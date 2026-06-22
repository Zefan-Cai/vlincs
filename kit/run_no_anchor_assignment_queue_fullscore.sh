#!/usr/bin/env bash
set -euo pipefail

# Deploy a no-anchor assignment queue to a Pluto node and run canonical DS1
# full scoring on selected assignment CSVs.  This launcher scores already
# materialized assignment files; it does not use anchors or labels for
# prediction.  Ground truth is used only inside the remote evaluator.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/localssd/vlincs_reid_by_search}"
REMOTE_RUNS="${REMOTE_RUNS:-/mnt/localssd/vlincs_reid_runs}"
REMOTE_PY="${REMOTE_PY:-/mnt/localssd/vlincs_reid_venv/bin/python}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-20}"
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-900}"
PLUTO_PASSWORD="${PLUTO_PASSWORD:-colligo}"
QUEUE_JSON="${QUEUE_JSON:-local_runs/no_anchor_assignment_summary_proxy_fullscore_queue_20260620.json}"
SELECTION_RANKS="${SELECTION_RANKS:-1,2,3}"
RUN_NAME="${RUN_NAME:-no_anchor_assignment_queue_fullscore_20260620}"
FOREGROUND=0
DRY_RUN=0
JOB_LIST=()

usage() {
  cat >&2 <<'EOF'
usage: kit/run_no_anchor_assignment_queue_fullscore.sh [--job JOB] [--queue-json PATH] [--ranks 1,2,3] [--run-name NAME] [--foreground] [--dry-run]

Defaults:
  jobs: h100-test-3 h100-test-2 test-video-0
  queue: local_runs/no_anchor_assignment_summary_proxy_fullscore_queue_20260620.json

Queue JSON may contain rows in any of these list fields:
  queue_rows, rows, selected, top_candidates, top

Each row should contain assignment_csv.  Rank is either row.rank or list order.

Environment overrides:
  REPO_ROOT, REMOTE_ROOT, REMOTE_RUNS, REMOTE_PY, CONNECT_TIMEOUT,
  REMOTE_TIMEOUT, PLUTO_PASSWORD, QUEUE_JSON, SELECTION_RANKS, RUN_NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB_LIST+=("${2:?--job requires a value}")
      shift 2
      ;;
    --queue-json)
      QUEUE_JSON="${2:?--queue-json requires a value}"
      shift 2
      ;;
    --ranks|--selection-ranks)
      SELECTION_RANKS="${2:?--ranks requires a value}"
      shift 2
      ;;
    --run-name)
      RUN_NAME="${2:?--run-name requires a value}"
      shift 2
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ${#JOB_LIST[@]} -eq 0 ]]; then
  JOB_LIST=(h100-test-3 h100-test-2 test-video-0)
fi

if [[ ! -f "${REPO_ROOT}/${QUEUE_JSON}" ]]; then
  echo "missing queue json: ${REPO_ROOT}/${QUEUE_JSON}" >&2
  exit 3
fi

mapfile -t assignment_files < <(
  cd "${REPO_ROOT}"
  python - "${QUEUE_JSON}" "${SELECTION_RANKS}" <<'PY'
import json
import sys
from pathlib import Path

queue = Path(sys.argv[1])
ranks_text = sys.argv[2]
data = json.loads(queue.read_text())
rows = []
if isinstance(data, list):
    rows = [row for row in data if isinstance(row, dict)]
elif isinstance(data, dict):
    for key in ("queue_rows", "rows", "selected", "top_candidates", "top"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
if not rows:
    raise SystemExit(f"no rows found in {queue}")
want = None
if ranks_text.strip().lower() not in ("", "all"):
    want = {int(part.strip()) for part in ranks_text.split(",") if part.strip()}
selected = []
for idx, row in enumerate(rows, start=1):
    rank = int(float(row.get("rank") or row.get("_selection_rank") or idx))
    if want is not None and rank not in want:
        continue
    path = row.get("assignment_csv") or row.get("assignments_out") or row.get("output_csv")
    if not path:
        raise SystemExit(f"selected row rank {rank} has no assignment_csv")
    selected.append(str(path))
for path in selected:
    print(path)
PY
)

if [[ ${#assignment_files[@]} -eq 0 ]]; then
  echo "no assignment files selected from ${QUEUE_JSON} ranks ${SELECTION_RANKS}" >&2
  exit 3
fi

files=(
  kit/evaluate_db_assignments_full.py
  kit/evaluate_sample_assignments_full.py
  kit/no_anchor_resolve_sweep.py
  "${QUEUE_JSON}"
)
for assignment in "${assignment_files[@]}"; do
  files+=("${assignment}")
done

for file in "${files[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${file}" ]]; then
    echo "missing required file: ${REPO_ROOT}/${file}" >&2
    exit 3
  fi
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
bundle="${tmp_dir}/vlincs_assignment_queue_fullscore.tgz"
(
  cd "${REPO_ROOT}"
  COPYFILE_DISABLE=1 tar --format ustar -czf "${bundle}" "${files[@]}"
)

expect_ssh() {
  local cfg="$1"
  local host="$2"
  local remote_cmd="$3"
  local escaped_cmd
  printf -v escaped_cmd "%q" "${remote_cmd}"
  EXPECT_ESCAPED_CMD="${escaped_cmd}" expect <<EOF
set timeout ${REMOTE_TIMEOUT}
spawn ssh -F "${cfg}" -S none -o ControlMaster=no -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=2 -o ConnectTimeout=${CONNECT_TIMEOUT} "${host}" "bash -lc \$env(EXPECT_ESCAPED_CMD)"
expect {
  -re "(?i)password:" { send -- "${PLUTO_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF
}

expect_scp() {
  local cfg="$1"
  local src="$2"
  local dst="$3"
  EXPECT_CFG="${cfg}" EXPECT_SRC="${src}" EXPECT_DST="${dst}" EXPECT_HOST="${host}" expect <<EOF
set timeout ${REMOTE_TIMEOUT}
spawn bash -lc {cat "\$EXPECT_SRC" | ssh -F "\$EXPECT_CFG" -S none -o ControlMaster=no -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=2 -o ConnectTimeout=${CONNECT_TIMEOUT} "\$EXPECT_HOST" "cat > \"\$EXPECT_DST\""}
expect {
  -re "(?i)password:" { send -- "${PLUTO_PASSWORD}\r"; exp_continue }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF
}

remote_run_cmd() {
  cat <<EOF
set -euo pipefail
cd '${REMOTE_ROOT}'
export PYTHONPATH='${REMOTE_ROOT}':"\${PYTHONPATH:-}"
export DATA_ROOT="\${DATA_ROOT:-/mnt/localssd/vlincs_reid_data}"
export PGHOST="\${PGHOST:-/mnt/localssd/pgsocket}"
export PGPORT="\${PGPORT:-55433}"
export PGUSER="\${PGUSER:-gallery}"
export PGPASSWORD="\${PGPASSWORD:-gallery}"
RUN_DIR='${REMOTE_RUNS}/${RUN_NAME}'
mkdir -p "\${RUN_DIR}" '${REMOTE_RUNS}/logs'
'${REMOTE_PY}' -m py_compile \
  kit/evaluate_db_assignments_full.py \
  kit/evaluate_sample_assignments_full.py \
  kit/no_anchor_resolve_sweep.py
cat > "\${RUN_DIR}/assignment_list.txt" <<'ASSIGNMENTS'
$(printf '%s\n' "${assignment_files[@]}")
ASSIGNMENTS
cat > "\${RUN_DIR}/run_assignment_queue_fullscore.sh" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_ROOT}'
export PYTHONPATH='${REMOTE_ROOT}':"\${PYTHONPATH:-}"
export DATA_ROOT="\${DATA_ROOT:-/mnt/localssd/vlincs_reid_data}"
export PGHOST="\${PGHOST:-/mnt/localssd/pgsocket}"
export PGPORT="\${PGPORT:-55433}"
export PGUSER="\${PGUSER:-gallery}"
export PGPASSWORD="\${PGPASSWORD:-gallery}"
RUN_DIR='${REMOTE_RUNS}/${RUN_NAME}'
RESULTS_JSONL="\${RUN_DIR}/full_results.jsonl"
: > "\${RESULTS_JSONL}"
while IFS= read -r assignment_csv; do
  [[ -n "\${assignment_csv}" ]] || continue
  stem="\$(basename "\${assignment_csv%.csv}")"
  out_json="\${RUN_DIR}/\${stem}_full.json"
  out_zip="\${RUN_DIR}/\${stem}.zip"
  '${REMOTE_PY}' kit/evaluate_db_assignments_full.py \
    --assignment-csv "\${assignment_csv}" \
    --json "\${out_json}" \
    --zip-out "\${out_zip}" | tee -a "\${RESULTS_JSONL}"
done < "\${RUN_DIR}/assignment_list.txt"
REMOTE_SCRIPT
chmod +x "\${RUN_DIR}/run_assignment_queue_fullscore.sh"
if [[ '${FOREGROUND}' == '1' ]]; then
  bash "\${RUN_DIR}/run_assignment_queue_fullscore.sh"
else
  log='${REMOTE_RUNS}/logs/${RUN_NAME}.log'
  nohup bash "\${RUN_DIR}/run_assignment_queue_fullscore.sh" >"\${log}" 2>&1 &
  echo "pid=\$! log=\${log} run_dir=\${RUN_DIR}"
fi
EOF
}

for job in "${JOB_LIST[@]}"; do
  cfg="${HOME}/.colligo/pluto_cli/ssh_configs/job_pluto-prod-zcai-${job}/config"
  host="pluto-prod-zcai-${job}-0"
  if [[ ! -f "${cfg}" ]]; then
    echo "skip ${job}: missing ssh config ${cfg}" >&2
    continue
  fi
  echo "probing ${job}..."
  if ! expect_ssh "${cfg}" "${host}" "echo REMOTE_OK; test -d '${REMOTE_ROOT}'; test -d '${REMOTE_RUNS}'"; then
    echo "skip ${job}: ssh probe failed" >&2
    continue
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry-run: would deploy ${bundle} to ${job} and score ${#assignment_files[@]} assignments"
    printf '  %s\n' "${assignment_files[@]}"
    exit 0
  fi
  remote_bundle="/tmp/vlincs_assignment_queue_fullscore_${USER}_$$.tgz"
  expect_scp "${cfg}" "${bundle}" "${remote_bundle}"
  expect_ssh "${cfg}" "${host}" "mkdir -p '${REMOTE_ROOT}' && tar -xzf '${remote_bundle}' -C '${REMOTE_ROOT}' && rm -f '${remote_bundle}'"
  remote_cmd="$(remote_run_cmd)"
  expect_ssh "${cfg}" "${host}" "${remote_cmd}"
  echo "started assignment queue full-score run on ${job}"
  exit 0
done

echo "no reachable Pluto job found among: ${JOB_LIST[*]}" >&2
exit 4
