#!/usr/bin/env bash
set -euo pipefail

# Materialize selected no-anchor full-score scheduler rows as assignment CSVs,
# then run the canonical DS1 full evaluator on the first reachable Pluto job.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/localssd/vlincs_reid_by_search}"
REMOTE_RUNS="${REMOTE_RUNS:-/mnt/localssd/vlincs_reid_runs}"
REMOTE_PY="${REMOTE_PY:-/mnt/localssd/vlincs_reid_venv/bin/python}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-20}"
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-600}"
PLUTO_PASSWORD="${PLUTO_PASSWORD:-colligo}"
SCHEDULER_JSON="${SCHEDULER_JSON:-local_runs/no_anchor_fullscore_scheduler_mass_features_diverse_20260620.json}"
BASE_ASSIGNMENT_CSV="${BASE_ASSIGNMENT_CSV:-${REMOTE_RUNS}/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv}"
SELECTION_RANKS="${SELECTION_RANKS:-1,2,3}"
RUN_NAME="${RUN_NAME:-no_anchor_scheduler_manifest_fullscore_20260620}"
FOREGROUND=0
DRY_RUN=0
JOB_LIST=()

usage() {
  cat >&2 <<'EOF'
usage: kit/run_no_anchor_scheduler_manifest_fullscore.sh [--job JOB] [--scheduler-json PATH] [--ranks 1,2,3] [--run-name NAME] [--foreground] [--dry-run]

Defaults:
  jobs: h100-test-3 h100-test-2 test-video-0
  scheduler: local_runs/no_anchor_fullscore_scheduler_mass_features_diverse_20260620.json

Environment overrides:
  REPO_ROOT, REMOTE_ROOT, REMOTE_RUNS, REMOTE_PY, CONNECT_TIMEOUT,
  REMOTE_TIMEOUT, PLUTO_PASSWORD, SCHEDULER_JSON, BASE_ASSIGNMENT_CSV,
  SELECTION_RANKS, RUN_NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB_LIST+=("${2:?--job requires a value}")
      shift 2
      ;;
    --scheduler-json)
      SCHEDULER_JSON="${2:?--scheduler-json requires a value}"
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

files=(
  kit/export_no_anchor_scheduler_manifest_assignments.py
  kit/no_anchor_fullscore_scheduler.py
  kit/evaluate_db_assignments_full.py
  kit/evaluate_sample_assignments_full.py
  kit/no_anchor_resolve_sweep.py
  "${SCHEDULER_JSON}"
  local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_stricttarget_pair_20260620.json
  local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_candidate_search_loose1_pair_20260620.json
  local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_narrow_selfplay_pair_20260620.csv
  local_runs/remote_h100_test_3_20260620/no_anchor_conflict_reassign_strict_narrow_selfplay_pair_20260620.json
)

for file in "${files[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${file}" ]]; then
    echo "missing required file: ${REPO_ROOT}/${file}" >&2
    exit 3
  fi
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
bundle="${tmp_dir}/vlincs_scheduler_manifest_fullscore.tgz"
(
  cd "${REPO_ROOT}"
  tar -czf "${bundle}" "${files[@]}"
)

expect_ssh() {
  local cfg="$1"
  local host="$2"
  local remote_cmd="$3"
  EXPECT_REMOTE_CMD="${remote_cmd}" expect <<EOF
set timeout ${REMOTE_TIMEOUT}
spawn ssh -F "${cfg}" -S none -o ControlMaster=no -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=2 -o ConnectTimeout=${CONNECT_TIMEOUT} "${host}" bash -lc "\$env(EXPECT_REMOTE_CMD)"
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
  expect <<EOF
set timeout ${REMOTE_TIMEOUT}
spawn scp -F "${cfg}" -o ControlMaster=no -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=2 -o ConnectTimeout=${CONNECT_TIMEOUT} "${src}" "${dst}"
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
mkdir -p '${REMOTE_RUNS}/logs' '${REMOTE_RUNS}/${RUN_NAME}'
'${REMOTE_PY}' -m py_compile \
  kit/export_no_anchor_scheduler_manifest_assignments.py \
  kit/no_anchor_fullscore_scheduler.py \
  kit/evaluate_db_assignments_full.py
'${REMOTE_PY}' kit/export_no_anchor_scheduler_manifest_assignments.py --self-test
cat > '${REMOTE_RUNS}/${RUN_NAME}/run_manifest_fullscore.sh' <<'REMOTE_SCRIPT'
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
ASSIGN_DIR="\${RUN_DIR}/assignments"
MANIFEST_JSON="\${RUN_DIR}/manifest_assignments.json"
RESULTS_JSONL="\${RUN_DIR}/full_results.jsonl"
mkdir -p "\${ASSIGN_DIR}"
'${REMOTE_PY}' kit/export_no_anchor_scheduler_manifest_assignments.py \
  --scheduler-json '${SCHEDULER_JSON}' \
  --base-assignment-csv '${BASE_ASSIGNMENT_CSV}' \
  --assignment-out-dir "\${ASSIGN_DIR}" \
  --manifest-json "\${MANIFEST_JSON}" \
  --selection-ranks '${SELECTION_RANKS}'
: > "\${RESULTS_JSONL}"
'${REMOTE_PY}' - "\${MANIFEST_JSON}" <<'PY' | while IFS= read -r assignment_csv; do
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
for item in manifest.get("outputs", []):
    print(item["output_csv"])
PY
  stem="\$(basename "\${assignment_csv%.csv}")"
  out_json="\${RUN_DIR}/\${stem}_full.json"
  out_zip="\${RUN_DIR}/\${stem}.zip"
  '${REMOTE_PY}' kit/evaluate_db_assignments_full.py \
    --assignment-csv "\${assignment_csv}" \
    --json "\${out_json}" \
    --zip-out "\${out_zip}" | tee -a "\${RESULTS_JSONL}"
done
REMOTE_SCRIPT
chmod +x '${REMOTE_RUNS}/${RUN_NAME}/run_manifest_fullscore.sh'
if [[ '${FOREGROUND}' == '1' ]]; then
  bash '${REMOTE_RUNS}/${RUN_NAME}/run_manifest_fullscore.sh'
else
  log='${REMOTE_RUNS}/logs/${RUN_NAME}.log'
  nohup bash '${REMOTE_RUNS}/${RUN_NAME}/run_manifest_fullscore.sh' >"\${log}" 2>&1 &
  echo "pid=\$! log=\${log} run_dir='${REMOTE_RUNS}/${RUN_NAME}'"
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
  if ! expect_ssh "${cfg}" "${host}" "echo REMOTE_OK; test -d '${REMOTE_ROOT}'; test -d '${REMOTE_RUNS}'; test -f '${BASE_ASSIGNMENT_CSV}'"; then
    echo "skip ${job}: ssh probe failed" >&2
    continue
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry-run: would deploy ${bundle} to ${job} and run scheduler ranks ${SELECTION_RANKS}"
    exit 0
  fi
  remote_bundle="/tmp/vlincs_scheduler_manifest_fullscore_${USER}_$$.tgz"
  expect_scp "${cfg}" "${bundle}" "${host}:${remote_bundle}"
  expect_ssh "${cfg}" "${host}" "mkdir -p '${REMOTE_ROOT}' && tar -xzf '${remote_bundle}' -C '${REMOTE_ROOT}' && rm -f '${remote_bundle}'"
  remote_cmd="$(remote_run_cmd)"
  expect_ssh "${cfg}" "${host}" "${remote_cmd}"
  echo "started scheduler manifest full-score run on ${job}"
  exit 0
done

echo "no reachable Pluto job found among: ${JOB_LIST[*]}" >&2
exit 4
