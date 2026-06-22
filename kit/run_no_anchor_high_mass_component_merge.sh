#!/usr/bin/env bash
set -euo pipefail

# Launch a no-anchor assignment-level component merge sweep that spends full
# scoring budget on high-mass false-split bridges instead of pair-F1 winners.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/localssd/vlincs_reid_by_search}"
REMOTE_RUNS="${REMOTE_RUNS:-/mnt/localssd/vlincs_reid_runs}"
REMOTE_PY="${REMOTE_PY:-/mnt/localssd/vlincs_reid_venv/bin/python}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-20}"
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-120}"
PLUTO_PASSWORD="${PLUTO_PASSWORD:-colligo}"
FOREGROUND=0
DRY_RUN=0
JOB_LIST=()

usage() {
  cat >&2 <<'EOF'
usage: kit/run_no_anchor_high_mass_component_merge.sh [--job JOB] [--foreground] [--dry-run]

Defaults:
  jobs: h100-test-3 h100-test-2 test-video-0

Environment overrides:
  REPO_ROOT, REMOTE_ROOT, REMOTE_RUNS, REMOTE_PY, CONNECT_TIMEOUT,
  REMOTE_TIMEOUT, PLUTO_PASSWORD
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB_LIST+=("${2:?--job requires a value}")
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
  kit/no_anchor_assignment_component_merge_sweep.py
  kit/no_anchor_component_merge_sweep.py
  kit/no_anchor_louvain_sweep.py
  kit/no_anchor_resolve_sweep.py
)

for file in "${files[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${file}" ]]; then
    echo "missing required file: ${REPO_ROOT}/${file}" >&2
    exit 3
  fi
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
bundle="${tmp_dir}/vlincs_high_mass_component_merge.tgz"
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
mkdir -p '${REMOTE_RUNS}/logs'
'${REMOTE_PY}' -m py_compile kit/no_anchor_component_merge_sweep.py kit/no_anchor_assignment_component_merge_sweep.py
'${REMOTE_PY}' kit/no_anchor_component_merge_sweep.py --self-test

base_cmd=(
  '${REMOTE_PY}' kit/no_anchor_assignment_component_merge_sweep.py
  --assignment-csv '${REMOTE_RUNS}/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv'
  --candidate-top-k 160
  --top-edge-k 12
  --centroid-weights 0.0,0.25,0.5,0.75
  --min-source-size 2
  --max-source-size 1000000
  --min-target-size 2
  --max-target-size 1000000
  --max-component-sizes 300,500,800
  --mutual-top-ks 0,1,2
  --thresholds 0.60,0.64,0.68,0.72,0.76,0.80,0.84
  --margins -1.0,0.0,0.02,0.04
  --accepted-preview-n 40
  --rank-by mass_proxy
  --full-top-n 12
)

run_one() {
  local name="\$1"
  local feature_npz="\$2"
  shift 2
  local log='${REMOTE_RUNS}/logs/no_anchor_high_mass_component_merge_'"${name}"'_20260620.log'
  local json='${REMOTE_RUNS}/no_anchor_high_mass_component_merge_'"${name}"'_20260620.json'
  local csv='${REMOTE_RUNS}/no_anchor_high_mass_component_merge_'"${name}"'_20260620.csv'
  local assignments='${REMOTE_RUNS}/no_anchor_high_mass_component_merge_'"${name}"'_top_assignments_20260620.csv'
  cmd=(
    "\${base_cmd[@]}"
    --merge-feature-npz "\${feature_npz}"
    "\$@"
    --json "\${json}"
    --csv "\${csv}"
    --assignments-out "\${assignments}"
  )
  if [[ '${FOREGROUND}' == '1' ]]; then
    "\${cmd[@]}"
  else
    nohup "\${cmd[@]}" >"\${log}" 2>&1 &
    echo "${name}_pid=\$! log=\${log}"
  fi
}

run_one fused '${REMOTE_RUNS}/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_20260619.npz'
run_one dino '${REMOTE_RUNS}/ds1_tracklet_dinov2base_s1_20260620.npz' \
  --centroid-weights 0.0,0.25,0.5 \
  --thresholds 0.54,0.58,0.62,0.66,0.70,0.74
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
  if ! expect_ssh "${cfg}" "${host}" "echo REMOTE_OK; test -d /mnt/localssd/vlincs_reid_by_search; test -d /mnt/localssd/vlincs_reid_runs"; then
    echo "skip ${job}: ssh probe failed" >&2
    continue
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry-run: would deploy ${bundle} to ${job} and run high-mass component merge"
    exit 0
  fi
  remote_bundle="/tmp/vlincs_high_mass_component_merge_${USER}_$$.tgz"
  expect_scp "${cfg}" "${bundle}" "${host}:${remote_bundle}"
  expect_ssh "${cfg}" "${host}" "mkdir -p ${REMOTE_ROOT} && tar -xzf ${remote_bundle} -C ${REMOTE_ROOT} && rm -f ${remote_bundle}"
  remote_cmd="$(remote_run_cmd)"
  expect_ssh "${cfg}" "${host}" "${remote_cmd}"
  echo "started high-mass component merge on ${job}"
  exit 0
done

echo "no reachable Pluto job found among: ${JOB_LIST[*]}" >&2
exit 4
