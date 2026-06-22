#!/usr/bin/env bash
set -euo pipefail

# Launch the next no-anchor false-split repair sweep on the first reachable
# Pluto job.  The sweep intentionally samples candidate assignments after
# skipping top source->target edge families, so full scoring is spent on
# distinct repair families instead of variants of the same first edit.

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
usage: kit/run_no_anchor_false_split_diversity.sh [--job JOB] [--foreground] [--dry-run]

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
  kit/no_anchor_assignment_conflict_reassign_sweep.py
  kit/no_anchor_assignment_provisional_relink_sweep.py
  kit/no_anchor_assignment_state_policy_sweep.py
  kit/no_anchor_component_merge_sweep.py
  kit/no_anchor_louvain_sweep.py
  kit/no_anchor_resolve_sweep.py
  kit/analyze_no_anchor_full_proxy_training.py
  local_runs/no_anchor_full_proxy_delivery_ridge_model_20260620.json
  local_runs/no_anchor_source_island_acceptor_ridgelogit_model_20260620.json
)

for file in "${files[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${file}" ]]; then
    echo "missing required file: ${REPO_ROOT}/${file}" >&2
    exit 3
  fi
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
bundle="${tmp_dir}/vlincs_false_split_diversity.tgz"
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
'${REMOTE_PY}' -m py_compile kit/no_anchor_assignment_conflict_reassign_sweep.py
PYTHONPATH='${REMOTE_ROOT}' '${REMOTE_PY}' kit/no_anchor_assignment_conflict_reassign_sweep.py --self-test

log='${REMOTE_RUNS}/logs/no_anchor_conflict_reassign_diverse_first_edge_committee_deliveryproxy_20260620.log'
cmd=(
  '${REMOTE_PY}' kit/no_anchor_assignment_conflict_reassign_sweep.py
  --assignment-csv '${REMOTE_RUNS}/no_anchor_softcut_then_softoverlap_best_assignments_20260619.csv'
  --primary-feature-npz '${REMOTE_RUNS}/ds1_tracklet_fused_match1_person025_color010_face005_osnet005_s7true_20260619.npz'
  --view "posecolor:${REMOTE_RUNS}/ds1_tracklet_posecolor_s3_full_20260618.npz:0.5"
  --view "colorhist:${REMOTE_RUNS}/ds1_tracklet_colorhist_s3_20260618.npz:0.5"
  --view "dino:${REMOTE_RUNS}/ds1_tracklet_dinov2base_s1_20260620.npz:0.3"
  --committed-min-sizes 8
  --pending-max-sizes 0
  --conflict-rate-thresholds 0.01
  --source-min-component-sizes 32
  --source-max-component-sizes 1000000
  --source-seed-sims 0.74
  --source-expand-sims 0.70
  --source-top-ks 8
  --source-min-group-sizes 2
  --source-max-group-sizes 8
  --source-min-conflicts-to-rest 1
  --source-min-margins 0.03
  --source-max-groups-per-component 1
  --source-max-total-groups 64
  --target-states committed,provisional
  --min-target-sizes 16
  --target-top-ks 3
  --min-target-best-sims 0.78
  --min-target-mean-sims 0.72
  --min-target-view-votes 0.5
  --min-target-qualities 0.0
  --target-view-sim-thresholds 0.70
  --min-target-margins 0.00
  --max-forbidden-pairs 0
  --max-sources-per-target 1,2
  --max-reassignments 1,2,4,8
  --candidate-search-top-n 256
  --candidate-search-prefixes 4,8,16,32,64,128
  --candidate-targets-per-source 1
  --candidate-skip-first-edge-families 0,1,2,4,8,16,32,64
  --rank-by committee_proxy
  --learned-proxy-json '${REMOTE_ROOT}/local_runs/no_anchor_full_proxy_delivery_ridge_model_20260620.json'
  --source-acceptor-json '${REMOTE_ROOT}/local_runs/no_anchor_source_island_acceptor_ridgelogit_model_20260620.json'
  --source-acceptor-rank-weight 0.025
  --source-acceptor-rank-floor 0.50
  --full-selection diverse_first_edge
  --full-top-n 8
  --json '${REMOTE_RUNS}/no_anchor_conflict_reassign_diverse_first_edge_committee_deliveryproxy_20260620.json'
  --csv '${REMOTE_RUNS}/no_anchor_conflict_reassign_diverse_first_edge_committee_deliveryproxy_20260620.csv'
  --assignments-out '${REMOTE_RUNS}/no_anchor_conflict_reassign_diverse_first_edge_committee_deliveryproxy_top_assignments_20260620.csv'
)
if [[ '${FOREGROUND}' == '1' ]]; then
  "\${cmd[@]}"
else
  nohup "\${cmd[@]}" >"\${log}" 2>&1 &
  echo "pid=\$! log=\${log}"
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
  if ! expect_ssh "${cfg}" "${host}" "echo REMOTE_OK; test -d /mnt/localssd/vlincs_reid_by_search; test -d /mnt/localssd/vlincs_reid_runs"; then
    echo "skip ${job}: ssh probe failed" >&2
    continue
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry-run: would deploy ${bundle} to ${job} and run diverse-first-edge committee full-proxy sweep"
    exit 0
  fi
  remote_bundle="/tmp/vlincs_false_split_diversity_${USER}_$$.tgz"
  expect_scp "${cfg}" "${bundle}" "${host}:${remote_bundle}"
  expect_ssh "${cfg}" "${host}" "mkdir -p ${REMOTE_ROOT} && tar -xzf ${remote_bundle} -C ${REMOTE_ROOT} && rm -f ${remote_bundle}"
  remote_cmd="$(remote_run_cmd)"
  expect_ssh "${cfg}" "${host}" "${remote_cmd}"
  echo "started diverse-first-edge committee full-proxy sweep on ${job}"
  exit 0
done

echo "no reachable Pluto job found among: ${JOB_LIST[*]}" >&2
exit 4
