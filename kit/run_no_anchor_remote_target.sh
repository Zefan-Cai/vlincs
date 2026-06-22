#!/usr/bin/env bash
set -euo pipefail

# Deploy the current no-anchor research scripts to the first reachable Pluto
# job and start the DS1 target sweep. This is intentionally a local driver:
# it never reads labels or anchors, and it only starts the no-anchor sweep on
# an already prepared DS1 gallery runtime.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/localssd/vlincs_reid_by_search}"
REMOTE_RUNS="${REMOTE_RUNS:-/mnt/localssd/vlincs_reid_runs}"
REMOTE_PY="${REMOTE_PY:-/mnt/localssd/vlincs_reid_venv/bin/python}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/mnt/localssd/vlincs_reid_data}"
CASE="${CASE:-target}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-12}"
FOREGROUND=0
DRY_RUN=0
JOB_LIST=()

usage() {
  cat >&2 <<'EOF'
usage: kit/run_no_anchor_remote_target.sh [--job JOB] [--case CASE] [--foreground] [--dry-run]

Defaults:
  jobs: h100-test-3 h100-test-2 test-video-0
  case: target

Environment overrides:
  REPO_ROOT, REMOTE_ROOT, REMOTE_RUNS, REMOTE_PY, REMOTE_DATA_ROOT, CONNECT_TIMEOUT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB_LIST+=("${2:?--job requires a value}")
      shift 2
      ;;
    --case)
      CASE="${2:?--case requires a value}"
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
  kit/no_anchor_global_id_model.py
  kit/no_anchor_sample_parquet_sweep.py
  kit/no_anchor_resolve_sweep.py
  kit/no_anchor_result_gate.py
  kit/no_anchor_sweep_advisor.py
  kit/run_no_anchor_ds1_pair_model_experiments.sh
  kit/extract_tracklet_foundation_features.py
)

for file in "${files[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${file}" ]]; then
    echo "missing required file: ${REPO_ROOT}/${file}" >&2
    exit 3
  fi
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
bundle="${tmp_dir}/vlincs_no_anchor_delta.tgz"

(
  cd "${REPO_ROOT}"
  tar -czf "${bundle}" "${files[@]}"
)

echo "prepared bundle: ${bundle}"
echo "bundle files:"
tar -tzf "${bundle}" | sed 's/^/  /'

for job in "${JOB_LIST[@]}"; do
  cfg="${HOME}/.colligo/pluto_cli/ssh_configs/job_pluto-prod-zcai-${job}/config"
  host="pluto-prod-zcai-${job}-0"
  if [[ ! -f "${cfg}" ]]; then
    echo "skip ${job}: missing ssh config ${cfg}" >&2
    continue
  fi

  echo "probing ${job} (${host})..."
  if ! ssh -o BatchMode=yes -o ConnectTimeout="${CONNECT_TIMEOUT}" -F "${cfg}" "${host}" \
    'test -d /mnt/localssd && hostname && date' >/tmp/vlincs_no_anchor_probe.$$ 2>/tmp/vlincs_no_anchor_probe_err.$$; then
    echo "skip ${job}: ssh probe failed" >&2
    sed 's/^/  /' /tmp/vlincs_no_anchor_probe_err.$$ >&2 || true
    rm -f /tmp/vlincs_no_anchor_probe.$$ /tmp/vlincs_no_anchor_probe_err.$$
    continue
  fi
  sed 's/^/  /' /tmp/vlincs_no_anchor_probe.$$
  rm -f /tmp/vlincs_no_anchor_probe.$$ /tmp/vlincs_no_anchor_probe_err.$$

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry-run: would deploy bundle to ${job}:${REMOTE_ROOT} and run case ${CASE}"
    exit 0
  fi

  remote_bundle="/tmp/vlincs_no_anchor_delta_${USER}_$$.tgz"
  scp -F "${cfg}" "${bundle}" "${host}:${remote_bundle}"
  ssh -F "${cfg}" "${host}" "mkdir -p '${REMOTE_ROOT}' '${REMOTE_RUNS}/logs' && tar -xzf '${remote_bundle}' -C '${REMOTE_ROOT}' && rm -f '${remote_bundle}'"

  run_id="no_anchor_${CASE}_${job}_$(date +%Y%m%d_%H%M%S)"
  remote_log="${REMOTE_RUNS}/logs/${run_id}.log"
  remote_cmd="cd '${REMOTE_ROOT}' && ROOT='${REMOTE_ROOT}' PY='${REMOTE_PY}' RUNS='${REMOTE_RUNS}' DATA_ROOT='${REMOTE_DATA_ROOT}' bash kit/run_no_anchor_ds1_pair_model_experiments.sh '${CASE}'"
  if [[ "${FOREGROUND}" == "1" ]]; then
    echo "running foreground on ${job}: ${remote_cmd}"
    ssh -F "${cfg}" "${host}" "${remote_cmd}"
  else
    quoted_remote_cmd="$(printf '%q' "${remote_cmd}")"
    echo "starting background sweep on ${job}"
    ssh -F "${cfg}" "${host}" "mkdir -p '${REMOTE_RUNS}/logs'; nohup bash -lc ${quoted_remote_cmd} >'${remote_log}' 2>&1 & echo \$! > '${REMOTE_RUNS}/${run_id}.pid'; echo log='${remote_log}'; echo pid_file='${REMOTE_RUNS}/${run_id}.pid'"
  fi
  echo "deployed_and_started job=${job} case=${CASE}"
  exit 0
done

echo "no reachable Pluto job found among: ${JOB_LIST[*]}" >&2
exit 4
