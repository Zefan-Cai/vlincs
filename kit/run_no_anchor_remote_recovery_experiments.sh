#!/usr/bin/env bash
set -euo pipefail

# Local recovery launcher for the next no-anchor VLINCS experiments.
#
# It deploys the small set of changed research scripts to the first reachable
# Pluto job and starts resumable background jobs.  It does not use anchors or
# labels.  Ground truth is only used by the remote scorer scripts after
# prediction.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/localssd/vlincs_reid_by_search}"
REMOTE_RUNS="${REMOTE_RUNS:-/mnt/localssd/vlincs_reid_runs}"
REMOTE_PY="${REMOTE_PY:-/mnt/localssd/vlincs_reid_venv/bin/python}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-20}"
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-60}"
PLUTO_PASSWORD="${PLUTO_PASSWORD:-colligo}"
CASE="${CASE:-both}"
DRY_RUN=0
JOB_LIST=()

usage() {
  cat >&2 <<'EOF'
usage: kit/run_no_anchor_remote_recovery_experiments.sh [--job JOB] [--case both|pervideo|state|switch|selector|targetsrc|all] [--dry-run]

Defaults:
  jobs: h100-test-3 h100-test-2 test-video-0
  case: both

Cases:
  pervideo  submission-level per-video confidence/area oracle
  state     assignment-level component state/cannot-link policy sweep
  switch    assignment-level video/source switch diagnostic
  selector  no-GT sparse-overlay source selector
  targetsrc generate target-agglom sparse sources, then run no-GT selector
  both      pervideo + state, kept stable for existing recovery runs
  all       pervideo + state + switch + selector + targetsrc

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
    --case)
      CASE="${2:?--case requires a value}"
      shift 2
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

case "${CASE}" in
  both|pervideo|state|switch|selector|targetsrc|all) ;;
  *)
    echo "bad --case ${CASE}; expected both, pervideo, state, switch, selector, targetsrc, or all" >&2
    exit 2
    ;;
esac

if [[ ${#JOB_LIST[@]} -eq 0 ]]; then
  JOB_LIST=(h100-test-3 h100-test-2 test-video-0)
fi

files=(
  kit/no_anchor_submission_pervideo_filter_oracle.py
  kit/no_anchor_assignment_state_policy_sweep.py
  kit/no_anchor_assignment_video_switch.py
  kit/no_anchor_assignment_source_selector.py
  kit/export_no_anchor_time_agglom_model.py
  kit/export_no_anchor_target_agglom_source.py
  kit/evaluate_sample_assignments_full.py
  kit/evaluate_submission_detection_filter.py
  kit/evaluate_db_assignments_detection_filter.py
  kit/submission_video_switch.py
  kit/no_anchor_assignment_component_split_sweep.py
  kit/no_anchor_component_split_sweep.py
  kit/no_anchor_louvain_sweep.py
  kit/no_anchor_resolve_sweep.py
  kit/submit.py
)

for file in "${files[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${file}" ]]; then
    echo "missing required file: ${REPO_ROOT}/${file}" >&2
    exit 3
  fi
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
bundle="${tmp_dir}/vlincs_no_anchor_recovery.tgz"

(
  cd "${REPO_ROOT}"
  tar -czf "${bundle}" "${files[@]}"
)

echo "prepared bundle: ${bundle}"
tar -tzf "${bundle}" | sed 's/^/  /'

expect_ssh() {
  local cfg="$1"
  local host="$2"
  local remote_cmd="$3"
  expect <<EOF
set timeout ${REMOTE_TIMEOUT}
spawn ssh -F "${cfg}" -S none -o ControlMaster=no -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive -o NumberOfPasswordPrompts=2 -o ConnectTimeout=${CONNECT_TIMEOUT} "${host}" ${remote_cmd}
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

remote_start_cmd() {
  local selected_case="$1"
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
if [[ '${selected_case}' == 'both' || '${selected_case}' == 'state' || '${selected_case}' == 'all' ]]; then
  '${REMOTE_PY}' kit/no_anchor_assignment_state_policy_sweep.py --self-test
fi
if [[ '${selected_case}' == 'both' || '${selected_case}' == 'pervideo' || '${selected_case}' == 'all' ]]; then
  log='${REMOTE_RUNS}/logs/no_anchor_quality060_pervideo_conf_oracle_20260619.log'
  nohup '${REMOTE_PY}' kit/no_anchor_submission_pervideo_filter_oracle.py \\
    --submission-zip '${REMOTE_RUNS}/no_anchor_quality060_osnet_s3_component_merge_submission_20260619.zip' \\
    --conf-quantiles 0,0.01,0.02,0.03,0.05,0.08,0.10 \\
    --area-quantiles 0 \\
    --resume \\
    --full-oracle \\
    --json '${REMOTE_RUNS}/no_anchor_quality060_pervideo_conf_oracle_20260619.json' \\
    --zip-out '${REMOTE_RUNS}/no_anchor_quality060_pervideo_conf_oracle_20260619.zip' \\
    >"\${log}" 2>&1 &
  echo "pervideo_pid=\$! log=\${log}"
fi
if [[ '${selected_case}' == 'both' || '${selected_case}' == 'state' || '${selected_case}' == 'all' ]]; then
  log='${REMOTE_RUNS}/logs/no_anchor_state_policy_quality060_20260619.log'
  nohup '${REMOTE_PY}' kit/no_anchor_assignment_state_policy_sweep.py \\
    --assignment-csv '${REMOTE_RUNS}/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv' \\
    --committed-min-sizes 4,8,16,32 \\
    --pending-max-sizes 0,1,2,4 \\
    --conflict-rate-thresholds 0,0.0005,0.001,0.003,0.01 \\
    --policies keep_all,color_forced,singleton_forced,drop_forced,color_pending_forced,singleton_pending_forced,drop_pending_forced \\
    --full-top-n 5 \\
    --sort-key tracklet_pair_f1 \\
    --json '${REMOTE_RUNS}/no_anchor_state_policy_quality060_20260619.json' \\
    --assignments-out '${REMOTE_RUNS}/no_anchor_state_policy_quality060_best_assignments_20260619.csv' \\
    --component-states-out '${REMOTE_RUNS}/no_anchor_state_policy_quality060_component_states_20260619.csv' \\
    >"\${log}" 2>&1 &
  echo "state_pid=\$! log=\${log}"
fi
if [[ '${selected_case}' == 'switch' || '${selected_case}' == 'all' ]]; then
  switch_args=(
    --source "current:${REMOTE_RUNS}/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv"
  )
  for spec in \\
    "small060:${REMOTE_RUNS}/no_anchor_small_attach_quality060_assignments_20260618.csv" \\
    "face_local:${REMOTE_RUNS}/no_anchor_louvain_face005_localgrid_e2ebest_assignments_20260618.csv" \\
    "face_small:${REMOTE_RUNS}/no_anchor_louvain_face005_small_attach_top_assignments_20260618.csv" \\
    "highres:${REMOTE_RUNS}/no_anchor_louvain_base_highres_e2ebest_assignments_20260618.csv" \\
    "paircal:${REMOTE_RUNS}/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_assignments_20260618.csv" \\
    "cannotlink_light:${REMOTE_RUNS}/no_anchor_quality060_cannotlink_split_light_assignments_20260619.csv" \\
    "verifier_split:${REMOTE_RUNS}/no_anchor_current_best_verifier_split_t040_m16_assignments_20260619.csv"
  do
    path="\${spec#*:}"
    if [[ -f "\${path}" ]]; then
      switch_args+=(--source "\${spec}")
    else
      echo "switch_skip_missing=\${path}"
    fi
  done
  if (( \${#switch_args[@]} <= 2 )); then
    echo "switch_skip_reason=no_candidate_sources_besides_current"
  else
    log='${REMOTE_RUNS}/logs/no_anchor_assignment_video_switch_quality060_20260619.log'
    nohup '${REMOTE_PY}' kit/no_anchor_assignment_video_switch.py \\
      "\${switch_args[@]}" \\
      --reference-source current \\
      --base-source current \\
      --max-greedy-iters 4 \\
      --json '${REMOTE_RUNS}/no_anchor_assignment_video_switch_quality060_20260619.json' \\
      --assignments-out '${REMOTE_RUNS}/no_anchor_assignment_video_switch_quality060_best_assignments_20260619.csv' \\
      >"\${log}" 2>&1 &
    echo "switch_pid=\$! log=\${log}"
  fi
fi
if [[ '${selected_case}' == 'selector' || '${selected_case}' == 'all' ]]; then
  selector_args=(
    --source "current:${REMOTE_RUNS}/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv"
  )
  for spec in \\
    "small060:${REMOTE_RUNS}/no_anchor_small_attach_quality060_assignments_20260618.csv" \\
    "face_local:${REMOTE_RUNS}/no_anchor_louvain_face005_localgrid_e2ebest_assignments_20260618.csv" \\
    "face_small:${REMOTE_RUNS}/no_anchor_louvain_face005_small_attach_top_assignments_20260618.csv" \\
    "highres:${REMOTE_RUNS}/no_anchor_louvain_base_highres_e2ebest_assignments_20260618.csv" \\
    "paircal:${REMOTE_RUNS}/no_anchor_pair_calibrator_fused_mpc010_f032_mcam05area12000_assignments_20260618.csv" \\
    "cannotlink_light:${REMOTE_RUNS}/no_anchor_quality060_cannotlink_split_light_assignments_20260619.csv" \\
    "verifier_split:${REMOTE_RUNS}/no_anchor_current_best_verifier_split_t040_m16_assignments_20260619.csv"
  do
    path="\${spec#*:}"
    if [[ -f "\${path}" ]]; then
      selector_args+=(--source "\${spec}")
    else
      echo "selector_skip_missing=\${path}"
    fi
  done
  if (( \${#selector_args[@]} <= 2 )); then
    echo "selector_skip_reason=no_candidate_sources_besides_current"
  else
    log='${REMOTE_RUNS}/logs/no_anchor_assignment_source_selector_quality060_20260619.log'
    nohup '${REMOTE_PY}' kit/no_anchor_assignment_source_selector.py \\
      "\${selector_args[@]}" \\
      --reference-source current \\
      --base-source current \\
      --json '${REMOTE_RUNS}/no_anchor_assignment_source_selector_quality060_20260619.json' \\
      --assignments-out '${REMOTE_RUNS}/no_anchor_assignment_source_selector_quality060_best_assignments_20260619.csv' \\
      >"\${log}" 2>&1 &
    echo "selector_pid=\$! log=\${log}"
  fi
fi
if [[ '${selected_case}' == 'targetsrc' || '${selected_case}' == 'all' ]]; then
  log='${REMOTE_RUNS}/logs/no_anchor_target_agglom_sources_quality060_20260619.log'
  target_script='${REMOTE_RUNS}/logs/no_anchor_target_agglom_sources_quality060_20260619.sh'
  cat >"\${target_script}" <<'TARGETSRC'
set -euo pipefail
cd '${REMOTE_ROOT}'
export PYTHONPATH='${REMOTE_ROOT}':"\${PYTHONPATH:-}"
export DATA_ROOT="\${DATA_ROOT:-/mnt/localssd/vlincs_reid_data}"
export PGHOST="\${PGHOST:-/mnt/localssd/pgsocket}"
export PGPORT="\${PGPORT:-55433}"
export PGUSER="\${PGUSER:-gallery}"
export PGPASSWORD="\${PGPASSWORD:-gallery}"
target_dir='${REMOTE_RUNS}/no_anchor_target_agglom_sources_quality060_20260619'
mkdir -p "\${target_dir}"
'${REMOTE_PY}' kit/export_no_anchor_target_agglom_source.py \
  --target-clusters 640,960,1280 \
  --output-min-dets-list 1,5,10 \
  --output-min-conf-list 0.65,0.75 \
  --out-dir "\${target_dir}" \
  --prefix no_anchor_target_agglom_quality060_20260619 \
  --json '${REMOTE_RUNS}/no_anchor_target_agglom_sources_quality060_20260619.json' \
  --full-top-n 0
selector_args=(
  --source "current:${REMOTE_RUNS}/no_anchor_quality060_osnet_s3_component_merge_best_assignments_20260619.csv"
)
for path in "\${target_dir}"/*_assignments.csv; do
  [[ -f "\${path}" ]] || continue
  base="\$(basename "\${path}" _assignments.csv)"
  name="\${base#no_anchor_target_agglom_quality060_20260619_}"
  selector_args+=(--source "\${name}:\${path}")
done
'${REMOTE_PY}' kit/no_anchor_assignment_source_selector.py \
  "\${selector_args[@]}" \
  --reference-source current \
  --base-source current \
  --selector-strategy balanced \
  --sparse-min-component-ratio 0.20 \
  --sparse-target-component-ratio 0.45 \
  --json '${REMOTE_RUNS}/no_anchor_assignment_source_selector_target_agglom_quality060_20260619.json' \
  --assignments-out '${REMOTE_RUNS}/no_anchor_assignment_source_selector_target_agglom_quality060_best_assignments_20260619.csv'
TARGETSRC
  nohup bash "\${target_script}" >"\${log}" 2>&1 &
  echo "targetsrc_pid=\$! log=\${log}"
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

  echo "probing ${job} (${host})..."
  if ! expect_ssh "${cfg}" "${host}" "\"echo SSH_OK && hostname && test -d '${REMOTE_ROOT}' && echo REPO_OK\"" >/tmp/vlincs_recovery_probe.$$ 2>/tmp/vlincs_recovery_probe_err.$$; then
    echo "skip ${job}: ssh probe failed" >&2
    sed 's/^/  /' /tmp/vlincs_recovery_probe_err.$$ >&2 || true
    sed 's/^/  /' /tmp/vlincs_recovery_probe.$$ >&2 || true
    rm -f /tmp/vlincs_recovery_probe.$$ /tmp/vlincs_recovery_probe_err.$$
    continue
  fi
  sed 's/^/  /' /tmp/vlincs_recovery_probe.$$
  rm -f /tmp/vlincs_recovery_probe.$$ /tmp/vlincs_recovery_probe_err.$$

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry-run: would deploy ${bundle} to ${job}:${REMOTE_ROOT} and start case=${CASE}"
    exit 0
  fi

  remote_bundle="/tmp/vlincs_no_anchor_recovery_${USER}_$$.tgz"
  expect_scp "${cfg}" "${bundle}" "${host}:${remote_bundle}"
  expect_ssh "${cfg}" "${host}" "\"mkdir -p '${REMOTE_ROOT}' '${REMOTE_RUNS}/logs' && tar -xzf '${remote_bundle}' -C '${REMOTE_ROOT}' && rm -f '${remote_bundle}'\""
  start_file="${tmp_dir}/remote_start.sh"
  remote_start_cmd "${CASE}" > "${start_file}"
  expect_scp "${cfg}" "${start_file}" "${host}:/tmp/vlincs_no_anchor_recovery_start_${USER}_$$.sh"
  expect_ssh "${cfg}" "${host}" "\"bash /tmp/vlincs_no_anchor_recovery_start_${USER}_$$.sh; status=\$?; rm -f /tmp/vlincs_no_anchor_recovery_start_${USER}_$$.sh; exit \$status\""
  echo "deployed_and_started job=${job} case=${CASE}"
  exit 0
done

echo "no reachable Pluto job found among: ${JOB_LIST[*]}" >&2
exit 4
