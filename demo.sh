#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="${ROOT}/reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/reproduce.sh"

usage() {
  cat <<'EOF'
Usage:
  ./demo.sh            # run MS02 gallery demo, then DS1 WISC no-anchor method reproduction
  ./demo.sh all        # same as default
  ./demo.sh ms02       # run only the MS02 gallery demo
  ./demo.sh ds1        # run only the DS1 WISC no-anchor method reproduction

The default command intentionally prints both:
  DONE: ms02 -> {...}
  DONE: ds1  -> {...}

No final assignment score is replayed. DS1 is regenerated from committed no-anchor
feature evidence and deterministic repair materialization before scoring.
EOF
}

run_ms02() {
  echo "[demo.sh] running MS02 gallery method reproduction..."
  local data_root="${DATA_ROOT:-/mnt/datastore2_videolincs/data}"
  local ms02_card="${data_root}/VLINCS_Performer-selected/MS02/MC0002/2018-03-Tc85"
  if [ ! -d "${ms02_card}" ]; then
    cat >&2 <<EOF
[demo.sh] missing MS02 DATA_ROOT card directory:
  ${ms02_card}

Set DATA_ROOT to the VLINCS datastore that contains VLINCS_Performer-selected
before running the combined demo:
  export DATA_ROOT=/path/to/vlincs/datastore
  ./demo.sh
EOF
    exit 2
  fi
  (
    cd "${ROOT}/kit"
    export DATA_ROOT="${data_root}"
    export DEMO_HEADLESS="${DEMO_HEADLESS:-1}"
    export GALLERY_DB_HOST_PORT="${GALLERY_DB_HOST_PORT:-0}"
    export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-gallery_demo_ms02_$$_$(date +%s)}"
    if [ "${DEMO_HEADLESS}" = "1" ]; then
      trap 'docker compose down -v || true' EXIT
    fi
    ./demo.sh ms02 --no-cannot-link "$@"
  )
}

run_ds1() {
  if [ ! -f "${REPRO}" ]; then
    echo "missing no-anchor residual feature-outlier combo reproduce script: ${REPRO}" >&2
    exit 2
  fi
  local filtered=()
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --no-cannot-link)
        ;;
      *)
        filtered+=("${arg}")
        ;;
    esac
  done
  echo "[demo.sh] running DS1 WISC no-anchor method reproduction..."
  bash "${REPRO}" "${filtered[@]}"
}

selected="${1:-all}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "${selected}" in
  all|"")
    if [ "$#" -gt 0 ]; then
      echo "[demo.sh] extra arguments are supported only for single-dataset runs." >&2
      usage >&2
      exit 2
    fi
    run_ms02
    run_ds1
    ;;
  ms02|debug|ds0000|ds0)
    run_ms02 "$@"
    ;;
  ds1|ds0001)
    run_ds1 "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[demo.sh] unknown dataset selector: ${selected}" >&2
    usage >&2
    exit 2
    ;;
esac
