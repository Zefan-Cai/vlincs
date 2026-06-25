#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="${ROOT}/reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/reproduce.sh"

usage() {
  cat <<'EOF'
Usage:
  ./demo.sh                      # run DS1/MS01 WISC no-anchor method reproduction
  ./demo.sh --run-dir PATH        # optional output directory override
  ./demo.sh --data-root PATH      # optional DS1 evaluator GT root override

The default command intentionally prints:
  DONE: ds1 -> {...}

No final assignment score is replayed. DS1 is regenerated from committed no-anchor
feature evidence and deterministic repair materialization before scoring.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "help" ]; then
  usage
  exit 0
fi

if [ ! -f "${REPRO}" ]; then
  echo "missing DS1 no-anchor reproduce script: ${REPRO}" >&2
  exit 2
fi

REPRO_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir|--data-root)
      if [ "$#" -lt 2 ]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 2
      fi
      REPRO_ARGS+=("$1" "$2")
      shift 2
      ;;
    ds1)
      echo "[demo.sh] note: dataset is fixed to DS1; positional 'ds1' is ignored." >&2
      shift
      ;;
    ms02|MS02)
      echo "[demo.sh] error: this handoff demo is fixed to DS1/MS01; MS02 is not selected here." >&2
      echo "[demo.sh] run the legacy MS02 diagnostic separately with an appropriate DATA_ROOT." >&2
      exit 2
      ;;
    *)
      echo "[demo.sh] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "[demo.sh] running DS1 WISC no-anchor method reproduction..."
bash "${REPRO}" "${REPRO_ARGS[@]}"
