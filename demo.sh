#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="${ROOT}/reports/vlincs_iterations/20260624_fresh_subpart_seq8367_p005_gain/reproduce.sh"

usage() {
  cat <<'EOF'
Usage:
  ./demo.sh                     # run DS1 WISC no-anchor method reproduction
  ./demo.sh --run-dir PATH       # optional output directory override
  ./demo.sh --data-root PATH     # optional DS1 evaluator GT root override

The default command intentionally prints:
  DONE: ds1  -> {...}

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

echo "[demo.sh] running DS1 WISC no-anchor method reproduction..."
bash "${REPRO}" "$@"
