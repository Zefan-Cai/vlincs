#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="${ROOT}/reports/vlincs_iterations/20260623_rank01_37to86_top32_p005_gain/reproduce.sh"

if [ ! -f "${REPRO}" ]; then
  echo "missing no-anchor top32 reproduce script: ${REPRO}" >&2
  exit 2
fi

exec bash "${REPRO}" "$@"
