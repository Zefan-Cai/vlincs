#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="${ROOT}/reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/reproduce.sh"

if [ ! -f "${REPRO}" ]; then
  echo "missing no-anchor visual-subcluster top2 reproduce script: ${REPRO}" >&2
  exit 2
fi

exec bash "${REPRO}" "$@"
