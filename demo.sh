#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
usage:
  ./demo.sh ds1 [kit/demo.py args...]
      Run the standard DS1 online-gallery demo from kit/pipelines/ds1.yaml.

  ./demo.sh no-anchor-top32 [reproduce args...]
  ./demo.sh ds1-no-anchor-top32 [reproduce args...]
      Recreate the delivered no-anchor top32 global-ID result:
      IDF1/HOTA/AssA = 0.668198/0.528747/0.539071.

Common no-anchor args:
  --data-root PATH   DATA_ROOT exposing Box/VLINCS_Performer DS1 GT.
  --run-dir PATH     Output directory for regenerated zips/json.
EOF
}

cmd="${1:-}"
case "${cmd}" in
  ""|-h|--help|help)
    usage
    ;;
  ds1)
    shift
    exec "${PYTHON_BIN}" "${ROOT}/kit/demo.py" --dataset ds1 "$@"
    ;;
  no-anchor-top32|ds1-no-anchor-top32|reproduce-no-anchor-top32)
    shift
    exec bash "${ROOT}/reports/vlincs_iterations/20260623_rank01_37to86_top32_p005_gain/reproduce.sh" "$@"
    ;;
  *)
    echo "unknown demo target: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
