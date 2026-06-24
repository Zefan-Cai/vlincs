#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="${ROOT}/reports/vlincs_iterations/20260624_feature_outlier_residual_rank06_07_p005_gain/reproduce.sh"

if [ ! -f "${REPRO}" ]; then
  echo "missing no-anchor residual feature-outlier combo reproduce script: ${REPRO}" >&2
  exit 2
fi

if [ "$#" -gt 0 ]; then
  case "$1" in
    ds1|ds0001)
      shift
      filtered=()
      for arg in "$@"; do
        case "${arg}" in
          --no-cannot-link)
            ;;
          *)
            filtered+=("${arg}")
            ;;
        esac
      done
      echo "[demo.sh] legacy DS1 arguments detected; running the WISC no-anchor replay instead."
      set -- "${filtered[@]}"
      ;;
    ms02|debug|ds0000|ds0)
      echo "[demo.sh] MS02 is still served by kit/demo.sh. The root demo.sh is the WISC DS1 no-anchor replay." >&2
      exit 2
      ;;
  esac
fi

exec bash "${REPRO}" "$@"
