#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PKG_DIR}/../../.." && pwd)"
if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "missing Python interpreter; install Python or set PYTHON_BIN=/path/to/python" >&2
    exit 2
  fi
fi
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/local_runs/reproduce_rank06_top32_20260623}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,80p' "${PKG_DIR}/README.md"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ensure_python_deps() {
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import numpy
import pandas
import pyarrow
import reid_hota
import sklearn
PY
  then
    export PYTHON_BIN
    return
  fi

  if [ "${VLINC_DEMO_NO_VENV:-0}" = "1" ]; then
    echo "Python dependencies missing; unset VLINC_DEMO_NO_VENV or install numpy pandas pyarrow scikit-learn reid-hota" >&2
    exit 2
  fi

  VENV_DIR="${REPO_ROOT}/.venv-demo"
  if [ -x "${VENV_DIR}/bin/python" ] && "${VENV_DIR}/bin/python" - <<'PY' >/dev/null 2>&1
import numpy
import pandas
import pyarrow
import reid_hota
import sklearn
PY
  then
    PYTHON_BIN="${VENV_DIR}/bin/python"
    export PYTHON_BIN
    return
  fi

  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "DEMO creating ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  echo "DEMO installing/verifying Python dependencies in ${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install --quiet numpy pandas pyarrow scikit-learn reid-hota
  PYTHON_BIN="${VENV_DIR}/bin/python"
  export PYTHON_BIN
}

ensure_python_deps

ASSIGNMENT_CSV="${PKG_DIR}/repro/input/rank06_component_subset_attach_source_assignments.csv"
P005_CONFIG="${PKG_DIR}/repro/input/p005_area_config.txt"
EXPECTED_P005="${PKG_DIR}/repro/expected/rank06_density_p005_area.json"

mkdir -p "${RUN_DIR}"

if [ ! -f "${ASSIGNMENT_CSV}" ]; then
  echo "missing assignment CSV: ${ASSIGNMENT_CSV}" >&2
  exit 2
fi
if [ ! -f "${P005_CONFIG}" ]; then
  echo "missing p005 config: ${P005_CONFIG}" >&2
  exit 2
fi

tracklet_parquets=("${REPO_ROOT}"/kit/demo_data/ds1/tracklets/*/tracklets.parquet)
if [ ! -f "${tracklet_parquets[0]}" ]; then
  echo "missing DS1 tracklet parquets; run git lfs pull for kit/demo_data/ds1" >&2
  exit 2
fi
if head -c 80 "${tracklet_parquets[0]}" | grep -q "version https://git-lfs.github.com/spec/v1"; then
  echo "DS1 tracklet parquet is still a Git LFS pointer, not an Apache Parquet file:" >&2
  echo "  ${tracklet_parquets[0]}" >&2
  echo "Install Git LFS, then run:" >&2
  echo '  git lfs pull --include="kit/demo_data/ds1/**"' >&2
  exit 2
fi

count_gt_files() {
  find "$1/Box/VLINCS_Performer/MS01/MC0001" -path '*2024-03-Tc*/*_v1.7.2.parquet' -type f 2>/dev/null | wc -l | tr -d ' '
}

if [ -z "${DATA_ROOT:-}" ]; then
  for candidate_root in \
    "${REPO_ROOT}/local_runs/local_data_root_20260622" \
    "/Users/zcai/Codex/vlincs_reid_by_search/local_runs/local_data_root_20260622" \
    "/mnt/localssd/vlincs_reid_data" \
    "/mnt/localssd/vlincs_reid_data_root"; do
    if [ "$(count_gt_files "${candidate_root}")" -ge 10 ]; then
      DATA_ROOT="${candidate_root}"
      break
    fi
  done
fi

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/local_runs/local_data_root_20260622}"

gt_count="$(count_gt_files "${DATA_ROOT}")"
if [ "${gt_count}" -lt 10 ]; then
  echo "missing DS1 GT under DATA_ROOT=${DATA_ROOT}" >&2
  echo "expected at least 10 files like Box/VLINCS_Performer/MS01/MC0001/2024-03-Tc6/*_v1.7.2.parquet" >&2
  exit 2
fi

echo "REPRO repo=${REPO_ROOT}"
echo "REPRO package=${PKG_DIR}"
echo "REPRO data_root=${DATA_ROOT}"
echo "REPRO run_dir=${RUN_DIR}"
echo "REPRO tracklet_parquets=${#tracklet_parquets[@]}"
echo "REPRO gt_files=${gt_count}"

echo "STAGE 1 direct_full_export_from_assignment"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/evaluate_sample_assignments_full.py" \
  --tracklet-parquet "${tracklet_parquets[@]}" \
  --assignments "${ASSIGNMENT_CSV}" \
  --fallback singleton \
  --json "${RUN_DIR}/rank06_full_export.json" \
  --zip-out "${RUN_DIR}/rank06_full_export.zip"

echo "STAGE 2 density_simple_delivery"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/no_anchor_pervideo_filter_selector.py" \
  --source-zip "${RUN_DIR}/rank06_full_export.zip" \
  --policies density_simple \
  --json "${RUN_DIR}/rank06_density_simple.json" \
  --zip-out "${RUN_DIR}/rank06_density_simple.zip"

echo "STAGE 3 p005_area_delivery"
CONFIG="$(cat "${P005_CONFIG}")"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/evaluate_submission_detection_filter.py" \
  --submission-zip "${RUN_DIR}/rank06_density_simple.zip" \
  --config "${CONFIG}" \
  --json "${RUN_DIR}/rank06_density_p005_area.json" \
  --zip-out "${RUN_DIR}/rank06_density_p005_area.zip"

"${PYTHON_BIN}" - "${RUN_DIR}/rank06_full_export.json" "${RUN_DIR}/rank06_density_simple.json" "${RUN_DIR}/rank06_density_p005_area.json" "${EXPECTED_P005}" <<'PY'
import json
import sys

direct_path, density_path, p005_path, expected_path = sys.argv[1:5]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

direct = load(direct_path)
density = load(density_path)["rows"][0]
p005 = load(p005_path)["rows"][0]
expected = load(expected_path)["rows"][0]

checks = [
    ("direct_idf1", direct["idf1"], 0.666005),
    ("density_idf1", density["idf1"], 0.668093),
    ("p005_idf1", p005["idf1"], expected["idf1"]),
    ("p005_hota", p005["hota"], expected["hota"]),
    ("p005_assa", p005["assa"], expected["assa"]),
]
for name, got, want in checks:
    if abs(float(got) - float(want)) > 1e-6:
        raise SystemExit(f"{name} mismatch: got {got}, expected {want}")
if p005.get("config_name") != "p005_area":
    raise SystemExit(f"config_name mismatch: {p005.get('config_name')!r}")
if int(p005.get("dropped_rows", 0)) <= 0:
    raise SystemExit("p005_area did not drop any rows")

print(json.dumps({
    "stage": "verified",
    "direct": {k: direct[k] for k in ("idf1", "hota", "assa")},
    "density_simple": {k: density[k] for k in ("idf1", "hota", "assa", "rows", "dropped_rows")},
    "p005_area": {k: p005[k] for k in ("idf1", "hota", "assa", "rows", "dropped_rows", "config_name")},
}, sort_keys=True))
PY
