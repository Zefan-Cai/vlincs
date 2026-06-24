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

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/local_runs/reproduce_feature_outlier_residual_rank06_07_p005_20260624}"

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
      sed -n '1,110p' "${PKG_DIR}/README.md"
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

ASSIGNMENT_CSV="${PKG_DIR}/repro/input/rank06_07_residual_feature_outlier_assignments.csv"
P005_CONFIG="${PKG_DIR}/repro/input/p005_area_config.txt"
EXPECTED_DIRECT="${PKG_DIR}/repro/expected/rank06_07_full_export.json"
EXPECTED_DENSITY="${PKG_DIR}/repro/expected/rank06_07_density_simple.json"
EXPECTED_P005="${PKG_DIR}/repro/expected/rank06_07_density_p005_area.json"
REPO_GT_ROOT="${REPO_ROOT}/kit/demo_data/ds1/gt"
REPO_GT_CHECKSUMS="${REPO_GT_ROOT}/checksums.sha256"

mkdir -p "${RUN_DIR}"

for required in "${ASSIGNMENT_CSV}" "${P005_CONFIG}" "${EXPECTED_DIRECT}" "${EXPECTED_DENSITY}" "${EXPECTED_P005}"; do
  if [ ! -f "${required}" ]; then
    echo "missing required replay file: ${required}" >&2
    exit 2
  fi
done

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
  DATA_ROOT="${REPO_GT_ROOT}"
fi
export DATA_ROOT

gt_count="$(count_gt_files "${DATA_ROOT}")"
if [ "${gt_count}" -lt 10 ]; then
  echo "missing DS1 GT under DATA_ROOT=${DATA_ROOT}" >&2
  echo "expected at least 10 files like Box/VLINCS_Performer/MS01/MC0001/2024-03-Tc6/*_v1.7.2.parquet" >&2
  echo "If this is a fresh clone, run:" >&2
  echo '  git lfs pull --include="kit/demo_data/ds1/**"' >&2
  exit 2
fi

first_gt="$(find "${DATA_ROOT}/Box/VLINCS_Performer/MS01/MC0001" -path '*2024-03-Tc*/*_v1.7.2.parquet' -type f | sort | head -1)"
if head -c 80 "${first_gt}" | grep -q "version https://git-lfs.github.com/spec/v1"; then
  echo "DS1 GT parquet is still a Git LFS pointer, not an Apache Parquet file:" >&2
  echo "  ${first_gt}" >&2
  echo "Install Git LFS, then run:" >&2
  echo '  git lfs pull --include="kit/demo_data/ds1/**"' >&2
  exit 2
fi

if [ -f "${REPO_GT_CHECKSUMS}" ]; then
  echo "REPRO verifying gt_checksums=${REPO_GT_CHECKSUMS}"
  (cd "${DATA_ROOT}" && shasum -a 256 -c "${REPO_GT_CHECKSUMS}" >/dev/null)
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
  --json "${RUN_DIR}/rank06_07_full_export.json" \
  --zip-out "${RUN_DIR}/rank06_07_full_export.zip"

echo "STAGE 2 density_simple_delivery"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/no_anchor_pervideo_filter_selector.py" \
  --source-zip "${RUN_DIR}/rank06_07_full_export.zip" \
  --policies density_simple \
  --json "${RUN_DIR}/rank06_07_density_simple.json" \
  --zip-out "${RUN_DIR}/rank06_07_density_simple.zip"

echo "STAGE 3 p005_area_delivery"
CONFIG="$(cat "${P005_CONFIG}")"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/evaluate_submission_detection_filter.py" \
  --submission-zip "${RUN_DIR}/rank06_07_density_simple.zip" \
  --config "${CONFIG}" \
  --json "${RUN_DIR}/rank06_07_density_p005_area.json" \
  --zip-out "${RUN_DIR}/rank06_07_density_p005_area.zip"

"${PYTHON_BIN}" - "${RUN_DIR}/rank06_07_full_export.json" "${RUN_DIR}/rank06_07_density_simple.json" "${RUN_DIR}/rank06_07_density_p005_area.json" "${EXPECTED_DIRECT}" "${EXPECTED_DENSITY}" "${EXPECTED_P005}" <<'PY'
import json
import sys

direct_path, density_path, p005_path, exp_direct_path, exp_density_path, exp_p005_path = sys.argv[1:7]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def primary(data):
    if isinstance(data, dict):
        for key in ("best", "primary"):
            if isinstance(data.get(key), dict):
                return data[key]
        rows = data.get("rows")
        if isinstance(rows, list) and rows:
            return rows[0]
    return data

direct = primary(load(direct_path))
density = primary(load(density_path))
p005 = primary(load(p005_path))
exp_direct = primary(load(exp_direct_path))
exp_density = primary(load(exp_density_path))
exp_p005 = primary(load(exp_p005_path))

checks = [
    ("direct_idf1", direct["idf1"], exp_direct["idf1"]),
    ("direct_hota", direct["hota"], exp_direct["hota"]),
    ("direct_assa", direct["assa"], exp_direct["assa"]),
    ("density_idf1", density["idf1"], exp_density["idf1"]),
    ("density_hota", density["hota"], exp_density["hota"]),
    ("density_assa", density["assa"], exp_density["assa"]),
    ("p005_idf1", p005["idf1"], exp_p005["idf1"]),
    ("p005_hota", p005["hota"], exp_p005["hota"]),
    ("p005_assa", p005["assa"], exp_p005["assa"]),
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
