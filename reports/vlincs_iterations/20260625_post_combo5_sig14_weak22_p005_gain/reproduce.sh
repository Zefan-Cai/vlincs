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

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/local_runs/reproduce_post_combo5_sig14_weak22_p005_20260625}"

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
      sed -n '1,120p' "${PKG_DIR}/README.md"
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

if [ -z "${DATA_ROOT:-}" ]; then
  DATA_ROOT="${REPO_ROOT}/kit/demo_data/ds1/gt"
fi
export DATA_ROOT

tracklet_parquets=("${REPO_ROOT}"/kit/demo_data/ds1/tracklets/*/tracklets.parquet)
if [ ! -f "${tracklet_parquets[0]}" ]; then
  echo "missing DS1 tracklet parquets; run git lfs pull for kit/demo_data/ds1" >&2
  exit 2
fi
if head -c 80 "${tracklet_parquets[0]}" | grep -q "version https://git-lfs.github.com/spec/v1"; then
  echo "DS1 tracklet parquet is still a Git LFS pointer, not an Apache Parquet file:" >&2
  echo "  ${tracklet_parquets[0]}" >&2
  echo 'Run: git lfs pull --include="kit/demo_data/ds1/**"' >&2
  exit 2
fi

PREV_REPRO="${REPO_ROOT}/reports/vlincs_iterations/20260624_scheduler_combo5_after_seq6257_p005_gain/reproduce.sh"
if [ ! -f "${PREV_REPRO}" ]; then
  echo "missing previous combo5 method reproduction: ${PREV_REPRO}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"

PREV_RUN_DIR="${RUN_DIR}/previous_combo5"
METHOD_DIR="${RUN_DIR}/method_reproduction"
DELIVERY_DIR="${RUN_DIR}/delivery/combo_sig14_weak22"
mkdir -p "${METHOD_DIR}" "${DELIVERY_DIR}"

echo "REPRO repo=${REPO_ROOT}"
echo "REPRO package=${PKG_DIR}"
echo "REPRO data_root=${DATA_ROOT}"
echo "REPRO run_dir=${RUN_DIR}"
echo "REPRO tracklet_parquets=${#tracklet_parquets[@]}"

echo "STAGE 1 reproduce_previous_combo5_from_feature_evidence"
bash "${PREV_REPRO}" --run-dir "${PREV_RUN_DIR}" --data-root "${DATA_ROOT}"

BASE_ASSIGNMENT="${PREV_RUN_DIR}/method_reproduction/scheduler_combo5_after_seq6257_assignments.csv"
SIGLIP_MANIFEST="${PREV_RUN_DIR}/method_reproduction/local_gated_siglip_primary/manifest.json"
WEAK_MANIFEST="${PREV_RUN_DIR}/method_reproduction/local_gated_weak_primary/manifest.json"
FINAL_ASSIGNMENT="${METHOD_DIR}/post_combo5_sig14_weak22_assignments.csv"
FINAL_MANIFEST="${METHOD_DIR}/post_combo5_sig14_weak22_manifest.json"
EXPECTED_ASSIGNMENT_SHA="7719d5494242d65a70613a769b13df05a846bd0af85d8af276894f6674e7648a"

for required in "${BASE_ASSIGNMENT}" "${SIGLIP_MANIFEST}" "${WEAK_MANIFEST}"; do
  if [ ! -f "${required}" ]; then
    echo "missing previous reproduction artifact: ${required}" >&2
    exit 2
  fi
done

echo "STAGE 2 compose_post_combo5_siglip14_weak22"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/compose_no_anchor_cross_manifest_repairs.py" \
  --base-assignment-csv "${BASE_ASSIGNMENT}" \
  --candidate "${SIGLIP_MANIFEST}:14" \
  --candidate "${WEAK_MANIFEST}:22" \
  --assignment-out "${FINAL_ASSIGNMENT}" \
  --json "${FINAL_MANIFEST}" \
  --decision-status post_combo5_reviewer_combo_probe

FINAL_ASSIGNMENT_SHA="$(shasum -a 256 "${FINAL_ASSIGNMENT}" | awk '{print $1}')"
if [ "${FINAL_ASSIGNMENT_SHA}" != "${EXPECTED_ASSIGNMENT_SHA}" ]; then
  echo "final assignment checksum mismatch:" >&2
  echo "  got      ${FINAL_ASSIGNMENT_SHA}" >&2
  echo "  expected ${EXPECTED_ASSIGNMENT_SHA}" >&2
  exit 2
fi
echo "REPRO final_assignment_sha256=${FINAL_ASSIGNMENT_SHA}"

DIRECT_JSON="${METHOD_DIR}/combo_sig14_weak22_full_export.json"
DIRECT_ZIP="${METHOD_DIR}/combo_sig14_weak22_full_export.zip"
DENSITY_JSON="${DELIVERY_DIR}/combo_sig14_weak22_density_simple.json"
DENSITY_ZIP="${DELIVERY_DIR}/combo_sig14_weak22_density_simple.zip"
P005_JSON="${DELIVERY_DIR}/combo_sig14_weak22_density_p005_area.json"
P005_ZIP="${DELIVERY_DIR}/combo_sig14_weak22_density_p005_area.zip"
P005_CONFIG="${PKG_DIR}/repro/input/p005_area_config.txt"

echo "STAGE 3 score_direct_full"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/evaluate_sample_assignments_full.py" \
  --tracklet-parquet "${tracklet_parquets[@]}" \
  --assignments "${FINAL_ASSIGNMENT}" \
  --fallback singleton \
  --json "${DIRECT_JSON}" \
  --zip-out "${DIRECT_ZIP}"

echo "STAGE 4 delivery_density_simple"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/no_anchor_pervideo_filter_selector.py" \
  --source-zip "${DIRECT_ZIP}" \
  --policies density_simple \
  --json "${DENSITY_JSON}" \
  --zip-out "${DENSITY_ZIP}"

echo "STAGE 5 delivery_p005_area"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/evaluate_submission_detection_filter.py" \
  --submission-zip "${DENSITY_ZIP}" \
  --config "$(cat "${P005_CONFIG}")" \
  --json "${P005_JSON}" \
  --zip-out "${P005_ZIP}"

echo "STAGE 6 verify_expected_metrics"
"${PYTHON_BIN}" - "${PKG_DIR}" "${DIRECT_JSON}" "${DENSITY_JSON}" "${P005_JSON}" <<'PY'
import json
import math
import sys
from pathlib import Path

pkg = Path(sys.argv[1])
actual_paths = {
    "direct": Path(sys.argv[2]),
    "density": Path(sys.argv[3]),
    "p005": Path(sys.argv[4]),
}
expected_paths = {
    "direct": pkg / "repro/expected/combo_sig14_weak22_full_export.json",
    "density": pkg / "repro/expected/combo_sig14_weak22_density_simple.json",
    "p005": pkg / "repro/expected/combo_sig14_weak22_density_p005_area.json",
}

def load_metric(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("rows"), list) and data["rows"]:
      data = data["rows"][0]
    return data

checks = {
    "direct": ("idf1", "hota", "assa"),
    "density": ("idf1", "hota", "assa", "dropped_rows"),
    "p005": ("idf1", "hota", "assa", "dropped_rows", "config_name"),
}
for name, keys in checks.items():
    actual = load_metric(actual_paths[name])
    expected = load_metric(expected_paths[name])
    for key in keys:
        av = actual.get(key)
        ev = expected.get(key)
        if isinstance(ev, float):
            if not math.isclose(float(av), ev, rel_tol=0.0, abs_tol=1e-6):
                raise SystemExit(f"{name}.{key} mismatch: got {av}, expected {ev}")
        else:
            if av != ev:
                raise SystemExit(f"{name}.{key} mismatch: got {av}, expected {ev}")

p005 = load_metric(actual_paths["p005"])
if p005.get("config_name") != "p005_area" or int(p005.get("dropped_rows", 0)) <= 0:
    raise SystemExit(f"invalid p005 validation: {p005.get('config_name')} dropped={p005.get('dropped_rows')}")

print(
    "DONE: ds1 -> "
    + json.dumps(
        {
            "dataset": "ds1",
            "idf1": p005["idf1"],
            "hota": p005["hota"],
            "assa": p005["assa"],
            "config_name": p005["config_name"],
            "dropped_rows": p005["dropped_rows"],
            "method_reproduction": True,
            "uses_anchors": False,
            "uses_gt_for_training_or_anchors": False,
            "uses_gt_for_evaluation_only": True,
            "assignment_sha256": "7719d5494242d65a70613a769b13df05a846bd0af85d8af276894f6674e7648a",
        },
        sort_keys=True,
    )
)
PY
