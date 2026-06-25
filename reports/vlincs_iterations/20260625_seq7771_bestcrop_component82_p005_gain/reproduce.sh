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

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/local_runs/reproduce_seq7771_bestcrop_component82_p005_20260625}"

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
      sed -n '1,140p' "${PKG_DIR}/README.md"
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
  echo "DS1 tracklet parquet is still a Git LFS pointer:" >&2
  echo "  ${tracklet_parquets[0]}" >&2
  echo 'Run: git lfs pull --include="kit/demo_data/ds1/**"' >&2
  exit 2
fi

PREV_REPRO="${REPO_ROOT}/reports/vlincs_iterations/20260625_post_combo5_sig14_weak22_p005_gain/reproduce.sh"
if [ ! -f "${PREV_REPRO}" ]; then
  echo "missing previous best method reproduction: ${PREV_REPRO}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"
PREV_RUN_DIR="${RUN_DIR}/previous_post_combo5_sig14_weak22"
METHOD_DIR="${RUN_DIR}/method_reproduction"
SIGLIP_DIR="${METHOD_DIR}/siglip_primary_focus_mcam08"
DELIVERY_DIR="${RUN_DIR}/delivery/siglip_r17_s11_to82_seq3403"
mkdir -p "${SIGLIP_DIR}/assignments" "${DELIVERY_DIR}"

echo "REPRO repo=${REPO_ROOT}"
echo "REPRO package=${PKG_DIR}"
echo "REPRO data_root=${DATA_ROOT}"
echo "REPRO run_dir=${RUN_DIR}"
echo "REPRO tracklet_parquets=${#tracklet_parquets[@]}"

echo "STAGE 1 reproduce_previous_post_combo5_sig14_weak22"
bash "${PREV_REPRO}" --run-dir "${PREV_RUN_DIR}" --data-root "${DATA_ROOT}"

BASE_ASSIGNMENT="${PREV_RUN_DIR}/method_reproduction/post_combo5_sig14_weak22_assignments.csv"
if [ ! -f "${BASE_ASSIGNMENT}" ]; then
  echo "missing previous reproduced base assignment: ${BASE_ASSIGNMENT}" >&2
  exit 2
fi

echo "STAGE 2 propose_siglip_component82_focus"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/propose_no_anchor_subpart_repair_candidates.py" \
  --assignment-csv "${BASE_ASSIGNMENT}" \
  --feature-npz "${REPO_ROOT}/kit/demo_data/ds1/features/ds1_tracklet_siglip2_person_reid_s1_20260620.npz" \
  --view "weak:${REPO_ROOT}/kit/demo_data/ds1/features/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz:0.70" \
  --view "dino:${REPO_ROOT}/kit/demo_data/ds1/features/ds1_tracklet_dinov2base_s1_20260620.npz:0.80" \
  --output-dir "${SIGLIP_DIR}/assignments" \
  --json "${SIGLIP_DIR}/manifest.json" \
  --top-n 24 \
  --min-source-component-size 20 \
  --max-source-component-size 700 \
  --min-target-component-size 3 \
  --max-target-component-size 300 \
  --min-component-conflict-edges 1 \
  --max-seeds-per-component 32 \
  --min-group-size 1 \
  --max-group-size 8 \
  --seed-sim 0.72 \
  --min-conflicts-to-rest 1 \
  --min-source-margin 0.0 \
  --min-target-sim 0.5 \
  --min-target-margin 0.0 \
  --targets-per-group 3 \
  --overlap-margin-frames 0 \
  --focus-videos vlincs_MS01_MC0001_MCAM08_2024-03-Tc6

RANK17_ASSIGNMENT="${SIGLIP_DIR}/assignments/rank17_subpart_s11_to82_1seq_assignments.csv"
EXPECTED_ASSIGNMENT_SHA="14a0c6a3a06724fbc7dc6abfd995f05ac48640aab72b15706876cd5bddc2d8cd"
if [ ! -f "${RANK17_ASSIGNMENT}" ]; then
  echo "missing expected rank17 assignment: ${RANK17_ASSIGNMENT}" >&2
  exit 2
fi
ASSIGNMENT_SHA="$(shasum -a 256 "${RANK17_ASSIGNMENT}" | awk '{print $1}')"
if [ "${ASSIGNMENT_SHA}" != "${EXPECTED_ASSIGNMENT_SHA}" ]; then
  echo "rank17 assignment checksum mismatch:" >&2
  echo "  got      ${ASSIGNMENT_SHA}" >&2
  echo "  expected ${EXPECTED_ASSIGNMENT_SHA}" >&2
  exit 2
fi
echo "REPRO final_assignment_sha256=${ASSIGNMENT_SHA}"

DIRECT_JSON="${METHOD_DIR}/siglip_r17_s11_to82_seq3403_full_export.json"
DIRECT_ZIP="${METHOD_DIR}/siglip_r17_s11_to82_seq3403_full_export.zip"
DENSITY_JSON="${DELIVERY_DIR}/siglip_r17_s11_to82_seq3403_density_simple.json"
DENSITY_ZIP="${DELIVERY_DIR}/siglip_r17_s11_to82_seq3403_density_simple.zip"
P005_JSON="${DELIVERY_DIR}/siglip_r17_s11_to82_seq3403_density_p005_area.json"
P005_ZIP="${DELIVERY_DIR}/siglip_r17_s11_to82_seq3403_density_p005_area.zip"
P005_CONFIG="${PKG_DIR}/repro/input/p005_area_config.txt"

echo "STAGE 3 score_direct_full"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/evaluate_sample_assignments_full.py" \
  --tracklet-parquet "${tracklet_parquets[@]}" \
  --assignments "${RANK17_ASSIGNMENT}" \
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
"${PYTHON_BIN}" - "${PKG_DIR}" "${DIRECT_JSON}" "${DENSITY_JSON}" "${P005_JSON}" "${ASSIGNMENT_SHA}" <<'PY'
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
assignment_sha = sys.argv[5]
expected_paths = {
    "direct": pkg / "repro/expected/siglip_r17_s11_to82_seq3403_full_export.json",
    "density": pkg / "repro/expected/siglip_r17_s11_to82_seq3403_density_simple.json",
    "p005": pkg / "repro/expected/siglip_r17_s11_to82_seq3403_density_p005_area.json",
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
if float(p005["idf1"]) <= 0.669466:
    raise SystemExit(f"p005 result did not beat previous best: {p005['idf1']}")

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
            "assignment_sha256": assignment_sha,
        },
        sort_keys=True,
    )
)
PY
