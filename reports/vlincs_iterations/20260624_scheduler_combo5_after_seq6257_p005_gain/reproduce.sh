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

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/local_runs/reproduce_scheduler_combo5_after_seq6257_p005_20260624}"

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

BASE_ASSIGNMENT_CSV="${REPO_ROOT}/reports/vlincs_iterations/20260624_visual_subcluster_ctf_top2_p005_gain/repro/input/rank02_visual_positive_subcluster_ctf_topk_source_assignments.csv"
P005_CONFIG="${PKG_DIR}/repro/input/p005_area_config.txt"
EXPECTED_DIRECT="${PKG_DIR}/repro/expected/combo5_full_export.json"
EXPECTED_DENSITY="${PKG_DIR}/repro/expected/combo5_density_simple.json"
EXPECTED_P005="${PKG_DIR}/repro/expected/combo5_density_p005_area.json"
EXPECTED_ASSIGNMENT_SHA="24010eea71583ecedb1afcec7e8ae53d33b711537c920be56f3571b110c23010"
EXPECTED_FINAL_ASSIGNMENT_SHA="42f8f62f641cfe5f126cc781f58c365f746654f33a9861bd343ecd985d6afd39"
EXPECTED_COMBO_ASSIGNMENT_SHA="ce6e9b2d9f78fb2302050a3e65087498ce116d4c47d40e19576d1ac726e32393"
EXPECTED_SEQ6257_ASSIGNMENT_SHA="5359c8675655b3795b4b485eff8705d5c63bbf3714e74e0ebe200b523cfa290e"
EXPECTED_COMBO5_ASSIGNMENT_SHA="a569b4f7a2bbb0d69cee810f1db7e3917b5796c45f4305c5ce0cff4c2b1e43a2"
REPO_GT_ROOT="${REPO_ROOT}/kit/demo_data/ds1/gt"
REPO_GT_CHECKSUMS="${REPO_GT_ROOT}/checksums.sha256"
FEATURE_DIR="${REPO_ROOT}/kit/demo_data/ds1/features"
FEATURE_CHECKSUMS="${FEATURE_DIR}/checksums.sha256"

mkdir -p "${RUN_DIR}"

for required in "${BASE_ASSIGNMENT_CSV}" "${P005_CONFIG}" "${EXPECTED_DIRECT}" "${EXPECTED_DENSITY}" "${EXPECTED_P005}"; do
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

feature_npzs=(
  "${FEATURE_DIR}/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz"
  "${FEATURE_DIR}/ds1_tracklet_dinov2base_s1_20260620.npz"
  "${FEATURE_DIR}/ds1_tracklet_siglip2_person_reid_s1_20260620.npz"
)
for feature_npz in "${feature_npzs[@]}"; do
  if [ ! -f "${feature_npz}" ]; then
    echo "missing DS1 no-anchor feature artifact: ${feature_npz}" >&2
    echo 'Run: git lfs pull --include="kit/demo_data/ds1/**"' >&2
    exit 2
  fi
  if head -c 80 "${feature_npz}" | grep -q "version https://git-lfs.github.com/spec/v1"; then
    echo "DS1 no-anchor feature artifact is still a Git LFS pointer:" >&2
    echo "  ${feature_npz}" >&2
    echo 'Run: git lfs pull --include="kit/demo_data/ds1/**"' >&2
    exit 2
  fi
done
if [ -f "${FEATURE_CHECKSUMS}" ]; then
  echo "REPRO verifying feature_checksums=${FEATURE_CHECKSUMS}"
  (cd "${FEATURE_DIR}" && shasum -a 256 -c "${FEATURE_CHECKSUMS}" >/dev/null)
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

METHOD_DIR="${RUN_DIR}/method_reproduction"
METHOD_CANDIDATES_DIR="${METHOD_DIR}/feature_outlier_candidates"
METHOD_SUMMARY_JSON="${METHOD_DIR}/feature_outlier_summary.json"
METHOD_SUMMARY_CSV="${METHOD_DIR}/feature_outlier_summary.csv"
GENERATED_COMBO_ASSIGNMENT="${METHOD_DIR}/generated_combo_01_03_04_05_assignments.csv"
GENERATED_COMBO_MANIFEST="${METHOD_DIR}/generated_combo_01_03_04_05_manifest.json"
GENERATED_ASSIGNMENT="${METHOD_DIR}/generated_rank06_07_residual_feature_outlier_assignments.csv"
GENERATED_MANIFEST="${METHOD_DIR}/generated_rank06_07_residual_feature_outlier_manifest.json"
SUBPART_DIR="${METHOD_DIR}/fresh_subpart_weak_primary"
SUBPART_MANIFEST="${SUBPART_DIR}/manifest.json"
FINAL_ASSIGNMENT="${SUBPART_DIR}/assignments/rank04_subpart_s10_to77_1seq_assignments.csv"
LOCAL_WEAK_DIR="${METHOD_DIR}/local_gated_weak_primary"
LOCAL_WEAK_MANIFEST="${LOCAL_WEAK_DIR}/manifest.json"
LOCAL_SIGLIP_DIR="${METHOD_DIR}/local_gated_siglip_primary"
LOCAL_SIGLIP_MANIFEST="${LOCAL_SIGLIP_DIR}/manifest.json"
LOCAL_COMBO_DIR="${METHOD_DIR}/local_gated_cross_manifest_combo"
LOCAL_COMBO_MANIFEST="${LOCAL_COMBO_DIR}/manifest.json"
LOCAL_COMBO_ASSIGNMENT="${LOCAL_COMBO_DIR}/combo_top3_siglip1_weak3_weak1_assignments.csv"
GENERATED_SEQ6257_ASSIGNMENT="${METHOD_DIR}/generated_positive_scheduler_seq6257_assignments.csv"
GENERATED_SEQ6257_MANIFEST="${METHOD_DIR}/generated_positive_scheduler_seq6257_manifest.json"
COMBO5_ASSIGNMENT="${METHOD_DIR}/scheduler_combo5_after_seq6257_assignments.csv"
COMBO5_MANIFEST="${METHOD_DIR}/scheduler_combo5_after_seq6257_manifest.json"

echo "STAGE 0 method_reproduction_feature_outlier_proposer"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/propose_no_anchor_feature_outlier_relinks.py" \
  --assignment-csv "${BASE_ASSIGNMENT_CSV}" \
  --feature "weak:${FEATURE_DIR}/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz:1.0" \
  --feature "dino:${FEATURE_DIR}/ds1_tracklet_dinov2base_s1_20260620.npz:0.75" \
  --feature "siglip:${FEATURE_DIR}/ds1_tracklet_siglip2_person_reid_s1_20260620.npz:0.75" \
  --assignments-dir "${METHOD_CANDIDATES_DIR}" \
  --summary-json "${METHOD_SUMMARY_JSON}" \
  --summary-csv "${METHOD_SUMMARY_CSV}" \
  --max-source-centroid 0.72 \
  --min-target-centroid 0.6 \
  --min-centroid-margin 0.02 \
  --min-neighbor-margin 0.02 \
  --view-vote-threshold 0.55 \
  --emit-top-groups 24 \
  --skip-pairs '37->86,55->26,74->35,66->2,42->3,76->26,86->78,30->38,91->88,89->87,82->11,85->8,79->4'

echo "STAGE 0b method_reproduction_materialize_promoted_repairs"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/apply_no_anchor_ranked_repairs.py" \
  --base-assignment-csv "${BASE_ASSIGNMENT_CSV}" \
  --summary-json "${METHOD_SUMMARY_JSON}" \
  --rank 1 --rank 3 --rank 4 --rank 5 \
  --decision-status feature_outlier_combo_relink \
  --recompute-component-size \
  --assignment-out "${GENERATED_COMBO_ASSIGNMENT}" \
  --json "${GENERATED_COMBO_MANIFEST}"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/apply_no_anchor_ranked_repairs.py" \
  --base-assignment-csv "${GENERATED_COMBO_ASSIGNMENT}" \
  --summary-json "${METHOD_SUMMARY_JSON}" \
  --rank 6 --rank 7 \
  --skip-already-target \
  --decision-status feature_outlier_after_combo_relink \
  --assignment-out "${GENERATED_ASSIGNMENT}" \
  --json "${GENERATED_MANIFEST}"

GENERATED_ASSIGNMENT_SHA="$(shasum -a 256 "${GENERATED_ASSIGNMENT}" | awk '{print $1}')"
if [ "${GENERATED_ASSIGNMENT_SHA}" != "${EXPECTED_ASSIGNMENT_SHA}" ]; then
  echo "generated assignment checksum mismatch:" >&2
  echo "  got      ${GENERATED_ASSIGNMENT_SHA}" >&2
  echo "  expected ${EXPECTED_ASSIGNMENT_SHA}" >&2
  exit 2
fi
echo "REPRO generated_assignment_sha256=${GENERATED_ASSIGNMENT_SHA}"

echo "STAGE 0c method_reproduction_fresh_subpart_proposer"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/propose_no_anchor_subpart_repair_candidates.py" \
  --assignment-csv "${GENERATED_ASSIGNMENT}" \
  --feature-npz "${FEATURE_DIR}/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz" \
  --primary-weight 1.0 \
  --view "dino:${FEATURE_DIR}/ds1_tracklet_dinov2base_s1_20260620.npz:0.80" \
  --view "siglip:${FEATURE_DIR}/ds1_tracklet_siglip2_person_reid_s1_20260620.npz:0.30" \
  --output-dir "${SUBPART_DIR}/assignments" \
  --json "${SUBPART_MANIFEST}" \
  --top-n 12 \
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
  --min-source-margin 0.00 \
  --min-target-sim 0.50 \
  --min-target-margin 0.00 \
  --targets-per-group 2

if [ ! -f "${FINAL_ASSIGNMENT}" ]; then
  echo "missing promoted fresh-subpart assignment: ${FINAL_ASSIGNMENT}" >&2
  exit 2
fi

FINAL_ASSIGNMENT_SHA="$(shasum -a 256 "${FINAL_ASSIGNMENT}" | awk '{print $1}')"
if [ "${FINAL_ASSIGNMENT_SHA}" != "${EXPECTED_FINAL_ASSIGNMENT_SHA}" ]; then
  echo "final assignment checksum mismatch:" >&2
  echo "  got      ${FINAL_ASSIGNMENT_SHA}" >&2
  echo "  expected ${EXPECTED_FINAL_ASSIGNMENT_SHA}" >&2
  exit 2
fi
echo "REPRO final_assignment_sha256=${FINAL_ASSIGNMENT_SHA}"

echo "STAGE 0d method_reproduction_local_gated_subpart_rescan"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/propose_no_anchor_subpart_repair_candidates.py" \
  --assignment-csv "${FINAL_ASSIGNMENT}" \
  --feature-npz "${FEATURE_DIR}/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz" \
  --primary-weight 1.0 \
  --view "dino:${FEATURE_DIR}/ds1_tracklet_dinov2base_s1_20260620.npz:0.80" \
  --view "siglip:${FEATURE_DIR}/ds1_tracklet_siglip2_person_reid_s1_20260620.npz:0.30" \
  --output-dir "${LOCAL_WEAK_DIR}/assignments" \
  --json "${LOCAL_WEAK_MANIFEST}" \
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
  --min-source-margin 0.00 \
  --min-target-sim 0.50 \
  --min-target-margin 0.00 \
  --targets-per-group 3
"${PYTHON_BIN}" "${REPO_ROOT}/kit/propose_no_anchor_subpart_repair_candidates.py" \
  --assignment-csv "${FINAL_ASSIGNMENT}" \
  --feature-npz "${FEATURE_DIR}/ds1_tracklet_siglip2_person_reid_s1_20260620.npz" \
  --primary-weight 1.0 \
  --view "weak:${FEATURE_DIR}/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz:0.70" \
  --view "dino:${FEATURE_DIR}/ds1_tracklet_dinov2base_s1_20260620.npz:0.80" \
  --output-dir "${LOCAL_SIGLIP_DIR}/assignments" \
  --json "${LOCAL_SIGLIP_MANIFEST}" \
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
  --min-source-margin 0.00 \
  --min-target-sim 0.50 \
  --min-target-margin 0.00 \
  --targets-per-group 3

echo "STAGE 0e method_reproduction_cross_manifest_combo"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/compose_no_anchor_cross_manifest_repairs.py" \
  --base-assignment-csv "${FINAL_ASSIGNMENT}" \
  --candidate "${LOCAL_SIGLIP_MANIFEST}:1" \
  --candidate "${LOCAL_WEAK_MANIFEST}:3" \
  --candidate "${LOCAL_WEAK_MANIFEST}:1" \
  --decision-status local_gated_cross_manifest_subpart_combo \
  --assignment-out "${LOCAL_COMBO_ASSIGNMENT}" \
  --json "${LOCAL_COMBO_MANIFEST}"

COMBO_ASSIGNMENT_SHA="$(shasum -a 256 "${LOCAL_COMBO_ASSIGNMENT}" | awk '{print $1}')"
if [ "${COMBO_ASSIGNMENT_SHA}" != "${EXPECTED_COMBO_ASSIGNMENT_SHA}" ]; then
  echo "combo assignment checksum mismatch:" >&2
  echo "  got      ${COMBO_ASSIGNMENT_SHA}" >&2
  echo "  expected ${EXPECTED_COMBO_ASSIGNMENT_SHA}" >&2
  exit 2
fi
echo "REPRO combo_assignment_sha256=${COMBO_ASSIGNMENT_SHA}"

echo "STAGE 0f method_reproduction_generated_positive_scheduler_seq6257"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/compose_no_anchor_cross_manifest_repairs.py" \
  --base-assignment-csv "${LOCAL_COMBO_ASSIGNMENT}" \
  --candidate "${LOCAL_SIGLIP_MANIFEST}:10" \
  --decision-status generated_positive_scheduler_probe \
  --assignment-out "${GENERATED_SEQ6257_ASSIGNMENT}" \
  --json "${GENERATED_SEQ6257_MANIFEST}"

SEQ6257_ASSIGNMENT_SHA="$(shasum -a 256 "${GENERATED_SEQ6257_ASSIGNMENT}" | awk '{print $1}')"
if [ "${SEQ6257_ASSIGNMENT_SHA}" != "${EXPECTED_SEQ6257_ASSIGNMENT_SHA}" ]; then
  echo "seq6257 assignment checksum mismatch:" >&2
  echo "  got      ${SEQ6257_ASSIGNMENT_SHA}" >&2
  echo "  expected ${EXPECTED_SEQ6257_ASSIGNMENT_SHA}" >&2
  exit 2
fi
echo "REPRO seq6257_assignment_sha256=${SEQ6257_ASSIGNMENT_SHA}"

echo "STAGE 0g method_reproduction_scheduler_combo5_after_seq6257"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/compose_no_anchor_cross_manifest_repairs.py" \
  --base-assignment-csv "${GENERATED_SEQ6257_ASSIGNMENT}" \
  --candidate "${LOCAL_SIGLIP_MANIFEST}:18" \
  --candidate "${LOCAL_SIGLIP_MANIFEST}:6" \
  --candidate "${LOCAL_WEAK_MANIFEST}:12" \
  --candidate "${LOCAL_SIGLIP_MANIFEST}:9" \
  --candidate "${LOCAL_SIGLIP_MANIFEST}:2" \
  --decision-status post_seq6257_combo_probe \
  --assignment-out "${COMBO5_ASSIGNMENT}" \
  --json "${COMBO5_MANIFEST}"

COMBO5_ASSIGNMENT_SHA="$(shasum -a 256 "${COMBO5_ASSIGNMENT}" | awk '{print $1}')"
if [ "${COMBO5_ASSIGNMENT_SHA}" != "${EXPECTED_COMBO5_ASSIGNMENT_SHA}" ]; then
  echo "combo5 assignment checksum mismatch:" >&2
  echo "  got      ${COMBO5_ASSIGNMENT_SHA}" >&2
  echo "  expected ${EXPECTED_COMBO5_ASSIGNMENT_SHA}" >&2
  exit 2
fi
echo "REPRO combo5_assignment_sha256=${COMBO5_ASSIGNMENT_SHA}"

echo "STAGE 1 direct_full_export_from_scheduler_combo5_after_seq6257"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/evaluate_sample_assignments_full.py" \
  --tracklet-parquet "${tracklet_parquets[@]}" \
  --assignments "${COMBO5_ASSIGNMENT}" \
  --fallback singleton \
  --json "${RUN_DIR}/combo5_full_export.json" \
  --zip-out "${RUN_DIR}/combo5_full_export.zip"

echo "STAGE 2 density_simple_delivery"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/no_anchor_pervideo_filter_selector.py" \
  --source-zip "${RUN_DIR}/combo5_full_export.zip" \
  --policies density_simple \
  --json "${RUN_DIR}/combo5_density_simple.json" \
  --zip-out "${RUN_DIR}/combo5_density_simple.zip"

echo "STAGE 3 p005_area_delivery"
CONFIG="$(cat "${P005_CONFIG}")"
"${PYTHON_BIN}" "${REPO_ROOT}/kit/evaluate_submission_detection_filter.py" \
  --submission-zip "${RUN_DIR}/combo5_density_simple.zip" \
  --config "${CONFIG}" \
  --json "${RUN_DIR}/combo5_density_p005_area.json" \
  --zip-out "${RUN_DIR}/combo5_density_p005_area.zip"

"${PYTHON_BIN}" - "${RUN_DIR}/combo5_full_export.json" "${RUN_DIR}/combo5_density_simple.json" "${RUN_DIR}/combo5_density_p005_area.json" "${EXPECTED_DIRECT}" "${EXPECTED_DENSITY}" "${EXPECTED_P005}" <<'PY'
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
score = {
    "dataset": "ds1",
    "idf1": p005["idf1"],
    "hota": p005["hota"],
    "assa": p005["assa"],
    "detre": p005.get("detre"),
    "n_ids": p005.get("predicted_ids"),
    "per_video": p005.get("per_video", {}),
}
print(f"DONE: ds1 -> {score!r}")
PY
