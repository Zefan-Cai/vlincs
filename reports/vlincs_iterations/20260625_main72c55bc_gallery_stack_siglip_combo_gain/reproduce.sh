#!/usr/bin/env bash
set -euo pipefail

# Reproduce the post-main no-anchor repair stack.
#
# This script does not run the latest-main gallery ingestion itself.  First run
# the main 72c55bc DS1 demo so Postgres contains gallery_ds1.  Then run this
# script from the wisc checkout to export that gallery state, regenerate the
# no-anchor candidates, materialize the selected combo, and score it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPAIR_REPO="${REPAIR_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GALLERY_REPO="${GALLERY_REPO:-${REPAIR_REPO}}"
OUT_DIR="${OUT_DIR:-${REPAIR_REPO}/local_runs/main_72c55bc_stack_reproduce}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DBNAME="${DBNAME:-gallery_ds1}"
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-55434}"
export PGUSER="${PGUSER:-gallery}"
export PGPASSWORD="${PGPASSWORD:-gallery}"
export DATA_ROOT="${DATA_ROOT:-${REPAIR_REPO}/kit/demo_data/ds1/gt}"

FEATURE_DIR="${FEATURE_DIR:-${REPAIR_REPO}/kit/demo_data/ds1/features}"
TRACKLET_ROOT="${TRACKLET_ROOT:-${GALLERY_REPO}/kit/demo_data/ds1/tracklets}"
EMBEDDING_ROOT="${EMBEDDING_ROOT:-${GALLERY_REPO}/kit/demo_data/ds1/embeddings}"

mkdir -p "${OUT_DIR}/siglip_primary/assignments"

echo "STAGE 1 export gallery DB assignments"
"${PYTHON_BIN}" "${REPAIR_REPO}/kit/export_gallery_db_assignments.py" \
  --dbname "${DBNAME}" \
  --tracklet-root "${TRACKLET_ROOT}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --decision-status main_72c55bc_two_tier_component \
  --assignment-out "${OUT_DIR}/exported_main72c55bc_assignments.csv" \
  --json "${OUT_DIR}/exported_main72c55bc_assignments.json"

echo "STAGE 2 propose no-anchor subpart candidates"
"${PYTHON_BIN}" "${REPAIR_REPO}/kit/propose_no_anchor_subpart_repair_candidates.py" \
  --assignment-csv "${OUT_DIR}/exported_main72c55bc_assignments.csv" \
  --feature-npz "${FEATURE_DIR}/ds1_tracklet_siglip2_person_reid_s1_20260620.npz" \
  --primary-weight 1.0 \
  --view "weak:${FEATURE_DIR}/ds1_tracklet_weakmetric_osnet_s7_fused_w002_20260620_w0p1.npz:0.70" \
  --view "dino:${FEATURE_DIR}/ds1_tracklet_dinov2base_s1_20260620.npz:0.80" \
  --top-n 48 \
  --min-source-component-size 20 \
  --max-source-component-size 900 \
  --min-target-component-size 3 \
  --max-target-component-size 400 \
  --max-seeds-per-component 40 \
  --min-group-size 1 \
  --max-group-size 8 \
  --seed-sim 0.72 \
  --min-source-margin 0.0 \
  --min-target-sim 0.5 \
  --min-target-margin 0.0 \
  --targets-per-group 3 \
  --focus-videos "vlincs_MS01_MC0001_MCAM05_2024-03-Tc6,vlincs_MS01_MC0001_MCAM06_2024-03-Tc6,vlincs_MS01_MC0001_MCAM04_2024-03-Tc6,vlincs_MS01_MC0001_MCAM03_2024-03-Tc8,vlincs_MS01_MC0001_MCAM08_2024-03-Tc6" \
  --output-dir "${OUT_DIR}/siglip_primary/assignments" \
  --json "${OUT_DIR}/siglip_primary/manifest.json"

echo "STAGE 3 materialize selected repair combo"
"${PYTHON_BIN}" "${REPAIR_REPO}/kit/apply_no_anchor_candidate_combo.py" \
  --base-assignment-csv "${OUT_DIR}/exported_main72c55bc_assignments.csv" \
  --manifest "${OUT_DIR}/siglip_primary/manifest.json" \
  --rank 4,1,11,16,19,20 \
  --assignment-out "${OUT_DIR}/combo_all_positive_r04_r01_r11_r16_r19_r20.csv" \
  --json "${OUT_DIR}/combo_all_positive_r04_r01_r11_r16_r19_r20.json"

echo "STAGE 4 score with tracklet-key delivery evaluator"
"${PYTHON_BIN}" "${REPAIR_REPO}/kit/evaluate_sample_assignments_full.py" \
  --tracklet-parquet "${TRACKLET_ROOT}"/*/tracklets.parquet \
  --assignments "${OUT_DIR}/combo_all_positive_r04_r01_r11_r16_r19_r20.csv" \
  --json "${OUT_DIR}/combo_all_positive_r04_r01_r11_r16_r19_r20_sample_score.json" \
  --zip-out "${OUT_DIR}/combo_all_positive_r04_r01_r11_r16_r19_r20_sample.zip"

echo "DONE"
cat "${OUT_DIR}/combo_all_positive_r04_r01_r11_r16_r19_r20_sample_score.json"
