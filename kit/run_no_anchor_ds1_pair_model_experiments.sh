#!/usr/bin/env bash
set -euo pipefail

# Fixed no-anchor DS1 global-ID experiment entrypoint.
# This script assumes the DS1 gallery DB and eval cache already exist on the
# Pluto node. It never uses anchors or GT labels for training; the eval cache is
# only read by the scoring path.

ROOT="${ROOT:-/mnt/localssd/vlincs_reid_by_search}"
PY="${PY:-/mnt/localssd/vlincs_reid_venv/bin/python}"
RUNS="${RUNS:-/mnt/localssd/vlincs_reid_runs}"
DBNAME="${DBNAME:-gallery_ds1}"
DATA_ROOT="${DATA_ROOT:-/mnt/localssd/vlincs_reid_data}"

export DATA_ROOT
export PYTHONPATH="${ROOT}:${ROOT}/kit${PYTHONPATH:+:${PYTHONPATH}}"
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-55433}"
export PGUSER="${PGUSER:-gallery}"
export PGPASSWORD="${PGPASSWORD:-gallery}"

cd "${ROOT}"

EVAL_CACHE="${RUNS}/ds1_eval_labels_iou050_gallery_ds1.npz"
MCAM05_M3="vlincs_MS01_MC0001_MCAM05_2024-03-Tc6:12000"

run_time_candidate_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver agglom \
    --train-top-k 30 --infer-top-k 15 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same camera \
    --pseudo-theta 0.018 --pseudo-top-k 15 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.42 \
    --random-negatives 60000 --max-neg-per-pos 4 \
    --thresholds 0.014,0.016,0.018,0.020,0.022,0.025,0.030,0.035,0.040 \
    --blends 0.00,0.15,0.25,0.35,0.50,0.75 \
    --output-min-area-by-video "${MCAM05_M3}" \
    --eval-cache "${EVAL_CACHE}" \
    --full-top-n 0 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_timecand_m3_20260617.joblib" \
    --json "${RUNS}/no_anchor_pair_model_hgb_timecand_m3_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_timecand_m3_20260617.csv"
}

run_support_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver support \
    --train-top-k 30 --infer-top-k 30 \
    --min-dets 10 --max-component-size 120 \
    --min-merge-support 2 --min-merge-support-ratio 0.0 \
    --exclude-same camera \
    --pseudo-theta 0.018 --pseudo-top-k 15 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.42 \
    --random-negatives 60000 --max-neg-per-pos 4 \
    --thresholds 0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
    --blends 0.00,0.25,0.50,0.75,1.00 \
    --output-min-area-by-video "${MCAM05_M3}" \
    --eval-cache "${EVAL_CACHE}" \
    --full-top-n 0 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_support_m3_20260617.joblib" \
    --json "${RUNS}/no_anchor_pair_model_hgb_support_m3_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_support_m3_20260617.csv"
}

run_consensus_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver consensus \
    --train-top-k 30 --infer-top-k 30 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same camera \
    --pseudo-theta 0.018 --pseudo-top-k 15 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --pseudo-ensemble --pseudo-ensemble-min-votes 2 --pseudo-ensemble-max-neg-votes 0 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.42 \
    --random-negatives 60000 --max-neg-per-pos 4 \
    --thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
    --blends 0.00 \
    --output-min-area-by-video "${MCAM05_M3}" \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_consensus_ens_m3_20260617.joblib" \
    --assignments-out "${RUNS}/no_anchor_pair_model_hgb_consensus_ens_m3_assignments_20260617.csv" \
    --json "${RUNS}/no_anchor_pair_model_hgb_consensus_ens_m3_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_consensus_ens_m3_20260617.csv"
}

run_consensus_auto_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver consensus \
    --train-top-k 30 --infer-top-k 30 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same camera \
    --pseudo-theta 0.018 --pseudo-top-k 15 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --pseudo-ensemble --pseudo-ensemble-min-votes 2 --pseudo-ensemble-max-neg-votes 0 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.42 \
    --random-negatives 60000 --max-neg-per-pos 4 \
    --thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
    --blends 0.00 \
    --output-auto-anomaly-admission \
    --output-auto-anomaly-metric both \
    --output-auto-anomaly-quantile 0.75 \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_consensus_ens_autoq75_20260617.joblib" \
    --assignments-out "${RUNS}/no_anchor_pair_model_hgb_consensus_ens_autoq75_assignments_20260617.csv" \
    --json "${RUNS}/no_anchor_pair_model_hgb_consensus_ens_autoq75_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_consensus_ens_autoq75_20260617.csv"
}

run_consensus_guard_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver consensus_guard \
    --train-top-k 30 --infer-top-k 30 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same camera \
    --pseudo-theta 0.018 --pseudo-top-k 15 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --pseudo-ensemble --pseudo-ensemble-min-votes 2 --pseudo-ensemble-max-neg-votes 0 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.42 \
    --random-negatives 60000 --max-neg-per-pos 4 \
    --thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
    --blends 0.00 \
    --output-min-area-by-video "${MCAM05_M3}" \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_ens_m3_20260617.joblib" \
    --assignments-out "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_ens_m3_assignments_20260617.csv" \
    --json "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_ens_m3_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_ens_m3_20260617.csv"
}

run_consensus_guard_auto_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver consensus_guard \
    --train-top-k 30 --infer-top-k 30 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same camera \
    --pseudo-theta 0.018 --pseudo-top-k 15 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --pseudo-ensemble --pseudo-ensemble-min-votes 2 --pseudo-ensemble-max-neg-votes 0 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.42 \
    --random-negatives 60000 --max-neg-per-pos 4 \
    --thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
    --blends 0.00 \
    --output-auto-anomaly-admission \
    --output-auto-anomaly-metric both \
    --output-auto-anomaly-quantile 0.75 \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_ens_autoq75_20260617.joblib" \
    --assignments-out "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_ens_autoq75_assignments_20260617.csv" \
    --json "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_ens_autoq75_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_ens_autoq75_20260617.csv"
}

run_consensus_guard_stream_auto_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver consensus_guard \
    --train-top-k 40 --infer-top-k 40 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same stream \
    --pseudo-theta 0.018 --pseudo-top-k 20 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --pseudo-ensemble --pseudo-ensemble-min-votes 2 --pseudo-ensemble-max-neg-votes 0 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.42 \
    --random-negatives 80000 --max-neg-per-pos 4 \
    --thresholds 0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80 \
    --blends 0.00 \
    --output-auto-anomaly-admission \
    --output-auto-anomaly-metric both \
    --output-auto-anomaly-quantile 0.75 \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_stream_ens_autoq75_20260617.joblib" \
    --assignments-out "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_stream_ens_autoq75_assignments_20260617.csv" \
    --json "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_stream_ens_autoq75_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_consensus_guard_stream_ens_autoq75_20260617.csv"
}

run_consensus_attach_auto_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver consensus_attach \
    --train-top-k 30 --infer-top-k 30 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same camera \
    --pseudo-theta 0.018 --pseudo-top-k 15 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --pseudo-ensemble --pseudo-ensemble-min-votes 2 --pseudo-ensemble-max-neg-votes 0 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --attach-threshold 0.68 --attach-margin 0.04 --attach-model-weight 0.65 \
    --attach-max-source-size 2 --attach-min-target-size 2 --attach-top-k 60 \
    --attach-min-edge-support 1 --attach-score-agg hybrid --attach-top-mean-k 3 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.42 \
    --random-negatives 60000 --max-neg-per-pos 4 \
    --thresholds 0.45,0.50,0.55,0.60,0.65,0.70 \
    --blends 0.00 \
    --output-auto-anomaly-admission \
    --output-auto-anomaly-metric both \
    --output-auto-anomaly-quantile 0.75 \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_ens_autoq75_20260617.joblib" \
    --assignments-out "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_ens_autoq75_assignments_20260617.csv" \
    --json "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_ens_autoq75_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_ens_autoq75_20260617.csv"
}

run_consensus_attach_stream_auto_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver consensus_attach \
    --train-top-k 40 --infer-top-k 40 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same stream \
    --pseudo-theta 0.018 --pseudo-top-k 20 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --pseudo-ensemble --pseudo-ensemble-min-votes 2 --pseudo-ensemble-max-neg-votes 0 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --attach-threshold 0.68 --attach-margin 0.04 --attach-model-weight 0.65 \
    --attach-max-source-size 2 --attach-min-target-size 2 --attach-top-k 80 \
    --attach-min-edge-support 1 --attach-score-agg hybrid --attach-top-mean-k 3 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.42 \
    --random-negatives 80000 --max-neg-per-pos 4 \
    --thresholds 0.45,0.50,0.55,0.60,0.65,0.70 \
    --blends 0.00 \
    --output-auto-anomaly-admission \
    --output-auto-anomaly-metric both \
    --output-auto-anomaly-quantile 0.75 \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_stream_ens_autoq75_20260617.joblib" \
    --assignments-out "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_stream_ens_autoq75_assignments_20260617.csv" \
    --json "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_stream_ens_autoq75_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_stream_ens_autoq75_20260617.csv"
}

run_consensus_attach_teacher_auto_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver consensus_attach \
    --train-top-k 45 --infer-top-k 45 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same camera \
    --pseudo-theta 0.018 --pseudo-top-k 24 \
    --pseudo-temporal-bonus 0.005 --pseudo-time-window-ms 1000 \
    --pseudo-ensemble --pseudo-ensemble-min-votes 2 --pseudo-ensemble-max-neg-votes 0 \
    --pseudo-consensus-pos-min-votes 3 --pseudo-consensus-pos-min-sim 0.66 \
    --candidate-time-bonus 0.005 --affinity-time-bonus 0.0 \
    --attach-threshold 0.66 --attach-margin 0.035 --attach-model-weight 0.65 \
    --attach-max-source-size 3 --attach-min-target-size 2 --attach-top-k 90 \
    --attach-min-edge-support 2 --attach-score-agg hybrid --attach-top-mean-k 3 \
    --pseudo-pos-min-sim 0.62 --pseudo-strong-pos-sim 0.76 --pseudo-neg-max-sim 0.42 \
    --random-negatives 90000 --max-neg-per-pos 4 \
    --thresholds 0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
    --blends 0.00 \
    --output-auto-anomaly-admission \
    --output-auto-anomaly-metric both \
    --output-auto-anomaly-quantile 0.75 \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_teacher_autoq75_20260617.joblib" \
    --assignments-out "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_teacher_autoq75_assignments_20260617.csv" \
    --json "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_teacher_autoq75_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_teacher_autoq75_20260617.csv"
}

run_consensus_attach_allpairs_auto_sweep() {
  "${PY}" kit/no_anchor_global_id_model.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --model-type hgb --solver consensus_attach \
    --train-top-k 60 --infer-top-k 60 \
    --min-dets 10 --max-component-size 120 \
    --exclude-same none \
    --pseudo-theta 0.018 --pseudo-top-k 30 \
    --pseudo-temporal-bonus 0.003 --pseudo-time-window-ms 1000 \
    --pseudo-ensemble --pseudo-ensemble-min-votes 3 --pseudo-ensemble-max-neg-votes 0 \
    --pseudo-consensus-pos-min-votes 4 --pseudo-consensus-pos-min-sim 0.68 \
    --candidate-time-bonus 0.003 --affinity-time-bonus 0.0 \
    --attach-threshold 0.72 --attach-margin 0.06 --attach-model-weight 0.70 \
    --attach-max-source-size 3 --attach-min-target-size 2 --attach-top-k 120 \
    --attach-min-edge-support 2 --attach-score-agg hybrid --attach-top-mean-k 3 \
    --pseudo-pos-min-sim 0.64 --pseudo-strong-pos-sim 0.78 --pseudo-neg-max-sim 0.40 \
    --random-negatives 120000 --max-neg-per-pos 4 \
    --thresholds 0.55,0.60,0.65,0.70,0.75,0.80,0.85 \
    --blends 0.00 \
    --output-auto-anomaly-admission \
    --output-auto-anomaly-metric both \
    --output-auto-anomaly-quantile 0.75 \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --model-out "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_allpairs_autoq75_20260617.joblib" \
    --assignments-out "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_allpairs_autoq75_assignments_20260617.csv" \
    --json "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_allpairs_autoq75_20260617.json" \
    --csv "${RUNS}/no_anchor_pair_model_hgb_consensus_attach_allpairs_autoq75_20260617.csv"
}

run_admission_sweep() {
  local out_csv="${RUNS}/no_anchor_time_agglom_m3_admission_quantile_20260617.csv"
  local out_json="${RUNS}/no_anchor_time_agglom_m3_admission_quantile_20260617.jsonl"
  : > "${out_json}"
  echo "kind,area_q,quality_q,json,csv" > "${out_csv}"
  for kind in area quality; do
    for q in 0.50 0.60 0.70 0.80 0.85 0.90; do
      local tag="q${q/./p}"
      local json="${RUNS}/no_anchor_best_time_m3_${kind}_${tag}_20260617.json"
      local csv="${RUNS}/no_anchor_best_time_m3_${kind}_${tag}_20260617.csv"
      local extra=()
      if [[ "${kind}" == "area" ]]; then
        extra=(--output-drop-area-quantile-by-video "vlincs_MS01_MC0001_MCAM05_2024-03-Tc6:${q}")
      else
        extra=(--output-drop-quality-quantile-by-video "vlincs_MS01_MC0001_MCAM05_2024-03-Tc6:${q}")
      fi
      "${PY}" kit/no_anchor_resolve_sweep.py \
        --dataset ds1 --dbname "${DBNAME}" --role resolve \
        --modes time_agglom \
        --thetas 0.018 --top-ks 15 --min-dets 10 --exclude-same camera \
        --temporal-bonus 0.005 --time-windows-ms 1000 \
        "${extra[@]}" \
        --eval-cache "${EVAL_CACHE}" \
        --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
        --full-top-n 1 \
        --json "${json}" \
        --csv "${csv}"
      echo "{\"kind\":\"${kind}\",\"q\":${q},\"json\":\"${json}\",\"csv\":\"${csv}\"}" >> "${out_json}"
      echo "${kind},${q},,${json},${csv}" >> "${out_csv}"
    done
  done
}

run_auto_admission_sweep() {
  local out_csv="${RUNS}/no_anchor_time_agglom_auto_admission_quantile_20260617.csv"
  local out_json="${RUNS}/no_anchor_time_agglom_auto_admission_quantile_20260617.jsonl"
  : > "${out_json}"
  echo "metric,q,json,csv" > "${out_csv}"
  for metric in quality both area; do
    for q in 0.50 0.60 0.70 0.75 0.80 0.85 0.90; do
      local tag="${metric}_q${q/./p}"
      local json="${RUNS}/no_anchor_best_time_auto_${tag}_20260617.json"
      local csv="${RUNS}/no_anchor_best_time_auto_${tag}_20260617.csv"
      "${PY}" kit/no_anchor_resolve_sweep.py \
        --dataset ds1 --dbname "${DBNAME}" --role resolve \
        --modes time_agglom \
        --thetas 0.018 --top-ks 15 --min-dets 10 --exclude-same camera \
        --temporal-bonus 0.005 --time-windows-ms 1000 \
        --output-auto-anomaly-admission \
        --output-auto-anomaly-metric "${metric}" \
        --output-auto-anomaly-quantile "${q}" \
        --eval-cache "${EVAL_CACHE}" \
        --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
        --full-top-n 1 \
        --json "${json}" \
        --csv "${csv}"
      echo "{\"metric\":\"${metric}\",\"q\":${q},\"json\":\"${json}\",\"csv\":\"${csv}\"}" >> "${out_json}"
      echo "${metric},${q},${json},${csv}" >> "${out_csv}"
    done
  done
}

run_target_agglom_sweep() {
  "${PY}" kit/no_anchor_resolve_sweep.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --modes agglom_n \
    --target-clusters 120,160,220,320,480,640,960,1280 \
    --top-ks 15,30,50 \
    --min-dets 1,5,10 \
    --exclude-same camera,stream \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --json "${RUNS}/no_anchor_target_agglom_n_20260617.json" \
    --csv "${RUNS}/no_anchor_target_agglom_n_20260617.csv"
}

run_target_agglom_nfc_sweep() {
  "${PY}" kit/no_anchor_resolve_sweep.py \
    --dataset ds1 --dbname "${DBNAME}" --role resolve \
    --nfc-k1 8 --nfc-k2 2 --nfc-eta 0.5 --nfc-exclude-same camera \
    --modes agglom_n \
    --target-clusters 240,280,300,320,340,360,400,480,640,960,1280 \
    --top-ks 15,30,50 \
    --min-dets 1,5,10 \
    --exclude-same camera,stream \
    --eval-cache "${EVAL_CACHE}" \
    --full-selection-keys tracklet_pair_f1,pair_gate_margin,tracklet_pair_precision,tracklet_pair_recall \
    --full-top-n 8 \
    --json "${RUNS}/no_anchor_target_agglom_nfc_k8_e050_20260617.json" \
    --csv "${RUNS}/no_anchor_target_agglom_nfc_k8_e050_20260617.csv"
}

run_gate() {
  "${PY}" kit/no_anchor_result_gate.py \
    "${RUNS}/no_anchor_pair_model_hgb_*20260617.json" \
    "${RUNS}/no_anchor_target_agglom_n_20260617.json" \
    "${RUNS}/no_anchor_target_agglom_nfc_k8_e050_20260617.json" \
    "${RUNS}/no_anchor_best_time_auto_*20260617.json" \
    "${RUNS}/no_anchor_best_time_m3_*20260617.json" \
    --global-metric tracklet_pair_f1 \
    --precision-metric tracklet_pair_precision \
    --recall-metric tracklet_pair_recall \
    --e2e-metric full_idf1 \
    --global-threshold 0.70 \
    --precision-threshold 0.70 \
    --recall-threshold 0.70 \
    --e2e-threshold 0.70 \
    --json-out "${RUNS}/no_anchor_gate_20260617.json" \
    --csv-out "${RUNS}/no_anchor_gate_20260617.csv"
}

run_advisor() {
  "${PY}" kit/no_anchor_sweep_advisor.py \
    "${RUNS}/no_anchor_gate_20260617.json" \
    --run-script "kit/run_no_anchor_ds1_pair_model_experiments.sh" \
    --json-out "${RUNS}/no_anchor_advisor_20260617.json" \
    --text-out "${RUNS}/no_anchor_advisor_20260617.txt"
}

case "${1:-time-candidate}" in
  time-candidate) run_time_candidate_sweep ;;
  support) run_support_sweep ;;
  consensus) run_consensus_sweep ;;
  consensus-auto) run_consensus_auto_sweep ;;
  consensus-guard) run_consensus_guard_sweep ;;
  consensus-guard-auto) run_consensus_guard_auto_sweep ;;
  consensus-guard-stream-auto) run_consensus_guard_stream_auto_sweep ;;
  consensus-attach-auto) run_consensus_attach_auto_sweep ;;
  consensus-attach-stream-auto) run_consensus_attach_stream_auto_sweep ;;
  consensus-attach-teacher-auto) run_consensus_attach_teacher_auto_sweep ;;
  consensus-attach-allpairs-auto) run_consensus_attach_allpairs_auto_sweep ;;
  target-agglom) run_target_agglom_sweep ;;
  target-agglom-nfc) run_target_agglom_nfc_sweep ;;
  admission) run_admission_sweep ;;
  auto-admission) run_auto_admission_sweep ;;
  gate) run_gate ;;
  advisor) run_advisor ;;
  target)
    run_consensus_guard_auto_sweep
    run_consensus_guard_stream_auto_sweep
    run_consensus_attach_auto_sweep
    run_consensus_attach_stream_auto_sweep
    run_consensus_attach_teacher_auto_sweep
    run_consensus_attach_allpairs_auto_sweep
    run_target_agglom_sweep
    run_target_agglom_nfc_sweep
    run_auto_admission_sweep
    run_gate
    run_advisor
    ;;
  all)
    run_time_candidate_sweep
    run_support_sweep
    run_consensus_sweep
    run_consensus_auto_sweep
    run_consensus_guard_sweep
    run_consensus_guard_auto_sweep
    run_consensus_guard_stream_auto_sweep
    run_consensus_attach_auto_sweep
    run_consensus_attach_stream_auto_sweep
    run_consensus_attach_teacher_auto_sweep
    run_consensus_attach_allpairs_auto_sweep
    run_target_agglom_sweep
    run_target_agglom_nfc_sweep
    run_admission_sweep
    run_auto_admission_sweep
    run_gate
    run_advisor
    ;;
  *)
    echo "usage: $0 {time-candidate|support|consensus|consensus-auto|consensus-guard|consensus-guard-auto|consensus-guard-stream-auto|consensus-attach-auto|consensus-attach-stream-auto|consensus-attach-teacher-auto|consensus-attach-allpairs-auto|target-agglom|target-agglom-nfc|admission|auto-admission|gate|advisor|target|all}" >&2
    exit 2
    ;;
esac
