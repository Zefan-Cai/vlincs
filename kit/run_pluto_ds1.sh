#!/usr/bin/env bash
set -euo pipefail

job_name="${1:?usage: run_pluto_ds1.sh <job-name>}"

repo_root="${REPO_ROOT:-/mnt/localssd/vlincs_reid_by_search}"
run_root="${RUN_ROOT:-/mnt/localssd/vlincs_reid_runs}"
venv="${VENV:-/mnt/localssd/vlincs_reid_venv}"
data_root="${DATA_ROOT:-/mnt/localssd/vlincs_reid_data}"
log_dir="${run_root}/logs"
log_file="${log_dir}/ds1-${job_name}.log"
submission="${run_root}/ds1_submission_${job_name}.zip"
weak_submission="${run_root}/ds1_weak_submission_${job_name}.zip"
weak_source="${WEAK_SOURCE:-bbox-auto}"
weak_resolve="${WEAK_RESOLVE:-1}"
weak_min_dets="${WEAK_MIN_DETS:-1}"
weak_embedding_role="${WEAK_EMBEDDING_ROLE:-resolve}"
auto_weak_labels="${AUTO_WEAK_LABELS:-1}"
weak_label_csv="${WEAK_LABEL_CSV:-}"
resolve_nfc_k1="${RESOLVE_NFC_K1:-0}"
resolve_nfc_k2="${RESOLVE_NFC_K2:-2}"
resolve_nfc_eta="${RESOLVE_NFC_ETA:-1.0}"
resolve_nfc_exclude_same_camera="${RESOLVE_NFC_EXCLUDE_SAME_CAMERA:-0}"

mkdir -p "${log_dir}"
rm -f "${repo_root}/gallery.py" "${repo_root}/online.py"

export PYTHONPATH="${repo_root}:${repo_root}/kit"
export DATA_ROOT="${data_root}"
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-55433}"
export PGUSER="${PGUSER:-gallery}"
export PGPASSWORD="${PGPASSWORD:-gallery}"
export GALLERY_CODE_SHA="${GALLERY_CODE_SHA:-$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || echo unknown)}"

"${venv}/bin/python" -m py_compile \
  "${repo_root}/vlincs_gallery/gallery.py" \
  "${repo_root}/vlincs_gallery/weak_graph.py" \
  "${repo_root}/vlincs_gallery/weak_labels.py" \
  "${repo_root}/kit/online.py" \
  "${repo_root}/kit/demo.py" \
  "${repo_root}/kit/cli.py"

if [[ -x /usr/lib/postgresql/14/bin/dropdb ]]; then
  /usr/lib/postgresql/14/bin/dropdb \
    -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" --if-exists gallery_ds1 >/dev/null 2>&1 || true
fi

cd "${repo_root}/kit"
extra_args=()
if [[ "${weak_resolve}" == "1" ]]; then
  extra_args+=(--weak-source "${weak_source}" --weak-resolve --weak-embedding-role "${weak_embedding_role}" --weak-min-dets "${weak_min_dets}")
  if [[ -n "${weak_label_csv}" ]]; then
    extra_args+=(--weak-label-csv "${weak_label_csv}")
  elif [[ "${auto_weak_labels}" == "1" ]]; then
    extra_args+=(--auto-weak-labels)
  fi
fi
if [[ "${resolve_nfc_k1}" != "0" ]]; then
  extra_args+=(--resolve-nfc-k1 "${resolve_nfc_k1}" --resolve-nfc-k2 "${resolve_nfc_k2}" --resolve-nfc-eta "${resolve_nfc_eta}")
  if [[ "${resolve_nfc_exclude_same_camera}" == "1" ]]; then
    extra_args+=(--resolve-nfc-exclude-same-camera)
  fi
fi

"${venv}/bin/python" demo.py --dataset ds1 --submit "${weak_submission}" "${extra_args[@]}" >"${log_file}" 2>&1

tail -n 80 "${log_file}"
ls -lh "${weak_submission}"
