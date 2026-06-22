#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <run_dir> <area_config> <assignment_csv> [assignment_csv ...]" >&2
  exit 2
fi

RUN_DIR="$1"
AREA_CONFIG="$2"
shift 2

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DATA_ROOT="${DATA_ROOT:-/mnt/localssd/vlincs_reid_data}"

mkdir -p "${RUN_DIR}"

if [[ "${AREA_CONFIG}" == b64:* ]]; then
  AREA_CONFIG="$("${PYTHON_BIN}" - "${AREA_CONFIG#b64:}" <<'PY'
import base64
import sys
print(base64.b64decode(sys.argv[1]).decode("utf-8"))
PY
)"
elif [[ "${AREA_CONFIG}" == @* ]]; then
  AREA_CONFIG="$(cat "${AREA_CONFIG#@}")"
fi
EXPECTED_CONFIG_NAME="${AREA_CONFIG%%;*}"

for ASSIGNMENT_CSV in "$@"; do
  STEM="$(basename "${ASSIGNMENT_CSV}" .csv)"
  FULL_JSON="${RUN_DIR}/${STEM}_full_export.json"
  FULL_ZIP="${RUN_DIR}/${STEM}_full_export.zip"
  DENSITY_JSON="${RUN_DIR}/${STEM}_density_simple.json"
  DENSITY_ZIP="${RUN_DIR}/${STEM}_density_simple.zip"
  AREA_JSON="${RUN_DIR}/${STEM}_density_p005_area.json"

  echo "STAGE ${STEM} export_full_zip_no_score"
  "${PYTHON_BIN}" "${REPO_ROOT}/kit/export_no_anchor_assignment_zip.py" \
    --assignment-csv "${ASSIGNMENT_CSV}" \
    --json "${FULL_JSON}" \
    --zip-out "${FULL_ZIP}"

  echo "STAGE ${STEM} density_simple_sourcezip"
  FILTER_SKIP_SCORE_ARGS=()
  if [[ "${NO_ANCHOR_FILTER_SKIP_SCORE:-0}" == "1" ]]; then
    FILTER_SKIP_SCORE_ARGS+=(--skip-score)
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/kit/no_anchor_pervideo_filter_selector.py" \
    --source-zip "${FULL_ZIP}" \
    --policies density_simple \
    --json "${DENSITY_JSON}" \
    --zip-out "${DENSITY_ZIP}" \
    "${FILTER_SKIP_SCORE_ARGS[@]}"

  echo "STAGE ${STEM} p005_area_on_density_zip"
  "${PYTHON_BIN}" "${REPO_ROOT}/kit/evaluate_submission_detection_filter.py" \
    --submission-zip "${DENSITY_ZIP}" \
    --config "${AREA_CONFIG}" \
    --json "${AREA_JSON}"

  "${PYTHON_BIN}" - "${AREA_JSON}" "${EXPECTED_CONFIG_NAME}" <<'PY'
import json
import sys

path, expected = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
rows = data.get("rows") or []
if not rows:
    raise SystemExit(f"{path}: missing rows")
row = rows[0]
actual = str(row.get("config_name", ""))
if expected and actual != expected:
    raise SystemExit(f"{path}: config_name {actual!r} != expected {expected!r}")
if expected == "p005_area" and int(row.get("dropped_rows", 0)) <= 0:
    raise SystemExit(f"{path}: p005_area did not drop any rows; check config plumbing")
PY

  echo "STAGE ${STEM} done"
done
