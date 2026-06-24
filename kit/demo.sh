#!/usr/bin/env bash
# One-click demo / auto-eval entrypoint.
#
# `docker compose run --rm app demo` alone is a one-shot: it populates the DB then the app container
# exits, and it never starts viz/ui — so there's nothing serving http://localhost:4200 afterwards.
# For local use this script brings up the persistent stack. In CI/headless mode it starts only the DB,
# runs the same demo command, and optionally tees stdout to DEMO_SCORE_FILE for auto-evaluation.
set -euo pipefail
cd "$(dirname "$0")"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://maxwell.novateur.com:9091}"

default=ms02
selected="${1:-$default}"
if [[ $# -gt 0 ]]; then
  shift
fi
extra_cli_args=("$@")
selected_desc="MS02 (DEBUG)"
DS1_flag=0
headless="${DEMO_HEADLESS:-0}"
if [[ -n "${BITBUCKET_BUILD_NUMBER:-}" || -n "${CI:-}" ]]; then
  headless="${DEMO_HEADLESS:-1}"
fi

has_cli_arg() {
  local wanted="$1"
  local arg
  for arg in "${extra_cli_args[@]}"; do
    [[ "$arg" == "$wanted" ]] && return 0
  done
  return 1
}

case "$selected" in
  "ms02"|"debug"|"ds0000"|"ds0")
    selected="ms02"
    ;;
  "ds1"|"ds0001")
    selected="ds1"
    selected_desc="DS0001 (36 IDs w/GT)"
    if [[ "${ALLOW_LEGACY_DS1_GALLERY_DEMO:-0}" != "1" ]]; then
      cat >&2 <<'EOF'
[kit/demo.sh] DS1 here is the legacy gallery/weak-graph demo, not the WISC no-anchor best replay.
[kit/demo.sh] It can score around 0.12 with weak bbox-auto forced output and should not be used to reproduce 0.668.
[kit/demo.sh] Run from the repository root instead:
  ./demo.sh

To intentionally run the legacy gallery demo, set:
  ALLOW_LEGACY_DS1_GALLERY_DEMO=1
EOF
      exit 2
    fi
    # The DS1 (MLflow) extras only PULL inputs from MLflow. If the offline Git LFS bundle is pulled
    # (demo_data/ds1/ holds real files, not LFS pointers), the base kit + pyyaml runs DS1 with no
    # MLflow / SDK / devpi. Otherwise build the MLflow extras to fetch the inputs.
    if find demo_data/ds1 -name '*.parquet' -size +100k 2>/dev/null | grep -q .; then
      DS1_flag=0
      echo "[demo.sh] DS1 offline bundle present -> base-kit build (no MLflow/SDK/devpi)"
    else
      DS1_flag=1
      echo "[demo.sh] DS1 offline bundle not pulled -> MLflow build ('git lfs pull' for offline)"
    fi
    ;;
  *)
    selected=$default
    ;;
esac
echo "Selected -> $selected"

echo "[demo.sh] building containers to account for changes..."
if [[ "$headless" == "1" ]]; then
  WITH_DS1=$DS1_flag docker compose build app
else
  WITH_DS1=$DS1_flag docker compose build app viz
fi

if [[ "$headless" == "1" ]]; then
  echo "[demo.sh] headless mode: bringing up db only for $selected_desc..."
  DATASET=$selected docker compose up -d db
else
  echo "[demo.sh] bringing up the persistent stack (db + pgadmin + viz + ui) for $selected_desc..."
  DATASET=$selected docker compose up -d db pgadmin
  DATASET=$selected docker compose up -d --force-recreate viz ui
fi

echo "[demo.sh] populating the gallery with the $selected_desc demo data (the app container is one-shot)..."
default_cli_args=()
if [[ "$selected" == "ds1" && "${WEAK_RESOLVE:-1}" == "1" ]]; then
  has_cli_arg "--weak-source" || default_cli_args+=(--weak-source "${WEAK_SOURCE:-bbox-auto}")
  has_cli_arg "--weak-resolve" || default_cli_args+=(--weak-resolve)
  has_cli_arg "--weak-embedding-role" || default_cli_args+=(--weak-embedding-role "${WEAK_EMBEDDING_ROLE:-resolve}")
  has_cli_arg "--weak-min-dets" || default_cli_args+=(--weak-min-dets "${WEAK_MIN_DETS:-1}")
  if [[ -n "${WEAK_LABEL_CSV:-}" ]]; then
    has_cli_arg "--weak-label-csv" || default_cli_args+=(--weak-label-csv "$WEAK_LABEL_CSV")
  elif [[ "${AUTO_WEAK_LABELS:-1}" == "1" ]]; then
    has_cli_arg "--auto-weak-labels" || default_cli_args+=(--auto-weak-labels)
  fi
fi

run_cmd=(docker compose run --rm app demo --dataset "$selected" "${default_cli_args[@]}" "${extra_cli_args[@]}")
if [[ -n "${DEMO_SCORE_FILE:-}" ]]; then
  "${run_cmd[@]}" 2>&1 | tee "$DEMO_SCORE_FILE"
else
  "${run_cmd[@]}"
fi

if [[ "$headless" == "1" ]]; then
  echo
  echo "[demo.sh] Done — headless auto-eval run completed for $selected_desc."
  exit 0
fi
cat <<'EOF'

[demo.sh] Done — the demo data is in the DB and the stack is STILL UP. Explore it:
  Gallery view : http://localhost:4200    decisions / identities / crops, live from the DB
  pgAdmin      : http://localhost:5050    DB inspector (server pre-registered; DB password: gallery)
  viz API      : http://localhost:8077    raw FastAPI endpoints

Stop everything:  docker compose down       (add -v to also wipe the DB volume)
EOF
