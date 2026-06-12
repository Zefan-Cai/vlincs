#!/bin/sh
# One-click MS02 demo that LEAVES THE GALLERY VIEW UP.
#
# `docker compose run --rm app demo` alone is a one-shot: it populates the DB then the app container
# exits, and it never starts viz/ui — so there's nothing serving http://localhost:4200 afterwards.
# This brings up the PERSISTENT stack (db + pgadmin + viz + ui) pointed at MS02, populates it with the
# shipped demo data, then leaves everything running so you can actually explore the result.
set -e
cd "$(dirname "$0")"

export MLFLOW_TRACKING_URI="http://maxwell.novateur.com:9091"

default=ms02
selected=$1
selected_desc="MS02 (DEBUG)"
DS1_flag=0

case "$selected" in
  "ms02"|"debug"|"ds0000"|"ds0")
    selected="ms02"
    ;;
  "ds1"|"ds0001")
    selected="ds1"
    selected_desc="DS0001 (36 IDs w/GT)"
    DS1_flag=1
    ;;
  *)
    selected=$default
    ;;
esac
echo "Selected -> $selected"

echo "[demo.sh] building containers to account for changes..."
WITH_DS1=$DS1_flag docker compose build app viz

echo "[demo.sh] bringing up the persistent stack (db + pgadmin + viz + ui) for $selected_desc..."
DATASET=$selected docker compose up -d db pgadmin
DATASET=$selected docker compose up -d --force-recreate viz ui

echo "[demo.sh] populating the gallery with the $selected_desc demo data (the app container is one-shot)..."
docker compose run --rm app demo --dataset "$selected"

cat <<'EOF'

[demo.sh] Done — the demo data is in the DB and the stack is STILL UP. Explore it:
  Gallery view : http://localhost:4200    decisions / identities / crops, live from the DB
  pgAdmin      : http://localhost:5050    DB inspector (server pre-registered; DB password: gallery)
  viz API      : http://localhost:8077    raw FastAPI endpoints

Stop everything:  docker compose down       (add -v to also wipe the DB volume)
EOF
