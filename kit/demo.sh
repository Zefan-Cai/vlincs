#!/bin/sh
# One-click MS02 demo that LEAVES THE GALLERY VIEW UP.
#
# `docker compose run --rm app demo` alone is a one-shot: it populates the DB then the app container
# exits, and it never starts viz/ui — so there's nothing serving http://localhost:4200 afterwards.
# This brings up the PERSISTENT stack (db + pgadmin + viz + ui) pointed at MS02, populates it with the
# shipped demo data, then leaves everything running so you can actually explore the result.
set -e
cd "$(dirname "$0")"

echo "[demo.sh] building containers to account for changes..."
docker compose build app viz

echo "[demo.sh] bringing up the persistent stack (db + pgadmin + viz + ui) for MS02..."
DATASET=ms02 docker compose up -d db pgadmin
# force-recreate viz + ui so they ALWAYS bind to GALLERY_DATASET=ms02 — `up -d` alone will reuse a viz
# container that was started earlier on the default ds1, leaving the view pointed at an empty gallery_ds1
# ("0 videos / 9 cameras" = it's reading DS1, not the MS02 the demo populated).
DATASET=ms02 docker compose up -d --force-recreate viz ui

echo "[demo.sh] populating the gallery with the shipped MS02 demo data (the app container is one-shot)..."
docker compose run --rm app demo

cat <<'EOF'

[demo.sh] Done — the demo data is in the DB and the stack is STILL UP. Explore it:
  Gallery view : http://localhost:4200    decisions / identities / crops, live from the DB
  pgAdmin      : http://localhost:5050    DB inspector (server pre-registered; DB password: gallery)
  viz API      : http://localhost:8077    raw FastAPI endpoints

Stop everything:  docker compose down       (add -v to also wipe the DB volume)
EOF
