#!/bin/sh
# serve -> the viz API (FastAPI); anything else -> passthrough (the `app` service overrides this with cli.py)
if [ "$1" = "serve" ]; then
  exec uvicorn vlincs_gallery.viz.app:app --host 0.0.0.0 --port 8077
fi
exec "$@"
