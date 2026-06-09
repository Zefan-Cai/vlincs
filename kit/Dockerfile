# CPU-only kit image: the gallery library + ingest API + scorer + viz API + CLI. No torch, no weights.
# Build context is the repo root (see docker-compose `context: ..`).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY kit/requirements.txt kit/requirements.txt
# reid_hota is internal (not on public PyPI). Install via the Novateur devpi index (it proxies upstream
# PyPI for the public deps too). If your build host can't reach it, vendor a reid_hota wheel into
# kit/vendor/ and pip-install that instead (see requirements.txt).
RUN pip install --no-cache-dir \
    --extra-index-url http://euclid.novateur.com:48731/novateur/dev/+simple/ \
    --trusted-host euclid.novateur.com \
    -r kit/requirements.txt
COPY vlincs_gallery/ vlincs_gallery/
COPY db/ db/
COPY kit/ kit/
ENV PYTHONPATH=/app:/app/kit DATA_ROOT=/data PGHOST=db PGPORT=5432
WORKDIR /app/kit
ENTRYPOINT ["/app/kit/entrypoint.sh"]
CMD ["serve"]
