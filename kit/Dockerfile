# CPU-only kit image: the gallery library + ingest API + scorer + viz API + CLI. No torch, no weights.
# Build context is the repo root (see docker-compose `context: ..`).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY kit/requirements.txt kit/requirements.txt
# Every dependency is on public PyPI — reid_hota included (NIST: github.com/usnistgov/reid_hota). No
# internal index, no auth, nothing to vendor: a clean clone builds anywhere with PyPI reachable.
RUN pip install --no-cache-dir -r kit/requirements.txt
COPY vlincs_gallery/ vlincs_gallery/
COPY db/ db/
COPY kit/ kit/
ENV PYTHONPATH=/app:/app/kit DATA_ROOT=/data PGHOST=db PGPORT=5432
WORKDIR /app/kit
ENTRYPOINT ["/app/kit/entrypoint.sh"]
CMD ["serve"]
