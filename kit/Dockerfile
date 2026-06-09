# CPU-only kit image: the gallery library + ingest API + scorer + viz API + CLI. No torch, no weights.
# Build context is the repo root (see docker-compose `context: ..`).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY kit/requirements.txt kit/requirements.txt
# reid_hota is internal — if your build host can't reach the devpi index, vendor a wheel into kit/vendor/
# (see requirements.txt). The rest are public PyPI.
RUN pip install --no-cache-dir -r kit/requirements.txt
COPY vlincs_gallery/ vlincs_gallery/
COPY scripts/gallery_db.py scripts/gallery_db.py
COPY db/ db/
COPY kit/ kit/
ENV PYTHONPATH=/app:/app/kit DATA_ROOT=/data PGHOST=db PGPORT=5432
WORKDIR /app/kit
ENTRYPOINT ["/app/kit/entrypoint.sh"]
CMD ["serve"]
