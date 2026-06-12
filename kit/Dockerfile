# CPU-only kit image: the gallery library + ingest API + scorer + viz API + CLI. No torch, no weights.
# Build context is the repo root (see docker-compose `context: ..`).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY kit/requirements.txt kit/requirements.txt
# Every dependency is on public PyPI — reid_hota included (NIST: github.com/usnistgov/reid_hota). No
# internal index, no auth, nothing to vendor: a clean clone builds anywhere with PyPI reachable.
RUN pip install --no-cache-dir -r kit/requirements.txt

# Optional DS1-demo extras (mlflow + vlincs-sdk[mlflow], the latter from the internal devpi index). The
# default image stays public/MS02-only; build with `--build-arg WITH_DS1=1` for the on-network DS1 demo.
ARG WITH_DS1=0
COPY kit/requirements-ds1.txt kit/requirements-ds1.txt
RUN if [ "$WITH_DS1" = "1" ]; then \
      pip install --no-cache-dir \
        --extra-index-url http://euclid.novateur.com:48731/novateur/dev/+simple/ \
        --trusted-host euclid.novateur.com \
        -r kit/requirements-ds1.txt; \
    fi
COPY vlincs_gallery/ vlincs_gallery/
COPY db/ db/
COPY kit/ kit/
ENV PYTHONPATH=/app:/app/kit DATA_ROOT=/data PGHOST=db PGPORT=5432
WORKDIR /app/kit
ENTRYPOINT ["/app/kit/entrypoint.sh"]
CMD ["serve"]
