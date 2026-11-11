#!/usr/bin/env bash
set -Eeuo pipefail

: "${INTERNAL_API_KEY:?INTERNAL_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"

python3 model_downloader.py

# Unlike image-lustify (shim process + separate ComfyUI process), there is
# only one process here: the pipeline runs in-process inside the shim's own
# worker thread, so there is nothing else to start or wait on.
exec uvicorn shim:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --proxy-headers \
    --forwarded-allow-ips 127.0.0.1
