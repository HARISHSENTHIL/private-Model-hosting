#!/usr/bin/env bash
set -Eeuo pipefail

: "${INTERNAL_API_KEY:?INTERNAL_API_KEY must be set}"
: "${CIVITAI_API_TOKEN:?CIVITAI_API_TOKEN must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"

python3 model_downloader.py

python3 main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --disable-auto-launch \
    --disable-metadata &
COMFY_PID=$!
SHIM_PID=""

shutdown() {
    trap - EXIT TERM INT
    if [[ -n "${SHIM_PID}" ]]; then
        kill "${SHIM_PID}" 2>/dev/null || true
    fi
    kill "${COMFY_PID}" 2>/dev/null || true
    wait "${SHIM_PID}" "${COMFY_PID}" 2>/dev/null || true
}
trap shutdown EXIT TERM INT

python3 - <<'PY'
import time
import urllib.error
import urllib.request

deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2):
            break
    except (OSError, urllib.error.URLError):
        time.sleep(1)
else:
    raise SystemExit("ComfyUI did not become ready within 300 seconds")
PY

uvicorn shim:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --proxy-headers \
    --forwarded-allow-ips 127.0.0.1 &
SHIM_PID=$!

set +e
wait -n "${COMFY_PID}" "${SHIM_PID}"
status=$?
set -e
exit "${status}"
