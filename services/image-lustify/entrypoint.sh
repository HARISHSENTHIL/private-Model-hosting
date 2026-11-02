#!/bin/sh
set -eu

: "${INTERNAL_API_KEY:?INTERNAL_API_KEY must be set}"

# ComfyUI's own server, headless/API mode, listening only on localhost —
# the FastAPI shim (the only thing that should be reachable from outside
# the container) proxies to it, not the other way around.
python3 main.py --listen 127.0.0.1 --port 8188 &
COMFY_PID=$!

# Give ComfyUI a moment to come up before the shim starts health-checking it.
sleep 5

uvicorn shim:app --host 0.0.0.0 --port 8000 &
SHIM_PID=$!

trap 'kill "$COMFY_PID" "$SHIM_PID" 2>/dev/null' TERM INT
wait -n "$COMFY_PID" "$SHIM_PID"
