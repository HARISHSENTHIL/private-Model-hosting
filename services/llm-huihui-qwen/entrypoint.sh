#!/bin/sh
set -eu

: "${INTERNAL_API_KEY:?INTERNAL_API_KEY must be set}"

exec python3 -m sglang.launch_server \
    --model-path huihui-ai/Huihui-Qwen3.8-27B-abliterated \
    --served-model-name huihui-qwen3-27b-abliterated \
    --mm-attention-backend fa3 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "${INTERNAL_API_KEY}"
