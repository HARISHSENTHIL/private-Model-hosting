#!/usr/bin/env bash
set -Eeuo pipefail

: "${INTERNAL_API_KEY:?INTERNAL_API_KEY must be set}"

exec sglang serve \
    --trust-remote-code \
    --model-path Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --revision d9748a51ae66354c4dad665aab2c71f26cf2c8cd \
    --served-model-name qwen3-vl-30b-a3b-instruct \
    --tp-size 1 \
    --context-length "${VLM_CONTEXT_LENGTH:-128000}" \
    --mem-fraction-static 0.80 \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --chunked-prefill-size 8192 \
    --max-running-requests "${VLM_MAX_RUNNING_REQUESTS:-16}" \
    --max-queued-requests "${VLM_MAX_QUEUED_REQUESTS:-64}" \
    --schedule-policy fcfs \
    --enable-metrics \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "${INTERNAL_API_KEY}"
