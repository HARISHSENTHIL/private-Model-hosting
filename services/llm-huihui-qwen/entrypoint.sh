#!/usr/bin/env bash
set -Eeuo pipefail

: "${INTERNAL_API_KEY:?INTERNAL_API_KEY must be set}"

exec sglang serve \
    --trust-remote-code \
    --model-path huihui-ai/Huihui-Qwen3.8-27B-abliterated \
    --revision 739e3c5b89849f6c238ce1e5b70008612ae42cdd \
    --served-model-name huihui-qwen3-27b-abliterated \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --context-length 32768 \
    --mem-fraction-static 0.80 \
    --attention-backend flashinfer \
    --chunked-prefill-size 2048 \
    --max-prefill-tokens 4096 \
    --disable-prefill-cuda-graph \
    --cuda-graph-max-bs-decode 8 \
    --max-running-requests "${LLM_MAX_RUNNING_REQUESTS:-8}" \
    --max-queued-requests "${LLM_MAX_QUEUED_REQUESTS:-128}" \
    --schedule-policy fcfs \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --mamba-full-memory-ratio 4.59 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-ssm-dtype float32 \
    --enable-metrics \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "${INTERNAL_API_KEY}"
