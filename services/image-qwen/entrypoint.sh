#!/usr/bin/env bash
set -Eeuo pipefail

: "${INTERNAL_API_KEY:?INTERNAL_API_KEY must be set}"

exec vllm serve Qwen/Qwen-Image-2.1 \
    --omni \
    --revision d26bb61231c349cf6b7896fa83353113880e1ba3 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "${INTERNAL_API_KEY}" \
    --dtype bfloat16 \
    --step-execution \
    --max-num-seqs "${QWEN_IMAGE_MAX_NUM_SEQS:-4}" \
    --max-num-active-seqs "${QWEN_IMAGE_MAX_NUM_SEQS:-4}" \
    --max-num-queued-reqs "${QWEN_IMAGE_MAX_QUEUED_REQUESTS:-36}" \
    --scheduling-policy fcfs
