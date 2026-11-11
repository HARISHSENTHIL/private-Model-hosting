#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

warn() {
    echo "WARNING: $*" >&2
}

command -v docker >/dev/null || fail "docker is not installed"
docker compose version >/dev/null || fail "Docker Compose v2 is not available"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is not installed"
[[ -f .env ]] || fail ".env is missing; copy .env.example first"

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${HF_TOKEN:?HF_TOKEN is missing from .env}"
: "${CIVITAI_API_TOKEN:?CIVITAI_API_TOKEN is missing from .env}"
: "${INTERNAL_API_KEY:?INTERNAL_API_KEY is missing from .env}"
: "${MODEL_CACHE_DIR:?MODEL_CACHE_DIR is missing from .env}"

[[ "${HF_TOKEN}" != "hf_replace_me" ]] || fail "HF_TOKEN still has its example value"
[[ "${CIVITAI_API_TOKEN}" != "replace_me" ]] || fail "CIVITAI_API_TOKEN still has its example value"
(( ${#INTERNAL_API_KEY} >= 32 )) || fail "INTERNAL_API_KEY must be at least 32 characters"
[[ "${MODEL_CACHE_DIR}" = /* ]] || fail "MODEL_CACHE_DIR must be an absolute path"

if mode=$(stat -c '%a' .env 2>/dev/null); then
    [[ "${mode}" = "600" ]] || warn ".env permissions are ${mode}; run chmod 600 .env"
elif mode=$(stat -f '%Lp' .env 2>/dev/null); then
    [[ "${mode}" = "600" ]] || warn ".env permissions are ${mode}; run chmod 600 .env"
fi

gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
(( gpu_count >= 4 )) || fail "four NVIDIA GPUs are required; found ${gpu_count}"

reported_cuda=$(nvidia-smi | awk -F'CUDA Version: ' '/CUDA Version:/{split($2,a," "); print a[1]; exit}')
if [[ -n "${reported_cuda}" ]] && ! awk -v version="${reported_cuda}" 'BEGIN { exit !(version + 0 >= 13.0) }'; then
    warn "driver reports CUDA ${reported_cuda}; vLLM-Omni 0.32 requires a CUDA 13-compatible driver"
fi

install -d -m 0750 \
    "${MODEL_CACHE_DIR}/huihui-qwen" \
    "${MODEL_CACHE_DIR}/qwen-image" \
    "${MODEL_CACHE_DIR}/lustify" \
    "${MODEL_CACHE_DIR}/hunyuan3d-2"

available_kb=$(df -Pk "${MODEL_CACHE_DIR}" | awk 'NR==2 {print $4}')
(( available_kb >= 500 * 1024 * 1024 )) || warn "less than 500 GiB is available under MODEL_CACHE_DIR"

docker compose config --quiet

echo "Preflight passed: Compose is valid, ${gpu_count} GPUs are visible, and model storage is ready."
