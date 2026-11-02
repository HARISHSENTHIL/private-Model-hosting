# Qwen3.5's hybrid Gated-DeltaNet + full-attention architecture needs SGLang's
# main branch (not yet in a stable pip release as of this research) — see
# docs/ARCHITECTURE.md. Pin this to an exact, verified commit SHA before
# building for real; do not leave this on a moving branch ref.
ARG SGLANG_GIT_REF=REPLACE_WITH_VERIFIED_COMMIT_SHA

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/sglang
RUN git clone https://github.com/sgl-project/sglang.git . \
    && git checkout "${SGLANG_GIT_REF}"

RUN pip install --no-cache-dir -e "python[all]"

ENV HF_HUB_ENABLE_HF_TRANSFER=1
EXPOSE 8000

COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Single GPU (docker-compose.yml device_ids ["0"]) — the 27B model fits on
# one 80GB H100, so no --tp flag is passed.
# --mm-attention-backend fa3: SGLang's recommended multimodal attention
# backend for H100 (this checkpoint's vision tower is unused for text chat,
# but the flag affects the hybrid-attention path generally per SGLang's docs).
# --mamba-cache-mode / GDN-specific flags: check SGLang's current Qwen3.5 docs
# at build time — flag names have changed across versions during this
# architecture's rollout.
ENTRYPOINT ["/opt/entrypoint.sh"]
