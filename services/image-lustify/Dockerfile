# LUSTIFY V10 "Krea 2" via headless ComfyUI (API mode) — see
# docs/ARCHITECTURE.md for why ComfyUI instead of raw Diffusers: Krea 2 plus
# this checkpoint's qwen3vl_4b text encoder + Wan 2.1 VAE combination is new
# enough that node/sampler support is more likely to land in ComfyUI's
# ecosystem first.
#
# OPEN ITEM (flagged in docs/ARCHITECTURE.md, not yet resolved): which exact
# ComfyUI custom-node pack adds Krea 2 + Wan 2.1 VAE support is unconfirmed.
# Identify it before building this image for real, then replace the
# placeholder `git clone` line below with the actual node pack repo, pinned
# to a verified commit SHA (not a moving branch).

ARG COMFYUI_GIT_REF=REPLACE_WITH_VERIFIED_COMMIT_SHA

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/comfyui

RUN git clone https://github.com/comfyanonymous/ComfyUI.git . \
    && git checkout "${COMFYUI_GIT_REF}" \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

# TODO: clone the actual Krea 2 / Wan 2.1 VAE custom-node pack here once
# identified, pinned to a verified commit SHA, e.g.:
#   RUN git clone --branch <tag> https://github.com/<org>/<repo>.git \
#         custom_nodes/<repo> \
#       && pip install --no-cache-dir -r custom_nodes/<repo>/requirements.txt

COPY requirements-shim.txt .
RUN pip install --no-cache-dir -r requirements-shim.txt

COPY shim.py .
COPY comfy_workflows/ comfy_workflows/

EXPOSE 8000

COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]
