ARG COMFYUI_GIT_REF=v0.43.0

FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

ARG COMFYUI_GIT_REF
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# The deployment host used for this stack has unreliable IPv6 and blocks
# Ubuntu's plain-HTTP mirrors. Keep package installation on IPv4 HTTPS.
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 \
    && sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g; s|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip==25.2 setuptools==80.9.0 wheel==0.45.1 \
    && pip install \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu129

WORKDIR /root/comfyui
RUN git clone --branch "${COMFYUI_GIT_REF}" --depth 1 \
        https://github.com/Comfy-Org/ComfyUI.git . \
    && test "$(git describe --tags --exact-match)" = "${COMFYUI_GIT_REF}" \
    && pip install -r requirements.txt

COPY requirements-shim.txt /opt/private-model-hosting/requirements-shim.txt
RUN pip install -r /opt/private-model-hosting/requirements-shim.txt

COPY model_downloader.py shim.py /root/comfyui/
COPY comfy_workflows/ /root/comfyui/comfy_workflows/
COPY entrypoint.sh /opt/private-model-hosting/entrypoint.sh
RUN chmod 0555 /opt/private-model-hosting/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/opt/private-model-hosting/entrypoint.sh"]
