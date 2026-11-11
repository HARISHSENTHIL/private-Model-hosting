# Pins the Tencent-Hunyuan/Hunyuan3D-2 GitHub CODE repo at main's current HEAD
# (confirmed via `git ls-remote` on 2026-11-11) - this repo has no version
# tags, unlike image-lustify's ComfyUI dependency, so there is no tag to pin
# to instead. Re-verify this SHA is still what you intend before any rebuild;
# it will silently stay pinned to this exact commit either way.
ARG HUNYUAN3D_GIT_REF=f8db63096c8282cb27354314d896feba5ba6ff8a

FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG HUNYUAN3D_GIT_REF
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="9.0"

# Same IPv4/HTTPS mirror workaround as services/image-lustify - this host
# blocks Ubuntu's plain-HTTP mirrors and has unreliable IPv6.
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 \
    && sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g; s|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git python3 python3-pip python3-venv \
        build-essential ninja-build \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip==25.2 setuptools==80.9.0 wheel==0.45.1 \
    && pip install \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu129

# devel (not runtime) base is required here, unlike image-lustify: the two
# custom CUDA extensions below invoke nvcc via torch.utils.cpp_extension,
# which the runtime image does not ship.
WORKDIR /root/hunyuan3d
# requirements.txt already pins trimesh, rembg, and onnxruntime (confirmed
# by reading it at this exact commit) - no separate install needed for those.
RUN git clone "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git" . \
    && git checkout "${HUNYUAN3D_GIT_REF}" \
    && test "$(git rev-parse HEAD)" = "${HUNYUAN3D_GIT_REF}" \
    && pip install -r requirements.txt \
    && pip install -e .

RUN cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install \
    && cd ../differentiable_renderer && python3 setup.py install

COPY requirements-shim.txt /opt/private-model-hosting/requirements-shim.txt
RUN pip install -r /opt/private-model-hosting/requirements-shim.txt

COPY model_downloader.py shim.py /root/hunyuan3d/
COPY entrypoint.sh /opt/private-model-hosting/entrypoint.sh
RUN chmod 0555 /opt/private-model-hosting/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/opt/private-model-hosting/entrypoint.sh"]
