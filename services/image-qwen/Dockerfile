# Diffusers' QwenImage21Pipeline is not yet in a stable pip release per
# Qwen-Image-2.1's own model card — install from source, pinned to a verified
# commit SHA. Do not leave this on a moving branch ref.
ARG DIFFUSERS_GIT_REF=REPLACE_WITH_VERIFIED_COMMIT_SHA

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

# CUDA-matched torch build first so diffusers' install doesn't pull a CPU wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

RUN git clone https://github.com/huggingface/diffusers.git /opt/diffusers \
    && cd /opt/diffusers \
    && git checkout "${DIFFUSERS_GIT_REF}" \
    && pip install --no-cache-dir -e .

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV HF_HUB_ENABLE_HF_TRANSFER=1
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
