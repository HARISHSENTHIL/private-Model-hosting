# Private model hosting

Production-oriented, OpenAI-compatible inference for a single 4× H100 80 GB
node. The stack keeps prompts and outputs on infrastructure you control.

| Service | Engine | GPUs | Endpoint |
|---|---|---:|---|
| Huihui Qwen3.8 27B | SGLang 0.5.19 | 0, TP=1 | `:8001/v1/chat/completions` |
| Qwen Image 2.1 | vLLM-Omni 0.32 | 3 | `:8002/v1/images/*` |
| LUSTIFY v10 Krea 2 Turbo | ComfyUI 0.43 + bounded shim | 2 | `:8003/v1/images/generations` |
| Qwen3-VL-30B-A3B Instruct FP8 | SGLang 0.5.19 | 1, TP=1 | `:8004/v1/chat/completions` |
| Hunyuan3D-2 turbo | in-process bounded shim | 2 (shared with LUSTIFY) | `:8005/v1/3d/generations` |

The BF16 LLM fits on one 80 GB H100, but with substantially less cache and
concurrency headroom than the former TP=2 layout. Qwen3-VL is a vision-capable
(image/video + text) model, stock-aligned (not abliterated), Apache-2.0
licensed, and needs no restricted/research Compose profile.

The images, model revisions, and downloaded artifacts are immutable-version or
SHA-256 pinned. APIs bind to loopback by default, receive only the secrets they
need, and expose health checks and metrics. The LLM and LUSTIFY services have
bounded admission controls; Qwen Image needs an upstream bounded API gateway.

## Production gates

Read [`docs/LICENSES.md`](docs/LICENSES.md) before starting either image model.
As of 2026-11-03, Qwen Image 2.1 is non-commercial without a separate license,
and LUSTIFY/Krea 2 is not cleared for a general paid API. LUSTIFY is therefore
behind the explicit Compose profile `restricted`. Hunyuan3D-2 is behind its
own `gated` profile (an MAU threshold + EU/UK/South Korea exclusion, not a
safety gate); its text-to-3D path additionally inherits Qwen Image's
non-commercial restriction since it calls `image-qwen` internally.

This repository supplies inference infrastructure. It does not by itself supply
the age verification, consent/NCII controls, content moderation, abuse response,
audit retention, or jurisdiction-specific compliance required for a public
adult-image service.

## Host prerequisites

- Linux x86_64, Docker Engine with Compose v2, and NVIDIA Container Toolkit.
- Four H100 80 GB GPUs visible to Docker as IDs `0,1,2,3`.
- An NVIDIA driver compatible with CUDA 13 containers. The SGLang image itself
  uses CUDA 12.9; vLLM-Omni 0.32 uses CUDA 13.0.
- 500 GB or more of fast local NVMe storage.
- Outbound HTTPS to Hugging Face, Civitai, Docker Hub, and GitHub while building
  and downloading models.

Run the preflight before the first build:

```bash
cp .env.example .env
chmod 600 .env
# Fill in the three secrets and set MODEL_CACHE_DIR.
./scripts/preflight.sh
```

The Civitai token shown in earlier development context must be rotated before
this stack is deployed.

## Build and start

The default Compose profile contains the commercially cleared LLM and VLM.
Build and start either restricted image service only after obtaining the
required approvals described above.

```bash
docker compose build llm-huihui-qwen vlm-qwen3
docker compose up -d llm-huihui-qwen vlm-qwen3

# Research/internal evaluation only until Qwen commercial licensing clears:
docker compose --profile research build image-qwen
docker compose --profile research up -d image-qwen

# Restricted/internal evaluation only until licensing and safety gates clear:
docker compose --profile restricted build image-lustify
docker compose --profile restricted up -d image-lustify

# Gated: image-to-3D works alone; text-to-3D additionally needs image-qwen
# (profile "research") running, and inherits its non-commercial restriction.
docker compose --profile gated build hunyuan3d-2
docker compose --profile gated up -d hunyuan3d-2
```

First startup downloads roughly 56 GB for the LLM, 30 GB for the VLM, 33 GB
for Qwen Image, about 34 GB for LUSTIFY, and about 15 GB for Hunyuan3D-2.
Health-check start periods account for this, but actual download time
depends on the host connection.

```bash
docker compose ps
docker compose logs -f llm-huihui-qwen
```

## Admission control and concurrency

- SGLang continuously batches LLM tokens. `LLM_MAX_RUNNING_REQUESTS` caps the
  active batch and `LLM_MAX_QUEUED_REQUESTS` rejects overload instead of growing
  memory without bound.
- vLLM-Omni uses step-wise diffusion scheduling.
  `QWEN_IMAGE_MAX_NUM_SEQS` controls scheduler capacity. Start at `1`: this
  execution mode is experimental and the engine does not expose a supported
  bounded-waiting-queue option in the pinned release. Enforce the public
  request and queue limits in an upstream API gateway before production use.
- ComfyUI renders one LUSTIFY image at a time. The shim keeps a bounded FCFS
  queue and returns HTTP 503 with `Retry-After` when full.
- Hunyuan3D-2's single worker thread generates one mesh at a time; the shim
  keeps a bounded FCFS queue (`HUNYUAN3D_MAX_QUEUE_SIZE`) the same way.
  Pipelines idle-unload from VRAM after `HUNYUAN3D_IDLE_UNLOAD_SECONDS` since
  it shares GPU2 with LUSTIFY.
- SGLang also batches VLM requests. `VLM_MAX_RUNNING_REQUESTS` and
  `VLM_MAX_QUEUED_REQUESTS` bound it the same way as the LLM; the starting
  values are deliberately lower since image/video inputs consume far more
  context per request than text alone.

The defaults are safe starting points, not throughput promises. Benchmark the
actual node, then tune from observed p95 latency, GPU memory, and queue depth.

## Verification

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8005/health

python bench/bench_llm.py \
  --url http://127.0.0.1:8001 \
  --api-key "$INTERNAL_API_KEY"

python bench/bench_image.py \
  --url http://127.0.0.1:8002 \
  --api-key "$INTERNAL_API_KEY"

python bench/bench_mesh.py \
  --url http://127.0.0.1:8005 \
  --api-key "$INTERNAL_API_KEY"
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the serving decisions,
capacity controls, operations, and rollback guidance.
