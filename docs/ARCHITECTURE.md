# Production serving architecture

Last verified against upstream documentation: 2026-11-03.

## Request flow

```text
proxyllm / private reverse proxy
  ├─ chat  :8001 → SGLang → Huihui Qwen3.8 27B on GPU 0
  ├─ image :8002 → vLLM-Omni → Qwen Image 2.1 on GPU 3
  └─ image :8003 → bounded FastAPI shim → ComfyUI → LUSTIFY on GPU 2

GPU 1 is deliberately unallocated for a fourth model or a future replica.
```

All public traffic must enter through proxyllm or a private reverse proxy. Host
ports bind to `127.0.0.1` unless `PRIVATE_BIND_ADDRESS` is explicitly changed.
TLS, user authentication, per-tenant quotas, and billing belong at that ingress;
the model services use a shared internal bearer key as a second boundary.

## Engine decisions

### LLM: SGLang 0.5.19

SGLang 0.5.19 is the first stable release with explicit Qwen3.8-27B support and
the last CUDA 12.9 build. The upstream Qwen3.8 recipe includes the hybrid
Gated-DeltaNet state-cache settings used here. The runtime image is pinned by
version and digest instead of compiling a moving Git branch.

The complete BF16 model, pinned to Hugging Face revision
`739e3c5b89849f6c238ce1e5b70008612ae42cdd`, runs on GPU 0 with tensor
parallel size 1. Its roughly
55.6 GB of weights fit on an 80 GB H100, but engine state, activations, attention
KV cache, and recurrent GDN state must share the remaining memory. SGLang still
performs continuous token batching and prefix caching, but the initial active
request limit is therefore reduced to 32. Active and queued requests are
bounded separately so overload fails predictably.

Do not enable FP8 weights or KV cache merely to increase a benchmark number.
Evaluate any quantized checkpoint against the BF16 deployment first.

### General image: vLLM-Omni 0.32

Qwen's official Qwen-Image-2.1 repository documents vLLM-Omni online serving,
step-wise execution, request/step continuous batching, prefix-KV reuse, CUDA
graphs, and OpenAI-compatible image routes. Those features directly address the
concurrency and thread-safety problems of the former single-process Diffusers
wrapper.

The service pins Qwen Image to revision
`d26bb61231c349cf6b7896fa83353113880e1ba3` and runs BF16 on GPU 3 with
step-wise execution. Four active sequences
and 32 waiting requests are initial limits. Reduce active sequences if 2K/edit
workloads approach GPU memory limits; increase only after a measured soak test.

The official vLLM-Omni 0.32 image uses CUDA 13.0. Upgrade the host driver before
deployment if it cannot start CUDA 13 containers.

### Restricted image: native ComfyUI 0.43

Current ComfyUI supports Krea 2, Qwen3-VL conditioning, and AuraFlow model
sampling natively; no custom node pack is required. The service uses:

- LUSTIFY v10 Krea 2 RAW BF16, Civitai file `3015314`, SHA-256 pinned.
- Qwen3-VL 4B BF16 text encoder at an immutable Hugging Face revision.
- Wan 2.1 VAE at an immutable Hugging Face revision, matching the checkpoint
  author's recommendation.
- A native-node-only API workflow: AuraFlow shift 4, Euler/Beta, 52 RAW steps.

The model downloader streams to a temporary file, verifies SHA-256, fsyncs, and
atomically renames. Existing files are verified again on restart.

ComfyUI executes one render at a time. The API shim admits one running request
plus a configurable bounded queue, validates dimensions/steps/count, uses
constant-time bearer comparison, cleans generated files, exposes Prometheus
metrics, and returns 503 on overload or failed readiness.

## Capacity model

| Resource | Initial setting | Overload behavior |
|---|---:|---|
| LLM active requests | 32 | queued by SGLang |
| LLM total queued | 128 | reject at engine limit |
| Qwen Image active sequences | 4 | step-wise scheduled |
| Qwen Image total in flight | 36 | HTTP 503 |
| LUSTIFY active renders | 1 | queued by shim |
| LUSTIFY waiting queue | 32 | HTTP 503 + `Retry-After` |

These values are deployment hypotheses. Capture at minimum p50/p95/p99 latency,
throughput, queue time, HBM high-water mark, failure rate, and generated image
size for the real prompt/resolution mix. Keep 15–20% HBM headroom.

## Operational controls

- Versioned/digest-pinned engine images and checksum-pinned model artifacts.
- Health checks with long first-download/model-load start periods.
- Restart policy, init process, graceful stop window, bounded JSON logs.
- Loopback port binding and least-privilege secret injection per container.
- `no-new-privileges` on all services.
- Metrics: SGLang and vLLM native metrics; LUSTIFY at `/metrics`.
- Model caches persist on host NVMe. Generated LUSTIFY files use a bounded
  tmpfs and are removed after delivery.
- Compose's default profile starts only the LLM. Qwen Image requires the
  `research` profile and LUSTIFY requires the `restricted` profile.

## Required production work outside this repository

1. Obtain commercial licenses for the image checkpoints before paid traffic.
2. Put the services behind TLS and tenant authentication; never expose these
   ports directly to the internet.
3. Enforce tenant rate/concurrency quotas and retry 503 responses with jitter.
4. For LUSTIFY, implement age assurance, consent/NCII protections, prompt and
   output moderation, abuse reporting, audit controls, and applicable legal
   review before any public deployment.
5. Export metrics to Prometheus and alert on unhealthy containers, queue
   saturation, GPU Xid errors, HBM pressure, and elevated 5xx rates.
6. Back up configuration, not generated content. Test node rebuild and model
   cache restoration before launch.

## Upgrade and rollback

Upgrade one engine/model at a time. Pin the new digest, build under a new image
tag, run smoke/evaluation/soak tests, then move traffic. Preserve the prior image
and model cache until the soak window closes. Rollback is a Compose image/config
revert; never mutate a version tag in place.

## Primary upstream references

- SGLang Qwen3.8 serving example: https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/examples.mdx
- SGLang 0.5.19 release: https://github.com/sgl-project/sglang/releases/tag/v0.5.19
- Qwen Image 2.1: https://github.com/QwenLM/Qwen-Image-2.1
- vLLM-Omni Qwen Image recipe: https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image-2.1.md
- ComfyUI Krea 2 guide: https://docs.comfy.org/tutorials/image/krea/krea-2
