# Production serving architecture

Last verified against upstream documentation: 2026-11-03.

## Request flow

```text
proxyllm / private reverse proxy
  ├─ chat  :8001 → SGLang → Huihui Qwen3.8 27B on GPU 0
  ├─ chat  :8004 → SGLang → Qwen3-VL-30B-A3B Instruct FP8 on GPU 1  (public: qtextvl.heyopen.ai)
  ├─ image :8002 → vLLM-Omni → Qwen Image 2.1 on GPU 3
  └─ image :8003 → bounded FastAPI shim → ComfyUI → LUSTIFY on GPU 2
```

`qtextvl.heyopen.ai` is the public hostname the reverse proxy terminates TLS for and
forwards to `127.0.0.1:8004`. That proxy (TLS, auth, rate limiting) lives outside
this repository — nothing here changes for a hostname mapping; it only needs DNS
pointed at the host and the proxy's vhost/route config updated.

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
request limit is therefore reduced to 8. Active and queued requests are
bounded separately so overload fails predictably.

Do not enable FP8 weights or KV cache merely to increase a benchmark number.
Evaluate any quantized checkpoint against the BF16 deployment first.

### VLM: SGLang 0.5.19

Qwen3-VL family support landed in SGLang v0.5.3; v0.5.19 is well past that, so
the service reuses the same pinned runtime image as the LLM rather than
tracking a second one. Unlike the LLM, this model is **not** abliterated —
stock-aligned Instruct behavior is intentional here, chosen for general
image/video understanding rather than uncensored chat.

The service pins `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` to Hugging Face revision
`d9748a51ae66354c4dad665aab2c71f26cf2c8cd` and runs on GPU 1 with tensor
parallel size 1. It is a mixture-of-experts checkpoint (30B total / 3B active
parameters per token), so its ~30 GB of FP8 weights leave far more headroom for
KV cache than a dense model of similar total size, and per-token compute is
cheap relative to its quality. Apache-2.0 licensed; no restricted/research
Compose profile is required.

`VLM_MAX_RUNNING_REQUESTS` and `VLM_MAX_QUEUED_REQUESTS` bound admission the
same way as the LLM service. The initial limits are set lower than the LLM's,
since image and video inputs consume substantially more context per request
than text tokens alone — tune only after measuring actual multimodal request
sizes.

Tool calling is enabled via `--tool-call-parser qwen`, which converts Qwen's
native `<tool_call>` tag syntax into the OpenAI-compatible structured
`tool_calls` response field. Without this flag SGLang returns the tag
verbatim as plain text with `finish_reason: "stop"` instead of
`"tool_calls"` — confirm with `--help` on the deployed SGLang version if the
Qwen3-VL checkpoint revision ever changes, since parser names are tied to
the model family's chat template, not guaranteed stable across releases.

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

Current ComfyUI supports Krea 2 and Qwen3-VL conditioning natively; no custom
node pack is required. The service uses:

- LUSTIFY v10 Krea 2 Turbo BF16, Civitai file `3015314`, SHA-256 pinned. This
  is the 8-step guidance- and timestep-distilled checkpoint, recommended by
  the LUSTIFY author for inference (the companion RAW checkpoint, Civitai file
  `3015315`, is meant for training/fine-tuning and is not pulled by default).
- Qwen3-VL 4B BF16 text encoder at an immutable Hugging Face revision.
- Qwen Image VAE at an immutable Hugging Face revision — the VAE Krea 2 is
  actually trained against (do not substitute the similarly-sized Wan 2.1 VAE;
  it is a different checkpoint and silently degrades decode quality).
- A native-node-only API workflow: no model-sampling shift node (Turbo's
  distillation bakes in its own schedule), Euler/Simple, 8 steps, CFG 1.0
  (ComfyUI's "no guidance" value).

The model downloader streams to a temporary file, verifies SHA-256, fsyncs, and
atomically renames. Existing files are verified again on restart.

ComfyUI executes one render at a time. The API shim admits one running request
plus a configurable bounded queue, validates dimensions/steps/count, uses
constant-time bearer comparison, cleans generated files, exposes Prometheus
metrics, and returns 503 on overload or failed readiness.

## Capacity model

| Resource | Initial setting | Overload behavior |
|---|---:|---|
| LLM active requests | 8 | queued by SGLang |
| LLM total queued | 128 | reject at engine limit |
| VLM active requests | 16 | queued by SGLang |
| VLM total queued | 64 | reject at engine limit |
| Qwen Image scheduler capacity | 3 (testing, was 1) | step-wise scheduled |
| Qwen Image waiting queue | Not bounded by the engine | Bound it in the upstream API gateway |
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
