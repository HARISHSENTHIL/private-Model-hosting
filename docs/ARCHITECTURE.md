# Private-tier self-hosted model serving (4x H100 80GB)

> This is the approved architecture plan for this repo. Update the numbers marked
> **estimated** with real measurements as each service comes up (see `Verification`
> at the bottom) — don't let this file drift into fiction.

## Context

proxyllm's "Private" privacy tier needs self-owned-GPU inference (vs. routing to
third-party vendors like OpenRouter/Wiro/Surplus) for zero-data-retention
guarantees. Hardware: 4x NVIDIA H100 80GB SXM (NVLink, single node, CUDA 12.x).

Three model slots, researched and locked in:

- **Chat LLM**: `huihui-ai/Huihui-Qwen3.8-27B-abliterated` — uncensored,
  Apache-2.0. **Not a plain dense 28B transformer.** Its real `config.json`
  architecture is `Qwen3_5ForConditionalGeneration` — Alibaba's hybrid Qwen3.5
  design: 64 layers total, only 16 of them (every 4th) are standard
  full-attention/KV-cache layers; the other 48 are Gated DeltaNet
  linear-attention layers with a fixed-size recurrent state that does **not**
  grow with context length. Also natively vision-capable (image/video token +
  vision tower) — unused for this text-chat use case, but available for free if
  ever needed.
- **General image (TTI + instruction edit + subject/reference personalization)**:
  `Qwen/Qwen-Image-2.1` (non-Turbo). Confirmed via the official HF model card:
  7B params (32 single-stream DiT layers), one checkpoint covers text-to-image,
  instruction-based local editing (circle/paint/mask-guided), and
  subject/reference personalization (up to 10 reference images, multi-reference
  composition). **License: Qwen Research License (non-commercial)** — commercial
  use needs a separate license from Qwen at `model-business@notice.qwencloud.com`.
  Accepted as an interim risk; see `LICENSES.md`. Revisit before this tier bills
  real customers.
- **NSFW-specialized image**: LUSTIFY V10 "Krea 2" (civitai.com/models/573152).
  Krea 2's base license caps free commercial use at **<$1M annual revenue**
  (above that, needs Krea's enterprise license). LUSTIFY V10's own
  version-specific license tab still needs a **manual logged-in check** on
  Civitai (WebFetch got a 403 on the page during research) — see `LICENSES.md`.

Hard constraints: no quality-degrading shortcuts (no distilled/smaller
substitutes, no quantization beyond what's documented as effectively lossless),
and all 3 model slots must stay live and simultaneously servable — this is a
production stack, not a batch job.

## GPU allocation

| GPUs | Model | Why |
|---|---|---|
| 0 + 1 (TP=2, NVLink) | Huihui-Qwen3.8-27B-abliterated | Only the LLM benefits from tensor-parallel splitting (KV-cache-bound concurrency scales with freed VRAM); diffusion models don't gain from TP the same way. |
| 2 | LUSTIFY V10 (Krea 2) | Dedicated GPU, own queue — isolated from the Qwen pair in case its tooling (ComfyUI) needs different handling than the Diffusers-based service. |
| 3 | Qwen-Image-2.1 (T2I + edit + personalization, one checkpoint) | ~33GB full pipeline — comfortably fits one 80GB GPU with large headroom for activations and/or a second worker process if benchmarking shows it helps. |

This uses all 4 GPUs, keeps every model slot live concurrently, and gives the
LLM — the one model type that actually benefits from it — the multi-GPU
allocation.

## Serving framework per model (with rejected alternatives)

**LLM — SGLang, not vLLM.** Generic "vLLM is the default" advice doesn't apply
cleanly here because this checkpoint is the hybrid Qwen3.5 architecture above,
not a plain dense transformer — the real question is which engine has specific
support for Gated DeltaNet + full-attention hybrid layers.

- **SGLang** (chosen): official docs/cookbook explicitly cover Qwen3.5's hybrid
  attention and its vision-language serving, with H100-specific guidance
  (`--mm-attention-backend fa3`).
- **vLLM** (rejected): supports Gated DeltaNet too (community Triton kernels,
  `fused_recurrent_gated_delta_rule`), but its own docs flag gaps for this class
  — `--mamba-cache-mode align` is required, align-mode prefix caching is still
  experimental, and GDN backends don't support vLLM's batch-invariant mode.
- **TensorRT-LLM** (rejected): ~28min engine-compile step per version, worst fit
  if this fine-tune gets updated, no clear evidence of maturity for this hybrid
  architecture yet.
- Both SGLang and vLLM ship a native OpenAI-compatible server, so either way the
  integration into proxyllm needs **zero custom litellm code** — confirmed
  against proxyllm's own `config/chat/*.yaml`, which already wires up
  OpenAI-compatible self-hosted/third-party servers via
  `model: openai/<name>` + `api_base` + `api_key` (see the Surplus entries,
  `config/chat/primary.yaml` ~line 701).
- **Open risk**: confirm SGLang's Qwen3.5 support actually covers this specific
  `huihui-ai` third-party fine-tune (not the official Qwen release) once loaded
  for real — architecture-class support usually transfers across fine-tunes of
  the same base, but this hasn't been verified end to end.

**Qwen-Image-2.1 — direct Diffusers + a thin FastAPI wrapper.** Diffusers ships
official `QwenImage21Pipeline` support for this exact checkpoint (one pipeline
class handles T2I, edit, and multi-reference personalization via its image-input
args) — no community-node dependency, no compatibility lag. Needs Diffusers
built from source + `transformers>=5.17.0` per the model card. proxyllm's OpenAI
image provider (`litellm/llms/openai/image_generation/dall_e_2_transformation.py`,
`convert_dict_to_response.py`) does zero vendor-specific translation — it just
parses whatever JSON comes back into `ImageResponse`
(`{"data":[{"url"|"b64_json":...}], "created":...}`). So as long as our FastAPI
wrapper emits that exact shape, this is also a **zero-litellm-code** integration
— proxyllm already has a self-hosted precedent for this exact pattern
(`config/image/primary.yaml` lines 18-21, `openai/flux-2-klein-4b` +
`api_base`).

**LUSTIFY V10 (Krea 2) — ComfyUI headless (API mode) behind the same thin
FastAPI→OpenAI-shape wrapper.** Rejected: raw Diffusers — Krea 2 plus this
checkpoint's unusual encoder/VAE combination is new enough that node/sampler
support is more likely to land in ComfyUI's ecosystem first; forcing it through
Diffusers risks a quality-compromising workaround if support is incomplete.
Rejected: exposing ComfyUI's native workflow-JSON API directly to proxyllm —
that wire format doesn't match OpenAI's shape and would need a Wiro-style custom
litellm provider (`litellm/llms/wiro/`: ~150-300 lines per capability, enum +
`ProviderConfigManager` branches). A thin FastAPI translator in front of ComfyUI
gets its checkpoint/node compatibility while keeping the proxyllm integration at
zero litellm code, same as the other two services.

**Quality stance**: BF16 everywhere by default — zero risk. FP8 (W8A8) for the
LLM only is an optional *measured* future upgrade (Neural Magic/Red Hat's
large-scale study found W8A8-FP "effectively lossless" on H100) — never applied
silently, needs an eval pass on this specific checkpoint first. Diffusion
weights stay BF16 outright; there isn't comparable published evidence FP8 is
lossless for these diffusion architectures.

## Per-model detailed breakdown (disk, VRAM, capabilities, libraries, concurrency)

Disk sizes are **measured** — pulled from each model's real file listing (HF API
tree, or civarchive's file-size table for the Civitai-only model), not from
parameter-count arithmetic.

### 1. Huihui-Qwen3.8-27B-abliterated (chat LLM)

- **Disk (measured)**: 55.6 GB total (single safetensors shard set, BF16).
- **VRAM**: ~27.8GB/GPU sharded TP=2 across GPUs 0+1, leaving ~44GB/GPU
  (~88GB combined, at 90% utilization of the ~72GB/GPU usable budget) for KV
  cache + GDN recurrent state + activations.
- **Capabilities**: uncensored multi-turn chat/instruction-following. Also
  natively vision-capable — unused here, available later at no extra model cost.
- **Libraries**: SGLang (chosen) or vLLM, PyTorch (CUDA 12.x matching the host
  driver), `transformers`, the engine's Triton kernels for the GDN layers, NCCL
  for TP=2.
- **Concurrency (estimated from the real architecture, not yet measured)**: only
  16/64 layers keep per-token KV cache. Per-token KV cost = 2(K+V) × 16
  full-attn layers × 4 KV heads × 256 head_dim × 2 bytes (BF16) = **64 KB/token**
  — small for a 28B-class model specifically because the hybrid design avoids
  per-token KV cost on 3/4 of its layers. At ~88GB combined KV budget:
  88e9 / 65536 ≈ 1.34M token-slots. At an 8K-context budget per request, that's
  an upper bound around 160 concurrent requests from attention-KV alone; GDN
  state + activation + engine overhead pulls the realistic number down.
  **Planning range: ~60-150 concurrent chat requests at ~8K context — pin this
  from the real SGLang startup log, not this estimate.**

### 2. Qwen-Image-2.1 (general TTI + instruction edit + reference personalization)

- **Disk (measured, HF API tree)**: 33.1 GB total — `text_encoder/` 17.5GB,
  `transformer/` (the DiT) 14.2GB, `vae/` 1.35GB.
- **VRAM**: ~33GB minimum (all three components must be resident), leaving
  ~47GB of the 80GB for activations at high resolution and/or a second worker.
- **Capabilities (confirmed against the HF model card)**: text-to-image;
  instruction-based local editing (circle/paint/mask-guided); subject/reference
  personalization (up to 10 reference images, multi-reference composition) —
  all via `QwenImage21Pipeline`.
- **Libraries**: Diffusers from source (not yet in a stable pip release per the
  model card), `transformers>=5.17.0`, `accelerate`, `pillow`, `safetensors`,
  CUDA-matched `torch`, FastAPI/uvicorn.
- **License**: Qwen Research License (non-commercial) — accepted interim risk,
  see `LICENSES.md`.
- **Concurrency**: diffusion doesn't batch like autoregressive LLMs — one H100
  is already compute-saturated by a single image at this size, so production
  services run batch size 1 and queue the rest. **1 image at a time** on this
  GPU. Per-image latency is **not yet measured** — benchmark before promising
  any throughput number.

### 3. LUSTIFY V10 "Krea 2" (NSFW-specialized image)

- **Disk (measured via civarchive, Civitai itself 403'd)**: V10 (Krea 2) "Raw"
  full-precision safetensors = 24.48 GB. A "Turbo" (distilled) variant and
  lossy options (fp8/int8 ~12GB, GGUF Q4 7.74GB, Q2 4.68GB) exist — **use Raw,
  full precision**, per the no-quality-compromise constraint. Needs a separate
  `qwen3vl_4b` text encoder + Wan 2.1 VAE (sizes not confirmed — budget
  **+8-9GB estimated**, total ~33GB, confirm once downloaded).
- **VRAM**: ~33GB (estimated) on GPU 2.
- **Capabilities**: NSFW-specialized text-to-image, Krea 2 base.
- **Libraries**: ComfyUI (headless/API mode) — chosen because Krea 2 + this
  encoder/VAE combo is new enough that ComfyUI community nodes are more likely
  to support it first. **Which exact custom-node pack is still unconfirmed** —
  identify this when building the service. Plus FastAPI/uvicorn for the
  translator shim.
- **License**: Civitai didn't show commercial-use terms for V10 specifically —
  **open follow-up**, manual check required, see `LICENSES.md`.
- **Concurrency**: same batch-size-1, one-GPU-one-image-at-a-time reasoning.
  Latency completely unmeasured — benchmark first.

**Combined image concurrency**: with 1 GPU per image model, aggregate
image-generation concurrency across the whole stack is **~2 in-flight renders
at once** — a real ceiling from the 4-GPU budget, not a research gap. If that's
not enough: dedicate more GPUs to images, accept queued wait time during
bursts, or benchmark whether micro-batching (>1) helps without hurting quality
or latency — not assumed to help by default.

**Total measured+estimated disk footprint**: 55.6 + 33.1 + ~33 ≈ **122GB of
model weights alone**. Real provisioning should be **300-500GB** of fast NVMe —
Docker images, SGLang/ComfyUI + dependency trees, ComfyUI custom nodes,
generated-output caching, and room for a second checkpoint version during
upgrades all add on top of the 122GB.

## Verification

1. Bring up `llm-huihui-qwen` alone (verify SGLang actually loads this
   third-party fine-tune of the hybrid Qwen3.5 architecture — the open risk
   above); `curl /v1/chat/completions`, confirm an OpenAI-shaped response; read
   the real KV-cache/concurrency numbers from SGLang's startup log and replace
   the ~60-150 estimate above with them.
2. Bring up `image-qwen`; `curl /v1/images/generations` and `/v1/images/edits`;
   time real per-image latency at the resolution/step count you intend to ship.
3. Bring up `image-lustify`; same latency measurement; **before enabling it for
   paying users**, manually check the Civitai V10 "about this version" / license
   tab (logged in) and log the result in `LICENSES.md`.
4. Add the proxyllm config entries (`openai/<name>` + `api_base` + `api_key`),
   smoke-test end-to-end through the gateway the same way `try_modes.sh` does
   for existing vendors.
5. Run `bench/` against all three services under realistic concurrent load
   (adapt the pattern in proxyllm's `tests/locust/` harness) to get real
   aggregate throughput numbers and update this doc from estimated to measured.
