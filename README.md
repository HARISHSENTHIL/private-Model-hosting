# private-model-hosting

Self-hosted inference stack for proxyllm's "Private" privacy tier — self-owned
GPUs, zero-data-retention, no routing through third-party vendors.

Hardware target: a single node with 4x NVIDIA H100 80GB SXM (NVLink).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design
(GPU allocation, serving framework choice per model, capacity math) and
[`docs/LICENSES.md`](docs/LICENSES.md) for what's cleared for commercial use
vs. what's an accepted interim risk — **read that before enabling any model for
paying users.**

## Services

| Service | GPU(s) | Model | Port |
|---|---|---|---|
| `llm-huihui-qwen` | 0, 1 (TP=2) | `huihui-ai/Huihui-Qwen3.8-27B-abliterated` via SGLang | 8001 |
| `image-qwen` | 3 | `Qwen/Qwen-Image-2.1` via Diffusers + FastAPI | 8002 |
| `image-lustify` | 2 | LUSTIFY V10 "Krea 2" via ComfyUI + FastAPI shim | 8003 |

Every service exposes an OpenAI-compatible API on its port
(`/v1/chat/completions` for the LLM, `/v1/images/generations` +
`/v1/images/edits` for the image services) so proxyllm can route to it with a
plain `openai/<name>` + `api_base` config entry — no custom litellm provider
code needed. See `ARCHITECTURE.md` for why.

## Running

```bash
cp .env.example .env   # fill in HF_TOKEN, CIVITAI_API_TOKEN, INTERNAL_API_KEY
docker compose up -d llm-huihui-qwen
docker compose up -d image-qwen
docker compose up -d image-lustify
```

Bring services up one at a time the first time — each one downloads tens of GB
of weights on first start (see `docs/ARCHITECTURE.md` for exact sizes).

## Benchmarking (required before any throughput promise)

None of the latency/concurrency numbers in `docs/ARCHITECTURE.md` are measured
yet — they're estimates from the model architectures and published research.
Run these against the real hardware and update that doc:

```bash
python bench/bench_llm.py --url http://localhost:8001 --max-concurrency 200
python bench/bench_image.py --url http://localhost:8002 --endpoint generations
python bench/bench_image.py --url http://localhost:8003 --endpoint generations
```

## Wiring into proxyllm

Once a service is up and benchmarked, add it to proxyllm's config as a plain
OpenAI-compatible entry (see `config/chat/primary.yaml` / `config/image/primary.yaml`
in the proxyllm repo for the existing pattern used for Surplus and the
self-hosted `flux-2-klein-4b` entry):

```yaml
- model_name: huihui-qwen-abliterated
  litellm_params:
    model: openai/huihui-qwen3-27b-abliterated
    api_base: http://<host>:8001/v1
    api_key: os.environ/PRIVATE_LLM_API_KEY
```

That's a follow-up change in the proxyllm repo itself, not in this one.
