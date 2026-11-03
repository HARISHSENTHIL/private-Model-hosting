# Headless LUSTIFY workflow

`txt2img_lustify.json` is a minimal ComfyUI API-format graph. It intentionally
uses only native ComfyUI nodes so production does not depend on community node
packs.

The graph is based on the first-pass settings from the model author's workflow
(Civitai model version `3123550`, SHA-256
`849f97ed9fc46a3c8e93dd4981aca44ac1359c2ae35a532a1ce59b77df824ac1`),
with these production-oriented changes:

- LUSTIFY v10 RAW BF16 is used instead of a quantized Turbo checkpoint.
- Qwen3-VL 4B BF16 is used instead of the FP8 text encoder.
- The author's recommended Wan 2.1 VAE is retained.
- Optional prompt enhancement, face/hand detailers, upscaling, and community
  nodes are omitted. They make latency and memory use unpredictable and are not
  part of the advertised `/v1/images/generations` contract.
- RAW uses 52 steps; the API may lower or raise this within its validated range.

Node IDs `4`, `7`, `8`, and `10` are part of the shim contract. Update the
constants and tests in `shim.py` if the graph changes.
