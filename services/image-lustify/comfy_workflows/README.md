# Workflow graphs

This directory is intentionally empty of real workflow JSON right now.

`shim.py` expects a ComfyUI API-format workflow graph at
`txt2img_lustify.json` (exported from ComfyUI's UI via "Save (API Format)",
not the regular "Save" format — they're different JSON shapes). It needs:

- the LUSTIFY V10 "Krea 2" checkpoint loader node, pointed at the Raw
  full-precision safetensors (not Turbo, not a quantized variant — see
  `docs/LICENSES.md` and `docs/ARCHITECTURE.md` for why),
- the `qwen3vl_4b` text encoder node,
- the Wan 2.1 VAE loader node,
- a text node whose value `shim.py` overwrites with the request's prompt
  (`PROMPT_NODE_ID` in `shim.py` must match that node's id in this file),
- and a `SaveImage`-equivalent output node whose id matches
  `OUTPUT_NODE_ID` in `shim.py`.

Build this for real once the custom-node pack for Krea 2 + Wan 2.1 VAE is
identified (see the `Dockerfile`'s open TODO) and a manual generation works
in ComfyUI's own UI first — don't write `shim.py` call sites against a graph
that's never actually been run.
