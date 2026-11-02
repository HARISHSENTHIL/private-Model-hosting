"""
OpenAI-shaped FastAPI translator in front of a headless ComfyUI instance
serving LUSTIFY V10 "Krea 2".

Why a translator instead of exposing ComfyUI directly: ComfyUI's native API
(POST /prompt with a node-graph JSON, then poll /history) doesn't match
OpenAI's image response shape, and proxyllm's OpenAI image provider does zero
vendor-specific translation (see litellm/llms/openai/image_generation/
dall_e_2_transformation.py in the proxyllm repo) — it just parses whatever
JSON comes back. Emitting that exact shape here keeps the proxyllm
integration at `openai/<name>` + `api_base`, zero custom litellm code, same
as the other two services. See ../../docs/ARCHITECTURE.md.

Requires comfy_workflows/txt2img_lustify.json to exist and match the node
ids below — see comfy_workflows/README.md. This file intentionally raises
loudly at startup if that workflow hasn't been built yet, rather than
pretending to work against a graph nobody has run.
"""

import asyncio
import base64
import copy
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

INTERNAL_API_KEY = os.environ["INTERNAL_API_KEY"]
COMFY_BASE_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = Path(__file__).parent / "comfy_workflows" / "txt2img_lustify.json"

# Must match the node ids in that exported workflow — see comfy_workflows/README.md.
PROMPT_NODE_ID = "PLACEHOLDER_PROMPT_NODE_ID"
SEED_NODE_ID = "PLACEHOLDER_SEED_NODE_ID"
OUTPUT_NODE_ID = "PLACEHOLDER_OUTPUT_NODE_ID"

app = FastAPI(title="lustify-comfyui-openai-shim")
_bearer = HTTPBearer()
_workflow_template: Optional[dict] = None


@app.on_event("startup")
def load_workflow_template() -> None:
    global _workflow_template
    if not WORKFLOW_PATH.exists():
        raise RuntimeError(
            f"{WORKFLOW_PATH} doesn't exist yet. Export it from a working "
            "ComfyUI session first — see comfy_workflows/README.md. "
            "This shim refuses to start against a workflow that's never "
            "actually been run."
        )
    _workflow_template = json.loads(WORKFLOW_PATH.read_text())


def require_api_key(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> None:
    if creds.credentials != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return 1024, 1024


class GenerationsRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    seed: Optional[int] = None


async def _run_comfy_job(client: httpx.AsyncClient, prompt_text: str, seed: int) -> bytes:
    assert _workflow_template is not None
    graph = copy.deepcopy(_workflow_template)
    graph[PROMPT_NODE_ID]["inputs"]["text"] = prompt_text
    graph[SEED_NODE_ID]["inputs"]["seed"] = seed

    client_id = str(uuid.uuid4())
    resp = await client.post(
        f"{COMFY_BASE_URL}/prompt", json={"prompt": graph, "client_id": client_id}
    )
    resp.raise_for_status()
    prompt_id = resp.json()["prompt_id"]

    # Poll /history rather than the websocket API — simpler, fine for the
    # batch-size-1 queue depth this service actually sees (see
    # docs/ARCHITECTURE.md's concurrency section).
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        hist = await client.get(f"{COMFY_BASE_URL}/history/{prompt_id}")
        hist.raise_for_status()
        entry = hist.json().get(prompt_id)
        if entry and entry.get("outputs"):
            outputs = entry["outputs"].get(OUTPUT_NODE_ID, {}).get("images", [])
            if not outputs:
                raise HTTPException(status_code=502, detail="ComfyUI job finished with no image output")
            img_ref = outputs[0]
            view = await client.get(
                f"{COMFY_BASE_URL}/view",
                params={
                    "filename": img_ref["filename"],
                    "subfolder": img_ref.get("subfolder", ""),
                    "type": img_ref.get("type", "output"),
                },
            )
            view.raise_for_status()
            return view.content
        await asyncio.sleep(0.5)

    raise HTTPException(status_code=504, detail="ComfyUI job did not finish in time")


@app.post("/v1/images/generations", dependencies=[Depends(require_api_key)])
async def generations(req: GenerationsRequest) -> dict:
    if req.response_format != "b64_json":
        raise HTTPException(
            status_code=400,
            detail="response_format=url is not implemented in this shim yet; use b64_json",
        )
    width, height = _parse_size(req.size)  # noqa: F841 — wire into the workflow's size nodes once they're known
    base_seed = req.seed if req.seed is not None else int(time.time())

    async with httpx.AsyncClient(timeout=180) as client:
        images = []
        for i in range(req.n):
            raw = await _run_comfy_job(client, req.prompt, base_seed + i)
            images.append({"b64_json": base64.b64encode(raw).decode("utf-8")})

    return {"created": int(time.time()), "data": images}


@app.get("/health")
async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{COMFY_BASE_URL}/system_stats")
            r.raise_for_status()
        return {"status": "ok"}
    except httpx.HTTPError:
        return {"status": "comfyui unreachable"}
