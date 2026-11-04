"""Bounded OpenAI-compatible API in front of native ComfyUI Krea 2 support."""

from __future__ import annotations

import asyncio
import base64
import copy
import hmac
import json
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, StringConstraints

INTERNAL_API_KEY = os.environ["INTERNAL_API_KEY"]
COMFY_BASE_URL = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188")
WORKFLOW_PATH = Path(__file__).parent / "comfy_workflows" / "txt2img_lustify.json"
OUTPUT_ROOT = Path(__file__).parent / "output"

MAX_QUEUE_SIZE = int(os.environ.get("LUSTIFY_MAX_QUEUE_SIZE", "32"))
JOB_TIMEOUT_SECONDS = int(os.environ.get("LUSTIFY_JOB_TIMEOUT_SECONDS", "900"))
MAX_MEGAPIXELS = float(os.environ.get("LUSTIFY_MAX_MEGAPIXELS", "4"))
MAX_IMAGES_PER_REQUEST = int(os.environ.get("LUSTIFY_MAX_IMAGES_PER_REQUEST", "4"))

PROMPT_NODE_ID = "4"
SIZE_NODE_ID = "7"
SAMPLER_NODE_ID = "8"
OUTPUT_NODE_ID = "10"
SIZE_PATTERN = re.compile(r"^(\d{3,4})x(\d{3,4})$")

REQUESTS = Counter("lustify_requests_total", "LUSTIFY API requests", ("result",))
JOB_SECONDS = Histogram(
    "lustify_job_duration_seconds",
    "End-to-end ComfyUI generation time",
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 900),
)
ADMITTED = Gauge("lustify_admitted_requests", "Running and queued API requests")
RUNNING = Gauge("lustify_running_requests", "Requests currently using the GPU")

_bearer = HTTPBearer(auto_error=False)
_workflow_template: dict | None = None
_admission_slots = asyncio.BoundedSemaphore(MAX_QUEUE_SIZE + 1)
_gpu_slot = asyncio.Semaphore(1)


def _load_workflow() -> dict:
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    required = {PROMPT_NODE_ID, SIZE_NODE_ID, SAMPLER_NODE_ID, OUTPUT_NODE_ID}
    missing = required.difference(workflow)
    if missing:
        raise RuntimeError(f"workflow is missing required nodes: {sorted(missing)}")
    return workflow


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _workflow_template
    _workflow_template = _load_workflow()
    yield


app = FastAPI(
    title="lustify-krea2-openai-api",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


def require_api_key(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    if creds is None or not hmac.compare_digest(creds.credentials, INTERNAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def parse_size(size: str) -> tuple[int, int]:
    match = SIZE_PATTERN.fullmatch(size.lower())
    if match is None:
        raise HTTPException(status_code=400, detail="size must use WIDTHxHEIGHT")
    width, height = (int(value) for value in match.groups())
    if not (512 <= width <= 2048 and 512 <= height <= 2048):
        raise HTTPException(status_code=400, detail="width and height must be 512-2048")
    if width % 64 or height % 64:
        raise HTTPException(
            status_code=400, detail="width and height must be multiples of 64"
        )
    if width * height > MAX_MEGAPIXELS * 1_000_000:
        raise HTTPException(
            status_code=400,
            detail=f"requested size exceeds {MAX_MEGAPIXELS:g} megapixels",
        )
    return width, height


Prompt = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)
]


class GenerationsRequest(BaseModel):
    prompt: Prompt
    model: str | None = None
    n: int = Field(default=1, ge=1)
    size: str = "1024x1024"
    response_format: Literal["b64_json"] = "b64_json"
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    num_inference_steps: int = Field(default=8, ge=8, le=80)

    model_config = {"extra": "forbid"}


async def _acquire_admission_slot() -> None:
    try:
        await asyncio.wait_for(_admission_slots.acquire(), timeout=0.01)
    except TimeoutError as exc:
        REQUESTS.labels("overloaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="generation queue is full; retry later",
            headers={"Retry-After": "10"},
        ) from exc
    ADMITTED.inc()


def _safe_output_path(image_ref: dict) -> Path | None:
    filename = image_ref.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        return None
    subfolder = image_ref.get("subfolder", "")
    if not isinstance(subfolder, str):
        return None
    candidate = (OUTPUT_ROOT / subfolder / filename).resolve()
    root = OUTPUT_ROOT.resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate


async def _delete_queued_job(client: httpx.AsyncClient, prompt_id: str) -> None:
    try:
        await client.post(f"{COMFY_BASE_URL}/queue", json={"delete": [prompt_id]})
    except httpx.HTTPError:
        pass


async def _delete_history(client: httpx.AsyncClient, prompt_id: str) -> None:
    try:
        await client.post(f"{COMFY_BASE_URL}/history", json={"delete": [prompt_id]})
    except httpx.HTTPError:
        pass


async def _run_comfy_job(
    client: httpx.AsyncClient,
    prompt_text: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
) -> bytes:
    if _workflow_template is None:
        raise HTTPException(status_code=503, detail="workflow is not loaded")

    graph = copy.deepcopy(_workflow_template)
    graph[PROMPT_NODE_ID]["inputs"]["text"] = prompt_text
    graph[SIZE_NODE_ID]["inputs"].update(width=width, height=height, batch_size=1)
    graph[SAMPLER_NODE_ID]["inputs"].update(seed=seed, steps=steps)

    client_id = str(uuid.uuid4())
    try:
        response = await client.post(
            f"{COMFY_BASE_URL}/prompt",
            json={"prompt": graph, "client_id": client_id},
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=502, detail="ComfyUI rejected the workflow"
        ) from exc

    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            history_response = await client.get(f"{COMFY_BASE_URL}/history/{prompt_id}")
            history_response.raise_for_status()
            entry = history_response.json().get(prompt_id)
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=502, detail="failed to read ComfyUI job status"
            ) from exc

        if entry:
            status_data = entry.get("status", {})
            if status_data.get("status_str") == "error":
                raise HTTPException(status_code=502, detail="ComfyUI generation failed")
            outputs = entry.get("outputs", {}).get(OUTPUT_NODE_ID, {}).get("images", [])
            if outputs:
                image_ref = outputs[0]
                try:
                    view = await client.get(
                        f"{COMFY_BASE_URL}/view",
                        params={
                            "filename": image_ref["filename"],
                            "subfolder": image_ref.get("subfolder", ""),
                            "type": image_ref.get("type", "output"),
                        },
                    )
                    view.raise_for_status()
                    content = view.content
                except (httpx.HTTPError, KeyError) as exc:
                    raise HTTPException(
                        status_code=502, detail="failed to retrieve generated image"
                    ) from exc
                output_path = _safe_output_path(image_ref)
                if output_path is not None:
                    output_path.unlink(missing_ok=True)
                await _delete_history(client, prompt_id)
                return content
        await asyncio.sleep(0.5)

    await _delete_queued_job(client, prompt_id)
    raise HTTPException(status_code=504, detail="ComfyUI generation timed out")


@app.post("/v1/images/generations", dependencies=[Depends(require_api_key)])
async def generations(request: GenerationsRequest) -> dict:
    if request.n > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"n must not exceed {MAX_IMAGES_PER_REQUEST}",
        )
    width, height = parse_size(request.size)
    await _acquire_admission_slot()
    started = time.monotonic()
    try:
        async with _gpu_slot:
            RUNNING.inc()
            try:
                timeout = JOB_TIMEOUT_SECONDS + 30
                async with httpx.AsyncClient(timeout=timeout) as client:
                    images = []
                    base_seed = (
                        request.seed
                        if request.seed is not None
                        else secrets.randbelow(2**63)
                    )
                    for offset in range(request.n):
                        raw = await _run_comfy_job(
                            client,
                            request.prompt,
                            base_seed + offset,
                            width,
                            height,
                            request.num_inference_steps,
                        )
                        images.append(
                            {"b64_json": base64.b64encode(raw).decode("ascii")}
                        )
            finally:
                RUNNING.dec()
        REQUESTS.labels("success").inc()
        return {"created": int(time.time()), "data": images}
    except HTTPException:
        REQUESTS.labels("error").inc()
        raise
    finally:
        JOB_SECONDS.observe(time.monotonic() - started)
        ADMITTED.dec()
        _admission_slots.release()


@app.get("/health")
async def health(response: Response) -> dict:
    if _workflow_template is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading"}
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            comfy_response = await client.get(f"{COMFY_BASE_URL}/system_stats")
            comfy_response.raise_for_status()
    except httpx.HTTPError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "comfyui_unreachable"}
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
