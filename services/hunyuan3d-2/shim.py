"""Bounded OpenAI-shaped API in front of the Hunyuan3D-2 turbo pipelines.

Unlike services/image-lustify/shim.py, there is no separate backend HTTP
process to proxy to: the shape (DiT) and texture (Paint, which internally
runs its own delight step) pipelines run in-process, owned exclusively by
one persistent worker thread (see _worker_loop). FastAPI
handlers enqueue a job and await an asyncio.Future that the worker thread
resolves via loop.call_soon_threadsafe - this is a new pattern in this repo,
not a copy of LUSTIFY's httpx-to-ComfyUI design.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import io
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Literal

import httpx
import trimesh
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, StringConstraints, model_validator

from model_downloader import MODELS_ROOT

INTERNAL_API_KEY = os.environ["INTERNAL_API_KEY"]
IMAGE_QWEN_BASE_URL = os.environ.get("IMAGE_QWEN_BASE_URL", "http://image-qwen:8000")

MAX_QUEUE_SIZE = int(os.environ.get("HUNYUAN3D_MAX_QUEUE_SIZE", "16"))
JOB_TIMEOUT_SECONDS = int(os.environ.get("HUNYUAN3D_JOB_TIMEOUT_SECONDS", "1200"))
IDLE_UNLOAD_SECONDS = float(os.environ.get("HUNYUAN3D_IDLE_UNLOAD_SECONDS", "600"))
MAX_IMAGE_BYTES = int(os.environ.get("HUNYUAN3D_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
_IDLE_POLL_SECONDS = 30

REQUESTS = Counter("hunyuan3d_requests_total", "Hunyuan3D-2 API requests", ("result",))
JOB_SECONDS = Histogram(
    "hunyuan3d_job_duration_seconds",
    "End-to-end mesh generation time",
    buckets=(5, 10, 20, 30, 60, 120, 300, 600, 900, 1200),
)
ADMITTED = Gauge("hunyuan3d_admitted_requests", "Running and queued API requests")
RUNNING = Gauge("hunyuan3d_running_requests", "Requests currently using the GPU")
PIPELINES_LOADED = Gauge("hunyuan3d_pipelines_loaded", "1 if pipelines are resident in VRAM")

_bearer = HTTPBearer(auto_error=False)
_admission_slots = asyncio.BoundedSemaphore(MAX_QUEUE_SIZE + 1)

# No separate GPU semaphore (unlike image-lustify's _gpu_slot): the single
# worker thread below already serializes all pipeline work by construction,
# since it pulls one item at a time off _job_queue.
_job_queue: "queue.SimpleQueue" = queue.SimpleQueue()
_pipelines: "_Pipelines | None" = None
_last_job_finished_at: float | None = None


def require_api_key(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    if creds is None or not hmac.compare_digest(creds.credentials, INTERNAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
            headers={"WWW-Authenticate": "Bearer"},
        )


Prompt = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)
]


class GenerationsRequest(BaseModel):
    prompt: Prompt | None = None
    image_b64: str | None = None
    response_format: Literal["glb_b64"] = "glb_b64"
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    texture: bool = True
    # Defaults match Hunyuan3DDiTFlowMatchingPipeline.__call__'s own
    # confirmed signature defaults (octree_resolution=384,
    # num_inference_steps=50) at the pinned commit - not independently
    # re-tuned for the turbo checkpoint specifically. Turbo (step-distilled)
    # checkpoints are typically meant to run well below their base model's
    # default step count, but no turbo-specific recommendation was found in
    # the README or examples - measure real latency/quality at a few step
    # counts via bench/bench_mesh.py before lowering this default.
    octree_resolution: int = Field(default=384, ge=128, le=512)
    num_inference_steps: int = Field(default=50, ge=1, le=100)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_exactly_one_input(self) -> "GenerationsRequest":
        if (self.prompt is None) == (self.image_b64 is None):
            raise ValueError("exactly one of prompt or image_b64 is required")
        return self


@dataclass(frozen=True)
class GenerationJob:
    prompt: str | None
    image_bytes: bytes | None
    seed: int | None
    texture: bool
    octree_resolution: int
    num_inference_steps: int


@dataclass
class _Pipelines:
    shape: object
    paint: object
    remove_background: object


class _UnloadJob:
    """Sentinel pushed onto _job_queue to unload pipelines between real jobs."""


class _Shutdown:
    """Sentinel pushed onto _job_queue to stop the worker thread."""


_UNLOAD_SENTINEL = _UnloadJob()
_SHUTDOWN_SENTINEL = _Shutdown()


def _should_unload(
    last_finished_at: float | None,
    now: float,
    threshold_seconds: float,
    queue_empty: bool,
) -> bool:
    """Pure decision helper, extracted so it is unit-testable without a GPU,
    a real worker thread, or real time passing (see tests/test_hunyuan3d_contract.py).
    """
    if last_finished_at is None or not queue_empty:
        return False
    return (now - last_finished_at) >= threshold_seconds


def _load_pipelines() -> _Pipelines:
    # Deferred import: hy3dgen pulls in torch + CUDA context init. Importing
    # it at module level would force every pure-logic unit test that loads
    # this module via importlib (admission control, request validation,
    # idle-unload timing) to also require a GPU and the full installed
    # package. image-lustify avoided this entirely by keeping its engine
    # out-of-process; this service's in-process design reintroduces the
    # problem, so the import is pushed to first real use instead.
    # Verified against hy3dgen/shapegen/pipelines.py, hy3dgen/texgen/pipelines.py,
    # and hy3dgen/rembg.py at the pinned GitHub commit (f8db630...), and
    # against the README's own usage example - NOT a guess, read from the
    # real source. Three corrections from the first draft of this function:
    #
    # 1. There is no standalone "Delight" pipeline class to import or call.
    #    Hunyuan3DPaintPipeline.from_pretrained() loads the delight model
    #    internally (from the same model_path's hunyuan3d-delight-v2-0
    #    subfolder) and applies it itself inside its own __call__, on the
    #    *untouched* reference image - see _run_generation_job below.
    # 2. Hunyuan3DPaintPipeline.from_pretrained(model_path, subfolder) takes
    #    exactly those two args - no **kwargs, no `torch_dtype`. Passing one
    #    would have raised TypeError.
    # 3. Hunyuan3DPaintPipeline has no `.to()` method at all (it hardcodes
    #    device='cuda' internally via Hunyuan3DTexGenConfig) - calling `.to()`
    #    on it would have raised AttributeError. Hunyuan3DDiTFlowMatchingPipeline
    #    does define its own `.to(device, dtype)`, but from_pretrained already
    #    calls it internally with its own cuda/fp16 defaults, so no explicit
    #    call is needed for either pipeline.
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(MODELS_ROOT), subfolder="hunyuan3d-dit-v2-0-turbo"
    )
    paint = Hunyuan3DPaintPipeline.from_pretrained(
        str(MODELS_ROOT), subfolder="hunyuan3d-paint-v2-0-turbo"
    )
    # rembg's new_session() downloads its own ONNX background-removal model
    # (u2net, ~176 MB) from a GitHub release on first use and caches it
    # under ~/.u2net - NOT pinned/SHA-256-verified the way every other
    # weight in this service is (open gap, not yet fixed - see docs/ARCHITECTURE.md).
    remove_background = BackgroundRemover()
    return _Pipelines(shape=shape, paint=paint, remove_background=remove_background)


def _ensure_pipelines_loaded() -> _Pipelines:
    global _pipelines
    if _pipelines is None:
        _pipelines = _load_pipelines()
        PIPELINES_LOADED.set(1)
    return _pipelines


def _unload_pipelines() -> None:
    global _pipelines
    if _pipelines is None:
        return
    import torch

    _pipelines = None
    torch.cuda.empty_cache()
    PIPELINES_LOADED.set(0)
    print("hunyuan3d-2: pipelines unloaded after idle timeout", flush=True)


def _decode_image_bytes(raw: bytes) -> "Image.Image":
    # Deliberately NOT forcing RGBA here: _run_generation_job below decides
    # whether to run background removal based on whether PIL decoded this as
    # "RGB" (no alpha - a plain photo) vs. already having an alpha channel
    # (pre-segmented), matching gradio_app.py's own
    # `if check_box_rembg or image.mode == "RGB": image = rmbg_worker(image)`
    # check. Converting to RGBA here would make that check never fire.
    try:
        return Image.open(io.BytesIO(raw))
    except Exception as exc:  # PIL raises several distinct exception types
        raise HTTPException(
            status_code=400, detail="image_b64 did not decode to a valid image"
        ) from exc


def _text_to_image(prompt: str) -> "Image.Image":
    # Synchronous call from inside the worker thread, not the event loop.
    # This blocks the one worker thread for the duration of the upstream
    # call, which is acceptable: GPU work here is already serialized to one
    # job at a time, so there is no second job this could be blocking.
    try:
        response = httpx.post(
            f"{IMAGE_QWEN_BASE_URL}/v1/images/generations",
            json={"prompt": prompt, "n": 1, "size": "1024x1024"},
            headers={"Authorization": f"Bearer {INTERNAL_API_KEY}"},
            timeout=180,
        )
        response.raise_for_status()
        b64 = response.json()["data"][0]["b64_json"]
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail="upstream image model (image-qwen) is unavailable"
        ) from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise HTTPException(
            status_code=502, detail="upstream image model returned an unexpected response"
        ) from exc
    return _decode_image_bytes(base64.b64decode(b64))


def _export_glb(mesh: trimesh.Trimesh) -> bytes:
    buffer = io.BytesIO()
    mesh.export(file_obj=buffer, file_type="glb")
    return buffer.getvalue()


def _run_generation_job(job: GenerationJob) -> bytes:
    import torch

    pipelines = _ensure_pipelines_loaded()
    image = (
        _decode_image_bytes(job.image_bytes)
        if job.image_bytes is not None
        else _text_to_image(job.prompt)  # type: ignore[arg-type]
    )

    # Matches gradio_app.py's own default behavior: run background removal
    # only when the image has no alpha channel yet (a plain photo, or
    # image-qwen's text-to-3D output, both decode as PIL mode "RGB"). An
    # already-segmented PNG (mode "RGBA"/"LA"/"P" with transparency) is left
    # as-is.
    if image.mode == "RGB":
        image = pipelines.remove_background(image)

    generator = (
        torch.Generator(device="cuda").manual_seed(job.seed)
        if job.seed is not None
        else None
    )
    # [0]: matches the Hunyuan3D-2 README's own
    # `mesh = pipeline(image=image)[0]` usage, confirmed at the pinned commit.
    mesh = pipelines.shape(
        image=image,
        octree_resolution=job.octree_resolution,
        num_inference_steps=job.num_inference_steps,
        generator=generator,
    )[0]
    if job.texture:
        # Paint delights `image` itself internally as part of its own
        # multi-view texture-painting call (hy3dgen/texgen/pipelines.py
        # loads and runs the delight model inside Hunyuan3DPaintPipeline -
        # there is no separate delight step to do here).
        mesh = pipelines.paint(mesh, image=image)
    return _export_glb(mesh)


def _set_future_result(future: asyncio.Future, result: bytes) -> None:
    if not future.done():
        future.set_result(result)


def _set_future_exception(future: asyncio.Future, exc: Exception) -> None:
    if not future.done():
        future.set_exception(exc)


def _worker_loop() -> None:
    global _last_job_finished_at
    while True:
        item = _job_queue.get()
        if isinstance(item, _Shutdown):
            return
        if isinstance(item, _UnloadJob):
            _unload_pipelines()
            continue

        job, future, loop = item
        try:
            result = _run_generation_job(job)
        except Exception as exc:  # must not kill the worker thread
            loop.call_soon_threadsafe(_set_future_exception, future, exc)
        else:
            loop.call_soon_threadsafe(_set_future_result, future, result)
        finally:
            _last_job_finished_at = time.monotonic()


async def _idle_unload_loop() -> None:
    while True:
        await asyncio.sleep(_IDLE_POLL_SECONDS)
        if _pipelines is None:
            continue
        if _should_unload(
            _last_job_finished_at, time.monotonic(), IDLE_UNLOAD_SECONDS, _job_queue.empty()
        ):
            _job_queue.put(_UNLOAD_SENTINEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = threading.Thread(target=_worker_loop, name="hunyuan3d-worker", daemon=True)
    worker.start()
    app.state.worker_thread = worker
    idle_task = asyncio.create_task(_idle_unload_loop())
    yield
    idle_task.cancel()
    _job_queue.put(_SHUTDOWN_SENTINEL)


app = FastAPI(
    title="hunyuan3d-2-openai-api",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


async def _acquire_admission_slot() -> None:
    try:
        await asyncio.wait_for(_admission_slots.acquire(), timeout=0.01)
    except TimeoutError as exc:
        REQUESTS.labels("overloaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="generation queue is full; retry later",
            headers={"Retry-After": "30"},
        ) from exc
    ADMITTED.inc()


@app.post("/v1/3d/generations", dependencies=[Depends(require_api_key)])
async def generations(request: GenerationsRequest) -> dict:
    image_bytes: bytes | None = None
    if request.image_b64 is not None:
        try:
            image_bytes = base64.b64decode(request.image_b64, validate=True)
        except binascii.Error as exc:
            raise HTTPException(
                status_code=400, detail="image_b64 is not valid base64"
            ) from exc
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=400, detail=f"image exceeds {MAX_IMAGE_BYTES} bytes"
            )

    await _acquire_admission_slot()
    started = time.monotonic()
    RUNNING.inc()
    try:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        job = GenerationJob(
            prompt=request.prompt,
            image_bytes=image_bytes,
            seed=request.seed,
            texture=request.texture,
            octree_resolution=request.octree_resolution,
            num_inference_steps=request.num_inference_steps,
        )
        _job_queue.put((job, future, loop))
        try:
            glb_bytes = await asyncio.wait_for(future, timeout=JOB_TIMEOUT_SECONDS + 30)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504, detail="mesh generation timed out"
            ) from exc
        REQUESTS.labels("success").inc()
        return {
            "created": int(time.time()),
            "data": [{"glb_b64": base64.b64encode(glb_bytes).decode("ascii"), "format": "glb"}],
        }
    except HTTPException:
        REQUESTS.labels("error").inc()
        raise
    finally:
        RUNNING.dec()
        JOB_SECONDS.observe(time.monotonic() - started)
        ADMITTED.dec()
        _admission_slots.release()


@app.get("/health")
async def health(response: Response) -> dict:
    worker = getattr(app.state, "worker_thread", None)
    if worker is None or not worker.is_alive():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "worker_dead"}
    # Unloaded-but-idle is the normal resting state (see idle-unload above),
    # not a degraded one - only worker-thread liveness gates health.
    return {"status": "ok", "pipelines_loaded": _pipelines is not None}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
