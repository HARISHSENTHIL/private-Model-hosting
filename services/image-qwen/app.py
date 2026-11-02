"""
OpenAI-shaped FastAPI wrapper around Qwen/Qwen-Image-2.1's QwenImage21Pipeline.

Why this shape: proxyllm's OpenAI image provider does zero vendor-specific
response translation (see litellm/llms/openai/image_generation/
dall_e_2_transformation.py and convert_dict_to_response.py in the proxyllm
repo) — it just parses whatever JSON comes back into its ImageResponse model.
Emitting that exact shape here means proxyllm can route to this service with
a plain `openai/<name>` + `api_base` config entry, no custom litellm provider
code. See ../../docs/ARCHITECTURE.md.

One pipeline instance handles all three of this model's capabilities:
text-to-image (/v1/images/generations with no input image), instruction-based
editing, and subject/reference personalization (/v1/images/edits with one or
more input images) — Qwen-Image-2.1 takes a list of 0-10 reference images as
one of QwenImage21Pipeline's own arguments, it isn't three separate models.
"""

import base64
import io
import os
import time
from typing import List, Optional

import torch
from diffusers import QwenImage21Pipeline
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel

INTERNAL_API_KEY = os.environ["INTERNAL_API_KEY"]
MODEL_ID = "Qwen/Qwen-Image-2.1"

app = FastAPI(title="qwen-image-2.1-openai-shim")
_bearer = HTTPBearer()

pipe: Optional[QwenImage21Pipeline] = None


@app.on_event("startup")
def load_pipeline() -> None:
    global pipe
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")


def require_api_key(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> None:
    if creds.credentials != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return 1024, 1024


def _image_to_response_item(image: Image.Image, response_format: str) -> dict:
    if response_format == "url":
        # No object storage wired up yet for this service — b64_json only
        # until one is. Don't silently fall back without saying so.
        raise HTTPException(
            status_code=400,
            detail="response_format=url is not implemented in this shim yet; use b64_json",
        )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"b64_json": base64.b64encode(buf.getvalue()).decode("utf-8")}


class GenerationsRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    # Diffusion-specific knobs OpenAI's schema doesn't have; optional passthrough.
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None


@app.post("/v1/images/generations", dependencies=[Depends(require_api_key)])
def generations(req: GenerationsRequest) -> dict:
    assert pipe is not None
    width, height = _parse_size(req.size)
    result = pipe(
        prompt=req.prompt,
        width=width,
        height=height,
        num_images_per_prompt=req.n,
        num_inference_steps=req.num_inference_steps or 50,
        guidance_scale=req.guidance_scale if req.guidance_scale is not None else 1.0,
    )
    return {
        "created": int(time.time()),
        "data": [
            _image_to_response_item(img, req.response_format) for img in result.images
        ],
    }


@app.post("/v1/images/edits", dependencies=[Depends(require_api_key)])
async def edits(
    prompt: str = Form(...),
    image: List[UploadFile] = File(...),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("b64_json"),
    num_inference_steps: Optional[int] = Form(None),
) -> dict:
    """
    Accepts 1-10 reference images (OpenAI's own schema allows one `image`
    field but repeated multipart fields of the same name are standard HTTP —
    Qwen-Image-2.1 supports up to 10 reference images for personalization and
    multi-reference composition; this is where that capability is exposed.
    """
    assert pipe is not None
    if not 1 <= len(image) <= 10:
        raise HTTPException(
            status_code=400, detail="Qwen-Image-2.1 accepts 1-10 reference images"
        )
    ref_images = [Image.open(io.BytesIO(await f.read())).convert("RGB") for f in image]
    width, height = _parse_size(size)
    result = pipe(
        prompt=prompt,
        image=ref_images,
        width=width,
        height=height,
        num_images_per_prompt=n,
        num_inference_steps=num_inference_steps or 50,
    )
    return {
        "created": int(time.time()),
        "data": [
            _image_to_response_item(img, response_format) for img in result.images
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if pipe is not None else "loading"}
