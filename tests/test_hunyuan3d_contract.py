from __future__ import annotations

import importlib.util
import sys
import urllib.request
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).parents[1]
SERVICE = ROOT / "services" / "hunyuan3d-2"


def load_shim(monkeypatch: pytest.MonkeyPatch | None = None):
    import os

    os.environ.setdefault("INTERNAL_API_KEY", "x" * 32)
    spec = importlib.util.spec_from_file_location("hunyuan3d_shim", SERVICE / "shim.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_downloader():
    # Registered under the bare name "model_downloader" (not a test-prefixed
    # one) because shim.py does `from model_downloader import MODELS_ROOT` -
    # in the real container both files live in the same WORKDIR and
    # model_downloader is importable as a plain sibling module; here it must
    # be loaded and cached in sys.modules under that exact name BEFORE shim.py
    # is loaded, so shim's import statement resolves against this instance.
    spec = importlib.util.spec_from_file_location(
        "model_downloader", SERVICE / "model_downloader.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_downloader"] = module
    spec.loader.exec_module(module)
    return module


DOWNLOADER = load_downloader()
SHIM = load_shim()


def test_artifact_sizes_and_hashes_are_pinned() -> None:
    artifacts = {artifact.relative_path: artifact for artifact in DOWNLOADER.ARTIFACTS}
    dit = artifacts["hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors"]

    assert dit.size_bytes == 4_930_777_530
    assert dit.sha256 == (
        "5ee5a81e4df08a1c65b79910bf5b145a90376e526794f4607a4d5d068d62f269"
    )
    assert len(DOWNLOADER.ARTIFACTS) == 9
    assert all(len(artifact.sha256) == 64 for artifact in DOWNLOADER.ARTIFACTS)
    assert all(artifact.size_bytes > 0 for artifact in DOWNLOADER.ARTIFACTS)
    assert all(artifact.token_env == "HF_TOKEN" for artifact in DOWNLOADER.ARTIFACTS)


def test_cross_origin_redirect_drops_authorization_but_keeps_range() -> None:
    request = urllib.request.Request(
        "https://huggingface.co/tencent/Hunyuan3D-2/resolve/x/model.fp16.safetensors",
        headers={"Authorization": "Bearer secret", "Range": "bytes=1024-"},
    )

    redirected = DOWNLOADER.TokenSafeRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://cdn-lfs.huggingface.co/signed-url",
    )

    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("Range") == "bytes=1024-"


def test_same_origin_redirect_retains_authorization() -> None:
    request = urllib.request.Request(
        "https://huggingface.co/old",
        headers={"Authorization": "Bearer secret"},
    )

    redirected = DOWNLOADER.TokenSafeRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://huggingface.co/new",
    )

    assert redirected.get_header("Authorization") == "Bearer secret"


def test_example_env_documents_hunyuan3d_vars() -> None:
    content = (ROOT / ".env.example").read_text()
    assert "HUNYUAN3D_MAX_QUEUE_SIZE" in content
    assert "HUNYUAN3D_IDLE_UNLOAD_SECONDS" in content
    assert "IMAGE_QWEN_BASE_URL" in content


def test_requires_exactly_one_of_prompt_or_image_b64() -> None:
    with pytest.raises(ValidationError):
        SHIM.GenerationsRequest()

    with pytest.raises(ValidationError):
        SHIM.GenerationsRequest(prompt="a chair", image_b64="aGVsbG8=")

    # Exactly one is valid either way.
    SHIM.GenerationsRequest(prompt="a chair")
    SHIM.GenerationsRequest(image_b64="aGVsbG8=")


@pytest.mark.parametrize("steps", [0, 101])
def test_rejects_out_of_range_inference_steps(steps: int) -> None:
    with pytest.raises(ValidationError):
        SHIM.GenerationsRequest(prompt="a chair", num_inference_steps=steps)


@pytest.mark.parametrize("resolution", [64, 1024])
def test_rejects_out_of_range_octree_resolution(resolution: int) -> None:
    with pytest.raises(ValidationError):
        SHIM.GenerationsRequest(prompt="a chair", octree_resolution=resolution)


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SHIM.GenerationsRequest(prompt="a chair", unknown_field=True)


@pytest.mark.parametrize(
    ("last_finished_at", "now", "threshold", "queue_empty", "expected"),
    [
        (None, 1000.0, 600.0, True, False),  # never ran a job yet
        (100.0, 500.0, 600.0, True, False),  # 400s idle, under threshold
        (100.0, 701.0, 600.0, True, True),  # 601s idle, over threshold
        (100.0, 701.0, 600.0, False, False),  # over threshold but queue not empty
    ],
)
def test_should_unload_decision(
    last_finished_at: float | None,
    now: float,
    threshold: float,
    queue_empty: bool,
    expected: bool,
) -> None:
    assert SHIM._should_unload(last_finished_at, now, threshold, queue_empty) is expected


def test_set_future_result_and_exception_are_idempotent_against_done_futures() -> None:
    import asyncio

    async def _check() -> None:
        loop = asyncio.get_running_loop()

        future: asyncio.Future = loop.create_future()
        SHIM._set_future_result(future, b"abc")
        SHIM._set_future_result(future, b"def")  # must not raise InvalidStateError
        assert future.result() == b"abc"

        other: asyncio.Future = loop.create_future()
        other.cancel()
        SHIM._set_future_exception(other, RuntimeError("boom"))  # must not raise

    asyncio.run(_check())
