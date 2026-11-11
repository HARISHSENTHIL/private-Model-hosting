"""Download and verify the immutable model artifacts required by Hunyuan3D-2.

Same resumable/checksum pattern as services/image-lustify/model_downloader.py,
reused near-verbatim. All artifacts here come from a single HF repo
(tencent/Hunyuan3D-2) at one pinned revision, so there is only one token_env
("HF_TOKEN") instead of LUSTIFY's mix of Civitai + Hugging Face sources.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Mirrors the HF repo's own directory layout (owner/name/<subfolder>/<file>)
# so that diffusers-style `XPipeline.from_pretrained(MODELS_ROOT, subfolder=...)`
# calls resolve the same way they would against a plain `huggingface_hub`
# cache. NOT yet confirmed against Hunyuan3D-2's actual from_pretrained calls
# in shim.py's pipeline-loading code - verify this once the cloned repo code
# is available and adjust both this path and shim.py's _load_pipelines()
# together if the real expectation differs.
MODELS_ROOT = Path("/root/hunyuan3d/models/tencent/Hunyuan3D-2")
CHUNK_SIZE = 16 * 1024 * 1024

# Pinned revision of the tencent/Hunyuan3D-2 HUGGING FACE WEIGHTS repo (not to
# be confused with the Tencent-Hunyuan/Hunyuan3D-2 GITHUB CODE repo pinned in
# the Dockerfile - these are two different repos with two different commit
# SHAs). Confirmed via `curl https://huggingface.co/api/models/tencent/Hunyuan3D-2?blobs=true`
# on 2026-11-11.
REVISION = "9cd649ba6913f7a852e3286bad86bfa9a2d83dcf"
_BASE_URL = f"https://huggingface.co/tencent/Hunyuan3D-2/resolve/{REVISION}"


class TokenSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow HTTPS redirects without forwarding credentials cross-origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None

        source = urllib.parse.urlsplit(req.full_url)
        target = urllib.parse.urlsplit(redirected.full_url)
        if target.scheme != "https":
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                "refusing non-HTTPS download redirect",
                headers,
                fp,
            )

        source_origin = (source.scheme, source.hostname, source.port)
        target_origin = (target.scheme, target.hostname, target.port)
        if source_origin != target_origin:
            # HF's resolve/ URLs can redirect to a signed CDN URL for large
            # LFS files. Kept for parity with image-lustify's downloader even
            # though HF's CDN has not been observed rejecting a forwarded
            # bearer token the way Civitai's does - safer to strip either way.
            redirected.remove_header("Authorization")
        return redirected


URL_OPENER = urllib.request.build_opener(TokenSafeRedirectHandler())


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    relative_path: str
    sha256: str
    size_bytes: int
    token_env: str


def _hf(relative_path: str) -> str:
    return f"{_BASE_URL}/{relative_path}"


# sha256 + size_bytes fetched directly from the HF API's `siblings[].lfs`
# metadata on 2026-11-11 (raw JSON, not an LLM-summarized transcription -
# long hex digests are exactly the kind of string a summarization pass can
# silently corrupt, so this was deliberately re-verified via a direct fetch
# and JSON parse before being hardcoded here).
ARTIFACTS = (
    Artifact(
        name="Hunyuan3D-2 DiT turbo (shape)",
        url=_hf("hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors"),
        relative_path="hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors",
        sha256="5ee5a81e4df08a1c65b79910bf5b145a90376e526794f4607a4d5d068d62f269"[:64],
        size_bytes=4_930_777_530,
        token_env="HF_TOKEN",
    ),
    Artifact(
        name="Hunyuan3D-2 Paint turbo image encoder",
        url=_hf("hunyuan3d-paint-v2-0-turbo/image_encoder/model.safetensors"),
        relative_path="hunyuan3d-paint-v2-0-turbo/image_encoder/model.safetensors",
        sha256="ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a"[:64],
        size_bytes=1_264_217_240,
        token_env="HF_TOKEN",
    ),
    Artifact(
        name="Hunyuan3D-2 Paint turbo text encoder",
        url=_hf("hunyuan3d-paint-v2-0-turbo/text_encoder/pytorch_model.bin"),
        relative_path="hunyuan3d-paint-v2-0-turbo/text_encoder/pytorch_model.bin",
        sha256="c3e254d7b61353497ea0be2c4013df4ea8f739ee88cffa0ba58cd085459ed565"[:64],
        size_bytes=1_361_671_895,
        token_env="HF_TOKEN",
    ),
    Artifact(
        name="Hunyuan3D-2 Paint turbo UNet",
        url=_hf("hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.safetensors"),
        relative_path="hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.safetensors",
        sha256="d6acffa4a22f4da61d87f446bfa83e7ac245481c1535fbf25b200fe4462d0b22"[:64],
        size_bytes=3_722_161_032,
        token_env="HF_TOKEN",
    ),
    Artifact(
        name="Hunyuan3D-2 Paint turbo VAE",
        url=_hf("hunyuan3d-paint-v2-0-turbo/vae/diffusion_pytorch_model.bin"),
        relative_path="hunyuan3d-paint-v2-0-turbo/vae/diffusion_pytorch_model.bin",
        sha256="1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc"[:64],
        size_bytes=334_707_217,
        token_env="HF_TOKEN",
    ),
    Artifact(
        name="Hunyuan3D-2 Delight text encoder",
        url=_hf("hunyuan3d-delight-v2-0/text_encoder/model.safetensors"),
        relative_path="hunyuan3d-delight-v2-0/text_encoder/model.safetensors",
        sha256="bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f"[:64],
        size_bytes=680_820_392,
        token_env="HF_TOKEN",
    ),
    Artifact(
        name="Hunyuan3D-2 Delight UNet",
        url=_hf("hunyuan3d-delight-v2-0/unet/diffusion_pytorch_model.safetensors"),
        relative_path="hunyuan3d-delight-v2-0/unet/diffusion_pytorch_model.safetensors",
        sha256="0ce61d15a43d11ba19079ab8f24dfce78b876d3f5291470079ef64b17e08ca58"[:64],
        size_bytes=3_463_772_592,
        token_env="HF_TOKEN",
    ),
    Artifact(
        name="Hunyuan3D-2 Delight VAE",
        url=_hf("hunyuan3d-delight-v2-0/vae/diffusion_pytorch_model.safetensors"),
        relative_path="hunyuan3d-delight-v2-0/vae/diffusion_pytorch_model.safetensors",
        sha256="3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a"[:64],
        size_bytes=167_335_342,
        token_env="HF_TOKEN",
    ),
    Artifact(
        name="Hunyuan3D-2 VAE turbo (shape decoder)",
        url=_hf("hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors"),
        relative_path="hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors",
        sha256="66b7af697118db2de00c22f63a385c75f9049709739281971837af9e4ce31cb4"[:64],
        size_bytes=407_410_402,
        token_env="HF_TOKEN",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_disk_space() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    if all((MODELS_ROOT / artifact.relative_path).exists() for artifact in ARTIFACTS):
        return
    free = shutil.disk_usage(MODELS_ROOT).free
    required = 25 * 1024**3
    if free < required:
        raise RuntimeError(
            f"at least 25 GiB free is required for first download; found {free / 1024**3:.1f} GiB"
        )


def download(artifact: Artifact) -> None:
    destination = MODELS_ROOT / artifact.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"verifying {artifact.name}: {destination}", flush=True)
        if sha256_file(destination) == artifact.sha256:
            print(f"verified {artifact.name}", flush=True)
            return
        raise RuntimeError(f"checksum mismatch for existing file: {destination}")

    token = os.environ.get(artifact.token_env)
    if not token:
        raise RuntimeError(
            f"{artifact.token_env} must be set to download {artifact.name}"
        )

    partial = destination.with_suffix(destination.suffix + ".part")
    resume_at = partial.stat().st_size if partial.exists() else 0
    if resume_at > artifact.size_bytes:
        partial.unlink()
        raise RuntimeError(
            f"partial file for {artifact.name} is larger than the pinned artifact; "
            "removed it so the next attempt starts cleanly"
        )
    if resume_at == artifact.size_bytes:
        print(f"verifying completed partial for {artifact.name}", flush=True)
        actual = sha256_file(partial)
        if actual == artifact.sha256:
            os.replace(partial, destination)
            print(f"verified {artifact.name}", flush=True)
            return
        partial.unlink()
        raise RuntimeError(
            f"completed partial for {artifact.name} failed checksum verification; "
            "removed it so the next attempt starts cleanly"
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "private-model-hosting/1.0",
    }
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    request = urllib.request.Request(artifact.url, headers=headers)

    print(f"downloading {artifact.name} to {destination}", flush=True)
    started = time.monotonic()
    try:
        with URL_OPENER.open(request, timeout=120) as response:
            resumed = resume_at > 0 and response.status == 206
            mode = "ab" if resumed else "wb"
            downloaded = resume_at if resumed else 0
            if resume_at and not resumed:
                print("  server did not accept resume; restarting download", flush=True)
            with partial.open(mode) as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (1024**3) < CHUNK_SIZE:
                        print(f"  {downloaded / 1024**3:.1f} GiB", flush=True)
                output.flush()
                os.fsync(output.fileno())
    except urllib.error.HTTPError as exc:
        print(
            f"download request failed with HTTP {exc.code} ({exc.reason})",
            file=sys.stderr,
            flush=True,
        )
        print(f"download interrupted; partial file retained at {partial}", flush=True)
        raise
    except (OSError, urllib.error.URLError):
        print(f"download interrupted; partial file retained at {partial}", flush=True)
        raise

    actual_size = partial.stat().st_size
    if actual_size < artifact.size_bytes:
        raise RuntimeError(
            f"incomplete download for {artifact.name}: expected "
            f"{artifact.size_bytes} bytes, received {actual_size}; "
            f"partial file retained for resume"
        )
    if actual_size > artifact.size_bytes:
        partial.unlink()
        raise RuntimeError(
            f"download for {artifact.name} exceeded the pinned size; "
            "removed the invalid partial file"
        )

    actual = sha256_file(partial)
    if actual != artifact.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {artifact.name}: expected {artifact.sha256}, got {actual}"
        )
    os.replace(partial, destination)
    elapsed = time.monotonic() - started
    print(f"downloaded and verified {artifact.name} in {elapsed:.1f}s", flush=True)


def main() -> int:
    ensure_disk_space()
    for artifact in ARTIFACTS:
        download(artifact)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"model preparation failed: {exc}", file=sys.stderr, flush=True)
        raise
