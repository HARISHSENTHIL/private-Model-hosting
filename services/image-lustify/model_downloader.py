"""Download and verify the immutable model artifacts required by LUSTIFY."""

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

MODELS_ROOT = Path("/root/comfyui/models")
CHUNK_SIZE = 16 * 1024 * 1024


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
            # Civitai redirects to a signed CDN URL. urllib otherwise copies the
            # bearer token to that host, which both leaks it and can make the
            # CDN reject the request because the URL already carries auth.
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


ARTIFACTS = (
    Artifact(
        name="LUSTIFY v10 Krea 2 RAW BF16",
        url="https://civitai.com/api/download/models/3112728?fileId=3015314",
        relative_path="diffusion_models/lustify-v10-krea-raw-bf16.safetensors",
        sha256="04571b6cf9b9c8a3868ed3695c94de48d8206a37670ea36c6a043e86dda327d3",
        size_bytes=26_283_344_816,
        token_env="CIVITAI_API_TOKEN",
    ),
    Artifact(
        name="Qwen3-VL 4B BF16 Krea text encoder",
        url=(
            "https://huggingface.co/Comfy-Org/Krea-2/resolve/"
            "588feaf955b08b9ac4723eeb751752ab045dba71/"
            "text_encoders/qwen3vl_4b_bf16.safetensors"
        ),
        relative_path="text_encoders/qwen3vl_4b_bf16.safetensors",
        sha256="36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34",
        size_bytes=8_875_719_384,
        token_env="HF_TOKEN",
    ),
    Artifact(
        name="Wan 2.1 VAE",
        url=(
            "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/"
            "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d/"
            "split_files/vae/wan_2.1_vae.safetensors"
        ),
        relative_path="vae/wan_2.1_vae.safetensors",
        sha256="2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        size_bytes=253_815_318,
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
    required = 45 * 1024**3
    if free < required:
        raise RuntimeError(
            f"at least 45 GiB free is required for first download; found {free / 1024**3:.1f} GiB"
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
