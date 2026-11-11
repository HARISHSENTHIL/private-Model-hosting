#!/usr/bin/env python3
"""
Measure real per-mesh latency, queuing behavior, and idle-unload cold-reload
cost for the Hunyuan3D-2 service, so docs/ARCHITECTURE.md's capacity-model
numbers for this service stop being pure hypotheses.

Mirrors bench/bench_image.py's baseline + N-concurrent probe, plus one extra
probe specific to this service: fire a request, wait past
HUNYUAN3D_IDLE_UNLOAD_SECONDS, fire again, and report the cold-reload delta -
LUSTIFY never unloads so has nothing equivalent to measure.

USAGE:
  python bench/bench_mesh.py --url http://localhost:8005 --api-key $INTERNAL_API_KEY
  python bench/bench_mesh.py --url http://localhost:8005 --api-key $INTERNAL_API_KEY --concurrent 3
  python bench/bench_mesh.py --url http://localhost:8005 --api-key $INTERNAL_API_KEY \
      --idle-reload-probe --idle-unload-seconds 600
"""

import argparse
import asyncio
import base64
import time

import httpx

# A tiny valid PNG (1x1 white pixel) - enough to exercise the image-to-3D
# path's decode/validation logic without needing a real test asset on disk.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)
PROMPT = "a low-poly wooden treasure chest"


async def _one_mesh(client: httpx.AsyncClient, *, use_prompt: bool) -> float:
    body = {"prompt": PROMPT} if use_prompt else {"image_b64": _TINY_PNG_B64}
    start = time.monotonic()
    resp = await client.post("/v1/3d/generations", json=body, timeout=1500)
    resp.raise_for_status()
    data = resp.json()["data"][0]
    assert base64.b64decode(data["glb_b64"])  # confirm it's at least decodable
    return time.monotonic() - start


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="e.g. http://localhost:8005")
    ap.add_argument("--api-key", required=True)
    ap.add_argument(
        "--concurrent",
        type=int,
        default=2,
        help="requests fired at once, to confirm the worker thread serializes jobs",
    )
    ap.add_argument(
        "--use-prompt",
        action="store_true",
        help="exercise the text-to-3D path (requires image-qwen running) instead of image-to-3D",
    )
    ap.add_argument(
        "--idle-reload-probe",
        action="store_true",
        help="additionally measure cold-reload latency after an idle-unload",
    )
    ap.add_argument(
        "--idle-unload-seconds",
        type=int,
        default=600,
        help="must match the service's HUNYUAN3D_IDLE_UNLOAD_SECONDS",
    )
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"}
    async with httpx.AsyncClient(base_url=args.url, headers=headers) as client:
        print("single-mesh baseline latency (warm or cold, whichever this is):")
        single = await _one_mesh(client, use_prompt=args.use_prompt)
        print(f"  {single:.2f}s")

        print(f"\n{args.concurrent} concurrent requests:")
        wall_start = time.monotonic()
        per_request = await asyncio.gather(
            *[_one_mesh(client, use_prompt=args.use_prompt) for _ in range(args.concurrent)]
        )
        wall_total = time.monotonic() - wall_start
        print(f"  per-request latencies: {[f'{t:.2f}s' for t in per_request]}")
        print(f"  wall-clock total: {wall_total:.2f}s")
        print(
            f"  if wall-clock total ~= {args.concurrent} x single-mesh baseline "
            f"({args.concurrent * single:.2f}s), the worker thread is serializing "
            "jobs as docs/ARCHITECTURE.md expects."
        )

        if args.idle_reload_probe:
            wait_for = args.idle_unload_seconds + 30
            print(
                f"\nwaiting {wait_for}s past HUNYUAN3D_IDLE_UNLOAD_SECONDS "
                f"({args.idle_unload_seconds}s) to force an idle-unload..."
            )
            await asyncio.sleep(wait_for)
            health = await client.get("/health", timeout=10)
            print(f"  /health before reload request: {health.json()}")
            reload_latency = await _one_mesh(client, use_prompt=args.use_prompt)
            print(f"  cold-reload request latency: {reload_latency:.2f}s")
            print(f"  warm baseline was: {single:.2f}s")
            print(
                f"  cold-reload overhead: {reload_latency - single:.2f}s "
                "(this is the pipeline-load cost paid on the next request after idle-unload)"
            )


if __name__ == "__main__":
    asyncio.run(main())
