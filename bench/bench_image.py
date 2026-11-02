#!/usr/bin/env python3
"""
Measure real per-image latency and queuing behavior for the two self-hosted
image services (image-qwen, image-lustify), so docs/ARCHITECTURE.md's
"per-image latency not yet measured" caveat can be replaced with a real
number.

Both services are expected to be batch-size-1 (one GPU, one image at a time
per docs/ARCHITECTURE.md) — this script confirms that empirically by firing
several requests concurrently and checking whether total time scales
linearly with count (queued) or stays flat (actually parallel), rather than
assuming the architecture doc's reasoning holds without checking.

USAGE:
  python bench/bench_image.py --url http://localhost:8002 --api-key $INTERNAL_API_KEY
  python bench/bench_image.py --url http://localhost:8003 --api-key $INTERNAL_API_KEY --concurrent 3
"""

import argparse
import asyncio
import time

import httpx

PROMPT = "a studio photograph of a red apple on a white background"


async def _one_image(client: httpx.AsyncClient, url: str) -> float:
    start = time.monotonic()
    resp = await client.post(
        "/v1/images/generations",
        json={"prompt": PROMPT, "n": 1, "size": "1024x1024"},
        timeout=180,
    )
    resp.raise_for_status()
    return time.monotonic() - start


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="e.g. http://localhost:8002")
    ap.add_argument("--api-key", required=True)
    ap.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="requests fired at once, to confirm queuing vs. real parallelism",
    )
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"}
    async with httpx.AsyncClient(base_url=args.url, headers=headers) as client:
        print("single-image baseline latency:")
        single = await _one_image(client, args.url)
        print(f"  {single:.2f}s")

        print(f"\n{args.concurrent} concurrent requests:")
        wall_start = time.monotonic()
        per_request = await asyncio.gather(
            *[_one_image(client, args.url) for _ in range(args.concurrent)]
        )
        wall_total = time.monotonic() - wall_start
        print(f"  per-request latencies: {[f'{t:.2f}s' for t in per_request]}")
        print(f"  wall-clock total: {wall_total:.2f}s")
        print(
            f"  if wall-clock total ~= {args.concurrent} x single-image baseline "
            f"({args.concurrent * single:.2f}s), requests are queuing serially "
            "on one GPU as docs/ARCHITECTURE.md expects. If it's much lower, "
            "the service is parallelizing more than assumed — update the docs."
        )


if __name__ == "__main__":
    asyncio.run(main())
