#!/usr/bin/env python3
"""
Load-test the self-hosted LLM service and report real latency/throughput at
increasing concurrency, so docs/ARCHITECTURE.md's "~60-150 concurrent
requests" estimate can be replaced with a measured number.

This measures end-to-end HTTP latency, not the engine's internal KV-cache
accounting — also read the SGLang startup log directly (grepped by
--show-startup-log) for its own "max concurrency" figure, which is the
authoritative number this script is approximating from outside.

USAGE:
  python bench/bench_llm.py --url http://localhost:8001 --api-key $INTERNAL_API_KEY \
      --concurrency 1 10 30 60 100
"""

import argparse
import asyncio
import statistics
import time

import httpx

PROMPT = "Explain, in two sentences, why the sky is blue."


async def _one_request(client: httpx.AsyncClient, url: str, model: str) -> float:
    start = time.monotonic()
    resp = await client.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 128,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return time.monotonic() - start


async def _run_at_concurrency(
    url: str, api_key: str, model: str, concurrency: int
) -> tuple[list[float], int, float]:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [_one_request(client, url, model) for _ in range(concurrency)]
        started = time.monotonic()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_time = time.monotonic() - started
    latencies = [result for result in results if isinstance(result, float)]
    errors = len(results) - len(latencies)
    return latencies, errors, wall_time


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="e.g. http://localhost:8001")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", default="huihui-qwen3-27b-abliterated")
    ap.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 10, 30, 60, 100, 150],
        help="concurrency levels to sweep",
    )
    args = ap.parse_args()

    print(
        f"{'concurrency':>12} {'p50_s':>8} {'p95_s':>8} "
        f"{'max_s':>8} {'req_s':>8} {'errors':>8}"
    )
    for c in args.concurrency:
        latencies, errors, wall_time = await _run_at_concurrency(
            args.url, args.api_key, args.model, c
        )
        if not latencies:
            print(f"{c:>12}  all requests failed ({errors} errors)")
            continue
        p50 = statistics.median(latencies)
        p95 = (
            statistics.quantiles(latencies, n=20)[18]
            if len(latencies) > 1
            else latencies[0]
        )
        throughput = len(latencies) / wall_time
        print(
            f"{c:>12} {p50:>8.2f} {p95:>8.2f} {max(latencies):>8.2f} "
            f"{throughput:>8.2f} {errors:>8}"
        )
        print(
            "  -> if p50/p95 jump sharply vs. the previous row, that's the "
            "concurrency ceiling — cross-check against the engine's own "
            "startup-log KV-cache figure, not just this number."
        )


if __name__ == "__main__":
    asyncio.run(main())
