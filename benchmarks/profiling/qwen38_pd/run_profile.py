# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm and profile an existing Qwen P/D server pair, one case at a time."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def batch(base_url, input_len, concurrency, output_len):
    async with httpx.AsyncClient(timeout=1800) as client:

        async def request(index):
            # Explicit token IDs fix the input length and avoid tokenizer timing.
            body = {
                "model": "qwen-profile",
                "prompt": [
                    1000 + ((i * 17 + index * 31) % 16000) for i in range(input_len)
                ],
                "max_tokens": output_len,
                "ignore_eos": True,
                "temperature": 0,
                "seed": 1234 + index,
                "stream": False,
            }
            begin = time.monotonic()
            response = await client.post(base_url + "/v1/completions", json=body)
            response.raise_for_status()
            data = response.json()
            assert data["usage"]["prompt_tokens"] == input_len, data["usage"]
            assert data["usage"]["completion_tokens"] == output_len, data["usage"]
            return {
                "index": index,
                "elapsed_s": time.monotonic() - begin,
                "usage": data["usage"],
            }

        begin = time.monotonic()
        requests = await asyncio.gather(*(request(i) for i in range(concurrency)))
        return {"batch_elapsed_s": time.monotonic() - begin, "requests": requests}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://127.0.0.1:8392")
    parser.add_argument("--prefill", default="http://127.0.0.1:8310")
    parser.add_argument("--decode", default="http://127.0.0.1:8320")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (
        [(128, 1, 8)]
        if args.pilot
        else [
            (8192, 1, 128),
            (8192, 8, 128),
            (8192, 32, 128),
            (128, 8, 128),
            (2048, 8, 128),
            (32768, 8, 128),
        ]
    )
    async with httpx.AsyncClient(timeout=1800) as client:
        for input_len, concurrency, output_len in cases:
            name = f"i{input_len}-c{concurrency}-o{output_len}"
            output = args.output / f"{name}.json"
            assert not output.exists(), f"Refusing to overwrite {output}"
            warm = await batch(args.proxy, input_len, concurrency, output_len)
            await asyncio.sleep(1)
            active = []
            try:
                for backend in (args.prefill, args.decode):
                    response = await client.post(
                        backend + "/start_profile",
                        json={
                            "profile_prefix": name,
                        },
                    )
                    response.raise_for_status()
                    active.append(backend)
                measured = await batch(args.proxy, input_len, concurrency, output_len)
            finally:
                for backend in reversed(active):
                    response = await client.post(backend + "/stop_profile")
                    response.raise_for_status()
            record = {
                "case": name,
                "input_len": input_len,
                "concurrency": concurrency,
                "output_len": output_len,
                "warmup": warm,
                "profiled": measured,
                "instrumented_latency_is_not_a_serving_benchmark": True,
            }
            output.write_text(json.dumps(record, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "case_complete": name,
                        "profiled_elapsed_s": measured["batch_elapsed_s"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
