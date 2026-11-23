# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run reproducible serving shapes against an already started 4P4D proxy."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

MAIN_LENGTHS = (
    (128, 128),
    (2048, 128),
    (8192, 128),
    (32768, 128),
    (128, 1024),
    (2048, 1024),
    (8192, 1024),
)
CAPACITY_LENGTHS = ((65536, 128), (32768, 1024))
CONCURRENCIES = (1, 8, 32, 64, 128)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--reference-arm", default="upstream_nixl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8192")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    model = manifest["models"][args.model]
    directory = args.output_root / args.model / args.arm
    if not args.dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        saved_manifest = directory / "experiment.json"
        if saved_manifest.exists() and saved_manifest.read_bytes() != manifest_bytes:
            raise ValueError("Result directory belongs to a different experiment")
        saved_manifest.write_bytes(manifest_bytes)
        from vllm.benchmarks.serve import add_cli_args, main_async
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        benchmark_parser = FlexibleArgumentParser()
        add_cli_args(benchmark_parser)

    for input_len, output_len in (*MAIN_LENGTHS, *CAPACITY_LENGTHS):
        concurrencies = CONCURRENCIES
        if (input_len, output_len) in CAPACITY_LENGTHS:
            concurrencies = (1, 8)
        elif args.round > 1:
            reference = args.output_root / args.model / args.reference_arm
            throughput = {}
            for concurrency in CONCURRENCIES:
                name = f"isl{input_len}-osl{output_len}-c{concurrency}-r1.json"
                result = json.loads((reference / name).read_text())
                if result["completed"] != result["num_prompts"] or result.get("failed"):
                    raise ValueError(f"Incomplete reference result: {reference / name}")
                throughput[concurrency] = result["output_throughput"]
            peak = max(throughput.values())
            saturation = min(
                c for c, value in throughput.items() if value >= 0.95 * peak
            )
            concurrencies = tuple(sorted({1, 8, saturation}))

        for concurrency in concurrencies:
            samples = max(8, 2 * concurrency)
            name = f"isl{input_len}-osl{output_len}-c{concurrency}-r{args.round}.json"
            output = directory / name
            if output.exists():
                result = json.loads(output.read_text())
                if result["completed"] == samples and not result.get("failed"):
                    continue
                raise ValueError(f"Incomplete result requires inspection: {output}")
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "bench",
                "serve",
                "--backend",
                "vllm",
                "--model",
                args.model,
                "--tokenizer",
                model["weight_path"],
                "--base-url",
                args.base_url,
                "--endpoint",
                "/v1/completions",
                "--dataset-name",
                "random",
                "--random-input-len",
                str(input_len),
                "--random-output-len",
                str(output_len),
                "--random-range-ratio",
                "0",
                "--random-prefix-len",
                "0",
                "--ignore-eos",
                "--temperature",
                "0",
                "--request-rate",
                "inf",
                "--max-concurrency",
                str(concurrency),
                "--num-prompts",
                str(samples),
                "--num-warmups",
                "1",
                "--seed",
                str(42 + args.round),
                "--disable-tqdm",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--metric-percentiles",
                "50,95,99",
                "--save-result",
                "--save-detailed",
                "--result-dir",
                str(directory),
                "--result-filename",
                name,
            ]
            if tokenizer_mode := model.get("tokenizer_mode"):
                command.extend(("--tokenizer-mode", tokenizer_mode))
            if args.dry_run:
                print(json.dumps(command))
                continue
            print(f"CASE_START {args.model} {args.arm} {name}", flush=True)
            benchmark_args = benchmark_parser.parse_args(command[5:])
            await asyncio.wait_for(main_async(benchmark_args), timeout=1800)
            result = json.loads(output.read_text())
            if result["completed"] != samples or result.get("failed"):
                raise RuntimeError(f"Incomplete requests in {output}")
            print(
                f"CASE_DONE {name} {result['output_throughput']:.3f} output tok/s",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
