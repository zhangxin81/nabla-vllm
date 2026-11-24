# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the real Qwen shared gate with its rounded sigmoid/multiply fusion."""

import argparse
import importlib.metadata
import json
import math
import statistics
import tempfile
import warnings
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.kernels.triton.fused_sigmoid_gate_mul import fused_sigmoid_gate_mul
from vllm.model_executor.layers.linear import ReplicatedLinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens", nargs="+", type=int, default=[1, 8, 32, 64, 128, 129, 1024]
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    assert int(importlib.metadata.version("cupti-python").split(".")[0]) >= 13
    assert all(tokens > 0 for tokens in args.num_tokens)
    props = torch.cuda.get_device_properties(0)
    results = []
    with (
        tempfile.TemporaryDirectory() as directory,
        set_current_vllm_config(VllmConfig()),
    ):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{directory}/store",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1, backend="gloo")
        try:
            with torch.inference_mode():
                with torch.device("cuda"):
                    gate = ReplicatedLinear(
                        2560, 1, bias=False, params_dtype=torch.bfloat16
                    )
                torch.manual_seed(args.seed)
                gate.weight.copy_(torch.randn_like(gate.weight) / math.sqrt(2560))
                gate.quant_method.process_weights_after_loading(gate)
                for tokens in args.num_tokens:
                    x = torch.randn(tokens, 2560, device="cuda", dtype=torch.bfloat16)
                    out = torch.randn_like(x)

                    def baseline(x=x, out=out):
                        return torch.sigmoid(gate(x)[0]) * out

                    def candidate(x=x, out=out):
                        projected_gate = gate(x)[0]
                        if 0 < out.shape[0] <= 8 and out.shape != projected_gate.shape:
                            return fused_sigmoid_gate_mul(out, projected_gate)
                        return torch.sigmoid(projected_gate) * out

                    expected, actual = baseline(), candidate()
                    assert torch.equal(actual, expected)
                    samples = {"baseline": [], "candidate": []}
                    for name, call in (
                        ("baseline", baseline),
                        ("candidate", candidate),
                        ("candidate", candidate),
                        ("baseline", baseline),
                    ):
                        call()
                        torch.accelerator.synchronize()
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", UserWarning)
                            samples[name].extend(
                                bench_gpu_time_with_cupti(
                                    call, use_cuda_graph=True, cold_l2_cache=True
                                )
                            )
                    medians = {
                        key: 1000 * statistics.median(value)
                        for key, value in samples.items()
                    }
                    row = dict(
                        tokens=tokens,
                        hidden=2560,
                        dtype="bfloat16",
                        median_us=medians,
                        speedup=medians["baseline"] / medians["candidate"],
                        samples_ms=samples,
                        bitwise=bool(torch.equal(actual, expected)),
                    )
                    results.append(row)
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "samples_ms"})
                    )
        finally:
            cleanup_dist_env_and_memory()
    if args.output_json:
        args.output_json.write_text(
            json.dumps(
                dict(
                    gpu=props.name,
                    capability=[props.major, props.minor],
                    sm_count=props.multi_processor_count,
                    torch=torch.__version__,
                    cupti=importlib.metadata.version("cupti-python"),
                    timer="CUPTI CUDA graph; cold L2; ReplicatedLinear plus gate tail",
                    seed=args.seed,
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
