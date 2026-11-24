# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the H100 unsorted fusion with native BF16 activation and UE8M0 quant."""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.fused_moe.experts import triton_moe
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[10, 40, 80, 640, 40960])
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if any(rows <= 0 for rows in args.rows):
        parser.error("rows must be positive")
    torch.manual_seed(59)
    records = []
    for rows in args.rows:
        x = torch.randn((rows, 1280), device="cuda", dtype=torch.bfloat16) * 3
        act = torch.empty((rows, 640), device="cuda", dtype=torch.bfloat16)

        def reference(x=x, act=act):
            torch.ops._C.silu_and_mul(act, x)
            return per_token_group_quant_fp8(act, 128, use_ue8m0=True)

        fused = partial(triton_moe._silu_mul_quant_ue8m0, x)
        expected_q, expected_scale = reference()
        actual_q, actual_scale = fused()
        assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(actual_scale, expected_scale)
        record = {"rows": rows, "width": 640, "exact": True}
        for cold in (False, True):
            for name, fn in (("baseline", reference), ("fused", fused)):
                times = bench_gpu_time_with_cupti(
                    fn, cold_l2_cache=cold, use_cuda_graph=True, repeat_time_ms=100
                )
                record[f"{name}_{'cold' if cold else 'hot'}_us"] = (
                    statistics.median(times) * 1000
                )
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "gpu": current_platform.get_device_name(),
        "torch": torch.__version__,
        "runtime_sha256": hashlib.sha256(
            Path(triton_moe.__file__).read_bytes()
        ).hexdigest(),
        "boundary": "GPU kernels only; no compilation, input copies or diagnostics",
        "records": records,
    }
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
