# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare replicated QSA selection with TP row ownership, including exchange.

Run with torchrun --standalone --nproc-per-node=4. CUDA events measure the
slowest rank per sample; allocations and Python dispatch are included between
events. Inputs are reused in device caches. This is not a serving benchmark.
"""

import argparse
import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
    _prefill_logits,
    qsa_prefill_logits_width,
    qsa_select_paged_prefill,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[799, 4096, 8192])
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[32768, 65536, 131072]
    )
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=208)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.page_size > 0 and args.samples > 0 and args.warmup > 0
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    init_distributed_environment(
        world_size=world, rank=rank, distributed_init_method="env://"
    )
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    try:
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=world)
            group = get_tp_group()
            results = []
            for context in args.contexts:
                for rows in args.rows:
                    if rows > context:
                        continue
                    torch.manual_seed(0)
                    pages = (context // 4 + args.page_size - 1) // args.page_size
                    q = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device="cuda")
                    cache = torch.randn(
                        pages, args.page_size, 1, 128, dtype=q.dtype, device=q.device
                    )
                    table = torch.arange(pages, dtype=torch.int32, device=q.device)[
                        None
                    ]
                    starts = torch.tensor([0, rows], dtype=torch.int32, device=q.device)
                    visible = (
                        torch.arange(
                            context - rows + 1,
                            context + 1,
                            dtype=torch.int32,
                            device=q.device,
                        )
                        // 4
                    )
                    out = torch.empty(rows, 512, dtype=torch.int32, device=q.device)
                    width = qsa_prefill_logits_width(context, 4, pages * args.page_size)
                    reference_logits = _prefill_logits(
                        q,
                        cache,
                        table,
                        starts,
                        visible,
                        torch.empty(rows * width, dtype=torch.float32, device=q.device),
                        rows,
                        width,
                        0,
                        rows,
                    )
                    columns = torch.arange(width, device=q.device)[None, :]
                    reference_logits.masked_fill_(
                        columns >= visible[:, None], -torch.inf
                    )
                    reference_scores = reference_logits.topk(512).values.sort().values
                    for epoch, shard in enumerate((False, True, True, False)):
                        call = partial(
                            qsa_select_paged_prefill,
                            q,
                            cache,
                            table,
                            starts,
                            visible,
                            2048,
                            4,
                            rows,
                            out,
                            context,
                            tp_group=group if shard else None,
                        )

                        for _ in range(args.warmup):
                            call()
                        torch.accelerator.synchronize()
                        assert ((out >= -1) & (out < visible[:, None])).all()
                        ordered = out.sort().values
                        assert not (
                            (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
                        ).any()
                        scores = reference_logits.gather(1, out.clamp_min(0).long())
                        scores.masked_fill_(out < 0, -torch.inf)
                        torch.testing.assert_close(
                            scores.sort().values, reference_scores, rtol=0, atol=0
                        )
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        samples = []
                        for _ in range(args.samples):
                            dist.barrier(group=group.cpu_group)
                            begin.record()
                            call()
                            end.record()
                            end.synchronize()
                            samples.append(begin.elapsed_time(end) * 1000)
                        all_samples = [None] * world
                        dist.all_gather_object(
                            all_samples, samples, group=group.cpu_group
                        )
                        critical = [max(values) for values in zip(*all_samples)]
                        result = dict(
                            rows=rows,
                            context=context,
                            epoch=epoch,
                            sharded=shard,
                            critical_us=critical,
                            median_us=statistics.median(critical),
                        )
                        results.append(result)
                        if rank == 0:
                            print(
                                json.dumps(
                                    {
                                        k: v
                                        for k, v in result.items()
                                        if k != "critical_us"
                                    }
                                ),
                                flush=True,
                            )
            if rank == 0:
                args.output.write_text(
                    json.dumps(
                        {
                            "tp": world,
                            "gpu": torch.cuda.get_device_name(),
                            "logits_mb": envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB,
                            "page_size": args.page_size,
                            "scope": (
                                "warm-cache selector plus production all-gather; "
                                "maximum-rank CUDA event time"
                            ),
                            "results": results,
                        },
                        indent=2,
                    )
                )
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
