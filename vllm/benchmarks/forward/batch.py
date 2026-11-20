# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a `SchedulerOutput` for an exact batch, without the scheduler."""

from dataclasses import dataclass

import numpy as np

from vllm.benchmarks.forward.shapes import BenchCase
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.cache_memory_arena import CacheMemoryArena
from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroup, KVCacheSpec


@dataclass(slots=True)
class BenchBatch:
    scheduler_output: SchedulerOutput
    # req_id -> token IDs of the query tokens, in order.
    query_token_ids: dict[str, np.ndarray]
    # Per KV cache group: allocated (begin, end) block runs.
    group_block_ranges: list[list[tuple[int, int]]]
    num_blocks: int
    # req_id -> the runner's expected (scheduled, computed, is_decode).
    expected: dict[str, tuple[int, int, bool]]
    group_capacities: tuple[int, ...] | None = None

    @property
    def num_tokens(self) -> int:
        return self.scheduler_output.total_num_scheduled_tokens

    @property
    def req_ids(self) -> list[str]:
        return list(self.scheduler_output.num_scheduled_tokens)

    def finish(self) -> SchedulerOutput:
        """A step that only removes this batch's requests."""
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = set(self.req_ids)
        return output


def kv_block_size(spec: KVCacheSpec, dcp_size: int) -> int:
    """Tokens per block as the scheduler counts them: DCP spreads each
    block's tokens over the DCP ranks."""
    return spec.block_size * (dcp_size if spec.dcp_sharded else 1)


def _num_blocks(
    group: KVCacheGroup, computed: int, kv_len: int, dcp_size: int
) -> tuple[int, int]:
    """Return (leading null blocks, live blocks) for one request."""
    block_size = resolve_dcp_kv_block_size(group, dcp_size)
    config = group.manager_config
    if isinstance(config.params, MambaManager.Params):
        return 0, 1 + config.params.num_speculative_blocks
    if issubclass(config.manager_cls, CircularBufferManager):
        return 0, 1
    total = -(-kv_len // block_size)
    skipped = 0
    if isinstance(config.params, SlidingWindowManager.Params):
        # Blocks wholly before the window are freed to the null block.
        skipped = max(0, computed - config.params.sliding_window + 1) // block_size
    return skipped, total - skipped


def block_ranges(ids: list[int]) -> list[tuple[int, int]]:
    """Coalesce consecutive physical IDs without spanning holes."""
    ranges: list[tuple[int, int]] = []
    for block_id in ids:
        if ranges and ranges[-1][1] == block_id:
            ranges[-1] = (ranges[-1][0], block_id + 1)
        else:
            ranges.append((block_id, block_id + 1))
    return ranges


def build_batch(
    case: BenchCase,
    kv_cache_groups,
    num_kv_blocks: int,
    vocab_size: int,
    rng: np.random.Generator,
    dcp_size: int = 1,
    lookahead: int = 0,
    sampling_params: SamplingParams | None = None,
    kv_cache_config: KVCacheConfig | None = None,
) -> BenchBatch:
    """Allocate disjoint blocks and build `NewRequestData` for every request.

    Legacy groups share unique block IDs; native groups use the real byte arena
    to reserve disjoint pages and state slots. Block 0 is the null block.
    Lookahead reserves space for the drafter's KV writes.
    Every request uses `sampling_params` (default: `SamplingParams()`).

    With prefix P = K - Q (KV length K, query length Q):

    | Kind               | prompt tokens | num_computed_tokens | spec tokens |
    | ------------------ | ------------- | ------------------- | ----------- |
    | Prefill            | K             | P                   | -           |
    | Decode             | P             | P                   | -           |
    | Verify (D drafts)  | P             | P                   | D           |

    Unless chunked, a prefill's prefix is a prefix hit, which replays its last
    R = min(window, P) tokens on models with SWA bounded replay (window 0
    otherwise), like the scheduler: it schedules R + Q tokens from
    replay_start = num_computed_tokens = P - R.
    """
    replay_window = max(
        (group.prefix_replay_tokens for group in kv_cache_groups),
        default=0,
    )
    if sampling_params is None:
        sampling_params = SamplingParams()
    expected: dict[str, tuple[int, int, bool]] = {}
    cursor = 1
    arena = None
    pools: dict[int, BlockPool] = {}
    capacities = None
    if kv_cache_config is not None and kv_cache_config.pool_block_strides is not None:
        strides = kv_cache_config.pool_block_strides
        if len(strides) != len(kv_cache_groups):
            raise ValueError("Native cache strides must match the benchmark groups")
        arena = CacheMemoryArena(kv_cache_config.arena_size_bytes, strides)
        capacities = tuple(
            kv_cache_config.num_blocks_for_group(i) for i in range(len(strides))
        )
        pools = {
            stride: BlockPool(
                kv_cache_config.arena_size_bytes // stride,
                False,
                1,
                memory_arena=arena,
                block_stride_bytes=stride,
            )
            for stride in set(strides)
        }
    new_reqs = []
    query_token_ids = {}
    group_block_ranges: list[list[tuple[int, int]]] = [[] for _ in kv_cache_groups]
    output = SchedulerOutput.make_empty()
    for i, req in enumerate(case.requests):
        req_id = f"bench-{i}"
        replay = min(replay_window, req.prefix_len) if req.prefix_hit else 0
        computed = req.prefix_len - replay
        block_ids = []
        counts = [
            _num_blocks(group, computed, req.kv_len + lookahead, dcp_size)
            for group in kv_cache_groups
        ]
        if arena is not None:
            needs = [
                (pools[stride], live) for stride, (_, live) in zip(strides, counts)
            ]
            if not arena.can_allocate(needs, prepare=True):
                raise ValueError(
                    f"Case {case.name!r} exceeds the native aligned byte budget. "
                    "Increase the KV cache size."
                )
        for group_id, ((skipped, live), ranges) in enumerate(
            zip(counts, group_block_ranges)
        ):
            if arena is None:
                ids = list(range(cursor, cursor + live))
                ranges.append((cursor, cursor + live))
            else:
                ids = [
                    block.block_id
                    for block in pools[strides[group_id]].get_new_blocks(live)
                ]
                ranges.extend(block_ranges(ids))
            block_ids.append([0] * skipped + ids)
            cursor += live
        tokens = rng.integers(0, vocab_size, req.kv_len, dtype=np.int64)
        prompt = tokens[: req.prefix_len] if req.is_decode else tokens
        query_token_ids[req_id] = tokens[computed:]
        expected[req_id] = (req.query_len + replay, computed, req.is_decode)
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt.tolist(),
                prefill_token_ids=prompt.tolist(),
                mm_features=[],
                sampling_params=sampling_params,
                pooling_params=None,
                block_ids=tuple(block_ids),
                num_computed_tokens=computed,
                lora_request=None,
                replay_start=computed if replay else 0,
            )
        )
        output.num_scheduled_tokens[req_id] = req.query_len + replay
        if req.num_draft_tokens:
            output.scheduled_spec_decode_tokens[req_id] = tokens[
                req.prefix_len + 1 :
            ].tolist()
    if arena is None and cursor > num_kv_blocks:
        raise ValueError(
            f"Case {case.name!r} needs {cursor} KV blocks, but only "
            f"{num_kv_blocks} exist. Increase the KV cache size."
        )
    output.scheduled_new_reqs = new_reqs
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    return BenchBatch(
        output, query_token_ids, group_block_ranges, cursor, expected, capacities
    )
