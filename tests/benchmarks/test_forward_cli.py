# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.core.utils import new_kv_cache_group
from vllm.benchmarks.forward.batch import build_batch
from vllm.benchmarks.forward.capture import write_input_ids
from vllm.benchmarks.forward.kv_init import touched_kv_views, zero_kv_blocks
from vllm.benchmarks.forward.main import run_case
from vllm.benchmarks.forward.profile import summarize
from vllm.benchmarks.forward.routing import RoutingFixture, zipf_topk_ids
from vllm.benchmarks.forward.shapes import jitter_kv_lens, parse_case
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.mla.cpu_mla import CPUMLABackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec, MLAAttentionSpec
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_synthetic_expert_ids_preserve_inactive_slots(keyword, dtype):
    """Replacing routes must not reactivate padding or partially masked rows."""
    from vllm.forward_context import ForwardContext, override_forward_context

    fixture = torch.from_numpy(zipf_topk_ids(4, 8, 2, np.random.default_rng(0)))
    expected = fixture.to(dtype).clone()
    ids = torch.tensor([[1, 2], [-1, -1], [3, -1], [-1, 4]], dtype=dtype)
    weights = torch.tensor([[0.6, 0.4], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    hook = RoutingFixture._substitute_ids(fixture, 1, [0], 0)
    context = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    with override_forward_context(context):
        args = (weights,) if keyword else (weights, ids)
        kwargs = {"topk_ids": ids} if keyword else {}
        replaced_args, replaced_kwargs = hook(torch.nn.Identity(), args, kwargs)
        routed = replaced_kwargs["topk_ids"] if keyword else replaced_args[1]
        assert routed.dtype == dtype
        assert torch.equal(routed[ids == -1], ids[ids == -1])
        assert torch.equal(routed[ids >= 0], expected[ids >= 0])
        assert replaced_args[0] is weights

        # A later replay can activate rows without inheriting an earlier mask.
        active_ids = torch.zeros_like(ids)
        args = (weights,) if keyword else (weights, active_ids)
        kwargs = {"topk_ids": active_ids} if keyword else {}
        replaced_args, replaced_kwargs = hook(torch.nn.Identity(), args, kwargs)
        routed = replaced_kwargs["topk_ids"] if keyword else replaced_args[1]
        assert torch.equal(routed, expected)


def test_zipf_expert_selection():
    """Popularity follows Zipf, with distinct experts and reproducible rows."""
    rng = np.random.default_rng(0)
    ids = zipf_topk_ids(50000, 32, 1, rng)
    frequencies = np.sort(np.bincount(ids.ravel(), minlength=32))[::-1] / len(ids)
    expected = np.arange(1, 33, dtype=float) ** -1.25
    expected /= expected.sum()
    np.testing.assert_allclose(frequencies, expected, atol=0.005, rtol=0)

    ids = zipf_topk_ids(2048, 384, 6, np.random.default_rng(0))
    assert ((ids >= 0) & (ids < 384)).all()
    assert (np.diff(np.sort(ids, axis=1), axis=1) > 0).all()
    assert np.array_equal(ids[:6], zipf_topk_ids(6, 384, 6, np.random.default_rng(0)))


@pytest.mark.parametrize("rank", [0, 1])
def test_pcp_query_tokens(rank):
    """PCP rows receive their query slice, including a continued prefix."""
    batch = build_batch(
        parse_case("mix=16@32c,1@64"), [], 1, 1000, np.random.default_rng(0)
    )
    buffers = InputBuffers(3, 12, torch.device("cpu"))
    local = InputBatch.make_dummy(3, 9, buffers)
    offsets = [4 * rank, 4 * (3 - rank)]
    local = replace(
        local,
        req_ids=["bench-1", "bench-0", "bench-0"],
        num_scheduled_tokens=np.array([1, 4, 4]),
        num_computed_tokens_np=np.array([63, 16 + offsets[0], 16 + offsets[1]]),
    )
    buffers.input_ids.fill_(-1)
    write_input_ids(batch, local, buffers.input_ids)
    prefill = batch.query_token_ids["bench-0"]
    expected = np.concatenate(
        [batch.query_token_ids["bench-1"]]
        + [prefill[start : start + 4] for start in offsets]
    )
    assert buffers.input_ids[:9].tolist() == expected.tolist()
    assert buffers.input_ids[9:].tolist() == [-1, -1, -1]


def _group(block_size: int, replay: int = 0, dcp_sharded: bool = False):
    return SimpleNamespace(
        block_size=block_size,
        prefix_replay_tokens=replay,
        dcp_sharded=dcp_sharded,
        manager_config=FullAttentionManager.config(),
    )


def test_build_batch_layout():
    """The runner sees each request kind as the scheduler would schedule it."""
    case = parse_case("c=2*1@101,4@64:3,32,8@48c,16@316")
    groups = [_group(16), _group(32, replay=128)]
    batch = build_batch(case, groups, 1000, 1000, np.random.default_rng(0))

    # (scheduled, computed, is_decode) per request: decode, verify, prefill,
    # chunked prefill, then a prefix hit replaying the last 128 hit tokens.
    assert list(batch.expected.values()) == [
        (1, 100, True),
        (1, 100, True),
        (4, 60, True),
        (32, 0, False),
        (8, 40, False),
        (16 + 128, 300 - 128, False),
    ]
    output = batch.scheduler_output
    assert output.total_num_scheduled_tokens == 1 + 1 + 4 + 32 + 8 + 144
    assert len(output.scheduled_spec_decode_tokens["bench-2"]) == 3
    assert output.scheduled_new_reqs[-1].replay_start == 300 - 128
    for req_id, (scheduled, _, _) in batch.expected.items():
        assert len(batch.query_token_ids[req_id]) == scheduled
    # Groups share one block pool: block IDs must not overlap.
    ranges = sorted(r for group in batch.group_block_ranges for r in group)
    assert ranges[0][0] == 1
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))

    with pytest.raises(ValueError, match="KV blocks"):
        build_batch(case, groups, 10, 1000, np.random.default_rng(0))

    # DCP spreads each block's tokens over the DCP ranks, like the scheduler.
    groups = [_group(16, dcp_sharded=True), _group(16)]
    batch = build_batch(
        parse_case("c=1@640"), groups, 1000, 1000, np.random.default_rng(0), 4
    )
    assert batch.group_block_ranges == [[(1, 11)], [(11, 51)]]
    with_drafts = build_batch(
        parse_case("c=1@640"),
        groups,
        1000,
        1000,
        np.random.default_rng(0),
        dcp_size=4,
        lookahead=4,
    )
    assert with_drafts.group_block_ranges == [[(1, 12)], [(12, 53)]]
    assert with_drafts.expected == batch.expected


def test_jitter_kv_lens():
    """Prefixes vary, capped at the max length; queries and fresh prefills don't.
    A case's draw depends only on the seed and its name, so adding or reordering
    other cases doesn't change it."""
    case = parse_case("c=1000*1@1001,4@64:3,32")
    jittered = jitter_kv_lens(case, 0.5, 1400, 0)
    assert jitter_kv_lens(case, 0.5, 1400, 0) == jittered
    other = jitter_kv_lens(replace(case, name="d"), 0.5, 1400, 0)
    assert other.requests != jittered.requests
    prefixes = np.array([r.prefix_len for r in jittered.requests[:1000]])
    assert prefixes.min() >= 500 and max(r.kv_len for r in jittered.requests) == 1400
    assert prefixes.std() > 200
    assert [r.query_len for r in jittered.requests] == [
        r.query_len for r in case.requests
    ]
    assert jittered.requests[-1] == case.requests[-1]
    assert jitter_kv_lens(case, 0, 2048, 0) == case
    with pytest.raises(ValueError, match="KV length"):
        jitter_kv_lens(case, 0, 1000, 0)


def test_bench_packed_kv_blocks():
    """Zero allocated blocks without corrupting neighboring layers or blocks."""
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        tokens_per_state=4,
    )
    group = AttentionGroup(CPUMLABackend, ["cache"], spec, 0, kernel_block_size=4)
    storage = torch.ones(16, 2, 1, 4, 8)
    manager_view = storage[:, 0]
    kernel_view = group.map_kv_cache_to_kernel_blocks(manager_view)
    batch = build_batch(
        parse_case("decode=1@129"),
        [new_kv_cache_group(["cache"], spec)],
        16,
        1000,
        np.random.default_rng(0),
    )
    caches = {"cache": kernel_view}
    zero_kv_blocks(caches, [group], batch, 16)
    allocated = torch.zeros(storage.shape[0], dtype=torch.bool)
    for begin, end in batch.group_block_ranges[0]:
        allocated[begin:end] = True
    assert not storage[allocated, 0].count_nonzero()
    assert (storage[~allocated, 0] == 1).all()
    assert (storage[:, 1] == 1).all()


def test_padded_page_spreads_kernel_blocks():
    """A padded page's kernel blocks sit evenly spaced, and its manager block
    spans all of them."""
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        tokens_per_state=4,
        page_size_padded=4 * 48,
    )
    group = AttentionGroup(CPUMLABackend, ["cache"], spec, 0, kernel_block_size=4)
    storage = torch.arange(4 * 60.0).view(4, 60)
    manager_view = storage.as_strided((4, 1, 4, 8), (60, 32, 8, 1))
    kernel_view = group.map_kv_cache_to_kernel_blocks(manager_view)
    assert kernel_view[:, 0, 0, 0].tolist() == list(range(0, 4 * 60 - 12, 12))
    manager_view = group.map_kv_cache_to_manager_blocks(kernel_view, 4)
    assert manager_view[:, 0, 0, 0].tolist() == [0, 60, 120, 180]
    assert manager_view[0].numel() >= 3 * 12 + 8


def test_bench_speculative_mamba_blocks():
    """Verify requests own and restore every recurrent scratch state."""
    spec = MambaSpec(
        block_size=16,
        shapes=((4,), (2, 2)),
        dtypes=(torch.float32, torch.float32),
    )
    num_speculative_blocks = 5
    config = MambaManager.config(num_speculative_blocks=num_speculative_blocks)
    batch = build_batch(
        parse_case("verify=2*6@128:5"),
        [new_kv_cache_group(["state"], spec, config)],
        13,
        1000,
        np.random.default_rng(0),
    )
    blocks = [req.block_ids[0] for req in batch.scheduler_output.scheduled_new_reqs]
    assert all(len(ids) >= 1 + num_speculative_blocks for ids in blocks)
    assert set(blocks[0]).isdisjoint(blocks[1])
    group = AttentionGroup(
        MambaAttentionBackendEnum.MAMBA2.get_class(), ["state"], spec, 0
    )
    caches = {"state": [torch.ones(13, *shape) for shape in spec.shapes]}
    zero_kv_blocks(caches, [group], batch, 13)
    views = touched_kv_views(caches, [group], batch, 13, 1)
    snapshots = [view.clone() for view in views]
    for state in caches["state"]:
        state[1:].fill_(7)
    for view, saved in zip(views, snapshots):
        view.copy_(saved)
    for state in caches["state"]:
        assert torch.equal(state[0], torch.ones_like(state[0]))
        assert torch.count_nonzero(state[1:]) == 0


@pytest.fixture(params=[False, True])
def native_cache_config(request):
    kv_spec = MLAAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float32
    )
    state_spec = MambaSpec(block_size=4, shapes=((16,),), dtypes=(torch.float32,))
    specs = [kv_spec, state_spec]
    if request.param:
        specs.reverse()
    groups = [new_kv_cache_group([f"cache{i}"], spec) for i, spec in enumerate(specs)]
    strides = tuple(spec.page_size_bytes for spec in specs)
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        pool_block_strides=strides,
        arena_size_bytes=512,
    )


def test_bench_native_cache_byte_ownership(native_cache_config):
    """Initialize and restore disjoint KV/state bytes without touching neighbors."""
    config = native_cache_config
    groups = config.kv_cache_groups
    specs = [group.distinct_specs[0] for group in groups]
    strides = config.pool_block_strides
    batch = build_batch(
        parse_case("decode=2*1@9"),
        groups,
        config.num_blocks,
        1000,
        np.random.default_rng(0),
        kv_cache_config=config,
    )
    spans = sorted(
        (block_id * stride, (block_id + 1) * stride)
        for req in batch.scheduler_output.scheduled_new_reqs
        for stride, ids in zip(strides, req.block_ids)
        for block_id in ids
        if block_id
    )
    assert spans[0][0] >= max(strides)
    assert all(end <= start for (_, end), (start, _) in zip(spans, spans[1:]))
    assert spans[-1][1] <= config.arena_size_bytes
    assert sum(end - start for start, end in spans) == 320
    assert any(
        any(b != a + 1 for a, b in zip(ids, ids[1:]))
        for req in batch.scheduler_output.scheduled_new_reqs
        for ids in req.block_ids
    )

    storage = torch.ones(128, dtype=torch.float32)
    caches, attn_groups = {}, []
    for i, spec in enumerate(specs):
        capacity = config.num_blocks_for_group(i)
        if isinstance(spec, MambaSpec):
            caches[f"cache{i}"] = storage.as_strided((capacity, 16), (16, 1))
            backend = MambaAttentionBackendEnum.MAMBA2.get_class()
        else:
            caches[f"cache{i}"] = storage.as_strided((capacity, 1, 4, 2), (8, 8, 2, 1))
            backend = CPUMLABackend
        attn_groups.append(AttentionGroup(backend, [f"cache{i}"], spec, i))
    zero_kv_blocks(caches, attn_groups, batch, config.num_blocks)
    owned = torch.zeros(128, dtype=torch.bool)
    for start, end in spans:
        owned[start // 4 : end // 4] = True
    assert storage[owned].count_nonzero() == 0
    assert (storage[~owned] == 1).all()
    views = touched_kv_views(caches, attn_groups, batch, config.num_blocks, 1)
    snapshots = [view.clone() for view in views]
    storage[owned] = 7
    for view, saved in zip(views, snapshots):
        view.copy_(saved)
    assert storage[owned].count_nonzero() == 0
    assert (storage[~owned] == 1).all()


def test_bench_native_cache_rejects_oversized_case(native_cache_config):
    """Reject native cases that exceed physical bytes despite valid virtual IDs."""
    config = native_cache_config
    with pytest.raises(ValueError, match="byte budget"):
        build_batch(
            parse_case("decode=4*1@9"),
            config.kv_cache_groups,
            config.num_blocks,
            1000,
            np.random.default_rng(0),
            kv_cache_config=config,
        )


@pytest.mark.parametrize("capacity", [0, 2])
def test_bench_forward_rejects_oversized_verify(capacity):
    """Reject drafts beyond the runner's buffer before launching CUDA kernels."""
    runner = SimpleNamespace(num_speculative_steps=capacity)
    case = parse_case("verify=2*4@64:3")
    with pytest.raises(
        ValueError,
        match=rf"Case 'verify' needs 3 draft tokens.*supports {capacity}",
    ):
        run_case(runner, case, SimpleNamespace(), SamplingParams())


def test_profile_summary():
    """Each replay's leading restore/rendezvous activities are left out, and
    kernels are grouped by name without template or function arguments."""

    def act(name, ts, dur, cat="kernel"):
        return {"cat": cat, "name": name, "ts": ts, "dur": dur}

    replay = [
        ("Memcpy DtoD (Device -> Device)", "gpu_memcpy", 5),
        ("void ns::gemm<128>(Params)", "kernel", 10),
        ("void ns::gemm<64>(Params)", "kernel", 20),
        ("std::enable_if<(1>0), void>::type ns::norm<float>(float*)", "kernel", 4),
    ]
    events = [{"cat": "cpu_op", "name": "x", "ts": 0, "dur": 1}]
    for r in range(2):
        for i, (name, cat, dur) in enumerate(replay):
            events.append(act(name, 1000 * r + 100 * i, dur, cat))
    summary = summarize(events, replays=2, skip=1)
    assert summary["pre_forward_us"] == 5
    assert summary["busy_us"] == 34
    assert summary["span_us"] == 204
    assert [(k["name"], k["us"], k["calls"]) for k in summary["kernels"]] == [
        ("ns::gemm", 30, 2),
        ("ns::norm", 4, 1),
    ]
    with pytest.raises(ValueError, match="replays"):
        summarize(events[:-1], replays=2, skip=1)
