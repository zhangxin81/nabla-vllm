# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare fixed-address PLE history and request offsets without temporaries."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs", "num_padded_reqs"])
def _prepare_ple_inputs_kernel(
    mapping,
    computed,
    history,
    source_qsl,
    context,
    target_qsl,
    num_reqs,
    num_padded_reqs,
    history_stride: tl.constexpr,
    context_len: tl.constexpr,
    capacity: tl.constexpr,
    eos: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = x // context_len
    col = x % context_len
    active = row < num_reqs
    req = tl.load(mapping + row, mask=active, other=0).to(tl.int64)
    end = tl.load(computed + req, mask=active, other=0).to(tl.int64)
    pos = end - context_len + col
    ids = tl.load(
        history + req * history_stride + pos, mask=active & (pos >= 0), other=eos
    )
    tl.store(context + x, ids, mask=x < capacity * context_len)
    qidx = tl.minimum(x, num_padded_reqs)
    q = tl.load(source_qsl + qidx, mask=x <= capacity, other=0)
    tl.store(target_qsl + x, q, mask=x <= capacity)


def prepare_ple_inputs(
    mapping: torch.Tensor,
    computed: torch.Tensor,
    history: torch.Tensor,
    query_start_loc: torch.Tensor,
    context: torch.Tensor,
    target_query_start_loc: torch.Tensor,
    num_reqs: int,
    num_padded_reqs: int,
    eos: int,
) -> None:
    """Fill `[capacity, depth]` history and `[capacity + 1]` request offsets.

    Inactive history rows contain EOS; unused offsets repeat the batch end.
    History may reside in GPU memory or mapped host memory.
    """
    capacity, depth = context.shape
    _prepare_ple_inputs_kernel[
        (triton.cdiv(max(capacity * depth, capacity + 1), 128),)
    ](
        mapping,
        computed,
        history,
        query_start_loc,
        context,
        target_query_start_loc,
        num_reqs,
        num_padded_reqs,
        history_stride=history.stride(0),
        context_len=depth,
        capacity=capacity,
        eos=eos,
        BLOCK=128,
    )
