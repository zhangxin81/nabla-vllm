# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical checkpoint retention; physical slot allocation belongs to the arena."""

from collections.abc import Sequence

from vllm.v1.kv_cache_interface import get_reusable_prefix_length


def prefill_chunk_checkpoint(
    chunk_end: int,
    num_prompt_tokens: int,
    num_cacheable_tokens: int,
    alignment_tokens: int,
) -> int | None:
    """Keep a complete prefill state before a prompt's possibly different suffix."""
    if (
        0 < chunk_end < num_prompt_tokens
        and chunk_end <= num_cacheable_tokens
        and chunk_end % alignment_tokens == 0
    ):
        return chunk_end
    return None


def replay_boundaries(
    num_prompt_tokens: int,
    num_hash_lookahead_tokens: int,
) -> tuple[int, int]:
    """Prompt positions that sparse retention keeps reachable.

    - Resend of the same prompt: `get_reusable_prefix_length`, which leaves
      the last token for logits and the lookahead tokens for the hash.
    - Request extending the prompt (e.g. the next turn): the full prompt. With
      Mamba and L >= 2 this adds nothing: the state is freed before its
      lookahead hash exists.

    `reachable_block_mask` floors both to the hit alignment, so they keep
    different tails only when a boundary falls in the prompt's last
    `max(1, num_hash_lookahead_tokens)` tokens.
    """
    resend = get_reusable_prefix_length(
        num_prompt_tokens, num_hash_lookahead_tokens=num_hash_lookahead_tokens
    )
    return resend, num_prompt_tokens


def mamba_checkpoint_mask(
    start_block: int,
    end_block: int,
    alignment_tokens: int | None,
    block_size: int,
    retention_interval: int | None,
    reachable_boundaries: Sequence[int],
) -> list[bool] | None:
    """Select resumable state positions using token history, independent of bytes."""
    if retention_interval is None or alignment_tokens is None:
        # Dense caching or no alignment constraint imposed.
        return None
    mask = [False] * (end_block - start_block)

    # (1) Segment-boundary states. Block ``i`` ends at token
    # ``(i + 1) * block_size``.
    segment_tokens = None if retention_interval == 0 else retention_interval
    if segment_tokens is not None:
        per_segment = segment_tokens // block_size
        if per_segment <= 1:
            # Interval at/below the block size: every block is a boundary.
            return None
        first_boundary = (start_block + per_segment) // per_segment * per_segment - 1
        for i in range(first_boundary - start_block, len(mask), per_segment):
            mask[i] = True

    # (2) Reachable-boundary states: the replay boundary (``num_prompt - 1``,
    # capped by ``get_computed_blocks``) and any shared-prefix junction, both
    # of which segments would otherwise skip under sparse retention. A Mamba
    # hit needs exactly the single state block ending on the boundary.
    for boundary_tokens in reachable_boundaries:
        aligned = boundary_tokens // alignment_tokens * alignment_tokens
        boundary_block = aligned // block_size - 1
        if start_block <= boundary_block < end_block:
            mask[boundary_block - start_block] = True

    return mask
