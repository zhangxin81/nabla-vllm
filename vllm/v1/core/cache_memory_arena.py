# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One byte budget for independently addressed cache pages and state slots."""

import weakref
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from math import gcd
from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv, round_up

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


class FreeByteRanges:
    """Aligned allocations with exact byte accounting and coalescing on release."""

    def __init__(self, capacity: int, reserved: int = 0) -> None:
        if not 0 <= reserved <= capacity:
            raise ValueError("Reserved bytes must fit the arena")
        self.capacity = capacity
        self.reserved = reserved
        self.free_bytes = capacity - reserved
        self.ranges = [(reserved, capacity)] if reserved < capacity else []
        self.allocated: set[tuple[int, int]] = set()
        self._free_slot_counts: dict[int, int] = {}
        self.allocation_cursors: dict[int, int] = {}

    def copy(self, include_ownership: bool = True) -> "FreeByteRanges":
        clone = FreeByteRanges(self.capacity, self.reserved)
        clone.ranges = self.ranges.copy()
        clone.allocated = self.allocated.copy() if include_ownership else set()
        clone.free_bytes = self.free_bytes
        clone.allocation_cursors = self.allocation_cursors.copy()
        if include_ownership:
            clone._free_slot_counts = self._free_slot_counts.copy()
        return clone

    def contains(self, start: int, size: int) -> bool:
        i = bisect_right(self.ranges, (start, self.capacity + 1)) - 1
        return i >= 0 and self.ranges[i][0] <= start < start + size <= self.ranges[i][1]

    def reserve(self, start: int, size: int) -> None:
        if size <= 0 or not self.contains(start, size):
            raise ValueError("Cache allocation overlaps live or reserved bytes")
        i = bisect_right(self.ranges, (start, self.capacity + 1)) - 1
        lo, hi = self.ranges[i]
        parts = [(a, b) for a, b in ((lo, start), (start + size, hi)) if a < b]
        for stride in self._free_slot_counts:
            self._free_slot_counts[stride] += sum(
                self._slots(a, b, stride) for a, b in parts
            ) - self._slots(lo, hi, stride)
        self.ranges[i : i + 1] = parts
        self.allocated.add((start, size))
        self.free_bytes -= size

    def first_fit(self, size: int) -> int | None:
        if size <= 0:
            raise ValueError("Cache allocation size must be positive")
        for lo, hi in self.ranges:
            start = round_up(lo, size)
            if start + size <= hi:
                return start
        return None

    def release(self, start: int, size: int) -> None:
        if (start, size) not in self.allocated:
            raise ValueError("Cannot release an unowned cache allocation")
        self.allocated.remove((start, size))
        i = bisect_right(self.ranges, (start, self.capacity + 1))
        end = start + size
        if i and self.ranges[i - 1][1] == start:
            i -= 1
            lo, hi = self.ranges.pop(i)
            for stride in self._free_slot_counts:
                self._free_slot_counts[stride] -= self._slots(lo, hi, stride)
            start = lo
        if i < len(self.ranges) and self.ranges[i][0] == end:
            lo, hi = self.ranges.pop(i)
            for stride in self._free_slot_counts:
                self._free_slot_counts[stride] -= self._slots(lo, hi, stride)
            end = hi
        for stride in self._free_slot_counts:
            self._free_slot_counts[stride] += self._slots(start, end, stride)
        self.ranges.insert(i, (start, end))
        self.free_bytes += size

    def free_slots(self, stride: int) -> int:
        if stride not in self._free_slot_counts:
            self._free_slot_counts[stride] = sum(
                self._slots(lo, hi, stride) for lo, hi in self.ranges
            )
        return self._free_slot_counts[stride]

    @staticmethod
    def _slots(lo: int, hi: int, stride: int) -> int:
        return max(0, hi // stride - cdiv(lo, stride))


class CacheMemoryArena:
    """Coordinate pool-local IDs over stable, overlapping GPU tensor views.

    Cached slots with no readers may be evicted by an overlapping allocation
    from another pool. Live references, including copies and transfers, reserve
    their full backing range. No KV/state ratio is reserved in advance.
    Initial needs guide contiguous placement while every pool shares free bytes.
    """

    def __init__(self, capacity: int, strides: Sequence[int]) -> None:
        if not strides or any(s <= 0 for s in strides):
            raise ValueError("Cache pool strides must be positive")
        self.unit_bytes = gcd(*strides)
        self.memory = FreeByteRanges(capacity, max(strides))
        self.num_gpu_blocks = capacity // self.unit_bytes
        self.pools: dict[int, weakref.ReferenceType[BlockPool]] = {}
        self.allocation_plan: dict[BlockPool, list[KVCacheBlock]] | None = None

    def register_pool(self, pool: "BlockPool", stride: int) -> None:
        if stride in self.pools:
            raise ValueError("A byte stride already has a registered pool")
        if pool.num_gpu_blocks != self.memory.capacity // stride:
            raise ValueError("Pool capacity does not match its byte stride")
        self.pools[stride] = weakref.ref(pool)

    def _pool(self, stride: int) -> "BlockPool":
        pool = self.pools[stride]()
        if pool is None:
            raise RuntimeError("Cache pool no longer exists")
        return pool

    def acquire(self, block: "KVCacheBlock", stride: int) -> None:
        start = block.block_id * stride
        self.memory.reserve(start, stride)
        if block.block_hash is None:
            self.memory.allocation_cursors[stride] = start + stride
        # A write through one view invalidates every cached overlapping view.
        for other_stride in self.pools:
            if other_stride == stride:
                continue
            pool = self._pool(other_stride)
            first = start // other_stride
            last = min(cdiv(start + stride, other_stride), pool.num_gpu_blocks)
            cached_ids = pool.cached_block_hash_to_block.cached_block_ids
            watched_ids = pool._reuse_watchers
            affected: Iterable[KVCacheBlock]
            if len(cached_ids) + len(watched_ids) < last - first:
                affected_ids = {i for i in cached_ids if first <= i < last}
                affected_ids.update(i for i in watched_ids if first <= i < last)
                affected = (pool.blocks[i] for i in affected_ids)
            else:
                affected = iter(pool.blocks[first:last])
            for other in affected:
                assert other.ref_cnt == 0
                if other.block_hash is not None:
                    pool._maybe_evict_cached_block(other)
                if pool._reuse_watchers:
                    pool._notify_reuse((other,))

    def release(self, block: "KVCacheBlock", stride: int) -> None:
        self.memory.release(block.block_id * stride, stride)
        if not self.memory.allocated:
            self.memory.allocation_cursors.clear()

    def select_blocks(
        self, pool: "BlockPool", count: int, memory: FreeByteRanges | None = None
    ) -> list["KVCacheBlock"] | None:
        """Select a complete allocation without changing pool or cache contents."""
        if count < 0:
            raise ValueError("Cache allocation count must be non-negative")
        memory = self.memory.copy(include_ownership=False) if memory is None else memory
        stride = pool.block_stride_bytes
        selected: list[KVCacheBlock] = []
        current = pool.free_block_queue.fake_free_list_head.next_free_block
        tail = pool.free_block_queue.fake_free_list_tail
        while len(selected) < count:
            block = None
            cursor = memory.allocation_cursors.get(stride)
            if cursor is not None and memory.contains(cursor, stride):
                candidate = pool.blocks[cursor // stride]
                if (
                    candidate.block_hash is None
                    and candidate.block_id not in pool._reuse_watchers
                ):
                    block = candidate
            if block is None:
                # Preserve cached-slot LRU order and bound cross-family scans.
                for _ in range(32):
                    if current is None or current is tail:
                        break
                    candidate = current
                    current = current.next_free_block
                    if memory.contains(candidate.block_id * stride, stride):
                        block = candidate
                        break
            if block is None:
                start = memory.first_fit(stride)
                if start is None:
                    return None
                block = pool.blocks[start // stride]
            memory.reserve(block.block_id * stride, stride)
            memory.allocation_cursors[stride] = (block.block_id + 1) * stride
            selected.append(block)
        return selected

    def can_allocate(
        self,
        needs: Sequence[tuple["BlockPool", int]],
        hits: Iterable["KVCacheBlock"] = (),
        prepare: bool = False,
    ) -> bool:
        if prepare:
            self.allocation_plan = None
        if all(count == 0 for _, count in needs):
            hits = tuple(hits)
            if not hits:
                if prepare:
                    self.allocation_plan = {}
                return True
        memory = self.memory.copy(include_ownership=False)
        seen: set[int] = set()
        for block in hits:
            if block.is_null or block.ref_cnt or id(block) in seen:
                continue
            seen.add(id(block))
            assert block.pool is not None and (pool := block.pool()) is not None
            memory.reserve(
                block.block_id * pool.block_stride_bytes, pool.block_stride_bytes
            )
        if not memory.allocation_cursors:
            weights: dict[int, int] = {}
            for pool, count in needs:
                if count > 0:
                    stride = pool.block_stride_bytes
                    weights[stride] = weights.get(stride, 0) + count * stride
            if len(weights) > 1:
                total = sum(weights.values())
                cumulative = 0
                for stride in sorted(weights, reverse=True):
                    memory.allocation_cursors[stride] = round_up(
                        memory.reserved
                        + (memory.capacity - memory.reserved) * cumulative // total,
                        stride,
                    )
                    cumulative += weights[stride]
        plan: dict[BlockPool, list[KVCacheBlock]] = {}
        # Reserve large state holes before smaller token pages can split them.
        for pool, count in sorted(
            needs, key=lambda need: need[0].block_stride_bytes, reverse=True
        ):
            selected = self.select_blocks(pool, count, memory)
            if selected is None:
                return False
            plan.setdefault(pool, []).extend(selected)
        if prepare:
            self.allocation_plan = plan
        return True

    def planned_blocks(self, pool: "BlockPool", count: int):
        if self.allocation_plan is None:
            return self.select_blocks(pool, count)
        planned = self.allocation_plan.get(pool, [])
        if len(planned) < count:
            raise RuntimeError("Cache allocation exceeded its admission reservation")
        selected = planned[:count]
        del planned[:count]
        return selected

    def get_num_free_blocks(self) -> int:
        return self.memory.free_bytes // self.unit_bytes

    def get_usage(self) -> float:
        usable = self.memory.capacity - self.memory.reserved
        return 1 - self.memory.free_bytes / usable if usable else 0.0

    @property
    def null_block(self):
        return self._pool(next(iter(self.pools))).null_block

    @property
    def num_hash_lookahead_tokens(self) -> int:
        return self._pool(next(iter(self.pools))).num_hash_lookahead_tokens

    def get_cached_block(self, block_hash, group_ids):
        return self._pool(next(iter(self.pools))).get_cached_block(
            block_hash, group_ids
        )

    def reset_prefix_cache(self) -> bool:
        pools = [self._pool(stride) for stride in self.pools]
        if self.memory.allocated:
            return False
        return all(pool.reset_prefix_cache() for pool in pools)

    def free_blocks(self, blocks: Iterable["KVCacheBlock"]) -> None:
        self._pool(next(iter(self.pools))).free_blocks(blocks)

    def evict_blocks(self, block_ids: set[int]) -> None:
        for stride in self.pools:
            pool = self._pool(stride)
            pool.evict_blocks({i for i in block_ids if i < pool.num_gpu_blocks})

    def take_events(self):
        return [
            event for stride in self.pools for event in self._pool(stride).take_events()
        ]

    def byte_ranges(self, pool: "BlockPool", block_ids: Iterable[int]):
        stride = pool.block_stride_bytes
        return [(block_id * stride, stride) for block_id in block_ids]
