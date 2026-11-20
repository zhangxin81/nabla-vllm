# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix lookup shared by independently allocated cache resources."""

from vllm.utils.gc_utils import gc_untrack
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock


class PrefixCacheIndex:
    """Index prefix hashes without using a pool's local IDs as global IDs.

    Physical pools keep their own IDs and free lists. The index assigns an
    internal offset to each pool; these IDs never reach a worker block table.
    """

    def __init__(self) -> None:
        self.blocks: list[KVCacheBlock] = []
        self.num_pools = 0
        self.cache: dict[BlockHashWithGroupId, int] = gc_untrack({})
        self.duplicate_ids: dict[BlockHashWithGroupId, dict[int, None]] = {}

    def register_pool(self, blocks: list[KVCacheBlock]) -> int:
        if any(block.block_id != i for i, block in enumerate(blocks)):
            raise ValueError("Pool block IDs must be contiguous and start at zero")
        offset = len(self.blocks)
        self.blocks.extend(blocks)
        self.num_pools += 1
        return offset

    def pop(self, key: BlockHashWithGroupId, index_id: int) -> KVCacheBlock | None:
        duplicates = self.duplicate_ids.get(key)
        if duplicates is None:
            if self.cache.get(key) != index_id:
                return None
            del self.cache[key]
        else:
            if duplicates.pop(index_id, False) is False:
                return None
            self.cache[key] = next(iter(duplicates))
            if len(duplicates) == 1:
                del self.duplicate_ids[key]
        return self.blocks[index_id]

    def clear_pool(self, offset: int, count: int) -> None:
        if self.num_pools == 1:
            self.cache.clear()
            self.duplicate_ids.clear()
            return
        end = offset + count
        for key, first_id in list(self.cache.items()):
            ids = self.duplicate_ids.get(key)
            for index_id in tuple(ids) if ids is not None else (first_id,):
                if offset <= index_id < end:
                    self.pop(key, index_id)

    def __len__(self) -> int:
        return len(self.cache)
