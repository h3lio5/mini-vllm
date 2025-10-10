"""Global tensor pool for paged KV cache."""
from __future__ import annotations

from typing import List

import torch

from .common import BlockRef, KVShape, bytes_for_tensor


class PagedBlockPool:
    """Owns the dense storage used by all sequences."""

    def __init__(
        self,
        kvshape: KVShape,
        *,
        blocks_per_layer: int,
        block_tokens: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ) -> None:
        layers, heads, head_dim = kvshape.as_tuple()
        self.kvshape = kvshape
        self.block_tokens = block_tokens

        shape = (layers, blocks_per_layer, heads, block_tokens, head_dim)
        self.K = torch.zeros(shape, dtype=dtype, device=device)
        self.V = torch.zeros_like(self.K)

        self._free_keys: List[List[int]] = [list(range(blocks_per_layer)) for _ in range(layers)]
        self._free_values: List[List[int]] = [list(range(blocks_per_layer)) for _ in range(layers)]

    # ------------------------------------------------------------------
    # Allocation helpers
    # ------------------------------------------------------------------
    def alloc(self, layer: int, *, is_key: bool) -> BlockRef:
        slots = self._free_keys[layer] if is_key else self._free_values[layer]
        if not slots:
            raise RuntimeError(f"Out of KV blocks at layer {layer}")
        return BlockRef(layer=layer, block=slots.pop(), is_key=is_key, used=0)

    def free(self, ref: BlockRef) -> None:
        target = self._free_keys if ref.is_key else self._free_values
        target[ref.layer].append(ref.block)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.K.device

    @property
    def dtype(self) -> torch.dtype:
        return self.K.dtype

    def memory_bytes(self) -> int:
        return bytes_for_tensor(self.K) + bytes_for_tensor(self.V)
