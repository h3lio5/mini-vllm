"""Paged KV cache built on top of the block pool."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

from .common import BlockRef, KVShape, SequencePage
from .pool import PagedBlockPool


class PagedKVCache:
    """High-level interface for paged KV storage."""

    def __init__(self, pool: PagedBlockPool) -> None:
        self.pool = pool
        self.kvshape = pool.kvshape

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------
    def new_sequence(self) -> SequencePage:
        return SequencePage.empty(self.kvshape)

    def release(self, seq: SequencePage) -> None:
        for per_layer in seq.keys:
            for ref in per_layer:
                self.pool.free(ref)
        for per_layer in seq.values:
            for ref in per_layer:
                self.pool.free(ref)
        seq.clear()

    # ------------------------------------------------------------------
    # Append & gather
    # ------------------------------------------------------------------
    def append_layer(self, seq: SequencePage, layer: int, k_proj: torch.Tensor, v_proj: torch.Tensor) -> None:
        """Append tokens for a single layer.

        Args:
            seq: Target sequence.
            layer: Layer index.
            k_proj: Tensor [T, H, D].
            v_proj: Tensor [T, H, D].
        """
        if k_proj.shape != v_proj.shape:
            raise ValueError("K and V projections must have the same shape")
        tokens, heads, head_dim = k_proj.shape
        shape = self.kvshape
        if heads != shape.heads or head_dim != shape.head_dim:
            raise ValueError("Projection shape mismatch with KVShape")

        self._append(seq.keys[layer], layer, k_proj, write_to_keys=True)
        self._append(seq.values[layer], layer, v_proj, write_to_keys=False)

    def gather_layer(self, seq: SequencePage, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = seq.token_count()
        shape = self.kvshape
        device = self.pool.device
        dtype = self.pool.dtype

        K_out = torch.empty((tokens, shape.heads, shape.head_dim), dtype=dtype, device=device)
        V_out = torch.empty_like(K_out)
        self._gather(seq.keys[layer], layer, K_out, read_from_keys=True)
        self._gather(seq.values[layer], layer, V_out, read_from_keys=False)
        return K_out, V_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append(self, slots: List[BlockRef], layer: int, src: torch.Tensor, write_to_keys: bool) -> None:
        pool_tensor = self.pool.K if write_to_keys else self.pool.V
        tokens = src.shape[0]
        offset = 0
        while offset < tokens:
            if not slots or slots[-1].used == self.pool.block_tokens:
                slots.append(self.pool.alloc(layer, is_key=write_to_keys))
            ref = slots[-1]
            free_tokens = self.pool.block_tokens - ref.used
            take = min(free_tokens, tokens - offset)
            chunk = src[offset : offset + take].permute(1, 0, 2)  # [T,H,D] -> [H,T,D]
            pool_tensor[layer, ref.block, :, ref.used : ref.used + take, :] = chunk
            ref.used += take
            offset += take

    def _gather(self, slots: Sequence[BlockRef], layer: int, dest: torch.Tensor, read_from_keys: bool) -> None:
        pool_tensor = self.pool.K if read_from_keys else self.pool.V
        offset = 0
        for ref in slots:
            chunk = pool_tensor[layer, ref.block, :, : ref.used, :]
            dest[offset : offset + ref.used] = chunk.permute(1, 0, 2)  # [H,T,D] -> [T,H,D]
            offset += ref.used
