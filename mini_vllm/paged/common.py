"""Shared data structures for paged KV cache."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import torch


@dataclass(frozen=True)
class KVShape:
    """Model-aware shape information for KV tensors."""

    layers: int
    heads: int
    head_dim: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return self.layers, self.heads, self.head_dim


@dataclass
class BlockRef:
    """Pointer to a block slice in the global pool."""

    layer: int
    block: int
    is_key: bool
    used: int = 0


@dataclass
class SequencePage:
    """Per-sequence KV cache expressed as block references."""

    kvshape: KVShape
    keys: List[List[BlockRef]] = field(default_factory=list)
    values: List[List[BlockRef]] = field(default_factory=list)

    @classmethod
    def empty(cls, kvshape: KVShape) -> "SequencePage":
        layers, _, _ = kvshape.as_tuple()
        return cls(
            kvshape=kvshape,
            keys=[[] for _ in range(layers)],
            values=[[] for _ in range(layers)],
        )

    def token_count(self) -> int:
        """Number of tokens materialised for the sequence."""
        if not self.keys:
            return 0
        if not self.keys[0]:
            return 0
        return sum(ref.used for ref in self.keys[0])

    def clear(self) -> None:
        for per_layer in self.keys:
            per_layer.clear()
        for per_layer in self.values:
            per_layer.clear()


def bytes_for_tensor(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()
