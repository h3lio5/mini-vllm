"""
Paged KV cache vs standard contiguous KV allocation.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

from mini_vllm.paged import KVShape, PagedBlockPool, PagedKVCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


class StandardKVCache:
    """Naïve contiguous KV cache that pre-allocates for max length."""

    def __init__(self, shape: KVShape, *, max_seq_len: int, max_batch: int, dtype: torch.dtype, device: str):
        layers, heads, head_dim = shape.as_tuple()
        tensor_shape = (max_batch, layers, heads, max_seq_len, head_dim)
        self.K = torch.zeros(tensor_shape, dtype=dtype, device=device)
        self.V = torch.zeros_like(self.K)

    def memory_bytes(self) -> int:
        return self.K.numel() * self.K.element_size() * 2


def bytes_to_mb(value: int) -> float:
    return value / (1024 ** 2)


def describe_sequences(sequences: Sequence[Tuple[str, int]]) -> None:
    print(f"Workload: {len(sequences)} sequences")
    print(f"Sequence lengths: {[length for _, length in sequences]}")
    print(f"Max sequence length: {max(length for _, length in sequences)}")


def simulate_sequences(cache: PagedKVCache, *, lengths: Iterable[int]) -> List:
    seqs = []
    shape = cache.kvshape
    dtype = cache.pool.dtype
    device = cache.pool.device

    for length in lengths:
        seq = cache.new_sequence()
        remaining = length
        while remaining > 0:
            take = min(cache.pool.block_tokens, remaining)
            k_proj = torch.zeros((take, shape.heads, shape.head_dim), dtype=dtype, device=device)
            v_proj = torch.zeros_like(k_proj)
            for layer in range(shape.layers):
                cache.append_layer(seq, layer, k_proj, v_proj)
            remaining -= take
        seqs.append(seq)
    return seqs


def tokens_in_refs(ref_lists: Iterable[Iterable]) -> int:
    total = 0
    for refs in ref_lists:
        for ref in refs:
            total += ref.used
    return total


def compare_memory() -> None:
    print("\n" + "=" * 70)
    print("PAGED KV CACHE - MEMORY COMPARISON")
    print("=" * 70)

    config = KVShape(layers=6, heads=8, head_dim=64)
    sequences = [
        ("short", 128),
        ("short", 150),
        ("medium", 256),
        ("medium", 280),
        ("long", 512),
        ("long", 490),
        ("very_long", 1024),
        ("very_long", 980),
    ]

    describe_sequences(sequences)

    max_seq_len = max(length for _, length in sequences)
    batch_size = len(sequences)

    standard = StandardKVCache(config, max_seq_len=max_seq_len, max_batch=batch_size, dtype=DTYPE, device=DEVICE)
    total_tokens = sum(length for _, length in sequences)
    element_size = torch.tensor([], dtype=DTYPE).element_size()

    used_bytes = 2 * config.layers * config.heads * total_tokens * config.head_dim * element_size

    print("\n" + "-" * 70)
    print("STANDARD CACHE")
    print("-" * 70)
    print(f"Allocated: {bytes_to_mb(standard.memory_bytes()):.2f} MB")
    print(f"Actually used: {bytes_to_mb(used_bytes):.2f} MB")

    block_tokens = 128
    blocks_needed = sum((length + block_tokens - 1) // block_tokens for _, length in sequences)
    blocks_per_layer = blocks_needed + 4

    pool = PagedBlockPool(
        config,
        blocks_per_layer=blocks_per_layer,
        block_tokens=block_tokens,
        dtype=DTYPE,
        device=DEVICE,
    )
    cache = PagedKVCache(pool)
    seq_objects = simulate_sequences(cache, lengths=[length for _, length in sequences])

    print("\n" + "-" * 70)
    print("PAGED CACHE")
    print("-" * 70)
    allocated_mb = bytes_to_mb(pool.memory_bytes())
    tokens_keys = sum(tokens_in_refs(seq.keys) for seq in seq_objects)
    tokens_values = sum(tokens_in_refs(seq.values) for seq in seq_objects)
    per_token_bytes = config.heads * config.head_dim * element_size
    paged_used_bytes = (tokens_keys + tokens_values) * per_token_bytes

    print(f"Allocated: {allocated_mb:.2f} MB")
    print(f"Blocks per layer: {blocks_per_layer} × {block_tokens} tokens")
    print(f"Actually used: {bytes_to_mb(paged_used_bytes):.2f} MB")

    savings_mb = bytes_to_mb(standard.memory_bytes() - pool.memory_bytes())
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Standard cache: {bytes_to_mb(standard.memory_bytes()):.2f} MB")
    print(f"Paged cache:    {allocated_mb:.2f} MB")
    print(f"Memory saved:   {savings_mb:.2f} MB")
    ratio = standard.memory_bytes() / pool.memory_bytes()
    print(f"Reduction:      {ratio:.2f}×")

    for seq in seq_objects:
        cache.release(seq)


if __name__ == "__main__":
    compare_memory()
