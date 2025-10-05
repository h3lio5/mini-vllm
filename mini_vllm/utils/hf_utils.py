"""Utilities to deal with Hugging Face LLM Models."""
from typing import Tuple
import torch

LayerKV = Tuple[torch.Tensor, torch.Tensor]
PastKV = Tuple[LayerKV, ...]

def ensure_pad_token(tok) -> None:
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token


def to_legacy_cache(cache) -> PastKV:
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def split_past_sample(past: PastKV, index: int) -> PastKV:
    return tuple((k[index : index + 1].contiguous(), v[index : index + 1].contiguous()) for k, v in past)

def pad_to(tensor: torch.Tensor, length: int, pad_id: int) -> torch.Tensor:
    if tensor.numel() >= length:
        return tensor
    pad = torch.full((length - tensor.numel(),), pad_id, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)