"""Shared utilities for scheduler implementations."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from mini_vllm.utils.hf_utils import pad_to, split_past_sample, to_legacy_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "sshleifer/tiny-gpt2"

LayerKV = Tuple[torch.Tensor, torch.Tensor]
PastKV = Tuple[LayerKV, ...]

DEFAULT_PREFILL_CHUNK = 128
DEFAULT_PREFILL_BATCH = 8
DEFAULT_DECODE_BATCH = 16
DEFAULT_TOKEN_BUDGET = 4096
DEFAULT_MAX_NEW = 64


@dataclass(frozen=True)
class SchedulerConfig:
    prefill_chunk: int = DEFAULT_PREFILL_CHUNK
    max_prefill_batch: int = DEFAULT_PREFILL_BATCH
    max_decode_batch: int = DEFAULT_DECODE_BATCH
    token_budget: int = DEFAULT_TOKEN_BUDGET
    max_new: int = DEFAULT_MAX_NEW


@dataclass
class Request:
    """Tracks scheduling state for a single inference request."""

    rid: int
    input_ids: torch.Tensor
    max_new: int = DEFAULT_MAX_NEW
    pos: int = 0
    past: Optional[PastKV] = None
    out_ids: List[int] = field(default_factory=list)
    done: bool = False
    _take: int = 0  # number of prompt tokens to consume on next prefill


# ---------------------------------------------------------------------------
# Helper utilities shared across schedulers
# ---------------------------------------------------------------------------
def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - unlikely, defensive only
        return torch.device("cpu")


def prepare_prefill_inputs(batch: List[Request], tokenizer) -> torch.Tensor:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_take = max(req._take for req in batch)
    slices = [
        pad_to(req.input_ids[req.pos : req.pos + req._take], max_take, pad_id)
        for req in batch
    ]
    return torch.stack(slices, dim=0)


def pack_past(requests: List[Request]) -> PastKV:
    n_layers = len(requests[0].past)  # type: ignore[index]
    packed: List[LayerKV] = []
    for layer in range(n_layers):
        keys = torch.cat([req.past[layer][0] for req in requests], dim=0)  # type: ignore[index]
        values = torch.cat([req.past[layer][1] for req in requests], dim=0)
        packed.append((keys, values))
    return tuple(packed)


def past_len(req: Request) -> int:
    return int(req.past[0][0].size(-2))  # type: ignore[index]


def decode_seed(batch_size: int, tokenizer, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)


def group_by_past_length(batch: List[Request]) -> List[List[Request]]:
    buckets = defaultdict(list)
    for req in batch:
        buckets[past_len(req)].append(req)
    return list(buckets.values())


def prefill_step(batch: List[Request], model, tokenizer, scheduler) -> None:
    if not batch:
        return
    device = _model_device(model)
    inputs = prepare_prefill_inputs(batch, tokenizer).to(device)
    out = model(input_ids=inputs, use_cache=True)
    past = to_legacy_cache(out.past_key_values)
    for index, req in enumerate(batch):
        req.pos += req._take
        req.past = split_past_sample(past, index)
        target = scheduler.to_decode if req.pos >= req.input_ids.numel() else scheduler.to_prefill
        target(req)


def decode_step(batch: List[Request], model, tokenizer, scheduler) -> None:
    if not batch:
        return
    device = _model_device(model)
    for group in group_by_past_length(batch):
        legacy = pack_past(group)
        cache = DynamicCache.from_legacy_cache(legacy)
        inputs = decode_seed(len(group), tokenizer, device)
        out = model(input_ids=inputs, past_key_values=cache, use_cache=True)
        past = to_legacy_cache(out.past_key_values)
        next_ids = out.logits[:, -1, :].argmax(dim=-1)
        for index, req in enumerate(group):
            req.past = split_past_sample(past, index)
            token_id = int(next_ids[index].item())
            req.out_ids.append(token_id)
            if len(req.out_ids) >= req.max_new or token_id == tokenizer.eos_token_id:
                req.done = True
            scheduler.recycle(req)


def refill_admissions(pending: Deque[Request], scheduler) -> None:
    while pending and scheduler.admit(pending[0]):
        pending.popleft()


def make_requests(tokenizer, n: int = 32, *, max_new: int = DEFAULT_MAX_NEW) -> List[Request]:
    prompts = [
        "Explain attention in one line.",
        "Write a haiku about GPUs.",
        "List 3 cities in India.",
        "Why is batching important?",
        "Give me a short riddle.",
    ]
    requests: List[Request] = []
    for rid in range(n):
        prompt = random.choice(prompts) * random.randint(1, 5)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        requests.append(
            Request(
                rid=rid,
                input_ids=encoded["input_ids"][0],
                max_new=max_new,
            )
        )
    return requests


def clone_requests(template: Iterable[Request]) -> List[Request]:
    clones: List[Request] = []
    for req in template:
        clones.append(
            Request(
                rid=req.rid,
                input_ids=req.input_ids.clone(),
                max_new=req.max_new,
            )
        )
    return clones
