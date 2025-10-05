from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from mini_vllm.utils.hf_utils import pad_to, split_past_sample, to_legacy_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "sshleifer/tiny-gpt2"

LayerKV = Tuple[torch.Tensor, torch.Tensor]
PastKV = Tuple[LayerKV, ...]


@dataclass(frozen=True)
class SchedulerConfig:
    prefill_chunk: int = 128
    max_prefill_batch: int = 8
    max_decode_batch: int = 16
    token_budget: int = 4096
    max_new: int = 64


CFG = SchedulerConfig()


@dataclass
class Request:
    rid: int
    input_ids: torch.Tensor
    max_new: int = CFG.max_new
    pos: int = 0
    past: Optional[PastKV] = None
    out_ids: List[int] = field(default_factory=list)
    done: bool = False
    _take: int = 0


class Scheduler:
    def __init__(self, cfg: SchedulerConfig = CFG):
        self.cfg = cfg
        self.q_new: Deque[Request] = deque()
        self.q_decode: Deque[Request] = deque()
        self.admitted = 0

    def admit(self, req: Request) -> bool:
        est = req.input_ids.numel() + req.max_new
        if self.admitted + est > self.cfg.token_budget:
            return False
        self.admitted += est
        self.to_prefill(req)
        return True

    def to_prefill(self, req: Request) -> None:
        self.q_new.append(req)

    def to_decode(self, req: Request) -> None:
        self.q_decode.append(req)

    def next_prefill_batch(self) -> List[Request]:
        batch: List[Request] = []
        while self.q_new and len(batch) < self.cfg.max_prefill_batch:
            req = self.q_new.popleft()
            remaining = req.input_ids.numel() - req.pos
            if remaining <= 0:
                self.to_decode(req)
                continue
            req._take = min(self.cfg.prefill_chunk, remaining)
            batch.append(req)
        return batch

    def next_decode_batch(self) -> List[Request]:
        batch: List[Request] = []
        while self.q_decode and len(batch) < self.cfg.max_decode_batch:
            batch.append(self.q_decode.popleft())
        return batch

    def recycle(self, req: Request) -> None:
        if req.done:
            self.admitted -= req.input_ids.numel() + req.max_new
        else:
            self.to_decode(req)

    def idle(self) -> bool:
        return not (self.q_new or self.q_decode)


def _prepare_prefill_inputs(batch: List[Request], tok) -> torch.Tensor:
    pad_id = tok.pad_token_id or tok.eos_token_id
    max_take = max(req._take for req in batch)
    slices = [
        pad_to(req.input_ids[req.pos : req.pos + req._take], max_take, pad_id)
        for req in batch
    ]
    return torch.stack(slices, dim=0).to(DEVICE)


def _pack_past(requests: List[Request]) -> PastKV:
    layers = len(requests[0].past)
    packed: List[LayerKV] = []
    for layer in range(layers):
        keys = torch.cat([req.past[layer][0] for req in requests], dim=0)
        values = torch.cat([req.past[layer][1] for req in requests], dim=0)
        packed.append((keys, values))
    return tuple(packed)


def _past_len(req: Request) -> int:
    return int(req.past[0][0].size(-2))


def prefill_step(batch: List[Request], model, tok, sched: Scheduler) -> None:
    inputs = _prepare_prefill_inputs(batch, tok)
    out = model(input_ids=inputs, use_cache=True)
    past = to_legacy_cache(out.past_key_values)
    for index, req in enumerate(batch):
        req.pos += req._take
        req.past = split_past_sample(past, index)
        target = (
            sched.to_decode if req.pos >= req.input_ids.numel() else sched.to_prefill
        )
        target(req)


def _decode_seed(batch_size: int, tok) -> torch.Tensor:
    return torch.full(
        (batch_size, 1), tok.eos_token_id, dtype=torch.long, device=DEVICE
    )


def decode_step(batch: List[Request], model, tok, sched: Scheduler) -> None:
    buckets = defaultdict(list)
    for req in batch:
        buckets[_past_len(req)].append(req)

    for group in buckets.values():
        legacy = _pack_past(group)
        cache = DynamicCache.from_legacy_cache(legacy)
        inputs = _decode_seed(len(group), tok)
        out = model(input_ids=inputs, past_key_values=cache, use_cache=True)
        past = to_legacy_cache(out.past_key_values)
        logits = out.logits[:, -1, :]
        next_ids = logits.argmax(dim=-1)
        for index, req in enumerate(group):
            req.past = split_past_sample(past, index)
            token_id = int(next_ids[index].item())
            req.out_ids.append(token_id)
            if len(req.out_ids) >= req.max_new or token_id == tok.eos_token_id:
                req.done = True
            sched.recycle(req)


def refill_admissions(pending: Deque[Request], sched: Scheduler) -> None:
    while pending and sched.admit(pending[0]):
        pending.popleft()


def make_requests(tok, n: int = 32) -> List[Request]:
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
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        requests.append(Request(rid=rid, input_ids=enc["input_ids"][0]))
    return requests
