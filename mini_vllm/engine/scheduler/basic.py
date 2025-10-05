"""Basic scheduler with admission control and prefill/decode queues."""
from __future__ import annotations

from collections import deque
from typing import Deque, List

from .common import Request, SchedulerConfig


class BasicScheduler:
    def __init__(self, cfg: SchedulerConfig = SchedulerConfig()) -> None:
        self.cfg = cfg
        self.q_new: Deque[Request] = deque()
        self.q_decode: Deque[Request] = deque()
        self.admitted = 0

    # ------------------------------------------------------------------
    # Admission control
    # ------------------------------------------------------------------
    def admit(self, req: Request) -> bool:
        est = req.input_ids.numel() + req.max_new
        if self.admitted + est > self.cfg.token_budget:
            return False
        self.admitted += est
        self.to_prefill(req)
        return True

    # ------------------------------------------------------------------
    # Queue transitions
    # ------------------------------------------------------------------
    def to_prefill(self, req: Request) -> None:
        self.q_new.append(req)

    def to_decode(self, req: Request) -> None:
        self.q_decode.append(req)

    # ------------------------------------------------------------------
    # Batch selection
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def recycle(self, req: Request) -> None:
        if req.done:
            self.admitted -= req.input_ids.numel() + req.max_new
        else:
            self.to_decode(req)

    def is_empty(self) -> bool:
        return not (self.q_new or self.q_decode)

    def idle(self) -> bool:
        return self.is_empty()
