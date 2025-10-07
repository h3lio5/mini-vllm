"""Bucketed scheduler that groups sequences by length for efficient batching."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

from .common import Request


class BucketedScheduler:
    """Scheduler with sequence-length bucketing for efficient batching."""

    BUCKETS = [64, 128, 256, 512]
    MAX_BATCH_PER_BUCKET = {64: 16, 128: 12, 256: 8, 512: 4}

    def __init__(
        self,
        token_budget: int = 4096,
        max_decode_batch: int = 24,
    ) -> None:
        self.token_budget = token_budget
        self.max_decode_batch = max_decode_batch

        self.buckets: Dict[int, Deque[Request]] = {size: deque() for size in self.BUCKETS}
        self.q_decode: Deque[Request] = deque()
        self.admitted = 0

    # ------------------------------------------------------------------
    # Admission & queue utilities
    # ------------------------------------------------------------------
    def admit(self, req: Request) -> bool:
        est = req.input_ids.numel() + req.max_new
        if self.admitted + est > self.token_budget:
            return False
        self.admitted += est
        self.to_prefill(req)
        return True

    def _bucket_for_length(self, length: int) -> int:
        capped = max(1, min(self.BUCKETS[-1], length))
        for size in self.BUCKETS:
            if size >= capped:
                return size
        return self.BUCKETS[-1]

    def to_prefill(self, req: Request) -> None:
        remaining = req.input_ids.numel() - req.pos
        if remaining <= 0:
            self.to_decode(req)
            return
        bucket = self._bucket_for_length(remaining)
        self.buckets[bucket].append(req)

    def to_decode(self, req: Request) -> None:
        self.q_decode.append(req)

    # ------------------------------------------------------------------
    # Batch selection
    # ------------------------------------------------------------------
    def next_prefill_batch(self) -> List[Request]:
        for bucket_size in self.BUCKETS:
            queue = self.buckets[bucket_size]
            if not queue:
                continue

            batch: List[Request] = []
            max_batch = self.MAX_BATCH_PER_BUCKET[bucket_size]
            while queue and len(batch) < max_batch:
                req = queue.popleft()
                remaining = req.input_ids.numel() - req.pos
                if remaining <= 0:
                    self.to_decode(req)
                    continue
                req._take = min(bucket_size, remaining)
                batch.append(req)

            if batch:
                return batch

        return []

    def next_decode_batch(self) -> List[Request]:
        batch: List[Request] = []
        while self.q_decode and len(batch) < self.max_decode_batch:
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
        return all(not bucket for bucket in self.buckets.values()) and not self.q_decode

    def idle(self) -> bool:
        return self.is_empty()
