"""Compare basic scheduler vs bucketed scheduler for batching efficiency."""
from __future__ import annotations

import random
import time
from collections import deque
from typing import Deque, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm.engine.scheduler import (
    DEVICE,
    MODEL,
    BasicScheduler,
    BucketedScheduler,
    Request,
    SchedulerConfig,
    clone_requests,
    decode_step,
    prefill_step,
    refill_admissions,
)
from mini_vllm.utils.hf_utils import ensure_pad_token

TOKEN_BUDGET = 4096
MAX_NEW = 64
REQUESTS = 80
SEED = 1337


# ---------------------------------------------------------------------------
# Request generation helpers
# ---------------------------------------------------------------------------
def generate_varied_requests(tokenizer) -> List[Request]:
    prompts = [
        "Explain attention.",
        "Haiku on GPUs.",
        "List cities.",
        "Why batching?",
        "Neural networks.",
    ]
    requests: List[Request] = []
    for rid in range(REQUESTS):
        prompt = random.choice(prompts) * random.randint(1, 40)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        requests.append(
            Request(
                rid=rid,
                input_ids=encoded["input_ids"][0],
                max_new=MAX_NEW,
            )
        )
    return requests


# ---------------------------------------------------------------------------
# Shared execution helpers
# ---------------------------------------------------------------------------
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE).eval()
    model.config.use_cache = True
    return tokenizer, model


def refill(pending: Deque[Request], scheduler) -> None:
    refill_admissions(pending, scheduler)


def run_prefill_decode_loop(scheduler, pending: Deque[Request], model, tokenizer) -> int:
    steps = 0
    with torch.no_grad():
        while not scheduler.is_empty() or pending:
            prefill_batch = scheduler.next_prefill_batch()
            if prefill_batch:
                prefill_step(prefill_batch, model, tokenizer, scheduler)

            decode_batch = scheduler.next_decode_batch()
            if decode_batch:
                decode_step(decode_batch, model, tokenizer, scheduler)
                steps += 1

            refill(pending, scheduler)
    return steps


def summarize(label: str, requests: List[Request], steps: int, elapsed: float) -> float:
    tokens_out = sum(len(req.out_ids) for req in requests)
    throughput = tokens_out / elapsed if elapsed > 0 else 0.0
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(f"Requests served: {len(requests)}")
    print(f"Tokens generated: {tokens_out}")
    print(f"Decode steps: {steps}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} tok/s")
    return throughput


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def run_basic_scheduler(template_requests: List[Request]) -> float:
    tokenizer, model = load_model_and_tokenizer()
    cfg = SchedulerConfig(
        token_budget=TOKEN_BUDGET,
        prefill_chunk=128,
        max_prefill_batch=8,
        max_decode_batch=16,
        max_new=MAX_NEW,
    )
    scheduler = BasicScheduler(cfg)
    requests = clone_requests(template_requests)
    pending: Deque[Request] = deque(requests)
    refill(pending, scheduler)

    start = time.perf_counter()
    steps = run_prefill_decode_loop(scheduler, pending, model, tokenizer)
    elapsed = time.perf_counter() - start

    return summarize("BASIC SCHEDULER (Iteration 3)", requests, steps, elapsed)


def run_bucketed_scheduler(template_requests: List[Request]) -> float:
    tokenizer, model = load_model_and_tokenizer()
    scheduler = BucketedScheduler(token_budget=TOKEN_BUDGET, max_decode_batch=24)
    requests = clone_requests(template_requests)
    pending: Deque[Request] = deque(requests)
    refill(pending, scheduler)

    start = time.perf_counter()
    steps = run_prefill_decode_loop(scheduler, pending, model, tokenizer)
    elapsed = time.perf_counter() - start

    return summarize("BUCKETED SCHEDULER (Iteration 5)", requests, steps, elapsed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ensure_pad_token(tokenizer)
    template_requests = generate_varied_requests(tokenizer)

    basic_tps = run_basic_scheduler(template_requests)
    bucketed_tps = run_bucketed_scheduler(template_requests)
    improvement = bucketed_tps / basic_tps if basic_tps else 0.0

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Basic scheduler:    {basic_tps:8.1f} tok/s")
    print(f"Bucketed scheduler: {bucketed_tps:8.1f} tok/s")
    print(f"Improvement:        {improvement:8.2f}×")
    print("\n" + "=" * 70)
    print("Key insight: Bucketing reduces padding waste and improves batching!")
    print("=" * 70)


if __name__ == "__main__":
    main()
