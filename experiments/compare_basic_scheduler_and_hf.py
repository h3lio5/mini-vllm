"""
Compare scheduler.py throughput vs HF baseline processing requests sequentially.
"""
import random
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm.engine.scheduler import (
    DEVICE,
    MODEL,
    BasicScheduler,
    SchedulerConfig,
    decode_step,
    make_requests,
    prefill_step,
    refill_admissions,
)
from mini_vllm.utils.timer import Timer
from mini_vllm.utils.hf_utils import ensure_pad_token


def run_scheduler_batched(requests, cfg: SchedulerConfig):
    """Run with continuous batching scheduler."""
    timer = Timer()

    with timer.span("load_model"):
        tok = AutoTokenizer.from_pretrained(MODEL)
        ensure_pad_token(tok)
        model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE).eval()
        model.config.use_cache = True

    sched = BasicScheduler(cfg)
    pending = deque(requests)
    refill_admissions(pending, sched)

    with timer.span("inference"):
        with torch.no_grad():
            while True:
                did_work = False

                prefill = sched.next_prefill_batch()
                if prefill:
                    prefill_step(prefill, model, tok, sched)
                    did_work = True

                decode = sched.next_decode_batch()
                if decode:
                    decode_step(decode, model, tok, sched)
                    did_work = True

                refill_admissions(pending, sched)

                if not did_work and not pending and sched.idle():
                    break

    tokens_out = sum(len(req.out_ids) for req in requests)
    return timer, tokens_out


def run_hf_sequential(requests):
    """Run HF baseline - process each request sequentially."""
    timer = Timer()

    with timer.span("load_model"):
        tok = AutoTokenizer.from_pretrained(MODEL)
        ensure_pad_token(tok)
        model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE).eval()
        model.config.use_cache = True

    total_tokens = 0

    with timer.span("inference"):
        with torch.no_grad():
            for req in requests:
                input_ids = req.input_ids.unsqueeze(0).to(DEVICE)
                out = model(input_ids=input_ids, use_cache=True)
                past = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                generated = 1
                while generated < req.max_new:
                    out = model(input_ids=next_id, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated += 1

                    if next_id.item() == tok.eos_token_id:
                        break

                total_tokens += generated

    return timer, total_tokens


def main():
    random.seed(1337)
    torch.manual_seed(1337)

    tok = AutoTokenizer.from_pretrained(MODEL)
    ensure_pad_token(tok)

    n_requests = 50
    requests_scheduler = make_requests(tok, n=n_requests)

    random.seed(1337)
    requests_hf = make_requests(tok, n=n_requests)

    print(f"Running comparison with {n_requests} requests...\n")

    print("=" * 60)
    print("SCHEDULER (Continuous Batching)")
    print("=" * 60)
    cfg = SchedulerConfig()
    timer_sched, tokens_sched = run_scheduler_batched(requests_scheduler, cfg)
    print(timer_sched.report())
    inference_time_sched = timer_sched.spans[1].ms / 1000
    throughput_sched = tokens_sched / inference_time_sched
    print(f"Tokens generated: {tokens_sched}")
    print(f"Throughput: {throughput_sched:.1f} tok/s\n")

    print("=" * 60)
    print("HF BASELINE (Sequential Processing)")
    print("=" * 60)
    timer_hf, tokens_hf = run_hf_sequential(requests_hf)
    print(timer_hf.report())
    inference_time_hf = timer_hf.spans[1].ms / 1000
    throughput_hf = tokens_hf / inference_time_hf
    print(f"Tokens generated: {tokens_hf}")
    print(f"Throughput: {throughput_hf:.1f} tok/s\n")

    print("=" * 60)
    print("SPEEDUP ANALYSIS")
    print("=" * 60)
    speedup = throughput_sched / throughput_hf
    time_saved = ((inference_time_hf - inference_time_sched) / inference_time_hf) * 100
    print(f"Throughput speedup: {speedup:.2f}x")
    print(f"Time reduction: {time_saved:.1f}%")
    print(f"Scheduler: {inference_time_sched:.2f}s vs HF: {inference_time_hf:.2f}s")


if __name__ == "__main__":
    main()
