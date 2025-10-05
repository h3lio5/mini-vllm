from dataclasses import dataclass
import contextlib
import time
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------------
# Timing utilities
# -------------------------------
@dataclass
class Stat:
    name: str
    ms: float


class Timer:
    def __init__(self):
        self.spans: List[Stat] = []

    @contextlib.contextmanager
    def span(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.spans.append(Stat(name, (t1 - t0) * 1_000))

    def report(self) -> str:
        total = sum(s.ms for s in self.spans)
        lines = [f"Total: {total:.2f} ms"]
        lines += [f" {s.name:<24} {s.ms:8.2f} ms" for s in self.spans]
        return "\n".join(lines)


# -------------------------------
# Config
# -------------------------------
MODEL = "sshleifer/tiny-gpt2"
PROMPT = "The quick brown fox jumps over the lazy dog."
MAX_NEW_TOKENS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else None


def load_model_and_tokenizer(timer: Timer):
    with timer.span("load_model"):
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE)
        model.to(DEVICE).eval()
    return model, tok


def tokenize(prompt: str, tok, timer: Timer):
    with timer.span("tokenize"):
        return tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]


def argmax_next(logits):
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def run_no_cache():
    t = Timer()
    model, tok = load_model_and_tokenizer(t)
    input_ids = tokenize(PROMPT, tok, t)

    model.config.use_cache = False

    with torch.no_grad():
        with t.span("prefill(full forward)"):
            out = model(input_ids=input_ids)
            next_id = argmax_next(out.logits)
            generated = torch.cat([input_ids, next_id], dim=1)

        with t.span("decode(no-cache loop)"):
            for _ in range(1, MAX_NEW_TOKENS):
                out = model(input_ids=generated)
                generated = torch.cat([generated, argmax_next(out.logits)], dim=1)

    print(t.report())
    print("Generated length:", generated.shape[1])


def run_with_cache():
    t = Timer()
    model, tok = load_model_and_tokenizer(t)
    input_ids = tokenize(PROMPT, tok, t)

    model.config.use_cache = True

    with torch.no_grad():
        with t.span("prefill(cache warmup)"):
            out = model(input_ids=input_ids, use_cache=True)
            past = out.past_key_values
            ids = argmax_next(out.logits)

        tokens = 1
        with t.span("decode(cached)"):
            while tokens < MAX_NEW_TOKENS:
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                past, ids = out.past_key_values, argmax_next(out.logits)
                tokens += 1

    print(t.report())
    print("New tokens:", tokens)


if __name__ == "__main__":
    run_no_cache()
    run_with_cache()
