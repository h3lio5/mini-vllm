"""
Comparision between Hugging Face's model inference time with and without in-built KV cache.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mini_vllm.utils.timer import Timer

# Config
MODEL = "sshleifer/tiny-gpt2"
PROMPT = "The quick brown fox jumps over the lazy dog."
MAX_NEW_TOKENS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else None


def load_model_and_tokenizer(timer: Timer):
    with timer.span("load_model"):
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE)
        model.to(DEVICE).eval()
    return model, tok


def tokenize(prompt: str, tok, timer: Timer):
    with timer.span("tokenize"):
        return tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]


def argmax_next(logits):
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def run_no_cache():
    """Baseline: No KV cache"""
    print("\n" + "="*60)
    print("BASELINE: No KV Cache")
    print("="*60)

    t = Timer()
    model, tok = load_model_and_tokenizer(t)
    input_ids = tokenize(PROMPT, tok, t)

    model.config.use_cache = False
    tokens_generated = 0

    with torch.no_grad():
        with t.span("prefill(full forward)"):
            out = model(input_ids=input_ids)
            next_id = argmax_next(out.logits)
            generated = torch.cat([input_ids, next_id], dim=1)
            tokens_generated = 1

        with t.span("decode(no-cache loop)"):
            for _ in range(1, MAX_NEW_TOKENS):
                out = model(input_ids=generated)
                generated = torch.cat([generated, argmax_next(out.logits)], dim=1)
                tokens_generated += 1

    print(t.report())

    # Calculate throughput
    decode_time_s = [s for s in t.spans if s.name == "decode(no-cache loop)"][0].ms / 1000
    throughput = tokens_generated / decode_time_s
    print(f"Tokens generated: {tokens_generated}")
    print(f"Throughput: {throughput:.1f} tok/s")

    return decode_time_s * 1000, throughput  # Return in ms


def run_with_cache():
    """With KV cache"""
    print("\n" + "="*60)
    print("WITH KV CACHE")
    print("="*60)

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

    # Calculate throughput
    decode_time_s = [s for s in t.spans if s.name == "decode(cached)"][0].ms / 1000
    throughput = tokens / decode_time_s
    print(f"Tokens generated: {tokens}")
    print(f"Throughput: {throughput:.1f} tok/s")

    return decode_time_s * 1000, throughput  # Return in ms


if __name__ == "__main__":
    no_cache_time, no_cache_throughput = run_no_cache()
    cache_time, cache_throughput = run_with_cache()

    print("\n" + "="*60)
    print("SPEEDUP ANALYSIS")
    print("="*60)
    print(f"No cache decode:  {no_cache_time:8.2f} ms | {no_cache_throughput:.1f} tok/s")
    print(f"With cache decode: {cache_time:8.2f} ms | {cache_throughput:.1f} tok/s")
    print(f"Speedup:          {no_cache_time/cache_time:8.1f}×")