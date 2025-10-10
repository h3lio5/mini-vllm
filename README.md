# mini-vllm
A mini implementation of the popular vLLM llm inference engine.

This repository tracks my attempt to rebuild the ideas behind high-throughput LLM serving a piece at a time. Every step has an accompanying experiment under `experiments/` and a short write-up in `WORKLOG.md`.

## Implemented Optimisations

1. **KV caching baseline** – Compare a naïve forward pass with and without Hugging Face’s built-in KV cache to establish the starting point.
   - CPU: `python -m experiments.hf_baseline_cache`
   - GPU (Modal H100): `modal run modal_app.py --script experiments/hf_baseline_cache.py --gpu h100`
2. **Continuous batching scheduler** – A two-queue scheduler that overlaps prefilling and decoding work. See `experiments/compare_scheduler.py`.
3. **Bucketed scheduler** – Length-aware batching that reduces padding waste for mixed prompt lengths. See `experiments/compare_bucketed_scheduler.py`.
4. **Paged KV cache** – Block-based KV storage that recycles memory as sequences finish. Memory demo: `experiments/paged_kv_cache.py`.
5. **Weight-only INT8 quantization** – Swap selected linear layers for per-channel INT8 weight-only variants. Benchmark: `experiments/quantization.py`.

## In Progress

6. Triton kernels / fused ops
7. CUDA graphs for decode loops
8. Speculative decoding strategies
