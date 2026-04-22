# LLM Inference Speed

A technical reference for how large language models are served efficiently — covering the algorithms, hardware trade-offs, and frameworks that determine inference latency, throughput, and cost.

---

## Contents

### Fundamentals

| Topic | What It Covers |
|---|---|
| [Inference Basics](fundamentals/inference_basics.md) | Autoregressive generation, prefill vs decode phases, KV cache, memory sizing |
| [Bottleneck Analysis](fundamentals/bottleneck_analysis.md) | Roofline model, compute-bound vs memory-bound, profiling |
| [Latency vs Throughput](fundamentals/latency_vs_throughput.md) | TTFT, TPOT, throughput, batch size trade-offs |
| [Memory & Compute Trade-offs](fundamentals/memory_compute_tradeoffs.md) | GPU memory hierarchy, bandwidth limits, model size arithmetic |

### Attention Optimization

| Topic | What It Covers |
|---|---|
| [Flash Attention](attention_optimization/flash_attention.md) | Tiling, kernel fusion, online softmax, IO complexity |
| [Flash Attention 2](attention_optimization/flash_attention_2.md) | Improved parallelism, reduced synchronization, H100 gains |
| [KV Caching](attention_optimization/kv_caching.md) | Cache mechanics, memory cost, GQA/MQA variants |
| [Paged Attention](attention_optimization/paged_attention.md) | Block tables, non-contiguous KV storage, copy-on-write |
| [Prefix Caching](attention_optimization/prefix_caching.md) | KV-cache reuse across requests with shared prefixes |

### Decoding Strategies

| Topic | What It Covers |
|---|---|
| [Greedy Decoding](decoding_strategies/greedy_decoding.md) | Argmax selection, repetition, when it works |
| [Beam Search](decoding_strategies/beam_search.md) | Top-k beam tracking, length bias, diversity |
| [Sampling Methods](decoding_strategies/sampling_methods.md) | Temperature, top-k, top-p (nucleus), combinations |
| [Speculative Decoding](decoding_strategies/speculative_decoding.md) | Draft-then-verify, rejection sampling, 2–3× latency reduction |

### Batching Strategies

| Topic | What It Covers |
|---|---|
| [Batching Strategies](batching_strategies.md) | Static, dynamic, continuous batching, chunked prefill |

### Quantization

| Topic | What It Covers |
|---|---|
| [Quantization Basics](quantization/quantization_basics.md) | PTQ vs QAT, symmetric/asymmetric, per-tensor/per-channel |
| [INT8 Quantization](quantization/int8_quantization.md) | LLM.int8(), mixed-precision, hardware support |
| [INT4 Quantization](quantization/int4_quantization.md) | Group quantization, NF4, double quantization |
| [GPTQ](quantization/gptq.md) | Optimal brain quantization, lazy batch updates |
| [AWQ](quantization/awq.md) | Activation-aware weight quantization, salient channel protection |
| [SmoothQuant](quantization/smoothquant.md) | Activation outlier smoothing, migration strength |
| [GGUF / GGML](quantization/gguf_ggml.md) | CPU inference, K-quantization types, llama.cpp |
| [Quantization Trade-offs](quantization/quantization_tradeoffs.md) | Method selection decision tree, memory vs quality spectrum |

### Serving Frameworks

| Topic | What It Covers |
|---|---|
| [vLLM](serving_frameworks/vllm.md) | PagedAttention, continuous batching, prefix caching |
| [TensorRT-LLM](serving_frameworks/tensorrt_llm.md) | Kernel fusion, quantization, NVIDIA GPU optimisation |
| [Text Generation Inference](serving_frameworks/text_generation_inference.md) | Rust backend, token streaming, grammar-constrained generation |
| [DeepSpeed Inference](serving_frameworks/deepspeed_inference.md) | ZeRO-Inference, kernel optimisations, model parallelism |
| [Triton Inference Server](serving_frameworks/triton_inference_server.md) | Multi-backend, ensemble models, Kubernetes integration |
| [Framework Comparison](serving_frameworks/framework_comparison.md) | Decision tree, feature matrix, multi-GPU trade-offs |
| [Disaggregated Prefill-Decode](serving_frameworks/disaggregated_prefill_decode.md) | Separate GPU pools for prefill and decode; Mooncake, DistServe |

### Test-Time Compute Scaling

| Topic | What It Covers |
|---|---|
| [Compute-Optimal Inference](test_time_compute/compute_optimal_inference.md) | Power laws, parallel vs sequential scaling, o1/R1 |
| [Best-of-N Sampling](test_time_compute/best_of_n_sampling.md) | Algorithm, reward models, cost vs quality trade-offs |
| [ORMs & PRMs](test_time_compute/orm_prm.md) | Step-level vs outcome scoring, tree search, training data |

---

## Key Mental Models

```
Inference has two distinct phases:

  Prefill (prompt processing)          Decode (token generation)
  ─────────────────────────            ──────────────────────────
  • Processes all input tokens          • Generates 1 token per step
    in parallel                         • Reuses KV cache
  • Compute-bound                       • Memory-bandwidth-bound
  • Time ∝ prompt length²               • Time ∝ output length × model size
  • Optimised by FlashAttention         • Optimised by batching,
                                          quantization, speculative decoding
```

**The decode phase is almost always the bottleneck in production.** Reducing its memory-bandwidth pressure — via quantization, GQA, paged attention, and continuous batching — is where most inference engineering effort goes.

---

- [Interview Q&A](qa.md) — Curated questions and answers across all topics
- [References](references.md) — Papers, tools, and benchmarks cited in this knowledge base
