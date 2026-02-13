## 1. Core Concepts

### Autoregressive Generation
- LLMs generate tokens sequentially: P(token_t | token_1, ..., token_{t-1})
- Each token requires full model forward pass
- Output of step t becomes input for step t+1

---

### Two-Phase Inference

**Prefill Phase (Prompt Processing)**

- Process entire input prompt in parallel
- Compute KV cache for all input tokens
- Computationally intensive, **compute-bound**
- Time complexity: O(n²d) for n tokens, d dimensions

**Decode Phase (Token Generation)**

- Generate one token at a time
- Reuse cached KV from previous tokens
- **Memory-bound** operation (fetching weights/KV cache)
- Continues until EOS token or max length

---

### Key Metrics

```
Time to First Token (TTFT) = Prefill time
Time Per Output Token (TPOT) = Average decode time per token
Total Latency = TTFT + (num_output_tokens × TPOT)
```

---

---

## 2. Model Architecture Components

### Transformer Blocks
- Multi-head self-attention: O(n²d) complexity
- Feed-forward network: O(nd_ff) where d_ff ≈ 4d
- Layer normalization
- Residual connections

---

### KV Cache
- Stores key/value matrices from previous tokens
- Size per layer: 2 × batch_size × seq_len × num_heads × head_dim × 2 bytes (FP16)
- **Example**: LLaMA-2-7B with seq_len=2048, batch=1
  - Per layer: 2 × 1 × 2048 × 32 × 128 × 2 ≈ 33 MB
  - Total (32 layers): ~1 GB

---

---

## 3. Memory Requirements

```
Total Memory = Model Weights + KV Cache + Activations + Overhead

Model Weights = num_params × bytes_per_param
- FP32: 4 bytes, FP16: 2 bytes, INT8: 1 byte, INT4: 0.5 bytes

Activations = temporary tensors during forward pass
Overhead = CUDA context, fragmentation (~10-20%)
```

**Example: LLaMA-2-7B (FP16)**
- Weights: 7B × 2 = 14 GB
- KV cache (batch=1, seq=2048): ~1 GB
- Activations: ~0.5-1 GB
- **Total: ~16-17 GB**

---

---

## 4. Common Interview Questions

**Q: Why is prefill compute-bound and decode memory-bound?**

- Prefill: Process many tokens in parallel → high arithmetic intensity, GPU cores saturated
- Decode: Generate 1 token → fetch entire model weights from memory, low compute utilization

---

**Q: How does batch size affect inference?**
- Prefill: Higher batch increases compute, remains compute-bound
- Decode: Higher batch increases memory for KV cache, can become compute-bound with large batches
- Sweet spot: Balance between throughput and latency

---

**Q: What limits maximum sequence length?**

- KV cache memory grows linearly with sequence length
- Attention computation grows quadratically O(n²)
- GPU memory capacity is primary constraint

---

**Q: Calculate memory for Mistral-7B (FP16) with batch=4, seq=4096?**

```
Weights: 7B × 2 = 14 GB
KV cache: 2 × 4 × 4096 × 32 × 128 × 2 × 32 layers ≈ 8 GB
Total: ~22-24 GB
```

---

**Q: Why can't we parallelize token generation?**

- Each token depends on all previous tokens
- Autoregressive dependency prevents parallelization
- Speculative decoding attempts to work around this

---

---

## 5. Modern Optimizations (2024-2025)

- **Grouped Query Attention (GQA)**: Reduce KV cache by sharing KV heads
- **Multi-Query Attention (MQA)**: Single KV head for all queries
- **FlashAttention-3**: Fused attention kernel, 2x faster on H100
- **Paged Attention (vLLM)**: Non-contiguous KV cache storage
- **Continuous Batching**: Dynamic batch assembly for throughput

---

---

## 6. Key Takeaways

1. Inference has distinct prefill (parallel) and decode (sequential) phases
2. KV cache is crucial for avoiding recomputation but consumes significant memory
3. Memory bandwidth is often the bottleneck during decode
4. Model size, sequence length, and batch size determine memory requirements
5. Understanding the compute vs memory bound distinction is critical

---