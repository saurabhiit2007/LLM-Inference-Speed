## 1. Core Innovation: PagedAttention

**Problem:** Traditional engines pre-allocate contiguous memory for max sequence length <br>
- 60-80% GPU memory wasted on over-reservation
- Internal fragmentation from unused allocated space

**Solution:** Paged memory management for KV cache <br>
- KV cache split into fixed-size blocks (pages)
- Non-contiguous physical memory mapped via block tables
- Reduces waste to <4%, enables 2-3x larger batch sizes

**Memory Formula:** <br>
```
KV Memory ≈ 2 × L × T × H × D_h × B
```

For Llama-3 8B (L=32, D=4096, FP16): ~0.5 MB per token <br>
- 2k tokens → ~1 GB
- 8k tokens → ~4 GB

---

---

## 2. Continuous Batching

**vs Static Batching:** Waits for entire batch to complete before accepting new requests

**vLLM Approach:** Iteration-level scheduling <br>
- New requests fill slots freed by completed sequences immediately
- Eliminates GPU idle time ("bubbles")
- Increases throughput by 20-30%

---

---

## 3. Prefill vs Decode Phases

| Phase | Processing | Bottleneck | vLLM Optimization |
|-------|-----------|------------|-------------------|
| Prefill | Parallel over tokens | Compute-bound | Chunked prefill |
| Decode | Sequential per token | Memory-bandwidth | PagedAttention |

**Chunked Prefill:** Breaks large prompts into chunks to prevent blocking decode operations

---

---

## 4. Modern Features (2025-2026)

### Speculative Decoding
- Small draft model generates k tokens
- Large target model verifies in single forward pass
- 2-3x latency reduction for heavy models

---

### Automatic Prefix Caching (APC)
- Shared KV blocks for common prefixes (system prompts, RAG contexts)
- Multiple requests reference same physical memory
- Critical for multi-turn chat and RAG applications

---

### Multi-LoRA Support
- Serve base model + hundreds of LoRA adapters simultaneously
- SGMV kernels enable batched computation across different adapters
- Ideal for multi-tenant SaaS deployments

---

---

## 5. Memory Pressure Handling

**Preemption Strategies:**
1. **Swap:** Move KV blocks to CPU memory (slower, preserves compute)
2. **Recompute:** Drop blocks and recalculate later (faster on modern GPUs)

Strategy selection based on GPU compute vs memory bandwidth ratio.

---

---

## 6. Interview Q&A

**Q: Why does PagedAttention improve throughput?** <br>
A: Eliminates memory fragmentation, allowing more concurrent requests to fit in GPU memory. With 60-80% waste reduced to <4%, effective batch size increases 2-3x.

---

**Q: When is prefill the bottleneck vs decode?** <br>
A: Prefill dominates for short outputs with long prompts (summarization). Decode dominates for long generations (creative writing). vLLM uses chunked prefill to balance both.

---

**Q: How does vLLM handle variable-length sequences in a batch?** <br>
A: Continuous batching removes completed sequences and adds new ones at iteration boundaries. Block tables allow each sequence to use non-contiguous memory independently.

---

**Q: Why use recompute over swap for preemption?** <br>
A: On H100/A100 GPUs with high compute, recomputing KV cache is faster than PCIe transfer to CPU. Swap preferred for older GPUs or when CPU memory is abundant.

---

**Q: How does APC differ from traditional caching?** <br>
A: Traditional caching stores entire request results. APC caches KV blocks at sub-request granularity, enabling partial reuse across different requests with shared prefixes.

---