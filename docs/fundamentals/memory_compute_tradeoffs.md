## 1. The Core Tradeoff

**Memory Savings → Compute Overhead (Usually)**

Techniques that reduce memory often require:

- Additional computation (quantization/dequantization)
- Recomputation instead of caching
- More complex kernels

---

---

## 2. Memory Bottlenecks in LLM Inference

### 1. Model Weights (Static)
- 70B model in FP16: 140 GB
- Must fit in GPU memory
- Loaded repeatedly during decode (memory bandwidth bound)

---

### 2. KV Cache (Dynamic)
- Grows with sequence length and batch size
- Often largest memory consumer in production
- **Formula**: `2 × B × S × L × H × D × bytes`
  - B=batch, S=seq_len, L=layers, H=heads, D=head_dim

---

### 3. Activations (Temporary)
- Intermediate tensors during forward pass
- Recomputed in inference (no backprop needed)
- ~5-10% of total memory

---

---

## 3. Quantization: Trading Precision for Memory

### Weight Quantization

**FP16 → INT8 (8-bit)**

- 2x memory reduction (2 bytes → 1 byte)
- Minimal accuracy loss (<1% typically)
- Faster on hardware with INT8 support (Tensor Cores)
- **Compute**: Dequantize to FP16 for matmul (overhead ~10%)

---

**FP16 → INT4 (4-bit)**

- 4x memory reduction
- Quality degradation possible (1-3% on benchmarks)
- Requires calibration data
- **Compute**: More dequant overhead (~20-30%)

---

**Techniques**:

```
Per-Tensor: Single scale for entire tensor
Per-Channel: Scale per output channel (better quality)
Group Quantization: Scale per 128 elements (GPTQ, AWQ)

GPTQ: Layer-wise quantization, minimizes error
AWQ: Activation-aware, protects important weights
```

---

### KV Cache Quantization

- KV cache in INT8 instead of FP16
- 2x memory savings → 2x larger batch or sequence length
- Quality loss typically <0.5%
- Growing adoption in production (2024+)

---

### Mixed Precision
- Keep critical layers in FP16 (first/last, attention)
- Quantize FFN layers to INT4
- Balance quality and memory

---

---

## 4. KV Cache Optimization

### Multi-Query Attention (MQA)
```
Standard: num_kv_heads = num_query_heads (e.g., 32)
MQA: num_kv_heads = 1

Memory reduction: 32x fewer KV parameters
Tradeoff: Slight quality degradation
Used in: Falcon, StarCoder
```

---

### Grouped Query Attention (GQA)
```
num_kv_heads < num_query_heads
Example: 8 KV heads, 32 query heads (4 queries per KV)

Memory reduction: 4x fewer KV parameters
Tradeoff: Minimal quality loss
Used in: LLaMA-2, Mistral, GPT-4 (rumored)
```

---

### Paged Attention (vLLM)
- KV cache in non-contiguous "pages" (like OS virtual memory)
- Eliminates fragmentation
- Enables ~2x higher batch size for same memory
- **Compute**: Slight overhead for page lookup

---

### Multi-Token Prediction
- Cache prefixes for common prompts
- Reduces redundant computation
- Memory: Store prompt KV cache (shared across requests)

---

---

## 4. Recomputation vs Caching

### Activation Checkpointing (Training)
- Not used in inference (no backprop)
- Mentioned for completeness

---

### Selective Recomputation
- Recompute cheap operations instead of storing
- Example: Recompute layer norm instead of caching
- Memory savings: ~10-20%
- Compute overhead: ~5-10%

---

---

## 5. Model Architecture Choices

### Width vs Depth
```
Wide: More hidden dimensions, fewer layers
- More memory for weights
- Less memory for KV cache (fewer layers)

Deep: More layers, smaller hidden dimensions
- Less memory for weights  
- More memory for KV cache (more layers)
```

---

### FFN Expansion Ratio
- Standard: `d_ff = 4 × d_model`
- Smaller ratio (2x or 3x): Less memory, potential quality loss
- MoE: Sparse activation, more parameters but same compute

---

---

## 6. Hardware-Specific Tradeoffs

### Memory Bandwidth vs Compute
```
A100: 1,935 GB/s bandwidth, 312 TFLOPS (FP16)
H100: 3,350 GB/s bandwidth, 989 TFLOPS (FP16)

Bandwidth-to-Compute Ratio:
A100: 6.2 GB/s per TFLOP
H100: 3.4 GB/s per TFLOP
```

**Implication**: H100 relatively more compute-bound, benefits more from quantization compute overhead

---

### Tensor Core Utilization
- FP16: Full tensor core speed
- INT8: 2x faster on Ampere/Hopper with DP4A
- INT4: 4x faster (requires specialized kernels)

**Tradeoff**: Quantization compute overhead offset by faster matmul

---

---

## 7. Memory-Compute Decision Matrix

| Technique | Memory Saved | Compute Overhead | Quality Impact |
|-----------|--------------|------------------|----------------|
| INT8 Quantization | 2x | +10% | <1% |
| INT4 Quantization | 4x | +30% | 1-3% |
| GQA (4:1) | 4x KV cache | Minimal | <0.5% |
| MQA | 32x KV cache | Minimal | 1-2% |
| KV Cache INT8 | 2x KV cache | +5% | <0.5% |
| FlashAttention | Minimal | -30% latency | None |

---

---

## 8. Common Interview Questions

**Q: You have a 70B model but only 40GB GPU memory. What do you do?**

```
Options:
1. INT4 quantization: 140GB → 35GB ✓
2. INT8 + model parallelism across 2 GPUs
3. Offload layers to CPU (slow, not recommended)
4. Use smaller model variant (13B/7B)
```

---

**Q: Explain the tradeoff in GQA (Grouped Query Attention)**

- Save memory: Fewer KV heads → smaller KV cache
- Minimal compute overhead: Attention computation slightly changes
- Quality: Negligible impact (<0.5% on benchmarks)
- Production: Widely adopted (Mistral, LLaMA-2)

---

**Q: Why is decode phase memory-bound?**

- Single token generation: Low arithmetic intensity
- Must fetch entire weight matrix from memory
- Memory bandwidth saturated, compute underutilized
- **Arithmetic Intensity**: FLOPs / Bytes Loaded ≈ 1-2 (very low)

---

**Q: When does quantization hurt performance?**

- Small batch size: Dequant overhead dominates
- Compute-bound workloads: Adding compute makes it worse
- Old hardware: No INT8 tensor core support
- **Generally**: Decode phase on modern GPUs (H100) benefits from quantization

---

**Q: Calculate KV cache size: LLaMA-2-70B, batch=16, seq=4096, FP16**

```
GQA with 8 KV heads (70B uses this)
2 × 16 × 4096 × 80_layers × 8_heads × 128_dim × 2_bytes
= 2 × 16 × 4096 × 80 × 8 × 128 × 2
= 1,073,741,824 bytes ≈ 1 GB per sample × 16 = 16 GB total

(If standard MHA with 64 heads: 128 GB - impractical!)
```

---

**Q: How does FlashAttention affect memory-compute tradeoff?**

- Reduces memory: Avoids materializing full attention matrix
- Reduces compute time: Fused kernel, better cache locality
- **Win-win**: Memory AND compute improvement
- No quality impact (mathematically equivalent)

---

---

## 9. Modern Techniques (2024-2025)

### AWQ (Activation-Aware Weight Quantization)
- Protect weights with high activation magnitude
- Better quality than naive INT4
- Used in production (Hugging Face TGI)

---

### SmoothQuant
- Migrate difficulty from weights to activations
- Enables better INT8 quantization
- Particularly for older models not trained for quantization

---

### FP8 (H100)
- Native FP8 support on Hopper
- 2x memory saving vs FP16
- Minimal quality loss
- **Compute**: Faster than FP16 (2x with tensor cores)

---

### QuIP# / AQLM
- Extreme quantization (2-3 bits)
- Lattice-based, better than naive 2-bit
- Research stage, not production yet

---

---

## 10. Practical Guidelines

1. **Start with INT8**: Minimal quality loss, 2x memory saving
2. **Use GQA architecture**: If designing new models
3. **Enable KV cache quantization**: Production-ready in vLLM
4. **FlashAttention is mandatory**: No downside
5. **INT4 for large models**: When GPU memory is the constraint
6. **Monitor quality**: Always benchmark on your task

---

---

## 11. Key Takeaways

1. Most memory optimizations have negligible compute cost (GQA, FlashAttention)
2. Quantization is a clear win on modern hardware (INT8 tensor cores)
3. KV cache often dominates memory in long-context scenarios
4. Decode phase is memory-bound: Reducing memory access helps latency
5. Hardware matters: H100 handles quantization overhead better than A100

---