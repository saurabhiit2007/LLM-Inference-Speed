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

## 5. Recomputation vs Caching

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

## 6. Model Architecture Choices

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

## 7. Hardware-Specific Tradeoffs

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

## 8. Memory-Compute Decision Matrix

| Technique | Memory Saved | Compute Overhead | Quality Impact |
|-----------|--------------|------------------|----------------|
| INT8 Quantization | 2x | +10% | <1% |
| INT4 Quantization | 4x | +30% | 1-3% |
| GQA (4:1) | 4x KV cache | Minimal | <0.5% |
| MQA | 32x KV cache | Minimal | 1-2% |
| KV Cache INT8 | 2x KV cache | +5% | <0.5% |
| FlashAttention | Minimal | -30% latency | None |

---
