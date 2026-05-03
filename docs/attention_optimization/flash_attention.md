## 1. Overview

FlashAttention is a fast and memory-efficient attention algorithm that computes **exact** attention without materializing the full $N \times N$ attention matrix. It's especially critical for long sequences (4k+ tokens) in modern LLMs.

**Key insight:** The bottleneck in attention isn't compute - it's **memory bandwidth** (moving data between GPU memory hierarchies).

---

---

## 2. Why Standard Attention is Slow

Standard attention formula:
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

---
### Problem 1: Quadratic Memory Growth

For sequence length $N = 16{,}384$ in FP16:

- Attention matrix: $N^2 = 268M$ elements
- Memory: $268M \times 2$ bytes $≈ 512$ MB (per layer, per head!)

---

### Problem 2: Excessive Memory Traffic

Standard attention performs multiple memory-heavy steps:

1. Compute $QK^T$ → write to global memory
2. Read $QK^T$ → apply softmax → write back
3. Read softmax output → compute with $V$ → write output

Result: GPUs become **memory-bound**, not compute-bound.

---

### Problem 3: Numerical Instability in FP16

- Large values in $QK^T$ cause overflow in $e^x$
- Small values underflow to zero
- Standard attention often requires FP32, increasing memory usage

---

---

## 3. How FlashAttention Works

FlashAttention uses three key techniques:

### 3.1 Tiling

Split Q, K, V into small **tiles** that fit in GPU shared memory (SRAM).

**Example:**

- Sequence length: $N = 16{,}384$
- Tile size: $B = 128$
- Memory per tile: $128 \times 128 = 16{,}384$ elements (vs. $268M$ for full matrix)

```python
# Conceptual tiling
for q_tile in Q_tiles:
    for k_tile, v_tile in zip(K_tiles, V_tiles):
        partial_scores = q_tile @ k_tile.T
        # accumulate incrementally
```

---

### 3.2 Kernel Fusion

Fuse all operations into a single kernel to keep intermediate results in fast shared memory:

1. Matrix multiplication ($Q \cdot K^T$)
2. Scaling ($1/\sqrt{d}$)
3. Softmax
4. Weighted sum with $V$

Standard attention writes/reads from global memory between each step. FlashAttention does everything in one pass.

---

### 3.3 Online Softmax

Compute softmax incrementally across tiles without storing the full attention matrix.

**Numerically stable approach:**

1. Maintain **running maximum** $m$ across tiles
   - Compute: $e^{x_i - m}$ (prevents overflow)
2. Maintain **running sum** of exponentials
3. Accumulate weighted output incrementally

**Example with 2 tiles:**

Tile 1: `[0.1, 0.5, 0.3]`, Tile 2: `[0.2, 0.4, 0.1]`

Processing:

1. **Tile 1:** $m = 0.5$, shifted exps: $[e^{-0.4}, e^{0}, e^{-0.2}]$, running sum $s_1$
2. **Tile 2:** update $m$, reweight previous results, add new exps, update sum $s_2$
3. **Final:** divide accumulated output by $s_2$

Result: **Exact same output** as standard attention, but in FP16/BF16 without overflow.

---

---

## 4. Performance Impact

### Memory Complexity

- Standard: $O(N^2)$
- FlashAttention: $O(N \cdot B)$ where $B$ is tile size

### Speedup

- 2–4x faster for long sequences on modern GPUs
- Enables 2–4x longer sequences or larger batch sizes

### Usage
```python
from flash_attn import flash_attn_func

# q, k, v shape: (batch, seq_len, num_heads, head_dim)
output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
```

### When FlashAttention Helps Most
✅ Long sequences (2k+ tokens)  
✅ FP16/BF16 precision  
✅ Modern NVIDIA GPUs with fast shared memory  

❌ Very short sequences  
❌ CPU-based inference  
❌ Custom attention patterns not supported by the kernels  

---
