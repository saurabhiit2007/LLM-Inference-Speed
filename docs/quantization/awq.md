## 1. Paper

Lin et al., 2023 - "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"

---

---

## 2. Core Insight

**Not all weights are equal** - 1% of salient weights (channels) matter disproportionately for model quality.

---

---

## 3. Key Observation
Salient weight channels correlate with large activation magnitudes. Protect these during quantization.

---

---

## 4. Algorithm

### 1. Identify Salient Channels

```python
# Collect activation statistics
salient_scores = activation_magnitude.mean(dim=samples)
# Top 1% channels by activation magnitude
```

---

### 2. Per-Channel Scaling

```python
# Scale up salient weights BEFORE quantization
s = compute_optimal_scale(W, X)  # Based on activation distribution
W_scaled = W * s  # Per-channel scale
W_quantized = quantize(W_scaled)

# At inference: 
# output = (W_quantized / s) @ X = W_quantized @ (X * s)
# Move scaling to activations (cheap)
```

---

### 3. Search Optimal Scales

Minimize: `||W·X - Q(W·s)·(X/s)||`

Grid search over s ∈ [0.5, 1.5] per channel

---

---

## 5. Why It Works

- Increases effective resolution for important weights
- Shifts dynamic range to match activation distribution
- Quantization error on salient weights reduced by 2-4×

---

---

## 6. Specifications

**Calibration**: 128 samples
**Quantization Time**: ~10 minutes for 7B model (much faster than GPTQ)
**Group Size**: 128 typical
**Bits**: Optimized for 4-bit, works for 3-bit

---

---

## 7. Performance Comparison

| Method | LLaMA-7B (4-bit) | Quantization Time |
|--------|------------------|-------------------|
| RTN | 73.2 PPL | seconds |
| GPTQ | 68.4 PPL | 4 hours |
| AWQ | 68.1 PPL | 10 min |

---

---

## 8. Advantages over GPTQ

1. **Speed**: 20-30× faster quantization
2. **Simplicity**: No Hessian computation
3. **Hardware-friendly**: Simple per-channel scales

---

---

## 9. TinyChat Integration
AWQ includes custom CUDA kernels for efficient INT4 inference:

- Fused dequantization + GEMM
- 3-4× speedup over FP16 on consumer GPUs

---

---

## 10. Implementation

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("model_name")
model.quantize(
    calib_data="wikitext",
    w_bit=4,
    q_group_size=128,
    version="GEMM"  # Inference kernel
)
```

---
