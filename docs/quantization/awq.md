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

---

## 11. Common Interview Questions

**Q1: How does AWQ differ from GPTQ philosophically?** <br>
A: GPTQ compensates errors across all weights. AWQ protects important weights from error in the first place. Prevention vs. compensation.

---

**Q2: Why is AWQ faster to quantize?** <br>
A: No iterative weight updates or Hessian computation. Just statistics collection + grid search for scales. Embarrassingly parallel.

---

**Q3: What's the "1% salient weights" finding?** <br>
A: 1% of weight channels (those with highest activation magnitude) contribute disproportionately. Protecting them preserves 90%+ of model quality.

---

**Q4: How are scales applied at inference?** <br>
A: Mathematically equivalent to scale weights (expensive) or scale activations (cheap). AWQ scales activations: `(W/s) @ X = (W) @ (X/s)`.

---

**Q5: Can AWQ work for INT8?** <br>
A: Yes, but less beneficial. INT8 already preserves most weights well. AWQ's advantage is strongest at 3-4 bits where bit budget is tight.

---

**Q6: What's the memory overhead of scales?** <br>
A: Per-channel FP16 scales: 0.1-0.2% overhead. Negligible compared to 8× weight reduction.

---

**Q7: AWQ vs. SmoothQuant?** <br>
A: SmoothQuant smooths activations for easier quantization. AWQ protects important weights. Can be combined: SmoothQuant for activation quantization, AWQ for weights.

---

**Q8: Why grid search for scales?** <br>
A: Closed-form solution doesn't exist for optimal scales. Grid search over reasonable range [0.5, 1.5] is fast and effective. Can use gradient-based for better results but slower.

---