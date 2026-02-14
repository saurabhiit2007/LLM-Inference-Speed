## 1. Overview

Maps FP16/FP32 values to 8-bit integers (256 discrete values). Standard for production LLM deployment.

---

---

## 2. Quantization Process

### Weight Quantization

```python
# Per-channel quantization
scale = max(abs(W)) / 127
W_int8 = round(W / scale).clip(-128, 127)
```

---

### Activation Quantization
```python
# Calibration phase (100-1000 samples)
min_val, max_val = collect_statistics(calibration_data)
scale = (max_val - min_val) / 255
zero_point = -round(min_val / scale)
```

---

---

## 3. LLM.int8() (Dettmers et al., 2022)

**Key Innovation**: Mixed-precision decomposition for outliers

**Process**:

1. Detect outlier features (>6σ threshold)
2. Separate matrix multiplication: FP16 for outliers, INT8 for rest
3. Typically, <0.1% outlier features, but they're critical

**Memory**: 2× reduction with minimal accuracy loss

---

---

## SmoothQuant Bridge

Often combined with SmoothQuant for activation smoothing before INT8 conversion.

---

---

## Hardware Support

- **NVIDIA Tensor Cores**: INT8 GEMM operations
- **Intel VNNI**: Vector Neural Network Instructions
- **ARM**: INT8 GEMM on modern CPUs

**Speedup**: 2-4× on modern hardware

---

---

## Common Interview Questions

**Q1: Why is INT8 considered the "sweet spot"?**
A: Best balance of compression (4× from FP32), hardware support, and accuracy preservation. INT4 needs more careful handling.

---

**Q2: What's the bottleneck in INT8 inference?**
A: Dequantization overhead and memory bandwidth. For small batches, compute isn't fully saturated.

---

**Q3: How does LLM.int8() handle outliers?**
A: Uses vector-wise quantization to detect outliers (values >6σ), processes them in FP16 while using INT8 for the remaining 99.9% of values.

---

**Q4: Can you quantize all layers to INT8?**
A: No. Embedding layers, layer norms, and sometimes first/last layers stay in FP16 for stability.

---

**Q5: What's absmax quantization?**
A: Symmetric quantization using absolute maximum: `scale = max(|W|) / 127`. Simple but can waste range if distribution is skewed.

---

**Q6: Calibration dataset size?**
A: 100-1000 samples from training distribution. More doesn't always help; diversity matters more than quantity.

---