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
