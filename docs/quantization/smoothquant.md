## 1. Paper

Xiao et al., 2022 - "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"

---

---

## 2. Problem Statement

**Activation outliers** make INT8 quantization difficult. Weights quantize well, activations don't.

### Observation

- Weight range: typically within [-3σ, 3σ]
- Activation range: 10-100× larger due to systematic outliers in specific channels

---

---

## 3. Core Idea: Smoothing

**Migrate difficulty from activations to weights** via mathematically equivalent transformation.

### Key Transformation

```
Y = XW = (X / s) · (W · s)
```

Where `s` is per-channel smoothing factor.

- Divide activations by s → reduces outliers
- Multiply weights by s → increases weight range (but weights easier to quantize)

---

---

## 4. Algorithm

### 1. Identify Outlier Channels
```python
# Collect calibration statistics
alpha_x = max(|X|, dim=tokens)  # Per-channel activation range
alpha_w = max(|W|, dim=input_dim)  # Per-channel weight range
```

---

### 2. Compute Smoothing Scales
```python
# Migration strength α ∈ [0, 1]
# α=0: no smoothing, α=1: full migration
s = alpha_x^α / alpha_w^(1-α)
```

---

### 3. Apply Smoothing
```python
# Offline transformation
W_smooth = W * s  # Fold into weights
# At runtime: X_smooth = X / s
```

---

---

## 5. Migration Strength α

**α = 0.5**: Balanced migration (default) <br>

- Geometric mean of activation and weight ranges
- Empirically optimal for most models

**α = 0.75**: More aggressive activation smoothing <br>

- Better for models with severe outliers (OPT)

---

---

## 6. Per-Token vs. Per-Tensor

**Per-Tensor Dynamic**: Single scale per activation tensor (fast, less accurate)
**Per-Token Dynamic**: Scale per token in sequence (better accuracy, slower)

SmoothQuant enables **per-tensor** quantization by smoothing outliers beforehand.

---

---

## 7. Performance

| Model | W8A8 Accuracy | vs FP16 |
|-------|---------------|---------|
| OPT-175B | 66.7% | -0.1% |
| BLOOM-176B | 68.4% | -0.3% |
| LLaMA-65B | 69.2% | -0.2% |

**Speedup**: 1.5-2× on A100 (INT8 Tensor Cores)

---

---

## 8. Integration with Other Methods

**SmoothQuant + AWQ**: <br>

- SmoothQuant for activation INT8
- AWQ for weight INT4
- Hybrid W4A8 quantization

**SmoothQuant + LLM.int8()**: <br>

- SmoothQuant pre-processing
- LLM.int8() for outlier handling
- Complementary techniques

---

---

## 9. Implementation

```python
from smoothquant.smooth import smooth_lm

# Apply smoothing transformation
model = smooth_lm(
    model,
    calibration_data,
    alpha=0.5  # Migration strength
)

# Then quantize with standard tools
quantized = quantize_model(model, w_bit=8, a_bit=8)
```

---
