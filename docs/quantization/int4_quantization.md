## 1. Overview

4-bit quantization (16 discrete values) achieves 8× memory reduction from FP32. Requires careful techniques to maintain quality.

---

---

## 2. Key Challenge

Limited range (16 values) makes naive quantization lossy. Need sophisticated methods: GPTQ, AWQ, or group quantization.

---

---

## 3. Group Quantization

**Concept**: Different scales for weight groups instead of entire layer

```python
# Group size typically 32-128
for group in split_weights(W, group_size=128):
    scale = max(abs(group)) / 7  # 4-bit signed: -8 to 7
    group_int4 = round(group / scale).clip(-8, 7)
```

**Tradeoff**: Better accuracy vs. more scales to store (usually 1-2% overhead)

---

---

## NormalFloat (NF4) - QLoRA

**Innovation**: Non-uniform quantization matching normal distribution

```
Standard INT4: [-8, -7, ..., 0, ..., 7]
NF4: [-1.0, -0.6962, -0.5251, -0.3949, ...]
```

**Why it works**: Pre-trained weights follow ~N(0, σ), NF4 bins optimally quantize normal distribution

**Usage**: QLoRA for parameter-efficient fine-tuning

---

---

## Double Quantization

Quantize the quantization scales themselves (QLoRA technique)

- FP16 scales → INT8 scales
- Saves additional 0.4 bits per parameter
- Minimal accuracy impact

---

---

## Inference Kernels

**Challenge**: No native INT4 arithmetic on most hardware

**Solution**: Pack two INT4 values per byte, unpack during compute
```
byte = (val1 << 4) | val2  # Pack
val1 = (byte >> 4) & 0xF   # Unpack
```

---

---

## Performance

- **Memory**: 8× reduction (2GB for 7B model)
- **Speed**: 1.5-2× faster than INT8 (memory-bound scenarios)
- **Accuracy drop**: 3-7% with naive methods, <2% with GPTQ/AWQ

---
