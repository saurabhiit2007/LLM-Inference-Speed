## 1. Overview

Quantization reduces the numerical precision of model weights (and optionally activations) from FP32/FP16 to lower-bit integers (INT8, INT4). This shrinks model memory footprint, reduces memory bandwidth pressure during decode, and enables faster integer arithmetic on hardware that supports it.

**Core formula:**

```
Q(x) = round(x / S) - Z
```

Where S = scale factor, Z = zero-point. Dequantization: `x ≈ S × (Q(x) + Z)`.

---

## 2. PTQ vs QAT

### Post-Training Quantization (PTQ)

Applied to an already-trained model — no retraining. Requires a small calibration dataset (hundreds of samples) to compute scales.

- **Static PTQ:** Scales computed offline from calibration data. Deterministic at inference time.
- **Dynamic PTQ:** Scales computed per-tensor at runtime. Slower but handles varied input distributions better.

### Quantization-Aware Training (QAT)

Simulates quantization noise during training using fake quantization nodes (straight-through estimator for gradients). Better final accuracy than PTQ, especially at 4-bit or below, but requires full training compute.

**In practice for LLMs:** PTQ dominates because retraining 7B–70B models is expensive.

---

## 3. Quantization Schemes

### Symmetric vs Asymmetric

| | Symmetric | Asymmetric |
|---|---|---|
| Zero-point | 0 | Non-zero |
| INT8 range | [-127, 127] | [0, 255] (UINT8) |
| Best for | Weights (symmetric distribution) | Activations (often non-negative) |

### Per-Tensor vs Per-Channel vs Per-Group

- **Per-tensor:** One scale for the whole tensor. Lowest overhead, worst accuracy — outliers force the scale high.
- **Per-channel:** One scale per output channel. Much better accuracy, negligible overhead. Standard for weights.
- **Per-group:** One scale per G consecutive weights (e.g., G=128). Used in 4-bit methods (GPTQ, AWQ) where per-channel alone is insufficient.

---

## 4. Memory Savings

| Precision | Bytes/param | 7B model | 70B model |
|---|---|---|---|
| FP32 | 4 | 28 GB | 280 GB |
| FP16 / BF16 | 2 | 14 GB | 140 GB |
| INT8 | 1 | 7 GB | 70 GB |
| INT4 | 0.5 | 3.5 GB | 35 GB |

INT8 enables a 7B model on a single 8 GB GPU. INT4 enables a 13B model on 8 GB.

---

## 5. Which Layers to Quantize

- **Linear projections (attention, FFN):** Primary target — large, well-behaved weight distributions.
- **Layer norm / RMS norm:** Very sensitive, small parameter count — leave in FP16.
- **Embedding and lm_head layers:** Often kept in FP16.

Mixed-precision (e.g., LLM.int8()) automatically keeps sensitive layers in FP16.

---

## 6. Accuracy vs Compression

| Method | Bit-width | Typical perplexity degradation vs FP16 |
|---|---|---|
| INT8 (LLM.int8(), SmoothQuant) | 8-bit | < 0.5 |
| GPTQ / AWQ | 4-bit | 0.5 – 1.5 |
| GGUF Q4_K_M | 4-bit | ~1.0 |
| GGUF Q2_K | 2-bit | 3.0 – 5.0 |

INT8 is essentially lossless for most tasks. Below 4-bit, quality degrades meaningfully.

---

## 7. Methods

| Method | See |
|---|---|
| LLM.int8() | [INT8 Quantization](int8_quantization.md) |
| GPTQ | [GPTQ](gptq.md) |
| AWQ | [AWQ](awq.md) |
| SmoothQuant | [SmoothQuant](smoothquant.md) |
| GGUF / llama.cpp | [GGUF & GGML](gguf_ggml.md) |
| Method selection | [Quantization Trade-offs](quantization_tradeoffs.md) |
