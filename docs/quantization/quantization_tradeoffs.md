## 1. Memory vs. Quality Spectrum

| Precision | Memory (7B) | Typical PPL Δ | Use Case |
|-----------|-------------|---------------|----------|
| FP16 | 14 GB | 0.0 (baseline) | Training, high-quality inference |
| INT8 | 7 GB | +0.1-0.5 | Production standard |
| INT4 (GPTQ/AWQ) | 3.5 GB | +0.5-1.5 | Commodity GPU inference |
| 3-bit | 2.6 GB | +1.5-3.0 | Extreme compression |
| Q2_K | 2 GB | +3.0-5.0 | Last resort |

---

---

## Speed vs. Quality

### Inference Latency (7B model, batch=1)
| Method | GPU (A100) | CPU (32-core) |
|--------|-----------|---------------|
| FP16 | 20 ms/token | N/A (OOM) |
| INT8 | 10 ms/token | N/A (OOM) |
| INT4 (AWQ) | 7 ms/token | 80 ms/token |
| GGUF Q4_K_M | 8 ms/token | 35 ms/token |

**Key insight**: CPU competitive for quantized models, especially with optimized kernels.

---

---

## Quantization Method Selection

### Decision Tree

**Need extreme compression (2-3 bit)?**
→ GPTQ (best quality at extreme compression)

**Standard 4-bit, fast quantization needed?**
→ AWQ (10 min vs 4 hours for GPTQ, similar quality)

**CPU deployment?**
→ GGUF with llama.cpp (optimized CPU kernels)

**GPU deployment, production quality?**
→ INT8 with SmoothQuant (robust, well-supported)

**Fine-tuning on limited memory?**
→ QLoRA with NF4 (efficient training)

---

---

## Layer-wise Quantization Strategy

### Typical Configuration
```
Embeddings: FP16 (critical for semantic space)
Attention Weights (Q, K, V): INT4/INT8
Attention Output: INT8
FFN Weights: INT4 (largest, most compressible)
FFN Activations: INT8
Layer Norm: FP16 (small, sensitive)
Final Layer: FP16 or INT8
```

---

### Rationale

- **FFN**: 66% of parameters, less sensitive → aggressive INT4
- **Attention**: 33% of parameters, more sensitive → INT8 or careful INT4
- **Norms/Embeddings**: <1% of parameters → keep FP16

---

---

## Mixed Precision Strategies

### W4A8 (Weight 4-bit, Activation 8-bit)

- Best of both worlds for many use cases
- Weights: AWQ/GPTQ 4-bit
- Activations: SmoothQuant INT8
- 6-8× memory reduction, <1% quality loss

---

### W8A8 (Both 8-bit)

- Production standard for quality-critical apps
- 4× memory reduction
- Hardware-accelerated on all modern platforms
- <0.5% quality loss with SmoothQuant

---

---

## Hardware Considerations

### NVIDIA GPUs

- **Tensor Cores**: INT8 (Turing+), INT4 (Hopper)
- **Recommendation**: INT8 for A100, INT4 for H100
- **Custom kernels**: AWQ's TinyChat, ExLlamaV2 for GPTQ

---

### AMD GPUs

- **ROCm**: INT8 support
- **Recommendation**: INT8, limited INT4 optimization
- **Ecosystem**: Less mature than NVIDIA

---

### Apple Silicon

- **Metal**: INT8, INT4 via llama.cpp
- **Recommendation**: GGUF Q4_K_M or Q6_K
- **Strength**: Unified memory architecture

---

### CPU (x86)

- **VNNI (Cascade Lake+)**: INT8 acceleration
- **AVX512**: INT8/INT4 kernels
- **Recommendation**: GGUF with llama.cpp, Q4_K_M sweet spot

---

---

## Calibration Data Tradeoffs

### Size

- **100 samples**: Usually sufficient, fast
- **1000 samples**: Marginal quality improvement
- **10000 samples**: No additional benefit, waste of time

---

### Diversity vs. Representativeness

- **In-domain**: Better for specialized models
- **General (WikiText)**: Better for general models
- **Mixed**: Best for production

---

---

## Dynamic vs. Static Quantization

### Static (PTQ)
**Pros**: Faster inference, lower memory
**Cons**: Fixed scales, may underfit outliers
**Best for**: Stable input distributions

---

### Dynamic
**Pros**: Adapts to inputs, better quality
**Cons**: Runtime overhead (scale computation)
**Best for**: Varied input distributions, activation quantization

---
