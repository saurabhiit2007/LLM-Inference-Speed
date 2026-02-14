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

---

## Common Interview Questions

**Q1: When would you use INT8 over INT4?**
A: (1) Quality-critical applications where 0.5% matters, (2) Hardware with INT8 acceleration but no INT4, (3) Activations (INT4 activations too lossy).

---

**Q2: What's the minimum model size for effective quantization?**
A: ~1B parameters. Smaller models have less redundancy, quantization hurts more. <500M models: stick to FP16.

---

**Q3: How do you decide between GPTQ and AWQ?**
A: GPTQ for 3-bit or when quality is paramount. AWQ for 4-bit, faster iteration, production deadlines. Quality difference minimal at 4-bit.

---

**Q4: What's the biggest failure mode of quantization?**
A: Outlier channels not handled properly. SmoothQuant, AWQ, or mixed-precision decomposition (LLM.int8()) essential for robust quantization.

---

**Q5: Can you quantize a fine-tuned model?**
A: Yes, but better to fine-tune with QLoRA (quantize-aware fine-tuning). Post-hoc quantization of fine-tuned models can be more sensitive than base models.

---

**Q6: What's the practical lower bound for useful quantization?**
A: 2-bit with current methods. Below that, quality degrades unacceptably even for large models (70B+). Active research on sub-2-bit.

---

**Q7: How much quality loss is acceptable?**
A: Domain-dependent. Chatbots: 2-3% acceptable. Code generation: <1%. Reasoning tasks: <0.5%. Benchmark on your specific use case.

---

**Q8: Should you quantize KV cache?**
A: Yes for long context (4K+). INT8 KV cache with SmoothQuant: 2× memory savings, <0.5% quality loss. Critical for 32K+ context.

---

**Q9: What's the ROI of quantization engineering time?**
A: High. 4 hours GPTQ quantization → 8× memory reduction → 8× more users per GPU → 8× cost reduction. One-time cost, continuous savings.

---

**Q10: Biggest misconception about quantization?**
A: "Lower bits always means faster." Reality: memory-bound scenarios see speedup, compute-bound scenarios don't. Batch size, context length matter more than bit width for speed.

---