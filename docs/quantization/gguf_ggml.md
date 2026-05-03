## 1. GGML (Georgi Gerganov Machine Learning)

C library for machine learning inference, optimized for CPU execution of LLMs.

**Key Features**: <br>

- Pure C/C++ (no Python runtime)
- CPU-optimized kernels (AVX2, AVX512, NEON)
- 4-bit to 16-bit quantization
- Memory-mapped model loading
- Apple Metal, CUDA, OpenCL backends

---

---

## GGUF (GGML Universal Format)

Successor to GGML format (deprecated). Single-file model container.

### File Structure
```
[Header]
- Magic number: "GGUF"
- Version
- Tensor count, KV metadata count

[Metadata]
- Model hyperparameters
- Tokenizer config
- Quantization scheme
- Author, license, etc.

[Tensor Info]
- Name, dimensions, type, offset

[Tensor Data]
- Actual quantized weights (memory-mapped)
```

---

### Advantages

- **Single file**: All model data + config in one .gguf
- **Memory mapping**: Load multi-GB models instantly, use minimal RAM
- **Extensible**: KV metadata for any additional info
- **Backward compatible**: Old GGML loaders fail safely

---

---

## Quantization Types

### K-Quantization (K-quants)
Optimized 2-6 bit quantization schemes:

| Type | Bits | Description | Use Case |
|------|------|-------------|----------|
| Q2_K | ~2.5 | Extreme compression | Large models on limited RAM |
| Q3_K_S | ~3.4 | Small, less accurate | Acceptable quality loss |
| Q3_K_M | ~3.7 | Medium quality | Balanced |
| Q4_K_S | ~4.0 | Small, good quality | Recommended default |
| Q4_K_M | ~4.5 | Medium, best quality | Best 4-bit option |
| Q5_K_S | ~5.0 | Small, very good | Low loss |
| Q6_K | ~6.0 | High quality | Near-FP16 quality |

---

### Legacy Quantization

- Q4_0: Original 4-bit (group size 32)
- Q4_1: 4-bit with per-group min (better than Q4_0)
- Q5_0, Q5_1: 5-bit variants
- Q8_0: INT8 quantization

---

### Importance Matrix (I-quants)
Uses importance scores to allocate more bits to salient weights:

- `IQ3_XXS`: 3-bit with importance weighting
- `IQ4_XS`: 4-bit with importance weighting

---

---

## llama.cpp

Reference implementation for GGUF inference.

```bash
# Convert HF model to GGUF
python convert-hf-to-gguf.py model_path --outtype q4_K_M

# Inference
./main -m model.gguf -p "Prompt" \
  -n 512 \              # Max tokens
  -c 4096 \             # Context length
  -t 8 \                # Threads
  --mlock               # Lock in RAM (prevent swapping)
```

---

---

## CPU Optimizations

### Kernel Fusion
```
attention = softmax(QK^T/√d) @ V
```
Fused into single kernel instead of 3 separate ops.

---

### Cache-Friendly Layout
Reorder tensors for sequential memory access. Dramatic speedup on CPU.

---

### Quantized Matrix Multiply
Custom AVX2/AVX512 kernels for INT4/INT8 GEMM. 4-8× faster than naive C.

---

---

## Performance

**M2 Max (Metal)**: <br>

- 7B Q4_K_M: ~40 tokens/sec
- 13B Q4_K_M: ~25 tokens/sec

**AMD 5950X (16-core)**: <br>

- 7B Q4_K_M: ~30 tokens/sec
- 13B Q4_K_M: ~15 tokens/sec

---
