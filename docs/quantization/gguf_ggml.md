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

---

## Common Interview Questions

**Q1: Why GGUF instead of safetensors or PyTorch?** <br>
A: GGUF designed for inference, not training. Memory-mapped loading, quantization metadata embedded, optimized for llama.cpp ecosystem.

---

**Q2: What's memory mapping and why does it matter?** <br>
A: OS maps file directly to virtual memory. Model loads "instantly" because data stays on disk until accessed. Enables running larger models than RAM.

---

**Q3: Why CPU inference for LLMs?** <br>
A: Consumer hardware accessibility. Most users don't have high-end GPUs. CPUs have large memory, enabling bigger models with quantization.

---

**Q4: What's the quality difference between Q4_K_M and Q8_0?** <br>
A: Q8_0: ~99% of FP16 quality. Q4_K_M: ~95-97%. For chat, Q4_K_M usually indistinguishable. For precise tasks, Q8_0 safer.

---

**Q5: How do K-quants improve on original Q4_0?** <br>
A: Better block structure, per-block scales and mins, optimized for specific bit rates. Q4_K_M beats Q4_0 quality while being similar size.

---

**Q6: Can GGUF models be used outside llama.cpp?** <br>
A: Yes. Libraries: llama-cpp-python (Python), whisper.cpp (audio), GPT4All, Ollama, LM Studio all support GGUF.

---

**Q7: What's the tradeoff between Q4_K_S and Q4_K_M?** <br>
A: Q4_K_S: Smaller (~4.0 bpw), faster. Q4_K_M: Slightly larger (~4.5 bpw), better quality. Difference: ~0.3 PPL for 7B models.

---

**Q8: Why multiple quantization types instead of just one?** <br>
A: Different hardware, use cases, quality requirements. Q2_K for extreme memory constraints, Q6_K for quality-critical applications, Q4_K_M for general use.

---

**Q9: What's "bpw" (bits per weight)?** <br>
A: Effective bits including quantization metadata overhead. Q4_K_M is labeled "4-bit" but actually ~4.5 bpw due to scales/mins storage.

---