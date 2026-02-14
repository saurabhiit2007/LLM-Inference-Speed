## 1. Core Architecture

**NVIDIA's optimization stack** for LLM inference on their GPUs <br>
- Built on TensorRT for kernel-level optimization
- Focuses on extracting maximum performance from NVIDIA hardware
- Trade-off: Complex setup vs peak performance

---

---

## 2. Key Technologies

### 1. In-flight Batching (Continuous Batching)
- Similar to vLLM's approach
- Dynamically adds/removes requests during execution
- Optimized specifically for NVIDIA GPU scheduling

### 2. Paged KV Cache
- Inspired by vLLM's PagedAttention
- NVIDIA-optimized memory management
- Custom CUDA kernels for memory operations

### 3. Kernel Fusion
- Combines multiple operations into single kernels
- Reduces memory transfers between GPU operations
- Examples: LayerNorm+Residual, QKV projection fusion

### 4. FlashAttention & FP8 Support
- Integrated FlashAttention-2 for memory-efficient attention
- Native FP8 quantization on Hopper GPUs (H100)
- 2x throughput vs FP16 with minimal accuracy loss

---

---

## 3. Quantization Support

**Weight-Only Quantization:** <br>
- INT8/INT4 weights, FP16 activations
- 2-4x memory reduction
- GPTQ, AWQ methods supported

**Activation Quantization:** <br>
- FP8 (Hopper GPUs only)
- SmoothQuant for INT8 activations

---

---

## 4. Model Parallelism

### Tensor Parallelism
- Splits model layers across GPUs
- Low-latency (intra-node communication)
- Best for latency-sensitive serving

### Pipeline Parallelism
- Splits model vertically into stages
- Higher throughput for large batches
- Micro-batching to reduce bubbles

### Combined TP+PP
- Multi-dimensional parallelism
- Example: 8-way TP × 4-way PP for 32 GPUs

---

---

## 5. Engine Building Process

**Two-Step Workflow:** <br>
1. **Build:** Model → Optimized TensorRT engine (slow, one-time)
2. **Runtime:** Load engine → Inference (fast)

**Key considerations:** <br>
- Engines are GPU-specific (H100 engine ≠ A100 engine)
- Rebuild required for different batch sizes or sequence lengths
- Trade flexibility for maximum performance

---

---

## 6. Multi-GPU Inference Modes

**KV Cache Transfer Optimization:**
- Custom NCCL/NVLink operations for KV cache
- Overlaps communication with computation
- Critical for tensor parallel setups

---

---

## 7. Interview Q&A

**Q: When to choose TensorRT-LLM over vLLM?** <br>
A: When you need absolute maximum throughput on NVIDIA GPUs and can handle complex setup. vLLM for ease of use and flexibility; TensorRT-LLM for peak performance.

---

**Q: Why is engine building necessary?** <br>
A: TensorRT optimizes compute graphs at compile time (kernel selection, fusion, memory layout). This specialization achieves maximum performance but loses runtime flexibility.

---

**Q: How does TensorRT-LLM handle dynamic shapes?** <br>
A: Uses optimization profiles with min/max ranges during build. Runtime performance varies by how well actual inputs match the profile. Too wide a range reduces optimization effectiveness.

---

**Q: What's the FP8 accuracy impact?** <br>
A: On Hopper GPUs with proper calibration, <1% accuracy degradation for most models. Requires per-tensor scaling and careful quantization of outlier features.

---

**Q: Why does TensorRT-LLM require specific CUDA versions?** <br>
A: Tightly integrated with CUDA toolkit for custom kernel launches, memory management, and GPU-specific optimizations. Newer releases exploit latest CUDA features (e.g., Hopper's Tensor Memory Accelerator).

---