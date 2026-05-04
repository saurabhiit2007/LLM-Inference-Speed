# TensorRT-LLM

**Optimization axis: latency via compile-time hardware specialization**

TensorRT-LLM is NVIDIA's inference toolkit that compiles a model into a GPU-specific engine ahead of serving. It consistently benchmarks at the highest per-token throughput of any framework — at the cost of a slow build step and hardware lock-in.

---

## 1. The Problem It Was Built to Solve

Generic inference runtimes (PyTorch, Hugging Face) run models through a general-purpose compute graph. They leave substantial NVIDIA hardware capability unused: Tensor Cores under-utilized, memory operations not fused, CUDA kernels not specialized for the exact GPU generation being targeted.

TensorRT-LLM's answer is to treat inference as a **compilation problem**: take the model once, spend time analyzing and optimizing it for a specific GPU, and then serve from the compiled artifact.

---

## 2. Core Insight: Compile Once, Serve Fast

The build step performs optimizations that are impossible at runtime:

- **Kernel fusion:** merge adjacent ops (e.g., LayerNorm + residual add + QKV projection) into a single CUDA kernel, eliminating intermediate memory writes
- **CUDA graph capture:** the entire forward pass for a fixed batch shape is recorded as a CUDA graph and replayed with minimal CPU overhead
- **Hardware-specific instructions:** H100 engines use FP8 Tensor Core instructions that don't exist on A100; A100 engines use INT8 Tensor Cores differently — the compiler selects the right path per GPU generation
- **Quantization folding:** GPTQ/AWQ quantization constants are folded into the kernel itself rather than applied at runtime

```
Model weights + architecture definition
          ↓  trtllm-build (5–30 min)
GPU-specific compiled engine
          ↓  runtime
Inference at hardware ceiling
```

The compiled engine is opaque and GPU-specific: an H100 engine will not run on an A100.

---

## 3. Architecture

### Two-step workflow

**Build phase (one-time):**
```bash
trtllm-build --model_dir ./llama-3-8b \
             --dtype float16 \
             --tp_size 1 \
             --output_dir ./engine
```
This generates a `.engine` file specific to the GPU and the chosen precision, batch size range, and sequence length range.

**Runtime phase:**
```python
from tensorrt_llm import LLM
llm = LLM(model="./engine")
output = llm.generate("The capital of France is")
```
The runtime loads the pre-compiled engine and serves with minimal overhead.

### In-flight batching and paged KV cache

TensorRT-LLM adopted continuous batching and a paged KV cache (inspired by vLLM's PagedAttention) with NVIDIA-optimized CUDA kernels. These run faster than vLLM's equivalent because the kernels are compiled for the specific GPU.

### FP8 on Hopper GPUs

On H100/H200, TensorRT-LLM supports FP8 quantization natively via Hopper's Tensor Memory Accelerator. With proper calibration, FP8 delivers roughly 2× the throughput of FP16 at under 1% accuracy degradation.

### Model parallelism

Supports tensor parallelism (TP) and pipeline parallelism (PP) with custom NCCL communication primitives that overlap computation and communication — faster than vLLM's Ray-based TP at high GPU counts.

---

## 4. Tradeoffs

| | |
|---|---|
| **Build time** | 5–30 minutes per model × GPU combination; any change (new quantization, different batch size range) requires a rebuild |
| **Hardware lock-in** | Engines are non-portable; H100 engines don't run on A100 |
| **Slower iteration** | Can't hot-swap models; unsuitable for rapid experimentation |
| **Operational complexity** | Requires ML engineering investment to maintain engine builds across model versions |

The performance advantage over vLLM at equivalent settings is typically 8–15%. Whether that justifies the operational overhead depends on the scale of the deployment.

---

## 5. When to Use

**Use TensorRT-LLM when:** you have a stable, production-hardened model on fixed NVIDIA hardware and need to extract every last token/second — for example, a high-traffic product with a dedicated inference cluster.

**Don't use TensorRT-LLM when:** you're iterating on models frequently, running on non-NVIDIA hardware, or don't have the engineering capacity to maintain engine builds.

**Common production pattern:** pair TensorRT-LLM as the inference backend with Triton Inference Server as the HTTP/gRPC frontend for enterprise-grade observability and model versioning.
