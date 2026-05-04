# DeepSpeed Inference

**Optimization axis: serving models too large to fit in GPU VRAM**

DeepSpeed Inference is Microsoft's inference library, best known for ZeRO-Inference — a technique that shards model weights across GPU, CPU, and NVMe storage so you can serve a model that doesn't fit in GPU memory at all. Its primary audience is teams already using DeepSpeed for training who want a unified toolchain.

---

## 1. The Problem It Was Built to Solve

A 70B-parameter model in FP16 requires roughly 140 GB of VRAM. An 8×A100 node has 640 GB of GPU memory — sufficient for a 70B model. But a 180B model doesn't fit even there. And most organizations can't afford eight A100s per deployment anyway.

The alternatives before ZeRO-Inference were narrow: model parallelism (requires custom code), quantization (accuracy hit), or just not serving large models. ZeRO-Inference added a third option: use the full memory hierarchy.

A secondary problem: teams that train models with DeepSpeed (using ZeRO-2 or ZeRO-3 for training memory efficiency) previously had to rewrite their inference pipeline entirely. DeepSpeed Inference closes that gap.

---

## 2. Core Insight: The Memory Hierarchy as a Single Pool

ZeRO-Inference treats GPU VRAM, CPU RAM, and NVMe SSD as a single unified memory pool for model weights.

```
NVMe SSD (TBs available, ~3 GB/s)
      ↑ on-demand load
CPU RAM (100s of GBs, ~50 GB/s)
      ↑ on-demand load
GPU VRAM (10s of GBs, ~2 TB/s)
      ← active layer runs here
```

Only the layers currently executing need to live in GPU VRAM. All other layers sit in CPU RAM or NVMe, loaded on demand as the forward pass proceeds. This lets you run a 70B model on a single GPU with 24 GB of VRAM — at the cost of latency proportional to how often weights need to be loaded from slower memory.

The tradeoff is explicit: you can serve the model at all, but inter-token latency is significantly higher than if the model fit entirely in GPU memory.

---

## 3. Architecture

### ZeRO-Inference offloading

Weights are partitioned and distributed across the memory hierarchy. At inference time, each layer's weights are loaded into GPU memory, the forward pass executes, and the weights are evicted to make room for the next layer. The eviction policy is configurable — you can pin frequently used layers in GPU VRAM to reduce reloads.

```python
import deepspeed
import torch

engine = deepspeed.init_inference(
    model,
    tensor_parallel={"tp_size": 4},
    dtype=torch.float16,
    replace_with_kernel_inject=True
)
output = engine("The capital of France is")
```

`replace_with_kernel_inject=True` swaps standard PyTorch Transformer ops with DeepSpeed's optimized CUDA kernels transparently.

### SplitFuse scheduling (FastGen)

DeepSpeed-FastGen (introduced 2023) added SplitFuse: long prefills are split into chunks and fused with ongoing decode operations in the same forward pass. This mirrors vLLM's chunked prefill concept — preventing a single long prompt from stalling all concurrent decode steps. In practice, SplitFuse reduces tail latency for workloads with mixed short and long prompts.

### Tensor and pipeline parallelism

For multi-GPU setups, DeepSpeed supports both tensor parallelism (split weight matrices across GPUs) and pipeline parallelism (split model layers across GPUs). Pipeline parallelism with micro-batching is particularly useful for models above 100B parameters where tensor parallelism alone can't keep all GPUs busy.

---

## 4. Tradeoffs

| | |
|---|---|
| **Latency with offloading** | Loading weights from CPU RAM or NVMe on each forward pass adds significant latency — 10–100× slower than GPU-resident inference |
| **Production maturity** | Less hardened than vLLM or TGI; smaller ecosystem; fewer production deployments to learn from |
| **Throughput ceiling** | No paged KV cache (uses simpler KV management); lower concurrent-request throughput than vLLM at the same GPU count |
| **Community size** | Primarily used within the DeepSpeed ecosystem; most inference-specific documentation assumes training context |

---

## 5. When to Use

**Use DeepSpeed Inference when:** you need to serve a model that exceeds your GPU VRAM budget, or you're already running a DeepSpeed training pipeline and want a single unified toolchain from training to serving.

**Don't use DeepSpeed Inference when:** your model fits in GPU VRAM and you need production-grade throughput — vLLM's PagedAttention and continuous batching will significantly outperform it. For models that do fit in GPU memory, vLLM or TensorRT-LLM are the better choices.

**The characteristic deployment:** a research team fine-tuning and serving 70B+ models on a limited GPU budget, where higher latency is acceptable and operational simplicity matters more than peak throughput.
