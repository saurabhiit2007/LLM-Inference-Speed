# Serving Framework Comparison

Every serving framework makes a different bet about what the binding constraint is. Understanding those bets explains why each framework is built the way it is, and which one is right for a given deployment.

---

## 1. The Design Axes

Each framework optimizes along a primary axis:

| Framework | Primary Axis | Core Mechanism |
|---|---|---|
| **vLLM** | Throughput via memory efficiency | PagedAttention eliminates KV cache fragmentation; 2–3× more concurrent requests on the same GPU |
| **TensorRT-LLM** | Latency via compile-time specialization | GPU-specific compiled engines; kernel fusion + CUDA graphs extract hardware-ceiling performance |
| **TGI** | Deployment simplicity | Rust router + HuggingFace ecosystem; batteries-included for any HF Hub model |
| **DeepSpeed Inference** | Model scale beyond GPU VRAM | ZeRO-Inference shards weights across GPU/CPU/NVMe; enables models that don't fit in memory |
| **Triton** | Multi-model orchestration | Framework-agnostic router wrapping other backends; versioning, pipelines, observability |

These axes are real tradeoffs, not marketing. A framework that optimizes for one axis typically sacrifices on the others.

---

## 2. The Throughput / Latency / Cost Trilemma

The three things every serving deployment wants are:

- **High throughput** (tokens/second across all concurrent users)
- **Low latency** (time-to-first-token and inter-token latency per user)
- **Low cost** (GPU-hours per million tokens)

You can't maximize all three simultaneously. The frameworks make different choices:

**vLLM maximizes throughput and cost-efficiency.** PagedAttention packs more concurrent requests into the same GPU memory, so each GPU produces more tokens per second and more tokens per dollar. Per-request latency is acceptable but not the lowest achievable.

**TensorRT-LLM minimizes latency.** Compiled kernels reduce per-token compute time by 8–15% over vLLM at equivalent settings. The cost: a 5–30 minute build cycle per model-GPU combination and an engineering team to maintain it.

**TGI is a Pareto-acceptable choice for teams that want neither extreme.** Solid throughput, acceptable latency, minimal setup. As of 2025, TGI is in maintenance mode — HuggingFace now recommends vLLM or SGLang for new deployments.

**DeepSpeed trades throughput and latency for model scale.** If you need to serve a model that doesn't fit in GPU memory at all, DeepSpeed is often the only option. Accept the latency penalty.

**Triton is orthogonal to this trilemma.** It doesn't change inference performance — it adds orchestration on top of another framework's engine.

---

## 3. Quick Selection Guide

| Scenario | Framework | Reason |
|---|---|---|
| New production deployment, single LLM | **vLLM** | Best throughput, active development, multi-LoRA, prefix caching |
| Fixed NVIDIA hardware, stability matters, max tokens/sec | **TensorRT-LLM** | Hardware-ceiling performance when you can absorb 30-min build cycles |
| Already running TGI in production | **TGI** | Don't migrate unless you're hitting its limits |
| Model larger than GPU VRAM, existing DeepSpeed training stack | **DeepSpeed Inference** | ZeRO-Inference is the tool for this specific problem |
| Multi-model pipeline, model versioning, enterprise observability | **Triton** | Not an engine — wraps vLLM or TensorRT-LLM with enterprise infrastructure |
| New deployment, exploring alternatives to vLLM | **SGLang** | Emerging competitor; strong performance on structured generation and multi-step programs |

---

## 4. Feature Matrix

| Feature | vLLM | TensorRT-LLM | TGI | DeepSpeed | Triton |
|---|---|---|---|---|---|
| PagedAttention / Paged KV | Yes | Yes (NVIDIA-optimized) | No | No | Via backend |
| Continuous batching | Yes | Yes | Yes | Yes (FastGen) | Via backend |
| Prefix / APC caching | Yes | Partial | No | No | Via backend |
| Multi-LoRA serving | Yes | Limited | No | No | Via backend |
| Grammar-constrained output | Via Outlines | No | Yes (native) | No | Via backend |
| FP8 (H100+) | Yes | Yes (native) | Yes | No | Via backend |
| CPU/NVMe weight offloading | No | No | No | Yes | No |
| Model versioning + A/B | No | No | No | No | Yes (core feature) |
| Ensemble pipelines | No | No | No | No | Yes (core feature) |
| Active development | Yes | Yes | Maintenance mode | Limited | Yes |

---

## 5. The Emerging Competitor: SGLang

SGLang (from the Stanford/UC Berkeley group behind vLLM) is a newer framework optimized for multi-step LLM programs — chains of calls, structured generation, and agent loops. Its RadixAttention extends prefix caching to arbitrary tree-structured KV reuse, which dramatically accelerates workloads where multiple requests share overlapping prefixes beyond a simple shared system prompt.

For straightforward single-turn serving, vLLM and SGLang perform similarly. For agentic workloads with complex prompt trees, SGLang's caching model can provide a substantial advantage. SGLang is worth evaluating for new deployments alongside vLLM.

---

## 6. The Common Production Stack

Most high-traffic production deployments end up at one of two configurations:

**Configuration A (most common):**
vLLM → direct HTTP API → load balancer

Simple, well-understood, low operational overhead. The default choice for teams that don't have specific reasons to go elsewhere.

**Configuration B (enterprise / NVIDIA-optimized):**
TensorRT-LLM → Triton → load balancer

Maximum GPU utilization, model versioning, enterprise observability. Requires an ML engineering team to maintain engine builds and Triton configurations. Justified at scale where the 8–15% throughput premium compounds into real cost savings.
