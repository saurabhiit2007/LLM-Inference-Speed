## 1. Quick Selection Guide

| Use Case | Recommended Framework | Rationale |
|----------|----------------------|-----------|
| Maximum throughput, multi-tenancy | **vLLM** | PagedAttention, multi-LoRA, continuous batching |
| Peak NVIDIA GPU performance | **TensorRT-LLM** | Hardware-specific optimization, FP8 support |
| Production stability, HF ecosystem | **TGI** | Rust reliability, grammar constraints, fast deploys |
| Research + training integration | **DeepSpeed-Inference** | Unified training/inference, ZeRO-Inference |
| Multi-model pipelines, enterprise | **Triton** | Framework-agnostic, model versioning, ensembles |

---

---

## 2. Feature Comparison Matrix

| Feature | vLLM | TensorRT-LLM | TGI | DeepSpeed | Triton |
|---------|------|--------------|-----|-----------|--------|
| **Memory Efficiency** | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ (via vLLM) |
| **Ease of Setup** | ★★★★★ | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| **Peak Throughput** | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★★ (via backends) |
| **Multi-LoRA** | ★★★★★ | ★★☆☆☆ | ☆☆☆☆☆ | ☆☆☆☆☆ | ★★★★★ (via vLLM) |
| **Model Support** | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★★★ |
| **Production Maturity** | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★★★ |

---

---

## 3. Technical Deep Dive

### Memory Management Approaches

**vLLM (PagedAttention):** <br>
- Paged KV cache with block tables
- <4% memory waste
- Best for variable-length sequences

**TensorRT-LLM:** <br>
- Paged KV cache inspired by vLLM
- NVIDIA-optimized CUDA kernels
- Tightly coupled with GPU architecture

**TGI:** <br>
- FlashAttention for memory efficiency
- No paging, simpler approach
- Good for single-tenant scenarios

**DeepSpeed:** <br>
- Basic KV cache management
- ZeRO-Inference for CPU/NVMe offloading
- Suited for extreme model sizes

---

### Batching Strategies

**Continuous Batching (vLLM, TGI, DeepSpeed-FastGen):** <br>
- Iteration-level scheduling
- Immediate slot filling
- 20-30% throughput improvement

**Static Batching (Traditional):** <br>
- Wait for full batch completion
- Simpler implementation
- GPU idle time

**Dynamic Batching (Triton):** <br>
- Time-window accumulation
- Less sophisticated than continuous
- Still effective for many workloads

---

### Quantization Comparison

| Framework | INT8 | INT4 | FP8 | Methods |
|-----------|------|------|-----|---------|
| vLLM | ✓ | ✓ | ✓ | AWQ, GPTQ, SmoothQuant |
| TensorRT-LLM | ✓ | ✓ | ✓ | Native + AWQ, GPTQ |
| TGI | ✓ | ✓ | ✓ | bitsandbytes, AWQ, GPTQ, EETQ |
| DeepSpeed | ✓ | ✓ | ✗ | ZeroQuant |
| Triton | Depends on backend | | | |

**FP8 Note:** Only on NVIDIA Hopper (H100+), 2x throughput vs FP16

---

---

## 4. Latency Characteristics

### First Token Time to Time (TTFT)

**Best to Worst:** <br>
1. TGI (Rust + safetensors, optimized cold start)
2. vLLM (Python overhead but chunked prefill)
3. TensorRT-LLM (engine loading overhead)
4. DeepSpeed-Inference
5. Triton (abstraction layer overhead)

---

### Inter-Token Latency (ITL)

**Best to Worst:** <br>
1. TensorRT-LLM (maximum kernel optimization)
2. vLLM (PagedAttention efficiency)
3. TGI (FlashAttention + Rust)
4. Triton (depends on backend)
5. DeepSpeed-Inference

---

### Throughput (tokens/second)

**Best to Worst:** <br>
1. vLLM (PagedAttention + continuous batching)
2. TensorRT-LLM (hardware optimization)
3. TGI (solid continuous batching)
4. Triton + vLLM backend
5. DeepSpeed-Inference

---

---

## 5. Multi-GPU Considerations

### Tensor Parallelism Performance

**TensorRT-LLM:**  <br>
- Custom NCCL optimizations
- Lowest latency for TP

**vLLM:** <br>
- Ray-based distribution
- Good performance, more overhead

**TGI:** <br>
- Rust-based TP implementation
- Efficient but less optimized than TensorRT

---

### Pipeline Parallelism

- Best support: DeepSpeed-Inference, TensorRT-LLM
- Limited: vLLM (experimental)
- Not primary focus: TGI

---

---

## 6. Production Deployment Factors

### Containerization

**Easiest:** TGI, vLLM (official Docker images, simple configs)  
**Medium:** Triton (more complex configs)  
**Complex:** TensorRT-LLM (build dependencies), DeepSpeed

---

### Monitoring & Observability

**Most Comprehensive:** Triton > TGI > vLLM > DeepSpeed  
**Key Metrics:** Queue depth, batch size, KV cache utilization, token throughput

---

### Scaling Patterns

**Horizontal (Multiple Instances):** All support, TGI/Triton easiest  
**Vertical (Bigger GPUs):** TensorRT-LLM extracts most value  
**Multi-Model:** Triton's core strength

---

---

## 7. Interview Q&A

**Q: vLLM vs TensorRT-LLM for production?** <br>
A: vLLM for faster iteration, multi-LoRA, easier ops. TensorRT-LLM when you need absolute maximum throughput and have dedicated ML Eng team for maintenance.

---

**Q: Why doesn't everyone use TensorRT-LLM if it's fastest?** <br>
A: Setup complexity, need to rebuild engines for changes, GPU-specific builds, harder debugging. Speed gain (10-20%) often not worth operational overhead.

---

**Q: When is DeepSpeed-Inference the right choice?** <br>
A: When you're already using DeepSpeed for training and want unified tooling. Or when you need ZeRO-Inference for models larger than GPU memory. Not for general production serving.

---

**Q: Can you mix frameworks?** <br>
A: Yes via Triton backends. Run vLLM for LLM, TensorRT for embeddings, Python backend for custom logic. Single server, unified API.

---

**Q: How to choose between vLLM and TGI?** <br>
A: Similar performance. Choose TGI for HuggingFace integration, grammar constraints, Rust reliability. Choose vLLM for multi-LoRA, latest features, slightly higher throughput.

---

**Q: What's the main bottleneck each framework optimizes?** <br>
A: vLLM → memory fragmentation. TensorRT-LLM → compute efficiency. TGI → deployment stability. DeepSpeed → model size limits. Triton → pipeline complexity.

---

**Q: Impact of continuous batching on latency?** <br>
A: Slightly increases average latency per request (5-10%) but dramatically increases throughput (20-30%). Worth it for high-traffic scenarios, not for latency-critical single-user apps.

---