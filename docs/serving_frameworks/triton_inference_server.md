## 1. Overview

**NVIDIA's general-purpose inference server** <br>
- Framework-agnostic (PyTorch, TensorFlow, ONNX, TensorRT)
- Not LLM-specific, but increasingly optimized for them
- Focus: Enterprise deployment, multi-model serving, complex pipelines

---

---

## 2. Core Architecture

### Backend System
**Pluggable backends for different frameworks:**
- Python backend (custom inference logic)
- PyTorch backend (TorchScript models)
- TensorRT backend (TensorRT engines)
- vLLM backend (integration added 2024)

**Benefit:** Mix different model types in same server

---

### Model Repository
- Centralized model storage (local/S3/GCS/Azure)
- Version management
- Hot-reloading of model versions

---

---

## 3. LLM-Specific Features (2024-2025)

### vLLM Backend Integration
- Uses vLLM engine under the hood
- Triton API layer on top
- Get vLLM's PagedAttention + Triton's enterprise features

---

### TensorRT-LLM Backend
- Native integration with TensorRT-LLM engines
- Maximum performance for NVIDIA GPUs
- Requires pre-built TensorRT-LLM engines

---

---

## 4. Advanced Serving Capabilities

### Model Ensembles
**Multi-stage pipelines as single endpoint:**
- Preprocessing → Embedding → LLM → Postprocessing
- Automatic scheduling between stages
- Example: RAG pipeline with retrieval + generation

### Dynamic Batching
- Accumulates requests up to max batch size
- Timeout-based flushing
- More basic than vLLM/TGI continuous batching

### Sequence Batching
- For stateful models (e.g., streaming LLMs)
- Maintains state across multiple requests
- Useful for chat applications

---

---

## 5. Model Configuration

**Model config.pbtxt example:** <br>
```protobuf
backend: "vllm"
max_batch_size: 32

instance_group [{ 
  count: 1
  kind: KIND_GPU
}]

parameters: {
  key: "max_tokens"
  value: { string_value: "2048" }
}
```

---

---

## 6. Scaling & Deployment

### Kubernetes Native
- Official Helm charts
- Horizontal Pod Autoscaler support
- Integration with Istio/Envoy for traffic management

### Multi-Instance Serving
- Multiple model instances per GPU
- Rate limiting and priority queues
- Request routing based on model version

---

---

## 7. Metrics & Observability

**Comprehensive Monitoring:** <br>
- Prometheus metrics (latency, throughput, queue depth)
- Per-model and per-version metrics
- GPU utilization tracking
- Inference count, batch statistics

**Tracing:** <br>
- OpenTelemetry support
- Request-level tracing through pipeline stages

---

---

## 8. Performance Optimization

### Concurrent Model Execution
- Multiple models on same GPU (if memory allows)
- Scheduler balances execution
- Useful for A/B testing

### Instance Groups
- Multiple instances of same model
- Load balancing across instances
- Can specify different GPUs per instance

---

---

## 9. Interview Q&A

**Q: When to use Triton over vLLM/TGI?** <br>
A: When you need multi-framework support, complex model pipelines, or enterprise features (model versioning, ensembles). For pure LLM serving, vLLM/TGI are simpler.

---

**Q: How does Triton's vLLM backend differ from standalone vLLM?** <br>
A: Same core engine, but Triton adds: model versioning, ensemble pipelines, enterprise monitoring, multi-framework support. Trade-off: extra abstraction layer with slight overhead.

---

**Q: What's the benefit of model ensembles?** <br>
A: Single API call for multi-stage pipelines. Triton handles scheduling, batching, and data passing between stages. Reduces latency vs multiple network hops.

---

**Q: How does dynamic batching work in Triton?** <br>
A: Accumulates requests for up to max_batch_size or max_delay_ms. Simpler than continuous batching (no iteration-level scheduling). Better for CV/audio models than LLMs.

---

**Q: Why use Triton for LLMs when vLLM exists?** <br>
A: Multi-model serving (embeddings + LLM + reranker), existing NVIDIA infrastructure, need for A/B testing across model versions, enterprise governance requirements.

---

**Q: How does Triton handle model updates?** <br>
A: Model repository polling detects new versions. Can load new version without stopping server. Traffic routing supports gradual rollout (e.g., 90% v1, 10% v2).

---