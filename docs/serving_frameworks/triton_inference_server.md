# Triton Inference Server

**Optimization axis: multi-model orchestration and enterprise serving infrastructure**

Triton is NVIDIA's framework-agnostic inference server. The critical mental model: **Triton is not an LLM inference engine**. It is a router and orchestration layer that wraps existing engines — TensorRT-LLM, vLLM, ONNX Runtime, PyTorch — behind a unified HTTP/gRPC API with production observability built in.

---

## 1. The Problem It Was Built to Solve

A production AI application rarely runs a single model. A RAG pipeline, for example, involves an embedding model, a retriever, an optional reranker, the LLM itself, and sometimes a postprocessor. Without an orchestration layer, each stage is a separate server with its own endpoint, requiring the client to chain them — multiple network round-trips, no shared scheduling, no unified observability.

A second problem: enterprise deployments need model versioning (run v1 and v2 simultaneously for A/B testing), multi-framework support (some models in TensorFlow, others in PyTorch), and a single monitoring surface across all models.

Triton was built to provide this orchestration layer independently of which inference backend is doing the actual computation.

---

## 2. Core Insight: The Backend System

Triton's architecture separates *serving concerns* from *execution concerns* through a pluggable backend system.

```
Client (HTTP/gRPC)
        ↓
  Triton Server
  ┌─────────────────────────────────┐
  │  Request routing & scheduling   │
  │  Model repository management    │
  │  Metrics & observability        │
  └──────┬──────┬──────┬────────────┘
         ↓      ↓      ↓
   TensorRT  vLLM   ONNX    (backends)
    engine  engine  Runtime
```

Each backend handles execution for a specific framework. Triton handles everything else: HTTP/gRPC protocol, request queuing, batching policy, metrics export, version management, and health checks. You swap the backend without changing your client code.

### Ensemble pipelines

Triton can wire multiple models into a single endpoint using an ensemble configuration. A client sends one request; Triton routes it through preprocessing → embedding model → LLM → postprocessing sequentially, with each stage's output becoming the next stage's input. The entire pipeline is a single logical model from the client's perspective.

```
Single API call → [Tokenizer → LLM → Detokenizer] → response
                   (Python)   (TRT-LLM) (Python)
```

This eliminates multi-hop latency and simplifies client code.

### Model repository and versioning

Triton reads from a structured model repository (local filesystem, S3, or GCS). Each model directory contains numbered version subdirectories. Triton can serve multiple versions simultaneously and apply traffic policies (e.g., 90% to v2, 10% to v3 for canary rollouts) without server restarts.

```
model_repository/
  llama-3-8b/
    1/        ← v1 engine
    2/        ← v2 engine (active)
    config.pbtxt
```

### LLM backends (2024–2025)

Triton added first-class backends for both TensorRT-LLM and vLLM. The TensorRT-LLM backend gives maximum NVIDIA performance; the vLLM backend provides PagedAttention and multi-LoRA support. Both expose the same Triton API, so the orchestration layer is identical regardless of which engine is running underneath.

```protobuf
backend: "vllm"
max_batch_size: 32

instance_group [{ count: 1  kind: KIND_GPU }]

parameters: { key: "max_tokens"  value: { string_value: "2048" } }
```

---

## 3. Observability

Triton exports Prometheus metrics covering queue depth, batch size, per-model throughput, GPU utilization, and request latency — at the per-model and per-version granularity. OpenTelemetry tracing shows request-level timing through each pipeline stage.

This observability is the primary reason to use Triton over running vLLM directly in enterprise contexts: a single monitoring surface across all models, not per-model dashboards.

---

## 4. Tradeoffs

| | |
|---|---|
| **Not a standalone engine** | Triton adds no LLM inference capability on its own; it requires a backend (TensorRT-LLM, vLLM) configured separately |
| **Operational complexity** | config.pbtxt files, model repository structure, and backend-specific configuration add setup overhead vs. running vLLM directly |
| **Added latency** | The abstraction layer adds a small overhead; negligible for LLMs but measurable for low-latency embedding or CV models |
| **Overkill for single-model serving** | If you're serving one LLM and don't need versioning or pipelines, vLLM with its own API server is simpler |

---

## 5. When to Use

**Use Triton when:** you need to serve multiple models (e.g., embedding + LLM + reranker) as a unified pipeline, require model versioning and A/B testing, or are operating in an enterprise context that requires unified observability and Kubernetes-native deployment.

**Don't use Triton when:** you're serving a single LLM and don't need enterprise orchestration features — the setup overhead isn't justified. Run vLLM directly instead.

**Common production pattern:** TensorRT-LLM as the inference backend (for maximum token throughput) with Triton as the HTTP/gRPC frontend (for model versioning, metrics, and Kubernetes integration). This pairing is NVIDIA's recommended enterprise deployment stack.
