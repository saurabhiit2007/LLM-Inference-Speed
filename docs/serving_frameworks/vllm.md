# vLLM

**Optimization axis: throughput via memory efficiency**

vLLM is the most widely adopted open-source LLM serving framework. Its central contribution is PagedAttention — a rethink of how GPU memory is allocated for the KV cache that unlocks dramatically larger batch sizes.

---

## 1. The Problem It Was Built to Solve

Before vLLM, every serving engine pre-allocated a contiguous block of GPU memory for each request's KV cache — sized for the *maximum possible* sequence length. This caused two forms of waste:

- **Reservation waste:** a request generating 200 tokens still holds memory reserved for 2048
- **Fragmentation:** as requests finish at different times, the freed blocks cannot be recombined for new requests of different sizes

The result: 60–80% of KV cache memory sat unused at any given moment, capping the number of concurrent requests and, therefore, throughput.

---

## 2. Core Insight: PagedAttention

PagedAttention borrows the OS virtual memory idea and applies it to the KV cache.

Instead of one contiguous allocation per request, the KV cache is divided into fixed-size **blocks (pages)** — typically 16 tokens each. A **block table** maps each request's logical KV positions to non-contiguous physical blocks, exactly like a page table in an OS.

```
Request A:  [Block 3] → [Block 7] → [Block 12]   (scattered, but logically contiguous)
Request B:  [Block 1] → [Block 9]
Request C:  [Block 2] → [Block 5] → [Block 8] → [Block 11]
```

**Result:** memory waste drops to under 4%, fitting 2–3× more concurrent sequences on the same GPU. Higher concurrency means more tokens generated per second.

---

## 3. Architecture

### Continuous batching

vLLM schedules at the *iteration* level, not the request level. As soon as one sequence finishes, the freed KV blocks are immediately reassigned and a new request is added to the batch — no GPU idle time between requests.

### Automatic Prefix Caching (APC)

KV blocks for shared prefixes (system prompts, RAG contexts) are hashed and reused across requests. If 100 concurrent requests share the same 500-token system prompt, that prompt's KV blocks are computed once. This changes the economics of RAG and multi-turn chat dramatically.

### Multi-LoRA serving

vLLM can serve a base model plus hundreds of LoRA adapters simultaneously using SGMV (Segmented Gather-Scatter Matrix-Vector) kernels that batch computation across different adapters. A single GPU can handle multi-tenant deployments where each tenant has a fine-tuned adapter.

### Chunked prefill

Long prompts are split into chunks and interleaved with decode steps, preventing a single large prefill from stalling all decode operations on co-batched requests.

### Memory pressure handling

When the GPU runs out of free blocks, vLLM either **swaps** KV blocks to CPU RAM or **recomputes** them later. On modern GPUs with high compute-to-bandwidth ratios, recomputation is often faster than the PCIe transfer.

---

## 4. Tradeoffs

| | |
|---|---|
| **Python overhead** | vLLM's scheduler runs in Python; TensorRT-LLM's compiled CUDA graphs are faster at equivalent concurrency |
| **No grammar constraints** | Structured output requires external tools (Outlines, Guidance) |
| **Multi-node TP** | Tensor parallelism via Ray works but has more overhead than TensorRT-LLM's custom NCCL ops |

---

## 5. When to Use

**Use vLLM when:** you need maximum throughput, multi-tenant serving, or multi-LoRA support and want minimal setup complexity. It is the default production choice for most teams.

**Don't use vLLM when:** you need the absolute lowest per-token latency on fixed NVIDIA hardware and can accept a 20–30 minute engine build cycle — use TensorRT-LLM instead.
