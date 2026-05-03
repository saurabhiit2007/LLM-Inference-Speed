## 1. Core Concepts

### Latency

- **Time to complete a single request**
- Measured in seconds or milliseconds
- Critical for interactive applications (chatbots, code completion)
- Key metrics: TTFT, TPOT, E2E latency

---

### Throughput

- **Number of requests processed per unit time**
- Measured in tokens/sec or requests/sec
- Critical for batch processing, high-traffic services
- Maximize GPU utilization

---

---

## 2. The Fundamental Tradeoff

```
Latency ↑ as Throughput ↑

Higher batch size → Higher throughput, Higher latency per request
Lower batch size → Lower latency, Lower throughput
```

### Why They Conflict

**Batching Increases Throughput**

- Process multiple requests simultaneously
- Better GPU utilization (more parallel work)
- Amortize weight loading overhead

**But Hurts Latency**

- Requests wait for entire batch to complete
- Queueing delays increase
- Stragglers slow down entire batch

---

---

## 3. Key Metrics

```
Latency = Queue Time + Processing Time
Throughput = Batch Size / Processing Time (ignoring queue)

Utilization = (Actual Throughput) / (Max Theoretical Throughput)
```

---

---

## 4. Optimization Strategies

### For Low Latency (< 100ms)

**Batch Size = 1 or Small**

- Minimize queueing delay
- Accept lower GPU utilization
- Use smaller models (7B vs 70B)
- Quantization (INT8/INT4) for faster decode

**Prefill Optimization**

- FlashAttention for faster attention
- Tensor parallelism to split model across GPUs

**Infrastructure**

- Low-latency network
- GPU with high memory bandwidth (H100 > A100)
- Close to users (edge deployment)

---

### For High Throughput

**Large Batch Sizes**

- Batch 32-128+ requests
- Maximize GPU compute utilization
- Accept seconds of latency per request

**Continuous Batching**

- Don't wait for all sequences to finish
- Insert new requests as others complete
- Used by vLLM, TensorRT-LLM

**Paged Attention (vLLM)**

- Reduce memory fragmentation
- Pack more sequences in memory
- Enable larger effective batch size

**Chunked Prefill**

- Split long prefills into chunks
- Interleave with decode steps
- Balance latency and throughput

---

---

## 5. Hardware Considerations

### A100 (80GB)

- 1,935 GB/s memory bandwidth
- Good for batch inference
- Throughput: ~2000 tokens/sec (LLaMA-2-7B, batch=32)

### H100 (80GB)

- 3,350 GB/s memory bandwidth (1.7x A100)
- Better for both latency and throughput
- FlashAttention-3 support
- Throughput: ~3500 tokens/sec (same setup)

### L40S / L4

- Lower cost, lower bandwidth
- Good for latency-optimized serving (small batch)
- Not ideal for high throughput

---
