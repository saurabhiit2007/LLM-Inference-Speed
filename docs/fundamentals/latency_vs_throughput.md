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

## 5. Request-Level Batching Strategies

### Static Batching
- Wait for batch to fill before processing
- Simple but high latency variance
- Wasted time if batch doesn't fill

### Continuous Batching
```
t=0: Start batch [A, B, C]
t=1: A finishes → add D → [B, C, D]
t=2: B finishes → add E → [C, D, E]
```

- Dynamic batch composition
- Much better GPU utilization
- Lower average latency

### Priority Queuing
- Process short/urgent requests first
- Separate queues for interactive vs batch
- SLO-aware scheduling

---

---

## 6. Hardware Considerations

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

---

## 7. Common Interview Questions

**Q: You have 1000 QPS (queries per second). Optimize for p99 latency < 200ms. How?**

- Use continuous batching (vLLM)
- Target small effective batch (4-8)
- Replica scaling with load balancer
- Monitor queue depth, scale if needed

---

**Q: Batch size 1 vs 32: compare latency and throughput**

```
Batch=1:
- Latency: ~50ms
- Throughput: ~20 tokens/sec
- GPU utilization: ~15%

Batch=32:
- Latency: ~800ms (includes queueing)
- Throughput: ~500 tokens/sec
- GPU utilization: ~80%
```

---

**Q: How does continuous batching improve over static?**

- No waiting for batch to fill
- No wasted cycles when sequences finish at different times
- Typically, 2-3x better throughput at similar latency

---

**Q: When would you choose latency over throughput?**

- Real-time chat applications
- Code completion (100-200ms target)
- Interactive agents
- Premium API tiers

---

**Q: When would you choose throughput over latency?**

- Offline batch processing
- Data labeling/annotation
- Embedding generation
- Document summarization at scale

---

---

## 8. Production Patterns (2024-2025)

### Multi-Tier Serving
```
Tier 1 (Latency): Small models, batch=1-4, edge deployment
Tier 2 (Balanced): Medium models, continuous batching, batch=8-16
Tier 3 (Throughput): Large models, large batches, datacenter
```

---

### Speculative Decoding
- Draft model generates multiple tokens
- Target model verifies in parallel
- 2-3x speedup with same latency
- Best for latency-sensitive scenarios

---

### Disaggregated Serving (Splitwise)
- Separate prefill and decode clusters
- Prefill: GPU compute optimized (A100)
- Decode: Memory bandwidth optimized (H100)
- Transfer KV cache between clusters

---

---

## 9. Key Metrics to Monitor

```
P50, P95, P99 Latency - Distribution matters
Throughput (tokens/sec) - Absolute capacity
Queue Depth - Leading indicator of overload
GPU Utilization - Efficiency metric
Cost per 1M tokens - Business metric
```

---

---

## 10. Benchmarking Tips

- Measure real production traffic patterns
- Include cold start times if relevant
- Test at different concurrency levels
- Monitor long-tail latency (p99, p99.9)
- Account for sequence length variance

---

---

## 11. Key Takeaways

1. Latency and throughput are inversely related via batching
2. Continuous batching is standard for production (vLLM, TRT-LLM)
3. Different use cases need different optimization targets
4. Hardware choice matters: H100 better for both metrics vs A100
5. Monitor distributions (p99), not just averages

---