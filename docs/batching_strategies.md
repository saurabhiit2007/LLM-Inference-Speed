## 1. Why Batching Matters

**GPU Utilization Problem:** <br>
- Single request uses <10% of GPU compute capacity
- Memory bandwidth underutilized
- Expensive hardware sitting idle

**Batching Solution:** <br>
- Process multiple requests simultaneously
- Amortize memory access costs
- Achieve 5-10x throughput improvement

**Key Constraint:** GPU memory limits batch size (primarily KV cache)

---

---

## 2. Static Batching

### Mechanism
- Accumulate N requests
- Process entire batch together
- Wait for ALL sequences to complete before accepting new requests

---

### Characteristics

```
Batch: [Req1: 100 tokens, Req2: 20 tokens, Req3: 80 tokens]
All requests wait until Req1 (longest) completes
GPU idle during Req2, Req3 completion
```

**Throughput:** Improved vs single request  
**Latency:** High variance (tail latency problem)  
**GPU Utilization:** Poor (bubbles after short sequences finish)

---

### When to Use
- Offline batch processing
- Known sequence lengths
- Latency not critical

---

---

## 3. Dynamic Batching

### Mechanism
- Accumulate requests up to max_batch_size OR max_delay
- Whichever comes first triggers batch execution
- Still waits for full batch completion before next batch

---

### Key Parameters
```python
max_batch_size = 32      # Maximum requests per batch
max_delay_ms = 100       # Maximum wait time
```

**Improvement over Static:**
- Balances latency and throughput
- Reduces wait time for batch formation

**Still Limited:**
- Batch-level scheduling (not iteration-level)
- GPU idle time after short sequences complete

---

### When to Use
- Services with moderate traffic
- Simpler implementation than continuous batching
- Good for non-LLM models (CV, audio)

---

---

## 4. Continuous Batching (Iteration-Level Scheduling)

### Core Innovation
**Iteration-Level Scheduling:** Schedule at each decode step, not batch level

---

### Mechanism

```
Initial Batch: [Req1, Req2, Req3, Req4]

Iteration 1: All generate token 1
Iteration 2: All generate token 2
Iteration 3: Req2 completes → Req5 joins → [Req1, Req3, Req4, Req5]
Iteration 4: Req5, Req3 complete → Req6, Req7 join → [Req1, Req4, Req6, Req7]
...
```

**Key Benefit:** No GPU idle time, slots filled immediately

---

### Implementation Details

**Sequence Completion Detection:** <br>
- Monitor for EOS tokens
- Max length reached
- User cancellation

**Slot Management:** <br>
- Free KV cache blocks immediately
- Add waiting request to batch
- Update attention masks

**Memory Efficiency:** <br>
- Works best with PagedAttention (vLLM)
- Non-contiguous memory allocation
- Independent per-sequence management

---

### Performance Impact
- **Throughput:** +20-30% vs dynamic batching
- **Latency:** Lower average, more consistent tail latency
- **GPU Utilization:** 80-95% (vs 50-70% for static)

---

### Trade-offs
- Complex implementation
- Variable batch size per iteration
- Requires sophisticated memory management

---

---

## 5. Chunked Prefill

### Problem

**Long prompts block decode:** <br>
```
Prompt: 10,000 tokens (prefill) → 50 iterations
Short prompts: waiting in queue
Decode operations: starved
```

---

### Solution
Break prefill into chunks, interleave with decode

---

### Mechanism
```
Iteration 1: Chunk 1 of long prompt (512 tokens)
Iteration 2: Decode for ongoing requests
Iteration 3: Chunk 2 of long prompt (512 tokens)
Iteration 4: Decode for ongoing requests
...
```

**Parameters:**
- `max_prefill_tokens`: Tokens to prefill per iteration
- Balances prefill throughput vs decode latency

---

### Benefits
- Prevents large prompts from causing latency spikes
- Better tail latency for decode operations
- More predictable service times

---

### Implementation Challenges
- Partial KV cache management
- Attention mask complexity
- Scheduling overhead

---

---

## 6. Speculative Batching

### Concept
Combine speculative decoding with batching

---

### Mechanism
1. **Draft Phase:** Small model predicts k tokens for all requests in batch
2. **Verification Phase:** Large model verifies all k tokens in single pass
3. **Accept/Reject:** Keep correct tokens, retry from first error

---

### Batch-Level Optimization

```
Without Speculation: 5 iterations for 5 tokens
With Speculation (k=5): 1 iteration (if all accepted)
Effective Speedup: 2-3x for batch
```

---

### Challenges
- Variable acceptance rates across requests
- Synchronization points
- Draft model overhead

---

---

## 7. Prefix Caching in Batching

### Problem

Repeated prefixes waste computation

```
User1: [System Prompt] + User Query 1
User2: [System Prompt] + User Query 2
User3: [System Prompt] + User Query 3
```

---

### Solution
**Share KV cache blocks for common prefixes**

---

### Automatic Prefix Caching (APC)
- Hash prompt prefixes
- Reuse physical memory blocks
- Multiple requests point to same KV cache

---

### Batch-Level Benefits

```
Without APC:
  3 requests × 1000-token system prompt = 3000 tokens in KV cache

With APC:
  3 requests share 1000-token KV cache = 1000 tokens total
  Effective batch size: 3x larger
```

---

### Use Cases
- Multi-turn chat (common history)
- RAG (shared context documents)
- Agent frameworks (repeated tool descriptions)

---

---

## 8. Mixed Batch Scheduling

### Challenge
Different request types have different characteristics:

- **Prefill:** Compute-bound, parallel
- **Decode:** Memory-bound, sequential

---

### Naive Approach
Separate batches for prefill and decode → Underutilization

---

### Optimized Approach
**Mixed batches with resource allocation:** <br>

```
GPU Resources:
  80% compute → Prefill operations
  20% memory bandwidth → Decode operations
  
Single iteration:
  Process prefill chunks (compute-heavy)
  Process decode steps (memory-heavy)
  Overlap when possible
```

---

### SplitFuse (DeepSpeed-FastGen)
Dynamic algorithm that:
1. Monitors compute vs memory utilization
2. Adjusts prefill chunk size
3. Balances prefill/decode in each iteration

---

---

## 9. Batch Size Selection

### Factors

**Memory Constraint:** <br>

```
Available Memory = Model Weights + KV Cache + Activations
KV Cache = batch_size × seq_len × kv_memory_per_token
```

**Optimal Batch Size:** <br>
- Too small → GPU underutilized
- Too large → OOM, increased latency

---

### Heuristics

**For Decode:** <br>
```
optimal_batch_size = GPU_memory / (model_size + max_seq_len × kv_per_token)
```

**For Prefill:**
```
Limited by compute, not memory
Larger batches better (up to memory limit)
```

---

### Adaptive Batching

Monitor queue depth and adjust:

- High queue → Increase batch size (if memory allows)
- Low latency requirement → Decrease batch size
- Balance throughput vs latency SLO

---

---

## 10. Multi-Query Batching (MQA/GQA Context)

### Relevance to Batching

**Multi-Query Attention (MQA):**
- Fewer KV heads → Smaller KV cache
- Enables larger batch sizes

**Grouped-Query Attention (GQA):**
- Middle ground between MHA and MQA
- Llama 3, Mistral use GQA

---

### Impact

```
Llama 2 (MHA): 32 heads for K, 32 for V
Llama 3 (GQA): 8 KV heads, 32 Q heads

KV Cache Reduction: 4x
Batch Size Increase: ~4x at same memory
```

---

---

## 11. Interview Q&A

**Q: Why is continuous batching better than dynamic batching?** <br>
A: Dynamic batching schedules at batch level, causing GPU idle time when short sequences complete. Continuous batching schedules at iteration level, immediately filling freed slots with new requests. This eliminates bubbles and improves throughput by 20-30%.

---

**Q: What's the trade-off between batch size and latency?** <br>
A: Larger batches increase throughput but also increase latency per request (more compute per iteration, longer queue wait times). Optimal batch size depends on SLO requirements and whether you're optimizing for throughput or latency.

---

**Q: How does chunked prefill prevent latency spikes?** <br>
A: Without chunking, a 10k token prefill blocks the GPU for many iterations, starving decode operations. Chunked prefill breaks it into smaller pieces (e.g., 512 tokens) and interleaves with decode, keeping decode latency predictable.

---

**Q: Why does prefix caching improve effective batch size?** <br>
A: Common prefixes (system prompts, RAG contexts) are stored once in memory and shared across requests. If 10 requests share a 1k token prefix, you save 9k tokens of KV cache, allowing 9 more requests to fit in the same memory.

---

**Q: When would you use static batching instead of continuous batching?** <br>
A: Offline batch processing with known sequence lengths, where latency doesn't matter and implementation simplicity is valued. For production online serving, continuous batching is almost always better.

---

**Q: How does speculative decoding interact with batching?** <br>
A: Draft model generates k tokens for entire batch, then target model verifies all in parallel. Benefit multiplies across batch. Challenge is handling variable acceptance rates—some sequences may accept all k tokens while others reject after first token.

---

**Q: What determines the optimal prefill chunk size?** <br>
A: Balance between prefill throughput and decode latency. Smaller chunks (256-512 tokens) minimize decode latency impact but reduce prefill efficiency. Larger chunks (1024-2048) maximize prefill throughput but can cause latency spikes. Typically tuned based on workload mix.

---

**Q: Why does continuous batching require PagedAttention or similar?** <br>
A: Variable batch sizes per iteration need dynamic memory allocation. Traditional contiguous allocation can't handle requests joining/leaving mid-batch efficiently. Paged memory allows independent per-sequence management without fragmentation.

---
