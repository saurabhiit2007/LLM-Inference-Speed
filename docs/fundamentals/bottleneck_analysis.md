## 1. Understanding Bottlenecks

### The Three Primary Bottlenecks

**1. Compute-Bound**
- GPU cores underutilized
- Not enough arithmetic operations
- Common in: Prefill phase, large batches

**2. Memory-Bound**
- GPU cores waiting for data
- Memory bandwidth saturated
- Common in: Decode phase, small batches

**3. Overhead-Bound**
- Framework/system overhead dominates
- Kernel launch latency
- Common in: Very small models, batch=1

---

---

## 2. Roofline Model

```
Attainable Performance = min(Peak Compute, Arithmetic Intensity × Memory Bandwidth)

Arithmetic Intensity = FLOPs / Bytes Transferred

If Arithmetic Intensity < Compute/Bandwidth ratio → Memory-Bound
If Arithmetic Intensity > Compute/Bandwidth ratio → Compute-Bound
```

---

### Example: H100 GPU
```
Peak FP16 Compute: 989 TFLOPS
Memory Bandwidth: 3,350 GB/s
Ratio: 295 FLOP/Byte

Operation with AI=100 FLOP/Byte → Memory-bound
Operation with AI=500 FLOP/Byte → Compute-bound
```

---

---

## 3. Identifying Bottlenecks

### Method 1: GPU Utilization Metrics

**Compute Utilization**

```
nvidia-smi dmon -s u
# SM (Streaming Multiprocessor) utilization

High SM% (>80%) → Compute-bound
Low SM% (<40%) → Memory or overhead-bound
```

**Memory Utilization**

```
nvidia-smi dmon -s m
# Memory bandwidth utilization

High Mem% (>80%) → Memory-bound
Low Mem% (<40%) → Compute or overhead-bound
```

---

### Method 2: Profiling Tools

**NVIDIA Nsight Compute**

```bash
ncu --set full -o profile python inference.py
```
- Shows compute vs memory bottleneck per kernel
- Identifies optimization opportunities

**PyTorch Profiler**

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(input)
    
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Key Metrics to Check**:

- Kernel time distribution
- Memory copy overhead
- CPU-GPU sync points

---

### Method 3: Microbenchmarks

**Isolate Operations**

```python
# Test prefill vs decode separately
prefill_time = benchmark_prefill(prompt_tokens)
decode_time = benchmark_decode(num_output_tokens)

# Test different batch sizes
for batch_size in [1, 4, 8, 16, 32]:
    throughput[batch_size] = benchmark(batch_size)
```

**Expected Results**:

- Decode: Throughput plateaus early → Memory-bound
- Prefill: Throughput scales with batch → Compute-bound

---

---

## 4. Common Bottleneck Patterns

### Pattern 1: Decode Phase (Memory-Bound)

**Symptoms**:

- Low GPU compute utilization (20-40%)
- High memory bandwidth usage
- TPOT doesn't improve with smaller model quantization

**Root Cause**:

```
Single token generation = Load entire weight matrix
Arithmetic Intensity ≈ 1-2 FLOP/Byte (very low)
```

**Solutions**:

- Weight quantization (INT8/INT4) → Reduce bytes transferred
- Increase batch size → Amortize weight loading
- Use higher memory bandwidth GPU (H100 vs A100)
- Speculative decoding → Generate multiple tokens

---

### Pattern 2: Prefill Phase (Compute-Bound)

**Symptoms**:

- High GPU compute utilization (70-90%)
- Attention computation dominates
- Scales well with batch size

**Root Cause**:

```
Attention: O(n²d) operations
Long sequences = Quadratic compute growth
```

**Solutions**:

- FlashAttention → Fused kernel, reduce memory access
- Tensor parallelism → Split across GPUs
- Reduce sequence length if possible
- Use models with sliding window attention (Mistral)

---

### Pattern 3: KV Cache Transfer (Memory-Bound)

**Symptoms**:

- Performance degrades with sequence length
- Memory copy time visible in profiler

**Root Cause**:

```
KV cache size = 2 × seq_len × layers × heads × dim × bytes
Long sequences = Large cache to copy
```

**Solutions**:

- GQA/MQA → Reduce KV cache size
- KV cache quantization (INT8) → 2x reduction
- Paged attention (vLLM) → Better memory management

---

### Pattern 4: Kernel Launch Overhead

**Symptoms**:

- Low utilization despite small workload
- Many small kernels in profiler
- Performance doesn't scale with model size

**Root Cause**:

```
Each operation launches separate kernel
Overhead: ~5-20μs per kernel launch
```

**Solutions**:

- Kernel fusion (FlashAttention, torch.compile)
- Larger batch sizes
- Use CUDA graphs → Eliminate launch overhead

---

### Pattern 5: CPU-GPU Synchronization

**Symptoms**:

- GPU idle time between operations
- High "cudaDeviceSynchronize" time
- Low pipeline parallelism

**Root Cause**:

```
Explicit sync points or implicit Python overhead
GPU waits for CPU to issue next operation
```

**Solutions**:

- Asynchronous operations (CUDA streams)
- Reduce Python overhead (torch.compile, C++ inference)
- Pipeline parallelism

---

---

## 5. Systematic Analysis Framework

### Step 1: Measure Baseline
```
Metrics to collect:
- Total latency (TTFT + decode time)
- Tokens per second (throughput)
- GPU utilization (SM%, Mem%)
- Memory usage (weights, KV cache, activations)
```

### Step 2: Profile Critical Path
```
Use profiler to identify:
1. Which operations take most time?
2. Are they compute or memory-bound?
3. Where are sync points?
```

### Step 3: Apply Targeted Optimizations
```
If memory-bound → Reduce data movement
If compute-bound → Optimize kernels or reduce ops
If overhead-bound → Fuse kernels or increase batch
```

### Step 4: Validate Improvement
```
Measure again and compare
Check for regressions in quality
Ensure optimization applies to production workload
```

---

---

## 6. Profiling Example: LLaMA-2-7B

### Baseline (Batch=1, Seq=512)
```
Operation          | Time (ms) | % Total | Bottleneck
-------------------|-----------|---------|------------
Attention          | 8.2       | 45%     | Memory
FFN                | 6.5       | 35%     | Memory
Layer Norm         | 1.8       | 10%     | Overhead
KV Cache Update    | 1.2       | 7%      | Memory
Misc               | 0.5       | 3%      | -
-------------------|-----------|---------|------------
Total              | 18.2      | 100%    | Memory-bound
```

---

### After Optimization
```
Applied: FlashAttention, INT8 quantization, kernel fusion

Operation          | Time (ms) | % Total | Change
-------------------|-----------|---------|--------
Attention (Flash)  | 4.1       | 40%     | -50%
FFN (INT8)         | 3.8       | 37%     | -42%
Layer Norm (fused) | 0.9       | 9%      | -50%
KV Cache Update    | 1.0       | 10%     | -17%
Misc               | 0.4       | 4%      | -20%
-------------------|-----------|---------|--------
Total              | 10.2      | 100%    | -44%
```

---

---

## 7. Common Interview Questions

**Q: How do you determine if inference is compute or memory-bound?**

```
1. Check GPU metrics (SM% vs Mem%)
2. Profile with Nsight Compute (SOL Compute vs SOL Memory)
3. Test batch size scaling:
   - Compute-bound: Scales well with batch
   - Memory-bound: Plateaus quickly
4. Calculate arithmetic intensity vs hardware ratio
```

---

**Q: GPU shows 100% utilization but throughput is low. Why?**

- Could be memory-bound (100% memory utilization)
- Check if memory bandwidth saturated
- Verify you're looking at the right metric (compute vs memory)
- Could be inefficient kernels (high utilization, low throughput)

---

**Q: Describe how you'd optimize a memory-bound decode phase**

```
1. Profile to confirm bottleneck (low SM%, high Mem%)
2. Quantize weights (INT8) → 2x less data to transfer
3. Increase batch size → Better memory bandwidth utilization
4. Use H100 instead of A100 → 1.7x more bandwidth
5. Consider speculative decoding → Reduce number of decode steps
```

---

**Q: What's the impact of FlashAttention on prefill vs decode?**

```
Prefill (Compute-bound):
- Reduces memory access (no full attention matrix)
- Enables longer sequences without OOM
- 2-4x speedup typical

Decode (Memory-bound):
- Smaller benefit (already memory-bound on weights)
- Still helpful for very long context
- ~20-30% speedup
```

---

**Q: How do you profile Python overhead vs GPU computation?**

```python
# Method 1: Compare with/without CUDA sync
import time

# With sync (includes Python overhead)
t0 = time.time()
output = model(input)
torch.cuda.synchronize()
t1 = time.time()

# With events (pure GPU time)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = model(input)
end.record()
torch.cuda.synchronize()
gpu_time = start.elapsed_time(end)

python_overhead = (t1 - t0) - (gpu_time / 1000)
```

---

**Q: Explain the roofline model for LLM inference**

```
Roofline: Performance = min(Compute Peak, Bandwidth × Arithmetic Intensity)

Example: Decode single token on H100
- Matmul: [1, 4096] × [4096, 4096]
- FLOPs: 2 × 1 × 4096 × 4096 ≈ 33M
- Bytes: (4096×4096 + 4096×4096) × 2 (FP16) ≈ 67MB
- AI: 33M / 67M ≈ 0.5 FLOP/Byte

H100: 989 TFLOPS, 3350 GB/s → 295 FLOP/Byte ratio
AI (0.5) << Ratio (295) → Memory-bound
```

---

---

## 8. Advanced Profiling

### Tensor Core Utilization
```
ncu --metrics sm__sass_thread_inst_executed_op_dmma_inst,sm__sass_thread_inst_executed_op_hmma_inst

Check if matmuls use tensor cores (TC)
Low TC usage → Not using FP16/BF16 or improper dims
```

---

### Memory Transaction Efficiency
```
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

Efficiency = Sectors / (Requests × 4)
Low efficiency → Uncoalesced memory access
```

---

---

## 9. Key Takeaways

1. **Always profile before optimizing** - Don't guess the bottleneck
2. **Different phases have different bottlenecks**: Prefill (compute), Decode (memory)
3. **Use the right metric**: SM% for compute, Mem% for memory
4. **Batch size is a key diagnostic**: Scaling behavior reveals bottleneck
5. **Optimization must target the actual bottleneck**: Memory optimization won't help compute-bound workload
6. **Modern GPUs shift bottlenecks**: H100's higher compute/bandwidth ratio changes optimization strategy

---