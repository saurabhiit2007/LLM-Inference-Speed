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
