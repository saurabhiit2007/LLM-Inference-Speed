## 1. Overview

PagedAttention is a memory management technique for efficient LLM serving that stores KV cache in non-contiguous memory blocks (pages). It's the core innovation behind vLLM, enabling **near-zero waste** in KV cache memory and higher throughput.

**Key insight:** Treat KV cache like virtual memory in operating systems—use paging to eliminate fragmentation and enable flexible memory sharing.

---

---

## 2. The KV Cache Memory Problem

### What is KV Cache?

In autoregressive generation, transformers reuse Key and Value tensors from previous tokens:
- Without cache: Recompute K, V for all previous tokens at each step (wasteful)
- With cache: Store K, V tensors and only compute for new token

For a sequence of length $N$ with $L$ layers, $H$ heads, and dimension $d$:
$$
\text{KV cache size} = 2 \times N \times L \times H \times d
$$

**Example:** LLaMA-13B with 2048 tokens ≈ 800 MB per sequence

---

### Problems with Traditional KV Cache

#### Problem 1: Memory Fragmentation
**Issue:** Must pre-allocate contiguous memory for maximum sequence length

```
Sequence 1: [████████░░░░] (800 tokens, allocated for 2048)
Sequence 2: [███░░░░░░░░░] (300 tokens, allocated for 2048)
Sequence 3: [██████░░░░░░] (600 tokens, allocated for 2048)
```

- Actual usage: 1700 tokens
- Allocated: 6144 slots (3 × 2048)
- **Waste: 72% of memory unused!**

---

#### Problem 2: No Memory Sharing
- Cannot share KV cache across sequences (even with identical prompts)
- Parallel sampling requires duplicating entire cache
- Beam search creates multiple copies

---

#### Problem 3: Static Allocation
- Must allocate for worst case (max sequence length)
- Can't dynamically adjust based on actual needs
- Limits batch size and throughput

---

---

## 3. How PagedAttention Works

### 3.1 Core Concept: Paging

Divide KV cache into fixed-size **blocks** (pages), similar to OS virtual memory:

```
Logical sequence: [Token 0, Token 1, ..., Token N]
                         ↓
Physical memory:  [Block 0] → [Block 5] → [Block 2] (non-contiguous)
```

**Key properties:**
- Block size: Typically 16-64 tokens
- Blocks can be anywhere in physical memory
- Mapping tracked via **block table** (like OS page table)

---

### 3.2 Block Table

Each sequence has a block table mapping logical blocks to physical blocks:

```
Sequence 1:
  Logical Block 0 → Physical Block 3
  Logical Block 1 → Physical Block 7
  Logical Block 2 → Physical Block 1

Sequence 2:
  Logical Block 0 → Physical Block 3  (shared with Seq 1!)
  Logical Block 1 → Physical Block 9
```

---

### 3.3 Dynamic Allocation

Blocks are allocated **on-demand** as sequences grow:

```python
# Conceptual allocation
def generate_token(sequence):
    if sequence.last_block_is_full():
        new_block = allocate_free_block()
        sequence.block_table.append(new_block)
    
    # Compute attention using block table
    output = paged_attention(Q, sequence.block_table)
    return output
```

**Benefits:**
- Only allocate what's actually used
- No pre-allocation for max length
- Memory freed immediately when sequence completes

---

### 3.4 Memory Sharing via Copy-on-Write

Multiple sequences can share blocks (read-only):

```
Prompt: "Translate to French: "
         ↓
[Block 0: "Translate to French: "] ← Shared by all sequences

Seq 1: [Block 0] → [Block 3: "Hello → "]
Seq 2: [Block 0] → [Block 5: "Goodbye → "]
```

When modifying a shared block → **copy-on-write**:
1. Allocate new physical block
2. Copy contents
3. Update block table
4. Modify the copy

---

---

## 4. Attention Computation with Paging

Standard attention accesses KV cache contiguously. PagedAttention accesses via block table:

```python
# Simplified PagedAttention
def paged_attention(Q, block_table, K_blocks, V_blocks):
    output = 0
    for logical_idx, physical_idx in enumerate(block_table):
        # Fetch K, V from physical block
        K_block = K_blocks[physical_idx]
        V_block = V_blocks[physical_idx]
        
        # Compute attention for this block
        scores = Q @ K_block.T / sqrt(d)
        attn = softmax(scores)
        output += attn @ V_block
    
    return output
```

**Key insight:** The indirection (block table lookup) has minimal overhead compared to memory savings.

---

---

## 5. Performance Impact

### Memory Efficiency
- **Traditional:** 20-40% KV cache utilization (60-80% waste)
- **PagedAttention:** 80-95% utilization (5-20% waste)
- **2-3x more sequences** in same memory

### Throughput Improvement
- vLLM with PagedAttention: **2-4x higher throughput** vs traditional serving
- Batch size limited by memory → bigger batches with less waste

### Latency
- Minimal overhead from block table lookups (<5%)
- Often better latency due to higher batch efficiency

---

---

## 6. Interview Questions

### Q1: What problem does PagedAttention solve?
**Answer:** PagedAttention solves memory fragmentation and waste in KV cache management. Traditional approaches pre-allocate contiguous memory for max sequence length, wasting 60-80% of memory. PagedAttention uses non-contiguous blocks allocated on-demand, achieving 80-95% utilization and enabling 2-4x higher throughput.

---

### Q2: How is PagedAttention similar to OS virtual memory?
**Answer:** Both use paging:
- **Virtual memory:** Maps virtual addresses to physical pages via page table
- **PagedAttention:** Maps logical KV cache positions to physical blocks via block table

Both enable non-contiguous allocation, on-demand paging, and copy-on-write sharing.

---

### Q3: What's a typical block size and why?
**Answer:** Typically **16-64 tokens**. Trade-offs:
- **Too small:** High block table overhead, more lookups during attention
- **Too large:** Internal fragmentation (wasted space within partially-filled blocks)
- **Sweet spot:** 16-64 balances overhead vs. fragmentation (similar to OS page sizes like 4KB)

---

### Q4: How does PagedAttention enable memory sharing?
**Answer:** Multiple sequences can point to the same physical blocks (read-only). Common use cases:
- **Shared prompts:** All sequences share blocks containing the same prompt
- **Parallel sampling:** Multiple outputs from same prompt share prefix blocks
- **Beam search:** Different beams share common prefix

When a shared block needs modification → **copy-on-write**: allocate new block, copy contents, update that sequence's block table.

---

### Q5: What's the overhead of block table lookups?
**Answer:** Minimal (<5% typically) because:
- Block tables are small (fits in cache)
- Lookups are simple integer indexing
- Attention computation dominates (matrix ops on blocks)
- Modern GPUs handle indirection efficiently

The memory savings far outweigh this small overhead.

---

### Q6: How does PagedAttention improve throughput?
**Answer:** By reducing memory waste:
1. Traditional: Can fit 10 sequences (60% waste)
2. PagedAttention: Can fit 25 sequences (10% waste) in same memory
3. Bigger batches → better GPU utilization → higher throughput

Typical improvement: **2-4x** more requests/second.

---

### Q7: What happens when we run out of physical blocks?
**Answer:** Memory management strategies:
- **Preemption:** Evict lower-priority sequences, save their state
- **Swapping:** Move blocks to CPU memory (like OS swap)
- **Recomputation:** Drop blocks and recompute if needed
- **Blocking:** Wait until blocks free up

vLLM typically uses preemption for fairness and efficiency.

---

### Q8: Can PagedAttention work with FlashAttention?
**Answer:** Yes! They're complementary:
- **FlashAttention:** Optimizes attention computation (tiling, kernel fusion)
- **PagedAttention:** Optimizes KV cache memory management (paging, sharing)

You can use both together: FlashAttention for fast computation, PagedAttention for efficient memory. vLLM does exactly this.

---

### Q9: What's the difference between block size and tile size?
**Answer:**
- **Block size (PagedAttention):** Memory management granularity (16-64 tokens)
  - Determines allocation unit for KV cache storage
- **Tile size (FlashAttention):** Computation granularity (128-256 tokens)
  - Determines how much data loads into shared memory at once

They're independent concepts operating at different levels (memory management vs computation).

---

### Q10: What are the limitations of PagedAttention?
**Answer:**
- **Complexity:** More complex implementation than contiguous allocation
- **Indirection overhead:** Small cost from block table lookups
- **GPU kernel changes:** Requires custom attention kernels that understand block tables
- **Internal fragmentation:** Last block in sequence may be partially empty

Despite these, benefits (2-4x throughput) far outweigh costs for LLM serving.

---

---

## 7. PagedAttention in Practice (vLLM)

### Key Features
```python
# vLLM with PagedAttention
from vllm import LLM

llm = LLM(model="meta-llama/Llama-2-7b")

# Automatic memory management
outputs = llm.generate(prompts, sampling_params)
# - Blocks allocated on-demand
# - Shared prompts reuse blocks
# - Memory freed automatically
```

---

### Use Cases Where It Shines
✅ High-throughput serving (many concurrent requests)  
✅ Long sequences (less pre-allocation waste)  
✅ Parallel sampling / beam search (shared prefixes)  
✅ Shared system prompts across requests  

❌ Single-sequence inference (no sharing benefits)  
❌ Very short sequences (overhead not amortized)  

---

---

## 8. Key Takeaways for Interviews

1. **Main idea:** Treat KV cache like OS virtual memory—use paging for efficient, flexible allocation
2. **Problem solved:** Memory fragmentation (60-80% waste → 5-20% waste)
3. **Mechanism:** Block table maps logical positions to physical memory blocks
4. **Sharing:** Copy-on-write enables multiple sequences to share read-only blocks
5. **Performance:** 2-4x throughput improvement in LLM serving
6. **Complementary:** Works alongside FlashAttention (computation vs memory optimization)

---

---

## 9. Comparison Table

| Aspect | Traditional KV Cache | PagedAttention |
|--------|---------------------|----------------|
| **Allocation** | Contiguous, pre-allocated | Non-contiguous, on-demand |
| **Memory waste** | 60-80% | 5-20% |
| **Max sequences** | Limited by pre-allocation | 2-4x more in same memory |
| **Sharing** | No sharing | Copy-on-write sharing |
| **Complexity** | Simple | More complex |
| **Overhead** | None | <5% (block lookups) |
| **Throughput** | Baseline | 2-4x higher |

---