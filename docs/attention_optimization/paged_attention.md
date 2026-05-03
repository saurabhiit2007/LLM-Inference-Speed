## 1. Overview

PagedAttention is a memory management technique for efficient LLM serving that stores KV cache in non-contiguous memory blocks (pages). It's the core innovation behind vLLM, enabling **near-zero waste** in KV cache memory and higher throughput.

**Key insight:** Treat KV cache like virtual memory in operating systems—use paging to eliminate fragmentation and enable flexible memory sharing.

---

---

## 2. The KV Cache Memory Problem

### 2.1 What is KV Cache?

In autoregressive generation, transformers reuse Key and Value tensors from previous tokens:

- Without cache: Recompute K, V for all previous tokens at each step (wasteful)
- With cache: Store K, V tensors and only compute for new token

For a sequence of length $N$ with $L$ layers, $H$ heads, and dimension $d$:
$$
\text{KV cache size} = 2 \times N \times L \times H \times d
$$

**Example:** LLaMA-13B with 2048 tokens ≈ 800 MB per sequence

---

### 2.2 Problems with Traditional KV Cache

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
