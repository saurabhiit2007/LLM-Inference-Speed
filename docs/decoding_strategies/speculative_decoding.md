# Speculative Decoding - Interview Prep Guide

## 1. Overview

Speculative decoding reduces **inference latency** in autoregressive LLMs by using a small **draft model** to propose multiple tokens that a large **target model** verifies in parallel. It produces **exact same output distribution** as standard decoding—not an approximation.

**Key insight:** Instead of 1 token per expensive model call, verify K tokens in one call by having a cheap model guess ahead.

**Typical speedup:** 2-3x latency reduction with no quality loss

---

---

## 2. The Problem with Standard Decoding

### Sequential Bottleneck

Autoregressive generation is inherently sequential:

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

**Standard decoding loop:**
```python
for _ in range(max_length):
    logits = large_model(tokens)      # Expensive!
    next_token = sample(logits[-1])   # Only use last position
    tokens.append(next_token)         # Generate 1 token
```

**Problems:**

- Generate 1 token per forward pass
- Model sits mostly idle (memory-bound, not compute-bound)
- Latency grows linearly with output length
- KV cache helps computation but doesn't break sequential dependency

**Example:** For 100 tokens, need 100 expensive forward passes of the large model.

---

---

## 3. How Speculative Decoding Works

### Core Idea

**Separate token proposal from verification:**

1. **Draft model (small, fast):** Proposes K tokens autoregressively
2. **Target model (large, expensive):** Verifies all K tokens in **one parallel forward pass**
3. **Accept/reject:** Use rejection sampling to maintain correct distribution

### Visual Example

```
Prompt: "The capital of France is"

Draft model proposes:  "Paris , which is"  (K=4 tokens)
                        ↓
Target model verifies all 4 tokens in one pass
                        ↓
Accept? [Paris: ✓] [,: ✓] [which: ✓] [is: ✗]
                        ↓
Output: "The capital of France is Paris , which"
Continue from "which" (3 tokens in 1 target pass instead of 3!)
```

---

---

## 4. Step-by-Step Algorithm

### Setup
- **Draft model** $q$: Small, fast (e.g., 1B params)
- **Target model** $p$: Large, accurate (e.g., 70B params)
- **Draft length** $K$: Number of tokens to propose (typically 4-8)

### Step 1: Draft Proposes K Tokens

Draft model generates K tokens autoregressively:

$$
y_i \sim q(\cdot \mid x, y_{<i}) \quad \text{for } i = 1, \dots, K
$$

**Important:** Sample tokens (don't use greedy) and record $q(y_i \mid x, y_{<i})$ for each.

```python
# Draft phase (cheap, K sequential calls to small model)
draft_tokens = []
draft_probs = []

for i in range(K):
    logits = draft_model(prompt + draft_tokens)
    probs = softmax(logits[-1])
    
    token = sample(probs)  # Sample, not greedy!
    draft_tokens.append(token)
    draft_probs.append(probs[token])  # Record q(y_i)
```

### Step 2: Target Verifies All Tokens

Target model runs **once** on the entire sequence:

$$
\text{Input: } [x, y_1, y_2, \dots, y_K]
$$

This produces probabilities for all draft positions:

$$
p(y_i \mid x, y_{<i}) \quad \text{for } i = 1, \dots, K
$$

```python
# Verification phase (1 parallel call to large model)
full_sequence = prompt + draft_tokens
target_logits = target_model(full_sequence)  # Shape: [seq_len, vocab_size]

# Extract logits for draft positions only
target_probs = []
for i in range(K):
    pos = len(prompt) + i - 1  # Position before token i
    probs = softmax(target_logits[pos])
    target_probs.append(probs[draft_tokens[i]])  # p(y_i)
```

**Key:** Transformer naturally computes logits for all positions—speculative decoding just uses them.

### Step 3: Accept or Reject Each Token

Use rejection sampling to maintain correct distribution:

$$
\alpha_i = \min\left(1, \frac{p(y_i \mid x, y_{<i})}{q(y_i \mid x, y_{<i})}\right)
$$

```python
accepted = []

for i in range(K):
    # Acceptance probability
    acceptance_prob = min(1.0, target_probs[i] / draft_probs[i])
    
    # Random test
    if random.random() < acceptance_prob:
        accepted.append(draft_tokens[i])
    else:
        # First rejection: stop here
        break
```

**Stop at first rejection** (can't accept token 3 if token 2 was rejected).

### Step 4: Handle Rejection

If token $j$ is rejected:

- Discard $y_j, y_{j+1}, \dots, y_K$
- Sample replacement from target model at position $j$:

$$
x_{\text{next}} \sim p(\cdot \mid x, y_{<j})
$$

```python
if len(accepted) < K:  # Some token was rejected
    # Sample next token from target model
    rejection_pos = len(accepted)
    next_token = sample(target_logits[rejection_pos])
    accepted.append(next_token)

# Continue with accepted tokens
prompt = prompt + accepted
```

If all K tokens accepted, continue with next draft batch.

---

---

## 5. Why It's Faster

### Speedup Analysis

**Standard decoding for N tokens:**

- N forward passes of target model
- Cost: $N \times C_{\text{target}}$

**Speculative decoding:**

- Draft proposes K tokens: $K \times C_{\text{draft}}$ (cheap)
- Target verifies in 1 pass: $1 \times C_{\text{target}}$
- If acceptance rate = $\alpha$, generate $\alpha \times K$ tokens per cycle

**Expected tokens per target call:**
$$
E[\text{tokens}] = 1 + \sum_{i=1}^{K} \alpha^i \approx \frac{1 - \alpha^{K+1}}{1 - \alpha}
$$

**Example:**

- $K=4$, $\alpha=0.7$ → ~2.4 tokens per target call
- 2.4× speedup (vs 1 token per call in standard decoding)

### Why Parallel Verification Works

Transformers compute logits for **all positions** in one pass:
```
Input:  ["The", "cat", "sat"]
Output: [logits_0, logits_1, logits_2]
         ↓         ↓         ↓
      P(·|"The") P(·|"The cat") P(·|"The cat sat")
```

Standard decoding only uses `logits_2` (last position).  
Speculative decoding uses `logits_0`, `logits_1`, `logits_2` to verify multiple tokens.

**This is not new capability**—it's how Transformers always work during training.

---

---

## 6. Why It's Exact (Not Approximate)

### Rejection Sampling Proof

The acceptance rule:
$$
\alpha_i = \min\left(1, \frac{p(y_i)}{q(y_i)}\right)
$$

is standard **rejection sampling**:

- If $p(y_i) \geq q(y_i)$: always accept (draft underestimated)
- If $p(y_i) < q(y_i)$: accept with probability $p/q$ (draft overestimated)

**Result:** Accepted tokens have distribution $p$, not $q$.

**Mathematical guarantee:** Output distribution is **identical** to sampling from target model directly.

### When Rejection Happens

```
Draft assigns:  q("Paris") = 0.8
Target assigns: p("Paris") = 0.9
→ Accept always (α = min(1, 0.9/0.8) = 1.0)

Draft assigns:  q("London") = 0.6
Target assigns: p("London") = 0.3
→ Accept 50% of time (α = 0.3/0.6 = 0.5)
```

Rejection corrects for draft model errors while maintaining exact target distribution.

---

---

## 7. Practical Considerations

### Draft Model Selection

**Options:**

1. **Smaller version of target** (e.g., 1B vs 70B parameters)
2. **Quantized target model** (INT8 vs FP16)
3. **Distilled model** trained to match target
4. **Early-exit from target** (use intermediate layers)

**Requirements:**

- Fast enough (≥10× faster than target)
- Similar enough to target (high acceptance rate)

### Draft Length K

**Trade-off:**

- **Small K (2-4):** Lower overhead, guaranteed speedup
- **Large K (8-16):** Higher potential speedup, but more likely to reject

**Optimal K depends on:**

- Acceptance rate (higher α → larger K beneficial)
- Draft/target speed ratio
- Memory constraints

**Typical:** K=4-5 works well in practice

### Acceptance Rate

**Factors affecting α:**

- Draft-target model similarity
- Task complexity (simple text → higher α)
- Prompt context (more context → better draft predictions)

**Typical rates:**

- Good draft: α=0.7-0.9
- Poor draft: α=0.3-0.5

**Below α≈0.3:** Speculative decoding may be slower than standard (overhead dominates).

---

---

## 8. Interaction with KV Cache

### KV Cache in Both Models

**Draft model:**

- Maintains its own KV cache
- Generates K tokens autoregressively (K cache updates)

**Target model:**

- Computes KV cache for entire draft window in one pass
- Accepted tokens' KV states are kept
- Rejected tokens' KV states are discarded

### Memory Considerations

```
Standard decoding: 1 model's KV cache
Speculative decoding: 2 models' KV cache (draft + target)
```

**Memory cost:** Minimal—draft model is small, so its cache is negligible.

---
