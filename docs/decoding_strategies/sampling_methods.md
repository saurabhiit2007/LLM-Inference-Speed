# Sampling Methods - Interview Prep Guide

## 1. Overview

Sampling methods introduce **controlled randomness** into text generation by probabilistically selecting tokens rather than deterministically choosing the highest-probability token. This enables diverse, creative outputs while maintaining coherence.

**Key insight:** The best sequence isn't always the highest-probability one—controlled randomness can produce more natural, interesting text.

**Three main techniques:**
1. **Temperature sampling** - Controls randomness via scaling
2. **Top-k sampling** - Samples from top K tokens
3. **Top-p sampling (nucleus)** - Samples from smallest set with cumulative probability ≥ p

---

## 2. Temperature Sampling

### 2.1 How It Works

Temperature scales logits before applying softmax, controlling distribution "sharpness":

$$\text{P}_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

Where:
- $z_i$ = logit for token i
- $T$ = temperature (T > 0)
- $T = 1$ → original distribution
- $T < 1$ → sharper (more deterministic)
- $T > 1$ → flatter (more random)

### 2.2 Example

Original probabilities after "The cat sat on the":

| Token | Prob (T=1) | Prob (T=0.5) | Prob (T=2.0) |
|-------|------------|--------------|--------------|
| mat   | 0.40       | 0.58         | 0.28         |
| floor | 0.25       | 0.26         | 0.23         |
| sofa  | 0.15       | 0.10         | 0.18         |
| bed   | 0.10       | 0.04         | 0.14         |
| roof  | 0.05       | 0.01         | 0.09         |
| moon  | 0.03       | 0.00         | 0.05         |
| pizza | 0.02       | 0.00         | 0.03         |

**T=0.5 (Low):** Distribution sharpens → "mat" dominates → near-greedy behavior  
**T=2.0 (High):** Distribution flattens → "moon", "pizza" become viable → more random

### 2.3 Extreme Cases

**T → 0:**
- Distribution becomes one-hot (probability → 1.0 for argmax)
- Equivalent to greedy decoding
- Zero randomness

**T → ∞:**
- Uniform distribution (all tokens equally likely)
- Maximum randomness
- Often produces gibberish

**Typical values:**
- **T=0.7:** Focused, coherent (factual Q&A)
- **T=1.0:** Balanced (default)
- **T=1.2-1.5:** Creative, diverse (story writing)

### 2.4 Code Example

```python
import torch
import torch.nn.functional as F

def temperature_sampling(logits, temperature=1.0):
    """
    Sample token with temperature scaling.
    
    Args:
        logits: Raw model outputs [vocab_size]
        temperature: Scaling factor (T > 0)
    
    Returns:
        Sampled token ID
    """
    # Scale logits
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sample
    token_id = torch.multinomial(probs, num_samples=1)
    return token_id.item()

# Usage
# next_token = temperature_sampling(model_logits, temperature=1.2)
```

---

## 3. Top-k Sampling

### 3.1 How It Works

Top-k restricts sampling to the **k most probable tokens**:

1. Sort tokens by probability (descending)
2. Keep only top-k tokens
3. Set all other probabilities to zero
4. Renormalize the top-k probabilities
5. Sample from renormalized distribution

### 3.2 Example

Probabilities after "The cat sat on the":

| Token | Probability |
|-------|-------------|
| mat   | 0.40        |
| floor | 0.25        |
| sofa  | 0.15        |
| bed   | 0.10        |
| roof  | 0.05        |
| moon  | 0.03        |
| pizza | 0.02        |

**Top-k with k=3:**

Kept tokens:
- mat: 0.40
- floor: 0.25
- sofa: 0.15

Renormalized:
- mat: 0.40/0.80 = 0.50
- floor: 0.25/0.80 = 0.31
- sofa: 0.15/0.80 = 0.19

Removed: bed, roof, moon, pizza

**Output:** One of {mat, floor, sofa} with renormalized probabilities

### 3.3 Key Limitation: Fixed K

Top-k doesn't adapt to distribution shape:

**Case 1: Confident model**
- Probabilities: [0.85, 0.07, 0.03, 0.02, 0.02, 0.01]
- k=5 → keeps 5 tokens even though model is very confident
- Inefficient: forces sampling from low-quality tokens

**Case 2: Uncertain model**
- Probabilities: [0.20, 0.20, 0.20, 0.20, 0.20]
- k=3 → keeps only 3 tokens, excludes equally valid options
- Too restrictive: removes valid choices

**Problem:** Same k value for different distribution shapes.

### 3.4 Typical Values

- **k=10-20:** Conservative, relatively safe outputs
- **k=40-50:** More diverse, creative
- **k=100+:** Very random, may include low-quality tokens

### 3.5 Code Example

```python
import torch

def top_k_sampling(logits, k=50, temperature=1.0):
    """
    Sample from top-k tokens.
    
    Args:
        logits: Raw model outputs [vocab_size]
        k: Number of top tokens to consider
        temperature: Optional temperature scaling
    
    Returns:
        Sampled token ID
    """
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Get top-k
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k)
    
    # Softmax over top-k
    probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample from top-k
    sampled_idx = torch.multinomial(probs, num_samples=1)
    
    # Map back to original vocabulary
    token_id = top_k_indices[sampled_idx]
    return token_id.item()

# Usage
# next_token = top_k_sampling(model_logits, k=40, temperature=0.9)
```

---

## 4. Top-p Sampling (Nucleus Sampling)

### 4.1 How It Works

Top-p selects the **smallest set of tokens** whose cumulative probability ≥ p:

1. Sort tokens by probability (descending)
2. Compute cumulative probability
3. Keep tokens until cumulative ≥ p
4. Renormalize and sample

**Key advantage:** Adaptive—number of tokens varies based on distribution.

### 4.2 Example

Probabilities after "The cat sat on the":

| Token | Probability | Cumulative |
|-------|-------------|------------|
| mat   | 0.40        | 0.40       |
| floor | 0.25        | 0.65       |
| sofa  | 0.15        | 0.80       |
| bed   | 0.10        | 0.90       | ← Stop here
| roof  | 0.05        | 0.95       |
| moon  | 0.03        | 0.98       |
| pizza | 0.02        | 1.00       |

**Top-p with p=0.9:**
- Keep: mat, floor, sofa, bed (cumulative = 0.90)
- Remove: roof, moon, pizza
- Effective k = 4

### 4.3 Adaptive Behavior

**Case 1: Confident model**
```
Probabilities: [0.85, 0.07, 0.03, 0.02, 0.02, 0.01]
p=0.9 → keeps 2 tokens [0.85, 0.07]
Effective k = 2 (adaptive reduction)
```

**Case 2: Uncertain model**
```
Probabilities: [0.20, 0.20, 0.20, 0.20, 0.20]
p=0.9 → keeps 5 tokens
Effective k = 5 (adaptive expansion)
```

**Advantage over top-k:** Automatically adjusts to model confidence.

### 4.4 Typical Values

- **p=0.9:** Standard for most applications (OpenAI default)
- **p=0.95:** More diverse, creative outputs
- **p=0.75-0.85:** More focused, conservative

### 4.5 Code Example

```python
import torch

def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Nucleus sampling (top-p).
    
    Args:
        logits: Raw model outputs [vocab_size]
        p: Cumulative probability threshold (0 < p ≤ 1)
        temperature: Optional temperature scaling
    
    Returns:
        Sampled token ID
    """
    # Apply temperature
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff: first position where cumulative > p
    # Include that position (so cumulative >= p)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Mask out tokens to remove
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )
    filtered_logits = scaled_logits.clone()
    filtered_logits[indices_to_remove] = float('-inf')
    
    # Sample from filtered distribution
    filtered_probs = F.softmax(filtered_logits, dim=-1)
    token_id = torch.multinomial(filtered_probs, num_samples=1)
    return token_id.item()

# Usage
# next_token = top_p_sampling(model_logits, p=0.9, temperature=1.0)
```

---

## 5. Combining Sampling Methods

In practice, sampling methods are often **combined**:

### Common Combination: Temperature + Top-p

```python
def sample_token(logits, temperature=0.9, top_p=0.9):
    # 1. Apply temperature first
    scaled_logits = logits / temperature
    
    # 2. Then apply top-p filtering
    token = top_p_sampling(scaled_logits, p=top_p)
    return token
```

**Why combine:**
- Temperature controls overall randomness
- Top-p prevents sampling from the very long tail
- Together: controlled creativity with safety

### Other Combinations

**Temperature + Top-k:**
```python
# Used in some older systems
token = top_k_sampling(logits, k=40, temperature=0.8)
```

**Top-k + Top-p:**
```python
# Apply top-k first as a hard cutoff
# Then apply top-p for adaptive filtering
# Less common, top-p usually sufficient
```

---

## 6. When to Use Each Method

### Temperature Sampling

**✅ Use when:**
- Need simple randomness control
- Working with other filtering methods (top-k, top-p)
- Want smooth transition between deterministic and random

**❌ Avoid when:**
- Used alone (no filtering from tail)
- Need adaptive behavior

### Top-k Sampling

**✅ Use when:**
- Need simple, predictable diversity control
- Fixed computational budget (always k tokens)
- Legacy systems (older standard)

**❌ Avoid when:**
- Distribution shape varies significantly
- Need adaptive behavior
- Modern systems (top-p preferred)

### Top-p Sampling

**✅ Use when:**
- Need adaptive diversity
- Model confidence varies
- General-purpose text generation
- Conversational AI, creative writing

**❌ Avoid when:**
- Need strict determinism
- Computational constraints (slightly more complex)

---

## 7. Interview Questions

### Q1: What problem do sampling methods solve?
**Answer:** Greedy and beam search produce repetitive, generic text by always choosing high-probability tokens. Sampling methods introduce controlled randomness, enabling diverse, creative outputs while preventing the model from getting stuck in repetitive loops. They balance coherence with variety.

---

### Q2: How does temperature affect the probability distribution?
**Answer:** Temperature scales logits before softmax. Low temperature (T<1) sharpens the distribution—high-probability tokens become more dominant (near-greedy). High temperature (T>1) flattens the distribution—low-probability tokens become more likely (more random). At T→0, it becomes greedy; at T→∞, it becomes uniform.

---

### Q3: What's the main limitation of top-k sampling?
**Answer:** **Fixed k doesn't adapt to distribution shape.** When the model is very confident, k=50 wastes computation on unlikely tokens. When uncertain with many valid options, k=50 might exclude good alternatives. Top-k treats all distributions the same, ignoring the model's confidence level.

---

### Q4: How is top-p better than top-k?
**Answer:** Top-p is **adaptive**—it automatically adjusts the number of candidate tokens based on distribution shape:
- Confident model → keeps fewer tokens (smaller effective k)
- Uncertain model → keeps more tokens (larger effective k)

This makes top-p more robust across different contexts without hyperparameter tuning.

---

### Q5: Can you use temperature=0.5 with top-p=0.9 together?
**Answer:** Yes, and this is common in practice:
1. Apply temperature=0.5 first → sharpen distribution (reduce randomness)
2. Apply top-p=0.9 → filter out low-probability tail
3. Sample from filtered distribution

Temperature controls overall randomness, top-p prevents sampling gibberish. Together they provide controlled, safe creativity.

---

### Q6: Why not just use temperature alone?
**Answer:** Temperature alone doesn't **filter out** low-probability tokens—it just reduces their probability. Even with low temperature, there's still a tiny chance of sampling nonsense tokens from the very long tail. Top-p/top-k provide a hard cutoff, ensuring we never sample from clearly bad options.

---

### Q7: What happens with very high temperature (T=5)?
**Answer:** The distribution becomes nearly **uniform**—all tokens have similar probability regardless of model's original confidence. This produces incoherent gibberish:

```
Input: "The capital of France is"
Output: "banana quantum seventh pencil"
```

High temperature destroys the model's learned knowledge. Typical max: T=1.5-2.0.

---

### Q8: How do you choose between p=0.9 vs p=0.95?
**Answer:**
- **p=0.9:** More focused, coherent (default for most applications)
- **p=0.95:** More diverse, creative (for storytelling, brainstorming)

Trade-off: Higher p → more diversity but higher risk of incoherence. Start with 0.9; increase for creativity, decrease for safety. Depends on task requirements.

---

### Q9: What's the computational cost of top-p vs top-k?
**Answer:**
- **Top-k:** O(V log k) — partial sort for top-k elements
- **Top-p:** O(V log V) — full sort to compute cumulative probabilities

Top-p is slightly more expensive, but the difference is negligible compared to model inference cost. The adaptive benefits of top-p outweigh the small computational overhead.

---

### Q10: Why do modern LLMs (GPT-4, Claude) prefer top-p over top-k?
**Answer:** **Adaptivity and robustness.** Top-p automatically adjusts to:
- Different contexts (formal vs casual)
- Varying model confidence
- Different domains (technical vs creative)

This makes it more reliable across diverse use cases without manual tuning. Top-k requires choosing k for each scenario, while top-p with p=0.9 works well universally.

---

## 8. Comparison Table

| Method | Randomness Control | Adaptive | Prevents Tail Sampling | Typical Usage | Complexity |
|--------|-------------------|----------|----------------------|---------------|------------|
| **Temperature** | Continuous (T) | No | No | + top-p/top-k | O(V) |
| **Top-k** | Discrete (k) | No | Yes | Legacy, simple control | O(V log k) |
| **Top-p** | Continuous (p) | Yes | Yes | Modern LLMs, general use | O(V log V) |
| **Greedy** | None | N/A | N/A | Debugging, deterministic | O(V) |
| **Beam** | None | No | N/A | Translation, ASR | O(K×V) |

---

## 9. Practical Guidelines

### For Most Applications
```python
# Recommended defaults
temperature = 0.8-1.0
top_p = 0.9
# Don't use top-k with top-p
```

### For Creative Writing
```python
temperature = 1.0-1.5
top_p = 0.95
```

### For Factual Q&A
```python
temperature = 0.3-0.7
top_p = 0.9
```

### For Code Generation
```python
temperature = 0.2-0.5
top_p = 0.9
# Or just use greedy (temperature=0)
```

---

## 10. Key Takeaways for Interviews

1. **Temperature:** Controls randomness by scaling logits; T<1 sharper, T>1 flatter
2. **Top-k:** Fixed number of candidate tokens; simple but not adaptive
3. **Top-p:** Adaptive cumulative probability threshold; modern standard
4. **Combination:** Use temperature + top-p together for best results
5. **Top-p advantage:** Automatically adjusts to model confidence
6. **Common values:** temperature=0.8-1.0, top-p=0.9
7. **Use case:** Sampling for creative/diverse tasks; greedy/beam for deterministic tasks

---

## References

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Introduces nucleus (top-p) sampling
- [Hugging Face: Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [OpenAI API: Sampling Parameters](https://platform.openai.com/docs/guides/text-generation)
