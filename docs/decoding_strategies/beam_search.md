# Beam Search - Interview Prep Guide

## 1. Overview

Beam search maintains **K candidate sequences** (beams) at each decoding step and selects the sequence with the highest **cumulative probability**. It balances between greedy decoding (K=1) and exhaustive search (K=vocab_size).

**Key insight:** Explore multiple promising paths simultaneously to avoid getting stuck in locally optimal but globally suboptimal sequences.

---

## 2. How It Works

### Algorithm

```python
def beam_search(model, prompt, beam_width=5, max_length=50):
    # Initialize: One beam with the prompt
    beams = [(prompt, 0.0)]  # (sequence, cumulative_log_prob)
    
    for step in range(max_length):
        candidates = []
        
        # Expand each beam
        for seq, score in beams:
            if seq[-1] == EOS:  # Completed sequence
                candidates.append((seq, score))
                continue
            
            logits = model(seq)
            probs = softmax(logits)
            
            # Consider all tokens
            for token_id, prob in enumerate(probs):
                new_seq = seq + [token_id]
                new_score = score + log(prob)  # Cumulative log probability
                candidates.append((new_seq, new_score))
        
        # Keep top K beams
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Stop if all beams are complete
        if all(seq[-1] == EOS for seq, _ in beams):
            break
    
    return beams[0][0]  # Return best sequence
```

### Core Concepts

**Beam width (K):**
- K=1: Greedy decoding
- K=5-10: Typical for translation
- K=50+: Exhaustive (expensive)

**Cumulative log probability:**
- Use log probabilities to avoid numerical underflow
- Score = log(P₁) + log(P₂) + ... + log(Pₙ)
- Equivalent to log(P₁ × P₂ × ... × Pₙ)

---

## 3. Example Walkthrough

**Prompt:** "The cat sat on the"

**Beam width:** 3

### Step 1: First Token

Model probabilities:
| Token | Probability | Log Prob |
|-------|-------------|----------|
| mat   | 0.40        | -0.92    |
| floor | 0.25        | -1.39    |
| sofa  | 0.15        | -1.90    |
| bed   | 0.10        | -2.30    |

**Top 3 beams:**
1. "The cat sat on the mat" → score: -0.92
2. "The cat sat on the floor" → score: -1.39
3. "The cat sat on the sofa" → score: -1.90

### Step 2: Second Token

Expand each beam:

**Beam 1:** "mat" + next token
- "mat ." → -0.92 + (-0.51) = **-1.43**
- "mat and" → -0.92 + (-1.61) = -2.53

**Beam 2:** "floor" + next token
- "floor ." → -1.39 + (-0.36) = **-1.75**
- "floor while" → -1.39 + (-1.20) = -2.59

**Beam 3:** "sofa" + next token
- "sofa ." → -1.90 + (-0.41) = -2.31
- "sofa when" → -1.90 + (-0.51) = **-2.41**

**New top 3 beams:**
1. "The cat sat on the mat ." → -1.43
2. "The cat sat on the floor ." → -1.75
3. "The cat sat on the sofa when" → -2.41

**Final output:** "The cat sat on the mat."

---

## 4. Key Characteristics

### Explores Multiple Paths
Unlike greedy, beam search maintains K hypotheses:
- Can recover from locally suboptimal choices
- Considers alternative continuations
- Better global optimization

### Length Bias Problem

Longer sequences accumulate more negative log probabilities:
```
Seq 1 (length 5): -0.5 + -0.6 + -0.4 + -0.5 + -0.3 = -2.3
Seq 2 (length 3): -0.5 + -0.6 + -0.4 = -1.5
```

Seq 2 has higher score despite being incomplete!

**Solution: Length normalization**
```python
normalized_score = score / length^α
# α = 0.6-0.8 typical
# α = 0: no normalization
# α = 1: full normalization
```

---

## 5. Common Problems

### Problem 1: Reduced Diversity

All beams often converge to similar outputs:

```
Prompt: "I think that"

Beam 1: "I think that we should focus on..."
Beam 2: "I think that we need to consider..."
Beam 3: "I think that this is important..."
```

All beams start with safe, high-probability tokens → similar continuations.

**Why:** Beam search favors **safe, high-probability paths** over **diverse, creative paths**.

### Problem 2: Generic Outputs

In open-ended generation:
```
Prompt: "Tell me a story about"

Greedy: "a boy who lived in"
Beam (K=5): "a young boy who lived in a small town"
```

Beam search produces grammatically perfect but boring text.

### Problem 3: Computational Cost

- **Memory:** O(K × T × V) for storing beam candidates
- **Time:** O(K × V) per step (vs O(V) for greedy)
- K=10 → 10× slower than greedy

---

## 6. Length Normalization

### Without Normalization
```python
# Shorter sequences preferred
beams = sorted(candidates, key=lambda x: x[1], reverse=True)
```

### With Normalization
```python
def length_penalty(length, alpha=0.6):
    return ((5 + length) / 6) ** alpha  # Google NMT formula

# Normalized score
normalized = score / length_penalty(len(seq), alpha=0.6)
beams = sorted(candidates, key=lambda x: normalized_score(x), reverse=True)
```

**Effect:**
- Encourages longer, more complete sequences
- Prevents premature termination
- Essential for translation and summarization

---

## 7. When to Use Beam Search

### ✅ Good Use Cases

**Structured tasks with clear objectives:**
- Machine translation
- Automatic speech recognition (ASR)
- Image captioning
- Summarization
- Question answering (extractive)

**When correctness matters more than creativity:**
- Technical documentation generation
- Code comment generation
- Medical report generation

### ❌ Poor Use Cases

**Creative or conversational tasks:**
- Story writing
- Dialogue systems
- Chatbots
- Poetry generation

**Tasks requiring diversity:**
- Brainstorming
- Multiple solution generation
- Creative writing

---

## 8. Variants and Improvements

### Diverse Beam Search
Force beams to be dissimilar using diversity penalty:
```python
diversity_penalty = 0.5
for beam_group in groups:
    # Penalize tokens already chosen by other groups
    adjusted_score = score - diversity_penalty * overlap_count
```

### Constrained Beam Search
Force inclusion of specific tokens/phrases:
```python
# E.g., must include "climate change" in summary
constraints = ["climate", "change"]
# Only keep beams that satisfy constraints
```

### Stochastic Beam Search
Add randomness to beam selection for more diversity.

---

## 9. Interview Questions

### Q1: What is beam search and how does it differ from greedy decoding?
**Answer:** Beam search maintains K candidate sequences (beams) instead of just one. At each step, it expands all beams, scores all possible continuations, and keeps the top K. This allows exploration of multiple paths and can recover from locally suboptimal choices, unlike greedy which commits to a single path.

---

### Q2: Why use log probabilities instead of regular probabilities?
**Answer:** Two reasons:
1. **Numerical stability:** Multiplying many small probabilities (0.3 × 0.4 × 0.2...) quickly underflows to zero in floating point
2. **Computational efficiency:** Log transforms products to sums: log(P₁ × P₂) = log(P₁) + log(P₂), which is more stable and efficient

---

### Q3: What is the length bias problem in beam search?
**Answer:** Longer sequences accumulate more negative log probabilities, making them score lower than shorter sequences even if they're more complete. For example, a 10-token sequence might score -8.5 while a 5-token incomplete sequence scores -3.2. Length normalization (dividing by sequence length raised to α) addresses this by favoring complete sentences.

---

### Q4: Why does beam search produce less diverse outputs than sampling?
**Answer:** Beam search is **deterministic** and **risk-averse**—it keeps the K highest-probability sequences. This means all beams tend to follow safe, high-probability paths, leading to similar outputs. Rare but creative continuations are discarded early. Sampling methods can explore lower-probability tokens, leading to more diversity.

---

### Q5: What's the computational complexity of beam search?
**Answer:**
- **Time per step:** O(K × V) where K=beam width, V=vocab size
  - Greedy: O(V), so beam is K× slower
- **Memory:** O(K × T) to store K beams of length T
- **Total:** For T steps, O(K × V × T) time

Typical K=5-10 for translation, but this 5-10× slowdown is significant.

---

### Q6: How do you choose the optimal beam width K?
**Answer:** Trade-off between quality and speed:
- **K=1:** Greedy (fast, low quality)
- **K=5-10:** Standard for translation (good balance)
- **K=50+:** Diminishing returns, very slow

Empirically, quality plateaus around K=10 for most tasks. Beyond that, you get marginal gains for significant computational cost.

---

### Q7: Can beam search guarantee finding the optimal sequence?
**Answer:** No. Beam search is a **heuristic** that prunes the search space. It only explores the top K paths at each step, potentially discarding paths that could lead to the globally optimal sequence later. Full exhaustive search (K=|V|^T) is computationally infeasible, so beam search is a practical approximation.

---

### Q8: What is diverse beam search and when is it useful?
**Answer:** Diverse beam search forces different beam groups to explore different areas of the search space by penalizing similarity. It's useful when you need multiple distinct outputs (e.g., generating K different translations or summaries). Standard beam search often produces K very similar sequences, which isn't helpful for diversity.

---

### Q9: When would you use beam search over sampling methods like top-p?
**Answer:** Use beam search for:
- **Objective quality metrics** (BLEU, ROUGE) that correlate with likelihood
- **Structured outputs** (translation, ASR) with one correct answer
- **Deterministic requirements** (reproducibility)

Use sampling for:
- **Creative tasks** requiring diversity
- **Conversational AI** needing personality
- **Open-ended generation** where many good answers exist

---

### Q10: How does beam search handle the EOS token?
**Answer:** When a beam generates EOS (end-of-sequence), it's marked as complete:
1. Complete beams stop expanding
2. They remain in the candidate pool with their final score
3. Active beams continue generating
4. Search terminates when all K beams complete or max_length reached
5. Return the complete beam with highest normalized score

Some implementations use length normalization to fairly compare complete sequences of different lengths.

---

## 10. Code Example

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple

def beam_search(
    model,
    input_ids: torch.Tensor,
    beam_width: int = 5,
    max_length: int = 50,
    length_penalty: float = 0.6,
    eos_token_id: int = 2
) -> torch.Tensor:
    """
    Beam search decoding.
    
    Args:
        model: Language model
        input_ids: Starting tokens [1, seq_len]
        beam_width: Number of beams
        max_length: Maximum sequence length
        length_penalty: Alpha for length normalization
        eos_token_id: End of sequence token
    
    Returns:
        Best sequence found
    """
    device = input_ids.device
    batch_size = input_ids.size(0)
    
    # Initialize beams: (sequence, score, finished)
    beams = [(input_ids[0].tolist(), 0.0, False)]
    
    for _ in range(max_length):
        candidates = []
        
        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, finished))
                continue
            
            # Get next token probabilities
            input_tensor = torch.tensor([seq], device=device)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
            
            # Expand beam with top-K tokens
            top_k_probs, top_k_ids = torch.topk(log_probs, beam_width)
            
            for prob, token_id in zip(top_k_probs, top_k_ids):
                new_seq = seq + [token_id.item()]
                new_score = score + prob.item()
                is_finished = (token_id.item() == eos_token_id)
                candidates.append((new_seq, new_score, is_finished))
        
        # Apply length penalty and keep top beams
        def normalized_score(item):
            seq, score, _ = item
            penalty = ((5 + len(seq)) / 6) ** length_penalty
            return score / penalty
        
        beams = sorted(candidates, key=normalized_score, reverse=True)[:beam_width]
        
        # Early stopping if all beams finished
        if all(finished for _, _, finished in beams):
            break
    
    # Return best sequence
    best_seq, _, _ = beams[0]
    return torch.tensor([best_seq], device=device)

# Usage
# output = beam_search(model, prompt_ids, beam_width=5)
```

---

## 11. Key Takeaways for Interviews

1. **Definition:** Maintains K candidate sequences, keeps top-K by cumulative probability
2. **Beam width K:** Trade-off between quality (higher K) and speed (lower K)
3. **Length normalization:** Essential to prevent bias toward shorter sequences
4. **Pros:** Better than greedy, explores alternatives, good for structured tasks
5. **Cons:** Computationally expensive (K× slower), low diversity, generic outputs
6. **Best for:** Translation, ASR, captioning, summarization
7. **Worst for:** Creative writing, dialogue, brainstorming
8. **Complexity:** O(K × V × T) time, O(K × T) space

---

## References

- [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) - Length normalization formula
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Discusses diversity issues
- [Diverse Beam Search](https://arxiv.org/abs/1610.02424)
