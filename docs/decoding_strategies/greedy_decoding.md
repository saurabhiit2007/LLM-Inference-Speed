# Greedy Decoding - Interview Prep Guide

## 1. Overview

Greedy decoding selects the token with the **highest probability** at each step during autoregressive text generation. It's the simplest and fastest decoding strategy but lacks diversity.

**Key insight:** Local optimization (best token at each step) doesn't guarantee global optimality (best overall sequence).

---

## 2. How It Works

### Algorithm
```python
def greedy_decode(model, prompt, max_length):
    tokens = prompt
    for _ in range(max_length):
        logits = model(tokens)  # Shape: (vocab_size,)
        next_token = argmax(logits)  # Pick highest probability
        tokens.append(next_token)
        if next_token == EOS:
            break
    return tokens
```

### Example

Given probability distribution after "The cat sat on the":

| Token | Probability |
|-------|-------------|
| mat   | 0.40        |
| floor | 0.25        |
| sofa  | 0.15        |
| bed   | 0.10        |

**Greedy choice:** `mat` (highest probability)

**Output:** "The cat sat on the mat"

---

## 3. Key Characteristics

### Deterministic
- Same input → always same output
- No randomness involved
- Fully reproducible

### Myopic (Short-sighted)
Each decision is locally optimal but may lead to suboptimal sequences:

```
Step 1: "I think" → model assigns:
  - "that" (0.6)
  - "the" (0.4)
  
Greedy picks: "that"

Step 2: "I think that" → model assigns:
  - "is" (0.3)
  - "maybe" (0.25)
  
But if we had picked "the" at step 1:
  "I think the" → "best" (0.7)
  
Final: "I think that is..." (score: 0.6 × 0.3 = 0.18)
Better: "I think the best..." (score: 0.4 × 0.7 = 0.28)
```

Greedy can't recover from early suboptimal choices.

---

## 4. Common Problems

### Problem 1: Repetition Loops

```
Prompt: "To be or not to"
Output: "be or not to be or not to be or not to be..."
```

**Why:** If "be" has slightly higher probability than alternatives at each step, greedy gets stuck.

### Problem 2: Generic Outputs

```
Prompt: "Write a creative story about"
Greedy: "a boy who lived in a small town and went to school..."
```

Always picks safe, high-probability continuations → boring text.

### Problem 3: Early Mistakes Propagate

```
Prompt: "The capital of France is"
If model assigns:
  - "Paris" (0.45)
  - "Lyon" (0.46)  ← greedy picks this (wrong!)
  
Output: "Lyon, which is known for..."
```

Once the wrong token is chosen, subsequent tokens try to justify it.

---

## 5. When to Use Greedy Decoding

### ✅ Good Use Cases

**Deterministic tasks:**
- Math problem solving
- Code generation (when exact output matters)
- Factual Q&A with clear answers
- Translation of standardized text

**Debugging:**
- Baseline comparisons
- Reproducible testing
- Fastest inference for quick experiments

**Confident models:**
- When model probability distributions are very peaked
- Single obvious correct answer

### ❌ Poor Use Cases

**Creative tasks:**
- Story writing
- Poetry generation
- Brainstorming

**Conversational AI:**
- Chatbots
- Dialogue systems
- Personality-driven responses

**Long-form generation:**
- Articles, essays
- Open-ended content

---

## 6. Practical Considerations

### Computational Complexity
- **Time:** O(T × V) where T = sequence length, V = vocab size
- **Space:** O(1) for decoding state (minimal memory)
- **Fastest** among all decoding strategies

### Modifications

**Greedy + repetition penalty:**
```python
def greedy_with_penalty(logits, previous_tokens, penalty=1.2):
    for token in previous_tokens:
        logits[token] /= penalty  # Reduce probability of repeated tokens
    return argmax(logits)
```

**Greedy + length normalization:**
Useful when comparing sequences of different lengths (not during decoding itself).

---

## 7. Interview Questions

### Q1: What is greedy decoding and how does it work?
**Answer:** Greedy decoding selects the token with the highest probability at each generation step. It's deterministic and fast but locally optimal—it picks the best next token without considering whether this leads to the best overall sequence.

---

### Q2: Why is greedy decoding called "greedy"?
**Answer:** It's "greedy" because it makes the locally optimal choice at each step (highest probability token) without considering future consequences. Like the greedy algorithm paradigm, it optimizes immediate reward rather than global optimality.

---

### Q3: What's the main disadvantage of greedy decoding?
**Answer:** **Lack of diversity and repetition.** Greedy often produces repetitive, generic text because it can't explore alternative paths. Once it makes a suboptimal choice, it can't backtrack, leading to error propagation and repetitive loops.

---

### Q4: How does greedy decoding differ from beam search?
**Answer:**
- **Greedy:** Keeps only 1 hypothesis (best token at each step)
- **Beam search:** Keeps K hypotheses (top-K sequences), explores multiple paths

Beam search can recover from locally suboptimal choices by considering alternative sequences. Greedy cannot.

---

### Q5: Can greedy decoding produce the optimal sequence?
**Answer:** Not necessarily. Greedy finds a **locally optimal** sequence but not necessarily **globally optimal**. The highest-probability sequence might require choosing a lower-probability token early on that leads to much higher probabilities later.

---

### Q6: Why does greedy decoding cause repetition?
**Answer:** If a token or phrase has slightly higher probability than alternatives, greedy will keep selecting it. For example, in "to be or not to be or not to be...", if "be" consistently has a small probability advantage, the model gets stuck in a loop with no mechanism to escape.

---

### Q7: How can you reduce repetition in greedy decoding?
**Answer:** Common techniques:
1. **Repetition penalty:** Divide logits of previously generated tokens by penalty factor (e.g., 1.2)
2. **N-gram blocking:** Prevent repeating same N-grams
3. **Switch to sampling:** Use temperature/top-p for diversity
4. **Beam search:** Explore multiple paths to avoid local minima

---

### Q8: What's the computational cost of greedy decoding?
**Answer:** Very efficient:
- **Per step:** O(V) to find argmax over vocabulary
- **Total:** O(T × V) for T tokens
- **Memory:** O(1) for decoding state
- **Fastest** decoding strategy, no beam/sample storage overhead

---

### Q9: When would you prefer greedy over beam search?
**Answer:**
- **Latency-critical applications** (greedy is much faster)
- **Deterministic outputs required** (reproducibility)
- **Single clear answer** (factual Q&A, simple math)
- **Resource-constrained environments** (minimal memory)

Beam search is overkill when diversity isn't needed and the model is confident.

---

### Q10: How does greedy decoding interact with temperature?
**Answer:** Greedy decoding **ignores temperature** because it always picks argmax. Temperature only affects sampling-based methods. You can think of greedy as temperature → 0 (infinitely peaked distribution where highest probability → 1.0, others → 0).

---

## 8. Comparison with Other Methods

| Aspect | Greedy | Beam Search | Sampling |
|--------|--------|-------------|----------|
| **Speed** | Fastest | Slow (K× cost) | Fast |
| **Diversity** | None | Low | High |
| **Determinism** | Yes | Yes | No |
| **Repetition** | High risk | Medium risk | Low risk |
| **Quality** | Variable | Higher | Variable |
| **Memory** | Minimal | O(K × T) | Minimal |

---

## 9. Code Example

```python
import torch

def greedy_decode(model, input_ids, max_length=50):
    """
    Simple greedy decoding implementation.
    
    Args:
        model: Language model with forward() method
        input_ids: Starting token IDs (tensor)
        max_length: Maximum sequence length
    
    Returns:
        Generated token IDs
    """
    generated = input_ids.clone()
    
    for _ in range(max_length):
        # Get logits for next token
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]  # Last token logits
        
        # Greedy selection
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Append to sequence
        generated = torch.cat([generated, next_token], dim=1)
        
        # Check for EOS
        if next_token.item() == model.config.eos_token_id:
            break
    
    return generated

# Usage
# output = greedy_decode(model, prompt_tokens, max_length=100)
```

---

## 10. Key Takeaways for Interviews

1. **Definition:** Always picks the highest probability token at each step
2. **Pros:** Fast, deterministic, simple, reproducible
3. **Cons:** No diversity, repetitive, locally optimal only
4. **Main problem:** Repetition loops and generic outputs
5. **Best for:** Deterministic tasks, debugging, factual Q&A
6. **Worst for:** Creative writing, conversations, long-form generation
7. **Complexity:** O(T × V) time, O(1) space for decoding

---

## References

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Discusses repetition issues
- [Hugging Face: Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
