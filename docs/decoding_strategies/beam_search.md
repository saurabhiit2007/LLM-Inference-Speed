## 1. Overview

Beam search maintains K candidate sequences (beams) at each decode step, retaining only the top K by cumulative log-probability. It trades compute for better global optimality compared to greedy.

```
score(sequence) = Σ log P(tokent | token<t)
```

K=1 reduces to greedy decoding. Typical values: K=4–6 for translation/ASR.

---

## 2. Core Issues

**Length bias:** Longer sequences accumulate more negative log-probabilities and score lower than shorter ones — even if they are better continuations. Fix: length normalisation `score / length^α` (α ≈ 0.6–0.8).

**Low diversity:** All K beams tend to share high-probability prefixes and diverge only in minor ways. The output set looks nearly identical. Diverse beam search adds a penalty for overlap across beams, but remains less effective than sampling for open-ended generation.

**Compute cost:** K × V candidates evaluated per step vs V for greedy; KV cache memory scales K×. At K=5 this is 5× the decode cost.

---

## 3. When to Use

| Use beam search | Prefer sampling |
|---|---|
| Machine translation | Chatbots, dialogue |
| ASR transcription | Creative writing |
| Structured / extractive QA | Open-ended generation |

**In modern LLM serving, beam search is almost never used.** Continuous batching systems assume each sequence generates one token per iteration — beam search would require K parallel sequences per user request, multiplying KV cache cost and breaking iteration-level scheduling. It remains relevant in offline tasks (translation, ASR) but not in interactive LLM APIs.

---

## 4. Beam Search vs Speculative Decoding

Both involve multiple sequences, but the goals differ: beam search explores K candidates to find the best sequence; speculative decoding uses a draft model to propose tokens that the target model verifies, aiming for the same output distribution but faster. They are not interchangeable.
