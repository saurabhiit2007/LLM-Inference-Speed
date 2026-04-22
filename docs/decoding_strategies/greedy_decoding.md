## 1. Overview

Greedy decoding selects the highest-probability token at every step — the simplest and fastest decoding strategy.

```
next_token = argmax P(token | context)
```

---

## 2. Key Problems

**Myopia:** Local optimality at each step does not guarantee global optimality. Choosing token A (prob 0.6) over B (prob 0.4) now can foreclose a much better sequence reachable only via B.

**Repetition loops:** If one token consistently scores highest, greedy gets stuck: *"be or not to be or not to be..."*

**Generic outputs:** Always picking the safe, high-probability continuation produces flat, predictable text with no diversity.

---

## 3. When to Use

**Use greedy when:** determinism is required (reproducible testing, CI pipelines), the task has a single correct answer (structured extraction, simple factual Q&A), or speed is the only concern.

**Do not use for:** chatbots, creative writing, open-ended generation — use temperature + top-p instead.

---

## 4. Cost

O(V) per step (argmax over vocabulary) — essentially free compared to the model forward pass. Fastest of all decoding strategies; no extra KV cache overhead.

---

## 5. Relation to Temperature Sampling

Greedy decoding is equivalent to temperature sampling with T → 0: as temperature approaches zero, the softmax distribution collapses to a one-hot on the argmax token.
