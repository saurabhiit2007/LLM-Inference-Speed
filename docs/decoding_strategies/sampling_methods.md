## 1. Overview

Sampling methods introduce controlled randomness into token selection. Instead of always picking the highest-probability token, they sample from a filtered distribution — enabling diverse, natural outputs while preventing low-quality tail tokens from being selected.

**Three core techniques:** temperature scaling, top-k filtering, top-p (nucleus) filtering. In production these are combined.

---

## 2. Temperature Sampling

Temperature T scales logits before softmax, controlling distribution sharpness:

$$P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

| Temperature | Effect | Use case |
|---|---|---|
| T → 0 | One-hot on argmax — equivalent to greedy | Deterministic tasks |
| T = 0.7 | Sharpened distribution, focused output | Factual Q&A |
| T = 1.0 | Original model distribution | Default |
| T = 1.2–1.5 | Flattened, more diverse | Creative writing |
| T → ∞ | Uniform distribution (gibberish) | Never |

Temperature alone does not prevent sampling from low-probability garbage tokens — always combine with top-k or top-p.

---

## 3. Top-k Sampling

Restrict sampling to the k highest-probability tokens, renormalise, then sample.

**Key limitation:** k is fixed regardless of distribution shape. When the model is very confident (one token at 95% probability), k=50 still forces diversity from 49 low-quality alternatives. When uncertain (flat distribution), k=50 may arbitrarily cut off valid options. Top-p solves this.

Typical values: k=40–50. Largely superseded by top-p in modern systems.

**Sampling from the renormalized distribution:** draw `u ~ Uniform(0, 1)`, then walk the cumulative distribution and select the first token where the running sum exceeds `u` — this is inverse CDF sampling, implemented by `torch.multinomial`. Equivalently, the **Gumbel-max trick** adds Gumbel-distributed noise to the filtered logits and takes the argmax, producing the same categorical distribution without an explicit softmax step.

---

## 4. Top-p (Nucleus) Sampling

Select the smallest set of tokens whose cumulative probability ≥ p, renormalise, then sample.

**Example (p=0.9):**

| Token | P | Cumulative |
|---|---|---|
| mat | 0.40 | 0.40 |
| floor | 0.25 | 0.65 |
| sofa | 0.15 | 0.80 |
| bed | 0.10 | 0.90 | ← stop |
| roof | 0.05 | 0.95 | excluded |

**Key advantage over top-k:** adapts automatically to model confidence. When the model is certain, the nucleus shrinks to 1–2 tokens. When uncertain, it expands to include many options. This is why top-p is the default in modern inference APIs.

Typical value: **p=0.9** (OpenAI default). p=0.95 for more diversity; p=0.8 for more focus. The same inverse CDF sampling mechanism applies after the nucleus is renormalized.

---

## 5. Production Recipe

Standard combination: **temperature + top-p**

1. Apply temperature to scale logits
2. Filter with top-p to remove the tail
3. Sample from the renormalised distribution

Adding top-k on top of top-p is redundant — top-p already handles adaptive cutoff.

---

## 6. Comparison

| Method | Adapts to confidence | Prevents tail | Production default |
|---|---|---|---|
| Greedy | — | — | Deterministic tasks only |
| Temperature only | No | No | Never alone |
| Top-k | No | Yes (fixed) | Legacy |
| Top-p | Yes | Yes | Yes |
| Temperature + Top-p | Yes | Yes | Standard |

---

## 7. Effect on Inference Cost

Sampling adds one multinomial draw per step — negligible overhead. The dominant cost is the model forward pass and KV cache access, which are identical regardless of sampling strategy.
