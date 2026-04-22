## 1. Overview

Best-of-N (BoN) sampling generates N candidate outputs from a model and selects the best one using a reward model or scoring function. It is the simplest form of test-time compute scaling.

```
x* = argmax R(xᵢ)  for i ∈ {1, ..., N}
```

Error reduction is approximately exponential: `error ∝ e^(-cN)` up to a saturation point.

---

## 2. Core Mechanics

**Algorithm:**

1. Sample N completions from the model (with temperature sampling for diversity)
2. Score each with a reward function R(x)
3. Return the highest-scoring completion

**Sampling for diversity:** Use temperature 0.7–1.0 or top-p nucleus sampling. Too-low temperature produces near-identical candidates; diversity is essential for BoN to work.

**Reward function options:**

| Type | Description | Use case |
|---|---|---|
| Trained reward model | Neural net trained on human preference data | General quality |
| Rule-based | Length, format compliance, keyword checks | Structured outputs |
| LLM-as-judge | Another LLM scores quality | Subjective tasks |
| Process Reward Model (PRM) | Scores each reasoning step | Math, code, multi-step |

---

## 3. Computational Cost

- **Time:** O(N × T) where T = generation time per sample
- **Space:** O(N × L) where L = average response length
- **Parallelizable:** All N samples can be generated in parallel given sufficient GPU memory

**Practical N values:** N=4–16 provides a good quality-cost balance. Beyond N=64–100, gains diminish significantly. Measure empirically on your task.

---

## 4. BoN vs. Other Techniques

| Technique | Training Cost | Inference Cost | Best Use Case |
|---|---|---|---|
| Best-of-N | None (uses existing reward model) | High (N×) | Quick quality boost, verifiable tasks |
| Beam Search | None | Medium | Deterministic, high-probability output |
| RLHF fine-tuning | Very high | Low | Production at scale |
| Speculative decoding | None | Lower latency | Speed, not quality improvement |

**BoN vs. beam search:** Beam search is deterministic — it tracks the top-k highest-probability prefixes. BoN samples independently with diversity and uses an external quality metric beyond log-probability.

---

## 5. Variants

**Weighted BoN:** Combine multiple high-scoring candidates via weighted averaging rather than hard selection — improves robustness when the top-1 reward model score is noisy.

**Adaptive N:** Determine N based on prompt difficulty or reward variance. Easy prompts: N=2–4. Hard reasoning: N=32–64+.

**Process-Based BoN:** Score using a PRM rather than an ORM — evaluates reasoning steps, enabling better selection for math and code. See [ORM vs PRM](orm_prm.md).

**Self-Consistency:** A special case of BoN without a reward model — generate N reasoning chains, take majority vote on the final answer. Works well for classification and math without needing a separate scorer.

---

## 6. Failure Modes

**Reward hacking:** Model exploits weaknesses in the reward model to score highly without genuine quality. Mitigations: ensemble reward models, KL regularisation from base model, diverse reward training data.

**Diminishing returns:** Beyond N*, additional samples provide negligible improvement. Always plot a quality-vs-N curve before committing to large N.

**Latency in production:** N × generation time is too slow for real-time applications. Use [speculative decoding](../decoding_strategies/speculative_decoding.md) or distil BoN-selected examples into a fine-tuned model for deployment.
