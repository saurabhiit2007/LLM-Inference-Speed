## 1. Overview

**Test-time compute scaling** (also called **inference-time scaling**) allocates additional computational resources during inference — rather than training — to improve performance on complex tasks. Instead of scaling model parameters, it uses techniques like multiple sampling, extended reasoning chains, or search algorithms at inference time.

### Core Principle

A smaller model with smart inference can outperform a larger model with standard decoding. Research shows this is often more cost-effective than training larger models for hard reasoning tasks.

---

## 2. Why It Matters

| Claim | Evidence |
|---|---|
| Smaller model + test-time search > larger model | 7B model with BoN/MCTS can match 34B greedy decoding on MATH |
| Inference compute can substitute for training compute | OpenAI o1/o3, DeepSeek R1 |
| System 2 thinking is unlockable at inference | Extended CoT enables deliberate step-by-step reasoning |

---

## 3. Core Techniques

### 3.1 Parallel Scaling

**Best-of-N (BoN):** Generate N candidate outputs, score each with a reward model or verifier, return the highest-scoring one. Error reduction follows approximately `error ∝ e^(-cN)`. See [Best-of-N Sampling](best_of_n_sampling.md).

**Majority Voting / Self-Consistency:** Sample N reasoning paths, select the most frequent answer. No reward model required — works well for multiple-choice and math tasks.

**Weighted Voting:** Assign confidence weights to candidates before aggregating.

---

### 3.2 Sequential Scaling

**Chain-of-Thought (CoT):** Decompose into intermediate steps. Longer reasoning chains improve performance on hard tasks.

**Self-Refinement:** Model critiques and revises its own output iteratively.

**Tree Search (MCTS, Beam Search + PRM):** Explore the reasoning tree with a [Process Reward Model](orm_prm.md) scoring intermediate steps. Prunes bad paths early; more efficient than BoN for structured problems.

---

### 3.3 Hybrid

Generate multiple chains in parallel, refine each sequentially, select the best final answer. Used internally by o1/o3 and R1.

---

## 4. Compute-Optimal Inference

Performance follows a power law: `Performance ∝ Compute^α` up to a saturation point N\* beyond which gains diminish.

**Allocation strategy by budget:**

| Budget | Recommended Approach |
|---|---|
| Low | Better base model, greedy decoding |
| Medium | Best-of-N (N=4–16) or CoT |
| High | Tree search + PRM; extended CoT |

**Memory bottleneck:** Long reasoning chains stress the KV-cache. Sparse attention (top-k or block-sparse) mitigates this.

---

## 5. Training for Test-Time Scaling

Models need to be trained to exploit extra compute effectively.

**Supervised Fine-Tuning (SFT):** Train on long CoT examples (synthetic or distilled). Models learn to imitate extended reasoning.

**Reinforcement Learning (RL):** Train with outcome rewards on verifiable tasks (math, code). RL discovers reasoning strategies without labeled chains. DeepSeek R1-Zero achieved strong reasoning from pure RL with no SFT.

**RLVR (RL from Verifiable Rewards):** Automatic reward generation in domains with ground truth — enables scalable RL training without human labeling.

---

## 6. Key Models (2024–2025)

| Model | Approach | Notable Result |
|---|---|---|
| OpenAI o1 (Sept 2024) | Hidden CoT + test-time search | Strong STEM benchmarks |
| OpenAI o3 (Dec 2024) | Deliberative alignment | 87.7% GPQA Diamond, 87.5% ARC-AGI |
| DeepSeek R1 (Jan 2025) | Pure RL, visible `<think>` tags, MoE (671B / 37B active) | 97.3% MATH-500, ~20× cheaper than o1 |

---

## 7. When to Use

**Use test-time scaling for:**

- Complex multi-step reasoning (math, code, planning)
- Tasks with verifiable correctness
- When a smaller model must match a larger one

**Avoid when:**

- Latency is the primary constraint
- Task is simple factual lookup
- Compute budget is very tight

---

## 8. Interaction with Other Inference Topics

- **Speculative decoding** can reduce latency of long reasoning chains — see [Speculative Decoding](../decoding_strategies/speculative_decoding.md)
- **KV-cache pressure** from long CoT is managed via [Paged Attention](../attention_optimization/paged_attention.md)
- **BoN and PRM** are the two core primitives — covered in this section
