## 1. Overview

**ORM (Outcome Reward Model)** and **PRM (Process Reward Model)** are two paradigms for scoring model outputs during test-time compute. They differ in granularity: ORMs judge the final answer only; PRMs judge each intermediate reasoning step.

Both are used to guide [Best-of-N sampling](best_of_n_sampling.md) and tree search at inference time.

---

## 2. ORM vs PRM

| Aspect | ORM | PRM |
|---|---|---|
| Granularity | Final answer only | Each reasoning step |
| Feedback | Single scalar | Vector of per-step scores |
| Supervision cost | Low (binary correct/incorrect) | High (step-level human annotation) |
| Error localisation | Cannot identify where reasoning fails | Identifies the failing step |
| Tree/beam search | Poor — only terminal rewards | Excellent — prunes bad paths early |
| Interpretability | Low | High |

**Analogy:** ORM grades only the final answer on a test. PRM grades the work shown for each step.

---

## 3. How They Work

### ORM

```python
def orm_score(problem, solution):
    return 1.0 if verify_final_answer(solution) else 0.0
```

Single forward pass per complete solution. Cheap at inference time.

### PRM

```python
def prm_score(problem, reasoning_steps):
    return [verify_step(step, context) for step in reasoning_steps]
```

One forward pass per step — O(n) passes where n = number of steps. PRM scores are aggregated (min, mean, or product) across steps to rank candidates.

---

## 4. Training Data Requirements

**ORM:** `(problem, complete_solution) → binary label`. Straightforward to collect.

**PRM:** `(problem, step_1, ..., step_n) → step-level labels`. Expensive. Two approaches to reduce annotation cost:

- **Math-Shepherd (2024):** Automated step-level annotation using Monte Carlo rollouts — estimate step value by sampling completions and checking final answer correctness.
- **Weak-to-strong supervision:** Use a weaker model to generate step labels for a stronger model.

---

## 5. Role in Test-Time Search

PRMs unlock tree search at inference time:

```
Query
  │
  ▼
Generate k partial reasoning paths
  │
  ▼
PRM scores each step → prune low-scoring branches
  │
  ▼
Extend surviving paths → repeat until complete
  │
  ▼
ORM re-scores final candidates → return best
```

This is more efficient than BoN: bad reasoning paths are abandoned early rather than completed before scoring.

---

## 6. Key Research

- **"Let's Verify Step by Step" (OpenAI, 2023):** PRMs substantially outperform ORMs on MATH benchmark. Best-of-N with PRM >> Best-of-N with ORM.
- **Math-Shepherd (2024):** Automated PRM data collection; reduces human labelling cost.
- **DeepSeek R1 (2025):** Trains with RLVR (RL from Verifiable Rewards) — effectively learns an internal ORM signal without a separate model.

---

## 7. When to Use Each

**Use ORM when:**

- Task has clear right/wrong answers
- Limited annotation budget
- Fast inference is required

**Use PRM when:**

- Multi-step reasoning (math, code, planning)
- Tree search is in the pipeline
- Interpretability and step-level debugging matter
- Combining with [Best-of-N sampling](best_of_n_sampling.md) for complex tasks
