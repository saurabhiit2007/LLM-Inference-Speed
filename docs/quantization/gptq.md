## 1. Paper

Frantar et al., 2022 - "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

---

---

## 2. Core Idea
Quantize weights one-by-one while compensating errors in remaining weights using second-order information (Hessian).

---

---

## 3. Algorithm (Simplified)

```python
# For each layer's weight matrix W
H = 2 * X^T @ X / n_samples  # Hessian (input correlations)

for i in range(n_columns):
    # Quantize column i
    w_q = quantize(W[:, i])
    error = W[:, i] - w_q
    
    # Distribute error to remaining columns using Hessian
    W[:, i+1:] -= error * H[i, i+1:] / H[i, i]
    
    W[:, i] = w_q
```

---

---

## 4. Key Innovation: Optimal Brain Quantization (OBQ)

- Uses second-order Taylor expansion to minimize quantization loss
- Iteratively quantizes weights in order that minimizes Hessian-weighted error
- Compensates each quantization error before next step

---

---

## Lazy Batch Updates (Efficiency)

- Don't update all weights individually
- Process in blocks of 128 columns
- Dramatic speedup with minimal quality loss

---

---

## Specifications

- **Calibration**: 128 samples typical
- **Time**: 3-4 hours for 175B model on single GPU
- **Group Size**: 128 (default), lower for better quality
- **Bits**: Designed for 3-4 bit, works for INT8 too

---

---

## Accuracy

| Model | Bits | Group Size | Perplexity Δ |
|-------|------|------------|--------------|
| LLaMA-7B | 4 | 128 | +0.2 |
| LLaMA-13B | 3 | 128 | +0.9 |
| OPT-175B | 4 | 128 | +0.1 |

---

---

## vs. Round-To-Nearest (RTN)

- RTN: Fast, 5-10% degradation at 4-bit
- GPTQ: Slow quantization, <2% degradation at 4-bit

---

---

## Implementation

```python
# AutoGPTQ library
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(
    "model_name",
    quantize_config={
        "bits": 4,
        "group_size": 128,
        "desc_act": False  # Activation ordering
    }
)
```

---
