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

---

## Common Interview Questions

**Q1: Why does GPTQ outperform naive quantization?** <br>
A: It compensates each quantization error by adjusting remaining weights based on input correlations (Hessian), preventing error accumulation.

---

**Q2: What's the computational complexity?** <br>
A: O(n²) in weight dimensions due to Hessian computation and updates. Lazy batching reduces this to O(n²/b) where b = block size.

---

**Q3: Why use Hessian instead of just gradient?** <br>
A: Second-order information captures weight interactions. First-order (gradient) only gives local slope, not curvature of loss landscape.

---

**Q4: What's "desc_act" in GPTQ?** <br>
A: Reorders activation channels by importance before quantization. Helps but adds complexity; often disabled for speed.

---

**Q5: Can GPTQ quantize activations?** <br>
A: No, GPTQ is weight-only. Activations typically stay FP16 or use runtime INT8 quantization.

---

**Q6: GPTQ vs. AWQ - when to use which?** <br>
A: GPTQ: Better for extreme compression (3-bit). AWQ: Faster quantization, better at 4-bit, protects important weights rather than compensating errors.

---

**Q7: Why is calibration data needed?** <br>
A: To compute Hessian (H = X^T X), which captures input statistics. Need representative samples to estimate weight importance accurately.

---