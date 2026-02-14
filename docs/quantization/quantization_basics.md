## 1. Core Concept
Quantization reduces model precision from FP32/FP16 to lower bit representations (INT8, INT4) to decrease memory and increase inference speed.

**Key Formula**: `Q(x) = round(x/S) - Z` where S = scale, Z = zero-point

---

---

## 2. Types

### Post-Training Quantization (PTQ)
- Applied after training
- No retraining needed
- Calibration dataset required
- Common methods: MinMax, Percentile, MSE

---

### Quantization-Aware Training (QAT)
- Simulates quantization during training
- Better accuracy but requires full training
- Fake quantization nodes in forward pass

---

---

## 3. Quantization Schemes

**Symmetric**: Zero-point = 0, range = [-127, 127] for INT8 <br>
**Asymmetric**: Zero-point ≠ 0, range = [0, 255] for UINT8

**Per-Tensor**: Single scale for entire tensor <br>
**Per-Channel**: Different scale per output channel (better accuracy)

---

---

## 4. Memory Savings

- FP32 → INT8: 4× reduction
- FP32 → INT4: 8× reduction
- Attention and FFN layers: Primary targets

---

---

## 5. Common Interview Questions

**Q1: Why does quantization work for LLMs?**
A: LLMs have activation/weight distributions that cluster around certain values. Most information is captured in relative magnitudes rather than absolute precision.

---

**Q2: What's the difference between static and dynamic quantization?**
A: Static uses calibration data to determine scales offline. Dynamic computes scales at runtime (slower but more accurate for varied inputs).

---

**Q3: Which layers are hardest to quantize?**
A: Layer normalization and first/last layers are most sensitive. Activations often need higher precision than weights.

---

**Q4: How do you measure quantization quality?**
A: Perplexity on validation set, task-specific metrics (accuracy, F1), and activation distribution analysis (KL divergence).

---

**Q5: What's the typical accuracy drop for INT8 quantization?**
A: Well-executed INT8 PTQ: <1% degradation. INT4: 2-5% depending on model size and method.

---