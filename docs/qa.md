# Interview Q&A — LLM Inference Speed

Curated questions and answers across all topics. Each answer is written for a technical interview context — concise, precise, and covering the key points an interviewer expects.

---

## Chapter 1: Inference Fundamentals

**Q1. Why is LLM inference split into prefill and decode phases, and why does this matter for optimization?**

Prefill processes the entire input prompt in parallel — all input tokens attend to each other simultaneously. This is compute-bound: the GPU cores are saturated doing matrix multiplications. Decode generates one token at a time, reusing the KV cache for previous tokens. This is memory-bandwidth-bound: the GPU spends most of its time fetching weights and cached KVs from HBM, not doing compute. The distinction matters because different bottlenecks require different solutions — prefill benefits from FlashAttention (reducing memory traffic for the attention matrix); decode benefits from quantization, batching, and paged attention (reducing memory pressure and increasing utilization).

**Q2. What is the KV cache and what does it cost?**

The KV cache stores the key and value matrices computed for each previous token, so they don't need to be recomputed in subsequent decode steps. Without it, each new token would require attending to all previous tokens from scratch — O(n) full forward passes instead of O(1).

Cost per layer in FP16: `2 × batch_size × seq_len × num_heads × head_dim × 2 bytes`

For LLaMA-2-7B (32 layers, 32 heads, head_dim=128) at batch=1, seq=2048: ~1 GB total. This grows linearly with sequence length and batch size, making it the primary memory constraint in long-context serving.

**Q3. Walk through the roofline model and how you use it to diagnose an inference bottleneck.**

The roofline model defines the maximum achievable performance as: `min(Peak Compute, Arithmetic Intensity × Memory Bandwidth)`. Arithmetic Intensity = FLOPs / bytes transferred. For an H100: peak FP16 = 989 TFLOPS, memory bandwidth = 3,350 GB/s, giving a ridge point of ~295 FLOP/byte. An operation with AI < 295 is memory-bound; above that it's compute-bound. In practice: decode (batch=1) has very low AI (~1-10 FLOP/byte depending on model size) — deeply memory-bound. Prefill with large batches can reach compute-bound territory. The fix for memory-bound ops is reducing data movement (FlashAttention, quantization); for compute-bound ops it's better hardware utilization.

**Q4. What is TTFT and TPOT, and what drives each?**

- **TTFT (Time to First Token):** Time from request submission to the first output token. Determined by prefill latency, which scales with prompt length and is compute-bound.
- **TPOT (Time Per Output Token):** Average time to generate each subsequent token. Determined by decode latency, which is memory-bandwidth-bound and scales with model size.

`Total latency = TTFT + (output_tokens × TPOT)`. For interactive applications, TTFT dominates perceived responsiveness. For long-form generation, TPOT × output_tokens dominates total cost.

---

## Chapter 2: Attention Optimization

**Q5. How does FlashAttention avoid materialising the full attention matrix?**

Standard attention writes the full N×N attention matrix to GPU global memory (HBM), then reads it back for softmax, then reads it again for the weighted sum with V — three round trips to HBM. For N=16k in FP16 that's ~512 MB per layer per head.

FlashAttention tiles Q, K, V into blocks that fit in SRAM (fast on-chip memory). It computes softmax in a numerically stable online fashion (tracking the running max and normaliser across tiles) without ever materialising the full N×N matrix. All intermediate results stay in SRAM; only the final output is written to HBM. This reduces memory reads/writes from O(N²) to O(N), making attention IO-complexity match its compute complexity.

**Q6. Explain Paged Attention and the problem it solves.**

Before PagedAttention, serving systems pre-allocated a contiguous chunk of GPU memory per sequence for its KV cache (maximum possible sequence length × model dimensions). This caused two problems: (1) internal fragmentation — most of the pre-allocated space was wasted for short sequences; (2) no sharing — sequences with identical prefixes (e.g., same system prompt) each got their own copy.

PagedAttention manages KV cache like OS virtual memory. Each sequence's cache is divided into fixed-size blocks (~16 tokens each). A block table maps virtual block IDs to physical pages, which can be non-contiguous in memory. Copy-on-write enables prefix sharing — multiple sequences can point to the same physical pages for their shared prefix. Result: ~30-40% reduction in memory waste, enabling 2-4× higher throughput for the same GPU memory.

**Q7. What is prefix caching and when does it provide the most benefit?**

Prefix caching stores computed KV values for prompt prefixes and reuses them across requests with the same starting tokens. Benefit is highest when: (a) a long system prompt (512+ tokens) is shared by all requests, (b) RAG retrieval returns the same documents for similar queries, or (c) multi-turn chat reuses the growing conversation history. Requires exact byte-level match for the cached prefix. Implemented via shared physical pages in PagedAttention (vLLM) or explicit cache control (Anthropic API, OpenAI API).

**Q8. How do MQA and GQA reduce KV cache size?**

Standard Multi-Head Attention (MHA) has one set of K,V heads per Q head. Multi-Query Attention (MQA) uses a single shared K,V head for all Q heads — reduces KV cache by `num_heads`×, with some quality loss. Grouped Query Attention (GQA) groups Q heads and shares one K,V per group (e.g., 4 Q heads per K,V group for Llama-3) — a middle ground. Most modern models (Llama-3, Mistral, Gemma) use GQA. The KV cache saving is `num_kv_heads / num_q_heads`.

---

## Chapter 3: Decoding Strategies

**Q9. Compare greedy decoding, beam search, and sampling. When would you use each?**

| Strategy | Mechanism | Strength | Weakness |
|---|---|---|---|
| Greedy | Always pick highest-probability token | Deterministic, fast | Repetitive, myopic — misses globally better sequences |
| Beam search | Track top-k sequences at each step | Better global optimum than greedy | Prefers short sequences (length bias); less diverse; expensive at k>1 |
| Sampling (temp/top-p) | Sample from distribution | Diverse, creative output | Non-deterministic; can produce low-quality tokens |

Use greedy for deterministic tasks (classification, code with tests). Beam search for translation and summarisation where quality > diversity. Sampling for creative writing, chatbots, open-ended generation.

**Q10. Explain speculative decoding. Why does it produce the exact same output distribution?**

A small draft model (e.g., 7B) autoregressively proposes K tokens. The large target model (e.g., 70B) evaluates all K tokens in a single parallel forward pass (possible because the draft tokens can be attended to simultaneously). Each token is accepted or rejected using rejection sampling: token k is accepted with probability `min(1, p_target(x_k) / p_draft(x_k))`. If rejected, a corrected token is sampled from the residual distribution `p_target - p_draft`. This guarantees the accepted sequence has exactly the same marginal distribution as sampling from the target model alone. Speedup (2-3×) comes from the target model doing K verifications per forward pass instead of 1 generation.

**Q11. What is top-p (nucleus) sampling and why is it preferred over top-k?**

Top-k always samples from exactly the k highest-probability tokens, regardless of how the probability mass is distributed. If the model is very confident (one token has 99% probability), top-k still includes k options; if it's very uncertain (flat distribution), top-k still caps at k. Top-p dynamically selects the smallest set of tokens whose cumulative probability ≥ p. This adapts to the model's confidence — narrow vocabulary at high-confidence positions, wider at uncertain ones. In practice top-p=0.9 with temperature=0.7 is a strong default for creative tasks.

---

## Chapter 4: Batching & Throughput

**Q12. What is continuous batching and why is it critical for LLM serving throughput?**

Static batching waits for a batch of requests to finish before accepting new ones. Because sequences have variable lengths, the entire batch waits for the longest sequence — short sequences waste GPU time sitting idle.

Continuous batching (iteration-level scheduling, Orca 2022) inserts new requests into the batch at every decode step. As soon as a sequence emits EOS, its slot is immediately filled by a waiting request. GPU utilization stays near 100% because there's always work to do. This typically gives 10-23× throughput improvement over static batching at similar latency.

**Q13. What is chunked prefill and why does it help latency?**

Long prompts have long prefill phases that block decode steps for all other requests in the batch (a "prefill bubble"). Chunked prefill splits a long prompt's prefill into multiple chunks processed across several iterations, interleaved with decode steps from other requests. This reduces the latency spike that long-prompt requests impose on co-batched requests. Implemented in Sarathi-Serve and vLLM. The tradeoff is slightly higher total prefill latency for the chunked request in exchange for lower P99 latency for all other requests.

---

## Chapter 5: Quantization

**Q14. Explain the quantization formula and the difference between symmetric and asymmetric quantization.**

Quantization maps floating-point values to integers: `Q(x) = round(x/S) - Z` where S is the scale factor and Z is the zero-point.

- **Symmetric (Z=0):** Range is symmetric around zero, e.g., [-127, 127] for INT8. Simpler, slightly lower representational range. Good for weights.
- **Asymmetric (Z≠0):** Range shifted, e.g., [0, 255] for UINT8. Better for activations (often non-negative after ReLU). Adds complexity in dequantization.

Per-tensor uses one scale for the whole tensor; per-channel uses a different scale per output channel — significantly better accuracy at minimal overhead.

**Q15. How does GPTQ work and what makes it better than simple round-to-nearest quantization?**

GPTQ is based on Optimal Brain Quantization (OBQ). It quantizes one weight at a time and compensates for the quantization error by updating the remaining unquantized weights in that row using the inverse Hessian. The Hessian captures how sensitive the loss is to each weight — high-Hessian weights are quantized last (minimising error impact). GPTQ uses lazy batch updates to process columns in blocks (128 at a time) for GPU efficiency, reducing the 4-bit quantization of a 7B model to ~4 hours. Result: <1% perplexity degradation at 4-bit vs ~3% for round-to-nearest.

**Q16. What problem does AWQ solve compared to GPTQ, and what is its core insight?**

AWQ's insight is that not all weights are equally important — ~1% of weights ("salient channels") account for most of the accuracy. These are the channels with large activation magnitudes. GPTQ quantizes all weights with the same procedure. AWQ identifies salient channels by analysing activation statistics, then scales them up (multiply weight by s, divide activations by s) before quantization so they get more precision. This hardware-compatible scaling requires no grouping or mixed-precision — the quantized format is uniform INT4. AWQ is 10-20× faster to quantize than GPTQ (minutes vs hours) with comparable or better quality.

**Q17. Why does SmoothQuant focus on activations rather than weights?**

Weights in LLMs are relatively easy to quantize — their distributions are smooth and well-behaved. Activations have extreme outliers: a small number of channels (often the same channels across tokens) have values 100× larger than typical. These outliers push the quantization scale up, wasting precision on normal values. SmoothQuant migrates the quantization difficulty from activations to weights: multiply each activation channel by `1/s` (smoothing it) and multiply the corresponding weight channel by `s`. Mathematically equivalent, but now both activations and weights are quantizable at INT8. The migration strength `α` (typically 0.5) controls how much difficulty shifts to weights.

---

## Chapter 6: Serving Frameworks

**Q18. What makes vLLM's architecture well-suited for high-throughput LLM serving?**

vLLM combines three key innovations: (1) **PagedAttention** — eliminates KV cache fragmentation and enables prefix sharing, increasing effective memory capacity by 30-40%; (2) **Continuous batching** — iteration-level scheduling keeps GPU utilization near 100%; (3) **CUDA graph capture** — captures the decode step as a static computation graph, eliminating Python overhead per step. Together these give vLLM the highest throughput of open-source serving frameworks for most workloads. Its weakness is limited support for custom model architectures and slower performance than TensorRT-LLM for NVIDIA-specific workloads.

**Q19. When would you choose TensorRT-LLM over vLLM?**

TensorRT-LLM is better when: (a) maximum raw throughput on NVIDIA hardware matters more than flexibility — TRT-LLM uses kernel fusion and hardware-specific optimizations that vLLM doesn't; (b) you need FP8 or INT4 inference optimized for H100 tensor cores; (c) you have a standard architecture (Llama, Mistral, Falcon) already supported by TRT-LLM's engine builder. vLLM is better when: you need fast iteration, support for more model architectures, or OpenAI-compatible API without NVIDIA lock-in.

**Q20. What is ZeRO-Inference in DeepSpeed and when does it apply?**

ZeRO-Inference shards model weights across GPUs (like ZeRO-3 in training) rather than replicating them. Each GPU holds only a fraction of the weights; during inference, the needed shard is gathered on-demand via NVLink/InfiniBand. This allows serving models that don't fit on a single GPU without the overhead of tensor parallelism's all-reduce operations at every layer. Best for very large models (100B+) on multi-GPU servers with high interconnect bandwidth, where the model can't fit even on a single A100/H100. Less efficient than tensor parallelism for batch-heavy production serving.

---

## Chapter 7: Test-Time Compute Scaling

**Q21. What is test-time compute scaling and what is the core trade-off?**

Test-time compute scaling allocates additional compute during inference — via generating multiple samples, extended reasoning chains, or search — rather than training larger models. The core trade-off: test-time scaling improves output quality linearly in cost (Best-of-N costs N× more), whereas training-time scaling improves quality logarithmically in compute (doubling training compute gives diminishing returns). For hard reasoning tasks, test-time scaling can be more cost-efficient: a 7B model with BoN-32 can match a 34B model with greedy decoding. The limit is latency — extended reasoning is incompatible with low-latency requirements.

**Q22. Explain Best-of-N sampling: when is it effective and what determines the optimal N?**

Best-of-N generates N candidate outputs independently (with temperature sampling for diversity), scores each with a reward model, and returns the highest-scoring one. Effective when: (a) the task has verifiable correctness (math, code — reward model = unit tests or verifier), (b) the model already knows how to solve the problem but sometimes makes mistakes (N gives it multiple chances). Error reduction follows approximately `error ∝ e^(-cN)`. Optimal N depends on task difficulty, model capability, and reward model quality. In practice, N=4–16 gives strong gains; beyond N=64, returns diminish. Plot a quality-vs-N curve empirically — don't assume more is always better.

**Q23. What is the difference between an ORM and a PRM, and which is more useful for test-time search?**

An ORM (Outcome Reward Model) scores the final answer only: correct or incorrect. A PRM (Process Reward Model) scores each intermediate reasoning step. For test-time search (MCTS, beam search over reasoning steps), PRMs are far more useful: they provide value estimates at intermediate states, allowing bad reasoning paths to be pruned before they're completed. ORMs only give a terminal reward, meaning all N paths must be fully generated before ranking. OpenAI's "Let's Verify Step by Step" (2023) showed PRMs substantially outperform ORMs on MATH. The cost: PRM training requires step-level annotations, which are expensive to collect. Math-Shepherd (2024) addressed this with automated Monte Carlo rollout labelling.

---

## Quick Reference Cheat Sheet

| Concept | One-Line Answer |
|---|---|
| Prefill phase | Parallel processing of input — compute-bound |
| Decode phase | Sequential token generation — memory-bandwidth-bound |
| KV cache cost | ~1 GB per 2048 tokens for 7B FP16 model |
| FlashAttention | Tiled attention in SRAM — O(N) IO instead of O(N²) |
| Paged Attention | Non-contiguous KV blocks; eliminates fragmentation |
| Continuous batching | Insert new requests every decode step; ~10-23× throughput |
| Speculative decoding | Draft K tokens with small model, verify in parallel — 2-3× speedup |
| GPTQ | Layer-wise quantization with Hessian compensation |
| AWQ | Scale salient activation channels before quantizing |
| SmoothQuant | Migrate activation outliers to weights via per-channel scaling |
| Best-of-N | Generate N samples, pick best — error ∝ e^(-cN) |
| ORM vs PRM | ORM scores final answer; PRM scores each step |
