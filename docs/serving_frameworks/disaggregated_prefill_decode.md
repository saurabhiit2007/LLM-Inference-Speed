## 1. Overview

Disaggregated prefill-decode (PD disaggregation) is an inference serving architecture that assigns **prefill** and **decode** operations to separate pools of GPUs rather than running both on the same GPU. This is a major architectural shift in large-scale LLM serving that addresses a fundamental tension in the two-phase inference pipeline.

---

## 2. The Problem with Co-located Prefill and Decode

In standard serving (e.g., vLLM with continuous batching), prefill and decode share the same GPU:

- **Prefill** is compute-bound, runs for many iterations on a long prompt, and during this time **blocks all decode operations** for co-batched requests — creating a "prefill bubble."
- **Decode** is memory-bandwidth-bound, requires low latency, and is sensitive to interruption.

The conflict: a single 10,000-token prefill can cause hundreds of milliseconds of decode stall for every other active request. Chunked prefill mitigates this but does not eliminate it.

---

## 3. PD Disaggregation Architecture

```
Prefill Pool (compute-optimised GPUs)    Decode Pool (memory-bandwidth-optimised GPUs)
────────────────────────────────────     ──────────────────────────────────────────────
• Receives new requests                  • Receives KV cache via high-speed interconnect
• Runs prefill to completion             • Runs decode until EOS
• Produces KV cache                      • Handles all active request queues
• Transfers KV → Decode pool             • Optimised for throughput / latency SLO
```

KV cache is transferred from the prefill GPU to the decode GPU over NVLink, InfiniBand, or RDMA after prefill completes.

---

## 4. Why This Helps

**For decode latency:** Decode GPUs are never interrupted by prefill computation. P99 decode latency drops significantly.

**For prefill throughput:** Prefill GPUs can be sized and batched independently — larger batches, larger models, or longer prompts without impacting active sessions.

**For hardware flexibility:** Prefill is compute-bound → benefits from H100 BF16/FP8 throughput. Decode is memory-bandwidth-bound → benefits from high-HBM-bandwidth GPUs. You can mix GPU types within one serving system.

---

## 5. Challenges

**KV cache transfer overhead:** Transferring a full KV cache (e.g., 1–2 GB for a 7B model with long context) introduces latency proportional to context length. Requires high-bandwidth interconnects (NVLink 3.0: ~900 GB/s, InfiniBand HDR: ~200 GB/s).

**Load balancing:** Prefill and decode have very different compute profiles and must be independently scaled. A sudden spike in long prompts can saturate the prefill pool while decode idles.

**Implementation complexity:** Session routing, KV transfer protocols, and pool auto-scaling are non-trivial to implement correctly.

---

## 6. Real-World Implementations

**Mooncake (ByteDance, 2024):** First published large-scale PD disaggregation system. Reports ~75% reduction in P99 TTFT and significant throughput gains on production traffic.

**DistServe (Zhong et al., 2024):** Academic system showing PD disaggregation achieves 2–3× better goodput (requests meeting SLO) vs co-location.

**vLLM v0.6+ / SGLang:** Both production frameworks are adding PD disaggregation support.

---

## 7. When to Use

PD disaggregation is most beneficial when:

- Traffic has a mix of short and long prompts (long prompts disrupt short-prompt latency)
- TTFT SLOs are strict (interactive applications)
- Scale is large enough that dedicated GPU pools are cost-effective (hundreds of GPUs)
- Hardware interconnect bandwidth is sufficient for KV transfer

For small deployments or workloads with uniform prompt lengths, co-located continuous batching with chunked prefill is simpler and often sufficient.

---

## 8. Relation to Other Optimisations

| Topic | Relationship |
|---|---|
| Chunked prefill | Predecessor/simpler alternative — reduces but doesn't eliminate prefill-decode interference |
| Paged attention | Needed on the decode side for efficient KV memory management |
| Prefix caching | Can be applied independently on either pool |
| Speculative decoding | Applied on the decode pool; orthogonal to disaggregation |
