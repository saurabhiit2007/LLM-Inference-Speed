## 1. Overview

Prefix caching (also called prompt caching or KV-cache reuse) avoids recomputing the KV-cache for portions of the prompt that are identical across requests. When a system prompt, few-shot examples, or document is shared between many queries, the attention keys and values for that prefix are computed once and reused.

---

## 2. How It Works

Standard inference computes keys and values for every token in every request — including the system prompt — from scratch.

With prefix caching:

```
Request 1: [system prompt (512 tokens)] + [user query A]
  → compute KV for all 512 + query-A tokens
  → cache KV for the 512-token prefix

Request 2: [system prompt (512 tokens)] + [user query B]
  → cache HIT: reuse KV for the 512-token prefix
  → compute KV for query-B tokens only
```

The savings scale with prefix length. A 2048-token system prompt shared across 1000 requests avoids 2048 × 1000 KV computations.

---

## 3. Requirements

- **Exact prefix match:** The cached prefix must be byte-identical. Even a single token difference misses the cache.
- **Static prefix ordering:** The shared prefix must appear at the start of every prompt — variable prefixes cannot be cached.
- **KV cache storage:** Cached prefixes occupy GPU or CPU memory. Requires a cache eviction policy (typically LRU).

---

## 4. Relation to Paged Attention

[Paged Attention](paged_attention.md) manages KV-cache memory in fixed-size pages (like OS virtual memory) to eliminate fragmentation and enable flexible allocation across requests.

Prefix caching layers on top: shared prefixes are stored as shared physical pages, copy-on-write when a request begins to diverge. vLLM implements both together — prefix caching determines which pages to share; paged attention manages their physical allocation.

---

## 5. Where It Helps Most

| Use Case | Shared Prefix | Speedup |
|---|---|---|
| Chatbots with long system prompts | System prompt repeated every turn | High |
| RAG with fixed context | Retrieved documents identical across retries | High |
| Few-shot inference | Same examples prepended to every query | High |
| Multi-turn conversations | Previous turns are a growing shared prefix | Moderate |
| Unique one-off queries | No shared prefix | None |

---

## 6. Framework Support

| Framework | Prefix Caching Support |
|---|---|
| vLLM | Yes — automatic prefix caching with paged attention |
| TGI (Text Generation Inference) | Yes |
| TensorRT-LLM | Yes |
| Anthropic API | Yes — prompt caching API (explicit cache control) |
| OpenAI API | Automatic for prompts >1024 tokens |

---

## 7. Limitations

- Cache only valid while the prefix is in memory — evicted under memory pressure
- Adds complexity to cache management (eviction, invalidation)
- No benefit for streaming or single-use prompts
- Cache hit rates depend on traffic patterns — workloads with high prefix diversity get little benefit
