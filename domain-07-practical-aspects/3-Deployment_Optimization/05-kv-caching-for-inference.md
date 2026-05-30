# KV Caching for Inference

## The Reusable Scaffolding Analogy

Imagine you're building a house floor by floor. Instead of dismantling the scaffolding after each floor and rebuilding it from scratch for the next, you leave it in place and simply add new scaffolding for the new floor. That's KV caching for LLMs: during generation, we reuse the Key and Value vectors from previous tokens instead of recomputing them for every new token.

In LLM inference, KV caching is the single most important optimization. Without it, generating a 1000-token response would be 500,000x slower (quadratic complexity). With caching, it's linear in sequence length—the difference between waiting minutes and getting instant responses.

---

## The Problem: Redundant Computation

### Without Caching, We Recompute Everything

```python

def without_caching():
    """
    What happens without KV caching
    """
    print("Without KV Caching: Recomputing Everything")
    print("=" * 60)

    print("""
    Generating "The cat sat" token by token:

    Step 1: Input: [The]
            Compute attention for all positions: [The]

    Step 2: Input: [The, cat]
            Compute attention for ALL positions again: [The, cat]
            └── We recomputed The's attention from scratch! 😱

    Step 3: Input: [The, cat, sat]
            Compute attention for ALL positions again: [The, cat, sat]
            └── Recomputed The and cat again! 😱😱

    For N tokens, we do O(N²) attention computations!
    """)

without_caching()
```

### The Quadratic Cost

```python

def quadratic_cost():
    """
    The computational cost without caching
    """
    print("The Quadratic Problem")
    print("=" * 60)

    import math

    tokens = [10, 100, 1000, 10000, 100000]

    print("Tokens | Attention Computations (without cache)")
    print("-" * 45)

    for n in tokens:
        # Without cache: sum of squares
        computations = n * (n + 1) // 2  # 1 + 2 + 3 + ... + n
        print(f"{n:6,d} | {computations:15,d}")

    print("\nFor 1000 tokens: 500,500 computations")
    print("For 10000 tokens: 50 million computations!")
    print("This doesn't scale!")

quadratic_cost()
```

---

## The Solution: KV Caching

### What We Cache

In transformer attention, we compute three matrices for each token:

- Query (Q): What this token is looking for
- Key (K): What information this token has
- Value (V): The actual information content

For generation, previous tokens' Keys and Values don't change—so we cache them!

```python

def kv_cache_intro():
    """
    The basic concept of KV caching
    """
    print("KV Caching: Don't Recompute the Past")
    "=" * 60

    print("""
    With KV caching:

    Step 1: Input: [The]
            Compute K1, V1 for "The"
            Store in cache: K=[K1], V=[V1]

    Step 2: Input: [cat]
            Compute K2, V2 for "cat"
            Use cached K1, V1 for "The"
            Update cache: K=[K1, K2], V=[V1, V2]

    Step 3: Input: [sat]
            Compute K3, V3 for "sat"
            Use cached K1,K2 and V1,V2
            Update cache: K=[K1,K2,K3], V=[V1,V2,V3]

    For N tokens: O(N) total computations! 🎉
    """)

kv_cache_intro()
```

### Visualizing the Cache

```text

Without Cache:                    With KV Cache:

Step 1:                           Step 1:
[The]─────┐                       [The]─────┐
    Compute all                    Compute new
    │                               │
    ▼                               ▼
[The]                            Store K1,V1 in cache

Step 2:                           Step 2:
[The, cat]─────┐                  [cat]─────┐
    Compute all                    Compute new
    │                               │
    ▼                               ▼
[The, cat]                       Use cached K1,V1 + new K2,V2

Step 3:                           Step 3:
[The, cat, sat]─────┐             [sat]─────┐
    Compute all                    Compute new
    │                               │
    ▼                               ▼
[The, cat, sat]                  Use cached K1,K2 + new K3,V3

Cache size grows linearly with sequence length.
Computation stays constant per new token!
```

---

## How KV Caching Works in Practice

### Step-by-Step Generation

```python

def generation_with_cache():
    """
    Detailed generation with KV cache
    """
    print("Generation with KV Cache: Step by Step")
    "=" * 60

    # Initialize
    prompt = "The cat sat"
    cache_k = []  # Keys cache
    cache_v = []  # Values cache

    print(f"Prompt: '{prompt}'\n")

    # Step 1: Process prompt (prefill)
    print("Step 1 (Prefill): Process all prompt tokens")
    tokens = prompt.split()
    for i, token in enumerate(tokens):
        print(f"  Token {i+1}: '{token}'")
        print(f"    Compute K{i+1}, V{i+1}")
        cache_k.append(f"K{i+1}")
        cache_v.append(f"V{i+1}")

    print(f"\nCache after prefill: K={cache_k}, V={cache_v}")

    # Step 2: Generate next tokens one by one
    print("\n" + "=" * 40)
    print("Step 2 (Generation): Generate new tokens")
    print("=" * 40)

    new_tokens = ["on", "the", "mat"]
    for i, token in enumerate(new_tokens, start=len(tokens)+1):
        print(f"\nGenerating token {i}: '{token}'")
        print(f"  Current cache size: {len(cache_k)} tokens")
        print(f"  Compute K{i}, V{i} for '{token}'")
        print(f"  Use cached K1..K{i-1} and V1..V{i-1} for attention")
        cache_k.append(f"K{i}")
        cache_v.append(f"V{i}")
        print(f"  Cache now: {len(cache_k)} tokens")

generation_with_cache()
```

### Memory Usage

```python

def cache_memory():
    """
    Memory requirements for KV cache
    """
    print("KV Cache Memory Requirements")
    "=" * 60

    # Llama 70B specs
    layers = 80
    hidden_size = 8192
    batch_size = 1

    # Memory per token (16-bit precision)
    bytes_per_param = 2  # FP16
    kv_size = 2 * layers * hidden_size * bytes_per_param  # K and V

    print(f"Model: Llama 70B (80 layers, {hidden_size} hidden)")
    print(f"KV cache size per token: ~{kv_size/1e6:.1f} MB\n")

    for seq_len in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        cache_memory = seq_len * kv_size / 1e9  # GB
        print(f"Seq length {seq_len:6,d}: {cache_memory:.2f} GB")

    print("\n⚠️ KV cache grows linearly with sequence length!")
    print("For 32K context, cache needs ~13GB of memory.")

cache_memory()
```

---

## KV Cache Implementation

### Simple Python Simulation

```python

import numpy as np

def kv_cache_simulation():
    """
    Simulate KV caching during generation
    """
    print("KV Cache Simulation")
    "=" * 60

    class KVAttention:
        def __init__(self):
            self.k_cache = []
            self.v_cache = []

        def forward_with_cache(self, x):
            """
            Simplified attention with KV cache
            x: new token embedding
            """
            # Simulate computing K and V for new token
            k_new = np.random.randn(4)  # Simplified
            v_new = np.random.randn(4)

            # Add to cache
            self.k_cache.append(k_new)
            self.v_cache.append(v_new)

            # Attention would use all cached K,V
            print(f"  New token: cache now has {len(self.k_cache)} entries")
            print(f"  Would attend to all {len(self.k_cache)} previous tokens")

            return "next_token"  # Simulated output

    # Simulate generation
    attn = KVAttention()
    prompt_tokens = ["The", "cat", "sat"]

    print("Prefilling cache with prompt:")
    for token in prompt_tokens:
        attn.forward_with_cache(token)

    print("\nGenerating new tokens:")
    for i in range(3):
        attn.forward_with_cache(f"new_token_{i+1}")

kv_cache_simulation()
```

### Real Implementation (Conceptual)

```python

def real_implementation():
    """
    How KV cache is implemented in practice
    """
    print("Real KV Cache Implementation")
    "=" * 60

    print("""
    # PyTorch-style pseudo-code

    class CausalAttention(nn.Module):
        def __init__(self):
            self.k_cache = None
            self.v_cache = None

        def forward(self, x, use_cache=False):
            # x: (batch, seq_len, hidden)

            # Compute Q, K, V for current input
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            if use_cache and self.k_cache is not None:
                # Concatenate with cache
                k = torch.cat([self.k_cache, k], dim=1)
                v = torch.cat([self.v_cache, v], dim=1)

            # Update cache
            self.k_cache = k
            self.v_cache = v

            # Attention computation with all keys/values
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            attn_weights = attn_weights / math.sqrt(k.size(-1))
            attn_weights = torch.softmax(attn_weights, dim=-1)

            out = torch.matmul(attn_weights, v)
            return out
    """)

real_implementation()
```

---

## Performance Impact

### Speed Comparison

```python

def speed_comparison():
    """
    With vs without KV cache
    """
    print("Performance: With vs Without KV Cache")
    "=" * 60

    import math

    seq_len = 1000
    attention_cost_per_token = 1  # arbitrary unit

    # Without cache: O(n²)
    without_cache = sum(i * attention_cost_per_token for i in range(1, seq_len + 1))

    # With cache: O(n)
    with_cache = seq_len * attention_cost_per_token

    print(f"Generating {seq_len} tokens:")
    print(f"  Without cache: {without_cache:,} units")
    print(f"  With cache:    {with_cache:,} units")
    print(f"  Speedup:       {without_cache/with_cache:.1f}x")

    print("\nReal-world numbers (Llama 70B):")
    print("  Without cache: 10-20 seconds per token → 3-6 hours for 1000 tokens")
    print("  With cache:    50-100ms per token → 50-100 seconds for 1000 tokens")

speed_comparison()
```

### Memory vs Speed Tradeoff

| Seq Length | Without Cache (Time) | With Cache (Time) | Cache Memory |
| ---------- | -------------------- | ----------------- | ------------ |
| 512        | 2x                   | 1x                | ~2GB         |
| 1024       | 4x                   | 1x                | ~4GB         |
| 2048       | 8x                   | 1x                | ~8GB         |
| 4096       | 16x                  | 1x                | ~16GB        |
| 8192       | 32x                  | 1x                | ~32GB        |
| 32768      | 128x                 | 1x                | ~128GB       |

---

## KV Cache in Production

### Continuous Batching

```python

def continuous_batching():
    """
    KV caching enables continuous batching
    """
    print("Continuous Batching with KV Cache")
    "=" * 60

    print("""
    In production systems, KV cache enables continuous batching:

    Request 1: [The, cat] ──► cache: K1,K2,V1,V2
                  │
                  ▼
    Request 2: [Once, upon] ──► cache: K3,K4,V3,V4
                  │
                  ▼
    Request 1 continues: [sat] ──► uses cache + new K5,V5
                  │
                  ▼
    Request 3: [In, the] ──► cache: K6,K7,V6,V7

    The cache keeps growing for each ongoing request,
    allowing efficient batch processing!
    """)

continuous_batching()
```

### PagedAttention (vLLM)

```python

def paged_attention():
    """
    PagedAttention: Solving cache fragmentation
    """
    print("PagedAttention: Virtual Memory for KV Cache")
    "=" * 60

    print("""
    Problem: KV cache is large and causes memory fragmentation.

    Solution: PagedAttention (vLLM) treats KV cache like OS virtual memory:

    ┌─────────────────────────────────────────┐
    │ Logical KV Cache (per request)          │
    │ ┌────┬────┬────┬────┬────┬────┬────┐   │
    │ │K1,V1│K2,V2│K3,V3│K4,V4│K5,V5│...│   │
    │ └────┴────┴────┴────┴────┴────┴────┘   │
    └─────────────────────────────────────────┘
                       ↓
    ┌─────────────────────────────────────────┐
    │ Physical memory blocks                   │
    │ ┌────────┐ ┌────────┐ ┌────────┐        │
    │ │Block 1 │ │Block 2 │ │Block 3 │  ...   │
    │ │K1,V1,  │ │K3,V3,  │ │K2,V2,  │        │
    │ │K5,V5   │ │K7,V7   │ │K4,V4   │        │
    │ └────────┘ └────────┘ └────────┘        │
    └─────────────────────────────────────────┘

    Benefits:
    • 2-4x higher throughput
    • Better memory utilization
    • Support for larger batches
    """)

paged_attention()
```

---

## KV Cache Cheat Sheet

| Aspect              | Without Cache | With Cache |
| ------------------- | ------------- | ---------- |
| Time complexity     | O(n²)         | O(n)       |
| Memory per request  | O(n)          | O(n)       |
| Batch efficiency    | Poor          | Excellent  |
| Long context        | Impossible    | Possible   |
| Implementation      | Simple        | Complex    |
| Continuous batching | No            | Yes        |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  KV caching isn't just an optimization—it's what makes      ║
║  LLMs usable in the real world:                             ║
║                                                              ║
║  🚀 SPEED: Without caching, generating a 1000-token         ║
║     response would take 500,000x more computation.          ║
║     With caching, it's linear and fast.                     ║
║                                                              ║
║  💰 COST: 500,000x less compute means 500,000x less        ║
║     cost. This is the difference between a demo and         ║
║     a profitable business.                                  ║
║                                                              ║
║  📱 REAL-TIME APPS: Chat, coding assistants, and           ║
║     interactive applications depend on fast generation.     ║
║                                                              ║
║  🔄 CONTINUOUS BATCHING: KV cache enables serving           ║
║     thousands of concurrent users efficiently.              ║
║                                                              ║
║  🌍 LONG CONTEXT: Modern 100K+ context windows             ║
║     would be impossible without KV cache optimizations.     ║
║                                                              ║
║  Every time you use ChatGPT and get an instant response,    ║
║  KV caching is working behind the scenes.                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• KV caching stores Key and Value vectors from previous tokens during generation—like leaving scaffolding in place instead of rebuilding it for each new floor, avoiding redundant computation

• Without caching, generation would be O(n²) and impractical—for 1000 tokens, 500,000x more computation than with caching

• Modern optimizations like PagedAttention (vLLM) make KV caching even more efficient—treating the cache like virtual memory to reduce fragmentation and increase throughput

---

## Mental Hook

> "KV caching is like keeping your notes from previous chapters while writing a book—you don't reread the whole book every time you write a new sentence, you just glance at your notes and keep writing."
