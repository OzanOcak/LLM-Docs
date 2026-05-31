# Sparse Attention

## The Efficient Reader Analogy

Imagine you're reading a long book. You don't carefully read every single word on every page—you skim, you focus on important sections, you jump to the index when needed. Reading every word on every page would take forever. That's sparse attention: instead of having every word look at every other word (which gets impossibly expensive for long texts), we design patterns where each word only attends to a carefully chosen subset.

In LLMs, sparse attention is the key to handling very long contexts. Standard attention grows quadratically with sequence length—a 100K token context would need 10 billion attention computations! Sparse attention patterns make longer contexts computationally feasible, enabling models to process entire books or huge documents.

---

## What Is Sparse Attention?

### The Core Idea

Sparse attention restricts each token to attend to only a subset of other tokens, chosen by a fixed pattern.

```text

Dense attention (all pairs):        Sparse attention (selected pairs):
    [A] [B] [C] [D] [E]                 [A] [B] [C] [D] [E]
[A]  1   1   1   1   1              [A]  1   1   0   0   0
[B]  1   1   1   1   1              [B]  1   1   1   0   0
[C]  1   1   1   1   1              [C]  0   1   1   1   0
[D]  1   1   1   1   1              [D]  0   0   1   1   1
[E]  1   1   1   1   1              [E]  0   0   0   1   1

    O(n²) attention                     O(n) or O(n log n) attention
    (too expensive for long sequences)   (feasible for long sequences)
```

```python

def sparse_intro():
    """
    The basic concept of sparse attention
    """
    print("Sparse Attention: Being Selective")
    print("=" * 60)

    n = 100000  # 100K tokens
    dense_cost = n * n
    sparse_cost = n * 100  # If each token attends to 100 others

    print(f"Sequence length: {n:,} tokens")
    print(f"Dense attention comparisons: {dense_cost:,} ({dense_cost/e:.2e})")
    print(f"Sparse attention comparisons: {sparse_cost:,} ({sparse_cost/e:.2e})")
    print(f"Speedup: {dense_cost/sparse_cost:,.0f}x!")

    print("\nSparse attention makes long contexts POSSIBLE.")

sparse_intro()
```

---

## The Quadratic Problem

### Why Dense Attention Fails for Long Sequences

```python

def quadratic_problem():
    """
    The computational cost of dense attention
    """
    print("The Quadratic Problem: Why We Need Sparsity")
    print("=" * 60)

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    print("Seq Length | Attention Pairs | Memory (GB) | Feasible?")
    print("-" * 60)

    for L in seq_lengths:
        pairs = L * L
        memory_gb = (pairs * 2) / (1024**3)  # Rough estimate, 2 bytes per pair
        feasible = "✓" if memory_gb < 80 else "✗ (too big)"
        print(f"  {L:7,d} | {pairs:12,d} |    {memory_gb:6.2f} GB |   {feasible}")

    print("\nFor 131K tokens (a book chapter):")
    print("• 17 billion attention pairs")
    print("• 32+ GB just for attention scores")
    print("• Impossible on single GPU!")
    print("\nSparse attention is the solution.")

quadratic_problem()
```

---

## Sparse Attention Patterns

### 1. Sliding Window Attention

Each token attends only to nearby tokens within a window.

```python

def sliding_window():
    """
    Sliding window attention
    """
    print("Sliding Window (Local) Attention")
    print("=" * 60)

    print("""
    Pattern: Each token attends to W neighbors on each side

    Window size = 2:

    Tokens: A    B    C    D    E    F    G
            ↓    ↓    ↓    ↓    ↓    ↓    ↓
    A sees: A,B,C
    B sees: A,B,C,D
    C sees: A,B,C,D,E
    D sees: B,C,D,E,F
    E sees: C,D,E,F,G
    F sees: D,E,F,G
    G sees: E,F,G

    Complexity: O(n × W) instead of O(n²)
    Used in: Mistral, Longformer
    """)

sliding_window()
```

### 2. Dilated (Strided) Sliding Window

Like sliding window, but with gaps to capture longer-range dependencies.

```python

def dilated():
    """
    Dilated sliding window
    """
    print("Dilated (Strided) Sliding Window")
    print("=" * 60)

    print("""
    Regular window (size 3):           Dilated window (size 3, dilation 2):
    [A,B,C]                              [A, _, C]

    Tokens: A B C D E F G H I          Tokens: A B C D E F G H I
            ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓                  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    E sees: C,D,E,F,G                    E sees: C, _, E, _, G

    Benefits:
    • Wider receptive field with same compute
    • Can capture longer-range patterns
    • Multiple dilated layers = exponential receptive field
    """)

dilated()
```

### 3. Global Attention

Some tokens (like [CLS]) attend to everything, and everything attends to them.

```python

def global_attention():
    """
    Global attention tokens
    """
    print("Global Attention Tokens")
    print("=" * 60)

    print("""
    Designate certain tokens as "global":

    [CLS] The cat sat on the mat
      ↓    ↓   ↓   ↓   ↓   ↓   ↓
    Global token (G) attends to ALL tokens
    ALL tokens attend to global token(s)

    Pattern matrix:
         G  T  c  s  o  t  m
    G:   1  1  1  1  1  1  1   (global sees all)
    T:   1  1  0  0  0  0  0   (local sees global + local)
    c:   1  1  1  1  0  0  0
    s:   1  0  1  1  1  0  0
    ...

    Used in: Longformer, BigBird
    """)

global_attention()
```

### 4. Random Attention

Randomly select a few tokens to attend to, ensuring information can flow across the sequence.

```python

def random_attention():
    """
    Random attention
    """
    print("Random Attention")
    print("=" * 60)

    print("""
    Each token attends to R random tokens.

    Why random?
    • Guarantees connectivity
    • Any token can reach any other via random hops
    • Works well with other patterns

    BigBird combines:
    • Sliding window (local)
    • Global tokens (few)
    • Random attention (long-range connectivity)

    This achieves dense-like performance with O(n) compute!
    """)

random_attention()
```

---

## Sparse Attention Patterns Visualized

```python

def patterns_visualized():
    """
    Visual comparison of attention patterns
    """
    print("Sparse Attention Patterns Compared")
    print("=" * 60)

    print("""
    Dense (n=8, O(n²)):
    [1 1 1 1 1 1 1 1]
    [1 1 1 1 1 1 1 1]
    [1 1 1 1 1 1 1 1]
    [1 1 1 1 1 1 1 1]
    [1 1 1 1 1 1 1 1]
    [1 1 1 1 1 1 1 1]
    [1 1 1 1 1 1 1 1]
    [1 1 1 1 1 1 1 1]

    Sliding Window (w=3, O(n)):
    [1 1 1 0 0 0 0 0]
    [1 1 1 1 0 0 0 0]
    [1 1 1 1 1 0 0 0]
    [0 1 1 1 1 1 0 0]
    [0 0 1 1 1 1 1 0]
    [0 0 0 1 1 1 1 1]
    [0 0 0 0 1 1 1 1]
    [0 0 0 0 0 1 1 1]

    BigBird (combination):
    [1 1 1 0 1 0 1 0]  (window + global + random)
    [1 1 1 1 0 1 0 1]
    [1 1 1 1 1 0 1 0]
    [0 1 1 1 1 1 0 1]
    [1 0 1 1 1 1 1 0]
    [0 1 0 1 1 1 1 1]
    [1 0 1 0 1 1 1 1]
    [1 1 0 1 0 1 1 1]
    """)

patterns_visualized()
```

---

## Sparse Attention in Practice

### Longformer

```python

def longformer():
    """
    Longformer's attention pattern
    """
    print("Longformer: Combining Local and Global")
    print("=" * 60)

    print("""
    Longformer attention:

    1. Sliding window (local)
       • Each token attends to neighbors
       • Window size typically 512

    2. Global attention (task-specific)
       • Certain tokens (e.g., [CLS]) attend everywhere
       • Everything attends to them

    Result: Can process up to 4,096 tokens (original)
    Later: 32k+ tokens
    """)

longformer()
```

### BigBird

```python

def bigbird():
    """
    BigBird's attention pattern
    """
    print("BigBird: The Best of All Worlds")
    print("=" * 60)

    print("""
    BigBird combines three patterns:

    1. Sliding window (local attention)
       • Captures local context

    2. Global tokens
       • Few tokens attend to all
       • All attend to them

    3. Random attention
       • Each token attends to random others
       • Ensures connectivity

    Theoretical guarantee: This sparse pattern
    approximates dense attention arbitrarily well!

    Can process sequences up to 16,384 tokens.
    """)

bigbird()
```

### Mistral's Sliding Window

```python

def mistral():
    """
    Mistral's sliding window attention
    """
    print("Mistral: Sliding Window Attention")
    print("=" * 60)

    print("""
    Mistral uses sliding window attention:

    • Window size: 4,096 tokens
    • Each token attends to 4,096 neighbors
    • With 32 layers, effective receptive field ~131k

    How? Information propagates:
    Layer 1: See 4k neighbors
    Layer 2: See 4k neighbors of those → 8k
    Layer 3: → 12k
    ...
    Layer 32: → 131k tokens!

    This is why Mistral can handle long contexts
    with only O(n) compute per layer!
    """)

mistral()
```

---

## Sparse vs Dense Attention

| Aspect                 | Dense Attention | Sparse Attention                      |
| ---------------------- | --------------- | ------------------------------------- |
| Complexity             | O(n²)           | O(n) or O(n log n)                    |
| Max context            | ~8k (practical) | 32k to 1M+                            |
| Information flow       | Complete        | Pattern-dependent                     |
| Implementation         | Simple          | Complex                               |
| Hardware efficiency    | Good on GPUs    | May be slower due to irregular access |
| Theoretical guarantees | Perfect         | Approximate                           |

### When to Use Sparse Attention

```python

def when_sparse():
    """
    When to choose sparse attention
    """
    print("When to Use Sparse Attention")
    print("=" * 60)

    use_cases = [
        "Processing long documents (books, papers)",
        "Code with long contexts",
        "Multi-turn conversations",
        "Retrieval-augmented generation (RAG)",
        "Any task needing >8k context"
    ]

    print("Sparse attention is essential for:")
    for case in use_cases:
        print(f"  • {case}")

when_sparse()
```

---

## Why This Matters for LLMs

### 1. Enabling Long Context Windows

```python

def long_context():
    """
    Models with long context windows
    """
    print("Models with Long Context Windows")
    print("=" * 60)

    models = {
        "GPT-3": "2,048 (dense)",
        "GPT-4": "32,000-128,000 (sparse?)",
        "Claude": "100,000-200,000",
        "Gemini": "1,000,000 (1M)",
        "Mistral": "32,000 (sliding window)",
        "Longformer": "4,096-32,000",
        "BigBird": "16,384"
    }

    for model, context in models.items():
        print(f"  • {model}: {context} tokens")

long_context()
```

### 2. The Receptive Field Effect

```python

def receptive_field():
    """
    How sparse attention still captures long-range info
    """
    print("The Receptive Field Effect")
    print("=" * 60)

    print("""
    Even with local attention, deep networks can see far:

    Layer 1: sees tokens within 512
    Layer 2: sees tokens within 1024 (via Layer 1)
    Layer 3: sees tokens within 1536
    ...
    Layer 32: sees tokens within 16,384!

    This is why sliding window works—depth compensates.

    Each layer expands the receptive field by window size.
    """)

receptive_field()
```

### 3. Hardware Challenges

```python

def hardware():
    """
    GPU efficiency challenges
    """
    print("Hardware Challenges with Sparse Attention")
    print("=" * 60)

    print("""
    Sparse attention sounds great, but:

    • GPUs are optimized for dense matrices
    • Irregular patterns = poor memory locality
    • Custom kernels needed (FlashAttention, xformers)

    FlashAttention solves this by:
    • Tiling attention computation
    • Not materializing the full attention matrix
    • Still O(n²) but constant factor smaller

    Newer: FlashAttention-2, FlashAttention-3
    """)

hardware()
```

### 4. Theoretical Limits

```python

def theoretical():
    """
    Theoretical properties of sparse attention
    """
    print("Theoretical Properties")
    print("=" * 60)

    print("""
    Research questions:

    • How sparse can we go without losing performance?
    • What patterns preserve expressivity?
    • Can we learn attention patterns instead of fixing them?

    BigBird proved: With sliding window + global + random,
    you can approximate any dense attention function.

    But in practice, learned sparse patterns (Sparse Transformer)
    sometimes work better than fixed.

    Open area of research!
    """)

theoretical()
```

---

## Sparse Attention Cheat Sheet

| Pattern         | Complexity | Range          | Used In             |
| --------------- | ---------- | -------------- | ------------------- |
| Sliding window  | O(n × w)   | Local          | Mistral, Longformer |
| Dilated sliding | O(n × w)   | Extended local | Sparse Transformer  |
| Global tokens   | O(n × g)   | Full           | Longformer, BigBird |
| Random          | O(n × r)   | Random         | BigBird             |
| Combined        | O(n)       | Global         | BigBird             |
| Learnable       | O(n log n) | Adaptive       | Routing Transformer |

---

## Quick Recap

• Sparse attention restricts each token to attend to a subset of others—like an efficient reader who skims rather than reading every word, making long contexts computationally feasible

• Common patterns include sliding window (local), global tokens, and random attention—each providing different connectivity, often combined for best results

• Sparse attention enables today's long-context models—Claude's 200K, Gemini's 1M contexts would be impossible without sparsity, transforming what we can do with LLMs

---

## Mental Hook

> "Sparse attention is like being in a huge conference—you don't try to talk to everyone, but you chat with neighbors, a few key people, and random connections, and somehow through this network, information still reaches everyone who needs it."
