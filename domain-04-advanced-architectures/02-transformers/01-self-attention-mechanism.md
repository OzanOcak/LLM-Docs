# Self-Attention Mechanism

## The Committee Meeting Analogy

Imagine a committee discussing a topic. Each member comes in with their own perspective. During the discussion, they listen to others and update their own understanding based on what they hear. The marketing expert pays attention to the sales expert when discussing revenue, but focuses on the legal expert when discussing regulations. That's self-attention: each word in a sentence looks at all other words, decides how much attention to pay to each, and updates its own representation accordingly.

In modern LLMs, self-attention is the core innovation of the Transformer architecture. It allows every word to directly interact with every other word, capturing complex relationships regardless of their distance in the sentence. This is why models like GPT can understand long-range dependencies so effectively.

---

## What Is Self-Attention?

### The Core Idea

Self-attention lets each word in a sequence look at all other words and decide how much to focus on each.

```text

Traditional attention:                    Self-attention:
Decoder looks at encoder                   Each word looks at ALL words
                                           (including itself!)
    ↓                                               ↓
Query from decoder      ←→      Keys from encoder   Each word computes:
                                                    Query, Key, Value from itself
```

```python

def self_attention_intro():
    """
    The basic concept of self-attention
    """
    print("Self-Attention: Words Looking at Words")
    print("=" * 60)

    sentence = "The animal didn't cross the street because it was tired"

    print(f"Sentence: '{sentence}'")
    print("\nWhat does 'it' refer to?")
    print("Self-attention lets 'it' look at ALL other words:")
    print("""
    'it' attention weights:
    The:     0.02
    animal:  0.65  ←── Most attention here!
    didn't:  0.05
    cross:   0.08
    street:  0.03
    because: 0.02
    was:     0.01
    tired:   0.14

    The model learns that 'it' likely refers to 'animal'
    """)

    print("\nEach word gets a NEW representation that includes")
    print("context from the whole sentence!")

self_attention_intro()
```

---

## The Three Roles: Query, Key, Value

### The Library Analogy

To understand self-attention, think of a library search:

- Query: What you're looking for (your question)

- Key: The catalog entry for each book (what each book is about)

- Value: The actual book content (the information you'll get)

```python

def qkv_intro():
    """
    Explaining Query, Key, Value
    """
    print("Query, Key, Value: The Library Analogy")
    print("=" * 60)

    print("""
    For each word, we create three vectors:

    Query (Q): "What am I looking for?"
    Key (K):   "What information do I have?"
    Value (V): "What information do I contain?"

    Then:
    1. Compare Query with all Keys (compatibility scores)
    2. Use scores to weight the Values
    3. Sum weighted Values to get new representation
    """)

    print("\nAnalogy: You (Query) search the library catalog (Keys)")
    print("to find relevant books (Values) to check out.")

qkv_intro()
```

### Concrete Example

```python

def qkv_example():
    """
    Concrete example of QKV for a sentence
    """
    print("QKV in Action: 'The cat sat'")
    print("=" * 60)

    words = ["The", "cat", "sat"]

    print(f"Sentence: {' → '.join(words)}")
    print("\nEach word has its own Q, K, V (simplified):")

    # Simplified vectors (in reality, these are learned)
    qkv = {
        "The": {"Q": [0.1, 0.2], "K": [0.1, 0.1], "V": [0.3, 0.1]},
        "cat": {"Q": [0.8, 0.7], "K": [0.8, 0.8], "V": [0.9, 0.7]},
        "sat": {"Q": [0.3, 0.9], "K": [0.4, 0.9], "V": [0.5, 0.8]}
    }

    for word, vectors in qkv.items():
        print(f"\n  '{word}':")
        print(f"    Q: {vectors['Q']} (what I'm looking for)")
        print(f"    K: {vectors['K']} (what I offer)")
        print(f"    V: {vectors['V']} (what I contain)")

    print("\nNow 'cat' will look at all words:")
    print("  Compare cat's Q with everyone's K → attention weights")
    print("  Weighted sum of everyone's V → new 'cat' representation")

qkv_example()
```

---

## Self-Attention Step by Step

### Step 1: Create Q, K, V for Each Word

```python

def self_attention_step1():
    """
    Step 1: Create queries, keys, values
    """
    print("Step 1: Creating Q, K, V")
    print("=" * 60)

    # Input embeddings for each word
    embeddings = {
        "The": [1.0, 0.0, 1.0],
        "cat": [0.0, 1.0, 1.0],
        "sat": [1.0, 1.0, 0.0]
    }

    # Learned weight matrices (simplified)
    W_Q = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
    W_K = [[0.3, 0.0, 0.0], [0.0, 0.3, 0.0]]
    W_V = [[0.7, 0.0, 0.0], [0.0, 0.7, 0.0]]

    print("Input embeddings:")
    for word, emb in embeddings.items():
        print(f"  '{word}': {emb}")

    print("\nComputing Q = embedding × W_Q:")
    for word, emb in embeddings.items():
        q0 = emb[0]*W_Q[0][0] + emb[1]*W_Q[0][1] + emb[2]*W_Q[0][2]
        q1 = emb[0]*W_Q[1][0] + emb[1]*W_Q[1][1] + emb[2]*W_Q[1][2]
        print(f"  '{word}' Q: [{q0:.1f}, {q1:.1f}]")

self_attention_step1()
```

### Step 2: Calculate Attention Scores

```python

def self_attention_step2():
    """
    Step 2: Calculate attention scores
    """
    print("Step 2: Computing Attention Scores")
    print("=" * 60)

    # Q and K vectors (simplified)
    Q = {
        "The": [0.5, 0.0],
        "cat": [0.0, 0.5],
        "sat": [0.5, 0.5]
    }

    K = {
        "The": [0.3, 0.0],
        "cat": [0.0, 0.3],
        "sat": [0.3, 0.3]
    }

    print("For 'cat' looking at all words:")
    q_cat = Q["cat"]
    print(f"  cat's Q: {q_cat}")

    scores = {}
    for word, k in K.items():
        # Dot product score
        score = q_cat[0]*k[0] + q_cat[1]*k[1]
        scores[word] = score
        print(f"  score with '{word}' (K={k}): {score:.2f}")

    print("\nThese scores say how much 'cat' should attend to each word.")

self_attention_step2()
```

### Step 3: Apply Softmax (Normalize Scores)

```python

import math

def self_attention_step3():
    """
    Step 3: Softmax to get attention weights
    """
    print("Step 3: Softmax - Making Scores Sum to 1")
    print("=" * 60)

    # Raw scores from previous step
    scores = {
        "The": 0.15,
        "cat": 0.15,
        "sat": 0.30
    }

    print("Raw scores (for 'cat'):")
    total_raw = sum(scores.values())
    for word, score in scores.items():
        print(f"  '{word}': {score:.2f}")

    # Apply softmax
    exp_scores = {w: math.exp(s) for w, s in scores.items()}
    sum_exp = sum(exp_scores.values())

    print("\nAfter softmax (probabilities):")
    for word, exp_s in exp_scores.items():
        weight = exp_s / sum_exp
        print(f"  '{word}': {weight:.3f}")

    print(f"\nSum of weights: {sum(exp_s.values())/sum_exp:.1f} ✓")

self_attention_step3()
```

### Step 4: Weighted Sum of Values

```python

def self_attention_step4():
    """
    Step 4: Weighted sum of values
    """
    print("Step 4: Creating New Representations")
    print("=" * 60)

    # Attention weights (from softmax)
    weights = {
        "The": 0.20,
        "cat": 0.30,
        "sat": 0.50
    }

    # Value vectors
    V = {
        "The": [0.7, 0.0],
        "cat": [0.0, 0.7],
        "sat": [0.5, 0.5]
    }

    print("Attention weights for 'cat':")
    for word, w in weights.items():
        print(f"  '{word}': {w:.2f} × V{word}")

    # Compute weighted sum
    new_cat = [0, 0]
    for word, w in weights.items():
        v = V[word]
        new_cat[0] += w * v[0]
        new_cat[1] += w * v[1]

    print(f"\nNew 'cat' representation: [{new_cat[0]:.2f}, {new_cat[1]:.2f}]")
    print("This now contains context from the whole sentence!")

self_attention_step4()
```

---

## The Complete Self-Attention Formula

```python

def self_attention_formula():
    """
    The mathematical formulation
    """
    print("Self-Attention: The Complete Formula")
    print("=" * 60)

    print("""
    Attention(Q, K, V) = softmax( QK^T / √d_k ) V

    Where:
    • Q: Query matrix (each row is a word's query)
    • K: Key matrix (each row is a word's key)
    • V: Value matrix (each row is a word's value)
    • d_k: dimension of keys (for scaling)

    Step-by-step:
    1. Q × K^T : Compute compatibility between all pairs
    2. Scale by 1/√d_k : Prevent dot products from getting too large
    3. Softmax : Convert to probabilities
    4. Multiply by V : Get weighted sum of values
    """)

    print("\nEach word's new representation = weighted sum of all values,")
    print("with weights based on compatibility of its query with their keys.")

self_attention_formula()
```

---

## A Tiny Self-Attention Implementation

```python

import numpy as np

def self_attention_implementation():
    """
    Minimal self-attention implementation
    """
    print("Tiny Self-Attention Implementation")
    print("=" * 60)

    class SelfAttention:
        def __init__(self, d_model):
            self.d_model = d_model
            # Random Q, K, V matrices (in reality, these are learned)
            self.W_q = np.random.randn(d_model, d_model) * 0.1
            self.W_k = np.random.randn(d_model, d_model) * 0.1
            self.W_v = np.random.randn(d_model, d_model) * 0.1

        def forward(self, x):
            """
            x: (seq_len, d_model) input embeddings
            """
            seq_len = x.shape[0]

            # Step 1: Compute Q, K, V
            Q = x @ self.W_q  # (seq_len, d_model)
            K = x @ self.W_k
            V = x @ self.W_v

            print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

            # Step 2: Compute attention scores
            scores = Q @ K.T  # (seq_len, seq_len)
            print(f"\nAttention scores shape: {scores.shape}")

            # Step 3: Scale
            scores = scores / np.sqrt(self.d_model)

            # Step 4: Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            print(f"Attention weights shape: {weights.shape}")
            print("Each row sums to 1")

            # Step 5: Weighted sum of values
            output = weights @ V

            return output, weights

    # Create attention module
    attn = SelfAttention(d_model=4)

    # Example input: 3 words, each 4-dim
    x = np.random.randn(3, 4)
    print(f"Input shape: {x.shape} (3 words, 4-dim embeddings)")

    # Forward pass
    output, weights = attn.forward(x)

    print(f"\nOutput shape: {output.shape}")
    print("Each word now contains context from all words!")

self_attention_implementation()
```

---

## Why Self-Attention Is Powerful

### Advantages Over RNNs

| Aspect                  | RNNs                            | Self-Attention               |
| ----------------------- | ------------------------------- | ---------------------------- |
| Parallelization         | Sequential (slow)               | Parallel (fast)              |
| Long-range dependencies | Difficult (vanishing gradients) | Direct (O(1) path)           |
| Context                 | Compressed in hidden state      | All positions accessible     |
| Interpretability        | Hard to analyze                 | Attention weights show focus |

### The Receptive Field

```text

RNN: Information flows step by step
Word1 → Word2 → Word3 → Word4 → Word5
  ↑       ↑       ↑       ↑       ↑
  └───────┴───────┴───────┴───────┘
  Word5 needs 4 steps to see Word1

Self-Attention: Direct connections
Word1 ←→ Word2 ←→ Word3 ←→ Word4 ←→ Word5
  ↑       ↑       ↑       ↑       ↑
  └───────┴───────┴───────┴───────┘
  Any word sees any other in ONE step!
```

---

## Why This Matters for LLMs

### 1. Self-Attention Is the Heart of Transformers

```python

def transformer_heart():
    """
    Self-attention in transformers
    """
    print("Self-Attention: The Heart of Transformers")
    print("=" * 60)

    print("""
    Transformer block:

    Input Embeddings
           ↓
    [Multi-Head Self-Attention]  ←─── Here!
           ↓
    Add & LayerNorm
           ↓
    Feed-Forward Network
           ↓
    Add & LayerNorm
           ↓
    Output

    Repeated 12 to 96+ times!
    """)

    print("\nEach layer refines representations through self-attention.")

transformer_heart()
```

### 2. Multi-Head Attention

```python

def multi_head():
    """
    Multiple attention heads
    """
    print("Multi-Head Attention: Different Perspectives")
    print("=" * 60)

    print("""
    Instead of one attention mechanism, use multiple:

    Head 1: Focuses on subject-verb relationships
    Head 2: Focuses on adjective-noun relationships
    Head 3: Focuses on coreference (it → animal)
    Head 4: Focuses on positional relationships

    Then concatenate and project:
    MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
    """)

    print("\nEach head learns a different type of relationship!")
    print("This is why Transformers are so powerful.")

multi_head()
```

### 3. Causal Self-Attention (For GPT)

```python

def causal_attention():
    """
    Masked self-attention for generation
    """
    print("Causal Self-Attention: Can't See the Future")
    print("=" * 60)

    print("""
    GPT uses masked self-attention:

    When generating "The cat sat":

    For "The": can only see "The"
    For "cat": can see "The", "cat"
    For "sat": can see "The", "cat", "sat"

    Mask prevents looking at future words:

    Attention weights matrix:
    [1, 0, 0]  # "The" sees only itself
    [1, 1, 0]  # "cat" sees The, cat
    [1, 1, 1]  # "sat" sees all previous
    """)

    print("\nThis lets GPT generate left to right!")

causal_attention()
```

### 4. The Quadratic Cost Challenge

```python

def quadratic_challenge():
    """
    The computational cost of self-attention
    """
    print("The Quadratic Cost Challenge")
    print("=" * 60)

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    print("Sequence Length | Attention Computations | Memory (approx)")
    print("-" * 65)

    for L in seq_lengths:
        computations = L * L  # Each word attends to all words
        memory_gb = (L * L * 2) / (1024**3)  # Rough estimate
        print(f"     {L:5d}      |   {computations:12,d}   |   {memory_gb:.3f} GB")

quadratic_challenge()
```

---

## Self-Attention Variants

| Variant         | Description                              | Used In              |
| --------------- | ---------------------------------------- | -------------------- |
| Multi-Head      | Multiple attention heads in parallel     | All transformers     |
| Causal/Masked   | Prevent looking at future                | GPT, decoder-only    |
| Sparse          | Only attend to local windows             | Longformer, BigBird  |
| Linear          | Approximate attention for long sequences | Performer, Linformer |
| Cross-attention | Attend to different sequence             | Encoder-decoder (T5) |

---

## Quick Recap

• Self-attention lets every word look at every other word in the sequence—like a committee meeting where each member updates their understanding based on what everyone else says

• Each word creates Query (what it's looking for), Key (what it offers), and Value (what it contains)—attention weights are computed by matching Queries with Keys, then used to weight Values

• Self-attention is the core innovation of Transformers—it enables parallel processing, captures long-range dependencies directly, and multi-head attention allows learning different types of relationships, making modern LLMs possible

---

## Mental Hook

> "Self-attention is like a conversation where everyone listens to everyone else, then updates their own understanding based on what they heard—by the end, each person's knowledge contains context from the whole group."
