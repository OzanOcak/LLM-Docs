# Attention Mechanism (Original Version)

## The Interpreter's Notebook Analogy

Imagine a simultaneous interpreter translating a long speech. They don't just listen once and try to remember everything—they keep a notebook with key phrases, and as they translate each sentence, they glance back at relevant parts of the original speech. When translating "the cat," they look at where "cat" appeared; when translating "sat," they look at where "sat" appeared. That's attention: at each step of generation, the model looks back at the input and decides which parts are most relevant right now.

In the evolution toward LLMs, attention was the breakthrough that fixed the bottleneck problem of Seq2Seq models. It allowed models to handle long sequences effectively and laid the groundwork for the Transformer architecture that powers every modern LLM.

---

## The Problem: Seq2Seq's Bottleneck

### Before Attention

```python

def bottleneck_recap():
    """
    The problem attention solved
    """
    print("The Bottleneck Problem (Before Attention)")
    print("=" * 60)

    print("""
    Old Seq2Seq:

    Input: "The cat sat on the mat and then..."
                      ↓
    Encoder ──→ [One Context Vector] ←── Bottleneck!
                      ↓
    Decoder ──→ "Le chat s'est assis..."

    PROBLEM: One vector must capture EVERYTHING.
    For long sentences, details get lost.
    """)

    print("\nAnalogy: Summarizing a whole book into one sentence")
    print("You lose most of the information!")

bottleneck_recap()
```

### The Information Loss

```text

Long sentence: 50 words → One vector (512 numbers)

Each word gets less than 10 numbers to represent it!

At step 50 of decoding, the model has forgotten
what happened at the beginning.
```

---

## The Solution: Attention

### Core Idea

Attention lets the decoder look back at ALL encoder states, not just the final one.

```python

def attention_intro():
    """
    The basic concept of attention
    """
    print("Attention: Looking Back at the Source")
    print("=" * 60)

    print("""
    At each decoding step:

    1. Look at ALL encoder states (every word's representation)
    2. Decide how much to focus on each (attention weights)
    3. Take a weighted average to create a CONTEXT VECTOR
    4. Use this context + decoder state to generate next word

    The context vector is DIFFERENT at each step!
    """)

    print("\nAnalogy: Instead of one summary of the whole book,")
    print("you have a searchable index. When writing page 50,")
    print("you look up only what's relevant right now.")

attention_intro()
```

---

## How Attention Works Step by Step

### The Setup

```python

def attention_step_by_step():
    """
    Detailed walkthrough of attention
    """
    print("Attention: Step by Step")
    print("=" * 60)

    # Encoder states (representations of each input word)
    encoder_states = [
        "h₁ (The)",
        "h₂ (cat)",
        "h₃ (sat)",
        "h₄ (on)",
        "h₅ (the)",
        "h₆ (mat)"
    ]

    # Current decoder state (at step 3 of generation)
    decoder_state = "s₃ (currently translating 'sat')"

    print(f"Encoder states: {encoder_states}")
    print(f"Current decoder state: {decoder_state}")

    print("\n" + "=" * 50)
    print("STEP 1: Compute Attention Scores")
    print("=" * 50)
    print("For each encoder state, calculate relevance to current decoder state:")

    scores = [0.1, 0.7, 0.05, 0.05, 0.05, 0.05]
    words = ["The", "cat", "sat", "on", "the", "mat"]

    for word, score in zip(words, scores):
        print(f"  score({decoder_state}, {word}) = {score}")

    print("\n" + "=" * 50)
    print("STEP 2: Convert to Attention Weights (Softmax)")
    print("=" * 50)
    print("Make scores sum to 1 (probabilities):")

    weights = [0.1, 0.6, 0.06, 0.06, 0.06, 0.12]  # After softmax
    total = sum(weights)

    for word, weight in zip(words, weights):
        print(f"  attention on '{word}': {weight:.2f}")

    print(f"\nSum of weights: {total} ✓")

    print("\n" + "=" * 50)
    print("STEP 3: Create Context Vector")
    print("=" * 50)
    print("Weighted sum of encoder states:")
    print("  context = 0.1×h₁ + 0.6×h₂ + 0.06×h₃ + ...")

    print("\n" + "=" * 50)
    print("STEP 4: Generate Next Word")
    print("=" * 50)
    print("Combine context with decoder state to predict:")
    print("  P(next word) = f(context, decoder_state)")
    print("  Most attention on 'cat' → likely generating 'chat'")

attention_step_by_step()
```

---

## The Math of Attention

### The Three Key Components

```python

def attention_math():
    """
    The mathematical formulation
    """
    print("The Mathematics of Attention")
    print("=" * 60)

    print("""
    For each decoding step t:

    1. Attention Scores:
       e_ti = score(s_t, h_i)

       Common scoring functions:
       • Dot product: s_t · h_i
       • Additive: v^T tanh(W[s_t; h_i])
       • General: s_t^T W h_i

    2. Attention Weights:
       α_ti = softmax(e_ti) = exp(e_ti) / Σ_j exp(e_tj)

    3. Context Vector:
       c_t = Σ_i α_ti × h_i

    4. Final prediction uses [s_t; c_t]
    """)

    print("\nPlain English:")
    print("1. Score how relevant each input word is")
    print("2. Turn scores into probabilities (sum to 1)")
    print("3. Take weighted average of input representations")
    print("4. Use this to predict the next word")

attention_math()
```

### Numerical Example

```python

def attention_numerical():
    """
    Concrete numbers example
    """
    import math

    print("Attention with Real Numbers")
    print("=" * 60)

    # Encoder states (simplified - 2D vectors)
    h = {
        "The": [0.1, 0.2],
        "cat": [0.8, 0.7],
        "sat": [0.3, 0.9],
        "on": [0.2, 0.1],
        "the": [0.1, 0.2],
        "mat": [0.4, 0.3]
    }

    # Current decoder state
    s_t = [0.6, 0.5]

    print(f"Decoder state s_t: {s_t}")
    print("\nComputing dot product scores:")

    scores = {}
    for word, h_vec in h.items():
        # Dot product score
        score = s_t[0]*h_vec[0] + s_t[1]*h_vec[1]
        scores[word] = score
        print(f"  score({word}) = {s_t[0]}×{h_vec[0]} + {s_t[1]}×{h_vec[1]} = {score:.2f}")

    # Softmax
    exp_scores = {w: math.exp(s) for w, s in scores.items()}
    sum_exp = sum(exp_scores.values())

    print("\nAttention weights (after softmax):")
    for word in h.keys():
        alpha = exp_scores[word] / sum_exp
        print(f"  α({word}) = {alpha:.3f}")

    # Context vector
    context = [0, 0]
    for word, alpha in zip(h.keys(), [exp_scores[w]/sum_exp for w in h.keys()]):
        h_vec = h[word]
        context[0] += alpha * h_vec[0]
        context[1] += alpha * h_vec[1]

    print(f"\nContext vector c_t: [{context[0]:.3f}, {context[1]:.3f}]")
    print("This context emphasizes 'cat' the most!")

attention_numerical()
```

---

## A Tiny Attention Implementation

```python

import numpy as np

def attention_implementation():
    """
    Minimal attention implementation
    """
    print("Tiny Attention Implementation")
    print("=" * 60)

    class Attention:
        def __init__(self, hidden_size):
            self.hidden_size = hidden_size
            # Simple dot product attention (no learned parameters)

        def forward(self, decoder_state, encoder_states):
            """
            decoder_state: (hidden_size, 1)
            encoder_states: list of (hidden_size, 1)
            returns context vector
            """
            n_steps = len(encoder_states)

            print(f"\nComputing attention over {n_steps} encoder states")

            # Step 1: Compute scores (dot product)
            scores = []
            for i, h in enumerate(encoder_states):
                score = (decoder_state.T @ h)[0, 0]
                scores.append(score)
                print(f"  score with state {i+1}: {score:.3f}")

            # Step 2: Softmax
            exp_scores = np.exp(scores - np.max(scores))  # for stability
            weights = exp_scores / np.sum(exp_scores)

            print("\n  Attention weights:")
            for i, w in enumerate(weights):
                print(f"    α{i+1}: {w:.3f}")

            # Step 3: Weighted sum
            context = np.zeros_like(decoder_state)
            for i, h in enumerate(encoder_states):
                context += weights[i] * h

            print(f"\n  Context vector norm: {np.linalg.norm(context):.3f}")

            return context, weights

    # Create attention module
    attn = Attention(hidden_size=4)

    # Example data
    decoder_state = np.random.randn(4, 1)
    encoder_states = [np.random.randn(4, 1) for _ in range(5)]

    # Forward pass
    context, weights = attn.forward(decoder_state, encoder_states)

    print(f"\nFinal context shape: {context.shape}")

attention_implementation()
```

---

## Why Attention Was Revolutionary

### Before vs After

| Aspect              | Without Attention    | With Attention                |
| ------------------- | -------------------- | ----------------------------- |
| Context             | Single fixed vector  | Dynamic per step              |
| Long sentences      | Information lost     | Can focus on relevant parts   |
| Interpretability    | Black box            | Can see what model focuses on |
| Translation quality | Degrades with length | Maintains quality             |
| Gradient flow       | Long path            | Direct connections            |

### The Attention Heatmap

````text

Visualizing attention weights:

French: Le  chat  s'est  assis  sur  le  tapis
        ↓    ↓     ↓      ↓     ↓    ↓    ↓
English: The  0.9  0.1   0.0   0.0  0.0  0.0
         cat  0.1  0.8   0.1   0.0  0.0  0.0
         sat  0.0  0.1   0.8   0.1  0.0  0.0
         on   0.0  0.0   0.1   0.7  0.2  0.0
         the  0.0  0.0   0.0   0.2  0.7  0.1
         mat  0.0  0.0   0.0   0.0  0.1  0.9

Each French word looks at relevant English words!
```

* * *

## Why This Matters for LLMs

### 1. Attention Became The Foundation

```python

def attention_legacy():
    """
    How attention evolved
    """
    print("From Attention to Transformers")
    print("=" * 60)

    evolution = {
        "2014": "Bahdanau attention (additive) for Seq2Seq",
        "2015": "Luong attention (dot product) variants",
        "2017": "Self-attention and Transformers",
        "2018+": "Multi-head attention in BERT, GPT"
    }

    for year, milestone in evolution.items():
        print(f"  • {year}: {milestone}")

    print("\nKey insight: 'Attention is All You Need' (2017)")
    print("Transformers use ONLY attention, no RNNs!")

attention_legacy()
````

### 2. Self-Attention vs Cross-Attention

```python

def attention_types():
    """
    Different types of attention in modern LLMs
    """
    print("Types of Attention in LLMs")
    print("=" * 60)

    types = {
        "Self-attention": "Words attend to other words in SAME sequence",
        "Cross-attention": "Decoder attends to encoder states (in T5, BART)",
        "Causal attention": "Masked self-attention (GPT, can't see future)",
        "Multi-head attention": "Multiple attention mechanisms in parallel"
    }

    for attn_type, desc in types.items():
        print(f"  • {attn_type}: {desc}")

attention_types()
```

### 3. Interpretability Through Attention

```python

def attention_interpretability():
    """
    Using attention to understand models
    """
    print("Attention for Interpretability")
    print("=" * 60)

    print("""
    Attention weights show us what the model focuses on:

    "The animal didn't cross the street because it was tired"

    Attention from 'it' to other words:
    animal: 0.65  ←── The model knows 'it' refers to 'animal'
    street: 0.12
    tired:  0.23

    This helps us debug and understand model behavior!
    """)

    print("\nCaution: Attention ≠ Explanation, but it's a useful tool.")

attention_interpretability()
```

### 4. The Quadratic Cost

```python

def quadratic_cost():
    """
    The computational challenge of attention
    """
    print("The Quadratic Cost of Attention")
    print("=" * 60)

    lengths = [10, 100, 1000, 10000, 100000]

    print("Sequence length | Attention comparisons")
    print("-" * 40)

    for L in lengths:
        comparisons = L * L  # Each word attends to all words
        print(f"      {L:6d}     |   {comparisons:12,d}")

quadratic_cost()
```

---

## Attention Variants

| Variant                | Formula               | Use Case                   |
| ---------------------- | --------------------- | -------------------------- |
| Additive (Bahdanau)    | `v^T tanh(W₁h + W₂s)` | Original, more expressive  |
| Multiplicative (Luong) | `h^T W s`             | Faster, simpler            |
| Dot Product            | `h^T s`               | Used in transformers       |
| Scaled Dot Product     | `(QK^T)/√d`           | Transformer self-attention |

---

## Quick Recap

• Attention allows the decoder to look back at all encoder states at each step—like an interpreter with a searchable notebook, finding relevant information just when it's needed

• It computes relevance scores, turns them into weights, and creates a dynamic context vector—solving the bottleneck problem of Seq2Seq models and enabling handling of long sequences

• Attention was the breakthrough that led directly to Transformers—the insight that "attention is all you need" replaced RNNs entirely, and every modern LLM from GPT to Claude is built on attention mechanisms

---

## Mental Hook

> "Attention is like having a photographic memory that can instantly flip to exactly the right page when you need it—instead of trying to remember everything at once, you just look up what's relevant right now."
