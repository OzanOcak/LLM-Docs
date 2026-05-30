# Positional Encodings

## The Seating Chart Analogy

Imagine you walk into a meeting where everyone is wearing name tags but there are no assigned seats. You know who people are, but you have no idea who sits next to whom, who arrived first, or what the order of speakers is. That's the problem transformers face—they process all words in parallel, so without additional information, "cat sat" and "sat cat" look identical. Positional encodings are like assigning each person a seat number, so the model knows the order of words.

In LLMs, positional encodings are essential because transformers have no built-in sense of sequence. They need to be explicitly told where each word appears in the sentence. This seemingly simple addition is what allows models like GPT to understand that "dog bites man" is very different from "man bites dog."

---

## The Problem: Transformers Are Order-Blind

### Why Position Matters

```python

def order_matters():
    """
    Why word order is crucial
    """
    print("Why Word Order Matters")
    print("=" * 60)

    sentence1 = "dog bites man"
    sentence2 = "man bites dog"

    print(f"Sentence 1: '{sentence1}'")
    print(f"Sentence 2: '{sentence2}'")

    print("\nSame words, completely different meanings!")
    print("But transformers process all words in parallel.")
    print("\nWithout position information:")
    print("  'dog' at position 1 looks identical to 'dog' at position 3")
    print("  The model can't tell them apart!")

    print("\nWe need to inject position information somehow.")

order_matters()
```

### The Parallel Processing Challenge

```text

RNNs: Natural order (sequential)
Step 1: "dog" ──→ Step 2: "bites" ──→ Step 3: "man"
           ↑                         ↑
    Knows it's position 1       Knows it's position 3

Transformers: All at once
    "dog"    "bites"    "man"
      ↓         ↓         ↓
    [???]     [???]     [???]

    All words processed simultaneously!
    No inherent order information.
```

---

## The Solution: Positional Encodings

### Core Idea

Positional encodings are vectors added to word embeddings to give each word a sense of its position.

```python

def positional_intro():
    """
    The basic concept of positional encodings
    """
    print("Positional Encodings: Adding Order Information")
    print("=" * 60)

    print("""
    Word embedding:      Positional encoding:    Final input:
    [0.2, -0.5, 0.8]  +  [0.1, 0.0, -0.1]  =  [0.3, -0.5, 0.7]
         ↑                        ↑                        ↑
    Word meaning          Position info          Meaning + Position

    Each word gets a unique position signal added to its embedding.
    """)

    print("\nNow words at different positions have different vectors!")
    print("'dog' at position 1 ≠ 'dog' at position 3")

positional_intro()
```

### Visualizing the Addition

```text

Position 1:    "dog" embedding + pos₁ = input₁
Position 2:    "bites" embedding + pos₂ = input₂
Position 3:    "man" embedding + pos₃ = input₃

Even though "dog" and "man" use the same word embedding,
they get different positional signals → different inputs!
```

---

## Types of Positional Encodings

### 1. Sinusoidal Encodings (Original Transformer)

The original Transformer used sine and cosine functions of different frequencies.

```python

import math

def sinusoidal_demo():
    """
    Sinusoidal positional encodings
    """
    print("Sinusoidal Positional Encodings")
    print("=" * 60)

    def get_positional_encoding(pos, d_model=4):
        """Simplified positional encoding for demonstration"""
        pe = []
        for i in range(d_model):
            if i % 2 == 0:
                # Sine for even indices
                value = math.sin(pos / (10000 ** (i / d_model)))
            else:
                # Cosine for odd indices
                value = math.cos(pos / (10000 ** ((i-1) / d_model)))
            pe.append(value)
        return pe

    d_model = 4
    print(f"Positional encodings for dimension {d_model}:")
    print("-" * 50)

    for pos in range(1, 6):
        pe = get_positional_encoding(pos, d_model)
        rounded = [round(x, 3) for x in pe]
        print(f"Position {pos}: {rounded}")

    print("\nProperties:")
    print("• Each position has a unique encoding")
    print("• Works for any sequence length (can extrapolate)")
    print("• Pattern is consistent: pos₁ - pos₂ relationship is predictable")

sinusoidal_demo()
```

### Why Sine and Cosine?

```python

def why_sine_cosine():
    """
    The intuition behind sinusoidal encodings
    """
    print("Why Sine and Cosine?")
    print("=" * 60)

    print("""
    Sinusoidal encodings have useful properties:

    1. Unique for each position
       pos₁ ≠ pos₂ → encoding₁ ≠ encoding₂

    2. Bounded values (between -1 and 1)
       Stable for neural networks

    3. Linear relationships
       PE(pos + k) can be represented as linear function of PE(pos)
       This helps the model learn relative positions!

    4. Extrapolates to longer sequences
       Can handle sentences longer than those seen in training
    """)

    print("\nThe formula:")
    print("  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))")
    print("  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")

why_sine_cosine()
```

### 2. Learned Positional Embeddings

Some models (like BERT) learn positional embeddings during training.

```python

def learned_positional():
    """
    Learned positional embeddings
    """
    print("Learned Positional Embeddings")
    print("=" * 60)

    print("""
    Instead of fixed sine/cosine functions,
    let the model learn position representations:

    Position 1: learnable vector p₁
    Position 2: learnable vector p₂
    Position 3: learnable vector p₃
    ...

    These are trained like word embeddings!
    """)

    max_length = 512
    d_model = 768

    print(f"\nBERT uses learned positional embeddings:")
    print(f"  Max sequence length: {max_length}")
    print(f"  Embedding dimension: {d_model}")
    print(f"  Parameters: {max_length} × {d_model} = {max_length * d_model:,}")

    print("\nPros: Can adapt to data")
    print("Cons: Can't handle sequences longer than training max")

learned_positional()
```

---

## A Complete Example

```python

def complete_positional_example():
    """
    End-to-end example of positional encodings
    """
    print("Complete Example: Adding Position to Words")
    print("=" * 60)

    # Word embeddings (simplified)
    word_embeddings = {
        "dog": [0.1, 0.5, -0.2],
        "bites": [0.3, -0.1, 0.4],
        "man": [0.2, 0.3, 0.1]
    }

    # Positional encodings (simplified)
    positional_encodings = {
        1: [0.01, 0.02, 0.01],
        2: [0.02, 0.01, 0.02],
        3: [0.01, 0.03, 0.01]
    }

    sentence1 = ["dog", "bites", "man"]
    sentence2 = ["man", "bites", "dog"]

    print("Sentence 1: 'dog bites man'")
    print("-" * 40)
    for pos, word in enumerate(sentence1, 1):
        word_emb = word_embeddings[word]
        pos_enc = positional_encodings[pos]
        final = [word_emb[i] + pos_enc[i] for i in range(3)]
        print(f"  Position {pos} - '{word}':")
        print(f"    Word emb: {word_emb}")
        print(f"    + Pos enc: {pos_enc}")
        print(f"    = Final:   {[round(x,2) for x in final]}")

    print("\nSentence 2: 'man bites dog'")
    print("-" * 40)
    for pos, word in enumerate(sentence2, 1):
        word_emb = word_embeddings[word]
        pos_enc = positional_encodings[pos]
        final = [word_emb[i] + pos_enc[i] for i in range(3)]
        print(f"  Position {pos} - '{word}':")
        print(f"    Word emb: {word_emb}")
        print(f"    + Pos enc: {pos_enc}")
        print(f"    = Final:   {[round(x,2) for x in final]}")

    print("\nKey insight: 'dog' at position 1 ≠ 'dog' at position 3!")
    print("The model can now distinguish word order.")

complete_positional_example()
```

---

## A Tiny Positional Encoding Implementation

```python

import numpy as np

def positional_implementation():
    """
    Implement positional encodings from scratch
    """
    print("Implementing Positional Encodings")
    print("=" * 60)

    class PositionalEncoding:
        def __init__(self, d_model, max_len=100):
            self.d_model = d_model

            # Create positional encoding matrix
            pe = np.zeros((max_len, d_model))

            # Position indices (0 to max_len-1)
            position = np.arange(0, max_len).reshape(-1, 1)

            # Division term: 10000^(2i/d_model)
            div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

            # Apply sine to even indices
            pe[:, 0::2] = np.sin(position * div_term)

            # Apply cosine to odd indices
            pe[:, 1::2] = np.cos(position * div_term)

            self.pe = pe

        def get_encoding(self, pos):
            """Get encoding for a specific position"""
            if pos < len(self.pe):
                return self.pe[pos]
            else:
                return None

        def visualize(self, num_positions=10):
            """Show encodings for first few positions"""
            print(f"\nPositional encodings (first {num_positions} positions, first 4 dims):")
            print("Pos | Dim0 (sin) | Dim1 (cos) | Dim2 (sin) | Dim3 (cos)")
            print("-" * 55)

            for pos in range(num_positions):
                enc = self.pe[pos][:4]
                print(f"{pos:3d} |   {enc[0]:6.3f}   |   {enc[1]:6.3f}   |   {enc[2]:6.3f}   |   {enc[3]:6.3f}")

    # Create positional encoder
    pe = PositionalEncoding(d_model=6, max_len=20)

    # Show encodings
    pe.visualize(8)

    print("\nEach position has a unique pattern!")
    print("Notice how values stay between -1 and 1.")

positional_implementation()
```

---

## Relative Positional Encodings

### Beyond Absolute Position

Modern models often use relative positional encodings that focus on distances between words.

```python

def relative_positional():
    """
    Relative positional encodings
    """
    print("Relative Positional Encodings")
    print("=" * 60)

    print("""
    Absolute position: "This is word #5"
    Relative position: "This word is 3 words after that one"

    For many tasks, relative distance matters more:

    "The cat sat on the mat and then..."

    It's more useful to know:
    • "cat" is 2 words before "sat"
    • "mat" is 4 words after "sat"

    Than to know absolute positions 2 and 6.
    """)

    print("\nModern models like GPT use relative position information.")
    print("This helps with generalization to longer sequences.")

relative_positional()
```

### Relative vs Absolute

| Aspect          | Absolute                   | Relative                 |
| --------------- | -------------------------- | ------------------------ |
| What it encodes | Position number (1,2,3...) | Distance between words   |
| Extrapolation   | Limited by max length      | Works for any length     |
| Used in         | Original Transformer, BERT | GPT, modern LLMs         |
| Intuition       | "Word at position 5"       | "Word 3 positions later" |

---

## Why This Matters for LLMs

### 1. Every LLM Needs Position Information

```python

def llm_positional():
    """
    Positional encodings in popular models
    """
    print("Positional Encodings in Popular LLMs")
    print("=" * 60)

    models = {
        "Transformer (original)": "Sinusoidal encodings",
        "BERT": "Learned positional embeddings (max 512)",
        "GPT-2": "Learned positional embeddings",
        "GPT-3": "Learned positional embeddings",
        "Llama": "Rotary Position Embeddings (RoPE) - relative",
        "RoBERTa": "Learned positional embeddings"
    }

    for model, encoding in models.items():
        print(f"  • {model}: {encoding}")

llm_positional()
```

### 2. Rotary Position Embeddings (RoPE)

```python

def rope():
    """
    Rotary Position Embeddings (used in Llama)
    """
    print("Rotary Position Embeddings (RoPE)")
    print("=" * 60)

    print("""
    RoPE is a clever relative positional encoding:

    Instead of adding position to embeddings,
    it ROTATES the embedding vectors based on position!

    Think of it like:
    • Word embedding is an arrow pointing in some direction
    • Position determines how much to rotate that arrow
    • The angle between arrows encodes relative distance
    """)

    print("\nAdvantages:")
    print("• Naturally captures relative positions")
    print("• Decays with distance (nearby words more similar)")
    print("• Works for any sequence length")
    print("\nUsed in: Llama, PaLM, many modern LLMs")

rope()
```

### 3. The Context Window Challenge

```python

def context_window():
    """
    Positional encodings and context windows
    """
    print("Positional Encodings and Context Windows")
    print("=" * 60)

    print("""
    The context window (e.g., 2048, 8192, 128K tokens)
    is partly determined by positional encodings:

    • Learned embeddings: Hard limit (can't go beyond trained length)
    • Sinusoidal: Theoretically unlimited, but may degrade
    • RoPE: Can extrapolate better than learned

    This is why newer models have much larger context windows!
    """)

    models_context = {
        "BERT": "512 tokens",
        "GPT-3": "2048 tokens",
        "GPT-4": "32K-128K tokens",
        "Claude": "100K-200K tokens",
        "Gemini": "1M tokens (with special encoding)"
    }

    print("\nContext windows:")
    for model, context in models_context.items():
        print(f"  • {model}: {context}")

context_window()
```

### 4. Positional Information Flow

```text

Without position:    With position:
    [cat]                [cat+pos₁]
      ↓                     ↓
    Self-attention      Self-attention
      ↓                     ↓
    [???]                [cat+context]

    Can't tell if       Knows "cat" was
    "cat" was first     first word
    or third
```

---

## Positional Encoding Cheat Sheet

| Type       | Formula              | Pros                    | Cons                |
| ---------- | -------------------- | ----------------------- | ------------------- |
| Sinusoidal | sin/cos functions    | Extrapolates, no params | Fixed pattern       |
| Learned    | Trainable embeddings | Adapts to data          | Fixed max length    |
| Relative   | Based on distances   | Better for length       | More complex        |
| RoPE       | Rotation matrices    | Best of both            | Newer, less studied |

---

## Quick Recap

• Positional encodings add order information to word embeddings—like assigning seat numbers so the model knows that "dog bites man" is different from "man bites dog"

• They come in different flavors—sinusoidal (fixed math functions), learned (trained like embeddings), and relative (focus on distances between words)

• Modern LLMs use sophisticated positional encodings like RoPE—which rotate vectors based on position, naturally capturing relative distances and enabling much longer context windows

---

## Mental Hook

> "Positional encodings are the invisible ink that marks each word's place in line—without them, transformers would see 'cat sat' and 'sat cat' as identical twins, unable to tell who did what to whom."
