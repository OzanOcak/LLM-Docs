# RNNs (Recurrent Neural Networks)

## The Page-Turner Analogy

Imagine reading a mystery novel. You don't just look at each page in isolation—you carry forward your understanding from previous pages. "The butler entered the room" means something different if you remember that "the murder weapon was missing." That's what RNNs do: they maintain a hidden state that carries information from one step to the next, allowing them to process sequences like sentences, time series, or audio.

In the evolution toward LLMs, RNNs were the first serious attempt at handling sequences. While they've been superseded by Transformers, understanding RNNs helps grasp why we needed something better and what problems Transformers had to solve.

---

## What Is an RNN?

### The Core Idea

A Recurrent Neural Network processes sequences by maintaining a "memory" (hidden state) that gets updated at each step.

```text

Regular Neural Network:      RNN:
    Input ──→ Output            Input₁ ──→ Output₁
                                   ↑          ↓
                                Hidden₁ ──→ Hidden₂
                                               ↑
                                            Input₂ ──→ Output₂

```

Each step uses both current input and previous hidden state.

```python

def rnn_intro():
    """
    The basic concept of RNNs
    """
    print("RNNs: Processing Sequences with Memory")
    print("=" * 60)

    print("""
    At each time step:

    hidden_t = tanh( W_hidden × hidden_{t-1} + W_input × input_t + bias )
    output_t = W_output × hidden_t + bias_out

    Plain English:
    • New hidden state = f(old hidden state + current input)
    • Output = based on new hidden state
    """)

    print("\nThis lets information flow from step to step,")
    print("like carrying a sticky note of context as you read.")

rnn_intro()

```

---

## How RNNs Work Step by Step

### Processing a Sentence

```python

def rnn_processing():
    """
    Walk through RNN processing of a sentence
    """
    print("RNN Processing: Step by Step")
    print("=" * 60)

    sentence = ["The", "cat", "sat", "on", "the", "mat"]

    print(f"Sentence: {' → '.join(sentence)}")
    print("\nRNN processes word by word:")

    # Initialize hidden state (memory)
    hidden = "Start (no context)"

    for i, word in enumerate(sentence):
        print(f"\nStep {i+1}: Reading '{word}'")
        print(f"  Previous hidden state: {hidden}")

        # RNN combines current word with previous hidden state
        print(f"  Combining: '{word}' + previous memory")

        # Update hidden state
        hidden = f"Context now includes '{word}'"
        print(f"  New hidden state: {hidden}")

        # Generate output (e.g., next word prediction)
        print(f"  Output: prediction based on current hidden state")

    print("\nFinal hidden state contains context from entire sentence!")

rnn_processing()
```

### The Recurrent Connection

```text

Time Step 1:    Time Step 2:    Time Step 3:
Input: "The"    Input: "cat"    Input: "sat"
   ↓               ↓               ↓
   RNN ──→ h₁      RNN ──→ h₂      RNN ──→ h₃
   ↑        ↓      ↑        ↓      ↑        ↓
   h₀ ←─────┘      h₁ ←─────┘      h₂ ←─────┘

h₀ = initial state (usually zeros)
h₁ = f("The", h₀)
h₂ = f("cat", h₁)
h₃ = f("sat", h₂)

Information flows forward through time.
```

---

## A Tiny RNN Implementation

```python

import numpy as np

def tiny_rnn_demo():
    """
    A minimal RNN implementation
    """
    print("Tiny RNN: Learning to Predict Next Character")
    print("=" * 60)

    class TinyRNN:
        def __init__(self, input_size, hidden_size, output_size):
            # Weight matrices
            self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
            self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
            self.W_hy = np.random.randn(output_size, hidden_size) * 0.1
            self.b_h = np.zeros((hidden_size, 1))
            self.b_y = np.zeros((output_size, 1))

        def forward(self, inputs):
            """
            Process a sequence of inputs
            """
            h = np.zeros((self.W_hh.shape[0], 1))  # Initial hidden state
            outputs = []

            for i, x in enumerate(inputs):
                # Update hidden state: h_t = tanh(W_xh·x + W_hh·h_{t-1} + b_h)
                h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)

                # Generate output: y_t = W_hy·h + b_y
                y = self.W_hy @ h + self.b_y
                outputs.append(y)

                print(f"Step {i+1}:")
                print(f"  Input shape: {x.shape}")
                print(f"  Hidden state norm: {np.linalg.norm(h):.3f}")

            return outputs, h

    # Create a tiny RNN
    rnn = TinyRNN(input_size=10, hidden_size=5, output_size=10)

    # Generate random sequence of 3 inputs
    sequence = [np.random.randn(10, 1) for _ in range(3)]

    print("\nProcessing sequence of 3 vectors...")
    outputs, final_hidden = rnn.forward(sequence)

    print(f"\nFinal hidden state shape: {final_hidden.shape}")
    print("This hidden state contains information from all 3 steps!")

tiny_rnn_demo()
```

---

## What RNNs Can Do

### Applications

| Task                | How RNN Helps                                  |
| ------------------- | ---------------------------------------------- |
| Language Modeling   | Predict next word based on previous words      |
| Sentiment Analysis  | Understand sentence meaning from word sequence |
| Machine Translation | Read sentence, produce translation             |
| Speech Recognition  | Process audio frames sequentially              |
| Time Series         | Predict future values from past                |

### Language Model Example

```python

def rnn_language_model():
    """
    RNN as a simple language model
    """
    print("RNN Language Model: Predicting Next Words")
    print("=" * 60)

    # Simplified example
    context = ["The", "cat", "sat", "on"]

    print(f"Context: {' '.join(context)}")
    print("\nRNN processes each word and updates its state:")

    hidden_state = "Initial"
    for i, word in enumerate(context):
        print(f"  Step {i+1}: Read '{word}'")
        hidden_state = f"State after '{' '.join(context[:i+1])}'"
        print(f"           Hidden: {hidden_state}")

    print(f"\nFinal hidden state: {hidden_state}")
    print("This state is used to predict the next word.")
    print("Possible next words: 'the', 'a', 'mat', 'floor'...")

rnn_language_model()
```

---

## The Fundamental Problems with RNNs

### 1. Vanishing Gradients (The Forgetting Problem)

```python

def vanishing_rnn():
    """
    Why RNNs forget long-term information
    """
    print("The Vanishing Gradient Problem")
    print("=" * 60)

    print("""
    During training, gradients flow backward through time:

    Step 10 ← Step 9 ← Step 8 ← ... ← Step 2 ← Step 1

    Each step multiplies by a weight matrix.
    If weights are small (<1), gradient shrinks exponentially.

    After 10 steps: gradient × 0.9¹⁰ = 0.35
    After 50 steps: gradient × 0.9⁵⁰ = 0.005
    After 100 steps: essentially zero!

    Early words get NO learning signal.
    """)

    steps = [10, 20, 50, 100]
    for s in steps:
        remaining = 0.9 ** s
        print(f"  After {s:3d} steps: {remaining:.2e} of gradient remains")

vanishing_rnn()
```

### 2. Short-Term Memory

```text

RNN Memory Decay:

"The movie was fantastic... [50 words later] ...I loved it"

Word:     "fantastic"  → ... → "loved"
Time:         t₀                t₅₀
Influence:    1.0              0.00001

The RNN has almost completely forgotten "fantastic" by the time it reaches "loved".
It can't connect related words that are far apart.
```

### 3. Sequential Processing (Slow)

```python

def sequential_slow():
    """
    Why RNNs are slow
    """
    print("Sequential Processing: The Speed Bottleneck")
    print("=" * 60)

    sequence_length = 1000

    print(f"With {sequence_length} words:")
    print("""
    RNN: Must process word 1 → word 2 → word 3 → ... → word 1000
          (1000 sequential steps, can't parallelize)

    Transformer: Process ALL words at once
          (parallel processing on GPU)

    RNN time: O(sequence_length)
    Transformer time: O(1) with enough hardware
    """)

    print("\nThis is why RNNs are slow for long sequences!")

sequential_slow()
```

---

## RNN Architectures

### Types of RNNs

```text

1. One-to-One (Standard NN):
   [Input] ─→ [Output]

2. One-to-Many (Image Captioning):
   [Image] ─→ [Word1] → [Word2] → [Word3]

3. Many-to-One (Sentiment Analysis):
   [Word1] → [Word2] → [Word3] ─→ [Sentiment]

4. Many-to-Many (Translation):
   [Word1] → [Word2] → [Word3] ─→ [WordA] → [WordB] → [WordC]

5. Many-to-Many (Video Frame Processing):
   [Frame1] → [Frame2] → [Frame3] → [Frame4] ─→ [Label1] [Label2] [Label3] [Label4]
```

### Code for Different Types

```python

def rnn_types():
    """
    Different RNN architectures
    """
    print("RNN Architecture Types")
    print("=" * 60)

    architectures = {
        "One-to-One": "Image classification (single input, single output)",
        "One-to-Many": "Image captioning (image → sequence of words)",
        "Many-to-One": "Sentiment analysis (words → positive/negative)",
        "Many-to-Many (sync)": "Video frame labeling (each frame → label)",
        "Many-to-Many (async)": "Machine translation (English → French)"
    }

    for arch, example in architectures.items():
        print(f"  • {arch}: {example}")

rnn_types()
```

---

## Why This Matters for LLMs

### 1. RNNs Were the Foundation

```python

def rnn_legacy():
    """
    What LLMs inherited from RNNs
    """
    print("RNN Legacy in Modern LLMs")
    print("=" * 60)

    legacy = {
        "Sequential processing concept": "Language is sequential",
        "Hidden states": "Representations carry context",
        "Word-by-word generation": "Decoders still generate word by word",
        "Teacher forcing": "Training with correct previous words"
    }

    for concept, desc in legacy.items():
        print(f"  • {concept}: {desc}")

rnn_legacy()
```

### 2. RNN Limitations → Transformer Innovations

| RNN Problem           | Transformer Solution                     |
| --------------------- | ---------------------------------------- |
| Vanishing gradients   | Residual connections + attention         |
| Sequential processing | Parallel self-attention                  |
| Short memory          | Direct connections between all positions |
| Fixed context         | Attention can look anywhere              |

### 3. Where RNNs Still Appear

```python

def rnn_today():
    """
    Where RNNs are still used
    """
    print("RNNs in the Modern AI Landscape")
    print("=" * 60)

    current_uses = [
        "Audio processing (speech recognition)",
        "Time series forecasting",
        "Some edge devices (smaller than transformers)",
        "Hybrid models with attention",
        "Educational value (learning sequence models)"
    ]

    print("While Transformers dominate NLP, RNNs are still used in:")
    for use in current_uses:
        print(f"  • {use}")

rnn_today()
```

---

## RNN vs Transformer Comparison

| Aspect                  | RNN                             | Transformer               |
| ----------------------- | ------------------------------- | ------------------------- |
| Processing              | Sequential (word by word)       | Parallel (all at once)    |
| Memory                  | Hidden state (bottleneck)       | Attention (direct access) |
| Long-range dependencies | Difficult (vanishing gradients) | Natural (O(1) path)       |
| Training speed          | Slow (can't parallelize)        | Fast (GPU parallel)       |
| Context window          | Limited (practical ~100)        | Large (thousands)         |
| Parameter efficiency    | Good for small data             | Needs lots of data        |

---

## Quick Recap

• RNNs process sequences by maintaining a hidden state that carries information from step to step—like reading a book and remembering what happened on previous pages, allowing context to influence understanding

• They suffer from vanishing gradients and sequential processing—long-range information fades away, and they can't parallelize, making them slow for long sequences

• While Transformers have replaced RNNs for LLMs, RNNs laid the groundwork—introducing the concepts of sequence processing, hidden states, and word-by-word generation that still influence modern architectures

---

## Mental Hook

> "RNNs are like reading with a fading highlighter—you can mark important words as you go, but the earlier marks get fainter and fainter until they disappear, making it impossible to connect ideas from the beginning and end of a long book."
