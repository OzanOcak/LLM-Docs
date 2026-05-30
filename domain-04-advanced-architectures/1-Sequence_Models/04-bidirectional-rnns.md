# Bidirectional RNNs

## The Detective Analogy

Imagine a detective solving a mystery. They don't just read a case file from start to finish—they look forward to see what happened, then look backward to see what led up to it. A clue in Chapter 10 might explain a mystery in Chapter 2. That's what Bidirectional RNNs do: they process sequences both forward and backward, capturing context from both directions.

In the evolution toward LLMs, bidirectional processing was a crucial insight. It showed that understanding a word often requires looking at words that come both before AND after it. This concept directly influenced modern architectures like BERT, which uses bidirectional attention.

---

## What Is a Bidirectional RNN?

### The Core Idea

A Bidirectional RNN runs two separate RNNs—one forward through the sequence, one backward—and combines their outputs.

```text

Standard RNN:          Bidirectional RNN:
    → → → → →              ← ← ← ← ←
    [The] [cat] [sat]      [The] [cat] [sat]
                             → → → → →
                            Combine at each step
```

```python

def bidirectional_intro():
    """
    The basic concept of bidirectional RNNs
    """
    print("Bidirectional RNNs: Looking Both Ways")
    print("=" * 60)

    sentence = "The bank of the river"

    print(f"Sentence: '{sentence}'")
    print("\nWhat does 'bank' mean?")
    print("  Forward context: 'The' → tells us it's a noun")
    print("  Backward context: 'of the river' ← tells us it's a river bank, not financial")
    print("\nYou need BOTH directions to know!")

    print("\nBidirectional RNN processes:")
    print("  → Forward pass: The → bank → of → the → river")
    print("  ← Backward pass: river ← the ← of ← bank ← The")
    print("  → Concatenate forward and backward states for each word")

bidirectional_intro()
```

---

## How Bidirectional RNNs Work

### The Architecture

```text

Time:       1         2         3         4         5
Word:      The      bank       of       the     river
           ↓         ↓         ↓         ↓         ↓
Forward:  [h₁→] → [h₂→] → [h₃→] → [h₄→] → [h₅→]
            ↑         ↑         ↑         ↑         ↑
            └─────────┴─────────┴─────────┴─────────┘

            ┌─────────┬─────────┬─────────┬─────────┐
            ↓         ↓         ↓         ↓         ↓
Backward: [h₁←] ← [h₂←] ← [h₃←] ← [h₄←] ← [h₅←]
           ↑         ↑         ↑         ↑         ↑
Word:      The      bank       of       the     river
Time:       1         2         3         4         5

Final representation for each word:
    Word1: [h₁→, h₁←]  (forward state + backward state)
    Word2: [h₂→, h₂←]
    Word3: [h₃→, h₃←]
    etc.
```

```python

def bidirectional_architecture():
    """
    Detailed architecture explanation
    """
    print("Bidirectional RNN Architecture")
    print("=" * 60)

    print("""
    Step-by-step:

    1. Forward RNN processes sequence left to right
       h₁→ = f(The, h₀→)
       h₂→ = f(bank, h₁→)
       h₃→ = f(of, h₂→)
       h₄→ = f(the, h₃→)
       h₅→ = f(river, h₄→)

    2. Backward RNN processes sequence right to left
       h₅← = f(river, h₆←)  (h₆← is initial state)
       h₄← = f(the, h₅←)
       h₃← = f(of, h₄←)
       h₂← = f(bank, h₃←)
       h₁← = f(The, h₂←)

    3. Concatenate for each position
       word2_representation = [h₂→, h₂←]
    """)

    print("\nEach word's final representation contains")
    print("context from both sides!")

bidirectional_architecture()
```

---

## A Concrete Example: Disambiguating Words

### The "Bank" Problem

```python

def bank_example():
    """
    How bidirectional RNNs resolve ambiguity
    """
    print("Bidirectional RNNs: Resolving Ambiguity")
    print("=" * 60)

    sentence1 = "I went to the bank to deposit money"
    sentence2 = "I sat on the bank of the river"

    print(f"Sentence 1: '{sentence1}'")
    print(f"Sentence 2: '{sentence2}'")

    print("\nFor the word 'bank' in sentence 1:")
    print("  Forward context: 'I went to the'")
    print("  Backward context: 'to deposit money'")
    print("  → Combined: FINANCIAL BANK")

    print("\nFor the word 'bank' in sentence 2:")
    print("  Forward context: 'I sat on the'")
    print("  Backward context: 'of the river'")
    print("  → Combined: RIVER BANK")

    print("\nA unidirectional RNN would only see left context:")
    print("  'I went to the' → could be either!")
    print("  'I sat on the' → could be either!")

    print("\nBidirectional sees both sides → gets it right.")

bank_example()
```

### Step-by-Step Through the Sentence

```python

def bidirectional_step_by_step():
    """
    Trace through bidirectional processing
    """
    print("Tracing Bidirectional Processing")
    print("=" * 60)

    words = ["The", "cat", "sat", "on", "the", "mat"]

    print(f"Sequence: {' → '.join(words)}")
    print("\nForward pass (left to right):")

    forward_states = []
    prev_h = "START"
    for i, word in enumerate(words):
        new_h = f"h{i+1}→ (context up to '{word}')"
        forward_states.append(new_h)
        print(f"  Step {i+1}: Read '{word}' + {prev_h} → {new_h}")
        prev_h = new_h

    print("\nBackward pass (right to left):")
    backward_states = []
    prev_h = "END"
    for i, word in enumerate(reversed(words)):
        new_h = f"h{len(words)-i}← (context after '{word}')"
        backward_states.insert(0, new_h)
        print(f"  Step {i+1}: Read '{word}' + {prev_h} → {new_h}")
        prev_h = new_h

    print("\nFinal representations (forward + backward):")
    for i, word in enumerate(words):
        print(f"  '{word}': [{forward_states[i]}, {backward_states[i]}]")

bidirectional_step_by_step()
```

---

## Implementing a Bidirectional RNN

```python

import numpy as np

def bidirectional_rnn_demo():
    """
    Simple bidirectional RNN implementation
    """
    print("Bidirectional RNN: Simple Implementation")
    print("=" * 60)

    class SimpleRNN:
        def __init__(self, input_size, hidden_size):
            self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
            self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
            self.b_h = np.zeros((hidden_size, 1))

        def forward(self, x, prev_h):
            return np.tanh(self.W_xh @ x + self.W_hh @ prev_h + self.b_h)

    class BidirectionalRNN:
        def __init__(self, input_size, hidden_size):
            self.forward_rnn = SimpleRNN(input_size, hidden_size)
            self.backward_rnn = SimpleRNN(input_size, hidden_size)
            self.hidden_size = hidden_size

        def forward(self, sequence):
            seq_len = len(sequence)
            hidden_size = self.hidden_size

            # Initialize states
            fwd_h = np.zeros((hidden_size, 1))
            bwd_h = np.zeros((hidden_size, 1))

            fwd_states = []
            bwd_states = []

            print(f"Processing sequence of {seq_len} steps")

            # Forward pass
            print("\nForward pass:")
            for t, x in enumerate(sequence):
                fwd_h = self.forward_rnn.forward(x, fwd_h)
                fwd_states.append(fwd_h.copy())
                print(f"  Step {t+1}: forward hidden norm = {np.linalg.norm(fwd_h):.3f}")

            # Backward pass
            print("\nBackward pass:")
            for t, x in enumerate(reversed(sequence)):
                bwd_h = self.backward_rnn.forward(x, bwd_h)
                bwd_states.insert(0, bwd_h.copy())
                print(f"  Step {seq_len-t}: backward hidden norm = {np.linalg.norm(bwd_h):.3f}")

            # Combine
            combined = []
            for t in range(seq_len):
                combined.append(np.vstack([fwd_states[t], bwd_states[t]]))

            return combined

    # Create bidirectional RNN
    bi_rnn = BidirectionalRNN(input_size=10, hidden_size=5)

    # Create a sequence of 3 vectors
    sequence = [np.random.randn(10, 1) for _ in range(3)]

    # Forward pass
    outputs = bi_rnn.forward(sequence)

    print(f"\nFinal combined representations shape: {outputs[0].shape}")
    print("Each word now has context from both directions!")

bidirectional_rnn_demo()
```

---

## Bidirectional vs Unidirectional

### Comparison Table

| Aspect      | Unidirectional RNN         | Bidirectional RNN                  |
| ----------- | -------------------------- | ---------------------------------- |
| Context     | Only past words            | Past AND future words              |
| Use case    | Generation (left to right) | Understanding (all context)        |
| Processing  | One pass                   | Two passes (forward + backward)    |
| Latency     | Can process online         | Needs entire sequence              |
| Parameters  | Fewer                      | Double (two RNNs)                  |
| Performance | Good for language modeling | Better for classification, tagging |

### When to Use Each

```python

def when_to_use():
    """
    Choosing between unidirectional and bidirectional
    """
    print("Unidirectional vs Bidirectional: When to Use")
    print("=" * 60)

    tasks = {
        "Language modeling (next word prediction)": "Unidirectional (can't see future words!)",
        "Sentiment analysis": "Bidirectional (whole sentence available)",
        "Machine translation": "Encoder: Bidirectional, Decoder: Unidirectional",
        "Named entity recognition": "Bidirectional (context on both sides)",
        "Speech recognition": "Bidirectional (whole utterance known)",
        "Real-time processing": "Unidirectional (can't wait for future)"
    }

    for task, recommendation in tasks.items():
        print(f"  • {task}: {recommendation}")

when_to_use()
```

---

## Why This Matters for LLMs

### 1. Bidirectional Context in Modern Models

```python

def bert_bidirectional():
    """
    How BERT uses bidirectional context
    """
    print("Bidirectional Context in BERT")
    print("=" * 60)

    print("""
    BERT (Bidirectional Encoder Representations from Transformers)
    uses MASKED LANGUAGE MODELING:

    "The [MASK] sat on the mat"

    To predict 'cat', BERT can see:
    • Left context: "The"
    • Right context: "sat on the mat"

    This is the bidirectional principle applied to transformers!
    """)

    print("\nGPT (decoder-only) is unidirectional:")
    print("  'The cat sat on the' → predict next word")
    print("\nBERT (encoder-only) is bidirectional:")
    print("  'The [MASK] sat on the mat' → predict masked word")

bert_bidirectional()
```

### 2. Encoder-Decoder Architectures

```python

def encoder_decoder():
    """
    How bidirectional fits in encoder-decoder
    """
    print("Bidirectional in Encoder-Decoder Models")
    print("=" * 60)

    print("""
    Machine Translation (T5, BART):

    Encoder (bidirectional):
    "The cat sat on the mat"
        ↓
    [Full context representation]
        ↓
    Decoder (unidirectional):
        → "Le" → "chat" → "s'est" → "assis"...

    Encoder sees whole sentence (bidirectional)
    Decoder generates left to right (unidirectional)
    """)

    print("\nBest of both worlds!")

encoder_decoder()
```

### 3. The Trade-off: Understanding vs Generation

| Aspect       | Bidirectional (BERT)    | Unidirectional (GPT) |
| ------------ | ----------------------- | -------------------- |
| Sees future? | Yes (for understanding) | No (for generation)  |
| Best at      | Classification, QA      | Text generation      |
| Training     | Masked language model   | Next word prediction |
| Use case     | "Understand this text"  | "Continue this text" |

### 4. Computational Cost

```python

def computational_cost():
    """
    The cost of bidirectionality
    """
    print("Computational Cost of Bidirectional RNNs")
    print("=" * 60)

    seq_len = 100
    hidden_size = 256

    unidirectional_params = 4 * hidden_size * (hidden_size + 1)  # Simplified
    bidirectional_params = 2 * unidirectional_params

    print(f"Sequence length: {seq_len}")
    print(f"Hidden size: {hidden_size}")
    print(f"\nUnidirectional RNN parameters: ~{unidirectional_params:,}")
    print(f"Bidirectional RNN parameters: ~{bidirectional_params:,}")
    print(f"Memory: 2× larger, Compute: 2× more")

    print("\nFor transformers, the cost is different:")
    print("  BERT: bidirectional attention (all words attend to all)")
    print("  GPT: causal attention (only left context)")
    print("  Compute difference is architectural, not simply double")

computational_cost()
```

---

## Bidirectional Variants

| Variant       | Description                                   |
| ------------- | --------------------------------------------- |
| BiLSTM        | Bidirectional LSTM                            |
| BiGRU         | Bidirectional GRU                             |
| Deep BiRNN    | Multiple stacked bidirectional layers         |
| BiTransformer | Transformer encoder (naturally bidirectional) |

---

## Quick Recap

• Bidirectional RNNs process sequences both forward and backward—like a detective reading a case file from start to finish AND from end to beginning, capturing context from both sides

• Each word's representation combines past and future context—crucial for understanding ambiguous words like "bank" that need information from both directions to resolve

• This concept directly influenced modern LLMs—BERT uses bidirectional attention for understanding, while encoder-decoder models like T5 use bidirectional encoders paired with unidirectional decoders

---

## Mental Hook

> "Bidirectional RNNs are like reading a mystery novel twice—once forward to see what happens, once backward to see what clues you missed—so you understand every page in the context of the whole story."
