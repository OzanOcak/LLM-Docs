# Encoder vs Decoder Architecture

## The Translator vs Writer Analogy

Imagine two different language professionals. A translator reads an entire document in English, understands it fully, then produces a French version—they need to see the whole input before writing (encoder). A writer starts with a prompt and writes a story one word at a time, each word building on what they've already written—they can only look at past words (decoder). Both are language experts, but they work differently.

In LLMs, these two architectures serve different purposes. Encoder-only models (like BERT) understand and analyze text. Decoder-only models (like GPT) generate text. Encoder-decoder models (like T5) do both—they read an input and then generate an output. Understanding the difference is key to knowing which model to use for which task.

---

## The Three Architectures

### At a Glance

```text

Encoder-Only (BERT):           Decoder-Only (GPT):         Encoder-Decoder (T5):
    [The] [cat] [sat]              [The] → [cat] → [sat]      [The] [cat] [sat]
       ↓    ↓    ↓                     ↓       ↓       ↓           ↓    ↓    ↓
    [Encoder] [Encoder] [Encoder]   [Decoder] [Decoder] [Decoder] [Encoder]
       ↓    ↓    ↓                     ↓       ↓       ↓           ↓    ↓    ↓
    [Understanding]                  [Generation]                [Context]
                                                                     ↓
                                                                [Decoder]
                                                                     ↓
                                                              [Le] [chat] [s'est]
```

```python

def architectures_intro():
    """
    The three main transformer architectures
    """
    print("The Three Transformer Architectures")
    print("=" * 60)

    architectures = {
        "Encoder-Only": {
            "also_known": "BERT-style",
            "sees": "Full context (bidirectional)",
            "best_for": "Understanding tasks (classification, QA)",
            "examples": "BERT, RoBERTa, DistilBERT"
        },
        "Decoder-Only": {
            "also_known": "GPT-style",
            "sees": "Left context only (causal)",
            "best_for": "Generation tasks (text completion, chat)",
            "examples": "GPT series, Llama, Claude"
        },
        "Encoder-Decoder": {
            "also_known": "T5-style",
            "sees": "Encoder sees full input, decoder sees left context",
            "best_for": "Transformation tasks (translation, summarization)",
            "examples": "T5, BART, mT5"
        }
    }

    for arch, details in architectures.items():
        print(f"\n{arch}:")
        for key, value in details.items():
            print(f"  • {key}: {value}")

architectures_intro()
```

---

## Encoder-Only Architecture (BERT-style)

### How It Works

The encoder sees the entire input sequence at once—it's bidirectional.

```python

def encoder_only():
    """
    Encoder-only architecture
    """
    print("Encoder-Only: Understanding Like BERT")
    print("=" * 60)

    sentence = "The cat sat on the mat"

    print(f"Input: '{sentence}'")
    print("\nEncoder processes ALL words simultaneously:")
    print("""
    [The] ←→ [cat] ←→ [sat] ←→ [on] ←→ [the] ←→ [mat]
       ↑        ↑        ↑        ↑        ↑        ↑
       └────────┴────────┴────────┴────────┴────────┘
               Full bidirectional attention

    Each word can attend to EVERY other word!
    """)

    print("\nTraining objective: Masked Language Model")
    print("  Predict: The [MASK] sat on the mat → 'cat'")
    print("  Can use context from both sides!")

    print("\nUse cases:")
    print("  • Sentiment analysis")
    print("  • Named entity recognition")
    print("  • Question answering (extractive)")
    print("  • Sentence classification")

encoder_only()
```

### Encoder Architecture Diagram

```text

Input tokens:   [The]  [cat]  [sat]  [on]  [the]  [mat]
                  ↓      ↓      ↓      ↓      ↓      ↓
Embeddings:      E₁     E₂     E₃     E₄     E₅     E₆
                  ↓      ↓      ↓      ↓      ↓      ↓
            ┌─────────────────────────────────────────┐
            │         Transformer Encoder             │
            │         (12 layers of:                   │
            │    Self-Attention + Feed-Forward)       │
            └─────────────────────────────────────────┘
                  ↓      ↓      ↓      ↓      ↓      ↓
Output:         H₁     H₂     H₃     H₄     H₅     H₆
                  ↑      ↑      ↑      ↑      ↑      ↑
            Each output represents the word
            with FULL context from the sentence
```

---

## Decoder-Only Architecture (GPT-style)

### How It Works

The decoder generates text left to right, only seeing past words.

```python

def decoder_only():
    """
    Decoder-only architecture
    """
    print("Decoder-Only: Generation Like GPT")
    print("=" * 60)

    prompt = "The cat sat"

    print(f"Prompt: '{prompt}'")
    print("\nDecoder generates ONE WORD AT A TIME:")
    print("""
    Step 1: [The]    → can only see [The]
    Step 2: [cat]    → can see [The] [cat]
    Step 3: [sat]    → can see [The] [cat] [sat]
    Step 4: [on]     → can see [The] [cat] [sat] [on]

    Causal attention mask prevents looking at future words!
    """)

    print("\nTraining objective: Next Word Prediction")
    print("  Given 'The cat', predict 'sat'")
    print("  Can only use left context!")

    print("\nUse cases:")
    print("  • Text generation")
    print("  • Chatbots")
    print("  • Code completion")
    print("  • Story writing")

decoder_only()
```

### Decoder Architecture Diagram

```text

Input:         [The] → [cat] → [sat] → [on] → [?]
                  ↓       ↓       ↓       ↓
            ┌─────────────────────────────────┐
            │      Transformer Decoder         │
            │  (with masked self-attention)    │
            └─────────────────────────────────┘
                  ↓       ↓       ↓       ↓
Output:        H₁      H₂      H₃      H₄
                ↓       ↓       ↓       ↓
Predict:       cat     sat     on      the

Masked attention:
Step 4: "on" can attend to: [The] [cat] [sat] [on] ✓
                            [the] [mat] ✗ (future)
```

---

## Encoder-Decoder Architecture (T5-style)

### How It Works

The encoder reads the input, then the decoder generates output based on it.

```python

def encoder_decoder():
    """
    Encoder-decoder architecture
    """
    print("Encoder-Decoder: Translation Like T5")
    print("=" * 60)

    english = "The cat sat on the mat"
    french = "Le chat s'est assis sur le tapis"

    print(f"English input: '{english}'")
    print(f"French target: '{french}'")

    print("\n" + "=" * 50)
    print("ENCODING PHASE (bidirectional)")
    print("=" * 50)
    print("""
    [The] ←→ [cat] ←→ [sat] ←→ [on] ←→ [the] ←→ [mat]
       ↑        ↑        ↑        ↑        ↑        ↑
       └────────┴────────┴────────┴────────┴────────┘
               Encoder creates rich representations
    """)

    print("\n" + "=" * 50)
    print("DECODING PHASE (causal)")
    print("=" * 50)
    print("""
    Step 1: <START> → 'Le'     (attends to encoder + <START>)
    Step 2: 'Le' → 'chat'      (attends to encoder + 'Le')
    Step 3: 'chat' → 's'est'   (attends to encoder + 'Le chat')
    etc.
    """)

    print("\nUse cases:")
    print("  • Machine translation")
    print("  • Summarization")
    print("  • Text-to-text tasks")
    print("  • Question answering (generative)")

encoder_decoder()
```

### Encoder-Decoder Diagram

```text

Encoder (bidirectional):
[The] [cat] [sat] [on] [the] [mat]
  ↓     ↓     ↓     ↓     ↓     ↓
  └─────┴─────┴─────┴─────┴─────┘
              ↓
      Context representations
              ↓
Decoder (causal):
<START> → [Le] → [chat] → [s'est] → [assis] ...
   ↓        ↓       ↓         ↓         ↓
   └────────┴───────┴─────────┴─────────┘
   Each step can attend to encoder AND previous outputs
```

---

## Comparison Table

| Aspect       | Encoder-Only         | Decoder-Only             | Encoder-Decoder                |
| ------------ | -------------------- | ------------------------ | ------------------------------ |
| Attention    | Bidirectional (full) | Causal (left only)       | Encoder: full, Decoder: causal |
| Training     | Masked LM            | Next word prediction     | Seq2Seq (e.g., translation)    |
| Input/output | One sequence         | One sequence (generated) | Two sequences (in → out)       |
| Best at      | Understanding        | Generation               | Transformation                 |
| Examples     | BERT, RoBERTa        | GPT, Llama               | T5, BART                       |
| Context      | Full context         | Past only                | Full input + past output       |

---

## Attention Masks Visualized

```python

def attention_masks():
    """
    Different attention patterns
    """
    print("Attention Masks: What Each Architecture Sees")
    print("=" * 60)

    print("""
    Encoder-Only (BERT):
    [1, 1, 1, 1, 1, 1]  Each word sees ALL words
    [1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1]

    Decoder-Only (GPT):
    [1, 0, 0, 0, 0, 0]  Each word sees only itself and past
    [1, 1, 0, 0, 0, 0]
    [1, 1, 1, 0, 0, 0]
    [1, 1, 1, 1, 0, 0]
    [1, 1, 1, 1, 1, 0]
    [1, 1, 1, 1, 1, 1]

    Encoder-Decoder (T5):
    Encoder same as BERT
    Decoder same as GPT + can attend to encoder
    """)

attention_masks()
```

---

## A Tiny Implementation Comparison

```python

import numpy as np

def architecture_comparison():
    """
    Compare the three architectures
    """
    print("Architecture Comparison: Simple Implementation")
    print("=" * 60)

    class SimpleAttention:
        def __init__(self, name, mask_type='full'):
            self.name = name
            self.mask_type = mask_type

        def forward(self, seq_len):
            # Create attention scores (simplified)
            scores = np.ones((seq_len, seq_len))

            if self.mask_type == 'causal':
                # Mask out future tokens
                for i in range(seq_len):
                    for j in range(i+1, seq_len):
                        scores[i, j] = 0
            elif self.mask_type == 'encoder_decoder':
                # Encoder: full, Decoder: causal
                pass

            return scores

    seq_len = 4
    words = ["The", "cat", "sat", "down"]

    print(f"Sequence: {words}")

    # Encoder-only
    encoder = SimpleAttention("BERT", mask_type='full')
    mask = encoder.forward(seq_len)
    print(f"\nEncoder-only ({encoder.name}) attention mask:")
    for row in mask:
        print(f"  {row}")

    # Decoder-only
    decoder = SimpleAttention("GPT", mask_type='causal')
    mask = decoder.forward(seq_len)
    print(f"\nDecoder-only ({decoder.name}) attention mask:")
    for row in mask:
        print(f"  {row}")

    print("\nEncoder-Decoder combines both:")
    print("  • Encoder: full mask (like BERT)")
    print("  • Decoder: causal mask (like GPT)")
    print("  • Plus cross-attention from decoder to encoder")

architecture_comparison()
```

---

## Why This Matters for LLMs

### 1. Different Models for Different Tasks

```python

def task_to_architecture():
    """
    Which architecture for which task
    """
    print("Choosing the Right Architecture")
    print("=" * 60)

    tasks = {
        "Sentiment analysis": "Encoder-only (BERT) - need full context",
        "Text generation": "Decoder-only (GPT) - natural for generation",
        "Translation": "Encoder-decoder (T5) - input → output",
        "Named entity recognition": "Encoder-only - classify each token",
        "Chatbot": "Decoder-only - generate responses",
        "Summarization": "Encoder-decoder - read long, write short",
        "Code completion": "Decoder-only - predict next tokens",
        "Question answering": "Encoder-only (extractive) or Encoder-decoder (generative)"
    }

    for task, arch in tasks.items():
        print(f"  • {task}: {arch}")

task_to_architecture()
```

### 2. Hybrid and Variants

```python

def hybrid_architectures():
    """
    Modern variations
    """
    print("Hybrid and Modern Architectures")
    print("=" * 60)

    hybrids = {
        "Prefix LM": "Decoder-only but can see prefix bidirectionally",
        "UniLM": "Can switch between modes",
        "XLNet": "Permutation language modeling (all orders)",
        "Encoder-decoder with cross-attention": "Standard T5-style",
        "Mixture of Experts": "Sparse activation in any architecture"
    }

    for name, desc in hybrids.items():
        print(f"  • {name}: {desc}")

hybrid_architectures()
```

### 3. Parameter Count Comparison

| Architecture    | Typical Use   | Parameter Distribution       |
| --------------- | ------------- | ---------------------------- |
| Encoder-only    | Understanding | All in encoder               |
| Decoder-only    | Generation    | All in decoder               |
| Encoder-decoder | Translation   | Split (often encoder larger) |

### 4. The Trend Toward Decoder-Only

```python

def trend():
    """
    Why decoder-only models are dominant
    """
    print("The Trend Toward Decoder-Only")
    print("=" * 60)

    print("""
    Recent years have seen a shift:

    2018: BERT (encoder) dominates
    2019: GPT-2 shows decoder strength
    2020: GPT-3 proves scaling decoders
    2022+: Llama, Claude, GPT-4 all decoder-only

    Why? Decoder-only models:
    • Are simpler (one stack)
    • Scale better
    • Can do both generation AND understanding
    • Enable in-context learning
    • Work for chat applications
    """)

    print("\nEncoder-decoder still used for translation/summarization,")
    print("but decoder-only is taking over general-purpose LLMs.")

trend()
```

---

## Architecture Cheat Sheet

| Aspect         | Encoder-Only | Decoder-Only | Encoder-Decoder           |
| -------------- | ------------ | ------------ | ------------------------- |
| Sees future?   | Yes          | No           | Encoder: Yes, Decoder: No |
| Generation?    | No           | Yes          | Yes                       |
| Understanding? | Yes          | Limited      | Yes                       |
| Training cost  | Moderate     | Highest      | Highest                   |
| Inference cost | Low          | Moderate     | High                      |
| Typical size   | 110M-340M    | 1B-175B+     | 220M-11B                  |

---

## Quick Recap

• Encoder-only models (BERT) see the full context bidirectionally—like a translator who reads the entire document before writing, perfect for understanding tasks like classification and QA

• Decoder-only models (GPT) generate left to right, seeing only past words—like a writer building a story word by word, ideal for generation and chat applications

• Encoder-decoder models (T5) combine both—an encoder reads input, then a decoder generates output—best for transformation tasks like translation and summarization, though decoder-only models are increasingly dominant

---

## Mental Hook

> "Encoder-only models are like careful readers who absorb everything before speaking; decoder-only models are like storytellers who build tales one word at a time; encoder-decoder models are like translators who first listen completely, then speak."
