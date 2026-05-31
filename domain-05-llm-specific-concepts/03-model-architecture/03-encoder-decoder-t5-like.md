# Encoder-Decoder Models (T5-like)

## The Translator's Booth Analogy

Imagine a United Nations interpreter. First, they listen to the entire speech in English, taking in all the nuance, context, and meaning (encoding). Then, they step into their booth and deliver the translation in French, one sentence at a time, referring back to their notes of the original speech as needed (decoding). That's an encoder-decoder model: one part reads and understands the input, the other part generates the output based on that understanding.

In the LLM family, encoder-decoder models like T5 (Text-to-Text Transfer Transformer) are the versatile workhorses. They treat every task as text-to-text—translation, summarization, question answering, classification—all by mapping input text to output text. They combine the understanding power of encoders with the generation power of decoders.

---

## What Is an Encoder-Decoder Model?

### The Core Idea

Encoder-decoder models use both an encoder to process input and a decoder to generate output.

```text

Encoder-Decoder (T5):
    [The] [cat] [sat]  (input)
       ↓    ↓    ↓
    ┌─────────────────┐
    │    ENCODER      │  (bidirectional understanding)
    └─────────────────┘
       ↓    ↓    ↓
    [hidden representations]
              ↓
    ┌─────────────────┐
    │    DECODER      │  (causal generation)
    └─────────────────┘
       ↓    ↓    ↓
    [Le] [chat] [s'est] [assis]  (output)
```

```python

def encoder_decoder_intro():
    """
    The basic concept of encoder-decoder models
    """
    print("Encoder-Decoder Models: Text-to-Text Framework")
    print("=" * 60)

    print("""
    Architecture:
    • Encoder: Bidirectional (like BERT)
      - Understands input fully
      - Creates rich representations

    • Decoder: Causal (like GPT)
      - Generates output left-to-right
      - Can attend to encoder representations

    • Cross-attention: Decoder attends to encoder
      - Connects understanding to generation
    """)

encoder_decoder_intro()
```

---

## How Encoder-Decoder Models Work

### The Two-Phase Process

```python

def two_phase():
    """
    Encoding phase and decoding phase
    """
    print("The Two Phases: Encoding then Decoding")
    print("=" * 60)

    input_text = "The cat sat on the mat"
    output_text = "Le chat s'est assis sur le tapis"

    print(f"Input (English): '{input_text}'")
    print(f"Output (French): '{output_text}'")

    print("\n" + "=" * 40)
    print("PHASE 1: ENCODING")
    print("=" * 40)
    print("Encoder reads entire input bidirectionally:")
    print("  [The] ←→ [cat] ←→ [sat] ←→ [on] ←→ [the] ←→ [mat]")
    print("  Creates rich representations for each token")
    print("  Final encoder states capture full meaning")

    print("\n" + "=" * 40)
    print("PHASE 2: DECODING")
    print("=" * 40)
    print("Decoder generates output left-to-right:")
    print("  Step 1: <START> → 'Le'   (attends to encoder)")
    print("  Step 2: 'Le' → 'chat'    (attends to encoder + previous)")
    print("  Step 3: 'chat' → 's'est'")
    print("  Step 4: 's'est' → 'assis'")
    print("  ...")
    print("  Final: </s> (end token)")

two_phase()
```

### Cross-Attention

```python

def cross_attention():
    """
    How decoder attends to encoder
    """
    print("Cross-Attention: Connecting Encoder and Decoder")
    print("=" * 60)

    print("""
    Decoder has TWO attention mechanisms:

    1. Self-attention (causal)
       • Attends to previous decoder outputs
       • Like GPT's attention

    2. Cross-attention
       • Attends to encoder outputs
       • Queries from decoder, Keys/Values from encoder
       • Decides what input information to use NOW

    When generating "chat" (cat), cross-attention might focus on:
    • "cat" in encoder (0.8)
    • "sat" (0.1)
    • "mat" (0.05)
    • etc.
    """)

cross_attention()
```

---

## T5 Architecture

### Text-to-Text Framework

```python

def text_to_text():
    """
    T5's unifying text-to-text approach
    """
    print("T5: Text-to-Text Framework")
    print("=" * 60)

    print("""
    T5 treats EVERY task as text input → text output:

    Task                    Input                     Output
    ──────────────────────────────────────────────────────────
    Translation   "translate English to German: Hello"  "Hallo"

    Summarization "summarize: long article..."          "summary"

    Classification "mnli premise: ... hypothesis: ..."  "entailment"

    QA            "question: ... context: ..."          "answer"

    Regression    "stsb sentence1: ... sentence2: ..."  "3.8"

    All with the same model, same loss, same architecture!
    """)

text_to_text()
```

### T5 Model Sizes

```python

def t5_sizes():
    """
    T5 model variants
    """
    print("T5 Model Sizes")
    print("=" * 60)

    variants = {
        "T5-small": "60M",
        "T5-base": "220M",
        "T5-large": "770M",
        "T5-3B": "3B",
        "T5-11B": "11B"
    }

    for name, params in variants.items():
        print(f"  • {name}: {params} parameters")

t5_sizes()
```

---

## What Encoder-Decoder Models Can Do

### 1. Machine Translation

```python

def translation():
    """
    Translation with encoder-decoder
    """
    print("Machine Translation")
    print("=" * 60)

    examples = [
        ("translate English to German: Hello", "Hallo"),
        ("translate English to Spanish: Good morning", "Buenos días"),
        ("translate English to French: How are you?", "Comment allez-vous?")
    ]

    for inp, out in examples:
        print(f"\n  Input: '{inp}'")
        print(f"  Output: '{out}'")

    print("\nThis is what encoder-decoders were designed for!")

translation()
```

### 2. Summarization

```python

def summarization():
    """
    Text summarization
    """
    print("Text Summarization")
    print("=" * 60)

    article = """
    The quick brown fox jumps over the lazy dog. The fox was quick and brown.
    The dog was lazy and didn't move. This is a long article about foxes and dogs.
    It contains many sentences that need to be summarized into something shorter.
    """

    summary = "A quick brown fox jumps over a lazy dog."

    print(f"Long article: '{article.strip()}'")
    print(f"\nSummary: '{summary}'")

    print("\nEncoder reads full article, decoder generates summary!")

summarization()
```

### 3. Question Answering

```python

def qa():
    """
    Question answering
    """
    print("Question Answering")
    print("=" * 60)

    context = "The capital of France is Paris. It is known for the Eiffel Tower."

    examples = [
        ("What is the capital of France?", "Paris"),
        ("What is Paris known for?", "the Eiffel Tower")
    ]

    for question, answer in examples:
        print(f"\nContext: {context}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")

qa()
```

### 4. Classification as Text Generation

```python

def classification():
    """
    Classification treated as text generation
    """
    print("Classification as Text Generation")
    print("=" * 60)

    examples = [
        ("mnli premise: A dog is playing. hypothesis: An animal is outside.", "entailment"),
        ("mnli premise: A cat is sleeping. hypothesis: The cat is running.", "contradiction"),
        ("stsb sentence1: The car is red. sentence2: The vehicle is crimson.", "4.2")
    ]

    for inp, out in examples:
        print(f"\n  Input: {inp}")
        print(f"  Output: {out}")

classification()
```

---

## Encoder-Decoder vs Other Architectures

| Aspect            | Encoder-Decoder (T5) | Encoder-Only (BERT) | Decoder-Only (GPT) |
| ----------------- | -------------------- | ------------------- | ------------------ |
| Encoder attention | Bidirectional        | Bidirectional       | N/A                |
| Decoder attention | Causal + cross       | N/A                 | Causal             |
| Training          | Span corruption      | Masked LM           | Next word          |
| Input → output    | Yes                  | No (repr only)      | Yes (continuation) |
| Best for          | Transformation       | Understanding       | Generation         |
| Example           | Translation          | Classification      | Chat               |

### When to Use Encoder-Decoder

```python

def when_to_use():
    """
    When to choose encoder-decoder
    """
    print("When to Use Encoder-Decoder Models")
    print("=" * 60)

    use_cases = [
        "Machine translation",
        "Text summarization",
        "Question answering (generative)",
        "Data-to-text generation",
        "Grammar correction",
        "Paraphrasing",
        "Any task with different input/output formats"
    ]

    print("Perfect for tasks that transform text:")
    for task in use_cases:
        print(f"  • {task}")

when_to_use()
```

---

## Why This Matters for LLMs

### 1. T5's Influence

```python

def t5_influence():
    """
    T5's impact on NLP
    """
    print("T5's Impact on NLP")
    print("=" * 60)

    print("""
    T5 introduced several key ideas:

    1. Text-to-text framework
       • Unify all tasks
       • Same model, same loss, same decoding

    2. Span corruption (instead of masked LM)
       • Mask contiguous spans of text
       • Model must reconstruct them

    3. Systematic study of transfer learning
       • What data? What objectives? What scale?

    4. FLAN (Fine-tuned LAnguage Net)
       • Instruction tuning at scale
       • Led to better zero-shot performance
    """)

t5_influence()
```

### 2. Modern Encoder-Decoder Models

```python

def modern_ed():
    """
    Modern encoder-decoder models
    """
    print("Modern Encoder-Decoder Models")
    print("=" * 60)

    models = {
        "T5": "Original text-to-text",
        "FLAN-T5": "T5 + instruction tuning",
        "BART": "Denoising autoencoder (bidirectional encoder + autoregressive decoder)",
        "Pegasus": "Optimized for summarization",
        "M2M-100": "Multilingual translation",
        "mT5": "Multilingual T5"
    }

    for model, desc in models.items():
        print(f"  • {model}: {desc}")

modern_ed()
```

### 3. Strengths and Weaknesses

```python

def strengths_weaknesses():
    """
    Pros and cons of encoder-decoder
    """
    print("Encoder-Decoder: Strengths and Weaknesses")
    print("=" * 60)

    strengths = [
        "Best for transformation tasks",
        "Encoder can fully understand input",
        "Decoder can generate freely",
        "Cross-attention is powerful",
        "Separate understanding and generation"
    ]

    weaknesses = [
        "Larger than encoder-only",
        "Slower than decoder-only (two passes)",
        "More complex architecture",
        "Less scalable than decoder-only?",
        "Not as popular for chatbots"
    ]

    print("\nStrengths:")
    for s in strengths:
        print(f"  • {s}")

    print("\nWeaknesses:")
    for w in weaknesses:
        print(f"  • {w}")

strengths_weaknesses()
```

### 4. The Future of Encoder-Decoder

```python

def future():
    """
    Where encoder-decoder fits in modern AI
    """
    print("The Future of Encoder-Decoder Models")
    print("=" * 60)

    print("""
    Current trends:

    • Decoder-only dominates chat/completion
    • Encoder-only dominates understanding tasks
    • Encoder-decoder specialized for transformations

    But encoder-decoder still best for:
    • Translation
    • Summarization
    • Any task with different input/output

    Hybrid approaches emerging:
    • Non-autoregressive decoders (faster)
    • Diffusion models for text
    • Unified architectures (like UL2)
    """)

future()
```

---

## Encoder-Decoder Cheat Sheet

| Aspect            | Details                                |
| ----------------- | -------------------------------------- |
| Architecture      | Encoder + Decoder with cross-attention |
| Encoder attention | Bidirectional                          |
| Decoder attention | Causal + cross-attention               |
| Training          | Span corruption (T5) or denoising      |
| Best for          | Text transformation tasks              |
| Examples          | T5, BART, FLAN-T5, mT5                 |
| Key innovation    | Text-to-text framework                 |
| Scale             | Up to 11B (T5), larger possible        |

---

## Quick Recap

• Encoder-decoder models combine bidirectional understanding with autoregressive generation—like a translator who first listens to the whole speech, then delivers the translation sentence by sentence

• T5's text-to-text framework treats every task as mapping input text to output text—unifying translation, summarization, classification, and more under one architecture

• While decoder-only models dominate chatbots, encoder-decoders remain best for transformation tasks—translation, summarization, and any task where input and output have different structures

---

## Mental Hook

> "Encoder-decoder models are like the perfect translator—they listen completely before speaking, understand every nuance of the original, and then craft their response while still being able to glance back at their notes."
