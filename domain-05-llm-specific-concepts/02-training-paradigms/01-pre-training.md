# Pre-Training

## The Medical School Analogy

Imagine a doctor's education. First, they spend years in medical school learning general medicine—anatomy, physiology, biochemistry (pre-training). They read thousands of textbooks and research papers, building a broad understanding of medicine. Later, they specialize in cardiology (fine-tuning), applying their general knowledge to a specific domain. That's pre-training: teaching a model general language understanding on massive amounts of text before specializing it for specific tasks.

In LLMs, pre-training is the massive first phase where models learn language—grammar, facts, reasoning patterns, and world knowledge—by predicting words on internet-scale data. This is why models like GPT-3 can do so many things: they've already learned the fundamentals during pre-training.

---

## What Is Pre-Training?

### The Core Idea

Pre-training is the initial phase where a model learns general language patterns from large amounts of unlabeled text.

```text

Raw internet text (trillions of tokens):
"The cat sat on the mat"
"Paris is the capital of France"
"2 + 2 = 4"
"Once upon a time..."

            ↓
    [PRE-TRAINING]
    (Self-supervised learning)
            ↓

Base Model with general language understanding
(Knows grammar, facts, reasoning, patterns)
            ↓
    Can be fine-tuned for specific tasks
```

```python

def pretraining_intro():
    """
    The basic concept of pre-training
    """
    print("Pre-Training: Learning the Fundamentals")
    print("=" * 60)

    print("""
    Pre-training objectives:

    • GPT-style: Predict next word
      "The cat sat on the [???]" → "mat"

    • BERT-style: Predict masked words
      "The [MASK] sat on the mat" → "cat"

    The model learns:
    • Grammar and syntax
    • Word meanings
    • World knowledge
    • Reasoning patterns
    • Common sense
    """)

    print("\nAll from raw text, no labels needed!")

pretraining_intro()
```

---

## Pre-Training Objectives

### 1. Causal Language Modeling (GPT-style)

The model predicts the next word, seeing only previous words.

```python

def causal_lm():
    """
    Causal Language Modeling (GPT-style)
    """
    print("Causal Language Modeling (Decoder-Only)")
    print("=" * 60)

    sentence = "The cat sat on the mat"
    words = sentence.split()

    print(f"Sentence: '{sentence}'")
    print("\nTraining examples (predict next word):")

    for i in range(len(words) - 1):
        context = ' '.join(words[:i+1])
        target = words[i+1]
        print(f"  Context: '{context:20}' → Target: '{target}'")

    print("\nThe model learns to:")
    print("• Build language understanding left-to-right")
    print("• Develop next-word prediction skills")
    print("• Capture long-range dependencies")

causal_lm()
```

### 2. Masked Language Modeling (BERT-style)

The model predicts randomly masked words, seeing full context.

```python

def masked_lm():
    """
    Masked Language Modeling (BERT-style)
    """
    print("Masked Language Modeling (Encoder-Only)")
    print("=" * 60)

    sentence = "The cat sat on the mat"
    words = sentence.split()

    print(f"Original: '{sentence}'")
    print("\nMasked examples (predict masked words):")

    # Simulate masking different words
    masked_examples = [
        ("[MASK] cat sat on the mat", "The"),
        ("The [MASK] sat on the mat", "cat"),
        ("The cat [MASK] on the mat", "sat"),
        ("The cat sat [MASK] the mat", "on"),
        ("The cat sat on [MASK] mat", "the"),
        ("The cat sat on the [MASK]", "mat")
    ]

    for masked, target in masked_examples:
        print(f"  Input: '{masked:30}' → Target: '{target}'")

    print("\nAdvantages:")
    print("• Uses context from both sides")
    print("• Learns bidirectional understanding")
    print("• Better for classification tasks")

masked_lm()
```

### 3. Permutation Language Modeling (XLNet)

A hybrid approach that predicts words in random order.

```python

def permutation_lm():
    """
    Permutation Language Modeling (XLNet)
    """
    print("Permutation Language Modeling")
    print("=" * 60)

    print("""
    XLNet combines both approaches:

    Original: "The cat sat"

    Different prediction orders:
    Order 1: [The] → [cat] → [sat]
    Order 2: [cat] → [sat] → [The]
    Order 3: [sat] → [The] → [cat]

    Model must predict in ALL orders!

    Benefits:
    • Sees all context like BERT
    • Generates like GPT
    • Best of both worlds
    """)

permutation_lm()
```

---

## The Scale of Pre-Training

### Data and Compute

```python

def pretraining_scale():
    """
    The massive scale of pre-training
    """
    print("The Scale of Pre-Training")
    print("=" * 60)

    models = {
        "GPT-2": {
            "data": "40 GB (8M documents)",
            "tokens": "10B",
            "compute": "petaflop-days"
        },
        "GPT-3": {
            "data": "570 GB",
            "tokens": "300B",
            "compute": "3,640 petaflop-days"
        },
        "Llama 3": {
            "data": "15T tokens",
            "tokens": "15T",
            "compute": "~10⁷ GPU hours"
        },
        "GPT-4": {
            "data": "Unknown (estimated 13T tokens)",
            "tokens": "13T+",
            "compute": "$100M+ estimated"
        }
    }

    for model, stats in models.items():
        print(f"\n{model}:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")

pretraining_scale()
```

### What the Model Sees

```python

def pretraining_data():
    """
    Types of data in pre-training
    """
    print("What Models Learn During Pre-Training")
    print("=" * 60)

    data_sources = {
        "Books": "Grammar, narrative structure, long-range dependencies",
        "Wikipedia": "Facts, encyclopedia knowledge, formal writing",
        "Web pages": "Diverse styles, opinions, current information",
        "Code": "Logical structure, programming syntax",
        "Scientific papers": "Technical vocabulary, formal reasoning",
        "Social media": "Informal language, dialogue patterns"
    }

    for source, knowledge in data_sources.items():
        print(f"  • {source}: {knowledge}")

pretraining_data()
```

---

## The Pre-Training Process

### Step-by-Step

```python

def pretraining_process():
    """
    The pre-training pipeline
    """
    print("The Pre-Training Pipeline")
    print("=" * 60)

    steps = [
        "Step 1: Data Collection",
        "  • Crawl web, books, papers",
        "  • Clean and filter",
        "  • Remove duplicates, personal info",

        "Step 2: Tokenization",
        "  • Train tokenizer on corpus",
        "  • Convert text to token IDs",

        "Step 3: Model Initialization",
        "  • Random weights",
        "  • Set architecture (layers, heads, dims)",

        "Step 4: Training Loop",
        "  • For each batch of text:",
        "    - Forward pass (predict words)",
        "    - Calculate loss",
        "    - Backward pass (gradients)",
        "    - Update weights",

        "Step 5: Validation",
        "  • Check loss on held-out data",
        "  • Adjust hyperparameters",

        "Step 6: Save Checkpoints",
        "  • Save model periodically",
        "  • Final pre-trained model"
    ]

    for step in steps:
        print(f"\n{step}")

pretraining_process()
```

### The Loss Curve

```python

def loss_curve():
    """
    How loss decreases during pre-training
    """
    print("Pre-Training Loss Curve")
    print("=" * 60)

    print("""
    Loss
    ↑
    │    ███
    │   █   ██
    │  █      ██
    │ █         ██
    │█            ██
    │                ██
    │                   ██
    │                      ███
    └──────────────────────────→ Training steps

    Phase 1: Rapid learning (basic patterns)
    Phase 2: Slower improvement (refinement)
    Phase 3: Diminishing returns
    """)

    print("\nTraining stops when:")
    print("• Compute budget exhausted")
    print("• Loss plateau")
    print("• Validation metrics stop improving")

loss_curve()
```

---

## Why Pre-Training Works

### Transfer Learning

```python

def transfer_learning():
    """
    The magic of transfer learning
    """
    print("Transfer Learning: Standing on Shoulders of Giants")
    print("=" * 60)

    print("""
    Without pre-training (training from scratch):

    Task A: Need 1M labeled examples
    Task B: Need another 1M labeled examples
    Task C: Need another 1M labeled examples

    With pre-training:

    Pre-train on 1B unlabeled examples
        ↓
    Task A: Need 1,000 labeled examples
    Task B: Need 1,000 labeled examples
    Task C: Need 1,000 labeled examples

    The model already knows language!
    """)

    print("\nThis is why modern AI is so powerful.")

transfer_learning()
```

### Emergent Abilities

```python

def emergent_abilities():
    """
    Abilities that emerge during pre-training
    """
    print("Emergent Abilities During Pre-Training")
    print("=" * 60)

    print("""
    As models scale, new abilities appear:

    • Small models (100M): Basic grammar, word meanings
    • Medium models (1B): Sentence structure, simple reasoning
    • Large models (10B): Translation, summarization
    • Very large (100B+): Few-shot learning, arithmetic, code
    • GPT-3 scale: In-context learning, chain-of-thought

    These aren't programmed—they EMERGE from scale!
    """)

emergent_abilities()
```

---

## Pre-Training vs Fine-Tuning

| Aspect   | Pre-Training             | Fine-Tuning          |
| -------- | ------------------------ | -------------------- |
| Data     | Massive, unlabeled       | Small, labeled       |
| Goal     | Learn language generally | Learn specific task  |
| Time     | Weeks to months          | Hours to days        |
| Cost     | Millions of dollars      | Thousands of dollars |
| Model    | Base model               | Specialized model    |
| Examples | GPT-3 base               | ChatGPT (fine-tuned) |

### The Two-Stage Process

```text

Stage 1: Pre-training
[Internet text] → [Base Model] → General language understanding
                      ↓
Stage 2: Fine-tuning
[Task-specific data] → [Fine-tuned Model] → Specialized for task
                      ↓
Final: Deployment
[User queries] → [Fine-tuned Model] → Task outputs
```

---

## Why This Matters for LLMs

### 1. It's What Makes LLMs "Large"

```python

def why_large():
    """
    Why pre-training needs scale
    """
    print("Why Pre-Training Needs Scale")
    print("=" * 60)

    reasons = [
        "More data → broader knowledge",
        "More parameters → more capacity",
        "Longer training → deeper patterns",
        "Diverse sources → robust understanding"
    ]

    print("Scale enables:")
    for reason in reasons:
        print(f"  • {reason}")

why_large()
```

### 2. The Chinchilla Law

```python

def chinchilla():
    """
    Optimal pre-training scale
    """
    print("The Chinchilla Law: Optimal Pre-Training")
    print("=" * 60)

    print("""
    DeepMind's Chinchilla found:

    For optimal performance, model size and training tokens
    should scale together:

    Old approach: Make model bigger, same data
    Chinchilla approach: Scale both proportionally

    Example: If you double model size,
             double training tokens too!

    This changed how everyone pre-trains LLMs.
    """)

chinchilla()
```

### 3. Pre-Training Costs

```python

def pretraining_costs():
    """
    The real costs of pre-training
    """
    print("The Real Costs of Pre-Training")
    print("=" * 60)

    costs = {
        "GPT-3": "~$4.6M",
        "Llama 3 70B": "~$15-20M",
        "GPT-4": "~$100M+",
        "Google's Gemini": "~$200M+"
    }

    for model, cost in costs.items():
        print(f"  • {model}: {cost}")

    print("\nThis is why only big tech companies")
    print("pre-train large models from scratch.")

pretraining_costs()
```

### 4. Environmental Impact

```python

def environmental():
    """
    Environmental considerations
    """
    print("Environmental Impact of Pre-Training")
    print("=" * 60)

    print("""
    Pre-training a large model:

    • Energy: Gigawatt-hours of electricity
    • CO2: Equivalent to lifetime of 5 cars
    • Water: Millions of liters for cooling

    This has led to:
    • More efficient architectures
    • Smaller models (Llama 3 8B)
    • Distillation techniques
    """)

environmental()
```

---

## Pre-Training Cheat Sheet

| Aspect      | Details                                        |
| ----------- | ---------------------------------------------- |
| Objective   | Next word prediction (GPT) or masked LM (BERT) |
| Data        | Trillions of tokens from internet              |
| Time        | Weeks to months on thousands of GPUs           |
| Cost        | Millions of dollars                            |
| Output      | Base model with general language understanding |
| Key insight | Scale is all you need                          |

---

## Quick Recap

• Pre-training is the massive first phase where models learn general language from internet-scale data—like a doctor's medical school education, building broad knowledge before specializing

• Models learn through simple objectives like next-word prediction or masked word prediction—no labels needed, just raw text, yet they absorb grammar, facts, reasoning, and world knowledge

• Pre-training is why LLMs are so powerful and versatile—the base model can then be fine-tuned for specific tasks with minimal data, transferring all that general knowledge to new domains

---

## Mental Hook

> "Pre-training is like sending a model to school for years, reading millions of books, learning everything about language—by the time it graduates, it's ready to tackle almost any task with just a little specialized training."
