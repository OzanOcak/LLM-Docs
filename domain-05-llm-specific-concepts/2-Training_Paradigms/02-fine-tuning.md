# Fine-Tuning

## The Medical Specialist Analogy

Imagine a doctor who's completed medical school (pre-training). They understand medicine broadly, but now they want to become a cardiologist. They don't need to go through medical school again—they take their general knowledge and do specialized training on heart conditions, reading ECGs, and cardiac cases. That's fine-tuning: taking a pre-trained model and specializing it for specific tasks with smaller, targeted datasets.

In LLMs, fine-tuning is how we adapt general-purpose models for specific applications. ChatGPT started as GPT-3 (pre-trained), then was fine-tuned on conversational data to become a helpful assistant. Fine-tuning is much faster and cheaper than pre-training, making AI accessible to more people and use cases.

---

## What Is Fine-Tuning?

### The Core Idea

Fine-tuning takes a pre-trained model and continues training on a smaller, task-specific dataset.

```text

Pre-trained Model (generalist):
    ↓
[Fine-Tuning on task-specific data]
    ↓
Fine-tuned Model (specialist)
```

Example:
GPT-3 (general) → Fine-tuned on customer service chats → Customer service bot
GPT-3 (general) → Fine-tuned on legal documents → Legal assistant
GPT-3 (general) → Fine-tuned on conversational data → ChatGPT

```python

def finetuning_intro():
    """
    The basic concept of fine-tuning
    """
    print("Fine-Tuning: From Generalist to Specialist")
    print("=" * 60)

    print("""
    Pre-trained model knows:
    • Grammar, vocabulary, facts
    • Reasoning, patterns, world knowledge

    Fine-tuning adds:
    • Task-specific behavior
    • Domain expertise
    • Desired response style
    """)

    print("\nAll with much less data than pre-training!")

finetuning_intro()
```

---

## Why Fine-Tuning Works

### Transfer Learning in Action

```python

def transfer_learning():
    """
    How knowledge transfers from pre-training
    """
    print("Transfer Learning: Standing on Shoulders")
    print("=" * 60)

    print("""
    Without pre-training (training from scratch):

    Task: Sentiment analysis
    • Need 100,000 labeled examples
    • Train for weeks
    • Model starts from zero

    With pre-training + fine-tuning:

    Task: Sentiment analysis
    • Need 1,000 labeled examples
    • Fine-tune for hours
    • Model already understands language!
    """)

    print("\nThis is the magic of fine-tuning!")

transfer_learning()
```

### What Gets Updated

```python

def what_updates():
    """
    What changes during fine-tuning
    """
    print("What Changes During Fine-Tuning")
    print("=" * 60)

    print("""
    All parameters can be updated, but:

    • Early layers: General features (barely change)
    • Middle layers: Language patterns (small changes)
    • Late layers: Task-specific (biggest changes)

    Analogy:
    Early layers = "how language works"
    Late layers = "what to do with it"
    """)

    print("\nThis is why fine-tuning needs less data—")
    print("most knowledge is already there!")

what_updates()
```

---

## Types of Fine-Tuning

### 1. Full Fine-Tuning

Update all model parameters on task data.

```python

def full_finetuning():
    """
    Full fine-tuning: update everything
    """
    print("Full Fine-Tuning")
    print("=" * 60)

    print("""
    Process:
    1. Take pre-trained model (all weights)
    2. Train on task dataset
    3. Update ALL parameters

    Pros:
    • Maximum flexibility
    • Best performance
    • Can adapt completely

    Cons:
    • Expensive (updates all weights)
    • Need good task data
    • Risk of overfitting
    • Must store full model per task
    """)

full_finetuning()
```

### 2. Parameter-Efficient Fine-Tuning (PEFT)

Update only a small subset of parameters.

```python

def peft():
    """
    Parameter-Efficient Fine-Tuning
    """
    print("Parameter-Efficient Fine-Tuning (PEFT)")
    print("=" * 60)

    print("""
    Key insight: Most knowledge is already there!
    We just need small adjustments.

    Methods:

    • Adapters: Insert small trainable layers
    • Prefix tuning: Learn special tokens
    • LoRA: Low-rank adaptation (most popular)

    LoRA example:
    Instead of updating W (huge matrix),
    learn A×B where A and B are small matrices.

    Update size: 175B → few million parameters!
    """)

peft()
```

### 3. LoRA (Low-Rank Adaptation)

The most popular PEFT method.

```python

def lora_explained():
    """
    LoRA in detail
    """
    print("LoRA: Low-Rank Adaptation")
    print("=" * 60)

    print("""
    Instead of updating weight matrix W (size d×k):

    W_new = W_original + A×B

    Where:
    • A is d×r (small, r=8 or 16)
    • B is r×k (small)
    • r << d,k

    During fine-tuning:
    • W_original is frozen (no gradient)
    • Only A and B are trained

    After training:
    • Can merge back: W_original + A×B
    • Or keep separate for multiple tasks
    """)

    print("\nUsed in: Most modern fine-tuning pipelines")

lora_explained()
```

---

## Fine-Tuning Examples

### Sentiment Analysis Fine-Tuning

```python

def sentiment_finetuning():
    """
    Fine-tuning for sentiment analysis
    """
    print("Example: Fine-Tuning for Sentiment Analysis")
    print("=" * 60)

    # Pre-trained model knows language
    print("Pre-trained model can complete sentences:")
    print("  'I loved this movie, it was...' → 'great'")

    # Fine-tuning data
    finetune_data = [
        ("I loved this movie!", "positive"),
        ("This was terrible", "negative"),
        ("Best film ever!", "positive"),
        ("Waste of time", "negative"),
        ("Amazing experience", "positive"),
        ("Disappointing", "negative")
    ]

    print(f"\nFine-tuning data: {len(finetune_data)} examples")
    for text, label in finetune_data:
        print(f"  '{text}' → {label}")

    print("\nAfter fine-tuning:")
    print("  Model can now CLASSIFY sentiment directly")
    print("  'This movie is fantastic!' → positive")
    print("  'What a waste of money' → negative")

sentiment_finetuning()
```

### Chat Fine-Tuning (ChatGPT)

```python

def chat_finetuning():
    """
    How ChatGPT was created
    """
    print("Example: Fine-Tuning ChatGPT")
    print("=" * 60)

    print("""
    Stage 1: Start with GPT-3 (pre-trained)

    Stage 2: Supervised Fine-Tuning (SFT)
    Data: Human-written conversations
    Input: User message
    Output: Desired assistant response

    Stage 3: Reinforcement Learning from Human Feedback (RLHF)
    Data: Human preferences between responses
    Learn to prefer helpful, honest, harmless responses

    Result: ChatGPT behaves like a helpful assistant,
    not just a next-word predictor!
    """)

chat_finetuning()
```

### Instruction Fine-Tuning

```python

def instruction_tuning():
    """
    Fine-tuning on instructions
    """
    print("Instruction Fine-Tuning")
    print("=" * 60)

    examples = [
        ("Translate to French: Hello", "Bonjour"),
        ("Summarize: Long article...", "Short summary"),
        ("What's 2+2?", "4"),
        ("Write a poem about cats", "Whiskers twitch...")
    ]

    print("Training on (instruction, response) pairs:")
    for inst, resp in examples:
        print(f"  Instruction: {inst}")
        print(f"  Response: {resp}\n")

    print("After fine-tuning, model follows instructions!")
    print("This is how we get models like InstructGPT.")

instruction_tuning()
```

---

## Fine-Tuning vs Pre-Training

| Aspect    | Pre-Training             | Fine-Tuning            |
| --------- | ------------------------ | ---------------------- |
| Data size | Trillions of tokens      | Thousands to millions  |
| Data type | Raw, unlabeled           | Labeled, task-specific |
| Compute   | Thousands of GPUs, weeks | Single GPU, hours/days |
| Cost      | Millions of dollars      | Hundreds to thousands  |
| Goal      | Learn language generally | Learn specific task    |
| Output    | Base model               | Specialized model      |

### The Iceberg Analogy

```text

Pre-training (below water, massive):
    ██████████████████████████
    ██████████████████████████
    ██████████████████████████  ← Most knowledge here!
    ██████████████████████████

Fine-tuning (tip of iceberg):
            ███████
            ███████  ← Small task-specific adaptation
            ███████
```

---

## Why This Matters for LLMs

### 1. Democratizes AI Development

```python

def democratization():
    """
    Fine-tuning makes AI accessible
    """
    print("Fine-Tuning Democratizes AI")
    print("=" * 60)

    print("""
    Without fine-tuning:
    • Only Google, OpenAI, etc. can build LLMs
    • Need $100M+ and thousands of GPUs

    With fine-tuning:
    • Any company can adapt existing models
    • Need $100 and a single GPU
    • Custom AI for specific needs
    """)

    print("\nThis is why we have specialized models for:")
    print("• Legal research")
    print("• Medical diagnosis")
    print("• Customer service")
    print("• Code assistance")

democratization()
```

### 2. Multiple Tasks, One Base Model

```python

def multitask():
    """
    One base model, many specializations
    """
    print("One Base Model, Many Specializations")
    print("=" * 60)

    print("""
    Pre-trained GPT-3 (one model)
              ↓
    ┌────┬───┴───┬────┐
    ↓    ↓       ↓    ↓
    Chat Legal Code Medical
    Bot  Doc     Asst  Asst

    Each fine-tuned version shares the same
    base knowledge but specializes differently!
    """)

multitask()
```

### 3. Data Efficiency

```python

def data_efficiency():
    """
    How little data fine-tuning needs
    """
    print("Data Efficiency of Fine-Tuning")
    print("=" * 60)

    examples = {
        "Sentiment analysis": "100-1000 examples",
        "Question answering": "1,000-10,000",
        "Instruction following": "10,000-100,000",
        "Chat (ChatGPT)": "~100,000 conversations"
    }

    print("Typical data needed for fine-tuning:")
    for task, amount in examples.items():
        print(f"  • {task}: {amount}")

    print("\nCompare to pre-training: billions of tokens!")
    print("This is why fine-tuning is so practical.")

data_efficiency()
```

### 4. Catastrophic Forgetting

```python

def catastrophic_forgetting():
    """
    The risk of fine-tuning
    """
    print("The Risk: Catastrophic Forgetting")
    print("=" * 60)

    print("""
    Problem: Fine-tuning on new tasks can make
             model forget previous knowledge.

    Example:
    Fine-tune on medical data → great doctor
    But forgets how to write poetry!

    Solutions:
    • Mix in general data during fine-tuning
    • Use PEFT methods (LoRA)
    • Keep base model frozen
    • Multi-task learning
    """)

catastrophic_forgetting()
```

---

## Fine-Tuning Methods Comparison

| Method           | Parameters Updated | Storage per Task | Quality     | Speed |
| ---------------- | ------------------ | ---------------- | ----------- | ----- |
| Full fine-tuning | All                | Full model       | Best        | Slow  |
| LoRA             | <1%                | Few MB           | Nearly best | Fast  |
| Adapters         | <1%                | Few MB           | Good        | Fast  |
| Prefix tuning    | Tiny               | KB               | Good        | Fast  |
| Prompt tuning    | Tiny               | KB               | OK          | Fast  |

---

## Quick Recap

• Fine-tuning adapts pre-trained models to specific tasks with small datasets—like a doctor specializing in cardiology after medical school, building on general knowledge with targeted training

• It's much cheaper and faster than pre-training—hours instead of months, thousands instead of millions, making AI accessible to more people and use cases

• Parameter-efficient methods like LoRA make fine-tuning even more practical—updating tiny fractions of the model preserves general knowledge while adapting to new tasks

---

## Mental Hook

> "Fine-tuning is like taking a brilliant generalist who's read millions of books and giving them a week of specialized training to become an expert lawyer, doctor, or programmer—they already know the language, they just need to learn the specialty."
