# Few-Shot Learning

## The Intern's Superpower Analogy

Imagine a brilliant intern who joins your company. You don't need to train them for weeks—just show them 2-3 examples of how to do a task, and they immediately understand and can do it perfectly. That's few-shot learning: large language models that can learn new tasks from just a handful of examples, without any parameter updates, just by understanding the pattern from context.

In LLMs, few-shot learning is a magical emergent ability that appears only in very large models. GPT-3 showed that with enough scale, models can learn tasks from just a few examples in the prompt. This changed how we think about AI—instead of fine-tuning for every task, we can just show the model what we want.

---

## What Is Few-Shot Learning?

### The Core Idea

Few-shot learning is the ability to perform a new task given only a few examples in the prompt, without any gradient updates.

```text

Traditional ML:                 Few-Shot Learning:
Train on 1000s of examples      Give 3 examples in prompt
        ↓                               ↓
    Trained model                 Model understands pattern
        ↓                               ↓
    Makes predictions              Makes predictions
    (fixed task)                    (new task on the fly)
```

```python

def fewshot_intro():
    """
    The basic concept of few-shot learning
    """
    print("Few-Shot Learning: Learning from Examples")
    print("=" * 60)

    print("""
    No training! Just show examples in the prompt:

    Prompt:
    Translate English to French:
    English: hello
    French: bonjour
    English: cat
    French: chat
    English: dog
    French:

    Model: chien  (learned the pattern from 2 examples!)
    """)

    print("\nThis works because the model already understands")
    print("language—it just needs to see the pattern.")

fewshot_intro()
```

---

## The Spectrum: Zero-Shot to Many-Shot

### Zero-Shot Learning

No examples, just instructions.

```python

def zeroshot():
    """
    Zero-shot learning: just instructions
    """
    print("Zero-Shot Learning: Just Tell Me What to Do")
    print("=" * 60)

    prompt = """
    Translate the following English word to French:
    dog
    """

    print(f"Prompt: {prompt}")
    print("Model output: chien")

    print("\nZero-shot works for simple tasks but")
    print("can be unreliable for complex ones.")

zeroshot()
```

### One-Shot Learning

Single example.

```python

def oneshot():
    """
    One-shot learning: one example
    """
    print("One-Shot Learning: Show Me Once")
    print("=" * 60)

    prompt = """
    Translate English to French:
    English: cat
    French: chat
    English: dog
    French:
    """

    print(f"Prompt: {prompt}")
    print("Model output: chien")

    print("\nOne example often helps clarify the task.")
    print("Reduces ambiguity about what you want.")

oneshot()
```

### Few-Shot Learning (2-5 examples)

Multiple examples establish pattern.

```python

def fewshot():
    """
    Few-shot learning: 2-5 examples
    """
    print("Few-Shot Learning: Pattern Recognition")
    print("=" * 60)

    prompt = """
    Translate English to French:
    English: hello
    French: bonjour
    English: cat
    French: chat
    English: dog
    French: chien
    English: house
    French:
    """

    print(f"Prompt: {prompt}")
    print("Model output: maison")

    print("\nWith 3-4 examples, the model clearly sees")
    print("the pattern and can handle variations.")

fewshot()
```

### Many-Shot Learning

Many examples (still in context window).

```python

def manyshot():
    """
    Many-shot learning: many examples in context
    """
    print("Many-Shot Learning: Rich Demonstrations")
    print("=" * 60)

    print("""
    With large context windows (100K+ tokens),
    we can give dozens or hundreds of examples!

    Benefits:
    • Complex patterns emerge
    • Can learn formatting rules
    • Fewer errors

    This is like fine-tuning without updating weights—
    just using the model's context window as working memory.
    """)

manyshot()
```

---

## Why Few-Shot Learning Works

### In-Context Learning

```python

def incontext_learning():
    """
    How models learn from context
    """
    print("In-Context Learning: The Secret Sauce")
    print("=" * 60)

    print("""
    During pre-training, models see patterns like:

    "Q: What is 2+2? A: 4"
    "Q: What is 3+5? A: 8"
    "Q: What is the capital of France? A: Paris"

    They learn the pattern: Q: ... A: ...

    At inference, when we give:
    "Q: What is 6+7? A:"

    The model continues the pattern!

    Few-shot learning is just pattern matching
    at a higher level—recognizing the task pattern.
    """)

incontext_learning()
```

### Emergent Ability

```python

def emergent():
    """
    Few-shot learning emerges with scale
    """
    print("Few-Shot Learning: An Emergent Ability")
    print("=" * 60)

    print("""
    Small models (100M parameters):
    • Need fine-tuning for new tasks
    • Can't learn from few examples

    Medium models (1B parameters):
    • Show some few-shot ability
    • Inconsistent

    Large models (100B+ parameters):
    • Strong few-shot learning
    • Can learn complex tasks from few examples

    This ability EMERGES at scale—it's not programmed!
    """)

emergent()
```

---

## Few-Shot Examples

### Classification

```python

def classification_fewshot():
    """
    Few-shot for sentiment classification
    """
    print("Few-Shot Sentiment Classification")
    print("=" * 60)

    prompt = """
    Classify the sentiment as positive or negative:

    Text: I loved this movie!
    Sentiment: positive

    Text: This was terrible
    Sentiment: negative

    Text: Best film ever!
    Sentiment: positive

    Text: What a waste of time
    Sentiment: negative

    Text: Amazing experience
    Sentiment:
    """

    print("Prompt includes 4 examples (2 pos, 2 neg)")
    print("Model infers: positive")
    print("\nWithout examples, might just guess.")

classification_fewshot()
```

### Translation

```python

def translation_fewshot():
    """
    Few-shot for translation
    """
    print("Few-Shot Translation")
    print("=" * 60)

    prompt = """
    Translate to Spanish:

    English: hello
    Spanish: hola

    English: goodbye
    Spanish: adiós

    English: thank you
    Spanish: gracias

    English: how are you?
    Spanish:
    """

    print("Model learns the pattern from 3 examples")
    print("Output: ¿cómo estás?")
    print("\nHandles variations like punctuation correctly.")

translation_fewshot()
```

### Reasoning

```python

def reasoning_fewshot():
    """
    Few-shot for reasoning tasks
    """
    print("Few-Shot Reasoning")
    print("=" * 60)

    prompt = """
    Solve this math word problem:

    Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
       Each can has 3 tennis balls. How many tennis balls does he have now?
    A: Roger starts with 5 balls. 2 cans × 3 balls = 6 balls.
       5 + 6 = 11. The answer is 11.

    Q: The cafeteria had 23 apples. They used 20 to make lunch and bought 6 more.
       How many apples do they have?
    A: They start with 23. Use 20: 23 - 20 = 3. Buy 6: 3 + 6 = 9. The answer is 9.

    Q: There were 9 computers in the server room. They installed 5 more computers.
       Each computer has 2 monitors. How many monitors are there total?
    A:
    """

    print("Model sees reasoning pattern, applies to new problem")
    print("Output: 28 monitors (9+5=14 computers × 2 monitors)")

reasoning_fewshot()
```

---

## Few-Shot vs Fine-Tuning

| Aspect            | Few-Shot Learning | Fine-Tuning            |
| ----------------- | ----------------- | ---------------------- |
| Examples needed   | 2-100 in prompt   | 1000+ labeled          |
| Parameter updates | None              | Yes                    |
| Time              | Seconds           | Hours to days          |
| Cost              | Inference only    | Training cost          |
| Task switching    | Instant           | New fine-tuning needed |
| Performance       | Good              | Better                 |
| Context limit     | Limited by window | No limit               |

### When to Use Each

```python

def when_to_use():
    """
    Choosing between few-shot and fine-tuning
    """
    print("Few-Shot vs Fine-Tuning: When to Use")
    print("=" * 60)

    scenarios = {
        "Quick experiment": "Few-shot (instant results)",
        "Production task": "Fine-tuning (better performance)",
        "Many different tasks": "Few-shot (no overhead)",
        "Single repeated task": "Fine-tuning (optimize)",
        "Limited data": "Few-shot (works with few examples)",
        "Lots of data": "Fine-tuning (leverage it)",
        "Need low latency": "Fine-tuning (no long prompts)"
    }

    for scenario, choice in scenarios.items():
        print(f"  • {scenario}: {choice}")

when_to_use()
```

---

## Why This Matters for LLMs

### 1. Zero/Few-Shot Changed Everything

```python

def paradigm_shift():
    """
    The paradigm shift caused by few-shot learning
    """
    print("How Few-Shot Changed AI")
    print("=" * 60)

    print("""
    Before GPT-3 (pre-2020):
    • Need fine-tuning for every new task
    • Collect thousands of labels
    • Train separate models
    • Weeks of work per task

    After GPT-3:
    • Write a prompt with few examples
    • Get results instantly
    • No training needed
    • Try many tasks in one day

    This democratized AI research!
    """)

paradigm_shift()
```

### 2. In-Context Learning vs. Meta-Learning

```python

def incontext_vs_metalearning():
    """
    Different interpretations of few-shot learning
    """
    print("In-Context Learning vs. Meta-Learning")
    print("=" * 60)

    print("""
    In-Context Learning View:
    • Model recognizes pattern from examples
    • Uses attention to find similarities
    • Extrapolates from demonstrations

    Meta-Learning View:
    • Pre-training taught "how to learn"
    • Model has learned a learning algorithm
    • Few-shot triggers that algorithm

    Probably both are true to some extent!
    """)

incontext_vs_metalearning()
```

### 3. The Scaling Law

```python

def scaling_law():
    """
    How scale enables few-shot learning
    """
    print("Scale Enables Few-Shot Learning")
    print("=" * 60)

    print("""
    GPT-3 paper showed clear trend:

    Model Size  | Few-shot Performance
    100M        | Random chance
    1B          | Slight improvement
    10B         | Good
    100B+       | Excellent

    This suggests few-shot learning is an
    emergent property of scale—like a phase change!
    """)

scaling_law()
```

### 4. Prompt Design Matters

```python

def prompt_design():
    """
    Crafting good few-shot prompts
    """
    print("Prompt Design for Few-Shot Learning")
    print("=" * 60)

    tips = [
        "Use diverse examples covering edge cases",
        "Format consistently (same pattern each time)",
        "Include both positive and negative examples",
        "Put most relevant examples last",
        "Be clear about the task",
        "Use separators (--- or newlines) between examples"
    ]

    print("Tips for effective few-shot prompts:")
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")

prompt_design()
```

---

## Few-Shot Learning Cheat Sheet

| Aspect          | Details                           |
| --------------- | --------------------------------- |
| Zero-shot       | Just instructions, no examples    |
| One-shot        | One example                       |
| Few-shot        | 2-100 examples                    |
| Many-shot       | 100+ examples (with long context) |
| Key requirement | Large model (100B+ parameters)    |
| Mechanism       | In-context learning               |
| Best for        | Quick prototyping, varied tasks   |

---

## Quick Recap

• Few-shot learning lets models learn new tasks from just a handful of examples in the prompt—like an intern who can start working after seeing just 2-3 demonstrations, without any formal training

• This ability emerges only in very large models (100B+ parameters)—smaller models can't do it; it's a magical property that appears at scale, like a phase change

• Few-shot learning has revolutionized AI development—instead of fine-tuning for every task, we can now prototype and test ideas in minutes with just a few examples and a good prompt

---

## Mental Hook

> "Few-shot learning is like showing a brilliant colleague three examples of a new task and having them say 'I get it, let me handle the rest'—they've learned the pattern from just those few demonstrations, no further training needed."
