# Distillation

## The Master and Apprentice Analogy

Imagine a master sushi chef who has spent 40 years perfecting their craft. They can't be everywhere at once, so they train apprentices. The apprentices don't need 40 years—they learn from the master's examples, guidance, and corrections. After training, they can produce sushi that's almost as good, much faster, and they can work in multiple locations. That's distillation: a smaller "student" model learns from a larger "teacher" model, capturing its knowledge and reasoning patterns in a more efficient package.

In LLM deployment, distillation is how we create models like DistilBERT, TinyBERT, and Phi-3. These models are 40-60% smaller than their teachers, run 2-5x faster, yet retain 95-99% of the performance. It's not just compression—it's knowledge transfer.

---

## What Is Distillation?

### The Core Idea

Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" model's behavior.

```text

Traditional Training:                 Distillation:
┌──────────────┐                     ┌──────────────┐
│   Labels     │                     │   Teacher    │
│   (one-hot)  │                     │    Model     │
└──────┬───────┘                     └──────┬───────┘
       ↓                                    ↓
┌──────────────┐                     ┌──────────────┐
│   Student    │                     │   Student    │
│    Model     │                     │    Model     │
└──────────────┘                     └──────────────┘
       ↓                                    ↓
   "cat" (hard)                        [0.7, 0.2, 0.1] (soft)
                                      (cat, dog, rabbit)

Student learns from                  Student learns the
exact answers                        teacher's reasoning!
```

```python

def distillation_intro():
    """
    The basic concept of knowledge distillation
    """
    print("Distillation: Learning from the Master")
    print("=" * 60)

    print("""
    Why distill?

    • SMALLER MODELS: 40-60% size reduction
    • FASTER INFERENCE: 2-5x speedup
    • LOWER COST: Cheaper to serve
    • EDGE DEPLOYMENT: Runs on phones, browsers
    • KNOWLEDGE TRANSFER: Captures reasoning, not just answers
    """)

distillation_intro()
```

---

## How Distillation Works

### Hard Labels vs Soft Labels

```python

def hard_vs_soft():
    """
    The key difference: soft labels contain more information
    """
    print("Hard Labels vs Soft Labels")
    "=" * 60

    # Example: classifying an image of a cat
    hard_label = [1, 0, 0, 0, 0]  # "cat" only

    # Teacher's soft probabilities
    teacher_probs = [0.7, 0.2, 0.05, 0.03, 0.02]  # cat, dog, rabbit, fox, bear

    print(f"Hard label (traditional):")
    print(f"  [1, 0, 0, 0, 0]")
    print(f"  'It's definitely a cat'")

    print(f"\nSoft labels (distillation):")
    print(f"  {teacher_probs}")
    print(f"  'It's probably a cat, could be a dog, maybe a rabbit...'")

    print("\nSoft labels tell the student:")
    print("• The main answer (cat)")
    print("• Similar classes (dog is somewhat cat-like)")
    print("• Unrelated classes (bear is very different)")
    print("• The teacher's confidence")

hard_vs_soft()
```

### Temperature Scaling

```python

def temperature():
    """
    Temperature makes soft labels even softer
    """
    print("Temperature: Softening the Probabilities")
    "=" * 60

    import numpy as np
    import math

    def soften(probs, temperature):
        """Apply temperature scaling to probabilities"""
        logits = [math.log(p + 1e-10) for p in probs]
        scaled = [l / temperature for l in logits]
        exp_scaled = [math.exp(s) for s in scaled]
        total = sum(exp_scaled)
        return [e / total for e in exp_scaled]

    teacher_probs = [0.7, 0.2, 0.05, 0.03, 0.02]

    print(f"Original teacher probs: {teacher_probs}")

    for T in [1.0, 2.0, 5.0]:
        softened = soften(teacher_probs, T)
        rounded = [round(p, 3) for p in softened]
        print(f"\nTemperature T={T}: {rounded}")
        if T > 1:
            print(f"  Softer distribution—more information for student!")

temperature()
```

---

## Distillation Loss Function

```python

def distillation_loss():
    """
    How distillation loss is calculated
    """
    print("Distillation Loss: Learning from Teacher")
    "=" * 60

    print("""
    Total Loss = α × StudentLoss(hard_labels) + (1-α) × DistillationLoss(teacher_soft)

    Where:
    • StudentLoss: Traditional cross-entropy with true labels
    • DistillationLoss: Cross-entropy with teacher's soft labels
    • α: Balances the two (usually 0.5)

    DistillationLoss = -Σ teacher_soft × log(student_soft)

    The student learns to:
    1. Get the right answer (hard labels)
    2. Match the teacher's reasoning (soft labels)

    α=1: Pure supervised learning
    α=0: Pure imitation learning
    α=0.5: Balanced (most common)
    """)

distillation_loss()
```

---

## Types of Distillation

### 1. Response-Based Distillation

Learn from the teacher's final outputs (most common).

```python

def response_distillation():
    """
    Distilling from final predictions
    """
    print("Response-Based Distillation")
    "=" * 60

    print("""
    Student learns to match teacher's output probabilities.

    ┌─────────┐     ┌─────────┐
    │ Teacher │     │ Student │
    └────┬────┘     └────┬────┘
         ↓               ↓
    [0.7,0.2,0.1]    [0.6,0.3,0.1]
         ↓               ↓
    └──────┬──────┘
           ↓
    Minimize difference!

    Simple and effective.
    Used in: DistilBERT, TinyBERT
    """)

response_distillation()
```

### 2. Feature-Based Distillation

Learn from intermediate representations.

```python

def feature_distillation():
    """
    Distilling from hidden layers
    """
    print("Feature-Based Distillation")
    "=" * 60

    print("""
    Student learns to match teacher's hidden states.

    Teacher:                     Student:
    Layer 12 ──► [0.8,0.1,...]    Layer 6 ──► [0.7,0.2,...]
         ↓                               ↓
    Layer 8 ──► [0.5,0.3,...]    Layer 4 ──► [0.4,0.4,...]
         ↓                               ↓
    Layer 4 ──► [0.2,0.6,...]    Layer 2 ──► [0.3,0.5,...]

    Student's layers are "aligned" with teacher's layers.

    More complex but can capture more nuanced knowledge.
    Used in: MobileBERT, MiniLM
    """)

feature_distillation()
```

### 3. Relation-Based Distillation

Learn relationships between different parts of the input.

```python

def relation_distillation():
    """
    Distilling relationships
    """
    print("Relation-Based Distillation")
    "=" * 60

    print("""
    Student learns how different parts of the input relate.

    For sentence: "The cat sat on the mat"

    Teacher attention patterns:
    cat ←→ sat : 0.8
    cat ←→ mat : 0.3
    sat ←→ mat : 0.6

    Student learns to match these relationship matrices.

    Captures structural knowledge about the data.
    Used in: Distilling knowledge via relational learning
    """)

relation_distillation()
```

---

## Famous Distilled Models

```python

def distilled_models():
    """
    Popular distilled models and their performance
    """
    print("Famous Distilled Models")
    "=" * 60

    models = [
        {
            "name": "DistilBERT",
            "teacher": "BERT-base",
            "size": "40% smaller",
            "speed": "60% faster",
            "performance": "95% of BERT"
        },
        {
            "name": "TinyBERT",
            "teacher": "BERT-base",
            "size": "7.5x smaller",
            "speed": "9.4x faster",
            "performance": "96% of BERT"
        },
        {
            "name": "MobileBERT",
            "teacher": "BERT-large",
            "size": "4.3x smaller",
            "speed": "5.5x faster",
            "performance": "≈ BERT-base"
        },
        {
            "name": "MiniLM",
            "teacher": "BERT-large",
            "size": "similar to BERT-base",
            "speed": "2x faster",
            "performance": "Better than DistilBERT"
        },
        {
            "name": "Phi-3 (mini)",
            "teacher": "GPT-4 (via synthetic data)",
            "size": "3.8B params",
            "speed": "Runs on phones",
            "performance": "Amazing for its size"
        }
    ]

    for m in models:
        print(f"\n  • {m['name']}:")
        print(f"    Size: {m['size']}, Speed: {m['speed']}")
        print(f"    Performance: {m['performance']}")

distilled_models()
```

---

## Implementing Distillation

### Simple Distillation Example

```python

import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_training():
    """
    Simplified distillation training loop
    """
    print("Distillation Training Loop")
    "=" * 60

    print("""
    # Assume we have teacher_model (frozen) and student_model
    # and a dataset with inputs

    def distillation_step(inputs, labels, teacher_model, student_model):

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # Student forward pass
        student_logits = student_model(inputs)
        student_probs = F.softmax(student_logits / temperature, dim=-1)

        # Distillation loss (match teacher soft labels)
        distill_loss = F.kl_div(
            student_probs.log(),
            teacher_probs,
            reduction='batchmean'
        )

        # Student loss (match true labels)
        student_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        loss = alpha * student_loss + (1-alpha) * distill_loss * (temperature ** 2)

        return loss
    """)

    print("\nKey hyperparameters:")
    print("  • temperature: Usually 2-5, controls softness")
    print("  • alpha: Usually 0.5, balances the losses")

distillation_training()
```

---

## Distillation vs Other Compression Methods

| Method       | Size Reduction | Speedup | Quality | Training Required      |
| ------------ | -------------- | ------- | ------- | ---------------------- |
| Distillation | 40-90%         | 2-10x   | 95-99%  | Yes (student training) |
| Quantization | 50-87%         | 2-4x    | 97-99%  | No (post-training)     |
| Pruning      | 30-60%         | 1.5-3x  | 95-98%  | Sometimes              |
| Combined     | 90%+           | 10x+    | 90-95%  | Yes                    |

### When to Use Each

```python

def when_to_use():
    """
    Choosing the right compression method
    """
    print("Distillation vs Other Methods")
    "=" * 60

    scenarios = [
        ("Need brand new smaller model", "Distillation", "Creates efficient architecture from scratch"),
        ("Already have trained model", "Quantization", "Quick compression without retraining"),
        ("Model too big for hardware", "Pruning", "Remove specific components"),
        ("Maximum compression needed", "All three", "Distill → Prune → Quantize"),
        ("Phone deployment", "Distillation + Quantization", "Get tiny, fast models")
    ]

    for scenario, method, reason in scenarios:
        print(f"\n  • {scenario}:")
        print(f"    → Use {method}")
        print(f"    → Why: {reason}")

when_to_use()
```

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Distillation is more than compression—it's knowledge transfer:║
║                                                              ║
║  🧠 CAPTURES REASONING: Student learns not just answers,    ║
║     but the teacher's thinking process.                       ║
║                                                              ║
║  📱 EDGE AI: Enables state-of-the-art models on phones,     ║
║     browsers, and IoT devices. AI anywhere.                  ║
║                                                              ║
║  🚀 SPEED: 5-10x faster inference means real-time           ║
║     applications become possible.                            ║
║                                                              ║
║  💰 COST: Smaller models cost 10x less to serve.           ║
║     At scale, this changes business models.                  ║
║                                                              ║
║  🔬 RESEARCH: Distillation reveals what knowledge           ║
║     is essential vs. redundant in large models.              ║
║                                                              ║
║  🌟 DEMOCRATIZATION: Not everyone has GPU clusters.         ║
║     Distilled models make AI accessible to all.              ║
║                                                              ║
║  Distillation is how we take the magic of GPT-4 and         ║
║  put it in your pocket.                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Distillation trains a smaller "student" model to mimic a larger "teacher"—using soft labels that capture the teacher's reasoning, not just final answers

• Different types include response-based (outputs), feature-based (hidden layers), and relation-based (relationships)—each capturing different aspects of the teacher's knowledge

• Famous examples like DistilBERT achieve 95% of BERT's performance with 40% fewer parameters—proving that much of a large model's knowledge can be compressed into a smaller architecture

---

## Mental Hook

> "Distillation is like a master chef training apprentices—they don't need 40 years of experience; they learn from the master's examples, corrections, and techniques, and soon they're cooking almost as well, in half the time, in multiple restaurants."
