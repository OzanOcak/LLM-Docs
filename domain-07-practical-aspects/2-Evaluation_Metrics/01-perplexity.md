# Perplexity

## The Multiple-Choice Exam Analogy

Imagine you're taking a multiple-choice test where you have to predict the next word in a sentence. A confused student might think any of 10 words could come next (high perplexity). A confident student might narrow it down to 2 possibilities (medium perplexity). A genius knows exactly which word comes next (low perplexity). That's perplexity: a measure of how "surprised" or "confused" your language model is when trying to predict the next word.

In LLMs, perplexity is the most fundamental intrinsic evaluation metric. It tells you how well your model predicts a sample of text. Lower perplexity means the model is more confident and accurate in its predictions. It's like a built-in report card that works without needing human judgments or external benchmarks.

---

## What Is Perplexity?

### The Core Idea

Perplexity measures how well a probability model predicts a sample. In language modeling, it's the inverse probability of the test set, normalized by the number of words.

```python

def perplexity_intro():
    """
    The basic concept of perplexity
    """
    print("Perplexity: Measuring Model Confusion")
    print("=" * 60)

    print("""
    Intuition:

    Perplexity = "How many equally likely words is the model choosing from?"

    • Perplexity = 1: Model is certain (always picks the right word)
    • Perplexity = 10: Model is as confused as if it had to choose
                      from 10 equally likely options
    • Perplexity = 100: Model is very uncertain

    Lower perplexity = BETTER model!
    """)

perplexity_intro()
```

### The Math in Plain English

```python

def perplexity_math():
    """
    The mathematics of perplexity
    """
    print("The Math of Perplexity")
    print("=" * 60)

    print("""
    Perplexity = 2 ^ ( - (1/N) * Σ log₂ P(wordᵢ | context) )

    Step-by-step:

    1. For each word, get its probability according to model
    2. Take the log (average of negative logs)
    3. Exponentiate back

    Example with 3 words, each probability 0.5:

    log₂(0.5) = -1
    Average of -logs = 1
    2¹ = 2  (perplexity = 2)

    This means: Model is as confused as if it had to choose
                between 2 equally likely words.
    """)

perplexity_math()
```

---

## Perplexity Intuition with Examples

### Low Perplexity (Confident Model)

```python

def low_perplexity():
    """
    Example of low perplexity (good model)
    """
    print("Low Perplexity: Confident Model")
    print("=" * 60)

    sentence = "The cat sat on the mat"

    # Simulated high-confidence predictions
    predictions = [
        {"word": "The", "prob": 0.95},
        {"word": "cat", "prob": 0.90},
        {"word": "sat", "prob": 0.85},
        {"word": "on", "prob": 0.92},
        {"word": "the", "prob": 0.94},
        {"word": "mat", "prob": 0.88}
    ]

    import math

    log_probs = [math.log2(p["prob"]) for p in predictions]
    avg_neg_log = -sum(log_probs) / len(log_probs)
    perplexity = 2 ** avg_neg_log

    print(f"Sentence: '{sentence}'")
    print("\nWord probabilities:")
    for p in predictions:
        print(f"  '{p['word']}': {p['prob']:.2f}")

    print(f"\nAverage negative log₂: {avg_neg_log:.3f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("→ Model is very confident, perplexity near 1!")

low_perplexity()
```

### High Perplexity (Uncertain Model)

```python

def high_perplexity():
    """
    Example of high perplexity (poor model)
    """
    print("High Perplexity: Confused Model")
    print("=" * 60)

    sentence = "The cat sat on the mat"

    # Simulated uncertain predictions
    predictions = [
        {"word": "The", "prob": 0.25},
        {"word": "cat", "prob": 0.20},
        {"word": "sat", "prob": 0.15},
        {"word": "on", "prob": 0.18},
        {"word": "the", "prob": 0.12},
        {"word": "mat", "prob": 0.10}
    ]

    import math

    log_probs = [math.log2(p["prob"]) for p in predictions]
    avg_neg_log = -sum(log_probs) / len(log_probs)
    perplexity = 2 ** avg_neg_log

    print(f"Sentence: '{sentence}'")
    print("\nWord probabilities:")
    for p in predictions:
        print(f"  '{p['word']}': {p['prob']:.2f}")

    print(f"\nAverage negative log₂: {avg_neg_log:.3f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("→ Model is uncertain, perplexity high (choosing among many)")

high_perplexity()
```

---

## Perplexity Across Model Sizes

```python

def model_comparison():
    """
    Typical perplexity values for different models
    """
    print("Perplexity Across Model Sizes")
    print("=" * 60)

    models = [
        {"name": "Random guessing", "ppl": 50000, "desc": "Uniform over 50k vocab"},
        {"name": "N-gram model", "ppl": 200-500, "desc": "Simple statistical"},
        {"name": "Small LSTM", "ppl": 80-120, "desc": "Early neural models"},
        {"name": "GPT-2 small", "ppl": 35-40, "desc": "124M parameters"},
        {"name": "GPT-2 large", "ppl": 30-35, "desc": "774M parameters"},
        {"name": "GPT-3", "ppl": 15-20, "desc": "175B parameters"},
        {"name": "Human", "ppl": ~10, "desc": "Estimated human performance"}
    ]

    for model in models:
        if isinstance(model["ppl"], tuple):
            ppl_str = f"{model['ppl'][0]}-{model['ppl'][1]}"
        else:
            ppl_str = str(model["ppl"])
        print(f"  • {model['name']:15} → {ppl_str:8} perplexity")
        print(f"    {model['desc']}")

model_comparison()
```

---

## Computing Perplexity in Code

```python

import numpy as np
import math

def compute_perplexity_demo():
    """
    Complete perplexity calculation example
    """
    print("Computing Perplexity in Python")
    print("=" * 60)

    # Simulated language model predictions
    # For a sentence with 5 words, we have probability distributions
    vocab_size = 1000

    # Good model: high probabilities for correct words
    good_probs = [0.8, 0.7, 0.9, 0.75, 0.85]

    # Bad model: low probabilities
    bad_probs = [0.2, 0.15, 0.1, 0.25, 0.2]

    def perplexity(probabilities):
        """Calculate perplexity from word probabilities"""
        log_probs = [math.log(p) for p in probabilities]
        avg_neg_log = -sum(log_probs) / len(log_probs)
        return math.exp(avg_neg_log)

    good_ppl = perplexity(good_probs)
    bad_ppl = perplexity(bad_probs)

    print("Good model (confident):")
    print(f"  Probabilities: {[f'{p:.2f}' for p in good_probs]}")
    print(f"  Perplexity: {good_ppl:.2f}")

    print("\nBad model (uncertain):")
    print(f"  Probabilities: {[f'{p:.2f}' for p in bad_probs]}")
    print(f"  Perplexity: {bad_ppl:.2f}")

    # Convert to "effective vocabulary size"
    print(f"\nInterpretation:")
    print(f"  Good model is as confused as choosing from {good_ppl:.1f} words")
    print(f"  Bad model is as confused as choosing from {bad_ppl:.1f} words")

compute_perplexity_demo()
```

---

## Perplexity in Practice

### Training Curves

```python

def training_curves():
    """
    Perplexity during training
    """
    print("Perplexity Training Curves")
    print("=" * 60)

    epochs = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    train_ppl = [500, 200, 100, 60, 40, 25, 18, 15, 14, 13.5]
    val_ppl = [550, 250, 130, 80, 55, 35, 28, 26, 27, 29]

    print("Epoch | Train PPL | Val PPL | Status")
    print("-" * 45)

    for i, epoch in enumerate(epochs):
        status = ""
        if i > 0 and val_ppl[i] > val_ppl[i-1]:
            status = "⚠️ Overfitting (val increasing)"
        print(f"{epoch:3d}   |   {train_ppl[i]:5.1f}   |  {val_ppl[i]:5.1f}   | {status}")

    print("\nBest model at epoch 30 (lowest val, before overfitting)")

training_curves()
```

### Perplexity Across Languages

```python

def multilingual_ppl():
    """
    Perplexity varies by language
    """
    print("Perplexity Across Languages")
    print("=" * 60)

    print("""
    The same model can have different perplexities
    for different languages:

    Language   | Tokenization | Perplexity (example)
    ────────────────────────────────────────────────
    English    | 1 token/word  | 20
    Spanish    | 1 token/word  | 22
    German     | 1-2 tokens    | 28 (long compounds)
    Chinese    | 1-2 chars     | 35 (different script)
    Thai       | 3-5 chars     | 45 (no spaces)

    This doesn't mean the model is worse at Thai—
    it's just that tokenization differs!
    """)

multilingual_ppl()
```

---

## Limitations of Perplexity

### What Perplexity Doesn't Measure

```python

def limitations():
    """
    Limitations of perplexity
    """
    print("What Perplexity Misses")
    print("=" * 60)

    limitations = [
        "COHERENCE: A model can have low perplexity but generate nonsense",
        "FACTUAL ACCURACY: Perplexity doesn't measure truth",
        "CREATIVITY: Repetitive text can have low perplexity",
        "SAFETY: Doesn't detect harmful content",
        "TASK PERFORMANCE: Low perplexity ≠ good at QA, translation"
    ]

    print("Perplexity is useful but incomplete:")
    for lim in limitations:
        print(f"  • {lim}")

    print("\nExample: 'The the the the' has low perplexity but is garbage!")
    print("Always combine with other metrics.")

limitations()
```

### The Repetition Trap

```python

def repetition_trap():
    """
    Why repetition fools perplexity
    """
    print("The Repetition Trap")
    print("=" * 60)

    repetitive = "The cat sat on the mat. " * 10
    creative = "The cat lounged on the cushion while watching birds outside."

    print("Repetitive text: predictable → low perplexity")
    print("Creative text:   varied → higher perplexity")
    print("\nBut which is better? Creative, of course!")
    print("Perplexity alone would prefer the repetitive text.")

repetition_trap()
```

---

## Perplexity Cheat Sheet

| Perplexity | Interpretation         | Example Model          |
| ---------- | ---------------------- | ---------------------- |
| 1          | Perfect prediction     | Impossible in practice |
| 10         | Near-human performance | Best LLMs              |
| 20-30      | Good language model    | GPT-2, BERT            |
| 50-100     | Decent model           | Small LSTMs            |
| 100-500    | Poor model             | N-gram models          |
| >1000      | Almost random          | Untrained model        |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Perplexity is your model's built-in report card:           ║
║                                                              ║
║  📊 FAST FEEDBACK: You can compute perplexity on any        ║
║     text without human judgments. Iterate quickly!           ║
║                                                              ║
║  🔍 DEBUGGING TOOL: Sudden increase in perplexity           ║
║     often reveals data issues, bugs, or distribution shift  ║
║                                                              ║
║  📈 PROGRESS TRACKING: Watch perplexity decrease            ║
║     during training to know if your model is learning       ║
║                                                              ║
║  ⚖️ MODEL COMPARISON: Compare different architectures       ║
║     fairly on the same test set using perplexity            ║
║                                                              ║
║  🎯 EARLY STOPPING: When validation perplexity stops        ║
║     improving, it's time to stop training                   ║
║                                                              ║
║  Perplexity isn't perfect, but it's the fastest way         ║
║  to know if your model is heading in the right direction.   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Perplexity measures how "surprised" a model is by test data—lower is better, with 1 being perfect prediction and higher values indicating uncertainty

• It's calculated as the exponentiated average negative log probability—intuitively, it's the number of equally likely words the model is choosing between

• While useful for tracking training and comparing models, perplexity has limitations—it doesn't measure coherence, factuality, or task performance, so it should be combined with other metrics

---

## Mental Hook

> "Perplexity is like a lie detector for your language model—when it's confident, it barely flinches (low perplexity). When it's guessing, it sweats and stammers (high perplexity). The better the model, the calmer it stays."
