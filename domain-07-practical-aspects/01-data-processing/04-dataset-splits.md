# Dataset Splits

## The Exam Preparation Analogy

Imagine you're studying for a final exam. You have a textbook with practice problems. You'd never take the actual final exam as practice—that would ruin it as a measure of your learning. Instead, you split the problems: some for practice while studying, some for practice tests to check your progress, and the actual final exam is kept completely separate until the end. That's dataset splits: dividing your data into training, validation, and test sets to ensure you can honestly measure how well your model has learned.

In machine learning, dataset splits are the foundation of honest evaluation. The training set is for learning, the validation set is for tuning, and the test set is for final evaluation. Mixing these up is like practicing on the final exam—you'll think you're ready, but you've just memorized the answers.

---

## The Three Sets: Training, Validation, Test

### The Core Idea

```python

def splits_intro():
    """
    The three types of dataset splits
    """
    print("The Three Dataset Splits")
    print("=" * 60)

    print("""
    TRAINING SET (80-90%)
    • What the model learns from
    • Sees this data many times
    • Updates weights based on this

    VALIDATION SET (5-10%)
    • Used to tune hyperparameters
    • Check progress during training
    • Early stopping, model selection

    TEST SET (5-10%)
    • Final evaluation only
    • Used exactly ONCE
    • Measures real-world performance
    """)

splits_intro()
```

### Visualizing the Split

```text

All Data (100%)
┌─────────────────────────────────────────────────┐
│                                                 │
├───────────┬───────────┬─────────────────────────┤
│  TRAIN    │  VALID    │         TEST            │
│  (80%)    │  (10%)    │         (10%)           │
├───────────┴───────────┴─────────────────────────┤
│                                                 │
│  • Model sees      • Tune         • Final       │
│    repeatedly       hyperparams     evaluation  │
│                                                 │
└─────────────────────────────────────────────────┘

NEVER touch test set until the very end!
```

---

## Why We Need Three Sets

### The Problem with Only Training/Test

```python

def two_set_problem():
    """
    Why training/test isn't enough
    """
    print("Why Two Sets Aren't Enough")
    print("=" * 60)

    print("""
    If you only have training and test:

    1. Train model
    2. Test on test set
    3. See poor performance
    4. Tweak hyperparameters
    5. Test again on test set
    6. Repeat 100 times

    PROBLEM: You've now effectively trained on the test set!
    Your hyperparameters are optimized for that specific test set,
    and your final performance estimate is overly optimistic.

    SOLUTION: Validation set for tuning, test set only for final check.
    """)

two_set_problem()
```

### The Validation Set Protects the Test Set

```python

def validation_protects():
    """
    How validation set protects test set
    """
    print("Validation Set: The Test Set's Bodyguard")
    print("=" * 60)

    print("""
    Correct workflow:

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   TRAIN     │────>│   VALID     │────>│    TEST     │
    │   set       │     │   set       │     │    set      │
    └─────────────┘     └─────────────┘     └─────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    Learn weights      Tune hyperparams      Final report
    many times         many times             exactly once

    Test set remains PRISTINE until the end.
    """)

validation_protects()
```

---

## Common Split Ratios

### By Dataset Size

```python

def split_ratios():
    """
    Common split ratios for different dataset sizes
    """
    print("Split Ratios by Dataset Size")
    print("=" * 60)

    scenarios = [
        ("Small (<10K)", "70/15/15", "Need enough validation/test examples"),
        ("Medium (10K-100K)", "80/10/10", "Standard split"),
        ("Large (100K-1M)", "90/5/5", "Can afford smaller validation/test"),
        ("Very Large (>1M)", "98/1/1", "1% of 1M is still 10K examples"),
        ("Massive (LLM scale)", "99.9/0.05/0.05", "0.05% of 1B is 500K examples")
    ]

    for size, ratio, reason in scenarios:
        print(f"\n  • {size}: {ratio}")
        print(f"    → {reason}")

split_ratios()
```

### Why Ratios Change with Scale

```python

def ratio_math():
    """
    Why larger datasets need smaller validation percentages
    """
    print("The Math Behind Split Ratios")
    print("=" * 60)

    sizes = [1000, 10000, 100000, 1000000, 10000000]
    val_percent = 0.1  # 10%

    print("Size | Validation examples | Enough?")
    print("-" * 40)

    for size in sizes:
        val_size = int(size * 0.1)
        enough = "✓" if val_size >= 1000 else "✗ (too small)"
        print(f"{size:7,d} | {val_size:6,d} | {enough}")

    print("\nFor small datasets, 10% might be too few validation examples.")
    print("For large datasets, 1% gives plenty of validation data.")

ratio_math()
```

---

## Types of Splits

### Random Split

The simplest: randomly shuffle and split.

```python

def random_split():
    """
    Random splitting
    """
    print("Method 1: Random Split")
    print("=" * 60)

    import random

    # Simulate dataset
    data = list(range(100))

    # Shuffle
    random.seed(42)
    random.shuffle(data)

    # Split 80/10/10
    train = data[:80]
    val = data[80:90]
    test = data[90:]

    print(f"Total: {len(data)} examples")
    print(f"Train: {len(train)} examples (indices {min(train)}-{max(train)})")
    print(f"Val:   {len(val)} examples (indices {min(val)}-{max(val)})")
    print(f"Test:  {len(test)} examples (indices {min(test)}-{max(test)})")

    print("\n✅ Simple, works for most cases")
    print("❌ May not preserve class distribution")

random_split()
```

### Stratified Split

Preserves class proportions in each split.

```python

def stratified_split():
    """
    Stratified split for classification
    """
    print("Method 2: Stratified Split")
    print("=" * 60)

    from collections import Counter

    # Simulate imbalanced dataset
    data = (
        ["cat"] * 800 +
        ["dog"] * 150 +
        ["rabbit"] * 50
    )

    print("Original distribution:")
    for label, count in Counter(data).items():
        print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")

    print("\nStratified split ensures each split has same proportions:")
    print("  Train (80%): cat 640, dog 120, rabbit 40")
    print("  Val (10%):   cat 80, dog 15, rabbit 5")
    print("  Test (10%):  cat 80, dog 15, rabbit 5")

    print("\n✅ Preserves class balance")
    print("❌ Only works for classification")

stratified_split()
```

### Time-Based Split

For time series data, don't shuffle—split by time.

```python

def time_split():
    """
    Time-based split for temporal data
    """
    print("Method 3: Time-Based Split")
    "=" * 60

    print("""
    For time series data (stock prices, news, etc.):

    ❌ WRONG: Random shuffle
    • Model would see future data during training
    • Would appear to predict perfectly (cheating!)

    ✅ CORRECT: Split by time
    ─────────────────────────────────
    Train: 2020-2022
    Val:   Jan-June 2023
    Test:  July-Dec 2023

    This simulates real-world prediction:
    training on past, predicting future.
    """)

time_split()
```

---

## Code: Complete Split Example

```python

import numpy as np
from sklearn.model_selection import train_test_split

def split_demo():
    """
    Complete dataset split example
    """
    print("Complete Dataset Split Demo")
    print("=" * 60)

    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)  # 10 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # binary target

    print(f"Dataset size: {n_samples} samples")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {np.bincount(y)}")

    # First split: train+val vs test (90/10)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Second split: train vs val (89/11 of original, i.e., 80/10 of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
    )

    print("\nFinal splits:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")

    # Check class distribution
    print("\nClass distribution in splits:")
    print(f"  Train: {np.bincount(y_train)/len(y_train)*100:.1f}% class 1")
    print(f"  Val:   {np.bincount(y_val)/len(y_val)*100:.1f}% class 1")
    print(f"  Test:  {np.bincount(y_test)/len(y_test)*100:.1f}% class 1")

split_demo()
```

---

## Special Considerations for LLMs

### Document-Level Splits

```python

def document_splits():
    """
    Splitting at document level for LLMs
    """
    print("Document-Level Splits for LLMs")
    print("=" * 60)

    print("""
    For LLM training, we must split at DOCUMENT level,
    not sentence or paragraph level.

    ❌ WRONG: Split sentences randomly
    • Same document could be in train and test
    • Model has effectively seen test content
    • Cheating!

    ✅ CORRECT: Split documents
    • Entire documents go to one split only
    • No leakage between splits

    Example with 1000 documents:
    Train: 800 complete documents
    Val:   100 complete documents
    Test:  100 complete documents
    """)

document_splits()
```

### The "Test Set Contamination" Problem

```python

def contamination():
    """
    The problem of test set contamination
    """
    print("Test Set Contamination: A Major Issue")
    print("=" * 60)

    print("""
    PROBLEM: Many popular LLMs were trained on data
    that includes common test sets (like MMLU, HumanEval).

    Example:
    • GPT-3's training data included MMLU examples
    • When tested on MMLU, it had already seen many questions
    • Performance appeared artificially high!

    SOLUTION:
    • Create fresh test sets
    • Use canary strings in web data
    • Test on newly collected data
    • Continuous monitoring for contamination
    """)

contamination()
```

---

## Common Pitfalls

| Pitfall            | What Happens                     | How to Avoid                    |
| ------------------ | -------------------------------- | ------------------------------- |
| Data leakage       | Test data influences training    | Split before any preprocessing  |
| Temporal leakage   | Future in training, past in test | Time-based split                |
| Duplicate removal  | Same doc in train and test       | Deduplicate across splits       |
| Overfitting to val | Tuning too much on val           | Limit iterations, use cross-val |
| Test set reuse     | Multiple evaluations             | Keep test set locked            |
| Small test set     | High variance results            | Ensure sufficient size          |

---

## Code: Detecting Data Leakage

```python

def leakage_detection():
    """
    Simple leakage detection
    """
    print("Detecting Data Leakage")
    print("=" * 60)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Simulated train and test documents
    train_docs = [
        "The cat sat on the mat",
        "Dogs love to play fetch",
        "Cats are independent pets"
    ]

    test_docs = [
        "The cat sat on the mat",  # Exact duplicate!
        "Cats are independent animals",
        "Car maintenance tips"
    ]

    # Check for duplicates
    print("Checking for train/test overlap:")
    for i, test_doc in enumerate(test_docs):
        for j, train_doc in enumerate(train_docs):
            if test_doc == train_doc:
                print(f"  ⚠️ Test doc {i} duplicates train doc {j}")

    # Check for near-duplicates
    vectorizer = TfidfVectorizer().fit(train_docs + test_docs)
    train_vecs = vectorizer.transform(train_docs)
    test_vecs = vectorizer.transform(test_docs)

    sim_matrix = cosine_similarity(test_vecs, train_vecs)

    print("\nSimilarity matrix (test vs train):")
    for i, row in enumerate(sim_matrix):
        print(f"  Test {i}: {[f'{s:.2f}' for s in row]}")

leakage_detection()
```

---

## Dataset Splits Cheat Sheet

| Split      | Purpose          | How Often Used | Can We Tune On It?     |
| ---------- | ---------------- | -------------- | ---------------------- |
| Training   | Learn weights    | Many epochs    | YES (this is learning) |
| Validation | Tune hyperparams | Many times     | YES (but limited)      |
| Test       | Final evaluation | Once           | NO (never!)            |

| Dataset Size | Recommended Split | Validation Size | Test Size |
| ------------ | ----------------- | --------------- | --------- |
| <10K         | 70/15/15          | ≥1,500          | ≥1,500    |
| 10K-100K     | 80/10/10          | ≥1,000          | ≥1,000    |
| 100K-1M      | 90/5/5            | ≥5,000          | ≥5,000    |
| >1M          | 98/1/1            | ≥10,000         | ≥10,000   |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Dataset splits are the difference between knowing your     ║
║  model works and THINKING it works:                         ║
║                                                              ║
║  🎯 HONEST EVALUATION: Without a held-out test set,         ║
║     you can't trust your performance numbers                 ║
║                                                              ║
║  🔧 PROPER TUNING: Validation set lets you optimize         ║
║     without contaminating your final test                    ║
║                                                              ║
║  🚫 PREVENTS CHEATING: Stops you from (even                 ║
║     unintentionally) training on test data                   ║
║                                                              ║
║  📊 RELIABLE COMPARISONS: Same test set across              ║
║     models = fair comparison                                 ║
║                                                              ║
║  🔬 REPRODUCIBILITY: Fixed splits mean others               ║
║     can verify your results                                  ║
║                                                              ║
║  In research and production, incorrect splits               ║
║  have led to embarrassing retractions and                    ║
║  models that fail in the real world. Don't be that person.  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Training set is for learning, validation set is for tuning, test set is for final evaluation—like using practice problems, practice tests, and a final exam, each serving a distinct purpose

• Split ratios depend on dataset size—small datasets need larger percentages for validation/test, large datasets can use tiny percentages and still have enough examples

• Never let test data influence your model—no peeking, no tuning, no repeated evaluation. The test set is used exactly once, at the very end, to report final performance

---

## Mental Hook

> "Dataset splits are like the honor system in an exam—if you peek at the answers during practice, you'll never know if you really understand the material. The test set is the final exam you can't cheat on, and it's the only one that matters."
