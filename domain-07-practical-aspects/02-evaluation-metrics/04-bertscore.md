# BERTScore

## The Smart Grader Analogy

Imagine a teacher grading essays who doesn't just check for exact word matches but understands meaning. When a student writes "The feline rested on the rug" instead of "The cat sat on the mat," the teacher knows they captured the meaning even though every word is different. That's BERTScore: using BERT embeddings to measure semantic similarity between generated text and references, capturing meaning rather than just surface-level word matches.

In LLM evaluation, BERTScore represents a paradigm shift. Instead of counting matching words (like BLEU) or n-grams (like ROUGE), it uses deep language understanding to judge whether two texts mean the same thing. It correlates much better with human judgment, especially for tasks where there are many ways to say the same thing.

---

## What Is BERTScore?

### The Core Idea

BERTScore computes similarity by comparing the embeddings of words in the candidate and reference texts, using BERT's contextual understanding.

```text

Reference: "The cat sat on the mat"
              ↓    ↓   ↓   ↓   ↓
          BERT embeddings (contextual)
              ↓    ↓   ↓   ↓   ↓
Candidate: "A feline rested on the rug"
              ↓    ↓   ↓   ↓   ↓
          BERT embeddings (contextual)
              ↓    ↓   ↓   ↓   ↓
          Pairwise cosine similarity
              ↓    ↓   ↓   ↓   ↓
          Aggregate scores → BERTScore

Even with different words, similar meaning = high score!
```

```python

def bertscore_intro():
    """
    The basic concept of BERTScore
    """
    print("BERTScore: Measuring Meaning, Not Words")
    print("=" * 60)

    print("""
    Why BERTScore is different:

    • BLEU/ROUGE: Exact word matches only
      "cat" vs "feline" → match? NO

    • BERTScore: Semantic similarity
      "cat" vs "feline" → very close in embedding space → match! ✓

    • Contextual understanding
      "bank" (river) vs "bank" (money) → different meanings → distinguished!

    • Handles paraphrases, synonyms, rephrasing
    """)

bertscore_intro()
```

---

## How BERTScore Works Step by Step

### Step 1: Get Contextual Embeddings

```python

def step1_embeddings():
    """
    Getting BERT embeddings for each word
    """
    print("Step 1: Contextual Word Embeddings")
    print("=" * 60)

    reference = "The cat sat on the mat"
    candidate = "A feline rested on the rug"

    print(f"Reference: '{reference}'")
    print(f"Candidate: '{candidate}'")

    print("\nBERT processes each word in context:")
    print("  'cat' in reference → embedding_cat_ref [0.2, -0.5, 0.8, ...]")
    print("  'feline' in candidate → embedding_feline_cand [0.3, -0.4, 0.7, ...]")
    print("\nThese embeddings capture meaning, not just identity.")
    print("'cat' and 'feline' will be close in embedding space!")

step1_embeddings()
```

### Step 2: Compute Pairwise Similarities

```python

def step2_similarities():
    """
    Computing cosine similarity between all word pairs
    """
    print("Step 2: Pairwise Cosine Similarities")
    print("=" * 60)

    import numpy as np

    # Simulated embeddings (simplified to 2D for visualization)
    ref_words = ["The", "cat", "sat", "on", "the", "mat"]
    cand_words = ["A", "feline", "rested", "on", "the", "rug"]

    # Simulated similarity matrix (simplified)
    similarity_matrix = np.array([
        [0.9, 0.1, 0.1, 0.1, 0.8, 0.1],  # "The" vs each candidate word
        [0.1, 0.9, 0.2, 0.1, 0.1, 0.2],  # "cat" vs each candidate word
        [0.1, 0.2, 0.8, 0.3, 0.1, 0.1],  # "sat" vs each candidate word
        [0.1, 0.1, 0.2, 0.9, 0.2, 0.1],  # "on" vs each candidate word
        [0.8, 0.1, 0.1, 0.2, 0.9, 0.1],  # "the" vs each candidate word
        [0.1, 0.3, 0.1, 0.1, 0.1, 0.9],  # "mat" vs each candidate word
    ])

    print("Similarity matrix (reference rows vs candidate columns):")
    print("          A   feline  rested  on  the  rug")
    ref_words_display = ["The", "cat", "sat", "on", "the", "mat"]
    for i, ref_word in enumerate(ref_words_display):
        row = [f"{sim:.1f}" for sim in similarity_matrix[i]]
        print(f"{ref_word:6} {row}")

step2_similarities()
```

### Step 3: Precision, Recall, and F1

```python

def step3_scores():
    """
    Computing precision, recall, and F1
    """
    print("Step 3: Precision, Recall, and F1")
    print("=" * 60)

    import numpy as np

    # Simulated similarity matrix (each ref word vs each cand word)
    sim_matrix = np.array([
        [0.9, 0.1, 0.1, 0.1, 0.8, 0.1],  # "The"
        [0.1, 0.9, 0.2, 0.1, 0.1, 0.2],  # "cat"
        [0.1, 0.2, 0.8, 0.3, 0.1, 0.1],  # "sat"
        [0.1, 0.1, 0.2, 0.9, 0.2, 0.1],  # "on"
        [0.8, 0.1, 0.1, 0.2, 0.9, 0.1],  # "the"
        [0.1, 0.3, 0.1, 0.1, 0.1, 0.9],  # "mat"
    ])

    # Recall: for each reference word, best matching candidate
    recall_scores = np.max(sim_matrix, axis=1)
    recall = np.mean(recall_scores)

    # Precision: for each candidate word, best matching reference
    precision_scores = np.max(sim_matrix, axis=0)
    precision = np.mean(precision_scores)

    # F1: harmonic mean
    f1 = 2 * (precision * recall) / (precision + recall)

    print("For each reference word, best match in candidate:")
    ref_words = ["The", "cat", "sat", "on", "the", "mat"]
    for word, score in zip(ref_words, recall_scores):
        print(f"  '{word}': best match similarity = {score:.2f}")
    print(f"\nRECALL = average = {recall:.3f}")

    print("\nFor each candidate word, best match in reference:")
    cand_words = ["A", "feline", "rested", "on", "the", "rug"]
    for word, score in zip(cand_words, precision_scores):
        print(f"  '{word}': best match similarity = {score:.2f}")
    print(f"\nPRECISION = average = {precision:.3f}")

    print(f"\nF1 SCORE = {f1:.3f}")

step3_scores()
```

---

## Complete BERTScore Example

```python

import torch
import numpy as np

def bertscore_demo():
    """
    Complete BERTScore example (conceptual)
    """
    print("Complete BERTScore Example")
    print("=" * 60)

    # Note: This is a conceptual demo. Real BERTScore uses actual BERT models.

    test_cases = [
        {
            "ref": "The cat sat on the mat",
            "cand": "A feline rested on the rug",
            "desc": "Paraphrase (different words, same meaning)",
            "expected": "HIGH"
        },
        {
            "ref": "The cat sat on the mat",
            "cand": "The dog played in the park",
            "desc": "Different topic",
            "expected": "LOW"
        },
        {
            "ref": "The movie was excellent",
            "cand": "The film was fantastic",
            "desc": "Synonyms",
            "expected": "HIGH"
        },
        {
            "ref": "I love this product",
            "cand": "I hate this product",
            "desc": "Opposite sentiment",
            "expected": "LOW (negative correlation possible)"
        }
    ]

    # Simulated BERTScore results
    results = [0.92, 0.31, 0.89, 0.15]

    for i, case in enumerate(test_cases):
        print(f"\n{'-'*40}")
        print(f"Case {i+1}: {case['desc']}")
        print(f"Reference: {case['ref']}")
        print(f"Candidate: {case['cand']}")
        print(f"BERTScore: {results[i]:.2f} ({case['expected']})")
        print(f"Expected correlation: {case['expected']}")

bertscore_demo()
```

---

## BERTScore vs Traditional Metrics

### Comparison Table

| Aspect                    | BLEU          | ROUGE         | BERTScore           |
| ------------------------- | ------------- | ------------- | ------------------- |
| Matches                   | Exact n-grams | Exact n-grams | Semantic similarity |
| Handles synonyms?         | No            | No            | Yes                 |
| Contextual understanding? | No            | No            | Yes                 |
| Correlation with humans   | Moderate      | Moderate      | High                |
| Computational cost        | Very low      | Very low      | High                |
| Language required         | Any           | Any           | Needs BERT model    |
| Interpretability          | Easy          | Easy          | Harder              |

### Example Comparison

```python

def metric_comparison():
    """
    Comparing metrics on the same sentences
    """
    print("Comparing Metrics on Paraphrase")
    print("=" * 60)

    reference = "The cat sat on the mat"
    candidate = "A feline rested on the rug"

    # Simulated scores
    scores = {
        "BLEU": 0.12,
        "ROUGE-1": 0.25,
        "ROUGE-L": 0.30,
        "BERTScore": 0.92
    }

    print(f"Reference: {reference}")
    print(f"Candidate: {candidate}")
    print("\nScores:")
    for metric, score in scores.items():
        bar = "█" * int(score * 30)
        print(f"  {metric:10}: {score:.2f} {bar}")

    print("\nDespite meaning being almost identical,")
    print("BLEU/ROUGE penalize word choice differences.")
    print("BERTScore correctly captures semantic similarity.")

metric_comparison()
```

---

## Using BERTScore in Python

```python

# Note: Requires installation: pip install bert-score

def bertscore_usage():
    """
    Using the bert-score library
    """
    print("BERTScore with Python")
    print("=" * 60)

    print("""
    from bert_score import score

    # Candidate and reference lists
    candidates = [
        "A feline rested on the rug",
        "The dog played in the park"
    ]
    references = [
        "The cat sat on the mat",
        "The cat sat on the mat"
    ]

    # Compute BERTScore
    P, R, F1 = score(candidates, references, lang='en', verbose=True)

    print(f"Precision: {P.mean():.3f}")
    print(f"Recall:    {R.mean():.3f}")
    print(f"F1:        {F1.mean():.3f}")

    # For first candidate (paraphrase) vs second (unrelated):
    # F1[0] ≈ 0.92 (high), F1[1] ≈ 0.35 (low)
    """)

    print("\nKey parameters:")
    print("  • model_type: 'roberta-large', 'bert-base-uncased', etc.")
    print("  • lang: 'en', 'zh', 'de', etc.")
    print("  • rescale_with_baseline: normalize scores to 0-1 range")

bertscore_usage()
```

---

## Advantages of BERTScore

### What Makes It Better

```python

def advantages():
    """
    Key advantages of BERTScore
    """
    print("Advantages of BERTScore")
    print("=" * 60)

    advantages = [
        "SEMANTIC UNDERSTANDING: Captures meaning, not just word matches",
        "SYNONYM ROBUSTNESS: 'big'/'large' treated as similar",
        "CONTEXTUAL AWARENESS: 'bank' disambiguated by context",
        "PARAPHRASE RECOGNITION: Different structures, same meaning",
        "HIGH HUMAN CORRELATION: Matches human judgment better",
        "MULTILINGUAL: Works for many languages with appropriate models"
    ]

    for adv in advantages:
        print(f"  • {adv}")

advantages()
```

---

## Limitations of BERTScore

```python

def bertscore_limitations():
    """
    Limitations of BERTScore
    """
    print("Limitations of BERTScore")
    print("=" * 60)

    limitations = [
        "COMPUTATIONAL COST: Much slower than n-gram metrics",
        "MODEL DEPENDENT: Results vary by which BERT model used",
        "LANGUAGE SPECIFIC: Needs model for each language",
        "LESS INTERPRETABLE: Harder to explain why score is low",
        "LENGTH BIAS: May favor longer texts",
        "NOT CALIBRATED: Scores aren't directly comparable across tasks"
    ]

    for lim in limitations:
        print(f"  • {lim}")

bertscore_limitations()
```

---

## BERTScore Interpretation

| Score Range | Interpretation           | Example                           |
| ----------- | ------------------------ | --------------------------------- |
| > 0.9       | Nearly identical meaning | Perfect paraphrase                |
| 0.8 - 0.9   | Very similar meaning     | Minor word choice differences     |
| 0.6 - 0.8   | Somewhat similar         | Related but not equivalent        |
| 0.4 - 0.6   | Slightly related         | Shared topic, different specifics |
| < 0.4       | Unrelated                | Different meaning entirely        |

### Typical Scores

```python

def typical_scores():
    """
    Typical BERTScore ranges
    """
    print("Typical BERTScore Ranges")
    print("=" * 60)

    scenarios = [
        ("Identical sentences", "0.99-1.00"),
        ("Human paraphrase", "0.90-0.95"),
        ("GPT-4 summary vs human", "0.85-0.92"),
        ("Different translations", "0.80-0.90"),
        ("Related topics", "0.60-0.70"),
        ("Unrelated texts", "0.30-0.50")
    ]

    for scenario, score in scenarios:
        print(f"  • {scenario:25} → {score}")

typical_scores()
```

---

## BERTScore vs Human Evaluation

```python

def human_correlation():
    """
    How well BERTScore correlates with humans
    """
    print("BERTScore vs Human Judgment")
    print("=" * 60)

    correlations = {
        "BLEU": "0.3-0.4 (moderate)",
        "ROUGE": "0.3-0.5 (moderate)",
        "METEOR": "0.4-0.5 (moderate)",
        "BERTScore": "0.7-0.8 (strong)"
    }

    print("Correlation with human evaluations:")
    for metric, corr in correlations.items():
        print(f"  • {metric:10} → {corr}")

    print("\nBERTScore approaches human-level agreement,")
    print("making it the closest automatic metric to human judgment.")

human_correlation()
```

---

## BERTScore Cheat Sheet

| Metric Component | What It Measures                       | Range |
| ---------------- | -------------------------------------- | ----- |
| Precision        | How much of candidate is in reference? | 0-1   |
| Recall           | How much of reference is in candidate? | 0-1   |
| F1               | Harmonic mean of precision/recall      | 0-1   |

| Variant      | Description               | Use Case                |
| ------------ | ------------------------- | ----------------------- |
| BERTScore    | Base version with BERT    | General text similarity |
| RoBERTaScore | Using RoBERTa model       | Often slightly better   |
| Rescaled     | Normalized with baselines | Comparing across tasks  |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  BERTScore represents a fundamental shift in evaluation:    ║
║                                                              ║
║  🧠 MEANING OVER FORM: For the first time, we can           ║
║     automatically evaluate whether text MEANS the same,      ║
║     not just whether it uses the same words.                 ║
║                                                              ║
║  🎯 HUMAN-LEVEL CORRELATION: BERTScore matches human        ║
║     judgment better than any previous automatic metric.      ║
║                                                              ║
║  🌍 MULTILINGUAL CAPABILITY: Works across languages         ║
║     with appropriate models, enabling global evaluation.     ║
║                                                              ║
║  🔄 PARAPHRASE ROBUST: Perfect for tasks where there        ║
║     are many valid ways to express the same idea.            ║
║                                                              ║
║  📈 RESEARCH STANDARD: Increasingly used in papers          ║
║     as the primary automatic metric for generation tasks.    ║
║                                                              ║
║  BERTScore isn't perfect, but it's the closest we've        ║
║  come to automatic evaluation that truly understands.        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• BERTScore uses contextual embeddings to measure semantic similarity—instead of counting matching words, it checks whether the meanings align, capturing paraphrases and synonyms that traditional metrics miss

• It computes precision, recall, and F1 based on pairwise word similarities—for each word, finding the best match in the other sentence and aggregating the scores

• While computationally more expensive, BERTScore correlates much better with human judgment—making it the gold standard for evaluating tasks where meaning matters more than exact wording

---

## Mental Hook

> "BERTScore is like having a smart reader who doesn't care if you say 'feline' instead of 'cat'—they understand you're talking about the same thing. It's the difference between grading by dictionary and grading by comprehension."
