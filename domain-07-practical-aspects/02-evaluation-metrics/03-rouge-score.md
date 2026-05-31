# ROUGE Score

## The Summary Grading Analogy

Imagine you're a teacher grading student summaries of a long article. You have a perfect summary written by an expert. You check each student's work: Did they mention the main points? Did they use the right words? Did they miss anything important? That's ROUGE (Recall-Oriented Understudy for Gisting Evaluation) : a set of metrics that compares automatically generated summaries against human-written reference summaries.

In LLM evaluation, ROUGE is the standard for summarization tasks. Unlike BLEU (which focuses on precision), ROUGE emphasizes recall—how much of the important content from the reference appears in the generated summary. It tells you whether your model captured the key information, not just whether it used the right words.

---

## What Is ROUGE?

### The Core Idea

ROUGE measures the overlap between a generated summary and reference summaries, with a focus on recall (how much of the reference is captured).

```text

Reference: "The cat sat on the mat and slept peacefully"
Generated: "The cat slept on the mat"

ROUGE checks:
• Did we capture "cat"? ✓
• Did we capture "slept"? ✓
• Did we capture "mat"? ✓
• Did we capture "peacefully"? ✗ (missed)
• Did we capture "sat"? ✗ (missed)

Recall = 4/6 = 0.67 (caught 4 of 6 important pieces)
```

```python

def rouge_intro():
    """
    The basic concept of ROUGE
    """
    print("ROUGE: Measuring Summary Quality")
    print("=" * 60)

    print("""
    ROUGE has several variants:

    • ROUGE-N: N-gram overlap (like BLEU, but recall-focused)
    • ROUGE-L: Longest common subsequence (captures fluency)
    • ROUGE-S: Skip-bigram (allowing gaps between words)
    • ROUGE-W: Weighted LCS (favors consecutive matches)

    Each gives a different view of summary quality!
    """)

rouge_intro()
```

---

## ROUGE-N: N-gram Overlap

### Precision vs Recall in ROUGE

```python

def precision_recall():
    """
    Precision vs Recall in ROUGE
    """
    print("ROUGE-N: Precision vs Recall")
    print("=" * 60)

    reference = "the cat sat on the mat".split()
    candidate = "the cat slept".split()

    from collections import Counter

    ref_unigrams = Counter(reference)
    cand_unigrams = Counter(candidate)

    # Count overlaps
    overlap = 0
    for word, count in cand_unigrams.items():
        overlap += min(count, ref_unigrams.get(word, 0))

    # Precision = overlap / candidate length
    precision = overlap / len(candidate)

    # Recall = overlap / reference length
    recall = overlap / len(reference)

    # F1 = harmonic mean
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Reference ({len(reference)} words): {reference}")
    print(f"Candidate ({len(candidate)} words): {candidate}")
    print(f"\nOverlapping words: {overlap}")
    print(f"Precision: {overlap}/{len(candidate)} = {precision:.2f} (of candidate words, how many are good?)")
    print(f"Recall:    {overlap}/{len(reference)} = {recall:.2f} (of reference words, how many were captured?)")
    print(f"F1-score:  {f1:.2f}")

precision_recall()
```

### ROUGE-1, ROUGE-2, ROUGE-3

```python

def rouge_n():
    """
    ROUGE-1, ROUGE-2, ROUGE-3 examples
    """
    print("ROUGE-1, ROUGE-2, ROUGE-3")
    print("=" * 60)

    reference = "the cat sat on the mat and slept".split()
    candidate = "the cat slept on the mat".split()

    def rouge_n(ref, cand, n):
        """Calculate ROUGE-N F1 score"""
        ref_ngrams = [' '.join(ref[i:i+n]) for i in range(len(ref)-n+1)]
        cand_ngrams = [' '.join(cand[i:i+n]) for i in range(len(cand)-n+1)]

        ref_counts = Counter(ref_ngrams)
        cand_counts = Counter(cand_ngrams)

        # Count overlaps (with clipping)
        overlap = 0
        for ngram, count in cand_counts.items():
            overlap += min(count, ref_counts.get(ngram, 0))

        precision = overlap / len(cand_ngrams) if cand_ngrams else 0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    print(f"Reference: {' '.join(reference)}")
    print(f"Candidate: {' '.join(candidate)}\n")

    for n in range(1, 4):
        p, r, f = rouge_n(reference, candidate, n)
        print(f"ROUGE-{n}:")
        print(f"  Precision: {p:.3f}")
        print(f"  Recall:    {r:.3f}")
        print(f"  F1:        {f:.3f}")

rouge_n()
```

---

## ROUGE-L: Longest Common Subsequence

### Capturing Fluency and Word Order

```python

def lcs_explanation():
    """
    Longest Common Subsequence (LCS)
    """
    print("ROUGE-L: Longest Common Subsequence")
    print("=" * 60)

    print("""
    LCS finds the longest sequence of words that appear
    in both texts in the same order (but not necessarily consecutive).

    Example:
    Reference: "the cat sat on the mat"
    Candidate: "the cat slept on the mat"

    LCS: "the cat on the mat" (5 words)

    Why this matters:
    • Captures fluency (word order)
    • Allows for insertions/deletions
    • Rewards coherent structure
    """)

lcs_explanation()
```

### Computing ROUGE-L

```python

def rouge_l():
    """
    ROUGE-L calculation
    """
    print("ROUGE-L Calculation")
    print("=" * 60)

    def lcs_length(x, y):
        """Dynamic programming for LCS length"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    reference = "the cat sat on the mat".split()
    candidate = "the cat slept on the mat".split()

    lcs_len = lcs_length(reference, candidate)

    print(f"Reference: {' '.join(reference)}")
    print(f"Candidate: {' '.join(candidate)}")
    print(f"\nLCS length: {lcs_len}")
    print(f"Reference length: {len(reference)}")
    print(f"Candidate length: {len(candidate)}")

    # ROUGE-L scores
    recall = lcs_len / len(reference)
    precision = lcs_len / len(candidate)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nROUGE-L Recall:    {recall:.3f}")
    print(f"ROUGE-L Precision: {precision:.3f}")
    print(f"ROUGE-L F1:        {f1:.3f}")

rouge_l()
```

---

## ROUGE-S: Skip-Bigram

### Allowing Gaps Between Words

```python

def rouge_s():
    """
    ROUGE-S: Skip-bigram co-occurrence
    """
    print("ROUGE-S: Skip-Bigram")
    print("=" * 60)

    print("""
    Skip-bigrams are pairs of words in sentence order,
    allowing gaps of up to 4 words between them.

    Sentence: "the cat sat on the mat"

    Skip-bigrams (max gap 2):
    • (the, cat)   gap 0
    • (the, sat)   gap 1
    • (the, on)    gap 2
    • (cat, sat)   gap 0
    • (cat, on)    gap 1
    • (sat, on)    gap 0
    • etc.

    This captures relationships between words even when
    extra words are inserted in between.
    """)

rouge_s()
```

---

## Complete ROUGE Example

```python

from collections import Counter

def rouge_complete():
    """
    Complete ROUGE evaluation example
    """
    print("Complete ROUGE Evaluation")
    print("=" * 60)

    # Test data
    reference = "the cat sat on the mat and slept peacefully".split()
    candidates = {
        "Good summary": "the cat slept on the mat".split(),
        "Wordy summary": "the lovely cat was sleeping on the comfortable mat".split(),
        "Missing info": "something about cats".split(),
        "Too short": "cat slept".split()
    }

    def rouge_1(ref, cand):
        """ROUGE-1 F1 score"""
        ref_counts = Counter(ref)
        cand_counts = Counter(cand)

        overlap = 0
        for word, count in cand_counts.items():
            overlap += min(count, ref_counts.get(word, 0))

        precision = overlap / len(cand) if cand else 0
        recall = overlap / len(ref) if ref else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    def rouge_2(ref, cand):
        """ROUGE-2 F1 score"""
        if len(ref) < 2 or len(cand) < 2:
            return 0

        ref_bigrams = [' '.join(ref[i:i+2]) for i in range(len(ref)-1)]
        cand_bigrams = [' '.join(cand[i:i+2]) for i in range(len(cand)-1)]

        ref_counts = Counter(ref_bigrams)
        cand_counts = Counter(cand_bigrams)

        overlap = 0
        for bigram, count in cand_counts.items():
            overlap += min(count, ref_counts.get(bigram, 0))

        precision = overlap / len(cand_bigrams)
        recall = overlap / len(ref_bigrams)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    def rouge_l(ref, cand):
        """ROUGE-L F1 score (simplified)"""
        # Simple LCS implementation for demo
        m, n = len(ref), len(cand)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == cand[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_len = dp[m][n]

        recall = lcs_len / m
        precision = lcs_len / n
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    print(f"Reference: {' '.join(reference)}\n")

    for name, candidate in candidates.items():
        print(f"\n{name}: {' '.join(candidate)}")
        r1 = rouge_1(reference, candidate)
        r2 = rouge_2(reference, candidate)
        rl = rouge_l(reference, candidate)
        print(f"  ROUGE-1: {r1:.3f}")
        print(f"  ROUGE-2: {r2:.3f}")
        print(f"  ROUGE-L: {rl:.3f}")

rouge_complete()
```

---

## ROUGE in Python (Using Hugging Face)

```python

# Note: This requires installation: pip install evaluate
# import evaluate

def rouge_huggingface():
    """
    Using Hugging Face's evaluate library
    """
    print("ROUGE with Hugging Face Evaluate")
    print("=" * 60)

    print("""
    from evaluate import load
    rouge = load('rouge')

    predictions = [
        "the cat slept on the mat",
        "the cat sat on the mat"
    ]

    references = [
        "the cat sat on the mat and slept",
        "the cat sat on the mat"
    ]

    results = rouge.compute(
        predictions=predictions,
        references=references
    )

    print(results)
    # {'rouge1': 0.75, 'rouge2': 0.5, 'rougeL': 0.67, ...}
    """)

    print("\nBenefits of using a library:")
    print("• Handles tokenization")
    print("• Multiple reference support")
    print("• Aggregates scores correctly")
    print("• Standard implementation")

rouge_huggingface()
```

---

## ROUGE Score Interpretation

| ROUGE Score | Interpretation    | Example                    |
| ----------- | ----------------- | -------------------------- |
| > 0.5       | Excellent summary | Captures all key points    |
| 0.3 - 0.5   | Good summary      | Main ideas present         |
| 0.2 - 0.3   | Adequate summary  | Some key points missing    |
| 0.1 - 0.2   | Poor summary      | Misses most important info |
| < 0.1       | Unusable          | Completely off-topic       |

### Typical Scores for Systems

```python

def typical_rouge():
    """
    Typical ROUGE scores for different systems
    """
    print("Typical ROUGE-1 Scores")
    print("=" * 60)

    scores = {
        "Human summarizer": "0.45-0.55",
        "GPT-4 summarization": "0.40-0.50",
        "BART/LED": "0.35-0.45",
        "Extractive baseline": "0.30-0.40",
        "Lead-3 baseline": "0.25-0.35",
        "Random sentences": "0.05-0.10"
    }

    for system, score in scores.items():
        print(f"  • {system:25} → {score}")

typical_rouge()
```

---

## ROUGE vs BLEU

| Aspect              | ROUGE                        | BLEU                             |
| ------------------- | ---------------------------- | -------------------------------- |
| Focus               | Recall-oriented              | Precision-oriented               |
| Best for            | Summarization                | Translation                      |
| Key insight         | "Did we capture everything?" | "Is everything we said correct?" |
| Penalizes           | Missing content              | Extra/incorrect content          |
| Multiple references | Yes (averaged)               | Yes (clipped)                    |
| Variants            | ROUGE-N, L, S, W             | BLEU-1,2,3,4                     |

### When to Use Which

```python

def when_to_use():
    """
    Choosing between ROUGE and BLEU
    """
    print("ROUGE vs BLEU: When to Use")
    print("=" * 60)

    scenarios = [
        ("Summarization", "ROUGE", "Care about capturing key points"),
        ("Translation", "BLEU", "Care about fluency and correctness"),
        ("Question answering", "ROUGE", "Answer must contain correct info"),
        ("Image captioning", "Both", "Need both coverage and fluency"),
        ("Data-to-text", "BLEU", "Need accurate reporting of facts")
    ]

    for task, metric, reason in scenarios:
        print(f"\n  • {task:20} → {metric:6} | {reason}")

when_to_use()
```

---

## Limitations of ROUGE

```python

def rouge_limitations():
    """
    Limitations of ROUGE
    """
    print("Limitations of ROUGE")
    print("=" * 60)

    limitations = [
        "SYNONYMS: 'big' vs 'large' → penalized unfairly",
        "REORDERING: The same info in different order gets lower score",
        "LENGTH BIAS: Longer summaries can artificially inflate recall",
        "NO SEMANTICS: Doesn't understand meaning, only surface form",
        "CONTEXT IGNORED: Can't tell if info is used correctly",
        "REFERENCE DEPENDENT: Quality depends on human references"
    ]

    for lim in limitations:
        print(f"  • {lim}")

    print("\nExample of good summary that would score low:")
    print("  Reference: 'The cat sat on the mat and slept'")
    print("  Good alt:  'A feline rested on the rug, napping'")
    print("  ROUGE would be low despite similar meaning!")

rouge_limitations()
```

---

## ROUGE Cheat Sheet

| Metric  | What It Measures           | Range | Best Use            |
| ------- | -------------------------- | ----- | ------------------- |
| ROUGE-1 | Unigram overlap            | 0-1   | Content coverage    |
| ROUGE-2 | Bigram overlap             | 0-1   | Phrase fluency      |
| ROUGE-L | Longest common subsequence | 0-1   | Sentence structure  |
| ROUGE-S | Skip-bigram                | 0-1   | Flexible matching   |
| ROUGE-W | Weighted LCS               | 0-1   | Consecutive matches |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ROUGE is the standard for summarization evaluation:        ║
║                                                              ║
║  📰 NEWS SUMMARIZATION: Every paper on summarization        ║
║     reports ROUGE. It's the common language of the field.    ║
║                                                              ║
║  🎯 CONTENT COVERAGE: Unlike BLEU, ROUGE tells you          ║
║     if you captured the important information.               ║
║                                                              ║
║  📊 MULTIPLE PERSPECTIVES: Different ROUGE variants         ║
║     reveal different aspects of summary quality.             ║
║                                                              ║
║  🔍 DEBUGGING: Low ROUGE on certain topics helps            ║
║     identify where your summarizer is failing.               ║
║                                                              ║
║  🌍 WIDELY ADOPTED: From research papers to production      ║
║     systems, ROUGE is everywhere.                            ║
║                                                              ║
║  ROUGE isn't perfect, but it's the best tool we have        ║
║  for automatically evaluating summaries at scale.            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• ROUGE measures overlap between generated and reference summaries, focusing on recall—it tells you how much of the important content from the reference was captured

• Key variants include ROUGE-N (n-gram overlap), ROUGE-L (longest common subsequence), and ROUGE-S (skip-bigram)—each capturing different aspects of summary quality

• While standard for summarization, ROUGE has limitations—it misses synonyms, doesn't understand meaning, and correlates imperfectly with human judgment

---

## Mental Hook

> "ROUGE is like a teacher checking if you mentioned all the key points from the chapter—it doesn't care if you used fancy words, just whether you covered what mattered. If you missed a key fact, you lose points, even if your writing was beautiful."
