# BLEU Score

## The Translation Competition Analogy

Imagine two translators competing to see who can best translate a sentence from English to French. You can't just ask "which is better?"—that's subjective. Instead, you count how many words and phrases match a set of perfect human translations. The translator whose version shares more words and phrases with the human references wins. That's BLEU (Bilingual Evaluation Understudy) Score: an automatic way to measure how close machine-generated text is to human-quality reference translations.

In LLM evaluation, BLEU is most commonly used for translation, summarization, and other text generation tasks. It's fast, automatic, and correlates reasonably well with human judgment. While not perfect, it remains one of the most widely used metrics for comparing generation quality.

---

## What Is BLEU Score?

### The Core Idea

BLEU compares a candidate translation (from your model) against one or more reference translations (from humans) by counting matching n-grams.

```text

Candidate: "The cat sat on the mat"
Reference: "The cat sat on the rug"

Matching words: The, cat, sat, on, the (5/6 words)
Matching bigrams: The cat, cat sat, sat on, on the (4/5 bigrams)

BLEU combines these matches into a single score (0-1, higher is better)
```

```python

def bleu_intro():
    """
    The basic concept of BLEU score
    """
    print("BLEU Score: Measuring Translation Quality")
    print("=" * 60)

    print("""
    BLEU combines:

    1. N-gram precision: How many n-grams match references?
       • Unigrams (single words)
       • Bigrams (word pairs)
       • Trigrams (three-word phrases)
       • 4-grams (four-word phrases)

    2. Brevity penalty: Prevents short, "safe" translations from winning
       • Penalizes candidates that are too short

    Final score ranges from 0 to 1 (or 0-100 as percentage)
    """)

bleu_intro()
```

---

## How BLEU Works Step by Step

### Step 1: Calculate N-gram Precision

```python

def ngram_precision():
    """
    Calculating n-gram precision
    """
    print("Step 1: N-gram Precision")
    print("=" * 60)

    candidate = "the cat sat on the mat".split()
    reference = "the cat sat on the rug".split()

    def ngram_precision(cand, ref, n):
        """Simplified n-gram precision"""
        # Generate n-grams from candidate
        cand_ngrams = [' '.join(cand[i:i+n]) for i in range(len(cand)-n+1)]

        # Generate n-grams from reference
        ref_ngrams = [' '.join(ref[i:i+n]) for i in range(len(ref)-n+1)]

        # Count matches (simplified - real BLEU clips counts)
        matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
        total = len(cand_ngrams)

        return matches, total, cand_ngrams, ref_ngrams

    print(f"Candidate: {' '.join(candidate)}")
    print(f"Reference: {' '.join(reference)}\n")

    for n in range(1, 5):
        matches, total, cand_ng, ref_ng = ngram_precision(candidate, reference, n)
        precision = matches / total if total > 0 else 0
        print(f"{n}-gram precision: {matches}/{total} = {precision:.2f}")
        if n <= 2:
            print(f"  Candidate {n}-grams: {cand_ng}")
            print(f"  Reference {n}-grams:  {ref_ng}")

ngram_precision()
```

### Step 2: Apply Clipping (No Double-Counting)

```python

def clipping():
    """
    Clipping prevents overcounting common words
    """
    print("Step 2: Clipping - No Double-Counting")
    print("=" * 60)

    candidate = "the the the the the".split()
    reference = "the cat sat on the mat".split()

    # Without clipping
    from collections import Counter

    cand_unigrams = Counter(candidate)
    ref_unigrams = Counter(reference)

    print("Candidate unigrams:", dict(cand_unigrams))
    print("Reference unigrams:", dict(ref_unigrams))

    # Without clipping (bad)
    raw_matches = sum(cand_unigrams.values())
    raw_precision = raw_matches / len(candidate)
    print(f"\nWithout clipping: {raw_matches}/{len(candidate)} = {raw_precision:.2f}")
    print("  ❌ Model would score well by just repeating 'the'!")

    # With clipping
    clipped_matches = 0
    for word, count in cand_unigrams.items():
        clipped_matches += min(count, ref_unigrams.get(word, 0))

    clipped_precision = clipped_matches / len(candidate)
    print(f"With clipping: {clipped_matches}/{len(candidate)} = {clipped_precision:.2f}")
    print("  ✅ Much more reasonable!")

clipping()
```

### Step 3: Geometric Mean of N-gram Precisions

```python

def geometric_mean():
    """
    Combining n-gram precisions
    """
    print("Step 3: Geometric Mean of N-gram Precisions")
    print("=" * 60)

    # Example precisions from a good translation
    precisions = [0.8, 0.6, 0.4, 0.2]  # unigram to 4-gram

    import math

    # Arithmetic mean (not used)
    arith_mean = sum(precisions) / len(precisions)

    # Geometric mean (used by BLEU)
    log_sum = sum(math.log(p) for p in precisions)
    geo_mean = math.exp(log_sum / len(precisions))

    print(f"Precisions: {precisions}")
    print(f"Arithmetic mean: {arith_mean:.3f}")
    print(f"Geometric mean: {geo_mean:.3f}")

    print("\nWhy geometric?")
    print("• Punishes low scores more heavily")
    print("• If any n-gram precision is 0, overall score is 0")
    print("• Encourages good matches at all lengths")

geometric_mean()
```

### Step 4: Brevity Penalty

```python

def brevity_penalty():
    """
    Penalizing translations that are too short
    """
    print("Step 4: Brevity Penalty")
    print("=" * 60)

    def calculate_brevity_penalty(cand_len, ref_len):
        if cand_len > ref_len:
            return 1.0  # No penalty if candidate is longer
        else:
            return math.exp(1 - ref_len / cand_len)

    examples = [
        {"cand": 6, "ref": 6, "desc": "Perfect length"},
        {"cand": 6, "ref": 8, "desc": "Candidate too short"},
        {"cand": 8, "ref": 6, "desc": "Candidate too long"},
        {"cand": 3, "ref": 6, "desc": "Much too short"}
    ]

    import math

    for ex in examples:
        bp = calculate_brevity_penalty(ex["cand"], ex["ref"])
        print(f"{ex['desc']:20} | cand={ex['cand']}, ref={ex['ref']} | BP={bp:.3f}")

brevity_penalty()
```

---

## Complete BLEU Calculation

```python

import math
from collections import Counter

def bleu_calculation():
    """
    Complete BLEU score calculation
    """
    print("Complete BLEU Score Calculation")
    print("=" * 60)

    candidate = "the cat sat on the mat".split()
    references = [
        "the cat sat on the rug".split(),
        "a cat sat on the mat".split(),
        "the cat was sitting on the mat".split()
    ]

    print(f"Candidate: {' '.join(candidate)}")
    print("References:")
    for ref in references:
        print(f"  • {' '.join(ref)}")

    # Step 1: Calculate n-gram precisions with clipping
    precisions = []
    for n in range(1, 5):
        cand_ngrams = [' '.join(candidate[i:i+n]) for i in range(len(candidate)-n+1)]
        cand_counts = Counter(cand_ngrams)

        # Maximum count over all references (with clipping)
        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = [' '.join(ref[i:i+n]) for i in range(len(ref)-n+1)]
            ref_counts = Counter(ref_ngrams)
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

        # Count clipped matches
        matches = 0
        for ngram, count in cand_counts.items():
            matches += min(count, max_ref_counts.get(ngram, 0))

        precision = matches / len(cand_ngrams) if cand_ngrams else 0
        precisions.append(precision)
        print(f"\n{n}-gram precision: {matches}/{len(cand_ngrams)} = {precision:.3f}")

    # Step 2: Geometric mean
    log_sum = sum(math.log(p) for p in precisions if p > 0)
    if any(p == 0 for p in precisions):
        geo_mean = 0
        print("\n⚠️ Some n-gram precision is 0 → BLEU = 0")
    else:
        geo_mean = math.exp(log_sum / len(precisions))
        print(f"\nGeometric mean: {geo_mean:.3f}")

    # Step 3: Brevity penalty
    cand_len = len(candidate)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - cand_len))

    if cand_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / cand_len)

    print(f"\nCandidate length: {cand_len}")
    print(f"Closest reference length: {closest_ref_len}")
    print(f"Brevity penalty: {bp:.3f}")

    # Step 4: Final BLEU
    bleu = bp * geo_mean if geo_mean > 0 else 0
    print(f"\n{'='*40}")
    print(f"FINAL BLEU SCORE: {bleu:.3f} ({bleu*100:.1f})")
    print(f"{'='*40}")

bleu_calculation()
```

---

## BLEU Score Examples

### Good Translation

```python

def good_translation():
    """
    Example of a good translation (high BLEU)
    """
    print("Good Translation - High BLEU")
    print("=" * 60)

    candidate = "The cat sat on the mat"
    references = [
        "The cat sat on the rug",
        "A cat sat on the mat",
        "The cat was sitting on the mat"
    ]

    print(f"Candidate: {candidate}")
    print("References:")
    for ref in references:
        print(f"  • {ref}")

    # Simulate high BLEU
    print("\n✓ High word overlap")
    print("✓ Most n-grams match")
    print("✓ Good length match")
    print("→ BLEU ≈ 0.7-0.9")

good_translation()
```

### Mediocre Translation

```python

def mediocre_translation():
    """
    Example of a mediocre translation (medium BLEU)
    """
    print("Mediocre Translation - Medium BLEU")
    print("=" * 60)

    candidate = "The cat is on the mat"
    references = [
        "The cat sat on the rug",
        "A cat sat on the mat"
    ]

    print(f"Candidate: {candidate}")
    print("References:")
    for ref in references:
        print(f"  • {ref}")

    print("\n⚠ Some words match, some don't")
    print("⚠ Bigram matches missing ('cat sat')")
    print("→ BLEU ≈ 0.3-0.5")

mediocre_translation()
```

### Bad Translation

```python

def bad_translation():
    """
    Example of a bad translation (low BLEU)
    """
    print("Bad Translation - Low BLEU")
    print("=" * 60)

    candidate = "The feline on the carpet"
    references = [
        "The cat sat on the rug",
        "A cat sat on the mat"
    ]

    print(f"Candidate: {candidate}")
    print("References:")
    for ref in references:
        print(f"  • {ref}")

    print("\n✗ Few word matches")
    print("✗ Very different word choice")
    print("✗ N-gram precision near zero")
    print("→ BLEU ≈ 0.0-0.2")

bad_translation()
```

---

## BLEU in Python (Using NLTK)

```python

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def nltk_bleu():
    """
    Using NLTK's BLEU implementation
    """
    print("BLEU Score with NLTK")
    print("=" * 60)

    # Download required data (uncomment if needed)
    # nltk.download('punkt')

    candidate = "the cat sat on the mat".split()
    references = [
        "the cat sat on the rug".split(),
        "a cat sat on the mat".split(),
        "the cat was sitting on the mat".split()
    ]

    # Basic BLEU (1-gram to 4-gram)
    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0))
    bleu4 = sentence_bleu(references, candidate)

    print(f"Candidate: {' '.join(candidate)}")
    print(f"\nBLEU-1 (unigrams only): {bleu1:.3f}")
    print(f"BLEU-2 (unigrams+bigrams): {bleu2:.3f}")
    print(f"BLEU-4 (standard): {bleu4:.3f}")

    # Handle zero n-gram matches with smoothing
    chencherry = SmoothingFunction()

    # Problem case: no 4-gram matches
    bad_candidate = "the cat".split()
    bleu_unsmoothed = sentence_bleu(references, bad_candidate)
    bleu_smoothed = sentence_bleu(references, bad_candidate,
                                  smoothing_function=chencherry.method1)

    print(f"\nWithout smoothing (short candidate): {bleu_unsmoothed:.3f}")
    print(f"With smoothing: {bleu_smoothed:.3f}")

nltk_bleu()
```

---

## BLEU Score Interpretation

| BLEU Score | Interpretation        | Example                |
| ---------- | --------------------- | ---------------------- |
| > 0.6      | Excellent translation | Human-level or better  |
| 0.4 - 0.6  | Good translation      | Most meaning preserved |
| 0.2 - 0.4  | Understandable        | Main idea gets across  |
| 0.1 - 0.2  | Poor translation      | Many errors            |
| < 0.1      | Unusable              | Mostly wrong           |

### Typical Scores for Systems

```python

def typical_scores():
    """
    Typical BLEU scores for different systems
    """
    print("Typical BLEU Scores")
    print("=" * 60)

    scores = {
        "Human translation": "0.6-0.8",
        "Google Translate (2024)": "0.5-0.7",
        "GPT-4 translation": "0.5-0.7",
        "Statistical MT (pre-neural)": "0.3-0.4",
        "Rule-based MT": "0.1-0.2",
        "Random words": "<0.01"
    }

    for system, score in scores.items():
        print(f"  • {system:25} → {score}")

typical_scores()
```

---

## Limitations of BLEU

### What BLEU Misses

```python

def bleu_limitations():
    """
    Limitations of BLEU score
    """
    print("Limitations of BLEU")
    print("=" * 60)

    limitations = [
        "SYNONYMS: 'big' vs 'large' → penalized even though meaning is same",
        "WORD ORDER: Slight reordering can drop score significantly",
        "CREATIVITY: Penalizes novel but valid translations",
        "LENGTH BIAS: Short translations can be gamed",
        "NO SEMANTICS: Doesn't understand meaning, only surface form",
        "SINGLE REFERENCE: Multiple references help but still limited"
    ]

    for lim in limitations:
        print(f"  • {lim}")

    print("\nExample that would score low but is actually good:")
    print("  Candidate: 'The feline reclined upon the carpet'")
    print("  Reference: 'The cat sat on the mat'")
    print("  BLEU would be low, but meaning is similar!")

bleu_limitations()
```

---

## BLEU vs Other Metrics

| Metric    | What It Measures    | Pros                           | Cons                      |
| --------- | ------------------- | ------------------------------ | ------------------------- |
| BLEU      | N-gram overlap      | Fast, standard                 | Ignores meaning           |
| ROUGE     | Recall-oriented     | Good for summarization         | Similar limitations       |
| METEOR    | Synonyms + stemming | Better correlation with humans | Slower, complex           |
| BERTScore | Semantic similarity | Understands meaning            | Computationally expensive |
| chrF      | Character n-grams   | Good for morphologically rich  | Less common               |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  BLEU is the workhorse of NLP evaluation:                   ║
║                                                              ║
║  🏆 INDUSTRY STANDARD: Every translation paper reports       ║
║     BLEU. Knowing it lets you read and compare research.     ║
║                                                              ║
║  ⚡ FAST AUTOMATION: Evaluate thousands of translations      ║
║     in seconds without humans in the loop.                   ║
║                                                              ║
║  📈 PROGRESS TRACKING: Watch BLEU improve as your           ║
║     model gets better—or catch regressions.                  ║
║                                                              ║
║  🔍 DEBUGGING: Low BLEU on specific sentences               ║
║     helps identify where your model struggles.               ║
║                                                              ║
║  🌍 MULTILINGUAL: Works for any language, making            ║
║     it invaluable for global applications.                   ║
║                                                              ║
║  BLEU isn't perfect, but it's the fastest way to            ║
║  know if your translation system is improving.               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• BLEU measures how many n-grams match between candidate and reference translations—higher is better, with scores above 0.6 considered excellent

• It combines clipped n-gram precision with a brevity penalty—preventing both overcounting of common words and gaming by producing short translations

• While fast and standard, BLEU has significant limitations—it ignores meaning, penalizes synonyms, and doesn't correlate perfectly with human judgment

---

## Mental Hook

> "BLEU is like a strict teacher grading essays by counting how many words match an answer key—it's fast and consistent, but it might miss that 'the feline reclined upon the carpet' is just as good as 'the cat sat on the mat'."
