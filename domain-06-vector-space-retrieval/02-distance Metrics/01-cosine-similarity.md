# Cosine Similarity (Orientation Matters)

## The Compass Needle Analogy

Imagine two compass needles. They can be different lengths, but if they're both pointing north, they're aligned in the same direction. A short north-pointing needle and a long north-pointing needle are still pointing the same way. That's cosine similarity—it measures the angle between vectors, ignoring their length. In embedding space, it tells us if two words point in the same "meaning direction," regardless of how strongly they express that meaning.

In LLMs, cosine similarity is the most common way to measure semantic similarity. When we say "cat" and "kitten" are similar, we mean their vectors point in roughly the same direction, even if one is longer (maybe representing a stronger or more specific concept). This focus on orientation rather than magnitude is what makes cosine similarity perfect for meaning comparison.

---

## What Is Cosine Similarity?

### The Core Idea

Cosine similarity measures the angle between two vectors, giving a score from -1 to 1.

```text

Vectors that point the SAME way: cos ≈ 1
Vectors that are PERPENDICULAR: cos ≈ 0
Vectors that point OPPOSITE ways: cos ≈ -1

Magnitude (length) doesn't matter—only direction!
```

```python

def cosine_intro():
    """
    The basic concept of cosine similarity
    """
    print("Cosine Similarity: Measuring Alignment")
    print("=" * 60)

    print("""
    Formula:

    cos(θ) = (A·B) / (||A|| × ||B||)

    Where:
    • A·B = dot product (how much they overlap)
    • ||A|| = length of vector A
    • ||B|| = length of vector B

    Plain English:
    "How much do these vectors point in the same direction?"
    """)

cosine_intro()
```

---

## Visualizing Cosine Similarity

### Same Direction, Different Lengths

```python

def same_direction():
    """
    Vectors pointing the same way
    """
    print("Same Direction: Cosine Similarity = 1")
    print("=" * 60)

    print("""
    Vector A: [2, 1]    (length: 2.24)
    Vector B: [4, 2]    (length: 4.47)  (A × 2)

    Both point to the northeast, same angle!

    B (4,2)
      ↑
      │
      │   A (2,1)
      │  ╱
      │ ╱
      └────────→

    Cosine similarity = 1.0 (perfectly aligned)
    """)

same_direction()
```

### Perpendicular Vectors

```python

def perpendicular():
    """
    Vectors at right angles
    """
    print("Perpendicular: Cosine Similarity = 0")
    print("=" * 60)

    print("""
    Vector A: [2, 0]    (pointing east)
    Vector B: [0, 3]    (pointing north)

    They're at 90 degrees—completely unrelated directions!

    B (0,3)
      ↑
      │
      │
      │
      └───────→ A (2,0)

    Cosine similarity = 0 (no alignment)
    """)

perpendicular()
```

### Opposite Directions

```python

def opposite():
    """
    Vectors pointing opposite ways
    """
    print("Opposite Directions: Cosine Similarity = -1")
    print("=" * 60)

    print("""
    Vector A: [2, 1]    (northeast)
    Vector B: [-2, -1]  (southwest)

    They point exactly opposite!

         A (2,1)
          ╱
         ╱
        ╱
    ────┼────→
        ╲
         ╲
          ╲
         B (-2,-1)

    Cosine similarity = -1 (completely opposite)
    """)

opposite()
```

---

## Why Magnitude Doesn't Matter

### The Length vs Direction Distinction

```python

def magnitude_vs_direction():
    """
    Why we ignore length in cosine similarity
    """
    print("Magnitude vs Direction: What Matters for Meaning")
    print("=" * 60)

    print("""
    In embedding spaces:

    DIRECTION = Meaning
    "cat" and "kitten" point same direction (similar meaning)

    MAGNITUDE = Intensity or specificity
    "very happy" might be longer than "happy"
    "ecstatic" might be longer than "happy"

    Cosine similarity只看 direction:
    • "happy" and "joyful" → high cosine (similar meaning)
    • "happy" and "sad" → low or negative cosine (different meaning)

    Length doesn't matter for semantic similarity!
    """)

magnitude_vs_direction()
```

### Example: Same Direction, Different Lengths

```python

def same_dir_example():
    """
    Concrete example with word meanings
    """
    print("Example: Words with Similar Meaning")
    print("=" * 60)

    # Simplified 2D embeddings
    vectors = {
        "happy": [0.8, 0.6],     # length 1.0
        "joyful": [0.9, 0.7],     # length 1.14 (longer)
        "ecstatic": [1.6, 1.2],   # length 2.0 (much longer)
        "sad": [-0.8, -0.6],      # opposite direction
        "car": [0.9, -0.3]        # different direction
    }

    import math
    def cosine(v1, v2):
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        return dot / (len1 * len2)

    print("Comparing 'happy' with other words:")
    print(f"  happy vs joyful: {cosine(vectors['happy'], vectors['joyful']):.3f} (high)")
    print(f"  happy vs ecstatic: {cosine(vectors['happy'], vectors['ecstatic']):.3f} (high)")
    print(f"  happy vs sad: {cosine(vectors['happy'], vectors['sad']):.3f} (negative)")
    print(f"  happy vs car: {cosine(vectors['happy'], vectors['car']):.3f} (low)")

    print("\nKey insight: 'joyful' and 'ecstatic' both score high")
    print("with 'happy' despite different lengths—direction matters!")

same_dir_example()
```

---

## Computing Cosine Similarity

### Step-by-Step Calculation

```python

def cosine_calculation():
    """
    How to compute cosine similarity manually
    """
    print("Computing Cosine Similarity Step by Step")
    print("=" * 60)

    # Two vectors
    A = [3, 4]
    B = [2, 1]

    print(f"Vector A: {A}")
    print(f"Vector B: {B}")

    # Step 1: Dot product
    dot = A[0]*B[0] + A[1]*B[1]
    print(f"\nStep 1: Dot product = {A[0]}×{B[0]} + {A[1]}×{B[1]} = {dot}")

    # Step 2: Lengths
    lenA = (A[0]**2 + A[1]**2) ** 0.5
    lenB = (B[0]**2 + B[1]**2) ** 0.5
    print(f"Step 2: |A| = √({A[0]}² + {A[1]}²) = {lenA:.2f}")
    print(f"        |B| = √({B[0]}² + {B[1]}²) = {lenB:.2f}")

    # Step 3: Cosine
    cos = dot / (lenA * lenB)
    print(f"\nStep 3: cos(θ) = {dot} / ({lenA:.2f} × {lenB:.2f}) = {cos:.3f}")

    print(f"\nInterpretation: The vectors are {cos:.1%} aligned in direction.")

cosine_calculation()
```

### Python Implementation

```python

import numpy as np

def cosine_python():
    """
    Cosine similarity in Python
    """
    print("Cosine Similarity in Python")
    print("=" * 60)

    def cosine_similarity(v1, v2):
        """Calculate cosine similarity between two vectors"""
        v1 = np.array(v1)
        v2 = np.array(v2)

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot / (norm1 * norm2)

    # Test vectors
    test_cases = [
        ([1, 0], [2, 0], "Same direction"),
        ([1, 0], [0, 1], "Perpendicular"),
        ([1, 0], [-1, 0], "Opposite"),
        ([2, 3], [4, 6], "Same direction, different length"),
        ([1, 2, 3], [1, 2, 3], "Identical"),
        ([1, 2, 3], [-1, -2, -3], "Opposite")
    ]

    for v1, v2, desc in test_cases:
        sim = cosine_similarity(v1, v2)
        print(f"{desc:30} {v1} vs {v2}: {sim:.3f}")

cosine_python()
```

---

## Cosine Similarity in Embedding Space

### Real Word Comparisons

```python

def word_comparisons():
    """
    Comparing real word embeddings
    """
    print("Cosine Similarity Between Words")
    print("=" * 60)

    # Simulated word embeddings (simplified)
    words = {
        "cat": [0.8, 0.7, 0.2, 0.1],
        "kitten": [0.9, 0.7, 0.2, 0.1],
        "dog": [0.7, 0.8, 0.2, 0.1],
        "puppy": [0.8, 0.8, 0.2, 0.1],
        "car": [0.1, 0.2, 0.9, 0.8],
        "truck": [0.1, 0.2, 0.9, 0.9],
        "happy": [0.2, 0.8, 0.1, 0.2],
        "joyful": [0.2, 0.9, 0.1, 0.2]
    }

    def cosine(v1, v2):
        dot = sum(v1[i]*v2[i] for i in range(len(v1)))
        len1 = sum(x**2 for x in v1)**0.5
        len2 = sum(x**2 for x in v2)**0.5
        return dot / (len1 * len2)

    print("Similar words (high cosine):")
    print(f"  cat vs kitten: {cosine(words['cat'], words['kitten']):.3f}")
    print(f"  dog vs puppy:  {cosine(words['dog'], words['puppy']):.3f}")
    print(f"  car vs truck:  {cosine(words['car'], words['truck']):.3f}")
    print(f"  happy vs joyful: {cosine(words['happy'], words['joyful']):.3f}")

    print("\nDifferent words (low cosine):")
    print(f"  cat vs car:    {cosine(words['cat'], words['car']):.3f}")
    print(f"  dog vs truck:  {cosine(words['dog'], words['truck']):.3f}")
    print(f"  happy vs car:  {cosine(words['happy'], words['car']):.3f}")

word_comparisons()
```

---

## Cosine Similarity vs Other Metrics

| Metric             | What It Measures       | Range   | Best For            |
| ------------------ | ---------------------- | ------- | ------------------- |
| Cosine similarity  | Angle (direction)      | [-1, 1] | Semantic similarity |
| Euclidean distance | Straight-line distance | [0, ∞)  | Magnitude matters   |
| Dot product        | Similarity × magnitude | (-∞, ∞) | When length matters |
| Manhattan distance | Grid distance          | [0, ∞)  | Sparse vectors      |

### When to Use Cosine

```python

def when_cosine():
    """
    When cosine similarity is the right choice
    """
    print("When to Use Cosine Similarity")
    print("=" * 60)

    use_cases = [
        "Word/sentence embeddings (semantic similarity)",
        "Document retrieval (finding relevant texts)",
        "Recommendation systems (user/item similarity)",
        "Clustering texts by topic",
        "Semantic search (query vs documents)",
        "Any time direction matters more than magnitude"
    ]

    print("Cosine similarity is ideal for:")
    for case in use_cases:
        print(f"  • {case}")

when_cosine()
```

---

## Why This Matters for LLMs

### 1. Semantic Search

```python

def semantic_search():
    """
    Using cosine for search
    """
    print("Semantic Search with Cosine Similarity")
    print("=" * 60)

    query = "cute feline pets"
    query_embedding = [0.8, 0.7, 0.2]  # Simplified

    documents = {
        "Kitten care guide": [0.9, 0.7, 0.2],
        "Dog training tips": [0.7, 0.8, 0.3],
        "Car maintenance": [0.1, 0.2, 0.9],
        "Cat breeds": [0.8, 0.6, 0.2]
    }

    def cosine(v1, v2):
        dot = sum(v1[i]*v2[i] for i in range(len(v1)))
        len1 = sum(x**2 for x in v1)**0.5
        len2 = sum(x**2 for x in v2)**0.5
        return dot / (len1 * len2)

    print(f"Query: '{query}'")
    print("\nResults ranked by cosine similarity:")

    results = []
    for doc, emb in documents.items():
        sim = cosine(query_embedding, emb)
        results.append((doc, sim))

    for doc, sim in sorted(results, key=lambda x: -x[1]):
        print(f"  {sim:.3f}: {doc}")

semantic_search()
```

### 2. Finding Nearest Neighbors

```python

def nearest_neighbors():
    """
    Finding similar words/items
    """
    print("Finding Nearest Neighbors")
    print("=" * 60)

    print("""
    Given a word, find most similar words:

    Word: "king"

    Compute cosine with all other words:
    • queen: 0.85 (royal, female)
    • prince: 0.82 (royal, male)
    • monarch: 0.80 (royal)
    • ruler: 0.75 (leader)
    • throne: 0.70 (associated)
    • car: 0.15 (unrelated)

    Return top k = ["queen", "prince", "monarch", ...]
    """)

nearest_neighbors()
```

### 3. Clustering and Classification

```python

def clustering():
    """
    Grouping similar items
    """
    print("Clustering with Cosine Similarity")
    print("=" * 60)

    print("""
    Cosine similarity is perfect for clustering text:

    1. Convert all documents to embeddings
    2. Compute pairwise cosine similarities
    3. Group documents with high cosine scores

    This naturally creates topic clusters:

    Cluster A (high internal cosine):
    • "Kittens are cute"
    • "Cats make great pets"
    • "How to care for your cat"

    Cluster B:
    • "Stock market trends"
    • "Investment strategies"
    • "Retirement planning"
    """)

clustering()
```

---

## Cosine Similarity Cheat Sheet

| Value      | Meaning             | Example             |
| ---------- | ------------------- | ------------------- |
| 1.0        | Identical direction | "happy" vs "joyful" |
| 0.5-0.9    | Similar direction   | "cat" vs "kitten"   |
| 0.0-0.3    | Unrelated           | "cat" vs "car"      |
| -0.3 to 0  | Slightly opposite   | "hot" vs "cold"     |
| -0.5 to -1 | Opposite meaning    | "good" vs "evil"    |

---

## Quick Recap

• Cosine similarity measures the angle between vectors, ignoring their length—like comparing compass needles by where they point, not how long they are, making it perfect for semantic similarity

• Values range from 1 (same direction) through 0 (perpendicular) to -1 (opposite)—similar words have high positive cosine, unrelated words near zero, opposites negative

• Cosine similarity is the standard for comparing embeddings in LLMs—used in semantic search, recommendations, clustering, and finding nearest neighbors

---

## Mental Hook

> "Cosine similarity is like asking whether two arrows point the same way—a short arrow pointing north and a long arrow pointing north are both going north, and that's what matters for meaning, not how strongly they're pointing."
