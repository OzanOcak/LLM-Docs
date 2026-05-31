# How Embeddings Capture Semantics

## The Social Network Analogy

Imagine you're new to a city and you want to understand who's who. You notice that certain people always appear together at the same events—the art crowd, the tech entrepreneurs, the university professors. Without anyone telling you labels, you figure out that people who hang out together share similar interests and professions. That's how embeddings capture semantics: words that appear in similar contexts get similar vectors, purely from observing their "social network" of usage.

In LLMs, this is the magic—semantic meaning emerges automatically from co-occurrence patterns. The model never gets told that "cat" and "kitten" are related; it discovers this because they appear in similar sentences. This is why embeddings can capture nuance, analogy, and even cultural associations.

---

## The Distributional Hypothesis

### Core Principle

Words that appear in similar contexts have similar meanings.

```python

def distributional_hypothesis():
    """
    The fundamental idea behind semantic capture
    """
    print("The Distributional Hypothesis")
    print("=" * 60)

    print("""
    "You shall know a word by the company it keeps."
                                        - J.R. Firth (1957)

    Examples:

    Context for "cat":
    • "The ___ sat on the mat"
    • "I fed my ___ this morning"
    • "The ___ chased the mouse"

    Context for "dog":
    • "The ___ slept on the rug"
    • "I walked my ___ in the park"
    • "The ___ barked at the mailman"

    These contexts are SIMILAR → "cat" and "dog" are similar!
    """)

distributional_hypothesis()
```

### Context Vectors

```python

def context_vectors():
    """
    Building meaning from context
    """
    print("Building Context Vectors")
    print("=" * 60)

    # Simplified context counts
    contexts = {
        "cat": {
            "sat on": 50,
            "chased": 30,
            "meowed": 20,
            "purred": 15,
            "scratched": 10
        },
        "dog": {
            "sat on": 45,
            "chased": 35,
            "barked": 40,
            "wagged": 25,
            "scratched": 8
        },
        "car": {
            "drove": 60,
            "parked": 40,
            "fueled": 25,
            "washed": 20,
            "scratched": 5
        }
    }

    print("Context profiles (simplified):")
    for word, ctx in contexts.items():
        print(f"\n  '{word}':")
        for context, count in list(ctx.items())[:3]:
            print(f"    • appears with '{context}': {count} times")

    print("\n'cat' and 'dog' share many contexts → similar vectors")
    print("'car' shares fewer contexts → different vector")

context_vectors()
```

---

## How Training Creates Semantic Structure

### Step-by-Step Emergence

```python

def training_process():
    """
    How embeddings learn semantics during training
    """
    print("The Training Process: Semantics Emerge")
    print("=" * 60)

    print("""
    Step 1: Random initialization
    ──────────────────────────────────
    All words start with random vectors.
    "cat" and "dog" are far apart (random).

    Step 2: Prediction task
    ──────────────────────────────────
    "The ___ sat on the mat" → predict "cat"
    Model adjusts "cat" vector to help prediction.

    Step 3: Shared contexts
    ──────────────────────────────────
    "The ___ sat on the mat" also occurs with "dog"
    Model adjusts "dog" vector similarly.

    Step 4: Convergence
    ──────────────────────────────────
    After millions of updates,
    "cat" and "dog" end up close together
    because they help predict the same contexts.
    """)

training_process()
```

### The CBOW and Skip-Gram Models

```python

def word2vec():
    """
    Classic word2vec architectures
    """
    print("Word2Vec: How It Started")
    print("=" * 60)

    print("""
    Two classic architectures:

    1. CBOW (Continuous Bag of Words)
       Context → Predict target word
       ["the", "sat", "on", "the"] → "cat"

    2. Skip-gram
       Target word → Predict context
       "cat" → ["the", "sat", "on", "the"]

    Both force the model to learn word relationships
    by making words that appear in similar contexts
    have similar vectors.
    """)

word2vec()
```

---

## The Geometry of Meaning

### Clusters and Neighborhoods

```python

def semantic_clusters():
    """
    How words form semantic neighborhoods
    """
    print("Semantic Neighborhoods")
    print("=" * 60)

    print("""
    In embedding space, words naturally cluster:

                    ┌─────────────────┐
                    │   ANIMALS       │
                    │  cat───┐        │
                    │   │    │        │
                    │  dog───┼──kitten │
                    │   │    │        │
                    │  horse─┘        │
                    └─────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ↓                               ↓
    ┌─────────────┐                 ┌─────────────┐
    │   VEHICLES  │                 │   FRUIT     │
    │ car───truck │                 │ apple─orange│
    │  │     │    │                 │  │     │    │
    │ bus───van   │                 │ banana─pear │
    └─────────────┘                 └─────────────┘

    Each cluster represents a semantic category.
    The boundaries are fuzzy, but neighborhoods are clear.
    """)

semantic_clusters()
```

### Analogies as Vector Arithmetic

```python

def analogies():
    """
    How analogies emerge from geometry
    """
    print("Analogies: Vector Arithmetic in Meaning Space")
    print("=" * 60)

    # Simplified 2D embeddings
    embeddings = {
        "king": [0.9, 0.8],
        "queen": [0.9, 0.2],
        "man": [0.3, 0.8],
        "woman": [0.3, 0.2],
        "paris": [0.8, 0.7],
        "france": [0.8, 0.1],
        "rome": [0.2, 0.7],
        "italy": [0.2, 0.1]
    }

    print("Dimension 1: 'royalty/country' axis")
    print("Dimension 2: 'gender/capital' axis")

    print("\nAnalogy: king : queen :: man : woman")
    king = embeddings["king"]
    man = embeddings["man"]
    woman = embeddings["woman"]
    queen = embeddings["queen"]

    print(f"  king - man + woman = {[king[0]-man[0]+woman[0], king[1]-man[1]+woman[1]]}")
    print(f"  queen = {queen}")
    print("  → They match! The gender direction is consistent.")

    print("\nAnalogy: paris : france :: rome : italy")
    paris = embeddings["paris"]
    france = embeddings["france"]
    rome = embeddings["rome"]
    italy = embeddings["italy"]

    result = [paris[0] - france[0] + rome[0],
              paris[1] - france[1] + rome[1]]
    print(f"  paris - france + rome = {result}")
    print(f"  italy = {italy}")
    print("  → The 'capital-country' direction is consistent!")

analogies()
```

---

## Beyond Words: Sentences and Documents

### Sentence Embeddings

```python

def sentence_embeddings():
    """
    Embedding entire sentences
    """
    print("Sentence Embeddings: Meaning of Phrases")
    print("=" * 60)

    sentences = [
        "The cat sat on the mat",
        "A kitten rested on the rug",
        "The dog chased the ball",
        "I love eating pizza"
    ]

    print("Sentences with similar meaning have similar vectors:")
    print(f"\n  '{sentences[0]}'")
    print(f"  '{sentences[1]}' → SIMILAR (both about cats on surfaces)")
    print(f"  '{sentences[2]}' → LESS similar (dogs, not cats)")
    print(f"  '{sentences[3]}' → DIFFERENT (food topic)")

    print("\nHow it works:")
    print("• Average of word embeddings (simple but works)")
    print("• Pooling from transformer [CLS] token (BERT)")
    print("• Specialized models (Sentence-BERT)")

sentence_embeddings()
```

### Compositionality

```python

def compositionality():
    """
    How meaning combines
    """
    print("Compositionality: Building Complex Meaning")
    print("=" * 60)

    print("""
    Embeddings can capture compositional meaning:

    "good" + "movie" ≈ "good movie" (different from both!)
    "not" + "good" ≈ "bad" (opposite direction!)

    The embedding of a phrase is not just the sum
    of its parts—the transformer creates new
    contextualized representations that capture
    how words modify each other.

    "bank" (river) + "money" context → financial meaning
    "bank" (river) + "water" context → river meaning
    """)

compositionality()
```

---

## A Complete Example

```python

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semantic_demo():
    """
    Complete demo of semantic capture
    """
    print("Complete Example: Semantics in Action")
    print("=" * 60)

    # Simulated 5D embeddings for words
    words = {
        "cat": [0.8, 0.7, 0.2, 0.1, 0.3],
        "dog": [0.8, 0.6, 0.3, 0.1, 0.4],
        "kitten": [0.9, 0.7, 0.2, 0.1, 0.2],
        "puppy": [0.9, 0.6, 0.3, 0.1, 0.3],
        "car": [0.1, 0.2, 0.9, 0.8, 0.1],
        "truck": [0.1, 0.2, 0.9, 0.9, 0.1],
        "apple": [0.2, 0.8, 0.1, 0.2, 0.8],
        "orange": [0.2, 0.7, 0.1, 0.2, 0.9]
    }

    def similarity(w1, w2):
        return cosine_similarity([words[w1]], [words[w2]])[0][0]

    print("Semantic Similarities:")
    print("-" * 40)

    # Animal cluster
    print(f"\nAnimals:")
    print(f"  cat vs dog:    {similarity('cat', 'dog'):.3f}")
    print(f"  cat vs kitten: {similarity('cat', 'kitten'):.3f}")
    print(f"  dog vs puppy:  {similarity('dog', 'puppy'):.3f}")

    # Vehicle cluster
    print(f"\nVehicles:")
    print(f"  car vs truck:  {similarity('car', 'truck'):.3f}")
    print(f"  car vs cat:    {similarity('car', 'cat'):.3f}")

    # Fruit cluster
    print(f"\nFruits:")
    print(f"  apple vs orange: {similarity('apple', 'orange'):.3f}")
    print(f"  apple vs car:    {similarity('apple', 'car'):.3f}")

    print("\n" + "=" * 40)
    print("What this shows:")
    print("• Words in same category have HIGH similarity")
    print("• Words in different categories have LOW similarity")
    print("• The geometry captures semantic relationships")
    print("• This emerges from training, not manual labeling")

semantic_demo()
```

---

## Why This Matters for LLMs

### 1. Contextualized Embeddings

```python

def contextualized():
    """
    Modern LLMs have contextual embeddings
    """
    print("Contextualized Embeddings in LLMs")
    print("=" * 60)

    print("""
    Unlike static word2vec embeddings,
    modern LLMs create DIFFERENT embeddings
    for the same word in different contexts:

    "I went to the bank to deposit money"
           ↓
    bank = [0.8, 0.2, 0.7, ...]  (financial)

    "I sat on the bank of the river"
           ↓
    bank = [0.2, 0.9, 0.3, ...]  (river)

    Each layer of the transformer refines
    the embedding based on surrounding words.
    """)

contextualized()
```

### 2. Semantic Search and RAG

```python

def semantic_search_rag():
    """
    Embeddings enable semantic search
    """
    print("Embeddings Enable RAG and Semantic Search")
    print("=" * 60)

    print("""
    How RAG works:

    1. Convert all documents to embeddings
    2. Store in vector database
    3. Convert user query to embedding
    4. Find nearest document embeddings
    5. Use those documents as context

    This works because embeddings capture
    semantic meaning, not just keywords.

    Query: "cute feline pets"
    Matches: "Kitten care guide" (semantic match)
    Not: "Cute shoes for sale" (keyword match only)
    """)

semantic_search_rag()
```

### 3. Multilingual Semantics

```python

def multilingual():
    """
    Cross-lingual semantic alignment
    """
    print("Multilingual Embeddings")
    print("=" * 60)

    print("""
    In multilingual models, embeddings align across languages:

    English: "cat" → [0.8, 0.7, 0.2]
    French:  "chat" → [0.8, 0.7, 0.2]  (almost identical!)
    German:  "Katze" → [0.8, 0.7, 0.2]

    Japanese: "猫" → [0.8, 0.7, 0.2]  (same meaning, same vector!)

    This enables:
    • Cross-lingual transfer learning
    • Translation without parallel data
    • Multilingual semantic search
    """)

multilingual()
```

---

## Semantic Capture Cheat Sheet

| Aspect                    | How It Captures Meaning                        |
| ------------------------- | ---------------------------------------------- |
| Distributional hypothesis | Words in similar contexts have similar vectors |
| Clustering                | Related words form neighborhoods               |
| Directions                | Relationships become vector arithmetic         |
| Composition               | Phrases combine word meanings                  |
| Contextualization         | Same word changes meaning based on context     |
| Cross-lingual             | Same meaning aligns across languages           |

---

## Quick Recap

• Embeddings capture semantics through the distributional hypothesis—words that appear in similar contexts get similar vectors, purely from observing usage patterns, not manual labeling

• Semantic relationships become geometric—clusters form for categories (animals, vehicles), directions encode relationships (gender, plural), and analogies become vector arithmetic

• Modern LLMs create contextualized embeddings—the same word gets different vectors in different contexts (bank as river vs financial), capturing nuance and polysemy

---

## Mental Hook

> "Embeddings capture meaning like a social network captures people—you understand who someone is by who they hang out with, and words understood by the company they keep, forming neighborhoods of meaning that emerge naturally from billions of conversations."
