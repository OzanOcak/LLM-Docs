# What Are Embeddings? (The "Meaning Space")

## The City Map Analogy

Imagine you're looking at a map of a city. Every location has coordinates—the library is at (2,5), the school at (3,1), the hospital at (7,8). Places with similar purposes are near each other: all restaurants cluster in one area, all schools in another. That's what embeddings do for words—they turn them into coordinates in a "meaning space" where similar words live in the same neighborhood. "King" and "queen" are close together, while "king" and "table" are far apart.

In LLMs, embeddings are the foundation of everything. They convert words, sentences, even entire documents into mathematical vectors that capture meaning, context, and relationships. This is how the model knows that "cat" and "kitten" are related, or that "Paris" is to "France" as "Rome" is to "Italy."

---

## What Is an Embedding?

### The Core Idea

An embedding is a numerical representation of a word (or any object) as a vector in a high-dimensional space.

```text

Word: "king"  →  [0.2, -0.5, 0.8, 0.1, -0.3, 0.6, ...]  (768 numbers)
Word: "queen" →  [0.3, 0.5, 0.7, 0.2, -0.2, 0.5, ...]  (similar numbers)
Word: "table" →  [-0.8, 0.1, -0.3, -0.7, 0.9, -0.4, ...]  (different numbers)
```

```python

def embedding_intro():
    """
    The basic concept of embeddings
    """
    print("Embeddings: Words Become Coordinates")
    print("=" * 60)

    # Simplified embeddings (3 dimensions instead of 768)
    embeddings = {
        "king": [0.8, 0.7, 0.9],
        "queen": [0.9, 0.6, 0.8],
        "prince": [0.7, 0.6, 0.8],
        "apple": [0.1, 0.9, 0.1],
        "orange": [0.2, 0.8, 0.1],
        "car": [0.8, 0.1, 0.2],
        "truck": [0.7, 0.1, 0.3]
    }

    print("Word embeddings (simplified to 3D):")
    for word, vec in embeddings.items():
        print(f"  '{word}': {vec}")

    print("\nNotice: Royalty words cluster together (high first dimension)")
    print("Fruit words cluster together (high second dimension)")
    print("Vehicle words cluster together (low second dimension)")

embedding_intro()
```

---

## The Meaning Space

### Dimensions as Features

Each dimension in an embedding vector captures some aspect of meaning—though not in a way humans can easily label.

```python

def dimensions_as_features():
    """
    What embedding dimensions represent
    """
    print("Dimensions: The Axes of Meaning")
    print("=" * 60)

    print("""
    In a 768-dimensional embedding space:

    Dimension 47 might represent "royalty"
    • Words with high values: king, queen, prince, princess
    • Words with low values: farmer, baker, carpenter

    Dimension 102 might represent "gender"
    • Positive values: masculine (king, man, boy)
    • Negative values: feminine (queen, woman, girl)

    Dimension 83 might represent "size"
    • High: elephant, skyscraper, mountain
    • Low: ant, pin, molecule

    These aren't labeled—they emerge from training!
    """)

dimensions_as_features()
```

### The Analogy Property

```python

def analogy_property():
    """
    How embeddings capture relationships
    """
    print("The Famous Analogy: King - Man + Woman = Queen")
    print("=" * 60)

    # Simplified 2D embeddings
    embeddings = {
        "king": [0.9, 0.8],
        "queen": [0.9, 0.2],
        "man": [0.3, 0.8],
        "woman": [0.3, 0.2]
    }

    print("2D embedding space (dim1 = royalty, dim2 = gender):")
    for word, vec in embeddings.items():
        print(f"  {word}: royalty={vec[0]}, gender={vec[1]}")

    # Vector arithmetic
    king = embeddings["king"]
    man = embeddings["man"]
    woman = embeddings["woman"]

    result = [king[0] - man[0] + woman[0],
              king[1] - man[1] + woman[1]]

    print(f"\nking - man + woman = {result}")
    print("This approximates 'queen'! (0.9, 0.2)")

    print("\nThis works because embeddings capture")
    print("meaning as directions in space.")

analogy_property()
```

---

## From Words to Numbers

### How Embeddings Are Created

```python

def embedding_creation():
    """
    How we get embeddings
    """
    print("How Embeddings Are Created")
    print("=" * 60)

    print("""
    Method 1: Learned during pre-training

    The model starts with random vectors for each word.
    During training, it adjusts them based on context:

    "I ate an apple"
    "I ate a pear"

    The model learns that "apple" and "pear" appear in similar contexts,
    so their embeddings should be similar.

    Method 2: Pre-trained embeddings (Word2Vec, GloVe)

    Trained specifically to capture word relationships,
    then used as input to other models.

    Modern LLMs: Learn embeddings as part of training!
    """)

embedding_creation()
```

### Visualizing Embeddings

```python

def visualization():
    """
    Visualizing high-dimensional spaces
    """
    print("Visualizing the Meaning Space")
    print("=" * 60)

    print("""
    Problem: 768 dimensions is impossible to visualize.

    Solution: Dimensionality reduction (t-SNE, UMAP)

    Before (768D):                                 After (2D):
    [0.2, -0.5, 0.8, 0.1, -0.3, 0.6, ...]  ──→    (2.3, 4.1)

    Words that are close in 768D space
    stay close in the 2D visualization:

              cat ──── kitten
              /                2D map:
             /
        dog ────── puppy          cat •──• kitten
                                 /
                              dog •──• puppy
    """)

visualization()
```

---

## A Tiny Embedding Example

```python

import numpy as np

def embedding_demo():
    """
    Working with embeddings in code
    """
    print("Working with Embeddings in Python")
    print("=" * 60)

    # Simulated pre-trained embeddings (4 dimensions for simplicity)
    embeddings = {
        "king": np.array([0.8, 0.7, 0.9, 0.2]),
        "queen": np.array([0.9, 0.6, 0.8, 0.8]),  # Note last dim different (gender)
        "man": np.array([0.3, 0.8, 0.2, 0.1]),
        "woman": np.array([0.3, 0.2, 0.3, 0.9]),
        "cat": np.array([0.7, 0.1, 0.8, 0.5]),
        "dog": np.array([0.8, 0.1, 0.7, 0.5]),
        "car": np.array([0.1, 0.8, 0.1, 0.3]),
        "truck": np.array([0.2, 0.9, 0.1, 0.4])
    }

    def cosine_similarity(v1, v2):
        """Measure similarity between vectors (-1 to 1)"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    print("Similarity scores (1 = identical, 0 = unrelated, -1 = opposite):")

    # Royalty cluster
    print(f"\nKing vs Queen: {cosine_similarity(embeddings['king'], embeddings['queen']):.3f}")
    print(f"King vs Man:   {cosine_similarity(embeddings['king'], embeddings['man']):.3f}")

    # Animals cluster
    print(f"\nCat vs Dog:    {cosine_similarity(embeddings['cat'], embeddings['dog']):.3f}")
    print(f"Cat vs Car:    {cosine_similarity(embeddings['cat'], embeddings['car']):.3f}")

    # Vehicles cluster
    print(f"\nCar vs Truck:  {cosine_similarity(embeddings['car'], embeddings['truck']):.3f}")
    print(f"Car vs Cat:    {cosine_similarity(embeddings['car'], embeddings['cat']):.3f}")

    print("\nNotice: Similar concepts have HIGHER similarity scores!")
    print("Different concepts have LOWER scores.")

embedding_demo()
```

---

## Types of Embeddings

| Type                  | What It Embeds       | Example                         | Use Case                    |
| --------------------- | -------------------- | ------------------------------- | --------------------------- |
| Word embeddings       | Individual words     | "cat" → [0.2, -0.5, ...]        | Word similarity, analogy    |
| Sentence embeddings   | Whole sentences      | "The cat sat" → [0.3, 0.7, ...] | Semantic search, clustering |
| Document embeddings   | Entire documents     | "War and Peace" → vector        | Document similarity         |
| Token embeddings      | Subword tokens       | "cat" → vector (LLM input)      | Input to transformers       |
| Positional embeddings | Position in sequence | Position 5 → vector             | Adding order information    |
| Multimodal embeddings | Images, audio, etc.  | Image of cat → vector           | Cross-modal search          |

---

## Why This Matters for LLMs

### 1. Embeddings Are the Input

```python

def llm_input():
    """
    How LLMs use embeddings
    """
    print("Embeddings in LLMs")
    print("=" * 60)

    print("""
    Every LLM starts with an embedding layer:

    Input tokens: ["The", "cat", "sat"]
                       ↓       ↓       ↓
    Embedding layer:  [E₁]    [E₂]    [E₃]  (each 768-dim)
                       ↓       ↓       ↓
                  ┌─────────────────────┐
                  │   Transformer       │
                  └─────────────────────┘
                       ↓       ↓       ↓
                  [H₁]    [H₂]    [H₃]  (contextualized)

    The embeddings are refined by each layer,
    but they start as learned representations.
    """)

llm_input()
```

### 2. Semantic Search

```python

def semantic_search():
    """
    Finding meaning, not just keywords
    """
    print("Semantic Search with Embeddings")
    print("=" * 60)

    query = "cute feline pets"

    documents = [
        "Cats make wonderful companions",
        "Dogs are loyal friends",
        "Car maintenance tips",
        "Kitten care for beginners",
        "Best cat food brands"
    ]

    print(f"Query: '{query}'")
    print("\nTraditional keyword search would miss 'kitten' and 'cat'")
    print("Embedding search finds SEMANTIC matches:")
    print("  • Kitten care for beginners (similar to 'cute feline')")
    print("  • Cats make wonderful companions (similar to 'pets')")
    print("  • Best cat food brands")
    print("\nDogs and cars are far away in embedding space.")

semantic_search()
```

### 3. Recommendation Systems

```python

def recommendations():
    """
    Embeddings for recommendations
    """
    print("Embeddings in Recommendation Systems")
    print("=" * 60)

    print("""
    Netflix/movie recommendations:

    User embedding: [0.7, 0.2, 0.8, ...]  (prefers action, sci-fi)

    Movie embeddings:
    "Star Wars":     [0.8, 0.3, 0.7, ...]  (similar → recommend)
    "The Notebook":  [0.1, 0.9, 0.1, ...]  (different → don't)

    Find movies with embeddings closest to user embedding.
    """)

recommendations()
```

### 4. The Geometry of Meaning

```python

def geometry():
    """
    The mathematical structure of meaning space
    """
    print("The Geometry of Meaning")
    print("=" * 60)

    properties = [
        "Similarity = closeness (cosine similarity)",
        "Analogy = vector arithmetic (king - man + woman = queen)",
        "Hierarchy = nested regions (animal → mammal → dog)",
        "Contrast = opposite directions (hot vs cold)",
        "Polysemy = multiple meanings (bank can be near river OR money)"
    ]

    print("Embedding space has structure:")
    for p in properties:
        print(f"  • {p}")

geometry()
```

---

## Embeddings Cheat Sheet

| Aspect             | Details                               |
| ------------------ | ------------------------------------- |
| What it is         | Numerical vector representing meaning |
| Typical dimensions | 384, 768, 1024, 1536                  |
| Similar words      | Have similar vectors (close in space) |
| Relationships      | Captured as directions                |
| Creation           | Learned from context during training  |
| Use in LLMs        | Input layer + refined through network |
| Visualization      | t-SNE, UMAP (for 2D viewing)          |
| Similarity metric  | Cosine similarity most common         |

---

## Quick Recap

• Embeddings turn words into coordinates in a "meaning space"—like a map where similar concepts cluster together, with "king" and "queen" in the royal neighborhood, far from "table" and "chair"

• Relationships between words become geometric relationships—famous analogies like "king - man + woman ≈ queen" work because embeddings capture meaning as directions in space

• Embeddings are the foundation of all LLMs—every token becomes a vector before processing, and the entire transformer architecture is designed to refine these representations through layers of attention

---

## Mental Hook

> "Embeddings are like giving every word its own GPS coordinate in a city of meaning—words with similar meanings live on the same block, related concepts are connected by straight roads, and you can navigate from 'king' to 'queen' by following the 'gender' street."
