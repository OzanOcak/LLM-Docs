# Similarity Search Basics

## The Friend Finder Analogy

Imagine you're at a huge party and want to find people who share your interests. You don't go around asking everyone "what are your hobbies?"—that would take forever. Instead, you look for people wearing band t-shirts you like, or standing near topics you enjoy. That's similarity search: finding items that are "close" to your query in some meaningful way, without checking everything.

In vector databases, similarity search is the fundamental operation. You have a query vector (what you're looking for) and a collection of vectors (your data), and you want the ones most similar to your query. This powers semantic search, recommendations, RAG, and countless other AI applications.

---

## What Is Similarity Search?

### The Core Idea

Similarity search finds the vectors in a database that are closest to a query vector, according to some distance metric.

```text

Query: "cute cat photos"
    ↓
Embedding: [0.3, -0.2, 0.8, 0.1, ...]
    ↓
Vector Database:
┌─────────────────────────────────────┐
│ ID  | Vector                       │
│ 1   | [0.8, 0.7, 0.2, 0.1, ...]   │ ← cat photo (similar)
│ 2   | [0.1, 0.2, 0.9, 0.8, ...]   │ ← car photo (different)
│ 3   | [0.7, 0.8, 0.3, 0.2, ...]   │ ← kitten photo (very similar)
│ 4   | [0.2, 0.1, 0.8, 0.9, ...]   │ ← truck photo (different)
└─────────────────────────────────────┘
    ↓
Results: IDs 3, 1 (most similar first)
```

```python

def similarity_intro():
    """
    The basic concept of similarity search
    """
    print("Similarity Search: Finding What's Close")
    print("=" * 60)

    print("""
    Key components:

    1. Query: What you're looking for (as a vector)
    2. Database: Collection of vectors to search
    3. Distance metric: How we measure "closeness"
    4. Results: Top-k most similar items

    It's like "nearest neighbors" but at scale!
    """)

similarity_intro()
```

---

## The Search Process Step by Step

### Step 1: Convert Everything to Vectors

```python

def step1_vectors():
    """
    First step: embedding everything
    """
    print("Step 1: Create Vectors for Everything")
    print("=" * 60)

    print("""
    Documents to search:

    Doc 1: "The cat sat on the mat"
    Doc 2: "Dogs love to play fetch"
    Doc 3: "Cats are independent pets"
    Doc 4: "Cars need regular maintenance"

    ↓ Embed (using sentence transformer)

    Vectors:
    Doc 1: [0.2, -0.5, 0.8, 0.1, 0.3, ...]
    Doc 2: [0.8, 0.3, -0.2, 0.5, -0.1, ...]
    Doc 3: [0.3, -0.4, 0.7, 0.2, 0.4, ...]
    Doc 4: [-0.5, 0.2, 0.1, -0.3, 0.8, ...]
    """)

step1_vectors()
```

### Step 2: Query as Vector

```python

def step2_query():
    """
    Convert query to vector
    """
    print("Step 2: Embed Your Query")
    print("=" * 60)

    query = "cute feline friends"

    print(f"Query: '{query}'")
    print("\n↓ Embed using SAME model")

    query_vector = [0.4, -0.3, 0.9, 0.1, 0.2, ...]
    print(f"Query vector: {query_vector}...")

    print("\nThe query is now in the SAME vector space")
    print("as your documents—ready for comparison!")

step2_query()
```

### Step 3: Compute Similarities

```python

def step3_similarities():
    """
    Calculate how close each document is to query
    """
    print("Step 3: Measure Similarity")
    print("=" * 60)

    # Simulated vectors (simplified to 2D for visualization)
    docs = {
        "Cat article": [0.8, 0.7],
        "Dog article": [0.7, 0.8],
        "Car article": [0.1, 0.2],
        "Cat care": [0.9, 0.6]
    }

    query = [0.8, 0.6]

    import math
    def cosine(v1, v2):
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        return dot / (len1 * len2)

    print(f"Query vector: {query}")
    print("\nSimilarity scores:")

    for name, vec in docs.items():
        sim = cosine(query, vec)
        print(f"  {name:12}: {sim:.3f}")

step3_similarities()
```

### Step 4: Return Top-K

```python

def step4_topk():
    """
    Return the most similar items
    """
    print("Step 4: Return Top-K Results")
    print("=" * 60)

    results = [
        ("Cat care guide", 0.98),
        ("Cat article", 0.95),
        ("Dog article", 0.82),
        ("Car article", 0.23)
    ]

    k = 2
    print(f"Top-{k} results (k={k}):")
    for i, (doc, score) in enumerate(results[:k], 1):
        print(f"  {i}. {doc} (similarity: {score:.2f})")

step4_topk()
```

---

## Types of Similarity Search

### 1. k-Nearest Neighbors (k-NN)

Find the k closest vectors to your query.

```python

def knn_search():
    """
    k-Nearest Neighbors search
    """
    print("k-NN Search: Find the k Closest")
    print("=" * 60)

    print("""
    Query: "Find 5 documents most similar to this"

    Results:
    1. Most similar
    2. Second most similar
    3. Third most similar
    4. Fourth most similar
    5. Fifth most similar

    k is typically 5-100 depending on use case.
    For RAG, k=3-10 is common.
    """)

knn_search()
```

### 2. Range Search

Find all vectors within a certain distance.

```python

def range_search():
    """
    Range search: find everything within a threshold
    """
    print("Range Search: Find All Within a Radius")
    print("=" * 60)

    print("""
    Query: "Find all documents about cats"

    ┌─────────────────────────┐
    │    ○    ○               │
    │       ○○○   ○           │
    │    ○○●○○○     ○         │
    │     ○○○  ○              │
    │      ○                   │
    └─────────────────────────┘
       ↑ Query
       Radius r

    Return ALL vectors within radius r.
    Useful for: deduplication, anomaly detection.
    """)

range_search()
```

### 3. Maximum Inner Product Search (MIPS)

Find vectors with highest dot product (for recommendations).

```python

def mips():
    """
    Maximum Inner Product Search
    """
    print("MIPS: Maximum Inner Product Search")
    print("=" * 60)

    print("""
    Used in recommendation systems:

    User embedding: u
    Item embeddings: i₁, i₂, i₃, ...

    Score = u · i  (dot product)

    Higher dot product = better recommendation.
    Often used when vectors aren't normalized.
    """)

mips()
```

---

## Similarity Search in Practice

### Complete Example

```python

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def similarity_search_demo():
    """
    Complete similarity search example
    """
    print("Complete Similarity Search Demo")
    print("=" * 60)

    # Our vector database (simplified)
    documents = [
        {"id": 1, "text": "The cat sat on the mat", "vec": [0.8, 0.7, 0.2]},
        {"id": 2, "text": "Dogs love to play fetch", "vec": [0.7, 0.8, 0.3]},
        {"id": 3, "text": "Cats are independent pets", "vec": [0.9, 0.6, 0.2]},
        {"id": 4, "text": "Kittens are adorable", "vec": [0.8, 0.5, 0.2]},
        {"id": 5, "text": "Car maintenance tips", "vec": [0.1, 0.2, 0.9]},
        {"id": 6, "text": "Truck repair guide", "vec": [0.2, 0.1, 0.8]},
    ]

    # Query
    query = "cute feline friends"
    query_vec = [0.85, 0.6, 0.2]  # Simulated embedding

    print(f"Query: '{query}'")
    print(f"Query vector: {query_vec}")

    # Search
    def cosine_sim(v1, v2):
        dot = sum(v1[i]*v2[i] for i in range(len(v1)))
        norm1 = sum(x**2 for x in v1)**0.5
        norm2 = sum(x**2 for x in v2)**0.5
        return dot / (norm1 * norm2)

    print("\nSearching...")
    results = []
    for doc in documents:
        sim = cosine_sim(query_vec, doc["vec"])
        results.append((doc, sim))

    # Sort by similarity (highest first)
    results.sort(key=lambda x: -x[1])

    # Show top 3
    print("\nTop 3 results:")
    for i, (doc, sim) in enumerate(results[:3], 1):
        print(f"\n  {i}. [Score: {sim:.3f}]")
        print(f"     {doc['text']}")
        print(f"     Vector: {doc['vec']}")

similarity_search_demo()
```

---

## Similarity Search Parameters

| Parameter             | What It Does                 | Typical Values         | Tradeoff                        |
| --------------------- | ---------------------------- | ---------------------- | ------------------------------- |
| k (number of results) | How many neighbors to return | 3-100                  | Higher k = more context, slower |
| metric                | Distance measure             | cosine, euclidean, dot | Depends on data type            |
| efSearch (HNSW)       | Search effort                | 50-500                 | Higher = more accurate, slower  |
| nprobe (IVF)          | Clusters to search           | 1-100                  | Higher = more accurate, slower  |
| radius (range)        | Maximum distance             | Depends on data        | Tighter = fewer results         |

---

## Why This Matters for LLMs

### 1. RAG (Retrieval-Augmented Generation)

```python

def rag_similarity():
    """
    Similarity search in RAG
    """
    print("Similarity Search in RAG")
    print("=" * 60)

    print("""
    RAG pipeline:

    1. User asks: "What is the capital of France?"

    2. Convert to vector: [0.3, -0.2, 0.8, ...]

    3. Similarity search in document DB:
       Find documents about France, Paris, capitals

    4. Retrieved docs: "Paris is the capital of France..."

    5. LLM generates answer using retrieved context

    Without similarity search, LLM would guess or hallucinate.
    """)

rag_similarity()
```

### 2. Recommendations

```python

def recommendations():
    """
    Similarity search for recommendations
    """
    print("Similarity Search in Recommendations")
    print("=" * 60)

    print("""
    Netflix-style recommendations:

    User watched: "Inception"
        ↓
    Movie embedding: [0.8, 0.3, 0.7, ...]
        ↓
    Find similar movies:
    • "Interstellar" (0.95) ✓
    • "The Matrix" (0.92) ✓
    • "Shrek" (0.45) ✗

    All powered by similarity search!
    """)

recommendations()
```

### 3. Semantic Caching

```python

def semantic_caching():
    """
    Similarity search for caching
    """
    print("Semantic Caching with Similarity Search")
    print("=" * 60)

    print("""
    Instead of exact match caching, use semantic cache:

    Previous query: "What's the capital of France?"
    Response: "Paris"

    New query: "Capital of France?"
        ↓
    Similarity to previous: 0.98
        ↓
    Return cached "Paris" (saves LLM call)

    New query: "What's the weather in Paris?"
        ↓
    Similarity: 0.45
        ↓
    Call LLM (different intent)
    """)

semantic_caching()
```

---

## Similarity Search Cheat Sheet

| Step      | What Happens                 | Example                        |
| --------- | ---------------------------- | ------------------------------ |
| 1. Index  | Store vectors + metadata     | Document vectors in database   |
| 2. Query  | Convert user input to vector | "cute cats" → [0.3, -0.5, ...] |
| 3. Search | Find nearest neighbors       | Top 5 closest vectors          |
| 4. Rank   | Sort by similarity           | Highest score first            |
| 5. Return | Send back results            | Document IDs, texts, scores    |

---

## Quick Recap

• Similarity search finds vectors closest to your query—like finding people at a party who share your interests by looking for those standing near you, without checking everyone

• The process has four steps: embed everything, embed your query, compute similarities, return top results—simple in concept but requires efficient indexing at scale

• Similarity search powers RAG, recommendations, and semantic caching—it's the fundamental operation that connects LLMs to external knowledge, making them more accurate and useful

---

## Mental Hook

> "Similarity search is like having a magical friend who can instantly tell you who at a party shares your taste in music—not by asking everyone, but by sensing who's 'vibing' in the same frequency as you."
