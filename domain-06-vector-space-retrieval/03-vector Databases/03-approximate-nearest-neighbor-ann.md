# Approximate Nearest Neighbor (ANN) Search

## The Restaurant Recommender Analogy

Imagine you're in a huge city with 10,000 restaurants and want the 5 closest to your hotel. You could measure the exact distance to every single restaurant—exhausting and slow. Or you could ask a local: "Just give me some good restaurants nearby." They might miss the absolute closest, but they'll give you great options in seconds. That's Approximate Nearest Neighbor (ANN) search: trading a tiny amount of accuracy for MASSIVE speed gains, finding "good enough" neighbors almost instantly.

In vector databases, ANN is the magic that makes real-time similarity search possible. Instead of comparing a query to every single vector (which would take seconds or minutes), ANN algorithms find most of the nearest neighbors in milliseconds. This is what powers every RAG application, recommendation system, and semantic search you've ever used.

---

## What Is Approximate Nearest Neighbor Search?

### The Core Idea

ANN search finds vectors that are approximately closest to a query, not guaranteed to be the absolute closest.

````text

Exact Search (KNN):                    Approximate Search (ANN):
    Compare to ALL vectors                 Use index to guess
    Guarantee: 100% accurate                Guarantee: 95% accurate
    Time: O(n)                              Time: O(log n) or O(1)

    For 1M vectors:                         For 1M vectors:
    1 second per query                       10 milliseconds
    Too slow for real-time                   100 queries/second!
```

```python

def ann_intro():
    """
    The basic concept of ANN search
    """
    print("ANN: Good Enough, Fast Enough")
    print("=" * 60)

    print("""
    The tradeoff:

    ┌─────────────────────────────────────┐
    │                                     │
    │   Exact      ●                      │
    │   Search     │  Accuracy = 100%     │
    │              │  Speed = 1x          │
    │              │                      │
    │   ANN        ●───────┐              │
    │   Search     │       │ Accuracy = 95%│
    │              │       │ Speed = 100x │
    │              ▼       ▼              │
    │         You trade 5% accuracy       │
    │         for 100x speed!             │
    └─────────────────────────────────────┘
    """)

ann_intro()
````

---

## Why Exact Search Is Impractical

### The Reality of Scale

```python

def scale_reality():
    """
    Why exact search fails at scale
    """
    print("The Scale Problem: Exact Search Doesn't Scale")
    print("=" * 60)

    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    time_per_compare = 0.000001  # 1 microsecond

    print("Vectors  | Comparisons | Time (exact) | Real-time?")
    print("-" * 65)

    for n in sizes:
        time_s = n * time_per_compare
        realtime = "✓" if time_s < 0.1 else "✗"
        print(f"{n:9,d} | {n:11,d} | {time_s:7.3f} s | {realtime}")

    print("\nFor 10M vectors: 10 seconds per query!")
    print("For 100M vectors: 1.6 minutes per query!")
    print("\nReal-time applications need <100ms → ANN is essential.")

scale_reality()
```

### The Curse of Dimensionality

```python

def curse_dimensions():
    """
    Why high dimensions make it worse
    """
    print("The Curse of Dimensionality")
    print("=" * 60)

    print("""
    In high dimensions (768D for embeddings):

    • All points are far apart
    • Distances become less meaningful
    • Indexing gets harder

    But ANN algorithms are designed for this:
    • HNSW works well in high dimensions
    • IVF with product quantization helps
    • Still get good results despite the curse
    """)

curse_dimensions()
```

---

## How ANN Achieves Speed

### The Three Main Strategies

```python

def ann_strategies():
    """
    The three main approaches to ANN
    """
    print("ANN Strategies: How We Cheat for Speed")
    "=" * 60

    strategies = {
        "Graph-based (HNSW)": {
            "how": "Build a graph where each node connects to similar nodes",
            "search": "Navigate the graph from node to node",
            "speed": "O(log n)",
            "accuracy": "Very high (98-99%)"
        },
        "Clustering-based (IVF)": {
            "how": "Group vectors into clusters",
            "search": "Only search closest clusters",
            "speed": "O(log n) with clusters",
            "accuracy": "High (95-98%)"
        },
        "Hashing-based (LSH)": {
            "how": "Hash similar vectors to same buckets",
            "search": "Only search same bucket",
            "speed": "O(1)",
            "accuracy": "Lower (90-95%)"
        }
    }

    for name, details in strategies.items():
        print(f"\n  {name}:")
        for key, value in details.items():
            print(f"    • {key}: {value}")

ann_strategies()
```

### Visualizing the Search Space

```text

Exact search (check everything):
[██████████████████████████████] 100% of data

ANN graph search (HNSW):
    Layer 2: [██] (2% of data)
        ↓
    Layer 1: [████] (4% of data)
        ↓
    Layer 0: [██████] (6% of data)

Total: ~12% of data, 90+% accuracy!

ANN cluster search (IVF):
    Cluster 1: [██]
    Cluster 2: [██] ← search these 2 only
    Cluster 3: [  ]  (nprobe=2)
    Cluster 4: [  ]
    Cluster 5: [  ]

Total: 2/10 clusters = 20% of data, 95% accuracy!
```

---

## Measuring ANN Quality

### Recall@K

```python

def recall_metric():
    """
    How we measure ANN accuracy
    """
    print("Recall@K: The ANN Quality Metric")
    print("=" * 60)

    print("""
    Recall@K = (number of true nearest neighbors found) / K

    Example with K=5:

    True nearest neighbors (exact search): [A, B, C, D, E]
    ANN returns:                            [A, B, C, F, G]

    Found 3 of the true 5 → Recall@5 = 3/5 = 60%

    Good ANN aims for 95%+ recall at K=10.
    """)

recall_metric()
```

### Speed-Recall Tradeoff

```python

def speed_recall():
    """
    The fundamental tradeoff
    """
    print("Speed vs Recall: You Choose")
    print("=" * 60)

    import matplotlib.pyplot as plt
    import numpy as np

    # Simulated tradeoff curve
    speeds = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    recalls = [100, 99.5, 99, 98, 96, 93, 88, 80, 65, 40]

    print("Speedup | Recall | Quality")
    print("-" * 30)

    for s, r in zip(speeds[:8], recalls[:8]):
        stars = "★" * int(r/20)
        print(f"   {s:3d}x   |  {r:5.1f}% | {stars}")

    print("\nYou can tune parameters to get the tradeoff you need:")
    print("• HNSW: adjust efSearch")
    print("• IVF: adjust nprobe")
    print("• Higher recall = slower queries")
    print("• Lower recall = faster queries")

speed_recall()
```

---

## ANN in Practice

### HNSW Parameters for Recall/Speed

```python

def hnsw_tuning():
    """
    Tuning HNSW for different needs
    """
    print("Tuning HNSW: efSearch Controls the Tradeoff")
    print("=" * 60)

    print("""
    efSearch (number of candidates to check):

    efSearch=10:  Fast, lower recall
    • Search time: 2ms
    • Recall@10: 85%

    efSearch=50:  Balanced
    • Search time: 8ms
    • Recall@10: 95%

    efSearch=200: Accurate, slower
    • Search time: 30ms
    • Recall@10: 99%

    efSearch=500: Almost exact
    • Search time: 70ms
    • Recall@10: 99.5%
    """)

hnsw_tuning()
```

### IVF Parameters for Recall/Speed

```python

def ivf_tuning():
    """
    Tuning IVF with nprobe
    """
    print("Tuning IVF: nprobe Controls the Tradeoff")
    print("=" * 60)

    print("""
    nprobe (number of clusters to search):

    nprobe=1:  Fastest, lowest recall
    • Search 1/1000 of data
    • Recall@10: 80%

    nprobe=10: Good balance
    • Search 1% of data
    • Recall@10: 94%

    nprobe=50: More accurate
    • Search 5% of data
    • Recall@10: 98%

    nprobe=100: Almost exhaustive
    • Search 10% of data
    • Recall@10: 99%
    """)

ivf_tuning()
```

---

## Real-World ANN Example

```python

import numpy as np
import time

def ann_demo():
    """
    Demonstrate ANN vs exact search
    """
    print("ANN vs Exact Search: Live Demo")
    print("=" * 60)

    # Generate random dataset
    np.random.seed(42)
    n_vectors = 100000
    d = 128

    data = np.random.random((n_vectors, d)).astype('float32')
    query = np.random.random((1, d)).astype('float32')

    print(f"Dataset: {n_vectors:,} vectors, {d} dimensions")

    # Exact search (simulated time)
    exact_time = n_vectors * 0.000001  # 1 microsecond per comparison
    print(f"\nExact search would take: {exact_time*1000:.2f} ms")

    # Simulate different ANN approaches
    ann_methods = [
        {"name": "HNSW (ef=50)", "speedup": 100, "recall": 0.95},
        {"name": "HNSW (ef=200)", "speedup": 30, "recall": 0.99},
        {"name": "IVF (nprobe=10)", "speedup": 80, "recall": 0.94},
        {"name": "IVF (nprobe=50)", "speedup": 20, "recall": 0.98},
    ]

    print("\nANN Options:")
    for method in ann_methods:
        ann_time = exact_time / method["speedup"] * 1000
        print(f"\n  {method['name']}:")
        print(f"    • Time: {ann_time:.2f} ms ({method['speedup']}x faster)")
        print(f"    • Recall: {method['recall']*100:.0f}%")

    print("\n" + "=" * 60)
    print("In production, you'd choose based on needs:")
    print("• High traffic, less accuracy critical → fast settings")
    print("• Quality critical, lower traffic → accurate settings")

ann_demo()
```

---

## ANN vs Exact Search: Complete Comparison

| Aspect      | Exact (KNN)           | Approximate (ANN)       |
| ----------- | --------------------- | ----------------------- |
| Accuracy    | 100%                  | 90-99%                  |
| Speed       | O(n)                  | O(log n) or O(1)        |
| 1M vectors  | 1 second              | 10-50 ms                |
| Memory      | Low (no index)        | Higher (index overhead) |
| Index build | None                  | Minutes to hours        |
| Use case    | Baselines, small data | Production, large scale |

### When to Use Each

```python

def when_to_use():
    """
    Decision guide for exact vs ANN
    """
    print("Exact vs ANN: When to Use")
    print("=" * 60)

    scenarios = [
        ("< 10K vectors", "Exact", "Brute force is fast enough"),
        ("10K-100K vectors", "Either", "Depends on latency needs"),
        ("> 100K vectors", "ANN", "Exact becomes too slow"),
        ("Offline batch processing", "Exact", "Can wait for accuracy"),
        ("Real-time user queries", "ANN", "Must be fast"),
        ("Benchmarking", "Exact", "Get ground truth"),
        ("Production RAG", "ANN", "Sub-100ms required")
    ]

    for scenario, choice, reason in scenarios:
        print(f"\n  • {scenario}: {choice}")
        print(f"    → {reason}")

when_to_use()
```

---

## Why This Matters for LLMs

### 1. RAG Would Be Impossible Without ANN

```python

def rag_ann():
    """
    ANN enables RAG applications
    """
    print("ANN Makes RAG Possible")
    print("=" * 60)

    print("""
    Typical RAG pipeline:

    User Query → Embed (50ms) → Vector Search → LLM Generation

    With exact search on 1M docs:
    • Vector search: 1000ms
    • Total: >1 second → user abandons

    With ANN search:
    • Vector search: 20ms
    • Total: <300ms → feels instant

    ANN is what makes RAG practical for real users!
    """)

rag_ann()
```

### 2. Scaling to Millions/Billions of Documents

```python

def scaling():
    """
    ANN enables web-scale search
    """
    print("ANN Enables Web-Scale Search")
    print("=" * 60)

    scales = {
        "Startup": "100K docs → IVF or HNSW",
        "Medium": "1M docs → HNSW (if RAM available)",
        "Large": "10M docs → IVF with PQ",
        "Web scale": "100M-1B docs → IVF+PQ, distributed"
    }

    for scale, solution in scales.items():
        print(f"  • {scale}: {solution}")

scaling()
```

### 3. Real-World ANN Libraries

```python

def ann_libraries():
    """
    Popular ANN libraries
    """
    print("Popular ANN Libraries")
    print("=" * 60)

    libraries = {
        "FAISS": "Facebook - IVF, HNSW, PQ - Industry standard",
        "Annoy": "Spotify - Tree-based - Memory efficient",
        "HNSWlib": "Pure HNSW implementation - Very fast",
        "ScaNN": "Google - State-of-the-art speed/accuracy",
        "NMSLIB": "Non-Metric Space Library - Many algorithms"
    }

    for lib, desc in libraries.items():
        print(f"  • {lib}: {desc}")

ann_libraries()
```

---

## ANN Cheat Sheet

| Algorithm | Type             | Speed  | Accuracy | Best For                      |
| --------- | ---------------- | ------ | -------- | ----------------------------- |
| HNSW      | Graph            | ⭐⭐⭐ | ⭐⭐⭐   | High accuracy, <1M            |
| IVF       | Cluster          | ⭐⭐   | ⭐⭐     | Large scale, memory efficient |
| IVF+PQ    | Cluster+compress | ⭐⭐   | ⭐⭐     | Billion-scale, compressed     |
| LSH       | Hash             | ⭐⭐⭐ | ⭐       | Very fast, lower accuracy     |
| Annoy     | Tree             | ⭐⭐   | ⭐⭐     | Read-only, memory mapped      |

---

## Quick Recap

• ANN search trades a small amount of accuracy for massive speed gains—like asking a local for restaurant recommendations instead of checking every restaurant, getting great options in a fraction of the time

• Recall@K measures ANN quality—what percentage of the true nearest neighbors were found, typically 95%+ is considered good

• ANN is what makes RAG and real-time vector search possible—without it, searching millions of documents would take seconds, not milliseconds, making production LLM applications impractical

---

## Mental Hook

> "ANN search is like having a friend who knows the city—they won't find the absolute closest coffee shop to you, but they'll find one 95% as close in 1% of the time, and you'll be drinking your latte while everyone else is still measuring distances."
