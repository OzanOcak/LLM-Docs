# Indexing Algorithms (HNSW, IVF)

## The City Navigation Analogy

Imagine you're in a massive city and need to find the closest coffee shop. You could check every single street (brute force)—exhausting and slow. Or you could use a hierarchical map: first look at the city district (top level), then zoom into the neighborhood, then finally the specific street. That's HNSW. Alternatively, you could organize coffee shops into neighborhoods and only check the ones near you—that's IVF. Both are ways to find "nearest neighbors" without checking every single option.

In vector databases, indexing algorithms are what make similarity search fast. Without them, searching 1 million vectors would mean 1 million distance calculations per query—impossible for real-time apps. HNSW and IVF are the two most popular approaches, each with different tradeoffs between speed, accuracy, and memory.

---

## The Problem: Naive Search Doesn't Scale

### Brute Force Reality Check

```python

def brute_force_problem():
    """
    Why we need indexing algorithms
    """
    print("The Problem: Naive Search Is Too Slow")
    print("=" * 60)

    vector_counts = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    time_per_comparison = 0.000001  # 1 microsecond

    print("Vectors | Comparisons | Time (naive) | Real-time?")
    print("-" * 60)

    for n in vector_counts:
        time_ms = n * time_per_comparison * 1000  # convert to ms
        realtime = "✓" if time_ms < 100 else "✗"
        print(f"{n:7,d} | {n:11,d} | {time_ms:7.2f} ms | {realtime}")

    print("\nFor 1M vectors: 1 second per query ❌")
    print("For 10M vectors: 10 seconds per query ❌")
    print("\nWe need algorithms that find neighbors in milliseconds!")

brute_force_problem()
```

### The Accuracy vs Speed Tradeoff

```python

def accuracy_tradeoff():
    """
    The fundamental tradeoff in indexing
    """
    print("The Accuracy vs Speed Tradeoff")
    print("=" * 60)

    print("""
    Exact search (brute force):
    • 100% accurate
    • O(n) time
    • Too slow for large datasets

    Approximate search (indexing):
    • 90-99% accurate
    • O(log n) or O(1) time
    • Fast enough for real-time

    Indexing algorithms trade a tiny amount of accuracy
    for MASSIVE speed gains—worth it for most applications!
    """)

accuracy_tradeoff()
```

---

## HNSW: Hierarchical Navigable Small World

### What Is HNSW?

HNSW builds a multi-layer graph where each layer is a "zoomed out" view of the data.

```python

def hnsw_intro():
    """
    The basic concept of HNSW
    """
    print("HNSW: Hierarchical Navigable Small World")
    print("=" * 60)

    print("""
    Structure:

    Layer 3 (top):    A ─── B     (very sparse, long jumps)
                       │
    Layer 2:          A ─── C ─── D
                      │    │
    Layer 1:          A ─── C ─── E ─── F
                      │         │
    Layer 0 (bottom): A ─── G ─── H ─── I ─── J  (dense, local)

    Search process:
    1. Start at top layer (long jumps)
    2. Find closest node in current layer
    3. Move down to next layer
    4. Repeat until bottom layer
    5. Refine search locally
    """)

hnsw_intro()
```

### How HNSW Search Works

```python

def hnsw_search():
    """
    Step-by-step HNSW search
    """
    print("HNSW Search: Zooming In")
    print("=" * 60)

    print("""
    Searching for point Q:

    Layer 2 (top):    A ──── B ──── C
                       │
    Start at entry point A
    Find closest: B is closer than A? No, stay at A
    Move down to layer 1

    Layer 1:          A ──── D ──── E
                      │     │
    From A, check neighbors: D is closer
    From D, check neighbors: E is closer
    From E, no better neighbors
    Move down to layer 0

    Layer 0 (bottom): A ──── D ──── E ──── F ──── G
                      │     │     │
    From E, check neighbors: F is closer
    From F, check neighbors: G is closer
    From G, no better neighbors

    Return G as nearest neighbor!
    """)

hnsw_search()
```

### HNSW Parameters

```python

def hnsw_parameters():
    """
    Key parameters for HNSW
    """
    print("HNSW Parameters: Controlling the Graph")
    print("=" * 60)

    print("""
    M = max number of connections per node
    • Higher M = more accurate, more memory
    • Lower M = faster, less memory
    • Typical: 16-64

    efConstruction = build time accuracy
    • Higher = better index quality, slower build
    • Typical: 100-500

    efSearch = search time accuracy
    • Higher = more accurate search, slower queries
    • Typical: 100-500 (can be adjusted per query)

    Tradeoffs:
    M=16, ef=100  → Fast, memory-efficient, 95% accuracy
    M=64, ef=500  → Slower, memory-heavy, 99% accuracy
    """)

hnsw_parameters()
```

### HNSW Pros and Cons

```python

def hnsw_pros_cons():
    """
    Advantages and disadvantages of HNSW
    """
    print("HNSW: Pros and Cons")
    "=" * 60

    pros = [
        "Very fast search (logarithmic time)",
        "Excellent accuracy (often 99%+)",
        "No training phase (incremental)",
        "Works well for any distance metric"
    ]

    cons = [
        "High memory usage (stores graph)",
        "Build time can be slow",
        "Not ideal for billion-scale datasets",
        "More complex to implement"
    ]

    print("\n✅ Pros:")
    for p in pros:
        print(f"  • {p}")

    print("\n❌ Cons:")
    for c in cons:
        print(f"  • {c}")

hnsw_pros_cons()
```

---

## IVF: Inverted File Index

### What Is IVF?

IVF (Inverted File Index) clusters vectors and only searches relevant clusters.

```python

def ivf_intro():
    """
    The basic concept of IVF
    """
    print("IVF: Inverted File Index")
    print("=" * 60)

    print("""
    Structure:

    Step 1: Cluster all vectors (using K-means)

          Cluster 1          Cluster 2          Cluster 3
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │  A  B  C    │    │  D  E  F    │    │  G  H  I    │
        │     J       │    │     K       │    │     L       │
        └─────────────┘    └─────────────┘    └─────────────┘
              ↑                   ↑                   ↑
          Centroid 1          Centroid 2          Centroid 3

    Step 2: For each vector, store which cluster it belongs to

    Step 3: Search only the closest clusters to query
    """)

ivf_intro()
```

### How IVF Search Works

```python

def ivf_search():
    """
    Step-by-step IVF search
    """
    print("IVF Search: Check Neighborhoods First")
    print("=" * 60)

    print("""
    Searching for point Q:

    1. Find closest centroids
       Q → distances to centroids: C1: 0.2, C2: 0.8, C3: 1.5
       Select nprobe closest clusters (e.g., nprobe=2 → C1, C2)

    2. Search only within selected clusters
       Cluster 1: 100 vectors
       Cluster 2: 150 vectors
       Total to check: 250 instead of 10,000!

    3. Return nearest neighbors from these candidates

    nprobe parameter:
    • nprobe=1 → fastest, lowest accuracy
    • nprobe=10 → slower, higher accuracy
    • nprobe=100 → almost brute force
    """)

ivf_search()
```

### IVF Parameters

```python

def ivf_parameters():
    """
    Key parameters for IVF
    """
    print("IVF Parameters: Controlling Clusters")
    print("=" * 60)

    print("""
    nlist = number of clusters
    • Typical: sqrt(n) or 4*sqrt(n)
    • For 1M vectors: nlist = 1000-4000
    • More clusters = more accurate, slower training

    nprobe = number of clusters to search
    • Typical: 1-100
    • Higher = more accurate, slower queries
    • Lower = faster, less accurate

    Example with 1M vectors, nlist=1000:
    • nprobe=1 → search 1,000 vectors (0.1% of data)
    • nprobe=10 → search 10,000 vectors (1% of data)
    • nprobe=100 → search 100,000 vectors (10% of data)
    """)

ivf_parameters()
```

### IVF Pros and Cons

```python

def ivf_pros_cons():
    """
    Advantages and disadvantages of IVF
    """
    print("IVF: Pros and Cons")
    "=" * 60

    pros = [
        "Memory efficient (only stores cluster IDs)",
        "Fast training (K-means is well-understood)",
        "Scales to billions of vectors",
        "Simple to implement and tune"
    ]

    cons = [
        "Less accurate than HNSW for same recall",
        "Requires training phase (offline)",
        "Performance depends on cluster quality",
        "Adding vectors requires re-clustering or partial updates"
    ]

    print("\n✅ Pros:")
    for p in pros:
        print(f"  • {p}")

    print("\n❌ Cons:")
    for c in cons:
        print(f"  • {c}")

ivf_pros_cons()
```

---

## HNSW vs IVF: Comparison

### Side-by-Side Comparison

| Aspect                    | HNSW                 | IVF                  |
| ------------------------- | -------------------- | -------------------- |
| Search speed              | ⭐⭐⭐ (logarithmic) | ⭐⭐ (sub-linear)    |
| Accuracy                  | ⭐⭐⭐ (often 99%+)  | ⭐⭐ (90-98%)        |
| Memory usage              | ⭐ (high)            | ⭐⭐⭐ (low)         |
| Build time                | ⭐⭐ (moderate)      | ⭐ (fast training)   |
| Incremental updates       | ⭐⭐⭐ (yes)         | ⭐ (re-train needed) |
| Scalability               | ⭐⭐ (memory bound)  | ⭐⭐⭐ (billions)    |
| Implementation complexity | ⭐⭐                 | ⭐⭐⭐ (simpler)     |

### When to Choose Which

```python

def choose_algorithm():
    """
    Decision guide for choosing between HNSW and IVF
    """
    print("Choosing Between HNSW and IVF")
    "=" * 60

    scenarios = [
        ("Real-time search, high accuracy needed",
         "HNSW",
         "Best accuracy/speed tradeoff"),

        ("Billion-scale datasets",
         "IVF",
         "Memory efficient, scales better"),

        ("Frequently adding new vectors",
         "HNSW",
         "Incremental updates, no re-training"),

        ("Memory-constrained environment",
         "IVF",
         "Lower memory footprint"),

        ("Need 99%+ recall",
         "HNSW",
         "IVF typically caps around 95-98%"),

        ("Simple implementation",
         "IVF",
         "Easier to understand and tune")
    ]

    for scenario, choice, reason in scenarios:
        print(f"\n  Scenario: {scenario}")
        print(f"  → Choose: {choice}")
        print(f"  → Why: {reason}")

choose_algorithm()
```

---

## Practical Example

```python

import numpy as np
from sklearn.cluster import KMeans
import faiss  # Facebook AI Similarity Search (popular library)

def indexing_demo():
    """
    Simplified demo of indexing concepts
    """
    print("Indexing Algorithms Demo")
    print("=" * 60)

    # Generate random data
    np.random.seed(42)
    n_vectors = 10000
    d = 128  # dimension

    data = np.random.random((n_vectors, d)).astype('float32')
    query = np.random.random((1, d)).astype('float32')

    print(f"Dataset: {n_vectors} vectors, {d} dimensions")

    # Brute force (exact search)
    print("\n1. Brute Force (exact):")
    import time
    start = time.time()
    distances = np.linalg.norm(data - query, axis=1)
    idx = np.argmin(distances)
    brute_time = (time.time() - start) * 1000
    print(f"   Found nearest neighbor in {brute_time:.2f} ms")

    # IVF concept demo
    print("\n2. IVF Concept (simulated):")
    n_clusters = 100
    print(f"   Creating {n_clusters} clusters...")

    # Simulate clustering
    clusters = np.random.randint(0, n_clusters, n_vectors)

    # Find which clusters to search
    nprobe = 5
    print(f"   Searching only {nprobe} closest clusters")
    print(f"   Vectors to check: {n_vectors/n_clusters * nprobe:.0f} instead of {n_vectors}")
    print(f"   Speedup: {n_vectors / (n_vectors/n_clusters * nprobe):.1f}x")

    # HNSW concept demo
    print("\n3. HNSW Concept (simulated):")
    layers = [10000, 1000, 100, 10]  # Decreasing size
    print(f"   Multi-layer graph with {len(layers)} layers")
    print(f"   Top layer: {layers[-1]} nodes (long jumps)")
    print(f"   Bottom layer: {layers[0]} nodes (local detail)")
    print(f"   Search time: O(log {n_vectors}) instead of O({n_vectors})")

indexing_demo()
```

---

## Real-World Vector Databases Implementation

### FAISS (Facebook AI Similarity Search)

```python

def faiss_example():
    """
    FAISS implementation examples
    """
    print("FAISS: Industry Standard for Vector Search")
    print("=" * 60)

    print("""
    FAISS supports multiple index types:

    IndexFlatL2: Brute force (exact)
    • Use for: Small datasets, baseline

    IndexIVFFlat: IVF with flat vectors
    • Use for: Large datasets, memory efficient

    IndexHNSWFlat: HNSW with flat vectors
    • Use for: High accuracy, can fit in RAM

    IndexIVFPQ: IVF + Product Quantization
    • Use for: Billion-scale, memory constrained

    Code example:

    import faiss

    # IVF index
    quantizer = faiss.IndexFlatL2(d)  # how we compare
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(data)
    index.add(data)

    # HNSW index
    index = faiss.IndexHNSWFlat(d, M)
    index.add(data)
    """)

faiss_example()
```

---

## Why This Matters for LLMs

### 1. RAG Performance Depends on Indexing

```python

def rag_indexing():
    """
    How indexing affects RAG applications
    """
    print("Indexing in RAG Applications")
    print("=" * 60)

    print("""
    RAG pipeline speed is determined by:

    1. Query embedding (fast)
    2. Vector search (depends on index)
    3. LLM generation (slowest part)

    With 1M documents:
    • Brute force: +1 second per query → terrible UX
    • HNSW/IVF: +10-50ms per query → seamless

    Good indexing makes RAG practical!
    """)

rag_indexing()
```

### 2. Memory vs Speed Tradeoffs in Production

```python

def production_tradeoffs():
    """
    Real-world deployment considerations
    """
    print("Production Considerations")
    print("=" * 60)

    print("""
    HNSW:
    • Use when: High accuracy needed, enough RAM
    • Example: Product search, semantic caching
    • Memory: 10-20GB for 1M vectors (768d)

    IVF:
    • Use when: Memory constrained, billion-scale
    • Example: Large document collections, web search
    • Memory: 2-5GB for 1M vectors + clusters

    Hybrid approaches:
    • IVF+HNSW (IVF with HNSW for centroids)
    • Product Quantization (compressed vectors)
    """)

production_tradeoffs()
```

### 3. Choosing the Right Index

```python

def final_guide():
    """
    Practical guide for choosing indexes
    """
    print("Quick Guide: Which Index Should You Use?")
    print("=" * 60)

    guide = [
        ("< 10K vectors", "Brute force (IndexFlat)", "Simple, exact"),
        ("10K-100K vectors", "HNSW", "Best accuracy"),
        ("100K-1M vectors", "HNSW or IVF", "Depends on memory"),
        ("1M-10M vectors", "IVF", "Memory efficient"),
        ("> 10M vectors", "IVF+PQ", "Compressed, billion-scale"),
        ("High accuracy needed", "HNSW", "Best recall"),
        ("Memory constrained", "IVF", "Lower footprint"),
        ("Real-time updates", "HNSW", "Incremental")
    ]

    for scenario, index, reason in guide:
        print(f"\n  • {scenario}: {index}")
        print(f"    → {reason}")

final_guide()
```

---

## Indexing Algorithms Cheat Sheet

| Algorithm   | Search Time | Memory   | Accuracy | Updates | Best For                        |
| ----------- | ----------- | -------- | -------- | ------- | ------------------------------- |
| Brute force | O(n)        | Low      | 100%     | Easy    | Tiny datasets, baselines        |
| HNSW        | O(log n)    | High     | 99%+     | Easy    | High accuracy, <1M vectors      |
| IVF         | O(log n)    | Medium   | 95-98%   | Hard    | Large scale, memory constrained |
| IVF+PQ      | O(log n)    | Very low | 90-95%   | Hard    | Billion-scale, approximate      |

---

## Quick Recap

• HNSW builds a multi-layer graph for logarithmic search—like having a city map with zoom levels, you start at the top for long jumps and refine downward, achieving excellent accuracy with fast search

• IVF clusters vectors and only searches relevant clusters—like organizing a city into neighborhoods and only checking the ones near you, trading some accuracy for memory efficiency and scalability

• Choose HNSW for high accuracy and real-time updates, IVF for billion-scale datasets and memory constraints—the right algorithm depends on your data size, accuracy needs, and hardware limits

---

## Mental Hook

> "HNSW is like exploring a city with a zoomable map—you see the whole city from above, zoom into a district, then a street, then finally the exact building. IVF is like having a friend who says 'check these three neighborhoods first'—much faster than checking every house, and usually right."
