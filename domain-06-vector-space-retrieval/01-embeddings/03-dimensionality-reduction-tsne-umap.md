# Dimensionality Reduction for Visualization (t-SNE, UMAP)

## The 3D Sculpture Shadow Analogy

Imagine you have a beautiful, complex sculpture in a dark room. You can't see it directly, but you can shine a flashlight from different angles and look at its shadows on the wall. Each shadow is a 2D projection that reveals some aspects of the sculpture but loses others. Some angles show the shape clearly, others create a confusing mess. That's dimensionality reduction—taking high-dimensional data (768D embeddings) and projecting it down to 2D or 3D so we humans can see it, while trying to preserve the most important relationships.

In LLMs, dimensionality reduction is how we visualize the "meaning space." We can't see 768 dimensions, but with t-SNE or UMAP, we can create 2D maps where similar words cluster together, revealing the semantic structure the model has learned.

---

## The Problem: We Can't See 768 Dimensions

### Why Visualization Is Hard

```python

def visualization_problem():
    """
    The challenge of high-dimensional data
    """
    print("The Problem: We Live in 3D, Embeddings Live in 768D")
    print("=" * 60)

    print("""
    Human limitations:
    • We can see 2D (paper) and 3D (objects)
    • We cannot see 4D, let alone 768D

    Embedding dimensions:
    • BERT: 768 dimensions
    • GPT-3: 12,288 dimensions
    • Each word is a point in this space

    We need to PROJECT down to 2D/3D
    while preserving the structure!
    """)

    # Analogy
    print("\nAnalogy: Like trying to understand a 3D object")
    print("by looking at its 2D shadow—you lose information,")
    print("but you can still see the basic shape.")

visualization_problem()
```

### What Gets Lost

```python

def what_lost():
    """
    Tradeoffs in dimensionality reduction
    """
    print("What Gets Lost in Translation")
    print("=" * 60)

    print("""
    Original 768D space:           2D projection:
    • 768 axes of meaning           • Only 2 axes
    • Can separate concepts finely  • Must squash many axes together
    • Exact distances matter         • Approximate distances only

    Example:
    In 768D, "cat" and "kitten" are close (0.1 apart)
    "cat" and "tiger" are medium (0.3 apart)
    "cat" and "car" are far (0.8 apart)

    In 2D, we try to preserve these relationships,
    but some distortion is inevitable.
    """)

what_lost()
```

---

## t-SNE: t-Distributed Stochastic Neighbor Embedding

### How t-SNE Works

```python

def tsne_intuition():
    """
    The basic intuition behind t-SNE
    """
    print("t-SNE: Preserving Neighborhoods")
    print("=" * 60)

    print("""
    t-SNE's goal: Keep nearby points together in 2D.

    Step 1: In high-D space, compute "similarities"
           Each point has a probability of picking
           every other point as its neighbor.

    Step 2: In 2D space, do the same thing
           With random starting positions.

    Step 3: Gradually move 2D points until
           the neighbor probabilities match
           the high-D probabilities.

    Result: Points that were neighbors in 768D
            become neighbors in 2D.
    """)

tsne_intuition()
```

### The Perplexity Parameter

```python

def perplexity():
    """
    The most important t-SNE parameter
    """
    print("Perplexity: How Many Neighbors to Consider")
    print("=" * 60)

    print("""
    Perplexity ≈ number of neighbors each point considers

    Low perplexity (5-10):
    • Focuses on very local structure
    • Can create many small clusters
    • May break up meaningful groups

    Medium perplexity (30-50):
    • Balances local and global structure
    • Most common choice
    • Good for word embeddings

    High perplexity (100+):
    • Emphasizes global structure
    • Can merge distinct clusters
    • May lose fine details

    Rule of thumb: Start with 30, adjust based on results.
    """)

perplexity()
```

### t-SNE Visualization Example

```python

def tsne_visualization():
    """
    What t-SNE plots look like
    """
    print("What t-SNE Reveals")
    print("=" * 60)

    print("""
    t-SNE of word embeddings often shows:

                    ┌─────────────────┐
                    │    ANIMALS      │
                    │   cat   dog     │
                    │     rabbit      │
                    │   horse         │
                    └─────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ↓                               ↓
    ┌───────────────┐               ┌───────────────┐
    │   VEHICLES    │               │    FRUIT      │
    │  car   truck  │               │ apple orange  │
    │    bus        │               │  banana       │
    │  motorcycle   │               │   pear        │
    └───────────────┘               └───────────────┘

    Notice: Clear separation between categories!
    """)

tsne_visualization()
```

---

## UMAP: Uniform Manifold Approximation and Projection

### How UMAP Differs

```python

def umap_intuition():
    """
    UMAP: Faster, Better at Global Structure
    """
    print("UMAP: The Modern Alternative")
    print("=" * 60)

    print("""
    UMAP vs t-SNE:

    t-SNE:
    • Preserves local structure well
    • Global structure (far-apart clusters) can be misleading
    • Slow on large datasets
    • Random initialization

    UMAP:
    • Preserves local AND global structure better
    • Much faster (especially on large data)
    • More deterministic
    • Better at showing continuous patterns

    Both are useful, UMAP is often preferred now
    for large embedding visualizations.
    """)

umap_intuition()
```

### Key UMAP Parameters

```python

def umap_parameters():
    """
    UMAP's main controls
    """
    print("UMAP Parameters: Controlling the View")
    print("=" * 60)

    print("""
    n_neighbors: How many neighbors to consider
    • Small (5-15): Focus on very local structure
    • Large (50-200): Capture broader patterns

    min_dist: How tightly to pack points
    • Small (0.1): Points cluster densely
    • Large (0.5): Points spread out more

    n_components: Output dimensions
    • 2: For visualization
    • 3: For interactive 3D plots

    Metric: Distance measure (cosine, euclidean)
    • Cosine: Good for word embeddings
    • Euclidean: Good for continuous data
    """)

umap_parameters()
```

---

## t-SNE vs UMAP Comparison

| Aspect           | t-SNE                       | UMAP                       |
| ---------------- | --------------------------- | -------------------------- |
| Speed            | Slow (O(n²))                | Fast (O(n log n))          |
| Global structure | Poor (can mislead)          | Good                       |
| Local structure  | Excellent                   | Excellent                  |
| Deterministic    | No (varies between runs)    | More stable                |
| Parameters       | Perplexity                  | n_neighbors, min_dist      |
| Best for         | Small datasets, exploration | Large datasets, production |
| Memory usage     | High                        | Moderate                   |

### When to Use Which

```python

def when_to_use():
    """
    Choosing the right tool
    """
    print("t-SNE vs UMAP: When to Use Each")
    print("=" * 60)

    scenarios = {
        "Small dataset (<5000 points)": "Either works, t-SNE is classic",
        "Large dataset (>50,000 points)": "UMAP (t-SNE too slow)",
        "Exploring local clusters": "t-SNE (emphasizes neighborhoods)",
        "Understanding global structure": "UMAP (better at large-scale patterns)",
        "Research visualization": "Both, compare results",
        "Production pipeline": "UMAP (faster, deterministic)"
    }

    for scenario, choice in scenarios.items():
        print(f"  • {scenario}: {choice}")

when_to_use()
```

---

## A Practical Example

```python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def dimensionality_reduction_demo():
    """
    Simplified example of t-SNE and UMAP
    """
    print("Dimensionality Reduction in Action")
    print("=" * 60)

    # Generate synthetic word embeddings (100 words, 50 dimensions)
    np.random.seed(42)
    n_words = 100
    n_dims = 50

    # Create 3 clusters of words
    embeddings = []
    labels = []

    # Cluster 1: Animals (30 words)
    center1 = np.random.randn(n_dims) * 2
    for i in range(30):
        vec = center1 + np.random.randn(n_dims) * 0.5
        embeddings.append(vec)
        labels.append("animals")

    # Cluster 2: Vehicles (30 words)
    center2 = np.random.randn(n_dims) * 2 + 5
    for i in range(30):
        vec = center2 + np.random.randn(n_dims) * 0.5
        embeddings.append(vec)
        labels.append("vehicles")

    # Cluster 3: Fruits (40 words)
    center3 = np.random.randn(n_dims) * 2 - 3
    for i in range(40):
        vec = center3 + np.random.randn(n_dims) * 0.5
        embeddings.append(vec)
        labels.append("fruits")

    embeddings = np.array(embeddings)

    print(f"Original data: {n_words} points in {n_dims} dimensions")

    # t-SNE
    print("\nRunning t-SNE (simulated)...")
    print("  • Perplexity = 30")
    print("  • Iterations = 1000")
    print("  • Result: 2D projection preserving neighborhoods")

    # UMAP
    print("\nRunning UMAP (simulated)...")
    print("  • n_neighbors = 15")
    print("  • min_dist = 0.1")
    print("  • Result: 2D projection with better global structure")

    print("\n" + "=" * 40)
    print("What the visualizations would show:")
    print("""
    t-SNE view:                          UMAP view:
    ┌─────────────────┐                  ┌─────────────────┐
    │   Animals  ••   │                  │ Animals  ••     │
    │        ••       │                  │      ••    Fruits│
    │  Vehicles ••    │                  │Vehicles ••  ••  │
    │        ••       │                  │    ••      ••   │
    │    Fruits ••    │                  │  ••  Fruits     │
    └─────────────────┘                  └─────────────────┘

    t-SNE: Clusters clearly separated, but distances between
           clusters are meaningless (they could be rearranged).

    UMAP: Clusters separated AND relative positions preserved
           (animals closer to fruits than to vehicles, if true in high-D).
    """)

dimensionality_reduction_demo()
```

---

## Why This Matters for LLMs

### 1. Understanding What Models Learn

```python

def understanding_models():
    """
    Using visualization to understand LLMs
    """
    print("Visualization Helps Us Understand LLMs")
    print("=" * 60)

    insights = [
        "See semantic categories emerge",
        "Detect bias (gender, race clustering)",
        "Identify outliers and unusual words",
        "Track how representations change through layers",
        "Compare different models",
        "Debug embedding quality"
    ]

    print("Researchers use t-SNE/UMAP to:")
    for insight in insights:
        print(f"  • {insight}")

understanding_models()
```

### 2. Famous Visualizations

```python

def famous_viz():
    """
    Classic embedding visualizations
    """
    print("Famous Embedding Visualizations")
    print("=" * 60)

    print("""
    Word2Vec countries + capitals:

                    China
                      Beijing
                        └── Japan
                           Tokyo
    Germany                 └─── Korea
      Berlin                     Seoul
        └── France
           Paris
              └── Italy
                  Rome

    The vector (country → capital) is consistent!
    """)

famous_viz()
```

### 3. Layer-by-Layer Evolution

```python

def layer_evolution():
    """
    How representations change through layers
    """
    print("Visualizing Layer-by-Layer Evolution")
    print("=" * 60)

    print("""
    In a 12-layer BERT model:

    Layer 1:                    Layer 6:                    Layer 12:
    ┌─────────┐                 ┌─────────┐                 ┌─────────┐
    │  The    │                 │  The    │                 │  The    │
    │  cat    │  (mostly        │  cat    │  (semantic      │  cat    │
    │  sat    │   syntax)        │  sat    │   clusters      │  sat    │
    │  on     │                 │  on     │   forming)       │  on     │
    │  the    │                 │  the    │                 │  the    │
    │  mat    │                 │  mat    │                 │  mat    │
    └─────────┘                 └─────────┘                 └─────────┘

    t-SNE of each layer shows:
    • Early layers: words cluster by syntax (nouns together)
    • Middle layers: semantic clusters emerge
    • Late layers: task-specific patterns
    """)

layer_evolution()
```

---

## Practical Tips for Visualization

```python

def practical_tips():
    """
    Tips for creating good visualizations
    """
    print("Practical Tips for Embedding Visualization")
    print("=" * 60)

    tips = [
        "Start with UMAP (faster, better global structure)",
        "Try multiple perplexity/n_neighbors values",
        "Color by known categories (if available)",
        "Label interesting points",
        "Interactive plots help (plotly, bokeh)",
        "Don't overinterpret distances between far clusters",
        "Compare with random projections as baseline"
    ]

    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")

practical_tips()
```

---

## Dimensionality Reduction Cheat Sheet

| Technique         | Best For                     | Speed     | Parameters            | Caveats                      |
| ----------------- | ---------------------------- | --------- | --------------------- | ---------------------------- |
| t-SNE             | Local structure, small data  | Slow      | perplexity            | Global distances meaningless |
| UMAP              | Local + global, large data   | Fast      | n_neighbors, min_dist | Newer, less studied          |
| PCA               | Linear structure, first look | Very fast | n_components          | Misses non-linear patterns   |
| Random Projection | Baseline comparison          | Very fast | n_components          | Just random                  |

---

## Quick Recap

• Dimensionality reduction projects 768D embeddings down to 2D for visualization—like casting shadows of a complex sculpture to understand its shape, losing detail but revealing structure

• t-SNE preserves local neighborhoods well but can misrepresent global distances—great for seeing clusters, but distances between far-apart clusters are meaningless

• UMAP is faster and preserves both local and global structure better—making it the modern choice for visualizing large embedding spaces in LLMs

---

## Mental Hook

> "Dimensionality reduction is like creating a 2D map of a vast 768-dimensional universe—we can't see the real thing, but with techniques like t-SNE and UMAP, we can create a flattening that reveals the continents and islands of meaning."
