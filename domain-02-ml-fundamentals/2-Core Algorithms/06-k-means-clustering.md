# K-means clustering

**DOMAIN: MACHINE LEARNING FUNDAMENTALS | Sub domain: Core Algorithms**

---

**1. Why this concept matters for building intelligent systems**

Not all learning has labels. Sometimes you just have data—millions of images, thousands of customer records, a stream of sensor readings—and you need to discover structure without being told what to look for. K-means clustering is the simplest and most widely used method for unsupervised learning. It finds natural groupings in data by repeatedly assigning points to the nearest cluster center and updating those centers. This simple algorithm underlies image compression, customer segmentation, feature learning, and even some attention mechanisms in modern transformers. Understanding K-means means understanding how machines find patterns when no one provides the answers.

---

**2. Core idea**

**K-means clustering partitions data into K clusters, where each point belongs to the cluster with the nearest mean (centroid), and the algorithm iteratively updates centroids to minimize the sum of squared distances within each cluster.**

---

**3. Concrete analogy**

Imagine you run a delivery service with 100 customer locations on a map. You want to build K = 3 warehouses to minimize driving distance. Where should you put them?

You guess three warehouse locations. You assign each customer to the nearest warehouse. Then you move each warehouse to the average (centroid) of the customers assigned to it. Then reassign customers. Then recompute centroids. Repeat until nothing changes.

After a few iterations, warehouses settle into the natural clusters of customers. This is K-means. The algorithm does not know what "good" means—it only knows to minimize distance. Yet the emergent clusters often correspond to meaningful groupings: downtown, north side, south side.

---

**4. ASCII diagram**

```
K-means iteration steps (K=3):

    Before clustering (unlabeled data)     After convergence

    y ↑                                   y ↑
      │  ×   ×                             │  ×   ×
      │    ×   ×   ×                       │    ×   ×   ×
      │  ×   ×   ×   ×                     │  ┌─────┐×   ×
      │    ×   × ×   ×                     │  │  ×  │ ×   ×
      │  ×   ×   ×                         │  │ ××  │×
      │    ×                               │  │  ×  │
      │        ×   ×                       │  └─────┘  ×   ×
      │          ×   ×                     │           ×   ×
      │            ×   ×                   │             ×   ×
      └──────────────→ x                   └──────────────→ x

    Iteration 0: Random centroids          Iteration 3: Clusters found

    The algorithm:

    1. Initialize K centroids randomly (or cleverly with K-means++)
    2. Assign each point to nearest centroid
    3. Move each centroid to mean of its assigned points
    4. Repeat steps 2-3 until centroids stabilize

    Convergence criteria:
    • Centroids move less than threshold ε
    • Assignments stop changing
    • Maximum iterations reached
```

---

**5. Mathematical formulation**

**Given:** Dataset {x₁, x₂, ..., xₙ} with each x_i ∈ ℝᵈ, and integer K

**Goal:** Partition into K clusters to minimize within-cluster sum of squares:

$$
\min_{S_1, ..., S_K} \sum_{k=1}^K \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2
$$

(Unicode: minimize Σ*k Σ*{x in S_k} ‖x - μ_k‖²)

Where:

- S_k = set of points in cluster k
- μ*k = centroid of cluster k = (1/|S_k|) Σ*{x∈S_k} x

**Distance metric (usually Euclidean):**

$$
\|\mathbf{x} - \boldsymbol{\mu}_k\|^2 = \sum_{j=1}^d (x_j - \mu_{k,j})^2
$$

**Algorithm steps:**

Initialize μ₁, μ₂, ..., μ_K (random or K-means++)

Repeat until convergence:

1. **Assignment step:** Assign each point to nearest centroid

$$
S_k^{(t)} = \{\mathbf{x}_i : \|\mathbf{x}_i - \boldsymbol{\mu}_k^{(t)}\|^2 \leq \|\mathbf{x}_i - \boldsymbol{\mu}_j^{(t)}\|^2 \ \forall j\}
$$

2. **Update step:** Recompute centroids

$$
\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|S_k^{(t)}|} \sum_{\mathbf{x}_i \in S_k^{(t)}} \mathbf{x}_i
$$

**Inertia (within-cluster sum of squares):**

$$
\text{Inertia} = \sum_{k=1}^K \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2
$$

**Choosing K (Elbow method):**

Plot Inertia vs K. Look for "elbow" where marginal decrease slows.

**K-means++ initialization:**

Choose first centroid randomly. For each subsequent centroid, choose point with probability proportional to squared distance to nearest existing centroid. Spreads initial centroids.

---

**6. Worked example (step-by-step)**

**Step 1: Define data points in 1D**

Points: 1, 2, 4, 5. Choose K = 2.

**Step 2: Initialize centroids randomly**

Let μ₁ = 1, μ₂ = 2 (first two points as seeds)

**Step 3: Assignment (Iteration 1)**

Distance to μ₁=1:
Point 1: |1-1|=0 → cluster 1
Point 2: |2-1|=1
Point 4: |4-1|=3
Point 5: |5-1|=4

Distance to μ₂=2:
Point 1: |1-2|=1
Point 2: |2-2|=0 → cluster 2
Point 4: |4-2|=2 → cluster 2
Point 5: |5-2|=3 → cluster 2

Assignments: S₁ = {1}, S₂ = {2, 4, 5}

**Step 4: Update centroids (Iteration 1)**

μ₁ = 1 / 1 = 1

μ₂ = (2 + 4 + 5) / 3 = 11/3 ≈ 3.667

**Step 5: Assignment (Iteration 2)**

Distances to μ₁=1: 1→0 (S₁), 2→1, 4→3, 5→4

Distances to μ₂=3.667: 1→2.667, 2→1.667, 4→0.333 (S₂), 5→1.333 (S₂)

Assignments: S₁ = {1, 2}, S₂ = {4, 5}

**Step 6: Update centroids (Iteration 2)**

μ₁ = (1 + 2) / 2 = 1.5

μ₂ = (4 + 5) / 2 = 4.5

**Step 7: Assignment (Iteration 3)**

Distances to μ₁=1.5: 1→0.5 (S₁), 2→0.5 (S₁), 4→2.5, 5→3.5

Distances to μ₂=4.5: 1→3.5, 2→2.5, 4→0.5 (S₂), 5→0.5 (S₂)

Assignments: S₁ = {1, 2}, S₂ = {4, 5} (same as iteration 2)

**Step 8: Converged**

Final clusters: {1, 2} and {4, 5}. Centroids at 1.5 and 4.5.

Inertia = (1-1.5)² + (2-1.5)² + (4-4.5)² + (5-4.5)² = 0.25+0.25+0.25+0.25 = 1.0

---

**7. How this appears inside neural networks and LLMs**

- **Tokenization preprocessing:** K-means clusters subword units for byte-pair encoding variants. Some tokenizers learn clusters of character sequences.

- **Vector quantization:** Image compression and generative models (VQ-VAE) use K-means-like codebook learning to discretize continuous representations.

- **Feature learning:** Deep clustering methods use neural networks to learn representations, then apply K-means in embedding space to discover semantic clusters.

- **Memory-augmented networks:** Some architectures maintain "memory slots" updated similar to K-means centroids, storing prototypical representations.

- **Attention interpretability:** Clustering attention heads or patterns across layers reveals functional specialization.

- **Dataset summarization:** K-means centroids serve as representative examples for dataset distillation or coreset selection.

- **Initialization for RBF networks:** Centroids from K-means initialize radial basis function centers in shallow neural networks.

- **Explaining embeddings:** Applying K-means to LLM hidden states reveals how the model organizes concepts internally.

---

**8. Brain-like connection (category formation)**

The brain learns categories without explicit labels. Infants develop prototypes for "dog," "cat," "bird" by seeing many examples. The prototype is like a centroid—an average of experienced exemplars. When a new animal appears, the brain compares it to stored prototypes and activates the closest match. This is K-means in the brain, though implemented by neural circuits rather than explicit iteration. The hippocampus and prefrontal cortex work together to update category boundaries as new experiences arrive. Sleep may replay experiences to "recluster" memories, maintaining stable prototypes despite daily variation.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "K-means always finds the optimal clustering."

_Why it is wrong:_ K-means guarantees convergence to a local minimum, not the global minimum. Different random initializations lead to different final clusterings. The algorithm is sensitive to initialization—bad seeds produce bad clusters (e.g., one cluster with one outlier, another with everything else). This is why K-means++ (smarter initialization) exists. Additionally, K-means assumes spherical clusters of similar size and density. Elongated, nested, or varying-density clusters break K-means (it will cut clusters arbitrarily). Knowing these limitations is essential before applying K-means to real data.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Labels are expensive. Unlabeled data is everywhere.      |
|  K-means finds structure when no one tells you what to    |
|  look for. It compresses images, segments customers,      |
|  and initializes deeper models. The algorithm is          |
|  astonishingly simple: assign, update, repeat. Yet from   |
|  this simplicity emerges meaningful groupings that        |
|  reveal the hidden geometry of your data.                 |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

Points in 2D: (0,0), (0,1), (1,0), (10,10), (10,11), (11,10). K = 2.

**Question:** If initialized with centroids at (0,0) and (10,10), will K-means converge to the correct clustering? What will the final clusters be?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Yes, K-means will converge to the correct clustering.

Final clusters will be:
Cluster 1: (0,0), (0,1), (1,0) (the three points in bottom-left)
Cluster 2: (10,10), (10,11), (11,10) (the three points in top-right)

The initialization already placed centroids correctly. Within-cluster distances are small (max ~1.414), between-cluster distances are large (~14.14). Inertia will be low. This is an ideal case for K-means because clusters are well-separated and spherical.
