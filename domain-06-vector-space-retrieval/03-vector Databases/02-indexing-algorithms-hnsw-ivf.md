# Indexing algorithms (HNSW, IVF)

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Vector Databases: The Collective Memory**

---

### **1. Why this concept matters**

A vector database with 100 million vectors and no index is useless. A query would scan every vector—100 million distance computations, each 768 dimensions. That is 76 billion operations. One second per query? More like ten minutes. Indexing solves this by organizing vectors into structures that enable sub-linear search. HNSW (Hierarchical Navigable Small World) and IVF (Inverted File Index) are the two most popular algorithms. They reduce search time from O(n) to O(log n) or O(√n), making real-time retrieval possible at scale. Understanding indexing is understanding how vector databases can be fast.

---

### **2. Core idea**

**HNSW builds a multi-layer graph where each layer is a sparse proximity graph enabling logarithmic search via greedy traversal; IVF partitions vectors into clusters (Voronoi cells) and searches only the most promising clusters, both trading off accuracy for speed.**

---

### **3. Concrete analogy**

Imagine you are searching for a friend in a massive stadium with 100,000 people.

- **Linear scan:** Walk through every row, look at every face. Guaranteed to find them but takes hours.

- **IVF (Inverted File Index):** Group people by section (A1, A2, B1...). First find which section your friend likely sits in (based on ticket info), then search only that section. Faster, but if your guess is wrong, you might miss them.

- **HNSW:** Build a social network of "friends of friends." Start with a few acquaintances (top layer), ask them for people closer to your target, move down layers, refining. This is how small-world networks work. You find anyone in ~6 hops (six degrees of separation). Logarithmic search.

Real vector databases combine both: IVF for coarse partitioning, HNSW within partitions.

---

### **4. ASCII diagram**

```
HNSW multi-layer graph structure:

    Layer 2 (top, sparse):    ●────●────●
                              │    │    │
    Layer 1 (middle):         ●────●────●────●────●
                              │    │    │    │    │
    Layer 0 (bottom, dense):  ●────●────●────●────●────●────●────●

    Search path: Start at top layer, greedily move to nearest neighbor,
                 drop to next layer, repeat until bottom layer.
    Complexity: O(log n)


IVF (Inverted File Index):

    Step 1: Cluster vectors (k-means, k=100):

        Cluster 1 (centroid C1): ● ● ● ● ●
        Cluster 2 (centroid C2): ○ ○ ○ ○ ○
        Cluster 3 (centroid C3): □ □ □ □ □

    Step 2: At query time, find nearest centroid to query q.

        q is closest to C1 → search only Cluster 1

    Step 3: Brute-force within cluster.

    Complexity: O(√n) if clusters balanced.


Hybrid (IVF+HNSW):

    Cluster centroids indexed with HNSW (fast centroid selection)
    Within each cluster, vectors indexed with HNSW (fast intra-cluster search)
    Best of both worlds.
```

---

### **5. Mathematical formulation**

**HNSW (Hierarchical Navigable Small World):**

Build layers from bottom (layer 0, all vectors) to top (layer L, sparse).

Insertion: For each vector, randomly choose max layer l_max. Insert into layers 0...l_max.

At each layer, build graph connections: connect vector to its k nearest neighbors at that layer (using heuristic edge pruning).

Search: Starting at top layer, greedily traverse to nearest neighbor. Move to next layer, repeat.

**Search complexity:** O(log n) distance computations.

**IVF (Inverted File Index):**

Training phase: Run k-means on a subset of vectors to get k centroids {c₁...c_k}.

Indexing: For each vector v, assign to nearest centroid: id = argmin ‖v - c_i‖. Store v in inverted list i.

Query: Compute distances from q to all centroids. Select top n_probe centroids. Search only vectors in those inverted lists (brute-force or with secondary index).

**IVF complexity:** O(k) for centroid distances + O((n/k)·n_probe) for intra-list search.

**n_probe parameter:** Tradeoff between speed (small n_probe) and recall (large n_probe). Default 1-10.

**Product Quantization (PQ) for compression:**

Split vector into m subvectors, each quantized to 256 centroids (1 byte). Distance computed via precomputed tables.

Compression factor: 768 × 4 bytes (FP32) = 3KB → 768/8 × 1 byte = 96 bytes (32× compression).

---

### **6. Worked example (step-by-step)**

#### **Step 1: IVF clustering (k=3, n=12 vectors)**

Vectors in 2D (simplified).

Centroids learned via k-means:
C1 = [1, 1] (cluster A)
C2 = [5, 5] (cluster B)
C3 = [9, 1] (cluster C)

Assign each vector to nearest centroid.

#### **Step 2: Inverted lists**

Cluster A (C1): v1,v2,v3,v4 (4 vectors near [1,1])
Cluster B (C2): v5,v6,v7,v8 (4 vectors near [5,5])
Cluster C (C3): v9,v10,v11,v12 (4 vectors near [9,1])

#### **Step 3: Query q = [2, 2]**

Compute distances to centroids:
d(q, C1) = √[(2-1)²+(2-1)²] = √2 ≈ 1.41
d(q, C2) = √[(2-5)²+(2-5)²] = √18 ≈ 4.24
d(q, C3) = √[(2-9)²+(2-1)²] = √(49+1)=√50≈7.07

Nearest centroid: C1.

#### **Step 4: Search only cluster A (n_probe=1)**

Scan vectors in cluster A (4 of them) instead of all 12. 3× faster. If n_probe=2, also search cluster B. Slower but higher recall.

#### **Step 5: HNSW for intra-cluster search (optional)**

Instead of brute-force within cluster A, build HNSW graph within each cluster. Faster for large clusters.

---

### **7. How this appears inside neural networks and LLMs**

- **FAISS (Facebook AI Similarity Search):** The industry standard. Implements IVF, HNSW, PQ, and their combinations. Used by many RAG systems.

- **HNSW in vector databases (Pinecone, Qdrant, Milvus, Weaviate):** HNSW is the default index for many because of its excellent speed-recall tradeoff. Build once, query many.

- **IVF in production:** IVF is faster to build than HNSW (no graph construction). Good for dynamic datasets (frequent inserts/deletes). Supports adding vectors without rebuilding entire index.

- **PQ for large-scale RAG:** When you have >1M vectors, storing full vectors in memory is expensive. PQ compression reduces memory by 10-30×. Slightly lower recall but acceptable for many use cases.

- **GPU indexing (FAISS-GPU):** IVF can run on GPU (k-means clustering, distance computations). HNSW is CPU-only (graph traversal sequential).

- **Hybrid search:** Combine vector index with metadata filter. e.g., "Find products similar to query where price < 100". First filter by metadata, then vector search (or vice versa).

- **Index time vs query time tradeoff:** IVF is slower to build than HNSW? Actually HNSW build is O(n log n), IVF build is O(nk) (k-means). IVF faster to build for large n. Choose based on whether you update frequently.

---

### **8. Brain-like connection (hippocampal indexing)**

The hippocampus does not store every memory in a linear list. It uses an indexing structure similar to IVF. The dentate gyrus separates memories into distinct patterns (pattern separation), analogous to k-means clustering. The CA3 region performs pattern completion, retrieving similar memories given a partial cue—like HNSW traversal. The brain's memory search is sub-linear: you do not scan every memory; you navigate through associatively linked memories. This is hierarchical small-world navigation. Patients with hippocampal damage cannot retrieve memories efficiently—their "index" is broken. Vector database indexing algorithms are engineering implementations of hippocampal principles.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "HNSW is always better than IVF. It has O(log n) search vs IVF's O(√n). Why would anyone use IVF?"

_Why it is wrong:_ HNSW has higher memory usage (stores graph edges) and slower build time (O(n log n) edge comparisons). IVF is faster to build (k-means clustering, O(nk)) and supports dynamic updates (add new vectors without rebuilding). IVF can also be combined with PQ for compression, reducing memory. For very large datasets (>10M vectors) with limited memory, IVF+PQ is often better. For smaller datasets (<1M) with fast queries, HNSW wins. The choice depends on dataset size, memory budget, update frequency, and recall requirements. Neither is universally superior.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Linear scan over a million vectors is 1 million distance |
|  computations. Too slow for real-time. Indexing reduces   |
|  this to hundreds or thousands. HNSW gives O(log n)       |
|  search using layered graphs—like six degrees of          |
|  separation. IVF partitions vectors into clusters—like    |
|  searching only one bookshelf instead of the whole        |
|  library. Without indexing, vector databases are toys.    |
|  With indexing, they power production RAG systems.        |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have 10 million product embeddings (768 dimensions). You need real-time search (<50ms latency) for an e-commerce site. Updates are batch (once per day, not real-time). Memory is abundant (128GB). Which indexing algorithm would you choose and why? What parameters would you tune?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Choose **HNSW**. Here's why:

- **Batch updates (daily):** HNSW can be rebuilt offline during off-hours. No need for real-time insert.
- **Memory abundant (128GB):** 10M × 768 × 4 bytes (FP32) = 30.7GB for vectors. HNSW adds graph edges (≈ same again), total ≈ 60GB. Within 128GB budget.
- **Real-time latency requirement (<50ms):** HNSW typically achieves 10-30ms for 10M vectors with high recall (>0.99). IVF may be slower (requires probing multiple clusters).

**Tuned parameters:**

- **M (number of edges per node):** 16-32 (higher M = better recall, more memory, faster query)
- **efConstruction (build time):** 200 (higher = better graph, slower build)
- **efSearch (query time):** 100-200 (higher = better recall, slower query)
- **Distance metric:** Cosine (L2-normalize vectors first, then inner product)

If memory were limited (<32GB), you would choose IVF+PQ with compression. For your scenario (memory abundant, batch updates, real-time), HNSW is optimal. Use FAISS or Qdrant as implementation.
