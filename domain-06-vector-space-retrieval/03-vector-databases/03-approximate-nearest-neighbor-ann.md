# Approximate Nearest Neighbor (ANN) search

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Vector Databases: The Collective Memory**

---

### **1. Why this concept matters**

Exact nearest neighbor search guarantees you find the closest vector. But it requires scanning every vector—O(n) time. With 10 million vectors, that is 10 million distance computations. For real-time retrieval, that is impossible. Approximate Nearest Neighbor (ANN) search sacrifices perfect accuracy for speed. It returns "close enough" vectors, not necessarily the exact closest, but does it in O(log n) or O(√n) time. ANN is the reason vector databases can answer queries in milliseconds, not minutes. The tradeoff is recall: 99% recall (meaning 1% of true neighbors missed) is often acceptable for applications like search and recommendations.

---

### **2. Core idea**

**Approximate Nearest Neighbor search finds vectors that are likely close to the query (but not guaranteed to be the exact closest) using indexing structures that prune the search space, achieving sub-linear query time at the cost of a small reduction in recall.**

---

### **3. Concrete analogy**

Imagine you are looking for the tallest person in a stadium of 50,000 people. Exact search requires measuring every person's height—hours of work.

Approximate search: Look only at the basketball players section. You will find someone very tall, but possibly not the absolute tallest (there might be a tall person in the volleyball section). The tradeoff: minutes instead of hours, 99% as tall.

In vector search, ANN algorithms like HNSW, IVF, and PQ each use different strategies to prune the search space:

- **HNSW:** Follows "friends of friends" graph. Like asking people for someone taller, moving up the chain.
- **IVF:** Clusters vectors. Like searching only the most promising sections of the stadium.
- **PQ:** Compresses vectors. Like measuring height with a coarse ruler (faster but less precise).

---

### **4. ASCII diagram**

```
Exact vs Approximate nearest neighbor:

    Exact (linear scan):

    Query ●
           │
    ┌──────┼──────────────────────────┐
    │      ▼                          │
    │ Compare to all 50,000 vectors   │
    │ O(n) time, 100% recall          │
    └─────────────────────────────────┘


    Approximate (IVF, n_probe=2):

    Query ●
           │
           ▼
    Find nearest 2 cluster centroids
           │
    ┌──────┼──────────┐
    │      ▼          ▼
    │ Search only    │
    │ Cluster A      │ Cluster B
    │ (5,000 vectors)│ (5,000 vectors)
    └────────────────┘

    O(√n) time, ~95% recall


Recall-speed tradeoff curve:

    Recall
      1.0 ┤                    ●
          │                  ●
      0.9 ┤                ●
          │              ●
      0.8 ┤            ●
          │          ●
      0.7 ┤        ●
          │      ●
      0.6 ┤    ●
          └────────────────────→ Speed (queries/sec)
             1    10   100   1000

    You choose operating point based on application.
```

---

### **5. Mathematical formulation**

**Exact Nearest Neighbor (NN):**

Given query q, dataset X = {x₁...xₙ}, find:

$$
\text{NN}(q) = \arg\min_{x_i} \|q - x_i\|
$$

**Approximate NN:** Find x̂ such that with high probability:

$$
\|q - \hat{x}\| \leq (1+\epsilon) \|q - \text{NN}(q)\|
$$

Where ε ≥ 0 is approximation factor. ε=0 gives exact search.

**Recall@k metric:**

$$
\text{Recall@k} = \frac{|\text{ANN}_k(q) \cap \text{NN}_k(q)|}{k}
$$

Proportion of true top-k neighbors retrieved by ANN.

**Search complexity tradeoff:**

| Algorithm   | Exact Complexity | ANN Complexity       | Typical Recall |
| ----------- | ---------------- | -------------------- | -------------- |
| Linear scan | O(n·d)           | O(n·d) (no speedup)  | 100%           |
| HNSW        | N/A              | O(log n·d)           | 95-99%         |
| IVF         | N/A              | O(√n·d)              | 90-98%         |
| IVF+PQ      | N/A              | O(√n·d) + compressed | 85-95%         |

**Speed-recall tradeoff curve:**

For a given index (HNSW), you can vary efSearch (search effort). Higher efSearch = higher recall + slower query.

For IVF, vary n_probe (number of clusters to search). Higher n_probe = higher recall + slower query.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Dataset (2D, n=100,000)**

Vectors randomly distributed in 100 clusters. Query q = [0,0].

#### **Step 2: Exact nearest neighbor (brute-force)**

Compute distance to all 100,000 vectors. Time: 100,000 × 768 dims ≈ 76M operations. Takes ~0.5 seconds. Finds true nearest neighbor (guaranteed).

#### **Step 3: IVF (k=100 clusters, n_probe=1)**

Train k-means on dataset, get 100 centroids.

Compute distance from q to 100 centroids (fast, 100 ops). Nearest centroid C3.

Search only vectors in cluster C3 (~1000 vectors). Time: 100 (centroid) + 1000 (cluster) = 1100 ops. 0.005 seconds. 100× faster.

Recall: True nearest neighbor may be in a different cluster. If it is in C3, recall=100%. If in another cluster, recall=0% for n_probe=1. For n_probe=10 (search 10 clusters), recall >95%.

#### **Step 4: HNSW (M=16, efSearch=100)**

Graph built with each vector connected to ~16 neighbors.

Search: Start at top layer, greedily move to closer vectors, refining. Visits ~500 vectors instead of 100,000. Time: ~500 ops. 0.002 seconds. 200× faster. Recall ~98% for this dataset.

#### **Step 5: Tradeoff selection**

For product search: 98% recall acceptable (users may miss some less relevant items). For medical retrieval: need 99.9% recall, use slower index (more probes, higher efSearch) or fall back to exact for safety.

---

### **7. How this appears inside neural networks or LLMs**

- **RAG (Retrieval-Augmented Generation):** ANN search retrieves relevant documents for LLM context. 95% recall acceptable; LLM still answers correctly with slightly suboptimal context.

- **Vector databases (Pinecone, Qdrant, Milvus, Weaviate):** All implement ANN (HNSW or IVF) as default. Users control recall via parameters (efSearch, n_probe).

- **FAISS (Facebook):** Explicitly supports tradeoff. `index.search()` returns distances and IDs; you can increase `nprobe` for higher recall.

- **Recommendation systems:** Retrieve similar items to user's last purchase. 99% recall not necessary; 90% recall still improves recommendations.

- **Deduplication:** Find near-duplicate documents. ANN is sufficient; exact duplicates rare.

- **Outlier detection:** ANN approximations may miss outliers if not in probed clusters. Use higher recall for safety.

- **Neural network inference caching:** Store input-output pairs, retrieve cached response for similar inputs using ANN. Small recall loss acceptable for speed gain.

- **Approximate distance computation (PQ):** Compressed distances cheaper to compute. Recall loss compensated by larger n_probe.

---

### **8. Brain-like connection (recall vs precision tradeoff in memory)**

Human memory is approximate. When you try to recall a fact (e.g., "What is the capital of France?"), your brain does not scan every memory exactly. It uses approximate retrieval: "Paris" comes to mind quickly, but you might also recall "Lyon" or "Marseille" if the cue is weak. This is ANN in the brain—fast, approximate, but occasionally wrong. Eyewitness testimony is notoriously unreliable for the same reason: memory retrieval trades off recall for speed. Patients with hyperthymesia (superior autobiographical memory) may have more exact retrieval but at the cost of slower recall or distraction. The brain's ANN tradeoff is evolutionarily optimal: speed matters for survival; exactness is secondary.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "ANN is just an approximation. Why not always use exact search? It is more accurate."

_Why it is wrong:_ Exact search is O(n·d). For n=1 million, d=768, that is 768M operations per query. At 1ms per operation (optimistic), that is 768 seconds—unacceptable for real-time. Even with GPU optimization (parallel batch), exact search on 1M vectors takes >10ms, which may be acceptable. But at 100M vectors, exact search becomes impossible. ANN brings search time to ~1ms. The accuracy loss (98-99% recall) is negligible for most applications. Users do not notice missing the 100th nearest neighbor if they get the top 99. Speed is more important than perfection in production systems.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Exact nearest neighbor is a gold standard you cannot     |
|  afford. At scale, O(n) kills you. ANN gives you 1000×    |
|  speedup for 1% recall loss. For web search, 99.9% of     |
|  users click the top result—they never miss the 100th.    |
|  For RAG, LLMs are robust to slightly different context.  |
|  The tradeoff is worth it. ANN is not laziness; it is     |
|  engineering reality. Speed matters. Approximate is       |
|  often good enough.                                       |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are building a RAG system for legal document retrieval (10 million legal paragraphs). Accuracy is critical (lawyers need correct citations). You benchmark ANN (HNSW) with recall@10 = 98.5% and latency = 20ms. Exact search has 100% recall but latency = 5000ms.

**Question:** Would you use ANN or exact? If ANN, how would you mitigate the recall loss? If exact, how would you make it faster?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Use ANN, but with a fallback strategy. Here's why:

- 5000ms (exact) is too slow for interactive use. Lawyers will not wait 5 seconds per query.
- 98.5% recall means 1.5% of queries will miss relevant documents. Unacceptable for legal.

**Mitigation strategies:**

1. **Two-stage retrieval:** ANN retrieves top-100 (instead of top-10). Re-rank with exact distances (only 100 vectors, cheap). Recall improves to near 100%.

2. **Hybrid search:** Combine ANN with keyword search (BM25). If ANN confidence low, fall back to exact or hybrid.

3. **Higher recall ANN:** Tune HNSW parameters (increase efSearch from 100 to 500). Recall may reach 99.5% with latency 50ms. Acceptable.

4. **Index partitioning:** Pre-filter by metadata (jurisdiction, date range) before ANN. Smaller index per query.

5. **Exact on cache:** Cache frequent queries and their exact results. After a few days, most queries are cached.

**If you insisted on exact search:** Use GPU with batched queries (process 100 queries in parallel). Optimize with FAISS (GPU exact). May reduce latency to 200ms. Still slower than ANN but acceptable if queries are batched.

Recommendation: ANN + two-stage re-ranking. Fast enough, near-exact recall. This is standard practice in production RAG.
