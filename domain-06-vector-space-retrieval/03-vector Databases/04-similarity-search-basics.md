# Similarity search basics

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Vector Databases: The Collective Memory**

---

### **1. Why this concept matters**

You have a million documents, each converted to an embedding vector. Now you ask a question. How do you find the most relevant documents? You convert the question to a vector and search for the closest vectors in embedding space. That is similarity search: finding items that are "close" to your query. It is the core of RAG (Retrieval-Augmented Generation), recommendation systems, and semantic search. Without similarity search, you cannot retrieve relevant context for LLMs. With it, you can search by meaning, not just keywords.

---

### **2. Core idea**

**Similarity search finds items whose embeddings are closest to a query embedding according to a distance metric (cosine, Euclidean, dot product), enabling retrieval by semantic meaning rather than exact keyword matching.**

---

### **3. Concrete analogy**

Imagine you have a library of a million books. You want books about "space exploration." A keyword search looks for exact words: "space," "exploration," "rocket." It misses books about "cosmic travel" or "astronautics."

A similarity search works differently. You read a few pages of a book about space exploration, capture its "essence" (embedding), then find books with similar essence. You find "cosmic travel" even though it has no matching keywords.

Now imagine you have a genie that converts any text into a "meaning fingerprint" (embedding). You fingerprint your query "space exploration." The genie finds the most similar fingerprints in the library. That is similarity search.

---

### **4. ASCII diagram**

```
Similarity search pipeline:

    Documents (millions)                 Query
         │                                  │
         ▼                                  ▼
    [Embedding model]                  [Embedding model]
         │                                  │
         ▼                                  ▼
    Document vectors                 Query vector
    (768 dimensions)                 (768 dimensions)
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
              Similarity computation
              (cosine, Euclidean, dot)
                        │
                        ▼
                   Top-k nearest
                        │
                        ▼
              Retrieved documents


Visualization in 2D (simplified):

    Dimension 2
         ↑
         │   ● ●   ○ ○
         │  ●   ● ○   ○   (documents: ● relevant, ○ not)
         │ ●     ●○○
         │    ╱
         │  ╱   ★ (query)
         │ ╱
         │●
         └────────────────→ Dimension 1

    Nearest neighbors to ★ are ● documents (good).
    Far away are ○ (irrelevant).


Query types:

    Query: "space exploration"
    Embed → find similar document embeddings

    Example: "I want a shirt like this" (image query)
    Embed image → find similar image embeddings

    Query: "Find similar users to user 123"
    Embed user preferences → find similar user embeddings
```

---

### **5. Mathematical formulation**

**Given:** Query vector q ∈ ℝᵈ, dataset {x₁...xₙ} ⊂ ℝᵈ, distance metric d, integer k.

**Similarity search:** Find top-k indices minimizing (or maximizing) d:

$$
\text{Top-k}(q) = \arg\min_{i \in \{1...n\}}^k d(q, x_i) \quad \text{(for distance)}
$$

Or maximizing for similarity:

$$
\text{Top-k}(q) = \arg\max_{i \in \{1...n\}}^k \text{sim}(q, x_i)
$$

**Common distance/similarity metrics:**

Cosine similarity (range [-1, 1], higher is better):

$$
\text{cos}(q, x) = \frac{q \cdot x}{\|q\| \|x\|}
$$

Euclidean distance (range [0, ∞), lower is better):

$$
d(q, x) = \sqrt{\sum_{j=1}^d (q_j - x_j)^2}
$$

Dot product (range (-∞, ∞), higher is better for normalized vectors):

$$
\text{dot}(q, x) = \sum_{j=1}^d q_j x_j
$$

**Normalization for equivalence:**

If all vectors are L2-normalized (‖v‖=1), then:

- Cosine similarity = dot product
- Euclidean distance = √(2 - 2·dot)

Thus, all metrics are equivalent after normalization.

**k parameter:** Number of results to return. Typical values: 5-100.

**Thresholding:** Return all vectors with similarity > threshold. Used for anomaly detection.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Dataset of 4 documents (2D embeddings)**

Doc1 (cats): [0.9, 0.2]
Doc2 (dogs): [0.8, 0.3]
Doc3 (python): [0.1, 0.9]
Doc4 (snake): [0.2, 0.8]

#### **Step 2: Query embedding**

Query: "Canine pets" → v_q = [0.85, 0.25]

#### **Step 3: Compute cosine similarities**

cos(q, doc1) = (0.85×0.9 + 0.25×0.2) / (1.0×1.0) = (0.765 + 0.05) = 0.815
cos(q, doc2) = (0.85×0.8 + 0.25×0.3) = 0.68 + 0.075 = 0.755
cos(q, doc3) = (0.85×0.1 + 0.25×0.9) = 0.085 + 0.225 = 0.310
cos(q, doc4) = (0.85×0.2 + 0.25×0.8) = 0.17 + 0.20 = 0.370

#### **Step 4: Rank results (k=2)**

Top-2: Doc1 (0.815), Doc2 (0.755)

#### **Step 5: Return documents**

Retrieved: "Cats are cute pets", "Dogs are loyal animals"

#### **Step 6: Interpret**

Query about "canine pets" retrieved "dogs" (relevant) and also "cats" (less relevant but still animal-related). The search worked by semantic meaning, not keyword matching.

---

### **7. How this appears inside neural networks or LLMs**

- **RAG (Retrieval-Augmented Generation):** Query embedding → similarity search → retrieve top-k documents → LLM generates answer with context.

- **Semantic search (enterprise, web):** Replace BM25/keyword search with embedding similarity. Handles synonyms, misspellings, paraphrases.

- **Recommendation systems (Netflix, Amazon):** User embedding → find similar user/item embeddings → recommend.

- **Deduplication (near-duplicate detection):** Find documents with cosine > 0.95 → merge or flag.

- **Anomaly detection (fraud, outlier):** Find documents with low similarity to all stored vectors.

- **Question answering (closed domain):** Embed FAQ questions, retrieve nearest when user asks.

- **Code search (GitHub Copilot):** Embed code snippets, retrieve similar code given natural language query.

- **Multimodal search (CLIP):** Image → embedding, text → embedding, search across modalities.

---

### **8. Brain-like connection (cued recall)**

Human memory retrieval is similarity search. When you hear "dog," your brain retrieves related concepts: "cat," "pet," "bark," "leash." This is similarity search in neural embedding space. The hippocampus performs this search, retrieving memories similar to the current cue. Cued recall experiments show that people retrieve items semantically similar to the cue, not just identical. This is why you might say "cat" when asked "name a pet starting with 'c'?" The brain's search is approximate and semantic, just like vector similarity search. Damage to hippocampal regions impairs this ability, preventing retrieval of related memories.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Similarity search is just keyword search with better ranking. You can implement it with PostgreSQL and a vector column."

_Why it is wrong:_ Similarity search with a vector column in PostgreSQL still requires a linear scan (unless you add an index). For a million vectors, that is slow. Production similarity search requires specialized indexing (HNSW, IVF) for sub-linear time. Moreover, keyword search (BM25) is lexical, relying on exact word matches. Similarity search is semantic, capturing meaning even without word overlap. "Car" and "automobile" have zero keyword overlap but high semantic similarity. You cannot get that with keyword search, regardless of ranking algorithm. Similarity search is not "fancier keyword search"—it is a different paradigm.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Traditional search looks for words. Similarity search    |
|  looks for meaning. "Car" and "automobile" are close.     |
|  "Space exploration" and "cosmic travel" are close.       |
|  This is how RAG finds relevant context for LLMs, how     |
|  Netflix recommends movies, how Google finds images.      |
|  Similarity search is the engine of modern retrieval.     |
|  Master the basics—embed, index, search, rank—and you    |
|  can build systems that understand intent, not just text. |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have 100,000 customer reviews (text). You want to find reviews that are semantically similar to "The battery life is terrible." You have an embedding model that produces 384-dim vectors.

**Question:** Walk through the similarity search process step-by-step. What distance metric would you use? How would you handle the fact that "battery life" might be phrased as "battery longevity" in some reviews? What about documents that are long vs short?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

#### **Step-by-step process:**

1. **Offline indexing:** Run embedding model on all 100,000 reviews → store vectors in vector DB (FAISS, Pinecone, etc.)
2. **Query:** Embed "The battery life is terrible" → query vector q
3. **Search:** Use cosine similarity (standard for text embeddings) to find top-k nearest vectors
4. **Return:** Retrieve the corresponding review texts

**Distance metric:** Cosine similarity. Why? Reviews vary in length; cosine normalizes away length bias. A short review "Battery sucks" should match a long review despite length difference.

**Handling synonyms:** Cosine similarity captures semantic meaning. "Battery life" and "battery longevity" have similar embeddings because they appear in similar contexts during model training. No explicit synonym mapping needed.

**Handling document length:** Cosine similarity is length-invariant. A short review "Battery bad" and a long review analyzing battery in detail will have vectors pointing in similar directions (same meaning) despite different magnitudes. Euclidean or dot product would favor longer reviews.

**Implementation:** Use FAISS with L2-normalized vectors (so dot = cosine). Add metadata filter (e.g., date range, product category) if needed. Return top-10 reviews for human inspection or further analysis.

**Potential issue:** Very short reviews (2-3 words) may have noisy embeddings. Consider using a minimum length filter or using a sentence embedding model designed for short text (e.g., SBERT).
