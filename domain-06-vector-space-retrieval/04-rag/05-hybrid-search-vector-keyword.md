# Hybrid search (combining vector + keyword)

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: RAG (Retrieval-Augmented Generation)**

---

### **1. Why this concept matters**

Vector search is great at semantics: "car" finds "automobile." But it misses exact matches: a product code "XJ-100" might be semantically unrelated to anything, but the user wants exactly that code. Keyword search (BM25) excels at exact matches and rare terms but fails at synonyms. Hybrid search combines both: vector similarity for meaning, keyword matching for precision. It is the standard in production RAG systems because real-world queries mix semantic concepts ("wireless headphones") and specific terms ("model XJ-100, $50 price"). Neither pure vector nor pure keyword alone suffices.

---

### **2. Core idea**

**Hybrid search combines vector similarity (dense retrieval) and keyword matching (sparse retrieval, e.g., BM25) using a weighted sum, capturing both semantic meaning and exact term matches in a single retrieval system.**

---

### **3. Concrete analogy**

Imagine you are searching for a restaurant.

- **Vector search:** "cozy Italian place with outdoor seating" → finds restaurants with similar vibes, even if they don't have those exact words.
- **Keyword search:** "Joe's Pizza 123 Main St" → finds exact address match, even if it doesn't understand "cozy."

Now imagine you search for "Joe's Pizza near Central Park." You need both: exact name "Joe's Pizza" (keyword) and semantic "near Central Park" (vector). Hybrid search does both.

In RAG, a user might ask: "What is the return policy for order #R12345?" Vector search finds "return policy" (semantic). Keyword finds "R12345" (exact). Hybrid returns chunks containing both.

---

### **4. ASCII diagram**

```
Hybrid search pipeline:

    Query: "What is the return policy for order #R12345?"
         │
    ┌────┴────┬─────────────────────┐
    ▼         ▼                     ▼
    Vector    Keyword               (Optional)
    Search    Search (BM25)         Reranking
    │         │                     │
    ▼         ▼                     ▼
    scores_v  scores_k              │
    │         │                     │
    └────┬────┘                     │
         ▼                          │
    Weighted fusion:                 │
    score = α·norm(scores_v) + (1-α)·norm(scores_k)
         │
         ▼
    Top-k documents
         │
         ▼
    Retrieved chunks


Weighted fusion visualization (α=0.7):

    Document scores before fusion:

    Doc   | Vector (v) | Keyword (k) | Hybrid (0.7v + 0.3k)
    ------|------------|-------------|-------------------
    Doc1  | 0.9        | 0.1         | 0.66
    Doc2  | 0.5        | 0.9         | 0.62
    Doc3  | 0.8        | 0.2         | 0.62
    Doc4  | 0.2        | 0.8         | 0.38

    Doc1 wins (high vector). Doc2 and Doc3 tie.
    α balances vector vs keyword importance.
```

---

### **5. Mathematical formulation**

**BM25 (keyword) score for query q with terms t₁...tₘ, document d:**

$$
\text{BM25}(q, d) = \sum_{i=1}^{m} \text{IDF}(t_i) \cdot \frac{f(t_i, d) \cdot (k_1 + 1)}{f(t_i, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
$$

Where:

- f(t,d) = term frequency in document
- |d| = document length
- avgdl = average document length
- k₁ = 1.2 (term frequency saturation)
- b = 0.75 (length normalization)

**Normalization for fusion:**

Min-max normalize vector scores and BM25 scores to [0,1]:

$$
\text{norm}(s) = \frac{s - \min(s)}{\max(s) - \min(s)}
$$

Or use rank-based fusion (RRF) to avoid scaling issues.

**Weighted sum fusion:**

$$
\text{score}_{\text{hybrid}}(d) = \alpha \cdot \text{norm}(\text{vector}(q,d)) + (1-\alpha) \cdot \text{norm}(\text{BM25}(q,d))
$$

Where α ∈ [0,1] controls semantic vs keyword weight (α=0.7 common).

**Reciprocal Rank Fusion (RRF) - no normalization needed:**

$$
\text{RRF}(d) = \sum_{r \in \{\text{vector}, \text{keyword}\}} \frac{1}{k + \text{rank}_r(d)}
$$

Where k=60 (typical), rank_r(d) is position of d in result list from method r.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Query**

q = "RAG retrieval chunk size"

#### **Step 2: Document corpus (4 chunks)**

Doc1: "RAG stands for Retrieval-Augmented Generation."
Doc2: "Chunk size affects retrieval quality. Smaller chunks for QA."
Doc3: "Vector search uses embeddings for semantic similarity."
Doc4: "BM25 keyword search matches exact terms like 'chunk'."

#### **Step 3: Vector scores (cosine similarity)**

Vector scores (already normalized):
v = [0.1, 0.9, 0.5, 0.3]

#### **Step 4: Keyword scores (BM25)**

BM25 raw scores:
k_raw = [0.05, 0.8, 0.1, 0.6]

Min-max normalize to [0,1]:
k_norm = [0.0, 1.0, 0.07, 0.73]

#### **Step 5: Hybrid fusion (α=0.7)**

hybrid = 0.7*v + 0.3*k_norm
= [0.07, 0.93, 0.37, 0.43]

#### **Step 6: Top-2 results**

Doc2 (0.93), Doc4 (0.43)

#### **Step 7: Compare pure vector vs pure keyword**

- Pure vector top-2: Doc2 (0.9), Doc3 (0.5) → missed Doc4
- Pure keyword top-2: Doc2 (1.0), Doc4 (0.73) → correct but missed Doc3
- Hybrid: Doc2, Doc4 → best of both

#### **Step 8: Interpret**

Hybrid retrieved relevant Doc2 (chunk size) and Doc4 (BM25). Doc3 (vector search) was less relevant. α=0.7 favors semantics, but keyword still boosted Doc4.

---

### **7. How this appears inside neural networks or LLMs**

- **Vector databases with hybrid search:** Pinecone (hybrid via sparse-dense), Weaviate (native hybrid), Qdrant (hybrid with BM25), Milvus (hybrid via separate indices).

- **Reciprocal Rank Fusion (RRF):** Standard for combining rankings without normalization. Works well when vector and keyword scores have different distributions.

- **α tuning:** α=1.0 pure vector; α=0.0 pure keyword. Typical range: α=0.3-0.7. Tune on validation set.

- **Use cases for hybrid:** E-commerce (product name exact + semantic category), legal (citation exact + concept search), code (function name exact + semantic description), customer support (order number exact + issue semantics).

- **Hybrid with reranking:** Retrieve top-50 with hybrid, rerank with cross-encoder for top-10.

- **When to skip hybrid:** Pure semantic search (no exact terms needed). Small corpus (<10K docs) where keyword not beneficial.

- **Implementation in LangChain:** `from langchain.retrievers import EnsembleRetriever`, combine vector and BM25 retrievers with weights.

---

### **8. Brain-like connection (dual-stream retrieval)**

The brain has two retrieval systems. Semantic memory (temporal lobe) retrieves by meaning: "dog" → "cat." Episodic memory (hippocampus) retrieves by specific events: "what did I eat for breakfast?" This is hybrid search. The brain combines both: when searching for your keys, you use semantic ("small metal object") and episodic ("last saw them on the kitchen counter"). Patients with semantic dementia lose meaning-based retrieval but keep specific memory. Patients with hippocampal damage lose specific retrieval but keep semantics. Brain-damaged patients behave like pure vector (no keyword) or pure keyword (no vector) retrieval. Hybrid requires both systems intact.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Hybrid search is just running vector and keyword separately and averaging scores. It's trivial."

_Why it is wrong:_ Score normalization is nontrivial. Vector scores (cosine) range [-1,1]; BM25 scores are unbounded (0 to large). Direct averaging fails. Min-max normalization works but is sensitive to outliers. Rank-based fusion (RRF) avoids normalization but loses score magnitude. Choosing α requires tuning; wrong α degrades both. Moreover, hybrid search doubles compute (two retrieval systems). Engineering tradeoffs matter. Hybrid is not trivial—it is a carefully balanced system.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Pure vector search misses exact terms: product codes,    |
|  order numbers, proper names. Pure keyword misses         |
|  synonyms: "car" vs "automobile." Hybrid combines both.   |
|  Production RAG systems use hybrid because real queries   |
|  mix semantics and specifics. "Return policy for order    |
|  #R12345" needs both. Tune α. Validate on your data.      |
|  Hybrid is not optional for serious RAG—it is required.   |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are building a RAG system for a legal firm. Lawyers ask queries like: "What is the statute of limitations for breach of contract under Section 5 of the Commercial Code?" and "Show me the Rodriguez case from 2022."

**Question:** Why does this use case need hybrid search? What α would you start with (vector weight)? Why?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

**Why hybrid is needed:**

- **Semantic part:** "statute of limitations," "breach of contract" → vector search captures meaning, synonyms.
- **Exact part:** "Section 5," "Commercial Code," "Rodriguez case," "2022" → keyword search needed for exact citations, case names, dates.

Pure vector would miss exact section numbers. Pure keyword would miss semantically similar clauses.

**Starting α (vector weight):** α = 0.6 or 0.7

Reason: Legal queries are semantic-heavy (concepts, arguments) but require exact citations. Slight favor to vector (0.6) because meaning is primary, but keyword still significant (0.4). For case name queries ("Rodriguez case"), keyword needs higher weight; consider query classification to adjust α dynamically.

**Additional considerations:**

- For citations ("Section 5"), use exact phrase matching in keyword search (not just term matching).
- For case names, use proper name recognition (NER) before retrieval.
- Hybrid with reranking: retrieve top-50 hybrid, rerank with legal-specific cross-encoder.
- Evaluate on held-out legal queries with ground truth relevant documents. Tune α to maximize recall@10. Expect α between 0.4 and 0.7 depending on query mix.

**Advanced:** Use dynamic α based on query type. Parse query: if contains citation pattern (e.g., "Section \d+"), increase keyword weight (α=0.3). Otherwise, use default α=0.7.
