# How RAG works step-by-step

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: RAG (Retrieval-Augmented Generation)**

---

### **1. Why this concept matters**

You have heard of RAG. You know it retrieves documents before generating answers. But how does it actually work under the hood? What are the exact steps? What happens when you click "search" in a RAG chatbot? Understanding the pipeline—from document ingestion to query to answer—is essential for debugging, optimizing, and building your own RAG system. This section walks through every step, from raw documents to final answer, with concrete examples.

---

### **2. Core idea**

**RAG has two phases: indexing (offline) and retrieval-generation (online). Indexing chunks documents, embeds them, and stores vectors. Online, the query is embedded, similar chunks are retrieved, and the LLM generates an answer grounded in those chunks.**

---

### **3. Concrete analogy**

Imagine building a research assistant for a lawyer.

**Indexing phase (one-time setup):**

- Take all legal documents (10,000 pages).
- Cut them into bite-sized paragraphs (chunks).
- For each chunk, write a summary card (embedding).
- File cards in a massive card catalog indexed by meaning (vector DB).

**Query phase (each question):**

- Lawyer asks: "What are the notice requirements for termination?"
- You write a summary card for the question.
- Flip through the card catalog, find the most similar cards.
- Pull the original document chunks for those cards.
- Staple them together, hand to the lawyer.
- Lawyer reads chunks and answers the question.

The indexing happens once. The query happens per question. This is RAG.

---

### **4. ASCII diagram**

```
RAG two-phase architecture:

    PHASE 1: INDEXING (offline, once)

    Documents (PDFs, wikis, emails)
         │
         ▼
    Chunking (split into pieces)
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
    Chunk1   Chunk2   Chunk3   Chunk4
    │         │        │        │
    ▼         ▼        ▼        ▼
    Embedding model (e.g., SBERT)
    │         │        │        │
    ▼         ▼        ▼        ▼
    Vector   Vector   Vector   Vector
    │         │        │        │
    └────┬────┴────────┴────────┘
         ▼
    Vector Database (HNSW, IVF)


    PHASE 2: QUERY (online, per user)

    User query: "What is the refund policy?"
         │
         ▼
    Embed query (same model)
         │
         ▼
    Vector DB search (top-k by cosine)
         │
         ▼
    Retrieved chunks (k=3-5)
         │
         ▼
    Construct prompt:
    "Context: [chunk1] [chunk2] [chunk3]
     Question: [query]
     Answer based only on context:"
         │
         ▼
    LLM generates answer
         │
         ▼
    Return answer to user
```

---

### **5. Mathematical formulation**

**Phase 1: Indexing**

Given document corpus D, chunk size L (tokens), overlap O (tokens).

Chunk each document: C = {c₁...cₘ} where |c_i| ≈ L, adjacent chunks overlap by O.

Embed each chunk: v_i = E(c_i) ∈ ℝᵈ, where E is embedding model (e.g., `all-MiniLM-L6-v2`, d=384).

Store in vector DB: (v_i, c_i, metadata_i).

**Phase 2: Query**

User query q. Compute query vector v_q = E(q).

Retrieve top-k chunks by cosine similarity:

$$
\text{sim}(v_q, v_i) = \frac{v_q \cdot v_i}{\|v_q\| \|v_i\|}
$$

$$
\text{Top-k} = \arg\max_{i=1..m}^k \text{sim}(v_q, v_i)
$$

Augment prompt template:

```
Context:
{'\n'.join([c_i for i in Top-k])}

Question: {q}

Answer based only on the context above. If the answer is not in the context, say "I don't know."
```

Generate answer: a = LLM(prompt)

**Optional reranking:** Retrieve top-k' (e.g., 20) then rerank with cross-encoder for final top-k (e.g., 5).

---

### **6. Worked example (step-by-step)**

#### **Step 1: Documents (tiny corpus)**

Doc: "RAG stands for Retrieval-Augmented Generation. It combines LLMs with vector search. RAG reduces hallucinations and keeps answers current."

#### **Step 2: Chunking (L=30 chars, O=0 for simplicity)**

Chunk1: "RAG stands for Retrieval-Augmented Generation."
Chunk2: "It combines LLMs with vector search."
Chunk3: "RAG reduces hallucinations and keeps answers current."

#### **Step 3: Embed (2D example for visualization)**

Assume embedding model produces:
v1 = [0.9, 0.2]
v2 = [0.8, 0.3]
v3 = [0.1, 0.9]

#### **Step 4: Index in vector DB**

Store (v1, chunk1), (v2, chunk2), (v3, chunk3).

#### **Step 5: User query**

q = "What is RAG?"

#### **Step 6: Embed query**

v_q = [0.85, 0.25] (similar to v1, v2; dissimilar to v3)

#### **Step 7: Cosine similarity**

sim(v_q, v1) = 0.96
sim(v_q, v2) = 0.92
sim(v_q, v3) = 0.35

Top-2: v1 (chunk1), v2 (chunk2)

#### **Step 8: Augment prompt**

```
Context:
RAG stands for Retrieval-Augmented Generation.
It combines LLMs with vector search.

Question: What is RAG?

Answer based only on the context above:
```

#### **Step 9: LLM generates**

Output: "RAG stands for Retrieval-Augmented Generation. It combines LLMs with vector search."

#### **Step 10: Return to user**

User receives answer grounded in retrieved chunks.

---

### **7. How this appears inside neural networks or LLMs**

- **Chunking strategies:** Fixed-size (e.g., 512 tokens, overlap 50). Semantic chunking (split at sentence boundaries). Recursive (document structure). Overlap prevents cutting sentences.

- **Embedding models:** General: SBERT (`all-MiniLM-L6-v2`, 384 dims). Domain-specific: Legal-BERT, Bio-BERT, CodeBERT. OpenAI `text-embedding-ada-002` (1536 dims). Voyage, Cohere, etc.

- **Vector DB retrieval:** Cosine similarity, top-k (typically 3-10). Filter by metadata before search (pre-filter) or after (post-filter).

- **Prompt template:** Critical. Include clear instructions: "Answer based only on context." "If not in context, say 'I don't know'." Helps reduce hallucination.

- **Reranking:** Cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks retrieved chunks. Higher accuracy, slower. Use after initial vector search.

- **Context window management:** If top-k chunks exceed LLM context limit, compress (summarize) or truncate (keep most relevant sentences).

- **Citation:** LLM can be prompted to output citations: `Answer: ... [1]`. Map to retrieved chunk IDs.

---

### **8. Brain-like connection (autobiographical memory retrieval)**

Human memory retrieval follows a similar two-phase process. When asked "What did you do last summer?" your brain does not replay every memory. It embeds the query into a neural representation (hippocampus), searches memory via similarity, retrieves candidate episodes (pattern completion), and then "generates" a narrative. This is RAG: retrieval (hippocampus) + generation (prefrontal cortex). The brain's retrieval is approximate (like ANN), not exact. Patients with hippocampal damage can still generate answers but cannot retrieve specific memories—they hallucinate plausible but false narratives. This is analogous to an LLM without retrieval: fluent but not grounded.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "RAG retrieves documents first, then the LLM reads them. That is all. Nothing else matters."

_Why it is wrong:_ RAG has many knobs. Chunk size: too small → missing context; too large → irrelevant information. Embedding model: general model may fail on domain jargon. Top-k: too few → missing answer; too many → context window overflow. Prompt template: "Answer based only on context" prevents hallucination; without it, LLM may ignore retrieved docs. Reranking: vector search is approximate; cross-encoder can fix order. All these choices matter. A naive RAG (fixed chunk, default model, top-5, simple prompt) works for simple cases but fails for real-world complexity. RAG is not trivial—it is a pipeline of components that must be tuned together.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  RAG is not magic. It is a pipeline: chunk → embed →      |
|  index → retrieve → rerank → prompt → generate. Each      |
|  step has knobs. Turn them wrong, you get garbage.        |
|  Turn them right, you get answers grounded in your        |
|  documents. Understanding the step-by-step is not         |
|  academic—it is how you debug when RAG fails. Learn the   |
|  pipeline. Test each component. Build RAG that works.     |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You deploy a RAG system for customer support. Users complain that the bot is missing obvious answers that exist in the documentation. You suspect the retrieval step is failing (relevant documents not retrieved). How would you debug? List three things to check.

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Three things to check:

1. **Embedding model quality:** Test on sample queries. Embed a query and its relevant document. Cosine similarity should be >0.7. If low, try different embedding model (e.g., switch from `all-MiniLM-L6-v2` to `intfloat/e5-large-v2` or OpenAI `text-embedding-3-small`). Also check if model is domain-specific (legal, medical, code). Fine-tune embedding model on your data if needed.

2. **Chunking strategy:** Are answers split across chunks? Example: "Refund policy: Returns within 30 days. Refund issued to original payment method." If split into two chunks, retrieval may find only one. Increase chunk size or use overlapping chunks (50-100 token overlap). Use semantic chunking (split at sentence/paragraph boundaries).

3. **Top-k and similarity threshold:** Maybe relevant doc has similarity 0.65 but top-k=5 includes only docs >0.7. Increase k to 10 or lower threshold. Check distribution of similarity scores for relevant vs irrelevant docs.

4. **Query-document vocabulary mismatch:** User says "return item"; document says "refund policy." Embedding model may not capture synonymy. Use hybrid search (vector + BM25 keyword). Or use query expansion: LLM generates alternative phrasings.

5. **Metadata filtering:** Are you accidentally filtering out relevant docs by metadata (e.g., product category, date)? Check filter logic. Try removing filters temporarily.

6. **Reranking:** Retrieve top-20 with fast vector, rerank with cross-encoder. Cross-encoder may find relevance that vector search missed.

Debugging RAG requires tracing each step. Log retrieved chunk IDs and their similarity scores. Compare to ground truth relevant documents. Adjust components based on findings.
