# The solution: Retrieve relevant documents first

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: RAG (Retrieval-Augmented Generation)**

---

### **1. Why this concept matters**

LLMs know a lot but not everything. They have knowledge cutoffs, no access to private data, and limited context windows. The solution is not to build bigger models—it is to give existing models the ability to look things up. Retrieve relevant documents first, then generate the answer. This is RAG (Retrieval-Augmented Generation). It turns a static LLM into a dynamic system that can access your company wiki, recent news, or any external corpus. RAG reduces hallucinations, provides citations, and keeps answers current without retraining. It is the most practical way to deploy LLMs in production.

---

### **2. Core idea**

**Retrieval-Augmented Generation (RAG) first retrieves relevant documents from an external corpus using vector similarity, then augments the prompt with those documents, and finally generates an answer using the LLM grounded in the retrieved context.**

---

### **3. Concrete analogy**

Imagine you are taking an open-book exam. You do not need to memorize everything. When asked a question, you look up relevant chapters, read them, then write your answer. You are not cheating—you are using available information efficiently.

RAG works the same way:

- **Retrieve:** Search your notes (vector DB) for pages relevant to the question.
- **Augment:** Staple those pages to the exam question.
- **Generate:** Write your answer using both the question and the notes.

Without retrieval, you would rely on memory alone (hallucinate). With retrieval, you have evidence. The LLM is the student; the vector database is the textbook.

---

### **4. ASCII diagram**

```
RAG pipeline step-by-step:

    User query: "What is RAG?"
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 1. Embed query using same model                         │
    │    q = embed("What is RAG?")                            │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 2. Vector search over document corpus                   │
    │    results = vector_db.search(q, top_k=5)               │
    │    → doc1: "RAG stands for Retrieval-Augmented..."      │
    │    → doc2: "RAG combines LLMs with external memory..."  │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 3. Augment prompt with retrieved documents              │
    │    prompt = """                                          │
    │    Context:                                              │
    │    - RAG stands for Retrieval-Augmented Generation...   │
    │    - RAG combines LLMs with vector databases...         │
    │                                                         │
    │    Question: What is RAG?                               │
    │    Answer based only on the context above:"""           │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 4. LLM generates answer grounded in context             │
    │    output = llm(prompt)                                 │
    │    → "RAG (Retrieval-Augmented Generation) is a         │
    │       technique that combines LLMs with external        │
    │       memory retrieval to ground answers in             │
    │       relevant documents."                              │
    └─────────────────────────────────────────────────────────┘


RAG vs no-RAG comparison:

    No RAG:     Query → LLM → Answer (based only on training)

    RAG:        Query → Retrieve → [Context + Query] → LLM → Answer
                                ↑
                          (grounded in retrieved docs)
```

---

### **5. Mathematical formulation**

**RAG pipeline:**

Let D = {d₁...dₙ} be document corpus (e.g., company wiki).

Let E be embedding function (e.g., BERT, SBERT).

**Offline indexing:**
For each document d_i, compute vector v_i = E(d_i). Store (v_i, d_i, metadata) in vector DB.

**Online query:**

Given query q, compute query vector v_q = E(q).

Retrieve top-k documents by cosine similarity:

$$
\text{Top-k}(q) = \arg\max_{i=1..n}^k \frac{v_q \cdot v_i}{\|v_q\| \|v_i\|}
$$

Augment prompt:

$$
\text{prompt} = \text{template}(\text{context} = \{d_i\}_{i \in \text{Top-k}}, \text{query} = q)
$$

Generate answer:

$$
a = \text{LLM}(\text{prompt})
$$

**Retrieval quality metric: Recall@k**

$$
\text{Recall@k} = \frac{|\text{Relevant documents} \cap \text{Retrieved}_{k}|}{|\text{Relevant documents}|}
$$

Higher recall = more likely correct answer is in context.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Document corpus (toy example)**

Doc1: "RAG stands for Retrieval-Augmented Generation."
Doc2: "Retrieval-Augmented Generation combines LLMs with vector search."
Doc3: "GPT-4 is a large language model from OpenAI."
Doc4: "Vector databases store embeddings for similarity search."

#### **Step 2: User query**

q = "What is RAG?"

#### **Step 3: Embed and search (cosine similarity)**

Assume embeddings produce these similarities:

- Doc1: 0.95
- Doc2: 0.92
- Doc3: 0.20
- Doc4: 0.45

Top-2: Doc1, Doc2

#### **Step 4: Augment prompt**

```
Context:
1. RAG stands for Retrieval-Augmented Generation.
2. Retrieval-Augmented Generation combines LLMs with vector search.

Question: What is RAG?

Answer based only on the context above:
```

#### **Step 5: LLM generates**

Output: "RAG (Retrieval-Augmented Generation) is a technique that combines LLMs with vector search. It stands for Retrieval-Augmented Generation."

#### **Step 6: Compare to no-RAG**

Without RAG, the LLM might answer based on training data: "RAG is a technique for..." (could be correct but not grounded in specific docs). With RAG, the answer is explicitly grounded in retrieved documents.

---

### **7. How this appears inside neural networks or LLMs**

- **Naive RAG (retrieve then read):** Simplest form. Retrieve top-k, concatenate, generate. Works for many use cases.

- **Advanced RAG:** Re-rank retrieved documents (cross-encoder), filter by metadata, compress context (summarize retrieved docs before LLM).

- **Multi-step RAG (iterative retrieval):** LLM generates search queries based on initial retrieval, retrieves again. Used for complex questions.

- **Self-RAG:** LLM decides whether to retrieve at all, and whether retrieved content is relevant. Adds reflection steps.

- **HyDE (Hypothetical Document Embeddings):** LLM generates a hypothetical answer first, embeds that, then retrieves similar docs. Better for some queries.

- **RAG with citations:** LLM outputs citations linking answer sentences to retrieved document IDs.

- **RAG for code generation:** Retrieve relevant code snippets from codebase, generate new code.

- **Multimodal RAG:** Retrieve images, audio, or video alongside text. CLIP for cross-modal search.

---

### **8. Brain-like connection (cued recall with external memory)**

Human memory retrieval is not always successful. When you cannot recall a fact, you look it up (external memory). This is RAG. The brain's retrieval process is similar: you formulate a query (internal representation), search memory (hippocampus), and if unsuccessful, you use external aids (books, phone, internet). RAG automates this: the LLM is the brain, the vector database is external memory. The LLM does not need to know everything—it just needs to know how to retrieve and use information. This is how humans work. We have limited working memory but vast external storage (books, notes, search engines). RAG gives LLMs the same capability.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "RAG is just concatenating search results to the prompt. It is trivial."

_Why it is wrong:_ RAG seems simple, but production RAG has many subtleties: chunking strategy (how to split documents), embedding model choice (general vs domain-specific), re-ranking (retrieved docs may be noisy), context length management (top-k vs summarization), hybrid search (vector + keyword), and updating (incremental indexing). A naive RAG system (chunk by 512 chars, embed with generic model, top-5) often fails for real-world queries. Good RAG requires tuning each component. Moreover, RAG can be extended with iterative retrieval, self-ask, and agentic workflows (multi-step). It is not trivial.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  RAG is the difference between an LLM that guesses and    |
|  an LLM that knows. Retrieve first, then generate.        |
|  Ground answers in your documents, not the model's        |
|  fallible memory. Reduce hallucinations. Add citations.   |
|  Keep answers current without retraining. RAG is the      |
|  most practical LLM deployment pattern. Learn it. Use it. |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are building a RAG system for legal document retrieval. A lawyer asks: "What are the notice requirements for contract termination under Section 5?" Your corpus contains 10,000 legal documents. You retrieve top-5 chunks and get 3 relevant, 2 irrelevant.

**Question:** How can you improve retrieval quality? Name three strategies.

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Three strategies to improve retrieval quality:

1. **Hybrid search:** Combine vector similarity with keyword BM25. Legal terms ("Section 5," "notice requirements") are exact phrases that keyword search handles well. Use weighted sum: score = α·vector_sim + (1-α)·bm25. α=0.5 often works.

2. **Re-ranking:** Retrieve top-20 chunks with fast vector search, then re-rank with a slower but more accurate cross-encoder (e.g., BERT fine-tuned on legal relevance). Cross-encoder computes joint query-document relevance (not cosine). Move relevant docs to top.

3. **Chunking strategy:** Legal documents have structure (sections, subsections). Chunk by section boundaries, not fixed length. Include section headers in chunk. Use overlapping chunks (overlap = 10-20%) to avoid cutting off sentences.

4. **Query rewriting:** LLM expands query: "What are the notice requirements for contract termination under Section 5?" → generate multiple search queries: "Section 5 notice requirements termination", "contract termination notice period", "legal notice requirements for ending contract". Retrieve for each, deduplicate.

5. **Metadata filtering:** Pre-filter by document type (contract law), jurisdiction, date. Vector DB supports filtering before or after vector search (pre-filter usually faster).

In legal RAG, precision is critical (hallucination unacceptable). Use hybrid search + cross-encoder re-ranking + careful chunking. Expect >90% recall@5.
