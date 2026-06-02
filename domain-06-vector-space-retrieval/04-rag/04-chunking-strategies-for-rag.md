# Chunking strategies for RAG

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: RAG (Retrieval-Augmented Generation)**

---

### **1. Why this concept matters**

You have a 500-page PDF. You cannot feed it whole into an LLM—context window too small. You split it into chunks, embed each, retrieve relevant chunks at query time. But how do you split? Too small → each chunk misses context; answers span multiple chunks. Too large → each chunk has irrelevant information; retrieval precision drops. Chunking is the silent killer of RAG systems. Bad chunking breaks retrieval. Good chunking is invisible. This section covers chunking strategies: fixed-size, semantic, recursive, and document-structure aware.

---

### **2. Core idea**

**Chunking splits documents into smaller pieces for embedding and retrieval. Optimal chunk size balances context preservation (larger chunks) against retrieval precision (smaller chunks), with overlap preventing information loss at boundaries.**

---

### **3. Concrete analogy**

Imagine you are cutting a long sausage into pieces for a sandwich.

- **Too small (10mm):** You get many tiny slices. The flavor of each slice is incomplete. You lose the big picture (context).
- **Too large (100mm):** You get a few huge chunks. Each chunk has too much, making it hard to pick the right flavor for your sandwich (low precision).
- **Just right (50mm):** Each slice captures a complete section of flavor. You can pick the right slices for your sandwich.

Now imagine the sausage has natural boundaries (knots). Cutting at knots preserves structure (sections, paragraphs). This is semantic chunking.

Overlap: When you cut, you leave a 10mm overlap between slices. If a sentence straddles a cut, it appears in both slices. No information lost.

---

### **4. ASCII diagram**

```
Chunking strategies comparison:

    Document text (continuous):
    "Paragraph 1. This is the first paragraph. It contains multiple sentences.
     Paragraph 2. This is the second paragraph. It discusses chunking.
     Paragraph 3. This is the third paragraph. The conclusion."


    Fixed-size (L=50 chars, O=10):
    ┌────────────────────────────────────┐
    │ Chunk1: "Paragraph 1. This is the  │
    │ first paragraph. It contains"      │
    ├────────────────────────────────────┤
    │ Chunk2: "It contains multiple      │
    │ sentences. Paragraph 2. This is"   │ (overlap "It contains")
    ├────────────────────────────────────┤
    │ Chunk3: "is the second paragraph.  │
    │ It discusses chunking. Paragraph"  │
    └────────────────────────────────────┘


    Semantic (split at sentence boundaries):
    ┌────────────────────────────────────┐
    │ Chunk1: "Paragraph 1. This is the  │
    │ first paragraph."                  │
    ├────────────────────────────────────┤
    │ Chunk2: "It contains multiple      │
    │ sentences."                        │
    ├────────────────────────────────────┤
    │ Chunk3: "Paragraph 2. This is the  │
    │ second paragraph."                 │
    └────────────────────────────────────┘


    Recursive (document structure):

    Document
    ├── Section 1
    │   ├── Paragraph 1
    │   └── Paragraph 2
    └── Section 2
        ├── Paragraph 3
        └── Paragraph 4

    Chunk = Section if small, else Paragraph.
    Preserves hierarchy. Metadata: section title, subsection.


Overlap prevents boundary loss:

    No overlap:            With overlap:
    [A][B][C]              [A][AB][B][BC][C]
    Sentence at boundary   Sentence appears in both chunks
    lost.                  preserved.
```

---

### **5. Mathematical formulation**

**Fixed-size chunking (most common):**

Given document text as token sequence t₁...tₙ, chunk size L (tokens), overlap O (tokens).

Chunk i: tokens t*{i\*(L-O)+1} to t*{i\*(L-O)+L}

Number of chunks: ⌈(n - L)/(L - O)⌉ + 1

**Semantic chunking:**

Split at sentence boundaries (., !, ?, \n\n). Group sentences until chunk size ≈ L. Preserves linguistic units.

**Recursive chunking:**

Use document structure (headings, sections). If section size < L, keep whole section. Else, split section by paragraphs. If paragraph too large, fall back to fixed-size.

**Optimal chunk size heuristic:**

- For dense embeddings (general text): 200-500 tokens
- For code: 50-200 tokens (functions, classes)
- For legal/medical: 500-1000 tokens (long contexts)
- For question-answering: smaller chunks (100-200) for precise retrieval
- For summarization: larger chunks (500-1000) for complete context

**Overlap recommendation:** 10-20% of chunk size (e.g., 50 tokens overlap for 500 token chunks).

---

### **6. Worked example (step-by-step)**

#### **Step 1: Input document**

```
Section 1: Introduction
RAG is retrieval-augmented generation. It combines LLMs with vector search.
RAG reduces hallucinations.

Section 2: Chunking
Chunking splits documents into pieces. Fixed-size chunking uses tokens.
Overlap prevents information loss. Semantic chunking uses sentences.
```

#### **Step 2: Fixed-size (L=50 chars, O=10)**

Chunk1: "Section 1: IntroductionRAG is retrieval-augmented gene" (cuts word)
Chunk2: "ted generation. It combines LLMs with vector search. " (overlap "gene"?? This is messy)

**Problem:** Cuts words, loses section boundaries.

#### **Step 3: Semantic chunking (split at \n\n, then sentences)**

Chunk1: "Section 1: Introduction\nRAG is retrieval-augmented generation. It combines LLMs with vector search. RAG reduces hallucinations."

Chunk2: "Section 2: Chunking\nChunking splits documents into pieces. Fixed-size chunking uses tokens. Overlap prevents information loss. Semantic chunking uses sentences."

**Better:** Preserves section headers, complete sentences.

#### **Step 4: Recursive chunking**

Check section 1 size (150 chars) < L (500) → keep whole section as chunk.
Same for section 2.
Result: two chunks (one per section). Perfect.

#### **Step 5: Overlap for safety**

If a paragraph spans two sections (rare), overlap captures it. For semantic/recursive, overlap often unnecessary because boundaries are natural.

#### **Step 6: Metadata**

Store with each chunk: source document, section title, page number, chunk index.

Query retrieves chunk, user sees citation.

---

### **7. How this appears inside neural networks or LLMs**

- **Fixed-size chunking (LangChain default):** `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`. Simple, works for general text.

- **Semantic chunking:** `SemanticChunker` (LangChain experimental). Uses embedding similarity between sentences. Split when similarity drops.

- **Document structure aware:** `MarkdownHeaderTextSplitter` (headers), `PythonCodeSplitter` (functions), `RecursiveJsonSplitter`.

- **Small chunks (100-200 tokens):** Better for question-answering (retrieve exact answer sentence). Risk: missing context.

- **Large chunks (500-1000 tokens):** Better for summarization (complete sections). Risk: retrieval includes irrelevant text.

- **Overlap tuning:** 10-20%. Too low → boundary loss. Too high → duplicated content, wasted context window.

- **Metadata propagation:** When chunking, propagate metadata (filename, page, section title) to each chunk. Critical for citation.

- **Chunk size and embedding model:** Embedding models have max sequence length (SBERT: 512 tokens, OpenAI ada: 8191 tokens). Chunk size must fit.

---

### **8. Brain-like connection (episodic memory segmentation)**

The brain does not store memories as continuous video. It segments experience into events (chunks) at boundaries: scene changes, topic shifts, temporal gaps. This is chunking. Each event is stored as a separate memory (chunk). When recalling, the brain retrieves relevant events (chunks), not raw sensory data. Overlap: adjacent events share context (e.g., leaving a room and entering another). The brain's event boundaries are semantic (topic change), not fixed-size. Patients with hippocampal damage cannot segment experience into events—they remember continuous, undifferentiated streams (no chunks). This impairs retrieval. RAG chunking mimics this cognitive segmentation.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Chunk size is a hyperparameter you set once and forget. 500 tokens works for everything."

_Why it is wrong:_ Optimal chunk size depends on document type, query type, and embedding model. Code needs small chunks (functions). Legal contracts need larger chunks (clauses). Question-answering needs small chunks (100-200). Summarization needs large chunks (500-1000). Chunking for RAG over PDFs (extracted text may have broken sentences) requires different strategy than HTML (structured). Always experiment. Evaluate retrieval recall@k for your use case. Change chunk size; measure. Do not assume default works.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Bad chunking breaks RAG. Chunks too small: answers       |
|  straddle chunks, never retrieved. Chunks too large:      |
|  irrelevant text dilutes similarity, wrong chunks         |
|  retrieved. Overlap too low: boundary sentences lost.     |
|  Overlap too high: context wasted, duplicates. Chunking   |
|  is not a detail—it is the foundation of retrieval.       |
|  Get it wrong and your RAG fails silently. Get it right   |
|  and retrieval just works. Test your chunking.            |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are building RAG for a codebase with 10,000 Python functions. Each function is 50-200 lines. You need to answer questions like "Which function handles user authentication?" and "What does `validate_token()` do?"

**Question:** What chunking strategy would you use? What chunk size? Would you use overlap? What metadata would you store with each chunk?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

**Chunking strategy:** Use a code-aware splitter (e.g., LangChain's `PythonCodeSplitter` or tree-sitter). Split at function/class boundaries, not fixed token count. Each chunk = one function (including docstring, signature, body). For very long functions (>500 tokens), split into logical blocks (but rare).

**Chunk size:** Not fixed token size; use structural units (functions). However, for embedding, truncate to model's max length (e.g., 512 tokens for SBERT). Most Python functions fit.

**Overlap:** Minimal or none. Functions are semantically independent. Overlap would duplicate code across chunks (waste). However, for class methods, consider including class name in each method's metadata (not chunk overlap).

**Metadata per chunk:**

- `file_path`: where function lives
- `function_name`: `validate_token`
- `class_name` (if method)
- `docstring` (first line for description)
- `parameters` and `return_type` (parsed from signature)
- `line_start`, `line_end` (for citation)
- `dependencies` (functions called within)

**Additional strategies:**

- For queries like "function that handles X", also index natural language descriptions of each function (synthetic queries).
- For code search, use code-specific embedding models (`codebert`, `graphcodebert`, OpenAI `text-embedding-3-small`).
- Hybrid search: vector similarity + exact symbol name matching (keyword).

**Evaluation:** Test retrieval recall@5 on a held-out set of function queries. Adjust chunking based on results. For code, function-level chunking is standard and effective. Avoid fixed-size chunking (breaks functions in half).
