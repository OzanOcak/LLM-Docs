# How RAG Works Step-by-Step

## The Research Assistant Analogy

Imagine you have a brilliant research assistant. When you ask a question, they first dash to a massive library, grab the most relevant books and articles, and bring them back to you. Then you read those materials and craft your answer, citing your sources. That's exactly how RAG (Retrieval-Augmented Generation) works: a two-part system where a retriever finds relevant information, and an LLM reads it to generate an answer.

RAG isn't magic—it's a clear, step-by-step pipeline. Understanding each step helps you build better RAG applications, debug when things go wrong, and appreciate why this technique has become the standard for production LLM systems.

---

## The Two-Phase RAG Pipeline

RAG has two distinct phases: Indexing (done once ahead of time) and Querying (done for each user question).

```text

PHASE 1: INDEXING (Prepare the library)    PHASE 2: QUERYING (Answer questions)
──────────────────────────────────────    ──────────────────────────────────────
Documents ──→ Chunk ──→ Embed ──→ Store    User Question ──→ Embed ──→ Search
                                                    │                │
                                                    └────────┬───────┘
                                                             ↓
                                                   Retrieved Chunks
                                                             ↓
                                            Prompt = Question + Chunks
                                                             ↓
                                                          LLM
                                                             ↓
                                                        Answer
```

```python

def rag_pipeline_intro():
    """
    The complete RAG pipeline
    """
    print("The RAG Pipeline: Two Phases, One Goal")
    print("=" * 60)

    print("""
    PHASE 1: INDEXING (Do this once)
    ─────────────────────────────────────────────
    1. Load your documents (PDFs, websites, databases)
    2. Split them into chunks (small pieces)
    3. Create embeddings for each chunk
    4. Store embeddings in vector database

    PHASE 2: QUERYING (Do this per question)
    ─────────────────────────────────────────────
    5. User asks a question
    6. Embed the question
    7. Search vector DB for similar chunks
    8. Retrieve top-k most relevant chunks
    9. Create prompt with question + chunks
    10. LLM generates answer
    11. Return answer to user
    """)

rag_pipeline_intro()
```

---

## Phase 1: Indexing (Building Your Knowledge Base)

### Step 1: Load Documents

First, gather all the information you want your LLM to access.

```python

def step1_load():
    """
    Loading documents from various sources
    """
    print("Step 1: Load Your Documents")
    print("=" * 60)

    sources = [
        "Company policies (PDFs)",
        "Product documentation (websites)",
        "Customer support tickets (database)",
        "Recent news articles (RSS feeds)",
        "Internal wikis (Confluence, Notion)",
        "Scientific papers (PDFs)",
        "Emails and reports"
    ]

    print("RAG can work with ANY text source:")
    for src in sources:
        print(f"  • {src}")

    print("\nTools like LangChain have built-in loaders")
    print("for hundreds of different data sources.")

step1_load()
```

### Step 2: Split into Chunks

Documents are too long—we need smaller, focused pieces.

```python

def step2_chunk():
    """
    Chunking documents into manageable pieces
    """
    print("Step 2: Split into Chunks")
    print("=" * 60)

    long_doc = """
    The company remote work policy has been updated for 2024.
    Employees may work remotely up to three days per week.
    This requires manager approval. A home office stipend of
    $500 is available. Equipment must be returned if leaving
    the company. Security guidelines must be followed.
    VPN access is required. Regular team meetings must be
    attended in person at least twice per month.
    """ * 5  # Simulate a long document

    print(f"Original document: {len(long_doc)} characters")

    # Chunk into smaller pieces
    chunk_size = 200
    chunks = [long_doc[i:i+chunk_size] for i in range(0, len(long_doc), chunk_size)]

    print(f"After chunking: {len(chunks)} chunks")
    print(f"Sample chunk 1: {chunks[0][:100]}...")

    print("\nChunking strategies:")
    print("  • Fixed size (e.g., 500 characters)")
    print("  • Sentence-based")
    print("  • Paragraph-based")
    print("  • Recursive (smart splitting)")

step2_chunk()
```

### Step 3: Create Embeddings

Convert each chunk into a vector that captures its meaning.

```python

def step3_embed():
    """
    Creating embeddings for each chunk
    """
    print("Step 3: Create Embeddings")
    print("=" * 60)

    chunks = [
        "Remote work policy allows 3 days WFH per week",
        "$500 home office stipend available",
        "Manager approval required for remote work",
        "VPN must be used when working remotely"
    ]

    print(f"Chunks to embed: {len(chunks)}")
    print("\nUsing an embedding model (e.g., text-embedding-ada-002)")
    print("Each chunk → vector (e.g., 1536 dimensions)")

    # Simulated embeddings
    import numpy as np
    np.random.seed(42)

    for i, chunk in enumerate(chunks):
        # Simulate a 4-dim embedding for demo
        embedding = np.random.randn(4).round(2)
        print(f"\n  Chunk {i+1}: '{chunk[:30]}...'")
        print(f"  → Vector: {embedding}")

step3_embed()
```

### Step 4: Store in Vector Database

Save vectors and original text for fast retrieval.

```python

def step4_store():
    """
    Storing vectors in a database
    """
    print("Step 4: Store in Vector Database")
    print("=" * 60)

    print("""
    Vector database record for each chunk:

    ┌─────────────────────────────────────────┐
    │ ID: chunk_123                           │
    ├─────────────────────────────────────────┤
    │ Vector: [0.23, -0.45, 0.67, ...]        │ ← 768/1536 dims
    ├─────────────────────────────────────────┤
    │ Metadata: {                              │
    │   "text": "Remote work policy...",      │ ← Original text
    │   "source": "HR_policy_2024.pdf",       │ ← For citation
    │   "page": 3,                             │
    │   "date": "2024-01-15"                   │
    │ }                                        │
    └─────────────────────────────────────────┘

    This is what makes retrieval fast—indexing at insert time!
    """)

step4_store()
```

---

## Phase 2: Querying (Answering Questions)

### Step 5: User Asks a Question

```python

def step5_question():
    """
    User submits a question
    """
    print("Step 5: User Asks a Question")
    print("=" * 60)

    questions = [
        "What's the remote work policy?",
        "How much is the home office stipend?",
        "Do I need manager approval for WFH?",
        "What are the VPN requirements?"
    ]

    print("Example questions RAG can answer:")
    for q in questions:
        print(f"  • {q}")

step5_question()
```

### Step 6: Embed the Question

Convert the question to a vector using the SAME embedding model.

```python

def step6_embed_question():
    """
    Embedding the user's question
    """
    print("Step 6: Embed the Question")
    print("=" * 60)

    question = "What's the current remote work policy?"

    print(f"Question: '{question}'")
    print("\nUsing the SAME embedding model as Step 3")
    print("This ensures question and chunks are in the same vector space")

    # Simulated embedding
    import numpy as np
    np.random.seed(123)
    q_vector = np.random.randn(4).round(2)

    print(f"\nQuestion vector: {q_vector}")
    print("(Same format as document chunk vectors)")

step6_embed_question()
```

### Step 7: Search Vector Database

Find chunks most similar to the question.

```python

def step7_search():
    """
    Searching for relevant chunks
    """
    print("Step 7: Search Vector Database")
    print("=" * 60)

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Simulated data
    np.random.seed(42)
    chunks = [
        "Remote work policy allows 3 days WFH per week",
        "$500 home office stipend available",
        "Manager approval required for remote work",
        "VPN must be used when working remotely",
        "Health insurance covers dental and vision"
    ]

    # Simulated embeddings
    chunk_vectors = np.random.randn(5, 4)
    question_vector = np.random.randn(1, 4)

    # Calculate similarities
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]

    print("Finding chunks most relevant to the question:")
    for i, (chunk, sim) in enumerate(zip(chunks, similarities)):
        print(f"  {sim:.3f}: {chunk}")

    print("\nTop 2 most relevant chunks:")
    top_indices = np.argsort(similarities)[-2:][::-1]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {chunks[idx]} (score: {similarities[idx]:.3f})")

step7_search()
```

### Step 8: Retrieve Top-K Chunks

Get the actual text of the most relevant chunks.

```python

def step8_retrieve():
    """
    Retrieving the actual chunk text
    """
    print("Step 8: Retrieve Top-K Chunks")
    print("=" * 60)

    # Retrieved from vector DB
    retrieved_chunks = [
        {
            "text": "Remote work policy allows 3 days WFH per week. Manager approval required.",
            "source": "HR_policy_2024.pdf",
            "score": 0.92
        },
        {
            "text": "$500 home office stipend available for remote workers. Equipment must be returned if leaving.",
            "source": "benefits_guide.pdf",
            "score": 0.87
        }
    ]

    k = 2
    print(f"Retrieved top-{k} chunks with metadata:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\n  {i}. [Score: {chunk['score']}]")
        print(f"     Text: {chunk['text']}")
        print(f"     Source: {chunk['source']}")

step8_retrieve()
```

### Step 9: Create Augmented Prompt

Combine the question with retrieved chunks.

```python

def step9_prompt():
    """
    Creating the augmented prompt
    """
    print("Step 9: Create Augmented Prompt")
    print("=" * 60)

    question = "What's the current remote work policy?"
    chunks = [
        "Remote work policy allows 3 days WFH per week. Manager approval required.",
        "$500 home office stipend available for remote workers."
    ]

    print("Original prompt (just question):")
    print(f"  {question}")

    print("\nAugmented prompt with context:")
    print("-" * 40)
    print("Context information:")
    for chunk in chunks:
        print(f"• {chunk}")
    print(f"\nQuestion: {question}")
    print("\nAnswer based on the context above:")
    print("-" * 40)

    print("\nNow the LLM has ALL the information it needs!")

step9_prompt()
```

### Step 10: LLM Generates Answer

```python

def step10_generate():
    """
    LLM generates answer from augmented prompt
    """
    print("Step 10: LLM Generates Answer")
    print("=" * 60)

    print("""
    LLM sees:

    Context:
    • Remote work policy allows 3 days WFH per week.
      Manager approval required.
    • $500 home office stipend available for remote workers.

    Question: What's the current remote work policy?

    LLM generates:

    "Based on the company documents, the current remote work
    policy allows employees to work from home up to 3 days per
    week with manager approval. A $500 home office stipend is
    also available for eligible remote workers."

    Source: HR_policy_2024.pdf, benefits_guide.pdf
    """)

step10_generate()
```

### Step 11: Return Answer to User

```python

def step11_return():
    """
    Return the answer to the user
    """
    print("Step 11: Return Answer to User")
    print("=" * 60)

    print("""
    Final response to user:

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   Based on the company documents, the current       │
    │   remote work policy allows employees to work from  │
    │   home up to 3 days per week with manager approval. │
    │   A $500 home office stipend is also available for  │
    │   eligible remote workers.                          │
    │                                                     │
    │   Sources:                                          │
    │   • HR_policy_2024.pdf                              │
    │   • benefits_guide.pdf                              │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)

step11_return()
```

---

## The Complete RAG Pipeline (All Steps)

```python

def complete_pipeline():
    """
    The entire RAG pipeline summarized
    """
    print("RAG PIPELINE: COMPLETE VIEW")
    print("=" * 60)

    steps = [
        ("PHASE 1: INDEXING", [
            "1. Load documents (PDFs, websites, databases)",
            "2. Split into chunks (500-1000 tokens each)",
            "3. Create embeddings for each chunk",
            "4. Store vectors in database with metadata"
        ]),
        ("PHASE 2: QUERYING", [
            "5. User asks question",
            "6. Embed the question (same model)",
            "7. Search vector DB for similar chunks",
            "8. Retrieve top-k most relevant chunks",
            "9. Create prompt: question + retrieved chunks",
            "10. LLM generates answer using context",
            "11. Return answer with sources"
        ])
    ]

    for phase, substeps in steps:
        print(f"\n{phase}:")
        for step in substeps:
            print(f"  {step}")

complete_pipeline()
```

---

## Key Parameters in RAG

| Parameter            | What It Controls                  | Typical Value                | Tradeoff                                      |
| -------------------- | --------------------------------- | ---------------------------- | --------------------------------------------- |
| Chunk size           | How big each document piece is    | 500-1000 tokens              | Smaller = more focused, larger = more context |
| Chunk overlap        | Overlap between chunks            | 10-20%                       | Prevents cutting mid-sentence                 |
| Top-k                | Number of chunks to retrieve      | 3-10                         | Higher = more context, slower, may confuse    |
| Similarity threshold | Minimum relevance score           | 0.7-0.8                      | Higher = fewer but better results             |
| Embedding model      | Quality of vector representations | text-embedding-ada-002, etc. | Better model = better retrieval               |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Understanding each step of RAG lets you:                    ║
║                                                              ║
║  • Build better applications—tune chunk size, top-k,        ║
║    and embedding models for your specific use case           ║
║                                                              ║
║  • Debug when things go wrong—is retrieval failing?         ║
║    Is the LLM ignoring context? Each step can be checked     ║
║                                                              ║
║  • Optimize for speed and cost—caching, smaller models,     ║
║    better chunking strategies all make a difference          ║
║                                                              ║
║  • Add citations and sources—retrieved chunks become         ║
║    verifiable references, building user trust                ║
║                                                              ║
║  RAG isn't magic—it's engineering. And understanding the     ║
║  steps turns you from a user into a builder.                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• RAG has two phases: Indexing (prepare your knowledge base once) and Querying (answer each question by retrieving relevant information)

• The indexing phase loads documents, chunks them, creates embeddings, and stores them in a vector database—building your searchable knowledge base

• The querying phase embeds the user's question, finds similar chunks, retrieves them, augments the prompt, and generates an answer grounded in those sources

---

## Mental Hook

> "RAG is like having a librarian who, before you answer a question, dashes to the stacks, grabs the most relevant books, puts sticky notes on the important pages, and hands them to you—so you can give an answer that's accurate, current, and citable."
