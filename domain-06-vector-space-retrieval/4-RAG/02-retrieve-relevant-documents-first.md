# The Solution: Retrieve Relevant Documents First

## The Open-Book Exam Analogy

Imagine you're taking a history exam. In a traditional (closed-book) test, you have to rely entirely on what you memorized months ago—like an LLM with only its training data. But in an open-book exam, you're given relevant passages to reference while answering. You can look up specific facts, quote sources, and give much more accurate answers. That's Retrieval-Augmented Generation (RAG) : instead of making the LLM guess, we first retrieve relevant documents and let it use them as reference material.

RAG is the solution to the knowledge cutoff problem. By retrieving current information from external sources and feeding it to the LLM as context, we give the model a window into the present. This turns a frozen-in-time model into one that can answer questions about today's news, your company's internal docs, or any other up-to-date information.

---

## The RAG Solution at a Glance

### Before RAG: Closed-Book

```text

User Question
    ↓
[LLM - relies only on training data]
    ↓
Answer (may be outdated or hallucinated)

### After RAG: Open-Book
```

```text

User Question
    ↓
[Retrieve relevant documents] ← External knowledge base
    ↓
Question + Retrieved Documents
    ↓
[LLM - reads documents + answers]
    ↓
Answer (accurate, citable, up-to-date)
```

```python

def rag_solution_intro():
    """
    The core idea of RAG
    """
    print("RAG: Giving LLMs an Open Book")
    print("=" * 60)

    print("""
    Instead of asking the LLM to know everything:

    ❌ "What's the current CEO of Twitter?"
       (LLM trained in 2021: "Jack Dorsey")

    We give it the information it needs:

    ✓ Retrieve: Find recent article about Twitter/X
      "Elon Musk acquired Twitter in 2022, Linda Yaccarino is CEO"

    ✓ Generate: LLM reads this + answers question
      "Linda Yaccarino is the current CEO of X (formerly Twitter)"

    The LLM doesn't need to KNOW—it just needs to READ and UNDERSTAND!
    """)

rag_solution_intro()
```

---

## How RAG Works Step by Step

### Step 1: Index Your Documents

```python

def step1_index():
    """
    First step: prepare your knowledge base
    """
    print("Step 1: Index Your Documents")
    print("=" * 60)

    print("""
    Before you can retrieve, you need something to retrieve FROM.

    1. Collect documents:
       • Company policies
       • Recent news articles
       • Product documentation
       • Customer support tickets
       • Any information you want the LLM to access

    2. Split into chunks:
       • Documents → smaller pieces (e.g., 500 words each)
       • Preserves context while keeping chunks focused

    3. Create embeddings:
       • Each chunk → vector (using embedding model)
       • Store in vector database

    Result: Searchable knowledge base of your information!
    """)

step1_index()
```

### Step 2: Retrieve Relevant Documents

```python

def step2_retrieve():
    """
    Second step: find relevant information for each query
    """
    print("Step 2: Retrieve Relevant Documents")
    print("=" * 60)

    query = "What's the company policy on remote work?"

    print(f"User Question: '{query}'\n")

    # Simulated retrieval
    documents = [
        "Policy 101: Remote work policy updated March 2024: Employees may work remotely up to 3 days per week...",
        "HR Guidelines: Requesting remote work requires manager approval...",
        "Benefits: Home office stipend available for remote workers..."
    ]

    print("Vector database finds most relevant chunks:")
    for i, doc in enumerate(documents, 1):
        print(f"\n  {i}. {doc[:100]}...")

    print("\n✅ Retrieved documents are RELEVANT to the query")

step2_retrieve()
```

### Step 3: Augment the Prompt

```python

def step3_augment():
    """
    Third step: add retrieved docs to the prompt
    """
    print("Step 3: Augment the Prompt")
    print("=" * 60)

    query = "What's the company policy on remote work?"
    documents = [
        "Remote work policy updated March 2024: Employees may work remotely up to 3 days per week...",
        "Requesting remote work requires manager approval..."
    ]

    print("Original prompt would just be the question.")
    print("\nAugmented prompt with context:")
    print("-" * 40)
    print("Context information:")
    for doc in documents:
        print(f"• {doc[:100]}...")
    print(f"\nQuestion: {query}")
    print("Answer based on the context above:")
    print("-" * 40)

    print("\nNow the LLM has the information it needs RIGHT THERE!")

step3_augment()
```

### Step 4: Generate Answer

```python

def step4_generate():
    """
    Fourth step: LLM generates answer using retrieved docs
    """
    print("Step 4: Generate Answer")
    print("=" * 60)

    print("""
    LLM sees:

    Context:
    • Remote work policy updated March 2024: Employees may work
      remotely up to 3 days per week with manager approval.
    • Home office stipend of $500 available for remote workers.

    Question: What's the company policy on remote work?

    LLM generates:

    "The company's remote work policy (updated March 2024) allows
    employees to work remotely up to 3 days per week with manager
    approval. A $500 home office stipend is also available for
    remote workers."

    ✅ Accurate
    ✅ Up-to-date
    ✅ Citable (sources provided)
    """)

step4_generate()
```

---

## Complete RAG Pipeline Visualization

```text

                    RAG PIPELINE
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   INDEXING PHASE            QUERY PHASE             │
    │   (one-time)                (per question)          │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │   Documents                   User Question         │
    │       ↓                            ↓                 │
    │   Split into chunks           Embed question        │
    │       ↓                            ↓                 │
    │   Embed chunks                 Vector DB            │
    │       ↓                       Similarity Search     │
    │   Store in Vector DB ←─────────────┘                │
    │                              ↓                       │
    │                         Top relevant chunks         │
    │                              ↓                       │
    │                    ┌─────────────────────┐          │
    │                    │ Prompt = Question + │          │
    │                    │    Retrieved Docs   │          │
    │                    └─────────────────────┘          │
    │                              ↓                       │
    │                           LLM                        │
    │                              ↓                       │
    │                         Answer                       │
    │                                                      │
    └─────────────────────────────────────────────────────┘
```

---

## A Simple RAG Implementation

```python

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rag_demo():
    """
    Simplified RAG implementation
    """
    print("Simple RAG Demo")
    print("=" * 60)

    # Our document collection
    documents = [
        "The company remote work policy allows 3 days WFH per week.",
        "Health insurance covers dental and vision for all employees.",
        "The 401k matching program contributes up to 5% of salary.",
        "Vacation policy: 20 days PTO per year for full-time staff.",
        "Remote work stipend of $500 available for home office setup.",
        "Recent update March 2024: Remote work now requires manager approval."
    ]

    # Simulated embeddings (in reality, these would be 768D vectors)
    # For demo, we'll use 2D approximations
    doc_embeddings = np.array([
        [0.9, 0.8],  # remote work policy
        [0.1, 0.9],  # health insurance
        [0.2, 0.1],  # 401k
        [0.3, 0.8],  # vacation
        [0.8, 0.7],  # remote stipend
        [0.9, 0.9]   # recent remote update
    ])

    # User query
    query = "What's the current remote work policy?"
    print(f"Query: '{query}'\n")

    # Simulated query embedding
    query_embedding = np.array([0.85, 0.85])

    # Retrieve (find most similar documents)
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_k = 3
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    print(f"Retrieved top {top_k} documents:")
    retrieved_docs = []
    for i, idx in enumerate(top_indices, 1):
        print(f"\n  {i}. [Score: {similarities[idx]:.3f}]")
        print(f"     {documents[idx]}")
        retrieved_docs.append(documents[idx])

    # Augment prompt
    print("\n" + "=" * 40)
    print("Augmented Prompt:")
    print("=" * 40)
    print("\nContext:")
    for doc in retrieved_docs:
        print(f"• {doc}")
    print(f"\nQuestion: {query}")
    print("\nAnswer based on the context above:")

    # Generate answer (simulated)
    print("\n" + "=" * 40)
    print("LLM Response:")
    print("=" * 40)
    print("""
    Based on the company documents:

    The current remote work policy allows employees to work remotely
    up to 3 days per week. A $500 home office stipend is available.
    As of March 2024, remote work now requires manager approval.
    """)

rag_demo()
```

---

## What RAG Solves

| Problem            | Without RAG                      | With RAG                           |
| ------------------ | -------------------------------- | ---------------------------------- |
| Knowledge cutoff   | Can't answer about recent events | Retrieves current documents        |
| Private data       | Never seen company docs          | Can access internal knowledge base |
| Hallucinations     | Makes up plausible lies          | Grounded in retrieved facts        |
| Verifiability      | Can't show sources               | Retrieved docs = citations         |
| Specific knowledge | Only general training data       | Can access domain-specific docs    |

### Knowledge Cutoff Solved

```python

def cutoff_solved():
    """
    How RAG fixes the cutoff problem
    """
    print("RAG: Solving the Knowledge Cutoff")
    print("=" * 60)

    cutoff_date = "September 2021"
    today = "March 2024"

    print(f"Model trained until: {cutoff_date}")
    print(f"Today: {today}")
    print(f"Gap: 2.5 years of unknown information")

    print("\nWithout RAG:")
    print("  • User asks about recent event")
    print("  • Model: "I don't know" or hallucinates")

    print("\nWith RAG:")
    print("  • Recent articles indexed in vector DB")
    print("  • Retrieval finds relevant 2024 documents")
    print("  • Model reads them and answers correctly")
    print("  • Knowledge cutoff effectively ELIMINATED!")

cutoff_solved()
```

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  RAG transforms LLMs from frozen statues into living,       ║
║  breathing assistants that can access current information:  ║
║                                                              ║
║  • No more knowledge cutoff—models can answer about         ║
║    today's news, events, and developments                    ║
║                                                              ║
║  • Enterprise data becomes accessible—internal docs,         ║
║    private knowledge bases, customer information             ║
║                                                              ║
║  • Hallucinations drastically reduced—answers are            ║
║    grounded in retrieved facts, not made up                   ║
║                                                              ║
║  • Verifiable responses—every answer can cite its            ║
║    sources, building trust with users                        ║
║                                                              ║
║  This is why RAG is the most important technique             ║
║  for deploying LLMs in production today.                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• RAG solves the knowledge cutoff by retrieving relevant documents first—like giving a student an open book for an exam, the model can look up current information instead of relying on outdated training data

• The process has four steps: index documents (chunk + embed), retrieve relevant chunks for each query, augment the prompt with retrieved text, and generate an answer grounded in those sources

• RAG eliminates hallucinations, enables access to private data, and makes answers verifiable—it's the most important technique for deploying LLMs in production today

---

## Mental Hook

> "RAG is like giving your LLM a librarian who instantly finds the right book before it answers—instead of guessing about 2024 events based on 2021 knowledge, it reads the latest news and gives you accurate, citable answers."
