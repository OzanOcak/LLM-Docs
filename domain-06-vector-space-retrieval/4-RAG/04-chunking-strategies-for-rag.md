# Chunking Strategies for RAG

## The Pizza Slicing Analogy

Imagine you have a large pizza (your document) and you need to serve it to guests who will each only eat a few slices (your LLM's context window). How do you slice it? Too big, and a slice won't fit on a plate. Too small, and guests lose the context of what's on adjacent slices. Some slices should overlap so you don't miss the pepperoni that straddles two slices. That's chunking—deciding how to split your documents into pieces that are useful for retrieval.

In RAG, chunking is arguably the most important design decision. Get it right, and your retrievals are relevant and complete. Get it wrong, and you either miss crucial information or overwhelm the LLM with irrelevant context. Different documents need different slicing strategies.

---

## What Is Chunking and Why It Matters

### The Core Problem

```python

def chunking_intro():
    """
    Why we need to chunk documents
    """
    print("Chunking: Slicing Documents for Retrieval")
    print("=" * 60)

    print("""
    Constraints we're working with:

    1. LLM context windows are limited (4K-200K tokens)
    2. Embedding models work best on focused text
    3. Retrieval finds SIMILAR chunks, not whole docs
    4. We need chunks that are self-contained

    A single document might be:
    • A 100-page PDF (too big for context)
    • A 5000-word article (too big for one embedding)
    • A book (definitely needs splitting!)

    Chunking turns one document into many searchable pieces.
    """)

chunking_intro()
```

### The Goldilocks Problem

```python

def goldilocks():
    """
    Finding the right chunk size
    """
    print("The Goldilocks Problem: Not Too Big, Not Too Small")
    print("=" * 60)

    print("""
    Chunks TOO SMALL (50-100 tokens):
    • ✅ Focused, specific
    • ✅ High precision retrieval
    • ❌ Lose context (sentence fragments)
    • ❌ May miss relationships
    • ❌ Many chunks to search

    Chunks TOO LARGE (2000+ tokens):
    • ✅ Full context preserved
    • ✅ Fewer chunks overall
    • ❌ Embedding becomes diluted
    • ❌ May contain irrelevant info
    • ❌ May exceed context window

    Chunks JUST RIGHT (300-800 tokens):
    • ✅ Focused enough for similarity
    • ✅ Complete enough for context
    • ✅ Fits in most context windows
    • ✅ Good balance of precision/recall
    """)

goldilocks()
```

---

## Common Chunking Strategies

### 1. Fixed-Size Chunking

The simplest approach: split by character or token count.

```python

def fixed_size():
    """
    Fixed-size chunking
    """
    print("Strategy 1: Fixed-Size Chunking")
    print("=" * 60)

    text = """
    The company remote work policy has been updated for 2024.
    Employees may work remotely up to three days per week.
    This requires manager approval. A home office stipend of
    $500 is available. Equipment must be returned if leaving
    the company. Security guidelines must be followed.
    VPN access is required. Regular team meetings must be
    attended in person at least twice per month.
    """

    chunk_size = 100
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    print(f"Original text length: {len(text)} chars")
    print(f"Chunk size: {chunk_size} chars")
    print(f"Number of chunks: {len(chunks)}")

    print("\nChunk 1:")
    print(f"  {chunks[0][:50]}...")
    print("\nChunk 2:")
    print(f"  {chunks[1][:50]}...")

    print("\n✅ Pros: Simple, fast, predictable")
    print("❌ Cons: May cut sentences mid-thought")

fixed_size()
```

### 2. Overlapping Chunks

Add overlap to prevent cutting off context.

```python

def overlapping():
    """
    Overlapping chunks
    """
    print("Strategy 2: Overlapping Chunks")
    print("=" * 60)

    text = "The quick brown fox jumps over the lazy dog. " * 10

    chunk_size = 50
    overlap = 20

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap

    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"Number of chunks: {len(chunks)}")

    print("\nChunk 1 and 2 show the overlap:")
    print(f"  1: {chunks[0][:40]}...")
    print(f"  2: {chunks[1][:40]}...")
    print(f"  Shared content: '{chunks[0][-20:]}' appears in both")

    print("\n✅ Pros: Preserves context across boundaries")
    print("❌ Cons: More chunks, some redundancy")

overlapping()
```

### 3. Recursive Character Text Splitting

Split intelligently trying to keep paragraphs/sentences intact.

```python

def recursive():
    """
    Recursive character text splitting
    """
    print("Strategy 3: Recursive Character Text Splitting")
    print("=" * 60)

    print("""
    LangChain's recursive splitter works like this:

    1. Try to split by paragraphs ("\n\n")
       If chunks still too big →
    2. Try to split by sentences (". " or "?\n")
       If chunks still too big →
    3. Try to split by phrases (", " or "; ")
       If chunks still too big →
    4. Split by words (" ")
       If chunks still too big →
    5. Split by characters

    This keeps semantic units intact when possible!
    """)

    separators = ["\n\n", "\n", ". ", "? ", "! ", ", ", "; ", " "]
    print(f"Separators in order: {separators}")

recursive()
```

### 4. Semantic Chunking

Use embeddings to find natural boundaries.

```python

def semantic():
    """
    Semantic chunking
    """
    print("Strategy 4: Semantic Chunking")
    print("=" * 60)

    print("""
    Idea: Use embeddings to detect topic boundaries

    1. Slide a window across the document
    2. Compute embeddings for each window
    3. When similarity drops sharply → topic change
    4. Split at topic boundaries

    Example:

    [Topic A window]──[Topic A window]──[Topic A window]
           │                  │                  │
    similarity: 0.95      0.94           0.45 ← Topic change!
                                            ↓
                                      Split here

    This creates chunks that are semantically coherent!
    """)

semantic()
```

### 5. Document Structure-Aware Chunking

Use the document's natural structure.

```python

def structure_aware():
    """
    Document structure-aware chunking
    """
    print("Strategy 5: Structure-Aware Chunking")
    print("=" * 60)

    print("""
    Different document types have natural chunk boundaries:

    📄 MARKDOWN:
    • Split by headers (#, ##, ###)
    • Keep sections intact

    📊 HTML:
    • Split by <section>, <div>, <p> tags
    • Preserve heading structure

    📚 PDFs:
    • Split by pages (with metadata)
    • Keep tables together

    💻 CODE:
    • Split by functions/classes
    • Keep imports with usage

    🗂️ JSON/XML:
    • Split by logical objects
    • Preserve nested structure
    """)

structure_aware()
```

### 6. Agentic Chunking (Advanced)

Use an LLM to decide where to split.

```python

def agentic():
    """
    Agentic chunking (using LLM)
    """
    print("Strategy 6: Agentic Chunking")
    print("=" * 60)

    print("""
    Use an LLM to find optimal chunk boundaries:

    1. Feed document to LLM with prompt:
       "Split this document into coherent chunks.
        Each chunk should be self-contained and
        focused on a single topic or concept."

    2. LLM returns split points

    3. Split at those points

    This is expensive but can be optimal for complex docs.
    Often used for initial indexing, not real-time.
    """)

agentic()
```

---

## Chunk Size Comparison

| Size       | Tokens    | Use Case                   | Pros             | Cons                 |
| ---------- | --------- | -------------------------- | ---------------- | -------------------- |
| Very small | 50-100    | FAQ items, definitions     | High precision   | Loses context        |
| Small      | 200-300   | Paragraphs, short sections | Good focus       | May miss connections |
| Medium     | 500-800   | Document sections          | Balanced         | Most common choice   |
| Large      | 1000-2000 | Full pages                 | Complete context | Embedding dilution   |
| Very large | 2000+     | Whole small docs           | Max context      | May exceed limits    |

---

## Chunking Strategy Decision Tree

```python

def decision_tree():
    """
    How to choose a chunking strategy
    """
    print("Chunking Strategy Decision Tree")
    print("=" * 60)

    print("""
    START HERE
        ↓
    What type of document?
    ├─► MARKDOWN/HTML → Use structure-aware (headers/tags)
    ├─► CODE → Use function/class boundaries
    ├─► PDF/Scanned → Use recursive with overlap
    ├─► Long prose → Try semantic chunking
    └─► Mixed/Unknown → Start with recursive (LangChain default)
        ↓
    How important is boundary accuracy?
    ├─► Critical → Try agentic chunking (expensive)
    └─► Normal → Recursive is fine
        ↓
    Test and iterate!
    """)

decision_tree()
```

---

## A Complete Chunking Example

```python

def chunking_demo():
    """
    Complete chunking example with different strategies
    """
    print("Chunking Strategies in Action")
    print("=" * 60)

    # Sample document
    doc = """
    CHAPTER 1: INTRODUCTION

    Remote work has become increasingly common since 2020.
    Many companies have adopted hybrid models. This document
    outlines our company's remote work policy.

    SECTION 1.1: ELIGIBILITY

    All full-time employees are eligible for remote work.
    Part-time employees may request consideration.
    Probationary period must be completed first.

    SECTION 1.2: SCHEDULE

    Employees may work remotely up to 3 days per week.
    Core hours are 10am-3pm in employee's timezone.
    Team meetings require in-person attendance twice monthly.

    CHAPTER 2: EQUIPMENT

    Company provides laptop and monitor.
    Employees are responsible for internet connection.
    $500 home office stipend available after 6 months.
    """

    print("Original document has clear structure (chapters/sections)")

    # Strategy 1: Fixed-size (ignores structure)
    print("\n" + "=" * 40)
    print("STRATEGY 1: Fixed-size (100 chars)")
    print("=" * 40)
    fixed = [doc[i:i+100] for i in range(0, len(doc), 100)]
    for i, chunk in enumerate(fixed[:3]):
        print(f"Chunk {i+1}: {chunk[:50]}...")

    # Strategy 2: Structure-aware
    print("\n" + "=" * 40)
    print("STRATEGY 2: Structure-aware (by sections)")
    print("=" * 40)
    sections = doc.split("\n\n")
    for i, section in enumerate(sections[:3]):
        if section.strip():
            print(f"Section {i+1}: {section[:50]}...")

chunking_demo()
```

---

## Best Practices by Document Type

| Document Type      | Recommended Strategy                 | Chunk Size | Notes                        |
| ------------------ | ------------------------------------ | ---------- | ---------------------------- |
| News articles      | Recursive by paragraphs              | 300-500    | Keep bylines with articles   |
| Academic papers    | Section-aware (Intro, Methods, etc.) | 500-800    | Keep citations with content  |
| Legal documents    | Clause-based with overlap            | 400-600    | Critical to preserve context |
| Code repos         | Function/class boundaries            | 200-400    | Include imports per file     |
| Conversations/chat | By turn/message                      | 200-300    | Keep threads together        |
| Product docs       | Feature-based sections               | 300-500    | Link related features        |
| Books              | Chapter + subsection                 | 800-1000   | May need hierarchical        |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Chunking is the most underrated factor in RAG quality:     ║
║                                                              ║
║  • Bad chunking = bad retrieval = bad answers               ║
║    No matter how good your embedding model or LLM is         ║
║                                                              ║
║  • Different documents need different strategies            ║
║    A recipe collection chunks differently than legal docs    ║
║                                                              ║
║  • Chunk size affects EVERYTHING                            ║
║    Too small → lose context, miss connections               ║
║    Too large → noisy embeddings, waste context window       ║
║                                                              ║
║  • Overlap prevents information falling through cracks      ║
║    Critical sentences that span chunk boundaries get kept    ║
║                                                              ║
║  • Structure-aware chunking preserves document logic        ║
║    Headers, lists, and tables need to stay together          ║
║                                                              ║
║  Get chunking right, and your RAG sings. Get it wrong,      ║
║  and you'll be debugging retrieval failures forever.         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Chunking is how we slice documents into pieces for retrieval—like cutting a pizza into slices that are small enough to handle but big enough to be satisfying

• Different strategies suit different documents—fixed-size for simplicity, overlapping to preserve context, recursive to respect structure, semantic for topic boundaries

• The right chunking strategy can make or break your RAG application—it affects retrieval quality, context utilization, and ultimately the accuracy of your LLM's answers

---

## Mental Hook

> "Chunking is like slicing a pizza—cut it wrong and someone gets just crust or just toppings, but slice it right and everyone gets a perfect piece with context of what's next to it."
