# Hybrid Search (Combining Vector + Keyword)

## The Detective Duo Analogy

Imagine two detectives solving a case. One is a master of meaning and context—she can find connections no one else sees, but sometimes misses exact details (vector search). The other is a stickler for precise facts—he catches every exact match but misses the bigger picture (keyword search). Together, they're unstoppable. That's hybrid search: combining the semantic understanding of vector search with the precision of keyword search.

In RAG, hybrid search gives you the best of both worlds. Vector search finds documents *about* your topic, even if they use different words. Keyword search ensures you don't miss documents containing exact phrases, product codes, or proper names. Together, they produce richer, more accurate retrieval.

---

## The Problem: Neither Search Type Is Perfect

### Vector Search Weaknesses

```python

def vector_weaknesses():
    """
    Where vector search falls short
    """
    print("Vector Search: Great at Meaning, Weak at Exactness")
    print("=" * 60)

    print("""
    ✅ VECTOR SEARCH EXCELS AT:
    • Synonyms ("car" vs "automobile")
    • Concepts ("animals" finding "cats" and "dogs")
    • Semantic similarity
    • Understanding intent

    ❌ VECTOR SEARCH STRUGGLES WITH:
    • Exact product codes ("iPhone 15 Pro Max")
    • Proper names ("Wolfgang Amadeus Mozart")
    • Rare terms ("zygote")
    • Quotes and phrases
    • When you need PRECISE matches
    """)

vector_weaknesses()
```

### Keyword Search Weaknesses

```python

def keyword_weaknesses():
    """
    Where keyword search falls short
    """
    print("Keyword Search: Great at Exactness, Weak at Meaning")
    print("=" * 60)

    print("""
    ✅ KEYWORD SEARCH EXCELS AT:
    • Exact phrase matching
    • Product codes and IDs
    • Proper names
    • Rare technical terms
    • Boolean logic (AND, OR, NOT)

    ❌ KEYWORD SEARCH STRUGGLES WITH:
    • Synonyms ("car" misses "automobile")
    • Context ("bank" as river vs money)
    • Semantic meaning
    • Understanding intent
    • Misspellings
    """)

keyword_weaknesses()
```

---

## Hybrid Search: The Best of Both Worlds

### How Hybrid Search Works

```python

def hybrid_intro():
    """
    The core idea of hybrid search
    """
    print("Hybrid Search: Combining Strengths")
    print("=" * 60)

    print("""
    Hybrid search runs TWO searches and combines results:

                    User Query
                        ↓
            ┌───────────┴───────────┐
            ↓                       ↓
    Vector Search           Keyword Search
    (semantic)              (exact match)
            ↓                       ↓
    ┌───────────┴───────────┐
    ↓                       ↓
    Combine & Rerank
            ↓
    Final Results

    Each search type catches what the other misses!
    """)

hybrid_intro()
```

### The Combination Formula

```python

def combination_formula():
    """
    How scores are combined
    """
    print("Combining Scores: The Hybrid Formula")
    print("=" * 60)

    print("""
    Final Score = (α × vector_score) + ((1-α) × keyword_score)

    Where α (alpha) controls the balance:

    α = 1.0 → Pure vector search
    α = 0.0 → Pure keyword search
    α = 0.5 → Equal weight to both
    α = 0.7 → Emphasize semantics

    Scores are normalized first (0 to 1) so they're comparable.

    Some systems use reciprocal rank fusion (RRF):
    Score = 1 / (k + rank_vector) + 1 / (k + rank_keyword)
    This is rank-based instead of score-based.
    """)

combination_formula()
```

---

## Concrete Example

```python

def hybrid_example():
    """
    Real example showing hybrid search in action
    """
    print("Hybrid Search in Action")
    print("=" * 60)

    query = "iPhone 15 Pro Max camera review"

    documents = [
        {
            "text": "The iPhone 15 Pro Max features a groundbreaking camera system...",
            "vector_score": 0.92,
            "keyword_score": 0.45
        },
        {
            "text": "Latest smartphone camera comparisons including iPhone 15 models...",
            "vector_score": 0.88,
            "keyword_score": 0.30
        },
        {
            "text": "iPhone 15 Pro Max available now with free shipping",
            "vector_score": 0.45,
            "keyword_score": 0.95
        },
        {
            "text": "Android phones vs iPhones: which takes better photos?",
            "vector_score": 0.78,
            "keyword_score": 0.10
        }
    ]

    print(f"Query: '{query}'\n")

    # Pure vector search
    print("PURE VECTOR SEARCH (semantic):")
    vector_results = sorted(documents, key=lambda x: -x["vector_score"])[:2]
    for doc in vector_results:
        print(f"  • {doc['text'][:50]}... (score: {doc['vector_score']:.2f})")

    # Pure keyword search
    print("\nPURE KEYWORD SEARCH (exact):")
    keyword_results = sorted(documents, key=lambda x: -x["keyword_score"])[:2]
    for doc in keyword_results:
        print(f"  • {doc['text'][:50]}... (score: {doc['keyword_score']:.2f})")

    # Hybrid (alpha=0.5)
    print("\nHYBRID SEARCH (α=0.5):")
    for doc in documents:
        doc["hybrid_score"] = 0.5 * doc["vector_score"] + 0.5 * doc["keyword_score"]

    hybrid_results = sorted(documents, key=lambda x: -x["hybrid_score"])[:3]
    for doc in hybrid_results:
        print(f"  • {doc['text'][:50]}... (score: {doc['hybrid_score']:.2f})")

hybrid_example()
```

---

## When to Use Hybrid Search

### Perfect Hybrid Scenarios

```python

def hybrid_scenarios():
    """
    When hybrid search shines
    """
    print("Hybrid Search: Ideal Use Cases")
    print("=" * 60)

    scenarios = [
        ("Product search",
         "User searches 'blue running shoes size 10'",
         "Vector finds 'running shoes', keyword finds exact size"),

        ("Medical literature",
         "Doctor searches 'COVID-19 treatment guidelines 2024'",
         "Vector finds treatment concepts, keyword finds exact year"),

        ("Legal documents",
         "Lawyer searches 'Section 230 immunity social media'",
         "Vector finds immunity concepts, keyword finds exact section"),

        ("Customer support",
         "User searches 'iPhone 14 battery replacement cost'",
         "Vector finds battery issues, keyword finds exact model"),

        ("Academic search",
         "Researcher searches 'transformers attention mechanism 2017 paper'",
         "Vector finds attention papers, keyword finds exact year")
    ]

    for domain, query, reason in scenarios:
        print(f"\n  • {domain}:")
        print(f"    Query: '{query}'")
        print(f"    Why: {reason}")

hybrid_scenarios()
```

### When Pure Search Might Be Better

```python

def pure_scenarios():
    """
    When to stick with one search type
    """
    print("When Pure Search Might Be Better")
    print("=" * 60)

    print("""
    PURE VECTOR IS BETTER WHEN:
    • Exploring concepts ("find me articles about consciousness")
    • No exact terms to match
    • Working with synonyms-heavy content

    PURE KEYWORD IS BETTER WHEN:
    • Searching product codes ("SKU-12345-XYZ")
    • Exact quotes needed
    • Boolean logic required
    • Working with structured data

    HYBRID IS BEST FOR MOST GENERAL SEARCH!
    """)

pure_scenarios()
```

---

## Implementing Hybrid Search

### With Weaviate (Example)

```python

def weaviate_hybrid():
    """
    Hybrid search in Weaviate
    """
    print("Hybrid Search in Weaviate")
    print("=" * 60)

    print("""
    import weaviate

    client = weaviate.Client("http://localhost:8080")

    # Hybrid search with alpha=0.7 (favor vector)
    result = client.query.get(
        "Document", ["title", "content"]
    ).with_hybrid(
        query="iPhone 15 Pro Max camera review",
        alpha=0.7  # 0.7 vector, 0.3 keyword
    ).with_limit(10).do()

    # Or use reciprocal rank fusion
    result = client.query.get(
        "Document", ["title", "content"]
    ).with_hybrid(
        query="iPhone 15 Pro Max camera review",
        fusion_type="ranked"  # Uses RRF
    ).with_limit(10).do()
    """)

weaviate_hybrid()
```

### With Qdrant (Example)

```python

def qdrant_hybrid():
    """
    Hybrid search in Qdrant
    """
    print("Hybrid Search in Qdrant")
    print("=" * 60)

    print("""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, Match

    client = QdrantClient("localhost", port=6333)

    # Qdrant supports hybrid via:
    # 1. Full-text index + vector search
    # 2. Fusion of results

    # Create payload index for keyword search
    client.create_payload_index(
        collection_name="my_docs",
        field_name="text",
        field_type="text"
    )

    # Then search with both
    vector_result = client.search(...)
    keyword_result = client.scroll(
        collection_name="my_docs",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="text",
                    match=Match(text="iPhone 15")
                )
            ]
        )
    )

    # Then fuse results in your application
    """)

qdrant_hybrid()
```

### Simple Python Implementation

```python

def simple_hybrid():
    """
    Simple hybrid search implementation
    """
    print("Simple Hybrid Search in Python")
    print("=" * 60)

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Sample documents
    docs = [
        "iPhone 15 Pro Max camera review",
        "Best smartphone cameras 2024",
        "iPhone 15 Pro Max available now",
        "Android vs iPhone camera comparison",
        "Smartphone photography tips and tricks"
    ]

    query = "iPhone 15 Pro Max camera"

    # Vector search (using TF-IDF as simple embedding)
    vectorizer = TfidfVectorizer().fit(docs + [query])
    doc_vectors = vectorizer.transform(docs)
    query_vector = vectorizer.transform([query])

    vector_scores = cosine_similarity(query_vector, doc_vectors)[0]

    # Keyword search (simple term matching)
    keyword_scores = []
    query_terms = set(query.lower().split())

    for doc in docs:
        doc_terms = set(doc.lower().split())
        overlap = len(query_terms & doc_terms)
        keyword_scores.append(overlap / len(query_terms))

    # Hybrid (alpha=0.6)
    alpha = 0.6
    hybrid_scores = alpha * vector_scores + (1-alpha) * np.array(keyword_scores)

    print(f"Query: '{query}'\n")
    print("Doc | Vector | Keyword | Hybrid")
    print("-" * 35)

    for i, doc in enumerate(docs):
        print(f"{i+1}. | {vector_scores[i]:.3f} | {keyword_scores[i]:.3f} | {hybrid_scores[i]:.3f}")
        print(f"   {doc}")

simple_hybrid()
```

---

## Hybrid Search Parameters

| Parameter     | What It Controls       | Typical Range    | Effect                             |
| ------------- | ---------------------- | ---------------- | ---------------------------------- |
| α (alpha)     | Balance vector/keyword | 0.3-0.7          | Higher = more semantic             |
| Fusion method | How to combine         | Weighted/RRF     | RRF is rank-based                  |
| k (for RRF)   | Rank fusion constant   | 60               | Higher = less impact of high ranks |
| Normalization | Score scaling          | min-max, z-score | Makes scores comparable            |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Hybrid search isn't just a nice-to-have—it's essential     ║
║  for production RAG applications:                           ║
║                                                              ║
║  • Vector search alone misses exact matches                 ║
║    Users searching for "iPhone 15" need that exact model    ║
║                                                              ║
║  • Keyword search alone misses semantic connections         ║
║    "Phone recommendations" won't find "best smartphones"    ║
║                                                              ║
║  • Real-world queries mix both types                        ║
║    "COVID-19 treatment guidelines 2024" needs both          ║
║                                                              ║
║  • Different users search differently                       ║
║    Some use precise terms, some describe concepts           ║
║                                                              ║
║  • Product names, codes, and IDs require exact matches      ║
║    But also need semantic understanding of context          ║
║                                                              ║
║  Hybrid search ensures you don't miss either type           ║
║  of relevant document. It's the difference between          ║
║  good search and great search.                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Vector search finds meaning but misses exact matches—great for synonyms and concepts, bad for product codes and proper names

• Keyword search finds exact matches but misses meaning—perfect for precise terms, useless for semantic understanding

• Hybrid search combines both—using a weighted score or rank fusion to give you the best of both worlds, catching documents that either method alone would miss

---

## Mental Hook

> "Hybrid search is like having both a bloodhound (vector) that follows the scent of meaning and a tracker (keyword) that follows exact footprints—together, they never lose the trail."
