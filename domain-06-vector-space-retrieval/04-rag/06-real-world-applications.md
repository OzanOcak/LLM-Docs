# Real-world applications

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: RAG (Retrieval-Augmented Generation)**

---

### **1. Why this concept matters**

RAG is not academic. It powers thousands of production systems: customer support chatbots that know your product docs, legal research assistants that cite cases, medical Q&A that references journals, and enterprise search over internal wikis. Understanding real-world applications helps you see where RAG fits—and where it does not. This section covers common use cases, their specific requirements (latency, accuracy, cost), and lessons learned from production deployments.

---

### **2. Core idea**

**RAG is deployed across domains where LLMs need access to private, current, or structured information: customer support, legal research, medical Q&A, enterprise search, code assistance, and education; each domain has unique requirements for latency, citation, and retrieval quality.**

---

### **3. Concrete analogy**

Imagine RAG as a Swiss Army knife. The same tool has different blades for different jobs:

- **Customer support (screwdriver):** Needs fast, accurate answers. No hallucinations. Must cite documentation.
- **Legal research (magnifying glass):** Needs precise citations, exact wording. Speed less critical. High recall essential.
- **Medical Q&A (scalpel):** High stakes. Must ground answers in peer-reviewed journals. Citation mandatory. Human review often required.
- **Enterprise search (flashlight):** Needs to surface relevant documents quickly. Users expect to read, not just get answers.

Same RAG pipeline, different configurations: chunk size, embedding model, top-k, reranking, citation format, latency targets. One size does not fit all.

---

### **4. ASCII diagram**

```
RAG applications and their characteristics:

    Application         | Latency | Recall | Citation | Hallucination
                        | target  | needed | required | tolerance
    --------------------|---------|--------|----------|-------------
    Customer support    | <2s     | 95%    | Yes      | Very low
    Legal research      | <10s    | 99%    | Critical | Zero
    Medical Q&A         | <5s     | 99%    | Critical | Zero
    Enterprise search   | <1s     | 90%    | No       | N/A (just retrieval)
    Code assistance     | <3s     | 85%    | Optional | Medium
    Education tutoring  | <2s     | 90%    | Optional | Low
    Personal assistant  | <1s     | 80%    | No       | Medium


Common RAG architectures by domain:

    Customer Support:
    Query → Hybrid search (product docs + FAQ) → Rerank → LLM → Answer + citation

    Legal Research:
    Query → Hybrid search (cases + statutes) → Cross-encoder rerank → LLM → Answer + pinpoint citation
    (often includes human-in-loop validation)

    Medical Q&A:
    Query → Vector search (peer-reviewed journals) → Filter by year, journal impact → LLM → Answer + citation
    (often includes disclaimer: "Consult physician")

    Enterprise Search:
    Query → Hybrid search (internal wikis + Slack + email) → Rank by recency + relevance → Return top documents
    (LLM optional; sometimes just retrieve)
```

---

### **5. Mathematical formulation**

#### **DOMAIN-specific metrics:**

Customer support:

$$
\text{Success} = \frac{\text{Resolved without escalation}}{\text{Total queries}}
$$

Legal research:

$$
\text{Recall@10} = \frac{\text{Relevant cases in top-10}}{\text{Total relevant cases}}
$$

Medical Q&A:

$$
\text{Grounding accuracy} = \frac{\text{Answers supported by retrieved context}}{\text{Total answers}}
$$

**Cost considerations:**

$$
\text{Cost per query} = \text{Cost}_{\text{embed}}(q) + \text{Cost}_{\text{search}} + \text{Cost}_{\text{LLM}}(\text{prompt})
$$

For high-volume applications (customer support), embedding and LLM costs dominate. Use smaller models (e.g., 7B instead of 70B) or caching.

**Latency breakdown (typical):**

- Embed query: 50ms
- Vector search: 20ms
- Reranking (optional): 100ms
- LLM generation (100 tokens): 500ms
- Total: ~670ms

---

### **6. Worked example (step-by-step)**

#### **Step 1: Customer support scenario**

Company: E-commerce site with 50,000 products, 500 FAQ articles.

Query: "How do I return a damaged item?"

#### **Step 2: RAG pipeline configuration**

- Chunking: FAQ articles split by question-answer pair. Product returns policy as separate chunks (500 tokens).
- Embedding: `all-MiniLM-L6-v2` (fast, cheap).
- Retrieval: Hybrid (vector + keyword), top-k=5, α=0.6.
- Reranking: Cross-encoder (optional, if budget allows).
- LLM: GPT-3.5-turbo (cheaper than GPT-4, sufficient for simple Q&A).
- Prompt: "Context: ... Question: ... Answer based only on context. If unsure, say 'I don't know'."

#### **Step 3: Retrieved chunks**

Top-3:

1. "Return policy: Items damaged during shipping can be returned within 30 days."
2. "How to start a return: Go to Orders, select Return, choose reason 'Damaged'."
3. "Return shipping: We provide free return labels for damaged items."

#### **Step 4: LLM generates answer**

"To return a damaged item, go to your Orders page, select the item, and choose 'Damaged' as the return reason. You have 30 days from delivery. We will provide a free return shipping label."

#### **Step 5: Citation**

Add footnote: "[1] Return policy, [2] Returns FAQ"

#### **Step 6: Metrics**

Latency: 800ms (embed 50ms + search 30ms + LLM 720ms). Cost: $0.002 per query. Success rate: 92% (tested on 1000 queries).

---

### **7. How this appears inside neural networks or LLMs**

- **Customer support (Zendesk, Intercom):** RAG over knowledge base. Typical stack: Pinecone/Qdrant + GPT-3.5. Metrics: deflection rate (resolved without human).

- **Legal research (Casetext, LexisNexis):** RAG over case law, statutes. Requires pinpoint citation (paragraph, page). Often uses legal-specific embeddings (Legal-BERT). Human-in-loop for high-stakes.

- **Medical Q&A (Glass Health, Doximity):** RAG over PubMed, journals. Requires high recall. Often includes disclaimers. Some use fine-tuned medical LLMs (Med-PaLM) + RAG.

- **Enterprise search (Glean, Glean, Microsoft Copilot):** RAG over internal documents (Slack, email, Drive, Confluence). Requires access control (filter by user permissions). Hybrid search essential.

- **Code assistance (GitHub Copilot, Cursor):** RAG over codebase. Retrieves relevant functions, classes, imports. Chunking at function level. Uses code embeddings (CodeBERT).

- **Education (Khanmigo, Duolingo):** RAG over curriculum. Needs age-appropriate responses. Retrieves from vetted sources only.

- **Personal assistant (Google Assistant, Alexa):** RAG over user's personal data (calendar, email, notes). Privacy critical. Often runs on-device or with confidential compute.

---

### **8. Brain-like connection (context-dependent retrieval)**

The brain retrieves memories differently depending on context. In a stressful situation, recall is fast but less precise (fight-or-flight). In relaxed study, recall is slower but more accurate. RAG applications mirror this: customer support needs speed (like stress response); legal research needs accuracy (like careful study). The brain also uses different memory systems for different domains: procedural memory (skills), semantic memory (facts), episodic memory (events). RAG systems use the same retrieval pipeline but with domain-tuned parameters—different chunk sizes, embeddings, top-k, reranking. This is cognitive specialization in software.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "RAG works the same for all applications. Just point it at your docs and it works."

_Why it is wrong:_ Customer support RAG fails for legal research (needs higher recall, citations). Legal research RAG fails for customer support (too slow, over-engineered). Medical RAG fails for code assistance (wrong embeddings). Each domain requires tuning: chunk size (code: small functions; legal: long clauses), embedding model (general vs domain-specific), retrieval (hybrid vs pure vector), reranking (optional vs mandatory), LLM size (7B vs 70B), latency targets (1s vs 10s). Do not assume a single RAG configuration works everywhere. Build for your domain.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  RAG is not a toy. It is deployed in customer support,    |
|  legal research, medicine, code assistance, and           |
|  education. Each domain has different requirements:       |
|  speed, accuracy, citations, cost. A customer support     |
|  RAG fails in legal. A medical RAG fails in code.         |
|  Understand your domain. Tune your RAG. Build for         |
|  production, not demos. RAG works—if you engineer it.     |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are building a RAG system for two different clients:

1. A fast-food chain with 500 locations (customer support for hours, menu, promotions)
2. A medical research hospital (doctors asking about rare disease treatments from journals)

**Question:** How would the RAG configuration differ between these two? List at least four differences (chunking, embedding, retrieval, LLM, latency, citation, etc.).

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Four key differences:

| Aspect                      | Fast-food (customer support)                      | Medical hospital (research)                                       |
| --------------------------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| **Chunking**                | Small chunks (100-200 tokens) for FAQ Q&A pairs   | Larger chunks (500-1000 tokens) for journal abstracts + full text |
| **Embedding model**         | General-purpose (`all-MiniLM-L6-v2`, fast, cheap) | Domain-specific (BioBERT, PubMedBERT, or fine-tuned)              |
| **Retrieval**               | Hybrid search (vector + keyword) for menu items   | Pure vector (or hybrid with high recall). Exact citations needed  |
| **LLM**                     | GPT-3.5-turbo or Llama 7B (cheap, fast)           | GPT-4 or Med-PaLM (higher accuracy, costly)                       |
| **Latency target**          | <1 second (customer impatient)                    | <10 seconds (doctor can wait)                                     |
| **Citation**                | Optional (source article title)                   | Mandatory (PubMed ID, journal, year, page)                        |
| **Hallucination tolerance** | Low (frustrates customers)                        | Zero (life-critical)                                              |
| **Reranking**               | Optional (cost-benefit)                           | Mandatory (cross-encoder for high recall)                         |
| **Human review**            | No                                                | Yes (doctor validates before treatment)                           |
| **Cost per query**          | <$0.001 (high volume, cheap model)                | <$0.10 (low volume, expensive model)                              |

**Additional considerations:**

- Fast-food: Use caching for common queries (store hours, prices). High volume.
- Medical: Access control (HIPAA). Audit logging. Privacy filters (remove PHI). Fallback to "consult physician" disclaimer.

Same RAG pipeline, completely different engineering. Domain dictates design.
