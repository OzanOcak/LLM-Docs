# Multi-head attention

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Transformers (The LLM Revolution)**

---

### **1. Why this concept matters**

Single self-attention computes one set of relationships: "cat" attends to "sat" in one way. But words have multiple relationships simultaneously. "Cat" has syntactic role (subject), semantic meaning (animal), coreference (it), and positional context (first noun). One attention head cannot capture all of these. Multi-head attention runs multiple self-attention operations in parallel, each with different learned projections. One head might focus on syntax, another on semantics, another on distant dependencies. Then the outputs are concatenated. This is why transformers work: they model the many facets of language simultaneously, not as a single averaged view.

---

### **2. Core idea**

**Multi-head attention runs multiple self-attention mechanisms in parallel, each with its own learned projections for queries, keys, and values, then concatenates the results, allowing the model to attend to different types of relationships at different positions.**

---

### **3. Concrete analogy**

Imagine a team of detectives investigating a crime scene. Each detective has a specialty:

- Detective A looks at fingerprints (local relationships)
- Detective B looks at alibis (temporal relationships)
- Detective C looks at financial records (semantic relationships)
- Detective D looks at phone calls (long-range dependencies)

Each detective examines the same evidence (the input sequence) but through a different lens (different projection). They each produce a report (attention output). The lead investigator then combines all reports (concatenation) and draws a final conclusion.

If you had only one detective, you would miss critical evidence. The same is true for attention. A single head might learn to attend to nearby words for grammar, but ignore distant coreference. Another head does the opposite. Together, they capture the full complexity of language. Multi-head attention is the team of specialists.

---

### **4. ASCII diagram**

```
Multi-head attention (h = 4 heads):

    Input X (n × d_model)
           │
    ┌──────┼──────┬──────┬──────┐
    │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼
  Head1  Head2  Head3  Head4
    │      │      │      │      │
    │ Q₁K₁V₁ │ Q₂K₂V₂ │ Q₃K₃V₃ │ Q₄K₄V₄
    │   ↓    │   ↓    │   ↓    │   ↓
    │ Attn₁  │ Attn₂  │ Attn₃  │ Attn₄
    │   ↓    │   ↓    │   ↓    │   ↓
    │  Z₁    │  Z₂    │  Z₃    │  Z₄
    │      │      │      │      │
    └──────┼──────┴──────┴──────┘
           │
           ▼
    Concatenate: Z = [Z₁, Z₂, Z₃, Z₄]
           │
           ▼
    Linear projection: Z × W^O
           │
           ▼
        Output (n × d_model)


Each head computes:

    Q_i = X × W_i^Q   (d_model × d_k)
    K_i = X × W_i^K   (d_model × d_k)
    V_i = X × W_i^V   (d_model × d_v)

    Z_i = softmax(Q_i × K_i^T / √d_k) × V_i

With d_k = d_v = d_model / h (typically 64, so d_model=512 → 8 heads)
```

---

### **5. Mathematical formulation**

**For each head i = 1 to h:**

$$
\mathbf{Q}_i = \mathbf{X} \mathbf{W}_i^Q, \quad
\mathbf{K}_i = \mathbf{X} \mathbf{W}_i^K, \quad
\mathbf{V}_i = \mathbf{X} \mathbf{W}_i^V
$$

**Head output:**

$$
\text{head}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}}\right) \mathbf{V}_i
$$

(Unicode: head_i = softmax(Q_i·K_iᵀ / √d_k) × V_i)

**Concatenation:**

$$
\mathbf{Z} = \text{Concat}(\text{head}_1, ..., \text{head}_h)
$$

**Final linear projection:**

$$
\text{MultiHead}(\mathbf{X}) = \mathbf{Z} \mathbf{W}^O
$$

(Unicode: MultiHead(X) = Concat(head_1,...,head_h) × W^O)

Where W^O is d_model × d_model (projects back to original dimension)

**Parameter counts per head:**

- W_i^Q: d_model × d_k
- W_i^K: d_model × d_k
- W_i^V: d_model × d_v
- Total for h heads: 3 × h × d_model × (d_model/h) = 3 × d_model²

Plus W^O: d_model². Total ≈ 4 × d_model² parameters.

**Typical configurations:**

- BERT-base: d_model=768, h=12, d_k=64
- BERT-large: d_model=1024, h=16, d_k=64
- GPT-3: d_model=12288, h=96, d_k=128

---

### **6. Worked example (step-by-step)**

**Step 1: Setup**

d_model = 4, h = 2 heads, so d_k = d_v = 2

Sequence length n=3, X = [[1,0,0,0], [0,1,0,0], [0,0,1,0]] (simplified)

**Step 2: Head 1 projections (learned weights—simplified)**

W₁^Q (4×2): extracts first 2 dims → Q₁ = [[1,0], [0,1], [0,0]]
W₁^K (4×2): same → K₁ = [[1,0], [0,1], [0,0]]
W₁^V (4×2): same → V₁ = [[1,0], [0,1], [0,0]]

Attention scores: Q₁·K₁ᵀ = [[1,0,0], [0,1,0], [0,0,0]] / √2 ≈ same pattern
Softmax weights: Head 1 focuses on self (diagonal). Output Z₁ approximates V₁.

**Step 3: Head 2 projections**

W₂^Q, W₂^K, W₂^V extract last 2 dims → Q₂ = [[0,0], [0,0], [1,0]]? Actually X row3 has 1 in position 3, so last 2 dims of row3 = [1,0] only if d_model=4 and we take indices 2,3. Let's simplify: Head 2 focuses on position 3 (last token) attending to all.

**Step 4: Concatenate**

Z = [Z₁, Z₂] — each head output is n×2, concatenated to n×4

**Step 5: Final projection W^O (4×4)**

Mixes information across heads. Final output mixes Head1's self-focus with Head2's cross-attention.

**Step 6: Interpret**

Head 1 learned local/self relationships. Head 2 learned to broadcast information from the last token. Together, the model captures both. Without multiple heads, the model would have to choose one behavior or average them poorly.

---

### **7. How this appears inside neural networks and LLMs**

- **Transformer blocks:** Every attention layer in every transformer uses multi-head attention (typically 8-96 heads).

- **Head specialization:** Research shows heads specialize: positional heads (attend to adjacent words), syntactic heads (attend to subjects/objects), coreference heads (connect pronouns to nouns), rare-word heads (copy unusual tokens).

- **Redundancy:** Some heads learn redundant functions. Pruning heads often preserves performance, enabling model compression.

- **BERT-base:** 12 layers × 12 heads = 144 attention heads total. Each captures different linguistic patterns.

- **GPT-3:** 96 layers × 96 heads = 9,216 attention heads. The massive parallel attention is why LLMs are so capable.

- **Multi-query attention (MQA):** Shared keys/values across heads to save memory. Used in some LLMs (e.g., PaLM, Falcon) for faster inference.

- **Grouped-query attention (GQA):** Compromise between MQA and standard multi-head. Llama 2/3 use GQA.

---

### **8. Brain-like connection (parallel cortical columns)**

The brain's cortex processes information through parallel columns. Each column (about 0.5mm wide) performs similar operations but with different receptive fields and specializations. Visual cortex has columns tuned to orientation, color, motion, and depth—all processing the same retinal image in parallel, then integrating. This is multi-head attention in biology. The hippocampus has "place cells" and "grid cells" that encode location in different coordinate systems simultaneously—multiple "heads" tracking different aspects of spatial memory. Evolution discovered parallel specialized processing long before transformers formalized it.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "More heads are always better. I should use as many heads as my GPU memory allows."

_Why it is wrong:_ Heads provide diminishing returns. Beyond a point (typically d_model/64 or ~16-32 heads for large models), adding heads adds parameters without improving performance. Heads also become redundant—many learn the same pattern. Pruning studies show 50-80% of heads can be removed with minimal accuracy loss. The optimal number depends on d_model and task complexity. For d_model=768, 12 heads works. For d_model=4096, 32 heads works. Blindly doubling heads wastes compute. Let validation accuracy guide your choice, not the desire for more heads.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  One head sees one relationship. Twelve heads see twelve  |
|  facets of language: grammar, semantics, coreference,     |
|  position, negation, and more. Multi-head attention is    |
|  the reason transformers can handle the richness of human |
|  language. It is the parallel processing that captures    |
|  what single-attention models miss. Without multiple      |
|  heads, GPT would be GPT-1. With them, we have GPT-4.     |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

BERT-base has d_model=768 and h=12 heads. d_k = d_v = d_model/h = 64.

**Question:** How many total parameters are in the multi-head attention layer (including all head projections and the output projection W^O)? Show the calculation.

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

For each head:

- W^Q: 768 × 64 = 49,152
- W^K: 768 × 64 = 49,152
- W^V: 768 × 64 = 49,152
- Total per head: 147,456

12 heads: 12 × 147,456 = 1,769,472

Output projection W^O: 768 × 768 = 589,824

**Total multi-head attention parameters: 1,769,472 + 589,824 = 2,359,296**

About 2.36 million parameters, which is small compared to feed-forward layers (each ~4×d_model² ≈ 4×589,824 = 2.36M as well). In BERT-base (110M total), attention and FFN contribute roughly equally.
