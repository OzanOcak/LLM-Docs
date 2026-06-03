# Comparing distance metrics (tradeoffs and use cases)

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Distance Metrics: Measuring Similarity**

---

### **1. Why this concept matters**

Cosine, Euclidean, dot product—three metrics, one question: "How similar are these vectors?" But they give different answers for the same pair. Choosing the wrong metric can sink your retrieval system, bias your recommendations, or break your clustering. Cosine ignores magnitude. Euclidean includes it. Dot product amplifies it. There is no universally best metric. The right choice depends on your data (sparse vs dense), your task (retrieval vs clustering), and whether magnitude is signal or noise. This section compares all three so you can decide with confidence.

---

### **2. Core idea**

**Cosine similarity measures direction (ignores magnitude), Euclidean distance measures absolute position (includes magnitude), and dot product measures raw alignment (magnitude × direction); the optimal metric depends on whether vector length carries meaningful information for your task.**

---

### **3. Concrete analogy**

Imagine comparing two people's restaurant preferences on a 2D map where x = love for Italian, y = love for sushi.

- **Person A:** [8, 2] (loves Italian, likes sushi a bit)
- **Person B:** [4, 1] (same proportions, half the intensity)
- **Person C:** [2, 8] (loves sushi, Italian a bit)

Different metrics tell different stories:

- **Cosine:** A and B are identical (0.96). C is different (0.32). Good for taste _profile_.
- **Euclidean:** A to B distance = √[(4)²+(1)²]=√17≈4.1. A to C distance = √[(6)²+(6)²]=√72≈8.5. B is closer, but magnitudes matter.
- **Dot product:** A·B=32+2=34, A·C=16+16=32. A·B > A·C, but A·A=64 (self). Raw intensity dominates.

Which is correct? If you recommend restaurants, use cosine (proportions matter). If you predict total spending, use dot product (intensity matters). If you cluster geographical locations, use Euclidean. No single answer.

---

### **4. ASCII diagram**

```
Comparison of three metrics on three vectors:

    Vector space:
        ↑ (sushi)
        │
        │   C (2,8)
        │
        │
        │ A (8,2)
        │ B (4,1)
        └────────────────→ (Italian)


    Pairwise similarities/distances:

    Pair      | Cosine | Euclidean | Dot    | Interpretation
    ----------|--------|-----------|--------|--------------------------------
    A vs B    | 0.96   | 4.12      | 34     | Same direction, different magnitude
    A vs C    | 0.32   | 8.49      | 32     | Different direction, similar magnitude
    B vs C    | 0.39   | 7.81      | 16     | Different direction, B smaller
    A vs A    | 1.00   | 0.00      | 68     | Self-similarity


When each metric shines:

    Metric    | Best for                                    | Avoid when
    ----------|---------------------------------------------|------------------
    Cosine    | Text embeddings, user taste profiles       | Magnitude matters (e.g., revenue prediction)
    Euclidean | Physical coordinates, regression residuals | High-dimensional sparse data
    Dot       | Recommendation (active users), raw logits  | Vectors have widely varying lengths

Relationship summary:

    dot = ‖a‖·‖b‖·cosθ
    cos = dot / (‖a‖·‖b‖)
    Euclidean² = ‖a‖² + ‖b‖² - 2·dot
```

---

### **5. Mathematical formulation**

**Three metrics defined:**

Dot product:

$$
\text{dot}(\mathbf{a},\mathbf{b}) = \sum_i a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta
$$

Cosine similarity:

$$
\cos(\mathbf{a},\mathbf{b}) = \frac{\text{dot}(\mathbf{a},\mathbf{b})}{\|\mathbf{a}\|\|\mathbf{b}\|}
$$

Euclidean distance:

$$
d_{\text{Euc}}(\mathbf{a},\mathbf{b}) = \sqrt{\sum_i (a_i - b_i)^2} = \sqrt{\|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\,\text{dot}(\mathbf{a},\mathbf{b})}
$$

**Tradeoff summary:**

| Property                    | Cosine               | Euclidean        | Dot                  |
| --------------------------- | -------------------- | ---------------- | -------------------- |
| Range                       | [-1, 1]              | [0, ∞)           | (-∞, ∞)              |
| Invariant to vector length  | Yes                  | No               | No                   |
| Invariant to rotation       | Yes                  | Yes              | Yes                  |
| Computational cost          | O(d) + normalization | O(d)             | O(d)                 |
| Sensitive to magnitude      | No                   | Yes              | Yes                  |
| Sensitive to frequency bias | No                   | Yes (doc length) | Yes (freq dominates) |

**Normalization equivalence:**

If all vectors are L2-normalized (‖v‖=1), then:

$$
\text{dot} = \cos, \quad d_{\text{Euc}} = \sqrt{2 - 2\cdot\text{dot}}
$$

Thus, all three are equivalent after normalization. The difference is whether you normalize.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Three vectors in 3D (word counts for three documents)**

Doc1: [100, 10, 1] (long, mostly about topic A)
Doc2: [10, 1, 0] (short, same proportions as Doc1)
Doc3: [1, 100, 10] (long, mostly about topic B)

#### **Step 2: Compute all metrics**

Normalize first for cosine (divide by L2 norm):

‖Doc1‖ = √(10000+100+1) ≈ √10101 ≈ 100.5
‖Doc2‖ = √(100+1+0) = √101 ≈ 10.05
‖Doc3‖ = √(1+10000+100) ≈ √10101 ≈ 100.5

Doc1_norm = [0.995, 0.0995, 0.00995]
Doc2_norm = [0.995, 0.0995, 0.0]
Doc3_norm = [0.00995, 0.995, 0.0995]

#### **Step 3: Cosine similarities**

cos(D1,D2) = 0.995×0.995 + 0.0995×0.0995 + 0.00995×0 = 0.990 + 0.0099 + 0 = 0.9999 (almost identical proportions)
cos(D1,D3) = 0.995×0.00995 + 0.0995×0.995 + 0.00995×0.0995 ≈ 0.0099 + 0.0990 + 0.0010 = 0.1099
cos(D2,D3) = 0.995×0.00995 + 0.0995×0.995 + 0×0.0995 ≈ 0.0099 + 0.0990 + 0 = 0.1089

#### **Step 4: Euclidean distances (raw, unnormalized)**

d(D1,D2) = √[(100-10)² + (10-1)² + (1-0)²] = √(8100 + 81 + 1) = √8182 ≈ 90.5
d(D1,D3) = √[(100-1)² + (10-100)² + (1-10)²] = √(9801 + 8100 + 81) = √17982 ≈ 134.1
d(D2,D3) = √[(10-1)² + (1-100)² + (0-10)²] = √(81 + 9801 + 100) = √9982 ≈ 99.9

#### **Step 5: Dot products (raw)**

dot(D1,D2) = 100×10 + 10×1 + 1×0 = 1000 + 10 + 0 = 1010
dot(D1,D3) = 100×1 + 10×100 + 1×10 = 100 + 1000 + 10 = 1110
dot(D2,D3) = 10×1 + 1×100 + 0×10 = 10 + 100 + 0 = 110

#### **Step 6: Interpret rankings**

- **Cosine:** D1 and D2 are nearly identical (0.9999). D1 and D3 are different (~0.11). Good for semantic similarity.
- **Euclidean:** D1 and D2 are closer (90.5) than D1 and D3 (134.1). But both large. Hard to interpret.
- **Dot:** D1·D3 (1110) > D1·D2 (1010). Dot says D1 is more similar to D3! This is because D3 is long (large magnitude). Dot product is biased by document length. Not good for retrieval without normalization.

---

### **7. How this appears inside neural networks and LLMs**

- **Text retrieval (RAG, semantic search):** Almost always cosine similarity on normalized embeddings. Euclidean sometimes used after normalization (equivalent). Dot product rarely used raw.

- **Recommendation (collaborative filtering):** Dot product raw or with normalization depending on whether user activity level is predictive. Netflix reportedly uses dot product.

- **Clustering:** k-means uses Euclidean (standard). Spherical k-means uses cosine (normalizes first).

- **LLM attention:** Scaled dot product (Q·Kᵀ / √d_k). Not cosine, not Euclidean. Magnitude matters because queries/keys are not normalized.

- **Anomaly detection:** Mahalanobis distance (Euclidean after scaling by covariance). Sometimes Euclidean on standardized features.

- **Vector databases (FAISS, Pinecone):** Support all three. Cosine is default for text. Inner product (dot) often used after L2 normalization (making it equivalent to cosine).

- **Image retrieval (CNN features):** Usually cosine or Euclidean after normalization. ViT embeddings normalized before retrieval.

- **Cross-modal retrieval (CLIP):** Cosine similarity between image and text embeddings (both L2-normalized). Critical for zero-shot classification.

---

### **8. Brain-like connection (multiple distance metrics in the brain)**

The brain uses different distance metrics for different tasks. For spatial navigation, hippocampus computes Euclidean distance (path integration). For recognizing faces, inferotemporal cortex uses a metric closer to cosine (invariant to lighting, scale). For olfaction, the brain uses a metric based on chemical properties, not Euclidean. For value-based decisions (choose A or B), the brain computes a dot product between neural representations of options and current state (orbitofrontal cortex). The brain is a collection of specialized distance metrics, each tuned to its task. This is the neural basis of the "no free lunch" theorem: no single metric is optimal for all problems.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "For text embeddings, always use cosine similarity. It's the standard. Euclidean and dot are wrong."

_Why it is wrong:_ Cosine is standard but not always optimal. For sentences, cosine works well because length is noise. For paragraphs, length may carry information (longer documents cover more topics). For TF-IDF, dot product is equivalent to cosine if vectors are normalized. Many production systems use inner product (dot) after L2 normalization because it is faster. Some use Euclidean. The key is normalization, not the metric itself. If you normalize all vectors to unit length, dot = cosine, and Euclidean is monotonic. The real choice is: do you normalize? If yes, all three are equivalent. If no, the choice matters. For raw count vectors, dot is dominated by frequency; cosine corrects for length. But if you have normalized vectors already (e.g., learned embeddings), the distinction vanishes.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Cosine, Euclidean, dot—three metrics, one question.      |
|  Choose wrong and your search engine favors long          |
|  documents, your clustering collapses, or your            |
|  recommendations miss the mark. Normalize vectors to      |
|  unit length and all three become equivalent. But when    |
|  magnitude is signal, skip normalization. The choice is   |
|  not about math—it is about whether length matters.       |
|  Know your data. Know your task. Choose accordingly.      |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a dataset of 10,000 product embeddings (768 dimensions each, L2-normalized) for an e-commerce search engine. A user queries "wireless headphones". You compute the query embedding (also normalized). Which distance metric would you use to retrieve top-100 products? Why? Would your answer change if the product embeddings were not normalized?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Use cosine similarity (or equivalently, dot product after normalization, or Euclidean after normalization—all equivalent for unit vectors).

Since all vectors are already L2-normalized (‖v‖=1), dot = cosine, and Euclidean distance = √(2-2·dot). All three metrics give the same ranking (monotonic). So you can use dot product for speed (no division, no sqrt) or cosine for interpretability. In practice, vector databases store normalized vectors and use dot product internally.

**If product embeddings were not normalized:**

Then the choice becomes critical:

- **Cosine:** Corrects for magnitude differences. If some products have larger magnitude due to more features or higher frequency, cosine prevents them from dominating unfairly.
- **Dot product:** Raw similarity. Would favor high-magnitude products, which may be desirable if magnitude correlates with relevance (e.g., popular products). Often undesirable.
- **Euclidean:** Also affected by magnitude. Less common for retrieval.

Recommendation: Normalize the embeddings to unit length, then use dot product (equivalent to cosine). This is standard practice in retrieval systems (CLIP, SBERT, etc.). If you cannot normalize (e.g., magnitude is meaningful), test both cosine and dot on a held-out relevance dataset to decide. For most semantic search tasks, cosine is safer.
