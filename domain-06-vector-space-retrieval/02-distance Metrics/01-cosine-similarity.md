# Cosine similarity (orientation matters)

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Distance Metrics: Measuring Similarity**

---

### **1. Why this concept matters**

Embeddings map meaning to geometry. But how do you measure if two meanings are similar? Euclidean distance is one way, but it fails when vectors have different lengths. A word embedding can be short or long—length often corresponds to frequency, not meaning. Cosine similarity ignores length. It measures only the angle between vectors: the direction, not the magnitude. Two vectors pointing the same way are similar, regardless of whether one is longer. This makes cosine similarity the default metric for embeddings. It is used in semantic search, recommendation systems, and clustering. Understanding cosine similarity is understanding how to compare meanings fairly.

---

### **2. Core idea**

**Cosine similarity measures the cosine of the angle between two vectors, ranging from -1 (opposite directions) to +1 (same direction), ignoring vector magnitude and focusing only on orientation, making it ideal for comparing normalized embeddings.**

---

### **3. Concrete analogy**

Imagine two arrows pointing in the same direction. One is long, one is short. They point the same way. Under cosine similarity, they are identical (score = 1.0). Under Euclidean distance, they are far apart because the arrowheads are in different positions.

Now imagine two arrows of the same length. One points north, one points slightly northeast. Cosine similarity captures this small difference. Two arrows pointing opposite directions (north vs south) give -1.0.

Why does this matter? In embeddings, vector length often correlates with word frequency or confidence. "Cat" and "cats" have similar meanings but may have different lengths. Cosine similarity ignores length, focusing on meaning (direction). This is why search engines use cosine similarity: "apple" (fruit) and "apples" should match even if their vector lengths differ.

---

### **4. ASCII diagram**

```
Vectors in 2D space (all same direction, different lengths):

    y ↑
      │
      │   ● (3,1)   length ≈ 3.16
      │  ╱
      │ ╱
      │╱
      ●────────────→ x
    (1.5,0.5) length ≈ 1.58

    Cosine similarity = 1.0 (same direction)
    Euclidean distance = 2.0 (far apart despite same meaning!)

    These vectors represent the same concept but different lengths.
    Cosine ignores the length, sees the meaning.


Cosine similarity visualized:

    Opposite (sim = -1)      Orthogonal (sim = 0)      Same (sim = 1)

    ↑                         ↑                         ↑
    │                         │                         │
    │                         │          ╱──●
    │                         │         ╱
    │                         │        ●─╱
    ●──────────→              └────●────→              ●──────→
    (pointing down)           (perpendicular)          (same angle)


Euclidean vs Cosine for embeddings:

    Euclidean sensitive to magnitude:
        "cat" (freq 100) → length 0.8
        "cats" (freq 10) → length 0.3
        Euclidean distance large (unfair)

    Cosine ignores magnitude:
        cos(cat, cats) ≈ 0.95 (similar meaning)
        Good for semantic comparison.
```

---

### **5. Mathematical formulation**

**Cosine similarity definition:**

$$
\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^d a_i b_i}{\sqrt{\sum_{i=1}^d a_i^2} \sqrt{\sum_{i=1}^d b_i^2}}
$$

Range: [-1, +1]

**Relationship to Euclidean distance (for normalized vectors):**

If $\|\mathbf{a}\| = \|\mathbf{b}\| = 1$, then:

$$
\|\mathbf{a} - \mathbf{b}\|^2 = 2(1 - \text{cosine}(\mathbf{a},\mathbf{b}))
$$

Thus, cosine similarity is equivalent to Euclidean distance for unit vectors.

**Conversion to angular distance:**

$$
\text{angular distance} = \frac{\arccos(\text{cosine}(\mathbf{a},\mathbf{b}))}{\pi}
$$

Range: [0, 1]

**When to use cosine vs Euclidean:**

| Scenario                            | Better metric                   | Reason                                     |
| ----------------------------------- | ------------------------------- | ------------------------------------------ |
| Embeddings (freq varies)            | Cosine                          | Ignores magnitude, focus on direction      |
| Normalized embeddings               | Cosine or Euclidean             | Equivalent after normalization             |
| Raw features (counts)               | Cosine if length not meaningful | e.g., TF-IDF vectors                       |
| Clustering spherical data           | Cosine                          | k-means on unit sphere = spherical k-means |
| Regression (absolute values matter) | Euclidean                       | Magnitude carries meaning                  |

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define two vectors**

a = [4, 3] (length = √(16+9)=5)
b = [8, 6] (length = √(64+36)=10)

Same direction (b = 2×a). Same angle.

#### **Step 2: Compute cosine similarity**

Dot product = 4×8 + 3×6 = 32 + 18 = 50

cosine = 50 / (5 × 10) = 50 / 50 = 1.0

#### **Step 3: Compute Euclidean distance**

Euclidean = √((8-4)² + (6-3)²) = √(16 + 9) = √25 = 5

Despite being directionally identical, Euclidean distance is large. Cosine says they are identical.

#### **Step 4: Example with different direction**

a = [4, 3]
c = [3, 4] (different direction, similar length)

Dot = 4×3 + 3×4 = 12 + 12 = 24
‖a‖ = 5, ‖c‖ = 5
cosine = 24 / 25 = 0.96

Angle ≈ 16 degrees, very similar.

#### **Step 5: Orthogonal vectors**

d = [4, 3]
e = [-3, 4]

Dot = 4×(-3) + 3×4 = -12 + 12 = 0
cosine = 0 (orthogonal, unrelated)

#### **Step 6: Opposite vectors**

f = [4, 3]
g = [-4, -3] (exactly opposite)

Dot = -16 + -9 = -25
‖f‖=5, ‖g‖=5, cosine = -25/25 = -1

---

### **7. How this appears inside neural networks and LLMs**

- **Semantic search:** Embed query and documents, compute cosine similarity, return top-k. Used in RAG (Retrieval-Augmented Generation).

- **Sentence similarity (SBERT):** Cosine between sentence embeddings. Scores correlate with human judgment (Spearman correlation).

- **Clustering embeddings:** k-means with cosine distance (spherical k-means) works by normalizing vectors to unit length first.

- **Word2vec evaluation:** Analogy accuracy often computed via cosine similarity (find d maximizing cosine with c + (b - a)).

- **Cross-encoder vs bi-encoder:** Bi-encoders compute cosine (fast, pre-compute). Cross-encoders compute joint attention (slow, accurate).

- **Cosine as unnormalized inner product:** If embeddings are normalized to unit length, cosine = dot product. Many libraries (FAISS) use inner product for speed.

- **Avoiding magnitude bias:** In retrieval, document length can vary. Cosine prevents long documents from dominating results unfairly.

- **Cosine in attention:** Self-attention uses scaled dot product, not cosine, because queries and keys are not normalized. But conceptually it is similarity.

---

### **8. Brain-like connection (direction coding in neural populations)**

The brain represents direction (not magnitude) in many neural codes. Motor cortex encodes direction of arm movement via population vectors. Individual neurons fire maximally for a preferred direction (e.g., 45°). The population vector sums contributions weighted by firing rates, producing a direction estimate. This is cosine similarity: the angle between the actual movement direction and each neuron's preferred direction determines its firing rate. The brain ignores overall firing rate magnitude (analogous to vector length) and focuses on direction. Similarly, place cells in hippocampus represent spatial location using direction from landmarks. Cosine similarity is not just a metric—it is a biologically plausible computation.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Cosine similarity always ranges from -1 to 1. A negative value means the vectors are unrelated."

_Why it is wrong:_ Negative cosine means vectors point in opposite directions. In many embedding spaces (e.g., Word2vec, GloVe, BERT sentence embeddings), vectors are mostly in the positive orthant (all dimensions positive). Cosine similarity rarely goes negative. Negative values are possible but uncommon. Even vectors for antonyms ("good" and "bad") often have positive cosine (0.3-0.5) because they share many contextual features (both adjectives, both evaluative). Therefore, negative cosine does not necessarily mean "unrelated"—it may be an artifact of the embedding space. Always check the distribution of cosine values for your specific model.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Embeddings have direction (meaning) and magnitude        |
|  (frequency, confidence). Cosine similarity ignores       |
|  magnitude, comparing only direction. This is why it is   |
|  the default metric for semantic search, recommendation,  |
|  and clustering. "Cat" and "cats" should match even if    |
|  their vectors differ in length. Euclidean would separate |
|  them. Cosine unites them. If you work with embeddings,   |
|  cosine is your compass.                                  |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have three embedding vectors (unit length after normalization):

A = [0.8, 0.6]
B = [0.6, 0.8]
C = [-0.8, 0.6]

**Question:** Compute cosine similarities cos(A,B), cos(A,C), cos(B,C). Which pair is most similar? Which pair is most dissimilar (closest to orthogonal)? Which pair is opposite direction?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Since all vectors are unit length (√(0.8²+0.6²)=1, etc.), cosine similarity = dot product.

cos(A,B) = 0.8×0.6 + 0.6×0.8 = 0.48 + 0.48 = 0.96

cos(A,C) = 0.8×(-0.8) + 0.6×0.6 = -0.64 + 0.36 = -0.28

cos(B,C) = 0.6×(-0.8) + 0.8×0.6 = -0.48 + 0.48 = 0

Most similar: A and B (0.96, angle ≈ 16°). Most dissimilar (orthogonal): B and C (0, exactly 90°). Opposite direction would be -1, not present. A and C have negative cosine (-0.28, angle ≈ 106°, more than 90° but not opposite).
