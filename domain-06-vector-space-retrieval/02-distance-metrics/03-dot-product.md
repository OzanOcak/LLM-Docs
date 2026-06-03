# Dot product (when to use which)

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Distance Metrics: Measuring Similarity**

---

### **1. Why this concept matters**

Cosine similarity ignores magnitude. Euclidean distance includes magnitude. Dot product is the raw, unnormalized inner product—it captures both similarity and magnitude in one number. It is the foundation of attention mechanisms, linear algebra, and recommendation systems. But when should you use dot product instead of cosine or Euclidean? The answer depends on whether vector magnitudes are meaningful. In neural networks, dot product with unnormalized vectors is efficient. In retrieval, dot product can be biased by frequent words. Understanding dot product means understanding the tradeoff between speed, simplicity, and fairness.

---

### **2. Core idea**

**Dot product computes the unnormalized similarity between vectors as the sum of elementwise products, measuring both alignment (angle) and magnitude, making it the basis for linear operations in neural networks but requiring careful interpretation for similarity search.**

---

### **3. Concrete analogy**

Imagine you are comparing two shopping lists. One list has 10 items, each checked once. Another list has 100 items, each checked once. The dot product counts how many items appear on both lists. But a long list might match by chance simply because it has more items.

- **Dot product:** Counts shared items. Long lists get higher scores even if the proportion of shared items is low. (Biased by magnitude)
- **Cosine similarity:** Measures the proportion of shared items relative to list length. Long lists are not favored. (Normalized for length)
- **Euclidean distance:** Measures the difference between lists. Long lists far from short lists.

Which to use? If you care about total purchases (e.g., a customer who buys 100 items is more valuable than one who buys 10), dot product is appropriate. If you care about preferences (e.g., taste similarity regardless of how many items), cosine is better.

---

### **4. ASCII diagram**

```
Dot product vs Cosine vs Euclidean:

    Vectors:
        a = [4, 0] (length 4, points east)
        b = [2, 0] (length 2, points east)
        c = [0, 4] (length 4, points north)

    Dot products:
        a·b = 8
        a·c = 0
        b·c = 0

    Cosine similarity:
        cos(a,b) = 8/(4×2)=1.0 (same direction)
        cos(a,c)=0/(4×4)=0 (perpendicular)

    Euclidean distance:
        d(a,b) = 2
        d(a,c) = √[(0-4)²+(4-0)²] = √32 ≈ 5.66


When to use each:

    Metric      | Best for                                      | Example
    ------------|-----------------------------------------------|-------------------
    Dot product | Raw similarity when magnitude matters         | Recommendation (user engagement)
    Cosine      | Similarity when magnitude is irrelevant       | Text embeddings, taste profiles
    Euclidean   | Distance when magnitude is signal (after norm) | Clustering, anomaly detection


Relationship between metrics (for unit vectors):

    If ‖a‖ = ‖b‖ = 1:
        dot = cos
        d_euclidean² = 2 - 2·dot

    If vectors are not normalized, dot = ‖a‖·‖b‖·cos
```

---

### **5. Mathematical formulation**

**Dot product (inner product):**

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^d a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta
$$

**Relationship to other metrics:**

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

$$
\|\mathbf{a} - \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2(\mathbf{a} \cdot \mathbf{b})
$$

**When to use dot product directly (without normalization):**

- **Recommendation systems:** User vector × item vector (engagement predicts clicks)
- **Neural network logits:** Final layer before softmax (raw scores)
- **Attention mechanisms:** Query · Key before scaling (unnormalized attention)
- **Linear algebra:** Projections, matrix multiplication

**When to avoid dot product for similarity search:**

- **Varying vector lengths:** Long documents dominate results
- **High-dimensional sparse data:** Frequent tokens dominate
- **When you need fair comparison** (normalize to unit vectors first)

**For similarity search (FAISS, vector DB):**

If all vectors are normalized (‖v‖=1), then:

$$
\text{cosine} = \text{dot}, \quad d_{\text{Euclidean}} = \sqrt{2 - 2\cdot\text{dot}}
$$

Thus, all three are equivalent. Many vector databases store normalized vectors and use dot product for speed.

---

### **6. Worked example (step-by-step)**

#### **Step 1: User and item vectors (recommendation)**

User vector (preferences for 3 genres):
u = [0.8, 0.3, 0.1] (likes action moderately, horror a little, romance not much)

Item vectors:
action_movie = [0.9, 0.1, 0.0]
horror_movie = [0.2, 0.9, 0.1]
romance_movie = [0.0, 0.1, 0.8]

#### **Step 2: Dot products**

u·action = 0.8×0.9 + 0.3×0.1 + 0.1×0.0 = 0.72 + 0.03 + 0 = 0.75
u·horror = 0.8×0.2 + 0.3×0.9 + 0.1×0.1 = 0.16 + 0.27 + 0.01 = 0.44
u·romance = 0.8×0.0 + 0.3×0.1 + 0.1×0.8 = 0 + 0.03 + 0.08 = 0.11

#### **Step 3: Interpretation**

Action movie gets highest dot product (0.75), then horror (0.44), then romance (0.11). This matches user preferences. Dot product works because vectors are already normalized (user preferences sum to 1.2? Not normalized; but magnitudes are similar). If user had large magnitude (e.g., u = [8,3,1]), dot would be 10× larger, but relative ordering may remain.

#### **Step 4: When dot product fails**

User u1 = [100, 0, 0] (active user, many interactions)
User u2 = [1, 0, 0] (light user)

Item i = [1, 0, 0]

u1·i = 100, u2·i = 1. Dot product says u1 is 100× more similar to i, but both have identical preferences (100% action). Cosine similarity would be 1.0 for both. For taste similarity, cosine is fairer. For total engagement (e.g., predicted clicks), dot product is correct.

---

### **7. How this appears inside neural networks and LLMs**

- **Attention mechanism (core of transformers):** Scores = Q·Kᵀ / √d_k. Dot product measures compatibility between query and key vectors. Scaling by √d_k prevents vanishing gradients.

- **Final layer logits:** Output = W·h + b. Dot product between hidden state h and each class vector (or token embedding). No normalization.

- **Embedding similarity search (vector DBs):** Many databases (FAISS, Pinecone, Milvus) support dot product indexing. For normalized vectors, dot = cosine; for unnormalized, dot favors large vectors.

- **Recommendation systems (collaborative filtering):** User embedding · item embedding = predicted rating. User and item magnitudes represent activity levels.

- **Logistic regression / linear layer:** logit = w·x + b. Dot product measures alignment between weight vector and input.

- **Cosine similarity variant:** After L2 normalization, dot product becomes cosine similarity. This is common in SBERT, CLIP, and retrieval models.

- **Inner product as similarity kernel:** In SVMs, kernel = x_i · x_j (linear kernel). Dot product measures similarity in input space.

- **Efficiency:** Dot product is O(d) and cache-friendly. Much faster than Euclidean or cosine (which require sqrt or normalization).

---

### **8. Brain-like connection (synaptic summation)**

The brain computes dot products at every synapse. A neuron receives input x_i from presynaptic neurons, each weighted by synaptic strength w_i. The total postsynaptic potential is Σ w_i x_i—a dot product between weight vector w (synaptic strengths) and input vector x (firing rates). This dot product determines whether the neuron fires (after thresholding). Magnitude matters: strong inputs produce larger potentials. The brain does not normalize inputs (divide by length) before summation—that would lose rate information. Dot product is the brain's native similarity metric for neural computation. Cosine would discard rate information; Euclidean would be slower to compute. Evolution chose dot product for speed and information preservation.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "For similarity search, dot product is just cosine similarity without the normalization. It is always worse."

_Why it is wrong:_ Dot product is not "un-normalized cosine." It is a different metric with different semantics. If magnitude matters, dot product is correct. For example, in recommendation: user who watches 100 action movies should be predicted to like action movies more than a user who watches 1, even if both have identical genre proportions. Dot product captures this; cosine does not. Many real-world systems (YouTube, Netflix) use dot product for this reason. The choice is not about normalization—it is about whether magnitude is signal or noise. Use dot when magnitude matters; use cosine when direction alone matters.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Dot product is the workhorse of linear algebra. It       |
|  powers neural network layers, attention mechanisms,      |
|  and recommendation systems. But when used for            |
|  similarity search, it favors large vectors. Sometimes    |
|  that is correct (active users get higher scores).        |
|  Sometimes it is bias (long documents crowd out short     |
|  ones). The key is knowing whether magnitude is signal    |
|  or noise. Normalize to unit vectors when it is noise.    |
|  Keep raw when it is signal. Dot product is not a poor    |
|  man's cosine—it is a different tool with different uses. |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a search engine for scientific papers. Each paper is represented by a TF-IDF vector (term frequency-inverse document frequency). TF-IDF vectors are already normalized for document length (TF part) but not for total vocabulary coverage (some papers use more unique terms). You need to rank papers by relevance to a short query. The query vector has the same normalization (each query term gets TF-IDF weight).

**Question:** Would you use dot product, cosine similarity, or Euclidean distance for ranking? Why? What would happen if you used dot product without normalization? How would you modify the vectors to make dot product equivalent to cosine?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Use cosine similarity (or equivalently, dot product after L2 normalization). Here's why:

- **TF-IDF vectors:** Document length (number of terms) is already normalized by TF, but the magnitude may still vary based on vocabulary coverage. Cosine similarity normalizes away this remaining magnitude variation.

- **Short query vs long document:** Dot product (without normalization) would favor longer documents (more terms, higher sum of weights). This is undesirable—relevance should depend on proportion of matches, not document length.

- **Euclidean distance:** Would also be biased by magnitude and less intuitive for retrieval (smaller distance = more similar, but scale varies).

**If you used dot product without normalization:** Long documents would systematically rank higher, even if less relevant. This is a known problem in early search engines.

**To make dot product equivalent to cosine:** L2-normalize each document vector and the query vector to unit length:

v' = v / ‖v‖

Then dot(v', q') = cosine(v, q). Many vector databases do this automatically or provide cosine as a primitive.

Thus: Use cosine or L2-normalize then dot. Avoid raw dot product for TF-IDF retrieval.
