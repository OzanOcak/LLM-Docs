# What are embeddings? (The "meaning space")

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Embeddings: Turning Words into Coordinates**

---

### **1. Why this concept matters**

A computer does not understand "king" or "queen." It understands numbers. But how do you turn a word into a number without losing meaning? You cannot assign 1=king, 2=queen—that would imply queen = 2×king. You need a richer representation. Embeddings solve this by mapping each word to a point in a high-dimensional space where similar words are close together. "King" and "queen" are near each other; "apple" and "orange" are near each other; "king" and "apple" are far apart. This "meaning space" is the foundation of every modern LLM. Without embeddings, a model cannot represent meaning at all.

---

### **2. Core idea**

**An embedding is a dense, low-dimensional vector representation of a discrete item (word, token, image, etc.) where semantically similar items are positioned close together in vector space, enabling mathematical operations on meaning.**

---

### **3. Concrete analogy**

Imagine a large map of a city. Each point of interest (restaurant, park, museum) has coordinates (x,y). Restaurants cluster together. Parks cluster together. The distance between points reflects their relationship—restaurants are near each other, far from parks.

Now imagine a 3D map where restaurants are not only grouped by type but also by cuisine: Italian restaurants form one cluster, Thai another. And a 4th dimension captures price range. And a 5th captures ambiance. This is embedding space: hundreds of dimensions, each capturing a different "feature" of meaning.

"King - man + woman = queen" becomes a geometric operation: take the vector for "king," subtract "man," add "woman," and you land at the vector for "queen." That is not a trick—it is a property of the embedding space. The vectors capture the underlying dimensions of meaning: royal, male, female, human.

---

### **4. ASCII diagram**

```
Embedding space (3D projection of 100D space):

    Dimension 2 (royalty)
         ↑
         │
         │       ● King
         │      ╱
         │     ╱
         │    ╱   ● Queen
         │   ╱
         │  ╱
         │ ╱
    ─────┼────────────────────→ Dimension 1 (gender)
         │      ● Man
         │
         │           ● Woman
         │
         │
         │               ● Apple
         │


Properties in embedding space:

    Similar words cluster together:
        ● King, ● Queen, ● Prince (royalty cluster)
        ● Apple, ● Orange, ● Banana (fruit cluster)

    Analogies as vector arithmetic:
        King - Man + Woman ≈ Queen

        Dog - Puppy + Cat ≈ Kitten

    Distance measures meaning:
        ‖King - Queen‖ < ‖King - Apple‖


Embedding space dimensions (learned, not labeled):

    Dimension 1: gender (male → female axis)
    Dimension 2: royalty (commoner → royal axis)
    Dimension 3: living (inanimate → animate)
    Dimension 4: size (small → large)
    Dimension 5: value (cheap → expensive)
    ...
    (Hundreds of dimensions, each capturing a semantic feature)
```

---

### **5. Mathematical formulation**

**Embedding lookup:**

Given vocabulary V of size |V| (e.g., 50,000), embedding matrix E ∈ ℝ^{|V| × d}, where d is embedding dimension (e.g., 300).

Word with index i maps to vector:

$$
\mathbf{e}_i = E[i, :] \in \mathbb{R}^d
$$

**Word2vec (skip-gram) objective:**

Maximize probability of context words given target word:

$$
L = \sum_{(w, c) \in \text{corpus}} \log \frac{\exp(\mathbf{e}_w \cdot \mathbf{e}_c)}{\sum_{v \in V} \exp(\mathbf{e}_w \cdot \mathbf{e}_v)}
$$

**GloVe (count-based):**

$$

L = \sum_{i,j} f(X_{ij}) (\mathbf{e}_i \cdot \mathbf{e}_j + b_i + b_j - \log X_{ij})^2


$$

Where X\_{ij} is co-occurrence count.

**Cosine similarity (measuring meaning closeness):**

$$
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \cos(\theta)
$$

Range: -1 (opposite) to +1 (same direction). 0 = unrelated.

**Analogy via vector arithmetic:**

$$
\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \approx \mathbf{e}_{\text{queen}}
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Tiny embedding space (d=3, |V|=5)**

Vectors (random initialization for illustration):

"king" = [0.8, 0.6, 0.1]
"queen" = [0.7, 0.5, 0.2]
"man" = [0.4, 0.8, 0.1]
"woman" = [0.3, 0.7, 0.2]
"apple" = [0.1, 0.1, 0.9]

#### **Step 2: Compute cosine similarities**

sim(king, queen) = (0.8×0.7 + 0.6×0.5 + 0.1×0.2) / (1.01×0.87) ≈ (0.56+0.30+0.02)/0.88 = 0.88/0.88 = 1.0? (Intentionally high)

sim(king, apple) = (0.8×0.1 + 0.6×0.1 + 0.1×0.9) / (1.01×0.91) = (0.08+0.06+0.09)/0.92 = 0.23/0.92 = 0.25

Interpretation: King and queen are very similar (0.99). King and apple are not (0.25). The embedding captures semantic relationships.

#### **Step 3: Vector arithmetic analogy**

Compute king - man + woman = [0.8-0.4+0.3, 0.6-0.8+0.7, 0.1-0.1+0.2] = [0.7, 0.5, 0.2]

This equals the "queen" vector! The embedding learned the gender dimension.

#### **Step 4: PCA visualization**

Project 300D embeddings to 2D for plotting. Clusters emerge: royalty cluster, gender subclusters, food cluster, etc. The "meaning space" is structured, not random.

---

### **7. How this appears inside neural networks and LLMs**

- **Word2Vec (2013):** The breakthrough. Static embeddings (one vector per word, no context). Still used for many classical NLP tasks.

- **GloVe (2014):** Count-based embeddings. Combines matrix factorization with local context.

- **FastText (2016):** Subword embeddings. Handles rare words by using character n-grams.

- **Contextual embeddings (BERT, GPT):** Dynamic embeddings—same word has different vectors based on context. "Bank" in "river bank" vs "money bank" differs.

- **Embedding layer in LLMs:** First layer of every LLM. Token ID → vector. Trained from scratch during pre-training.

- **Semantic search:** Compare query embedding to document embeddings (cosine similarity) to retrieve relevant documents.

- **Sentence embeddings (SBERT):** Pool token embeddings to produce fixed-length sentence vectors for similarity search.

- **Multilingual embeddings (XLM-R, LaBSE):** Same embedding space across languages. "Dog" (English) near "Chien" (French).

- **Downstream uses:** Recommendation (item embeddings), clustering (customer segments), outlier detection.

---

### **8. Brain-like connection (semantic space)**

The brain represents meaning in a high-dimensional space. Neuroimaging studies show that words like "dog" and "cat" activate similar neural patterns (temporal lobe). "Dog" and "car" activate different patterns. This is an embedding space in the brain. Moreover, the brain supports analogical reasoning through vector arithmetic: neural pattern for "king" minus "man" plus "woman" approximates pattern for "queen." The dimensions of this semantic space correspond to semantic features: animacy, size, danger, valence. Embeddings are not just a convenient ML trick—they are a mathematical model of how the brain organizes meaning.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Embeddings are just a way to convert words to numbers. One-hot encoding works too."

_Why it is wrong:_ One-hot encoding creates orthogonal vectors (all distances equal). "King" and "queen" are as similar as "king" and "apple" under one-hot (zero similarity). This loses all semantic information. Embeddings learn structure: similar words cluster, analogies hold, distances reflect meaning. One-hot is sparse, high-dimensional (|V|), and semantically meaningless. Embeddings are dense, low-dimensional (typically 100-1000), and capture meaning. The difference is not minor—it is the difference between a model that understands relationships and one that treats all words as unrelated symbols.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Embeddings are the Rosetta Stone of AI. They translate   |
|  discrete symbols (words) into continuous vectors that    |
|  capture meaning. In this space, "cat" is near "kitten",  |
|  "king - man + woman = queen", and semantic search        |
|  becomes nearest neighbor lookup. Every LLM begins with   |
|  an embedding layer—it is the first step from text to     |
|  thought. Without embeddings, there is no meaning.        |
|  With them, words become geometry.                        |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have trained a Word2Vec model on a large text corpus. You find that the vector for "doctor" is equally close to "man" and "woman" (cosine similarity 0.6 and 0.6). For "nurse", the vector is closer to "woman" (0.7) than to "man" (0.4).

**Question:** What does this indicate about bias in the training data? Could this cause problems in downstream applications? How would you mitigate it?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** This indicates gender bias in the training data. "Doctor" is equally associated with men and women (ideal), but "nurse" is more associated with women (biased). The model learned this bias from real-world text where nurses are disproportionately referred to as female.

This causes problems: if you use embeddings for resume screening, the model may unfairly rank female candidates lower for "doctor" roles? Actually here "doctor" is unbiased, but "nurse" would rank female candidates higher, potentially reinforcing stereotypes.

Mitigation strategies:

1. Debiasing algorithms (Hard-Debias, Soft-Debias) to remove gender dimensions from embeddings
2. Balanced training data (augment to equalize gender associations)
3. Post-hoc correction of embedding space
4. Use contextual embeddings (BERT) which may exhibit less bias but still require debiasing

Key lesson: Embeddings reflect training data biases. If your data is biased, your meaning space will encode those biases. Always audit embeddings for fairness before deployment.
