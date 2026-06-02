# Properties of embedding spaces (linear relationships, analogies)

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Embeddings: Turning Words into Coordinates**

---

### **1. Why this concept matters**

Embedding spaces are not random. They have rich linear structure: directions correspond to semantic dimensions (gender, tense, size), and analogies become vector arithmetic. "King - Man + Woman = Queen" is not a parlor trick—it is a fundamental property of the space. This linearity means you can solve analogies by vector addition, find word clusters via centroids, and even debias by subtracting unwanted dimensions. Understanding these properties is understanding how meaning organizes itself into geometry.

---

### **2. Core idea**

**Word embedding spaces exhibit linear substructure where semantic relationships (e.g., gender, tense, plurals) correspond to consistent vector offsets, enabling analogical reasoning through vector arithmetic and revealing the geometric organization of meaning.**

---

### **3. Concrete analogy**

Imagine a 2D map of a city. The vector from the library to the museum is (2 km east, 1 km north). The vector from the school to the hospital is also (2 km east, 1 km north). You discover a relationship: "library is to museum as school is to hospital." The city's geography has consistent offsets.

Now add a 3rd dimension: elevation. Rich neighborhoods have higher elevation. The vector from the library to the museum might have a small upward component, representing increased richness. This is an embedding space.

In word embeddings, each dimension captures a semantic feature. The vector from "man" to "woman" represents "gender." The same vector added to "king" yields "queen." The space is not just a set of points—it is a structured geometric object where directions correspond to meanings.

---

### **4. ASCII diagram**

```
Linear relationships in embedding space:

    Dimension: Gender
         ↑
         │   king─────────queen
         │    ╱           ╱
         │   ╱           ╱
         │  ╱           ╱
         │ ╱           ╱
         │man─────────woman
         │
         └────────────────────────→ Dimension: Royalty

    Offsets:
    Δ_gender = e_woman - e_man = (0.3, -0.1, 0.2)
    Δ_royalty = e_king - e_man = (0.5, 0.8, 0.1)

    e_queen = e_woman + Δ_royalty
    e_queen = e_king + Δ_gender


Common analogy types and their offsets:

    Type          | Example                    | Offset
    --------------|---------------------------|----------------
    Gender        | man:woman :: king:queen   | gender
    Tense         | walk:walked :: run:ran    | past-tense
    Plural        | cat:cats :: dog:dogs      | plural
    Country-capital| France:Paris :: Japan:Tokyo| capital-of
    Degree        | good:better :: fast:faster| comparative
    Antonym       | good:bad :: hot:cold      | negation


Vector arithmetic examples:

    e_king - e_man + e_woman ≈ e_queen
    e_paris - e_france + e_japan ≈ e_tokyo
    e_running - e_run + e_walk ≈ e_walking
```

---

### **5. Mathematical formulation**

**Analogy as vector offset:**

For analogy a:b :: c:d, there exists a relationship vector r such that:

$$
\mathbf{e}_b - \mathbf{e}_a \approx \mathbf{e}_d - \mathbf{e}_c = \mathbf{r}
$$

**Solving analogies via vector addition:**

$$
\mathbf{e}_d \approx \mathbf{e}_c + (\mathbf{e}_b - \mathbf{e}_a)
$$

**Cosine similarity evaluation:**

Find d' that maximizes cosine similarity:

$$
d^* = \arg\max_{v \in V, v \neq c} \cos(\mathbf{e}_v, \mathbf{e}_c + \mathbf{e}_b - \mathbf{e}_a)
$$

**Linear substructure discovery (PCA):**

Principal components of embedding space often align with meaningful dimensions:

$$
\mathbf{v}_{\text{gender}} \approx \text{PC}_1, \quad \mathbf{v}_{\text{tense}} \approx \text{PC}_2, \ldots
$$

**Bias direction identification:**

Compute gender direction vector:

$$
\mathbf{g} = \frac{1}{|M|} \sum_{m \in M} \mathbf{e}_m - \frac{1}{|F|} \sum_{f \in F} \mathbf{e}_f
$$

Where M and F are sets of gender-defining word pairs.

**Debiasing (hard de-biasing):**

Project vectors orthogonal to bias direction:

$$
\mathbf{e}' = \mathbf{e} - \frac{\mathbf{e} \cdot \mathbf{g}}{\|\mathbf{g}\|^2} \mathbf{g}
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Embedding vectors (hypothetical 3D)**

man = [0.4, 0.2, 0.1]
woman = [0.3, 0.5, 0.2]
king = [0.9, 0.3, 0.1]
queen = [0.8, 0.6, 0.2]

#### **Step 2: Compute gender offset**

Δ_gender = woman - man = [0.3-0.4, 0.5-0.2, 0.2-0.1] = [-0.1, 0.3, 0.1]

#### **Step 3: Apply offset to king**

king + Δ_gender = [0.9-0.1, 0.3+0.3, 0.1+0.1] = [0.8, 0.6, 0.2] = queen

Perfect! The gender offset generalizes.

#### **Step 4: Test a different analogy**

walk = [0.2, 0.1, 0.7]
walked = [0.3, 0.2, 0.5]
run = [0.4, 0.1, 0.8]

Compute tense offset: walked - walk = [0.1, 0.1, -0.2]

Prediction: run + tense_offset = [0.5, 0.2, 0.6]

Expected "ran" vector (hypothetical) = [0.5, 0.2, 0.6] — works!

#### **Step 5: Failure case (antonym)**

good = [0.8, 0.1, 0.2]
bad = [0.1, 0.8, 0.3]
fast = [0.7, 0.2, 0.4]

bad - good = [-0.7, 0.7, 0.1]

fast + (bad - good) = [0.0, 0.9, 0.5]

Expected "slow"? Not necessarily. Antonym offsets are less consistent than gender. The space may not have a linear "negation" dimension.

---

### **7. How this appears inside neural networks and LLMs**

- **Word2vec analogies benchmark:** Classic evaluation: "man:woman :: king:?" Accuracy ~75% for 300-dim embeddings on Google analogy dataset.

- **GloVe (Global Vectors):** Count-based embeddings show even stronger linear substructure. The offset "Paris - France" clusters near other country-capital pairs.

- **FastText:** Subword embeddings preserve analogies for rare words and morphological relationships (e.g., "run:running :: walk:walking").

- **Contextual embeddings (BERT, GPT):** Linear analogies hold less consistently because embeddings are context-dependent. Same word has different vectors in different sentences. However, static embeddings derived from pooling (e.g., SBERT) still show analogies.

- **Bias detection:** Linear direction for gender can be computed. Project profession words onto this direction: "doctor" has higher projection than "nurse" in biased spaces. Quantifies bias.

- **Debiasing algorithms:** Hard-debias, soft-debias, and conceptor debiasing all exploit linear substructure to remove unwanted dimensions.

- **Cross-lingual analogies:** After aligning embedding spaces, analogies hold across languages. "dog:puppy :: chien:chiot" (English to French).

- **Semantic search:** Use analogy to expand queries: "metal:iron :: material:steel" adds related terms.

---

### **8. Brain-like connection (continuous semantic spaces)**

Cognitive neuroscience suggests semantic memory is organized as a continuous space, not discrete categories. Dimensions include animacy, size, danger, valence. fMRI studies show that word meanings are represented in a smooth 2D space (posterior temporal lobe), with continuous gradations: "dog" near "cat", further from "car". This space supports analogical reasoning—the brain can compute vector offsets between concepts. Moreover, the brain's semantic space shows linearity: the neural pattern for "king" minus "man" plus "woman" approximates the pattern for "queen". Embedding spaces are not just mathematical constructs—they are models of how the brain organizes meaning.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Embedding analogies always work perfectly. If an embedding fails an analogy, it is a bad embedding."

_Why it is wrong:_ Analogies fail for many reasons. Polysemous words (bank) have multiple meanings—which sense should the analogy use? Rare words have poorly estimated embeddings. Relations that are not linear (negation, causality) may not be captured. The embedding space is only approximately linear. For contextual embeddings (BERT), analogies fail often because the same word has different vectors per context. Analogies are a useful property, but not a requirement. Many useful embedding spaces have poor analogy performance but excel at other tasks. Treat analogies as a diagnostic, not a hard requirement.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Embedding spaces are not just clusters—they have         |
|  geometry. Directions correspond to gender, tense,        |
|  plurals, and country-capital relationships. Analogies    |
|  become vector arithmetic. This linearity is powerful:    |
|  you can solve analogies by addition, debias by           |
|  projection, and discover hidden semantic dimensions      |
|  with PCA. Understanding this geometry means you can      |
|  manipulate meaning mathematically.                       |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a word embedding space. You compute the vector for "Spain" minus "Madrid" plus "Paris". The closest word to the result is "France" (cosine similarity 0.89). You repeat: "Japan" minus "Tokyo" plus "Paris". The closest word is "France" again (0.92).

**Question:** What property of the embedding space does this demonstrate? Why does "Paris" act as a "probe" for country vectors? Could you use this to identify capital cities for any country?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** This demonstrates that the embedding space has a consistent "country-capital" offset. The vector from a country to its capital is roughly constant: e_capital ≈ e_country + Δ. Therefore:

e_capital - e_country ≈ Δ

e_country ≈ e_capital - Δ

Given e_country + (e_paris - e_france) = e_country + Δ = e_capital (for that country).

Thus, "Japan" + (Paris - France) ≈ "Tokyo".

This works because the offset Δ is shared across countries. "Paris" acts as a probe because France is the reference country. To identify a capital city for any country X:

e_capital_X ≈ e_X + (e_paris - e_france)

You need to know the offset from a known country-capital pair (e.g., France-Paris). Then add that offset to any country vector. This is how analogy-based reasoning works: learn one relationship, apply to others.

Yes, you could use this to identify capital cities if the embedding space has learned the relationship. Test on "China" + (Paris - France) → "Beijing". Works if the space is sufficiently linear and the country vectors are well-separated. This is how early analogy solvers worked.
