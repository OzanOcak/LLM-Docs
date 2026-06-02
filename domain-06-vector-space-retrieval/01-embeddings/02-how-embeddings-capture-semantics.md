# How embeddings capture semantics

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Embeddings: Turning Words into Coordinates**

---

### **1. Why this concept matters**

Knowing that embeddings map words to vectors is one thing. Understanding how they capture meaning is another. The magic is not in the representation itself—it is in the geometry. Similar words cluster together. Directions in embedding space correspond to semantic dimensions (gender, tense, plurality). Analogies become vector arithmetic. This structured geometry is learned automatically from raw text, without human labels. Embeddings capture semantics because words that appear in similar contexts have similar meanings. This is the distributional hypothesis: "a word is characterized by the company it keeps."

---

### **2. Core idea**

**Embeddings capture semantics through the distributional hypothesis: words that appear in similar contexts have similar meanings, and this relationship emerges as geometric proximity and linear substructures (analogies) in the learned vector space.**

---

### **3. Concrete analogy**

Imagine you are an alien trying to understand English without a dictionary. You listen to conversations and notice:

- "I ate an apple" and "I ate a banana" appear in similar patterns.
- "The dog barked" and "The cat meowed" share structure.

You hypothesize: words that appear in the same contexts probably have similar meanings. This is the distributional hypothesis.

Now you create a map. You place words so that words appearing in similar contexts are close together. "Apple" and "banana" end up near each other (fruit cluster). "Dog" and "cat" near each other (pet cluster). But you also notice patterns: the vector from "man" to "woman" is similar to the vector from "king" to "queen". You have discovered the gender dimension.

This map is an embedding space. It was built without anyone telling you what "fruit" or "gender" means—just from observing word patterns.

---

### **4. ASCII diagram**

```
Distributional hypothesis visualization:

    Context patterns for "apple" and "banana":

    "I ate an ___"
    "She bought a ___"
    "The ___ was delicious"

    Similar contexts → similar meaning → nearby in embedding space.


Geometric structure of embedding space:

    Dimension: Gender
         ↑
         │   (king)  (queen)
         │     ●─────●
         │    ╱       ╲
         │   ╱         ╲
         │  ╱           ╲
         │ ●─────────────●
         │ (man)        (woman)
         │
         └────────────────────────→ Dimension: Royalty


Analogy as vector arithmetic:

    e(king) - e(man) = gender_vector
    e(queen) - e(woman) = gender_vector (same!)

    Therefore: e(king) - e(man) + e(woman) = e(queen)


Semantic dimensions discovered automatically:

    Dimension   |  Examples
    ------------|-----------------------------------
    Gender      |  man → woman, king → queen
    Tense       |  walk → walked, run → ran
    Plurality   |  cat → cats, child → children
    Country-capital | France → Paris, Japan → Tokyo
    Comparative |  fast → faster, good → better
```

---

### **5. Mathematical formulation**

**Distributional hypothesis formalized (Word2vec skip-gram):**

Maximize the probability of context words given target word w:

$$
P(c | w) = \frac{\exp(\mathbf{e}_w \cdot \mathbf{e}_c)}{\sum_{v \in V} \exp(\mathbf{e}_w \cdot \mathbf{e}_v)}
$$

This forces embeddings to represent words by their contextual co-occurrence.

**Cosine similarity (semantic similarity):**

$$
\text{sim}(w_1, w_2) = \frac{\mathbf{e}_{w_1} \cdot \mathbf{e}_{w_2}}{\|\mathbf{e}_{w_1}\| \|\mathbf{e}_{w_2}\|}
$$

Words with similar contexts have high cosine similarity.

**Analogy via vector offset:**

Given a:b :: c:d, the embedding satisfies:

$$
\mathbf{e}_b - \mathbf{e}_a \approx \mathbf{e}_d - \mathbf{e}_c
$$

Thus:

$$
\mathbf{e}_d \approx \mathbf{e}_c + (\mathbf{e}_b - \mathbf{e}_a)
$$

**Principal component analysis (PCA) of embedding space:**

The top principal components correspond to semantic dimensions like gender, tense, formality. These emerge automatically from the data.

**Context window size effect:**

- Small window (2-5 words): captures syntactic relationships (verb tense, pluralization)
- Large window (10+ words): captures semantic topics (fruit cluster, country cluster)

---

### **6. Worked example (step-by-step)**

#### **Step 1: Tiny corpus**

Sentences:

- "The cat chased the mouse"
- "The dog chased the cat"
- "The mouse ate the cheese"

#### **Step 2: Count contexts (window size 2)**

"cat" appears with: "the", "chased", "the", "dog", "chased", "the" (ignoring stopwords for simplicity)

"dog" appears with: "the", "chased", "the", "cat"

"mouse" appears with: "the", "chased", "the", "the", "ate", "the", "cheese"

#### **Step 3: Context vectors (simplified)**

Cat context: {chased:2, dog:1, cat:1} (self)
Dog context: {chased:2, cat:1}
Mouse context: {chased:1, ate:1, cheese:1}

#### **Step 4: Embeddings via matrix factorization**

After training Word2vec on this corpus, vectors might be:

"cat" = [0.8, 0.3, 0.1]
"dog" = [0.7, 0.4, 0.1]
"mouse" = [0.2, 0.8, 0.6]
"cheese" = [0.1, 0.7, 0.9]

#### **Step 5: Compute similarities**

cos(cat, dog) = high (0.95) — both are pets/animals
cos(cat, mouse) = medium (0.60) — both animals but different roles
cos(mouse, cheese) = high (0.85) — mouse associated with cheese

#### **Step 6: Analogy discovery**

e(cat) - e(kitten) approximates e(dog) - e(puppy). The vector for "young" emerges from animal pairs.

---

### **7. How this appears inside neural networks and LLMs**

- **Word2vec (2013):** Showed that analogies emerge from embeddings: "Paris - France + Italy ≈ Rome". First demonstration of semantic geometry.

- **GloVe (2014):** Global co-occurrence counts produce smoother embedding spaces with better analogy performance.

- **Contextual embeddings (BERT, GPT):** Same word gets different vectors based on context. "Bank" in "river bank" vs "money bank" maps to different regions of embedding space.

- **Sentence embeddings (SBERT, SimCSE):** Pool token embeddings to produce sentence vectors. Semantically similar sentences cluster.

- **Cross-lingual embeddings (MUSE, LASER):** Align embedding spaces across languages. "Dog" (English) and "Chien" (French) near each other.

- **Embedding arithmetic in practice:** Search engines use "query = embed(user_query) - embed(unwanted_concept) + embed(wanted_concept)" for semantic query reformulation.

- **Word analogies as evaluation:** Common benchmark: "man:woman :: king:?" Accuracy of embedding models on analogy tasks measures semantic quality.

- **Bias detection:** Embedding space geometry reveals biases: "doctor" nearer "man" than "woman" in biased corpora. Cosine difference quantifies bias.

---

### **8. Brain-like connection (semantic priming)**

In cognitive psychology, semantic priming shows that hearing "doctor" makes you faster to recognize "nurse" (related) than "table" (unrelated). The brain activates related concepts automatically—they are "near" in neural semantic space. Priming effects are symmetric: "doctor" primes "nurse" as much as "nurse" primes "doctor". This aligns with cosine similarity (symmetric). Moreover, analogical reasoning in the brain (A is to B as C is to D) appears to use vector arithmetic in neural population codes. The geometry of embedding space is not just a mathematical convenience—it mirrors how the human brain organizes semantic memory.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Embeddings capture word meaning directly. The vector for 'doctor' encodes all properties of doctors."

_Why it is wrong:_ Embeddings capture only _distributional_ meaning—similarity based on context. They do not capture entailment, negation, or logical relationships. "Dog" and "animal" are related, but the embedding might not capture that "dog" is a subclass of "animal" (hierarchical). "All dogs are animals" is not encoded as a geometric relationship. Embeddings tell you which words are similar, not which are true. A sentence "dogs can fly" would produce a similar embedding to "dogs can run" if both appear in similar contexts. Embeddings capture statistical patterns, not truth. This is why LLMs need massive scale—statistical patterns approximate truth, but the embedding itself does not represent logical semantics.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Embeddings turn meaning into geometry. In this space,    |
|  semantic similarity is cosine distance. Analogies are    |
|  vector sums. Dimensions correspond to features like      |
|  gender, tense, and topic. This geometry is not designed— |
|  it emerges from the distributional patterns of words     |
|  in text. Embeddings capture the subtle, multidimensional |
|  structure of meaning that eludes hand-coded rules.       |
|  Understanding this geometry is understanding how AI      |
|  "knows" what words mean.                                 |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have embeddings trained on a corpus. You find that:

cos(king, queen) = 0.75
cos(king, man) = 0.65
cos(queen, woman) = 0.68
cos(man, woman) = 0.60

**Question:** What is the approximate result of the analogy "king : man :: queen : ?" Show the vector arithmetic and explain why the embedding space supports this.

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** The analogy is "king is to man as queen is to woman". Vector arithmetic:

e(queen) ≈ e(woman) + (e(king) - e(man))

Compute difference vector: e(king) - e(man) = gender_offset (man→king adds royalty)

e(queen) should equal e(woman) + gender_offset

The embedding space supports this because:

1. The gender dimension (man→woman) is approximately parallel to the gender dimension in royalty (king→queen)
2. The royalty offset (man→king) is parallel to woman→queen

Given cosine similarities, we can infer that:

- e(king) - e(man) and e(queen) - e(woman) are approximately the same vector (royalty addition)
- Therefore, e(king) - e(man) + e(woman) ≈ e(queen)

The analogy holds if the embedding space has linear substructure for the gender and royalty dimensions. The high cosine similarities between king/queen and man/woman indicate this structure exists. The answer is "woman".
