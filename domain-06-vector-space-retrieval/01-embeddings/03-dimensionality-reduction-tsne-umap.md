# Dimensionality reduction for visualization (t-SNE, UMAP)

## **DOMAIN: VECTOR SPACE & RETRIEVAL | Sub domain: Embeddings: Turning Words into Coordinates**

---

### **1. Why this concept matters**

Word embeddings live in 100- to 1000-dimensional space. You cannot visualize 1000 dimensions. But you can project them down to 2D or 3D while preserving the structure—nearby points in high-D remain nearby in low-D, far points remain far. This is dimensionality reduction for visualization. t-SNE and UMAP are the most popular algorithms for this task. They reveal clusters: all fruit words together, all programming languages together, all emotions together. Without these techniques, embedding spaces would be invisible black boxes. With them, you can see meaning.

---

### **2. Core idea**

**t-SNE and UMAP are nonlinear dimensionality reduction techniques that project high-dimensional embeddings into 2D or 3D while preserving local neighborhood structure, enabling visualization of clusters and semantic relationships.**

---

### **3. Concrete analogy**

Imagine you have a 3D sculpture of a city. You want to photograph it from above, but a single 2D photo loses depth—some buildings that are far apart look close if they align in the photo.

t-SNE is like a magic camera that preserves neighborhood relationships: buildings that are close in 3D appear close in the photo. Distant buildings appear distant. The photo is distorted (no longer a linear projection), but it reveals the structure: residential clusters, industrial clusters, downtown cluster.

UMAP is like a similar camera but faster, with slightly different tradeoffs: it preserves more of the global structure (the overall shape of the city) while still keeping local clusters.

For embeddings, these algorithms project 300D vectors (word meanings) to 2D dots you can plot. "Cat" and "dog" become nearby dots. "Apple" (fruit) and "orange" cluster separately from "Apple" (company) if the embedding distinguishes them.

---

### **4. ASCII diagram**

```
t-SNE/UMAP process: 300D → 2D

    300D embedding space              2D visualization
    (invisible)                      (we can see)

    Each word is a point.            Nearby points in 300D
    Similar words cluster.           remain nearby in 2D.

    ○ ○ ○   ○ ○     ● ● ●   ● ●
     ○ ○ ○ ○ ○ ○    ● ● ● ● ● ●
       fruit        vegetables
        cluster        cluster

    ○ ○ ○   ○ ○     ○ ○ ○   ○ ○
    animals        emotions
     cluster         cluster


Comparison of algorithms:

    PCA:              t-SNE:                UMAP:
    Linear            Nonlinear             Nonlinear
    Preserves global  Preserves local       Preserves local + global
    structure         neighborhoods         structure
    Fast              Slow                  Medium-fast
    Distant points    Clusters only (may    Better global
    stay distant      split connected       layout
                        clusters)

    Best for:         Best for:            Best for:
    Overview,         Discovering clusters, Interactive,
    simple data       qualitative analysis  large datasets


Real t-SNE plot of word embeddings (conceptual):

    Dimension 2
         ↑
         │   animals
         │   cat●      ●fruits
         │   dog●     ●apple
         │    ●fish    ●banana
         │
         │   vehicles
         │   car●      ●emotions
         │   truck●   ●happy
         │            ●sad
         └────────────────────────→ Dimension 1
```

---

### **5. Mathematical formulation**

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**

Step 1: Convert high-D distances to probabilities (similarity of i to j):

$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

Step 2: Symmetrize: $p_{ij} = (p_{j|i} + p_{i|j}) / 2n$

Step 3: In low-D space (2D points y_i), define similar probabilities using t-distribution:

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
$$

Step 4: Minimize KL divergence between P and Q:

$$
\text{KL}(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
\]

**UMAP (Uniform Manifold Approximation and Projection):**

Constructs a graph in high-D (k-nearest neighbors), then optimizes a low-D layout that preserves the graph structure using cross-entropy loss and stochastic gradient descent.

**Key differences:**

- t-SNE uses Gaussian in high-D, t-distribution in low-D
- UMAP uses manifold assumptions (uniform distribution on a manifold)
- UMAP scales better to large datasets (millions of points)
- t-SNE tends to create tighter clusters; UMAP preserves more global structure

---

### **6. Worked example (step-by-step)**

#### **Step 1: Starting embeddings (300D, concept only)**

Assume 6 words with 300D embeddings (simplified to 2D for illustration):

cat = [0.8, 0.2]
dog = [0.7, 0.3]
fish = [0.6, 0.1]
apple = [0.1, 0.8]
banana = [0.2, 0.7]
car = [0.5, 0.5] (confused? Not realistic, just example)

#### **Step 2: t-SNE initialization**

Randomly place 2D points (perplexity=2). Iteratively move them to match high-D similarities.

#### **Step 3: After optimization**

t-SNE output (example):

cat = (1.2, 0.5)
dog = (1.1, 0.6)
fish = (1.0, 0.4)
apple = (-1.0, -0.8)
banana = (-0.9, -0.7)
car = (0.0, 0.0) (between clusters, maybe outlier)

#### **Step 4: Plot**

Scatter plot shows:
- Animals cluster (cat, dog, fish) near (1.0, 0.5)
- Fruits cluster (apple, banana) near (-0.9, -0.7)
- Car may be isolated (no cluster)

#### **Step 5: Interpretation**

The visualization reveals semantic categories: animals, fruits. Car might be too far from both—maybe missing from vocabulary. t-SNE makes cluster structure visible.

---

### **7. How this appears inside neural networks and LLMs**

- **Word embedding visualization:** Classic use. Word2vec, GloVe, FastText vectors projected to 2D reveal semantic clusters: animals, emotions, professions, etc.

- **LLM hidden state analysis:** Take activations from middle layers of BERT or GPT, apply t-SNE/UMAP to see how the model organizes concepts internally.

- **Attention pattern visualization:** Compute attention vectors (how much each token attends to others), reduce to 2D to see attention heads' focus.

- **Dataset exploration:** Embed sentences/documents, reduce to 2D, color by label. Reveals mislabeled data, outliers, or distribution shifts.

- **Model debugging:** Embed model predictions. If clusters of errors appear (e.g., all "cat" misclassified as "dog"), the embedding space may be insufficiently separated.

- **Embedding drift monitoring:** During fine-tuning, compare pre vs post embeddings with UMAP. If clusters shift dramatically, the model may be overfitting or forgetting.

- **Fasttext for visualization:** Subword embeddings produce smoother UMAP plots (rare words get meaningful positions).

- **Interactive exploration:** Tools like TensorFlow Projector, umap-learn, and openTSNE allow interactive visualization of embeddings.

---

### **8. Brain-like connection (tonotopic and retinotopic maps)**

The brain maps high-dimensional sensory inputs onto 2D cortical sheets with preserved neighborhoods. In auditory cortex, frequencies are mapped tonotopically: nearby neurons respond to similar pitches. In visual cortex, retinotopic maps preserve spatial adjacency: points near each other in the visual field activate nearby cortical neurons. This is biological dimensionality reduction: the brain projects high-D sensory data onto 2D cortical surfaces while preserving local topology. t-SNE and UMAP achieve something analogous computationally. The brain's maps are fixed by development; t-SNE optimizes maps per dataset. But the principle—local neighborhoods in high-D become local neighborhoods in 2D—is shared.

---

### **9. Common misunderstanding and why it is wrong**

*Misunderstanding:* "t-SNE preserves global distances. If two clusters are far apart in the plot, they are far apart in the original high-D space."

*Why it is wrong:* t-SNE preserves local neighborhoods but not global distances. Clusters that appear far apart in the plot might actually be moderately close in high-D if separated by empty space. Conversely, clusters that appear separated could be artificially split due to perplexity settings. t-SNE can also distort relative cluster sizes: a large cluster may appear smaller than a small cluster. Always interpret t-SNE plots qualitatively, not quantitatively. Do not measure distances between clusters—they are not reliable. UMAP preserves more global structure, but still distances are not Euclidean. Use PCA for global distance preservation. Use t-SNE/UMAP for discovering clusters, not measuring distances.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Embedding spaces are invisible. Hundreds of dimensions,  |
|  no way to see. t-SNE and UMAP are microscopes for       |
|  meaning space. They reveal clusters: animals, fruits,    |
|  programming languages, emotions. They expose biases:     |
|  "doctor" near "man", "nurse" near "woman". They help     |
|  debug models, explore data, and communicate findings.    |
|  Without them, embeddings are abstract math. With them,   |
|  you can see the geometry of meaning.                     |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have 1000 word embeddings (768 dimensions each) from BERT. You apply t-SNE to project to 2D and observe that "tiger", "lion", "leopard" form a tight cluster, while "car", "truck", "bus" form another cluster far away. "Jaguar" (the car brand) appears exactly midway between the animal cluster and the vehicle cluster.

**Question:** What does this tell you about BERT's embedding of "Jaguar"? Is this correct behavior? Would this cause problems for a search engine?

*(Answer hidden below)*

---

.

.

.

.

.

**Answer:** BERT's embedding of "Jaguar" captures its polysemy (multiple meanings). The word is ambiguous between animal (big cat) and car brand (luxury vehicle). In the embedding space, "Jaguar" is pulled toward both clusters, landing in between. This is semantically correct—the embedding represents ambiguity.

For a search engine, this could cause problems:
- Search for "Jaguar" might retrieve results about both animals and cars, which may or may not be desired.
- Query expansion "similar to Jaguar" would mix animal and car concepts, reducing precision.

Solutions:
1. Contextual disambiguation: Use surrounding words to determine sense ("Jaguar animal" vs "Jaguar car").
2. Sense-specific embeddings: Separate vectors per word sense (e.g., WordNet senses).
3. Use sentence embeddings not word embeddings for retrieval to leverage context.

The midway position is not a bug—it is a feature of distributional embeddings. It correctly reflects ambiguity in the training data. For applications needing sense disambiguation, use context or sense-specific models. For visualization, this midway point is informative.
$$
