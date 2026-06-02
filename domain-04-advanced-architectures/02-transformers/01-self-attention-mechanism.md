# Self-attention mechanism

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Transformers (The LLM Revolution)**

---

### **1. Why this concept matters**

Original attention let a decoder look back at an encoder. But what about within a single sequence? In the sentence "The animal didn't cross the street because it was too tired," what does "it" refer to? To know, the model needs to compare "it" to every preceding word and find the most relevant—"animal," not "street." Self-attention does exactly this: each position attends to all positions in the same sequence, including itself. This is the core innovation of the transformer. It captures dependencies regardless of distance, in parallel, without the sequential bottleneck of RNNs. Without self-attention, there is no GPT, no BERT, no modern LLM.

---

### **2. Core idea**

**Self-attention computes a weighted representation for each position in a sequence by comparing it to all positions (including itself), using queries, keys, and values all derived from the same input, allowing the model to capture contextual relationships regardless of distance.**

---

### **3. Concrete analogy**

Imagine a team meeting with 10 people. Each person needs to understand the discussion in context. Self-attention is like each person listening to everyone else, then summarizing: "What matters to me from what everyone said?"

For the word "it" in our sentence, self-attention asks:

- Query: "What am I looking for?" (from "it")
- Keys: "What do each of you offer?" (from each word)
- Values: "What information do you carry?"

"Animal" has a high key similarity to "it"'s query (pronoun likely refers to animate noun). "Street" has low similarity. So "it" pays high attention to "animal," modest to "crossed," low to "street." Its new representation blends information from relevant words. This happens for every word simultaneously, in parallel, not one after another.

---

### **4. ASCII diagram**

```
Self-attention for the sentence: "The animal crossed the street"

Step 1: Create Q, K, V from each word's embedding:

    "The"    → q₁, k₁, v₁
    "animal" → q₂, k₂, v₂
    "crossed"→ q₃, k₃, v₃
    "the"    → q₄, k₄, v₄
    "street" → q₅, k₅, v₅

Step 2: Compute attention scores (dot product between q and all k's):

For "animal" (position 2):
    score₂₁ = q₂·k₁  (The)
    score₂₂ = q₂·k₂  (animal)  ← highest (self)
    score₂₃ = q₂·k₃  (crossed)
    score₂₄ = q₂·k₄  (the)
    score₂₅ = q₂·k₅  (street)

Step 3: Softmax to get weights:

    α₂₁, α₂₂, α₂₃, α₂₄, α₂₅

Step 4: New representation for "animal":

    z₂ = α₂₁×v₁ + α₂₂×v₂ + α₂₃×v₃ + α₂₄×v₄ + α₂₅×v₅

Now "animal" knows about its context!


Matrix form (all positions at once):

    Input X (n × d_model)
         │
         ├───→ W_Q ──→ Q (n × d_k)
         ├───→ W_K ──→ K (n × d_k)
         └───→ W_V ──→ V (n × d_v)
                    │
                    ▼
            Attention = softmax(Q·Kᵀ / √d_k) · V
                    │
                    ▼
              Output Z (n × d_v)
```

---

### **5. Mathematical formulation**

**Input:** Sequence X of length n, each token as vector (d_model dimensions)

**Linear projections to lower dimensions:**

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}^V
$$

Where W^Q, W^K are d_model × d_k, W^V is d_model × d_v. Typically d_k = d_v = d_model / num_heads.

**Attention scores (logits):**

$$
\text{Scores} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}
$$

(Unicode: Scores = Q·Kᵀ / √d_k)

Dividing by √d_k prevents dot products from growing too large (which would push softmax into saturated region).

**Attention weights (probabilities):**

$$
\mathbf{A} = \text{softmax}(\text{Scores})
$$

(Each row sums to 1)

**Output (weighted sum of values):**

$$
\mathbf{Z} = \mathbf{A} \mathbf{V}
$$

**Complete formula:**

$$
\text{Attention}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{X}\mathbf{W}^Q (\mathbf{X}\mathbf{W}^K)^T}{\sqrt{d_k}}\right) \mathbf{X}\mathbf{W}^V
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: A tiny example**

Sequence length n=3, d_model=4. Simplified: skip W projections (use X directly as Q, K, V).

X = [[1, 0, 0, 0],   # token "cat"
     [0, 1, 0, 0],   # token "sat"
     [0, 0, 1, 0]] # token "down"

d_k = 4, so √d_k = 2

#### **Step 2: Compute Q·Kᵀ**

Scores = X·Xᵀ / 2

X·Xᵀ (dot products):

- Row1·Row1 = 1, Row1·Row2 = 0, Row1·Row3 = 0
- Row2·Row1 = 0, Row2·Row2 = 1, Row2·Row3 = 0
- Row3·Row1 = 0, Row3·Row2 = 0, Row3·Row3 = 1

Scores = [[0.5, 0, 0],
          [0, 0.5, 0],
          [0, 0, 0.5]]

#### **Step 3: Softmax (each row independently)**

Softmax of [0.5, 0, 0] = [exp(0.5), exp(0), exp(0)] = [1.648, 1, 1]; sum = 3.648
Weights = [0.452, 0.274, 0.274]

Similarly for each row:
A = [[0.452, 0.274, 0.274],
     [0.274, 0.452, 0.274],
     [0.274, 0.274, 0.452]]

#### **Step 4: Compute output Z = A × X**

Z₁ = 0.452×[1,0,0,0] + 0.274×[0,1,0,0] + 0.274×[0,0,1,0] = [0.452, 0.274, 0.274, 0]
Z₂ = [0.274, 0.452, 0.274, 0]
Z₃ = [0.274, 0.274, 0.452, 0]

#### **Step 5: Interpret**

Each output token now contains information from all input tokens, weighted by similarity. "cat" (position 1) kept most of itself (0.452) but also gained 0.274 from "sat" and 0.274 from "down." This is context! The model now knows "cat" is related to "sat" and "down" in this sentence.

---

### **7. How this appears inside neural networks and LLMs**

- **Transformer blocks:** Every transformer block has self-attention as its core component. Without it, the transformer is just feed-forward layers.

- **GPT (decoder-only):** Uses masked self-attention (causal mask) so positions cannot attend to future tokens. Enables autoregressive generation.

- **BERT (encoder-only):** Uses bidirectional self-attention (no mask). Each token sees full context left and right.

- **Long-range dependencies:** Unlike RNNs where path length = O(n), self-attention connects any two positions in O(1) operations.

- **Parallel processing:** All positions compute attention simultaneously (no sequential bottleneck). This is why transformers train fast on GPUs.

- **Inductive bias:** Self-attention has no built-in notion of position. The model treats "cat sat" same as "sat cat" unless positional encodings are added.

- **Attention heads:** Multiple self-attention mechanisms run in parallel (multi-head), each learning different relationship types (syntax, coreference, etc.).

---

### **8. Brain-like connection (relational memory)**

The brain's working memory operates like self-attention. The prefrontal cortex maintains representations of current stimuli (keys/values) while a "retrieval cue" (query) determines what is recalled. When you hear "She poured water from the pitcher into the glass until it was full," your brain retrieves "glass" (not "pitcher") for "it" based on semantic relationships. This is relational memory—binding representations based on similarity and relevance. Self-attention mathematically formalizes this cognitive process: weighted retrieval from a memory store based on query-key similarity. The transformer is, in essence, a differentiable implementation of content-addressable memory.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Self-attention is just computing similarity between all pairs of words. It's a simple lookup table."

_Why it is wrong:_ Self-attention does compute pairwise similarities (the Q·Kᵀ scores), but that is only the first step. The learned transformations W^Q, W^K, W^V are crucial. The model learns to project the same input X into three different spaces: query (what I'm looking for), key (what I offer), and value (what I communicate). These projections can emphasize different features. For example, in "bank," the query for "river" projects into a space where "bank" (river bank) has high similarity, while the query for "money" projects into a different space where "bank" (financial) is similar. Same word, different projections, different retrieved values. This is not a simple lookup.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Self-attention is the engine of modern AI. It replaces   |
|  the sequential chains of RNNs with a web of connections  |
|  between every pair of words. This parallel structure     |
|  unlocked training on massive datasets and led directly   |
|  to GPT, BERT, Llama, and every LLM you use today.        |
|  Understand self-attention and you understand the heart   |
|  of the transformer—the architecture that taught machines |
|  to pay attention to everything at once.                  |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

In self-attention, after computing scores = Q·Kᵀ / √d_k, we apply softmax row-wise. Why do we divide by √d_k before softmax? What would happen if d_k were 512 and we did not divide?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Dividing by √d_k prevents the dot products from becoming too large. For high-dimensional vectors (e.g., d_k=512), the expected magnitude of a dot product grows with √d_k. Without scaling, scores would be large (±50-100), pushing the softmax into regions where gradients are extremely small (saturation). The model would stop learning because gradients vanish. The √d_k scaling keeps the variance of the scores roughly 1, maintaining healthy gradient flow. This is critical for stable training of deep transformers.
