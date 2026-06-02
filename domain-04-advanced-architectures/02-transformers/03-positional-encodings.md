# Positional encodings

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Transformers (The LLM Revolution)**

---

### **1. Why this concept matters**

Self-attention has a fatal flaw: it is permutation invariant. The sentence "cat sat on mat" produces the same attention patterns as "mat on sat cat" because the mechanism treats all positions as a set, not a sequence. Word order is everything in language. Without knowing position, "dog bites man" and "man bites dog" are identical to a transformer. Positional encodings solve this by injecting information about each token's position into the input embeddings. This small addition—a carefully crafted sine/cosine pattern—is what gives transformers the ability to understand word order and reason about relative positions.

---

### **2. Core idea**

**Positional encodings add a unique vector to each token's embedding based on its position in the sequence, allowing the transformer to distinguish word order using either fixed sinusoidal functions or learnable position embeddings.**

---

### **3. Concrete analogy**

Imagine a row of numbered chairs. Each person (token) sits on a chair. Their identity (embedding) matters, but so does where they sit. "Alice before Bob" is different from "Bob before Alice."

Now imagine you give each person a colored badge. The badge encodes their seat number: red for seat 1, blue for seat 2, green for seat 3. Even if people swap chairs, the badge tells you their original position. This is like positional encoding—a signal added to each token that indicates its place in the sequence.

But there is a problem: if seats are numbered 1, 2, 3, a model might learn that seat 1000 is "large" and treat it differently from seat 1. Sinusoidal encodings solve this by using waves: position 1000 is a specific combination of sine and cosine values that is unique but not arbitrarily large. The model can also learn position embeddings directly—treating position like a vocabulary of 512 "position tokens."

---

### **4. ASCII diagram**

```
Positional encoding added to word embedding:

    Word Embedding (E) + Positional Encoding (P) = Input to Transformer

    "cat" at position 1:
    E_cat = [0.8, -0.2, 0.5,  0.1, ...]
    P_pos1= [0.1,  0.8, -0.3,  0.4, ...]  (from sine/cosine)
    X₁    = [0.9,  0.6,  0.2,  0.5, ...]  ← final input

    "sat" at position 2:
    E_sat = [0.3,  0.7, -0.1,  0.2, ...]
    P_pos2= [0.4,  0.2,  0.9, -0.1, ...]  (different pattern)
    X₂    = [0.7,  0.9,  0.8,  0.1, ...]


Sinusoidal positional encoding matrix (positions vs dimensions):

    Position → 0       1       2       3       4       5
    Dim ↓
    0        sin(0)  sin(1)  sin(2)  sin(3)  sin(4)  sin(5)
    1        cos(0)  cos(1)  cos(2)  cos(3)  cos(4)  cos(5)
    2        sin(0/2) sin(1/2) sin(2/2) sin(3/2) ...
    3        cos(0/2) cos(1/2) cos(2/2) cos(3/2) ...
    4        sin(0/4) sin(1/4) sin(2/4) ...
    5        cos(0/4) cos(1/4) cos(2/4) ...
    ...

    Each row is a wave with different frequency.
    Each position gets a unique fingerprint across dimensions.
```

---

### **5. Mathematical formulation**

**Sinusoidal positional encoding (Vaswani et al., 2017):**

For position pos and dimension i (0-indexed):

If i is even:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

If i is odd:

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

(Unicode: PE(pos,2i) = sin(pos / 10000^(2i/d_model)), PE(pos,2i+1) = cos(pos / 10000^(2i/d_model)))

**Learnable positional embedding:**

Treat position as a vocabulary of size max_len (e.g., 512). Each position gets a learnable vector of dimension d_model.

$$
\mathbf{P} \in \mathbb{R}^{\text{max\_len} \times d_{\text{model}}}
$$

During training, these vectors are updated via backpropagation.

**Final input to transformer:**

$$
\mathbf{X}_{\text{input}} = \mathbf{E} + \mathbf{P}
$$

(Elementwise addition, not concatenation)

**Relative positional encodings (used in some modern LLMs):**

Instead of absolute positions, encode relative distances: how many steps between token i and token j. Used in Transformer-XL, some T5 variants, and Llama (RoPE).

**Rotary Position Embedding (RoPE, used in Llama, GPT-NeoX):**

Rotates queries and keys by angles proportional to position. Preserves relative distance information naturally.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Small d_model=4, max_len=6**

Compute PE for pos=2 (third token), d_model=4 → i=0,1 (2i=0, 2i+1=1)

d_model=4, so i=0 and i=1 cover dimensions 0,1,2,3

#### **Step 2: Frequency for i=0**

divisor = 10000^(0/4) = 10000^0 = 1

dim 0 (even): sin(2 / 1) = sin(2) ≈ 0.909
dim 1 (odd): cos(2 / 1) = cos(2) ≈ -0.416

#### **Step 3: Frequency for i=1**

divisor = 10000^(2/4) = 10000^0.5 = 100

dim 2 (even): sin(2 / 100) = sin(0.02) ≈ 0.020
dim 3 (odd): cos(2 / 100) = cos(0.02) ≈ 0.9998

#### **Step 4: PE vector for position 2**

PE(2) = [0.909, -0.416, 0.020, 0.9998]

#### **Step 5: PE for position 5 (later token)**

For pos=5:

dim 0: sin(5) ≈ -0.959
dim 1: cos(5) ≈ 0.284
dim 2: sin(5/100)=sin(0.05)=0.050
dim 3: cos(5/100)=cos(0.05)=0.999

PE(5) = [-0.959, 0.284, 0.050, 0.999]

#### **Step 6: Interpret**

Each position has a unique fingerprint. Note how low-frequency dimensions (i=1, dims 2-3) change slowly across positions—they encode coarse position. High-frequency dimensions (i=0, dims 0-1) change rapidly—they encode fine-grained position differences. This multi-scale representation allows the model to distinguish both nearby and distant positions.

#### **Step 7: Why not just use position 1,2,3 as numbers?**

If we used raw numbers [1,2,3,...], large positions (e.g., 1000) would dominate the embedding magnitude. The model would struggle because position 1000 is "larger" than position 1, which has no linguistic meaning. Sinusoidal encodings keep values bounded between -1 and 1, and the pattern allows the model to extrapolate to longer sequences than seen during training.

---

### **7. How this appears inside neural networks and LLMs**

- **Original transformer (Vaswani et al.):** Used sinusoidal encodings. Required for the model to distinguish word order since self-attention is permutation-invariant.

- **BERT:** Uses learnable positional embeddings (absolute positions 0 to 511). Learns position-specific vectors from data rather than fixed sinusoids.

- **GPT-2/GPT-3:** Also use learnable absolute positional embeddings. Works well because pretraining data is massive.

- **Relative positional encodings (Transformer-XL, T5):** Encode distance between tokens, not absolute position. Better for long sequences because relative distances generalize beyond training length.

- **Rotary Position Embedding (RoPE) - Llama, GPT-NeoX, PaLM:** Rotates queries and keys in complex space. Preserves relative position information naturally without adding vectors to embeddings. Currently state-of-the-art for LLMs.

- **ALiBi (Attention with Linear Biases):** Adds a bias to attention scores based on distance (not added to embeddings). Simpler and extrapolates well to longer sequences.

- **No positional encoding?** Some recent work (NoPE) shows transformers can learn position from causal masking alone, but standard practice still includes positional encodings.

---

### **8. Brain-like connection (place cells and grid cells)**

The brain encodes position using two complementary systems. Place cells in the hippocampus fire at specific absolute locations—like absolute positional embeddings. Grid cells in entorhinal cortex fire in repeating hexagonal patterns across space—like sinusoidal positional encodings (periodic, multi-scale). The brain combines both to navigate. Similarly, transformers use both absolute (position embeddings) and relative (attention bias, RoPE) signals to understand word order. The multi-frequency sinusoidal encoding mirrors the brain's use of different spatial scales for coarse and fine localization. Evolution discovered that position encoding requires multiple scales; transformers independently arrived at the same solution.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Positional encodings are just a hack. The transformer should learn order from the data without them."

_Why it is wrong:_ Without positional encodings, self-attention treats sequences as sets. "A B C" and "C B A" produce identical attention outputs because the dot products Q·Kᵀ are the same (just permuted). The transformer has no way to know that order matters. You cannot learn order from data if the architecture has no mechanism to represent it. Positional encodings are not a hack—they are a fundamental requirement for sequence modeling in permutation-invariant architectures. Removing them collapses the transformer to a bag-of-words model, regardless of training data size.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Word order is half the meaning of language. "Dog bites    |
|  man" is news. "Man bites dog" is a different story.       |
|  Self-attention, left to its own devices, cannot tell     |
|  the difference. Positional encodings restore the         |
|  sense of order—the before and after, the cause and       |
|  effect, the subject and object. Without them, a          |
|  transformer is just counting words. With them, it        |
|  understands sentences.                                   |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

For a sinusoidal positional encoding with d_model=2, what is the PE vector for position pos=0? For pos=π? (Assume division by 10000^(0/2)=1)

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

For d_model=2, only i=0 (dimensions 0 and 1):

- dim 0 (even): sin(pos / 1) = sin(pos)
- dim 1 (odd): cos(pos / 1) = cos(pos)

For pos=0: PE = [sin(0), cos(0)] = [0, 1]

For pos=π: PE = [sin(π), cos(π)] = [0, -1]

The encoding produces a unit circle in 2D space. As position increases, the encoding traces the unit circle, returning to the same point every 2π steps. This periodic property means sinusoidal encodings are not unique—positions 0 and 2π have the same encoding! In practice, the division by 10000^(2i/d_model) makes frequencies very low for higher dimensions, so periodicity is extremely long (e.g., 10000 steps). For d_model=2 with no frequency scaling, the ambiguity is problematic, which is why real transformers use higher dimensions with different frequencies.
