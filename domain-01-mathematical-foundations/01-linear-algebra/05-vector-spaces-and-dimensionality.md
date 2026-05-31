# Vector spaces and dimensionality

## **DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Linear Algebra**

---

### **1. Why this concept matters for building intelligent systems**

You can have a collection of vectors. But not every collection is a vector space. A vector space is a closed world where you can add any two vectors and stay inside, scale any vector and stay inside. This closure property is what makes learning possible. If your AI's representations could drift outside the space, they would become meaningless. Dimensionality tells you how many independent directions exist in that space—how many "knobs" your system can turn independently. Understanding vector spaces means understanding the container that holds all possible thoughts your AI can have.

---

### **2. Core idea**

**A vector space is a set of vectors closed under addition and scalar multiplication; its dimension is the minimum number of independent directions needed to reach every vector in the space.**

---

### **3. Concrete analogy**

Imagine you are on a vast, flat, infinite field. You can walk in any direction. You can walk 3 steps north and 2 steps east. You can double that and walk 6 steps north and 4 steps east. You can add two journeys: north+west plus east+south equals north+east. You never leave the field. That field is a 2-dimensional vector space.

Now imagine you are tied to a straight railroad track. You can only move forward or backward along the track. No sideways movement. That is a 1-dimensional vector space (a line). Your freedom is limited. The dimension of your space is the number of independent directions you can move.

Now imagine you are in a 3D volume (like a room). You can move left/right, forward/back, up/down. That is a 3-dimensional vector space. Your AI's "thought space" might have 512 dimensions—you cannot visualize it, but the math works exactly the same.

---

### **4. ASCII diagram**

```text
Vector spaces of different dimensions:

1D (line)              2D (plane)             3D (volume)

    ←──●──→                 ↑                     z ↑
       │                  │                      │         0                  │                      │
                       ←───┼───→                 │
                           0                     /
                         │                      /
                         ↓                     /
                                              /
                                             /→ y
                                            x

Every point reachable     Every point reachable    Every point reachable
by one number (t)         by two numbers (x,y)     by three numbers (x,y,z)

Key concept: Basis vectors
    In 2D: e₁ = [1,0], e₂ = [0,1]
    Any vector v = a·e₁ + b·e₂
    Dimension = size of smallest basis = 2
```

---

### **5. Mathematical formulation**

A vector space V over real numbers must satisfy:

- Closure under addition: u,v ∈ V → u+v ∈ V
- Closure under scalar multiplication: c∈ℝ, v∈V → c·v ∈ V
- Contains zero vector: 0 ∈ V
- Additive inverses: for each v, there exists -v

**Basis:** A set of vectors {b₁, b₂, ..., bₙ} is a basis if:

1. They are linearly independent (no vector can be written as a combination of others)
2. They span V (every vector in V can be written as a combination)

**Dimension:** The number of vectors in any basis of V:

$$
\dim(V) = n
$$

**Linear combination:** Any vector v in V can be written uniquely as:

$$
\mathbf{v} = c_1\mathbf{b}_1 + c_2\mathbf{b}_2 + ... + c_n\mathbf{b}_n
$$

(Unicode: v = c₁·b₁ + c₂·b₂ + ... + cₙ·bₙ)

**Standard basis in ℝⁿ:**

$$
\mathbf{e}_1 = \begin{bmatrix}1\\0\\0\\\vdots\\0\end{bmatrix},\;
\mathbf{e}_2 = \begin{bmatrix}0\\1\\0\\\vdots\\0\end{bmatrix},\; ...\;,\;
\mathbf{e}_n = \begin{bmatrix}0\\0\\0\\\vdots\\1\end{bmatrix}
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define three vectors in ℝ³**

$$
\mathbf{v}_1 = \begin{bmatrix}1\\0\\0\end{bmatrix},\quad
\mathbf{v}_2 = \begin{bmatrix}0\\1\\0\end{bmatrix},\quad
\mathbf{v}_3 = \begin{bmatrix}1\\1\\0\end{bmatrix}
$$

#### **Step 2: Are they linearly independent?**

Check if c₁v₁ + c₂v₂ + c₃v₃ = 0 forces all c = 0.

c₁[1,0,0] + c₂[0,1,0] + c₃[1,1,0] = [c₁ + c₃, c₂ + c₃, 0] = [0,0,0]

This gives:
c₁ + c₃ = 0
c₂ + c₃ = 0

We can choose c₃ = 1, then c₁ = -1, c₂ = -1. Non-zero coefficients produce zero vector. Therefore {v₁, v₂, v₃} is linearly dependent.

#### **Step 3: Find a basis**

Remove v₃. Check {v₁, v₂}:

c₁[1,0,0] + c₂[0,1,0] = [c₁, c₂, 0] = [0,0,0] forces c₁ = 0, c₂ = 0. Independent.

Do they span? Any vector [x, y, 0] can be written as x·v₁ + y·v₂. But vectors with non-zero z-component cannot be reached.

#### **Step 4: Determine the subspace dimension**

The set of all vectors of form [x, y, 0] is a 2-dimensional subspace of ℝ³ (a plane through the origin). It is a vector space with dimension 2. It is not all of ℝ³ because you cannot reach any point with z ≠ 0.

#### **Step 5: Compare to full ℝ³**

Full ℝ³ has basis {v₁, v₂, [0,0,1]}. Dimension = 3.

The subspace {[x,y,0]} has basis {v₁, v₂}. Dimension = 2.

---

### **7. How this appears inside neural networks and LLMs**

- **Embedding dimension**: Word embeddings live in a vector space of dimension 300, 768, 4096, or more. This dimension determines how many independent features the model can use to distinguish concepts.

- **Hidden state dimension**: In transformers, the hidden state vector's dimension (d_model) defines the representational capacity of the network at each position.

- **Low-rank approximations**: Many matrices in LLMs are approximately low-rank, meaning most of the information lives in a lower-dimensional subspace. This is exploited for model compression.

- **Manifold hypothesis**: Real-world data (images, text, audio) lies on a low-dimensional manifold embedded in a high-dimensional space. The intrinsic dimension is much smaller than the pixel or token dimension.

- **Bottleneck layers**: Autoencoders force information through a low-dimensional bottleneck (small vector space), compelling the network to learn compressed representations.

- **Attention rank**: The attention matrix can be decomposed into lower-rank components for efficiency (e.g., Linformer, Nyströmformer).

- **Feature spaces**: Each layer of a neural network defines a new vector space where the input representation has been transformed. The dimension might change (e.g., 256 → 512 → 1024).

---

### **8. Brain-like connection (representational capacity)**

The brain's representational space is vast but constrained. Each cortical area processes information in a space defined by its neural population. The dimension of that space is roughly the number of independent activity patterns the population can produce. For a region with N neurons, the theoretical maximum dimension is N (if every neuron is independent), but correlations reduce the effective dimension.

This is why brain damage can be compensated—the representational space has redundancy. Similarly, in AI, you can prune 90% of parameters without losing much accuracy because the effective dimension of the solution space is much smaller than the parameter count. Understanding dimensionality helps you know how much "thinking power" your system truly has versus how much is wasted.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Dimension means the number of numbers in a vector. A vector with 5 numbers lives in 5 dimensions."

_Why it is wrong:_ That is only true if you are using the standard basis (the default coordinate axes). A vector with 5 numbers could represent a point in a 2-dimensional subspace if the numbers are correlated. For example, the vector [x, y, x+y, 2x, 2y] has five numbers but lives in a 2-dimensional space because every component is determined by x and y alone. The dimension is the number of independent degrees of freedom, not the length of the list. This distinction matters for understanding why embeddings work—they pack high-dimensional meaning into lower-dimensional structure.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  A vector space is the universe where your AI thinks.     |
|  Its dimension is how many independent concepts your AI   |
|  can juggle at once. Too few dimensions and the AI        |
|  cannot represent necessary distinctions. Too many and    |
|  you waste compute on meaningless degrees of freedom.     |
|  Mastering vector spaces means mastering the container    |
|  of intelligence itself.                                  |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have vectors in ℝ⁴:
v₁ = [1,0,0,0]
v₂ = [0,1,0,0]
v₃ = [1,1,0,0]
v₄ = [0,0,1,0]

**Question:** What is the dimension of the subspace spanned by {v₁, v₂, v₃, v₄}? Which vectors can be removed without reducing the span?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** The dimension is 3.

- v₁ and v₂ are independent (different directions in first two coordinates)
- v₃ = v₁ + v₂, so v₃ adds nothing new (linearly dependent)
- v₄ adds a third independent direction (third coordinate)

Basis: {v₁, v₂, v₄}. v₃ can be removed without reducing the span. The subspace consists of all vectors of the form [a, b, 0, c] (third coordinate always zero).
