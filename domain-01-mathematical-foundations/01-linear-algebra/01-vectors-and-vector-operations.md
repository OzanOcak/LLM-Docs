<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

# Vectors and vector operations

## ## **DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Linear Algebra**

---

### **1. Why this concept matters for building intelligent systems**

Every intelligent system must represent information. A single number is too small to capture meaning. A rigid database table cannot transform and learn. What you need is a flexible container that holds rich meaning while remaining mathematically manipulable. That container is the vector. When you build a system that thinks, every input (sight, sound, text), every memory, and every decision becomes a vector. Without vectors, your AI cannot compare, transform, or learn anything.

---

### **2. Core idea**

**A vector is an ordered list of numbers that exists in space, allowing you to measure similarity, direction, and magnitude between any two representations.**

---

### **3. Concrete analogy**

Imagine you are a detective profiling three suspects using three features: age, height in centimeters, and estimated annual income in thousands.

- Suspect A = [32, 175, 50]
- Suspect B = [45, 160, 120]
- Suspect C = [34, 178, 55]

Each suspect becomes a point in 3D space. The distance between A and C is small (they are similar). The distance between A and B is large (they are different). If a witness describes a new suspect as "around 33 years old, average height, moderate income," you can find who matches best by comparing vectors. This is exactly how an AI finds similar images, words, or sounds.

---

### **4. ASCII diagram**

```
Vector in 2D space (two features: age and height)

Age ↑
 50 |
    |
 40 |       • B (45, 160)
    |
 30 |   • A (32, 175)  • C (34, 178)
    |
 20 |
    +----------------→ Height
       150   170   190

• Each point = one vector
• Arrow from origin (0,0) to point = the vector
• Distance between points = difference between vectors
• Angle between arrows = similarity of direction
```

---

### **5. Mathematical formulation**

A vector v with n dimensions is written as:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \\ \vdots \\ v_n \end{bmatrix}
$$

Where each vᵢ is a number (scalar) representing one feature.

**Core operations:**

**Vector addition** (combine two things):

$$
\mathbf{a} + \mathbf{b} = [a_1+b_1, a_2+b_2, ..., a_n+b_n]
$$

(Unicode: a + b = [a₁+b₁, a₂+b₂, ..., aₙ+bₙ])

**Scalar multiplication** (amplify or shrink):

$$
c \cdot \mathbf{v} = [c \cdot v_1, c \cdot v_2, ..., c \cdot v_n]
$$

(Unicode: c·v = [c·v₁, c·v₂, ..., c·vₙ])

**Dot product** (measure similarity):

$$
\mathbf{a} \cdot \mathbf{b} = a_1b_1 + a_2b_2 + ... + a_nb_n
$$

(Unicode: a·b = a₁b₁ + a₂b₂ + ... + aₙbₙ)

**Magnitude** (length, or "how much total stuff"):

$$
\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}
$$

(Unicode: ‖v‖ = √(v₁² + v₂² + ... + vₙ²))

**Cosine similarity** (similarity normalized by length):

$$
\text{similarity} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

(Unicode: similarity = (a·b) / (‖a‖ ‖b‖))

Result ranges from -1 (opposite) to 0 (unrelated) to +1 (identical direction).

---

### **6. Worked example (step-by-step)**

Three word-vectors in a tiny 2D "meaning space":

| Word   | Power dimension | Trust dimension |
| ------ | --------------- | --------------- |
| King   | 0.9             | 0.4             |
| Queen  | 0.8             | 0.5             |
| Tyrant | 0.95            | 0.05            |

#### **Step 1: Write as vectors**

- King = [0.9, 0.4]
- Queen = [0.8, 0.5]
- Tyrant = [0.95, 0.05]

#### **Step 2: Compute similarity between King and Queen**

Dot product:

$$
0.9 \times 0.8 + 0.4 \times 0.5 = 0.72 + 0.20 = 0.92
$$

Magnitudes:

$$
\|\text{King}\| = \sqrt{0.9^2 + 0.4^2} = \sqrt{0.81 + 0.16} = \sqrt{0.97} = 0.985
$$

$$
\|\text{Queen}\| = \sqrt{0.8^2 + 0.5^2} = \sqrt{0.64 + 0.25} = \sqrt{0.89} = 0.943
$$

Cosine similarity:

$$
\frac{0.92}{0.985 \times 0.943} = \frac{0.92}{0.929} = 0.99
$$

(Unicode: 0.92 / (0.985 × 0.943) = 0.92 / 0.929 = 0.99)

#### **Step 3: Compare King to Tyrant**

Dot product:

$$
0.9 \times 0.95 + 0.4 \times 0.05 = 0.855 + 0.02 = 0.875
$$

Magnitude of Tyrant:

$$
\|\text{Tyrant}\| = \sqrt{0.95^2 + 0.05^2} = \sqrt{0.9025 + 0.0025} = \sqrt{0.905} = 0.951
$$

Cosine similarity:

$$
\frac{0.875}{0.985 \times 0.951} = \frac{0.875}{0.936} = 0.935
$$

**Interpretation:** King and Queen are nearly identical in meaning direction (0.99). Tyrant shares "power" but lacks "trust," making it less similar to King (0.935).

---

### **7. How this appears inside neural networks and LLMs**

- **Word embeddings**: Every word in an LLM's vocabulary is stored as a vector (typically 300 to 4096 dimensions). "King" and "Queen" have nearly parallel vectors.

- **Attention mechanism**: When an LLM reads a sentence, it computes dot products between the vector of the current word and vectors of all previous words. Higher dot product = more attention paid.

- **Hidden states**: At each processing step, the LLM maintains a hidden state vector summarizing "what it thinks so far."

- **Output logits**: The final vector is compared against all word vectors in the vocabulary using dot product. The word whose vector points most similarly becomes the next word generated.

- **Similarity search**: When you ask a question, the system converts your question into a vector and finds the most similar document vectors stored in memory.

- **Classification**: A neural network's last layer often computes which class vector (cat, dog, bird) is closest to the input vector.

---

### **8. Brain-like connection (memory)**

The hippocampus stores memories as distributed patterns across neurons—functionally equivalent to vectors. When you encounter a familiar smell (cinnamon), the brain converts it into a neural activity vector and performs something like a dot product search across stored memory vectors. The memory with the highest similarity (grandma's kitchen) surfaces into consciousness.

This is why similarity matters more than exact matching. You never experience the exact same sensory input twice, but your brain finds the closest match. Vector operations are the computational substrate of associative memory.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "A vector is just an array or list. Why make it special?"

_Why it is wrong:_ An array in programming is a passive container. A vector comes with geometric operations that let you ask meaningful questions:

- How similar are these two things? (dot product)
- How different are they? (Euclidean distance)
- If I combine them, what do I get? (addition)
- What is the direction of change? (subtraction)

Without these operations, your "list" cannot learn, compare, or transform. With them, you have the foundation of every neural network. Calling a vector an "array" is like calling a house "a pile of bricks." The structure and operations are the entire point.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  When you build a system that thinks, everything becomes  |
|  a vector. Every camera pixel, every audio sample, every  |
|  word in every sentence you feed it. The difference       |
|  between a dumb program and an intelligent one is whether |
|  you can compare, combine, and transform these vectors    |
|  fluidly. Master vectors and you master the native        |
|  language of machine intelligence.                        |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have three vectors in a 2D emotion space:

- Happy = [0.9, 0.8] (pleasure, arousal)
- Calm = [0.7, 0.2]
- Anxious = [0.3, 0.9]

**Question:** Which pair has the higher cosine similarity: (Happy, Calm) or (Happy, Anxious)? Show your work.

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Happy · Calm = (0.9×0.7)+(0.8×0.2) = 0.63+0.16 = 0.79

Happy · Anxious = (0.9×0.3)+(0.8×0.9) = 0.27+0.72 = 0.99

‖Happy‖ = √(0.81+0.64) = √1.45 = 1.20

‖Calm‖ = √(0.49+0.04) = √0.53 = 0.73

‖Anxious‖ = √(0.09+0.81) = √0.90 = 0.95

Cosine(Happy, Calm) = 0.79 / (1.20×0.73) = 0.79 / 0.876 = 0.90

Cosine(Happy, Anxious) = 0.99 / (1.20×0.95) = 0.99 / 1.14 = 0.87

**Happy and Calm are more similar** (0.90 > 0.87). Despite Anxious having higher raw dot product, the normalization reveals that Calm's direction aligns better with Happy when accounting for vector lengths.
