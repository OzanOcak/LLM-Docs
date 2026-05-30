# Linear transformations

**DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Linear Algebra**

---

**1. Why this concept matters for building intelligent systems**

A vector is a point in space. A matrix is a static grid of numbers. But intelligence is neither static nor a single point—it is a process of transformation. A linear transformation is the rule that takes an input vector and produces an output vector. Every layer of every neural network is a linear transformation (followed by a nonlinear activation). Every time information flows from perception to thought to action, a transformation just occurred. Understanding linear transformations means understanding how intelligence reshapes information.

---

**2. Core idea**

**A linear transformation is a function that maps vectors to other vectors while preserving two operations: vector addition and scalar multiplication.**

---

**3. Concrete analogy**

Imagine a rubber sheet stretched across a table. Draw a grid on it. Now push, pull, rotate, or shear the sheet. Any straight line remains straight. The origin (0,0) stays fixed. Grid lines that were parallel remain parallel. This is a linear transformation.

Now drop a coffee mug onto the sheet. The mug's coordinates change, but the transformation rule applies to every point equally. If you know where four key points go (say, (1,0) and (0,1)), you know where every point goes.

Your brain does this constantly. When you rotate your head, the image of the world on your retina undergoes a linear transformation (plus some projection). The object is the same. The coordinates changed. The transformation bridges them.

---

**4. ASCII diagram**

```
Linear transformation as grid deformation:

    Before (identity)              After (shear transform)

    y ↑                           y ↑
      │   ┌───┬───┐                 │   ◇───◇───◇
      │   │   │   │                 │  ╱    │    │
      │   ├───┼───┤                 │ ╱     │    │
      │   │   │   │                 │╱      │    │
      │   ├───┼───┤                 ◇───◇───◇
      │   │   │   │                ╱
      └───┴───┴───→ x             └──────→ x

    Properties preserved:
    • Straight lines remain straight
    • Origin stays fixed (0,0 → 0,0)
    • Parallel lines stay parallel
    • Grid spacing can change uniformly

    What defines the transformation:
    Where does (1,0) go? → First column of matrix
    Where does (0,1) go? → Second column of matrix
```

---

**5. Mathematical formulation**

A linear transformation T: ℝⁿ → ℝᵐ satisfies two rules for all vectors u, v and scalars c:

**Additivity:**

$$
T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})
$$

(Unicode: T(u+v) = T(u) + T(v))

**Homogeneity:**

$$
T(c\mathbf{v}) = c T(\mathbf{v})
$$

(Unicode: T(c·v) = c·T(v))

**Matrix representation:** Every linear transformation corresponds to a unique matrix A:

$$
T(\mathbf{v}) = \mathbf{A}\mathbf{v}
$$

Where:

- A has dimensions m × n
- Column j of A = T(eⱼ), where eⱼ is the j-th standard basis vector (1 at position j, 0 elsewhere)

**Geometric types of linear transformations in 2D:**

- **Rotation by angle θ:**

  $$
  \mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
  $$

- **Scaling (by factors sₓ, sᵧ):**

  $$
  \mathbf{S} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}
  $$

- **Shear (horizontal):**

  $$
  \mathbf{H} = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}
  $$

- **Reflection across x-axis:**
  $$
  \mathbf{F} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
  $$

---

**6. Worked example (step-by-step)**

Define a linear transformation that rotates vectors by 90° counterclockwise and then scales x-coordinates by 2.

**Step 1: Find where basis vectors go**

Standard basis vectors in ℝ²:
e₁ = [1, 0]ᵀ
e₂ = [0, 1]ᵀ

**Step 2: Apply rotation by 90°**

90° rotation matrix:

$$
\mathbf{R} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

T_rotate(e₁) = [0, 1]ᵀ
T_rotate(e₂) = [-1, 0]ᵀ

**Step 3: Apply scaling (x2 on x-coordinate only)**

Scale matrix:

$$
\mathbf{S} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}
$$

Final transformation: T = S × R (apply R first, then S)

First column = T(e₁) = S × [0, 1]ᵀ = [0, 1]ᵀ
Second column = T(e₂) = S × [-1, 0]ᵀ = [-2, 0]ᵀ

**Step 4: Write the matrix**

$$
\mathbf{A} = \begin{bmatrix} 0 & -2 \\ 1 & 0 \end{bmatrix}
$$

**Step 5: Test on a sample vector**

Take v = [1, 1]ᵀ (point at (1,1))

Without transformation: T(v) = A × [1, 1]ᵀ = [0×1 + -2×1, 1×1 + 0×1]ᵀ = [-2, 1]ᵀ

Interpretation: The point (1,1) rotated 90° becomes (-1,1), then x-coordinate doubled becomes (-2,1). Works as expected.

---

**7. How this appears inside neural networks and LLMs**

- **Fully connected layer:** y = Wx + b is an affine transformation (linear + translation). The linear part Wx is a linear transformation.

- **Embedding layers:** Mapping a one-hot vector (word index) to a dense vector is a linear transformation. The embedding matrix times the one-hot selects a row.

- **Attention's Q, K, V projections:** Each input vector is multiplied by a matrix (W_Q, W_K, W_V). These are linear transformations into query, key, and value spaces.

- **Convolutional layers:** A convolution is a linear transformation with a special structure (Toeplitz matrix) that enforces locality and weight sharing.

- **Principal Component Analysis (PCA):** Projecting data onto principal components is a linear transformation that rotates and discards dimensions.

- **Residual connections:** Adding the input to the output (y = F(x) + x) combines a nonlinear transformation with the identity transformation, which is linear.

- **Layer Normalization:** The normalization step subtracts mean and divides by standard deviation. Both are linear operations (though the standard deviation depends on the input in a nonlinear way).

---

**8. Brain-like connection (sensory processing)**

The primary visual cortex (V1) performs something remarkably close to a linear transformation on retinal images. Each neuron responds to a specific orientation of an edge (like a Gabor filter). The collection of these neurons forms a basis—the brain is projecting the incoming image onto orientation-sensitive "feature vectors." This transformation is approximate and has nonlinearities, but the core idea is linear projection: the brain re-represents the world in a format useful for higher cognition.

Similarly, the cochlea (inner ear) performs a linear transformation called the Fourier transform (approximately) on sound waves, converting time-domain pressure signals into frequency-domain representations. Your auditory system then processes these transformed vectors to recognize speech, music, and danger.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Linear transformations are too simple. Neural networks use nonlinear activation functions, so linearity doesn't matter."

_Why it is wrong:_ Neural networks are built as alternating linear and nonlinear layers. The linear transformations do the heavy lifting of mixing and reweighting information. The nonlinearities just add the ability to make decisions (thresholds) and represent non-linear manifolds. Without the linear transformations, you would just have independent nonlinearities on each input dimension, which cannot mix features. Without the nonlinearities, the entire network would collapse into a single linear transformation (since composition of linear functions is linear). Both are essential. The linear part gives expressive power; the nonlinear part gives representational flexibility.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Every intelligent system transforms information.         |
|  A linear transformation is the simplest nontrivial       |
|  transformation—easy to compute, easy to compose,         |
|  easy to learn. Master linear transformations and you     |
|  understand the atomic operation of every neural network  |
|  layer, every embedding, and every projection. They are   |
|  the gears inside the clockwork of machine intelligence.  |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

You have a linear transformation T: ℝ² → ℝ² defined by:

T([1, 0]ᵀ) = [2, 1]ᵀ
T([0, 1]ᵀ) = [-1, 3]ᵀ

**Question:** What is the matrix A representing T? What is T([3, 2]ᵀ)?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Matrix A = [[2, -1], [1, 3]] (first column = T(e₁), second column = T(e₂))

T([3, 2]ᵀ) = A × [3, 2]ᵀ = [2×3 + -1×2, 1×3 + 3×2]ᵀ = [6-2, 3+6]ᵀ = [4, 9]ᵀ
