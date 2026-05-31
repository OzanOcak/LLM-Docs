# Multivariable calculus basics

## **DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Calculus**

---

### **1. Why this concept matters for building intelligent systems**

Single-variable calculus handles one knob. A neural network has millions of knobs (weights). You cannot optimize each one in isolation because they interact. Changing weight A changes how weight B affects the output. Multivariable calculus is the mathematics of many interacting knobs. It gives you the tools to understand surfaces in high-dimensional spaces, measure curvature, find valleys, and navigate toward better solutions. Every modern AI system is built on the foundation of multivariable calculus, from gradient descent to attention mechanisms to variational autoencoders.

---

### **2. Core idea**

**Multivariable calculus extends differentiation and integration to functions with more than one input, using partial derivatives, gradients, directional derivatives, and second-order information to describe how the function behaves in multiple directions simultaneously.**

---

### **3. Concrete analogy**

Imagine you are a farmer on a hillside. Your crop yield depends on two factors: how much fertilizer you add (x) and how much water you provide (y). This is a function f(x,y).

Single-variable calculus would ask: "If you change only fertilizer (hold water constant), how does yield change?" That is ∂f/∂x.

But you want to know: "If you change both fertilizer and water simultaneously along a certain ratio (e.g., add 2 units of fertilizer for every 1 unit of water), how fast does yield change?" That is the directional derivative.

You also want to know: "Which direction on the hillside increases yield fastest?" That is the gradient.

Finally: "Am I standing in a valley (minimum), on a peak (maximum), or on a saddle (like a mountain pass)?" That requires second derivatives organized into a matrix called the Hessian.

Multivariable calculus gives you the full topographical map of your optimization landscape.

---

### **4. ASCII diagram**

```text
Multivariable function landscape (2 inputs, 1 output):

    f(x,y) [height]
         ↑
        /|\
       / | \
      /  |  \
     /   |   \
    /    |    \
   /     |     \
  /      |      \
 /       |       \
/        |        \
└────────┼────────┼──→ y (water)
         │        │
         └────────┴──→ x (fertilizer)

Key concepts on this surface:

    1. Partial derivative ∂f/∂x = slope in x-direction
    2. Partial derivative ∂f/∂y = slope in y-direction
    3. Gradient ∇f = [∂f/∂x, ∂f/∂y] points steepest uphill
    4. Contour lines = constant height slices
    5. Directional derivative = slope in any chosen direction
    6. Hessian matrix = curvature in all directions

Saddle point (neither min nor max):

         f(x,y) = x² - y²

         /\          /\
        /  \  /\    /  \
       /    \/  \  /    \
      /     /\   \/      \
     /     /  \  /\       \
    /_____/    \/  \_______\

    (curves up in x-direction, down in y-direction)
```

---

### **5. Mathematical formulation**

**Gradient (vector of first partial derivatives):**

$$
\nabla f = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

(Unicode: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ)

**Directional derivative (rate of change in direction of unit vector u):**

$$
D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}
$$

(Unicode: D_u f = ∇f · u)

**Hessian matrix (second partial derivatives):**

$$
\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

(Unicode: H_ij = ∂²f/∂x_i∂x_j)

**Taylor expansion in multiple dimensions (up to second order):**

$$
f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot \Delta\mathbf{x} + \frac{1}{2} \Delta\mathbf{x}^T \mathbf{H}(\mathbf{x}) \Delta\mathbf{x}
$$

(Unicode: f(x+Δx) ≈ f(x) + ∇f(x)·Δx + ½ Δxᵀ H(x) Δx)

**Chain rule for multivariable functions:**

If z = f(x₁, x₂, ..., xₙ) and each x_i = g_i(t), then:

$$
\frac{dz}{dt} = \frac{\partial f}{\partial x_1}\frac{dx_1}{dt} + \frac{\partial f}{\partial x_2}\frac{dx_2}{dt} + ... + \frac{\partial f}{\partial x_n}\frac{dx_n}{dt}
$$

(Unicode: dz/dt = Σ (∂f/∂x_i)(dx_i/dt))

**Laplacian (sum of second partials, measures "smoothness"):**

$$
\nabla^2 f = \frac{\partial^2 f}{\partial x_1^2} + \frac{\partial^2 f}{\partial x_2^2} + ... + \frac{\partial^2 f}{\partial x_n^2}
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define a function**

f(x,y) = x²y + 3xy²

#### **Step 2: Compute first partial derivatives**

∂f/∂x = 2xy + 3y² (treat y constant, differentiate x terms)

∂f/∂y = x² + 6xy (treat x constant, differentiate y terms)

#### **Step 3: Evaluate gradient at point (x=1, y=2)**

∂f/∂x at (1,2) = 2×1×2 + 3×4 = 4 + 12 = 16

∂f/∂y at (1,2) = 1² + 6×1×2 = 1 + 12 = 13

∇f(1,2) = [16, 13]ᵀ

#### **Step 4: Compute directional derivative in a specific direction**

Unit vector u = [0.6, 0.8] (points at 53° from x-axis, check: √(0.6²+0.8²)=1)

D_u f = ∇f · u = 16×0.6 + 13×0.8 = 9.6 + 10.4 = 20

Interpretation: Moving from (1,2) in direction u increases f at rate 20 per unit step.

#### **Step 5: Compute second partial derivatives for Hessian**

∂²f/∂x² = 2y

∂²f/∂y² = 6x

∂²f/∂x∂y = 2x + 6y (differentiate ∂f/∂x with respect to y)

Check mixed partials: ∂²f/∂y∂x = 2x + 6y (same, symmetric)

#### **Step 6: Hessian at (1,2)**

∂²f/∂x² = 2×2 = 4

∂²f/∂y² = 6×1 = 6

∂²f/∂x∂y = 2×1 + 6×2 = 2 + 12 = 14

$$
\mathbf{H} = \begin{bmatrix} 4 & 14 \\ 14 & 6 \end{bmatrix}
$$

The Hessian tells curvature. Positive eigenvalues (check later) indicate local minimum.

---

### **7. How this appears inside neural networks and LLMs**

- **Loss landscape optimization**: The gradient (first derivatives) guides descent. The Hessian (second derivatives) helps understand curvature, saddle points, and enables second-order optimizers like Newton's method.

- **Backpropagation through time**: The multivariable chain rule unrolls recurrent networks across time steps, with each time step contributing a term to the total gradient.

- **Attention mechanism**: Computing attention scores involves dot products of query and key vectors—a multilinear operation that multivariable calculus helps analyze.

- **Jacobian matrices**: For vector-valued functions (e.g., a layer mapping from 512 inputs to 1024 outputs), the Jacobian matrix contains all partial derivatives (∂output_i/∂input_j). Used in adversarial robustness and neural ODEs.

- **Natural gradient**: Uses the Fisher information matrix (related to second derivatives) to adjust update directions based on the geometry of parameter space, not just Euclidean distance.

- **Saddle point escape**: In high dimensions, most critical points are saddles, not minima. Multivariable calculus reveals that gradient descent with noise can escape saddles by moving along negative curvature directions.

- **Variational autoencoders**: Use the reparameterization trick, which relies on the chain rule for multivariable functions to backpropagate through random sampling.

---

### **8. Brain-like connection (high-dimensional navigation)**

The brain does not visualize 1000-dimensional spaces, but it navigates them. Each neuron is one dimension. A pattern of firing across 1 million neurons is a point in 1-million-dimensional space. Learning rewires connections, moving that point. The brain's plasticity rules approximate gradient descent in this unimaginably high-dimensional space. Multivariable calculus provides the language to describe what the brain does: it follows the gradient of a reward function, navigates curved manifolds of neural activity, and finds low-dimensional attractor states that represent memories and concepts.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Multivariable calculus is just single-variable calculus done multiple times independently."

_Why it is wrong:_ The interactions between variables create entirely new phenomena. In single-variable calculus, a critical point (derivative=0) is either a min or max. In multivariable calculus, you also get saddle points—where the function curves up in some directions and down in others. The function f(x,y) = x² - y² has a critical point at (0,0) that is a minimum along x-direction but a maximum along y-direction. No single-variable analog exists. The chain rule also gains extra terms (sums appear, not just products). Ignoring these interactions leads to incorrect optimization.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Your AI lives in high-dimensional space. Every weight    |
|  is a dimension. Multivariable calculus is the map and    |
|  compass for that space. It tells you which direction is  |
|  uphill, where the valleys are, and how steep the slopes  |
|  get. Without it, you are wandering blind. With it, you   |
|  navigate the abstract geometry of intelligence itself.   |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have f(x,y) = 3x² + 2y² - 4xy.

**Question:** Compute the gradient ∇f. At (x=1, y=1), which direction (unit vector) gives the greatest increase in f? What is the rate of increase in that direction?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

∂f/∂x = 6x - 4y  
∂f/∂y = 4y - 4x

∇f = [6x - 4y, 4y - 4x]ᵀ

At (1,1):  
∂f/∂x = 6×1 - 4×1 = 2  
∂f/∂y = 4×1 - 4×1 = 0

∇f(1,1) = [2, 0]ᵀ

Greatest increase direction = direction of gradient = [1, 0] (positive x-axis)

Rate of increase = magnitude of gradient = √(2² + 0²) = 2
