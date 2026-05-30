# Partial derivatives

**DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Calculus**

---

**1. Why this concept matters for building intelligent systems**

A neural network has thousands or millions of parameters. Each parameter influences the final output. But how do you isolate the effect of a single weight when all the others are also changing? You need a way to hold everything constant except one variable, measure the sensitivity, then repeat for every variable. That is a partial derivative. Partial derivatives are the atomic units of the gradient. Without them, you cannot tell which knob is responsible for which part of the error. With them, every parameter gets its own personalized learning signal.

---

**2. Core idea**

**A partial derivative measures how a multivariable function changes when you vary one input variable while holding all other variables fixed.**

---

**3. Concrete analogy**

Imagine you are baking bread. The final quality Q depends on three variables: temperature T (oven), time t (baking minutes), and hydration H (water percentage). You want to improve your bread. But changing temperature also changes how time behaves. How do you isolate each factor?

- Hold time constant at 30 minutes and hydration at 65%. Vary temperature up by 10°. Measure quality change. That is ∂Q/∂T.
- Hold temperature at 180°C and hydration at 65%. Vary time up by 2 minutes. That is ∂Q/∂t.
- Hold temperature and time constant. Vary hydration. That is ∂Q/∂H.

Each partial derivative tells you the isolated effect of one ingredient. The collection of all three tells you the complete sensitivity profile of your recipe. In a neural network, each weight has a partial derivative showing how much that specific weight contributes to the final error.

---

**4. ASCII diagram**

```
Visualizing partial derivatives on a 3D surface:

    f(x,y) (height)
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
└────────┼────────┼──→ y
         │        │
         └────────┴──→ x

    ∂f/∂x at point P:
        Slice the surface parallel to x-axis (fix y)
        Find slope of that slice

    ∂f/∂y at point P:
        Slice the surface parallel to y-axis (fix x)
        Find slope of that slice

    At a specific point (x₀, y₀):

         y
         ↑
         │    ┌─────┐
         │   ╱ ∂f/∂y ╲
         │  ╱ (y-direction) ╲
         │ ╱         ↑      ╲
         │╱          │       ╲
         ●───────────┼──────────→ x
          ╲          │       ╱
           ╲    ∂f/∂x→      ╱
            ╲   (x-direction)╱
             ╲              ╱
              └────────────┘
```

---

**5. Mathematical formulation**

For a function f(x₁, x₂, ..., xₙ), the partial derivative with respect to x_i is:

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, ..., x_i + h, ..., x_n) - f(x_1, ..., x_i, ..., x_n)}{h}
$$

(Unicode: ∂f/∂x*i = lim*{h→0} (f(..., x_i+h, ...) - f(...)) / h)

**Notation alternatives:**

$$
\frac{\partial f}{\partial x} = f_x = \partial_x f
$$

**Higher-order partial derivatives:**

$$
\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right)
$$

**Mixed partials (order usually doesn't matter for smooth functions):**

$$
\frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right) = \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}
$$

(Unicode: ∂²f/∂x∂y = ∂²f/∂y∂x)

**Key rule: Treat other variables as constants**

- For ∂/∂x of f(x,y) = x²y³: treat y³ as constant → 2x × y³
- For ∂/∂y of f(x,y) = x²y³: treat x² as constant → x² × 3y²

---

**6. Worked example (step-by-step)**

**Step 1: Define a simple neural network loss**

Network with two weights w₁ and w₂, input x=2, target t=3.

Prediction: ŷ = w₁ × x + w₂ = 2w₁ + w₂

Loss (mean squared error): L = (ŷ - t)² = (2w₁ + w₂ - 3)²

**Step 2: Compute partial derivative with respect to w₁**

Treat w₂ as constant. Let u = 2w₁ + (w₂ - 3)

∂L/∂w₁ = 2(2w₁ + w₂ - 3) × ∂/∂w₁(2w₁ + w₂ - 3)

∂/∂w₁(2w₁ + w₂ - 3) = 2

Therefore: ∂L/∂w₁ = 2(2w₁ + w₂ - 3) × 2 = 4(2w₁ + w₂ - 3)

**Step 3: Compute partial derivative with respect to w₂**

Treat w₁ as constant. ∂/∂w₂(2w₁ + w₂ - 3) = 1

Therefore: ∂L/∂w₂ = 2(2w₁ + w₂ - 3) × 1 = 2(2w₁ + w₂ - 3)

**Step 4: Evaluate at specific weights**

Current weights: w₁ = 1, w₂ = 0

Compute error term: 2×1 + 0 - 3 = 2 - 3 = -1

∂L/∂w₁ = 4 × (-1) = -4

∂L/∂w₂ = 2 × (-1) = -2

**Step 5: Interpret the partial derivatives**

- ∂L/∂w₁ = -4: Increasing w₁ by a small amount reduces loss by approximately 4× that amount
- ∂L/∂w₂ = -2: Increasing w₂ reduces loss by approximately 2× that amount

w₁ has twice the impact of w₂ on the loss at this point.

**Step 6: Gradient**

$$
\nabla L = \begin{bmatrix} -4 \\ -2 \end{bmatrix}
$$

Negative gradient points downhill: increase both w₁ and w₂ to decrease loss.

---

**7. How this appears inside neural networks and LLMs**

- **Backpropagation core**: The chain rule for partial derivatives is applied layer by layer. ∂L/∂w for layer l depends on ∂L/∂output of layer l (the "error signal" from above) and the input to that layer.

- **Parameter updates**: Each weight gets its own partial derivative. In SGD, w_new = w_old - α × ∂L/∂w. A bias term in layer 3 has no direct dependence on input pixels; its partial derivative only sees error propagated through later layers.

- **Sparsity in gradients**: For ReLU activation, ∂L/∂input = 0 when the neuron is inactive. This makes gradients sparse and speeds computation.

- **Batch gradients**: For a batch of examples, ∂L_total/∂w = (1/batch_size) × Σ(∂L_i/∂w). Partial derivatives average across examples.

- **Gradient checking**: During debugging, you can numerically approximate ∂L/∂w by (L(w+ε) - L(w-ε))/(2ε) and compare to analytical result.

- **Layer normalization**: The derivative of the normalization step requires partial derivatives with respect to mean and variance, then back through those to inputs.

---

**8. Brain-like connection (isolating causes)**

The brain cannot compute explicit partial derivatives, but it solves the credit assignment problem: which synapse caused which part of the error? Local learning rules (like spike-timing-dependent plasticity) approximate partial derivatives by correlating pre-synaptic activity with post-synaptic error signals. The cerebellum uses a clever trick: it has parallel fibers (inputs) and climbing fibers (error signals). When a climbing fiber fires, it modifies all parallel fiber synapses that were recently active, effectively computing a partial derivative of error with respect to each input. Evolution found a biological implementation of ∂E/∂w before calculus was formalized.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "A partial derivative treats other variables as constant, so they don't matter at all."

_Why it is wrong:_ "Treat as constant" only means you do not differentiate with respect to them in that step. The other variables still appear in the resulting expression. For f(x,y) = x²y, ∂f/∂x = 2xy. The y is still there—it is treated as a constant during differentiation, but its value multiplies the result. If y = 0, ∂f/∂x = 0 regardless of x. The other variables absolutely matter; you just are not taking their derivative at that moment.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Every parameter in your neural network lives in a        |
|  crowd of other parameters. The partial derivative is     |
|  how you isolate one voice in that crowd. It answers:     |
|  "If I tweak only this knob, holding everything else      |
|  perfectly still, what happens to the error?" Without     |
|  partial derivatives, learning is a chaotic mess. With    |
|  them, every parameter gets its own tailored instruction. |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

You have a function f(x,y) = x²y + 3x - y².

**Question:** Compute ∂f/∂x and ∂f/∂y. Then evaluate both at the point (x=2, y=1).

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

∂f/∂x = 2xy + 3 (treat y as constant, differentiate x²→2x, 3x→3)

∂f/∂y = x² - 2y (treat x as constant, differentiate y→1, -y²→-2y)

At (x=2, y=1):

∂f/∂x = 2×2×1 + 3 = 4 + 3 = 7

∂f/∂y = 2² - 2×1 = 4 - 2 = 2
