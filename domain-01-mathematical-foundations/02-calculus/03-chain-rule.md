# Chain rule

**DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Calculus**

---

**1. Why this concept matters for building intelligent systems**

A neural network is a chain of functions: input → layer 1 → activation → layer 2 → activation → ... → output → loss. An error at the end must propagate backward through every link in the chain to update weights at the beginning. How does a change in layer 1 affect the final loss after passing through five intermediate transformations? The chain rule is the answer. Without it, you cannot train deep networks. Backpropagation is the chain rule applied systematically. The chain rule is the glue that connects each layer's blame to the final mistake.

---

**2. Core idea**

**The chain rule tells you how to differentiate a composite function: the derivative of the outer function multiplied by the derivative of the inner function, extended through any number of nested functions.**

---

**3. Concrete analogy**

Imagine you are a manager at a factory. You want to know how changing the raw material order (R) affects the final profit (P). But there are intermediate steps:

- Raw material R goes into Production → Output Q
- Output Q goes into Sales → Revenue S
- Revenue S minus Costs → Profit P

You can measure:

- How much does output change when raw material changes? (dQ/dR)
- How much does revenue change when output changes? (dS/dQ)
- How much does profit change when revenue changes? (dP/dS)

The chain rule says: multiply these sensitivities together.

dP/dR = (dP/dS) × (dS/dQ) × (dQ/dR)

If each step amplifies or dampens the signal, the product tells the total effect. In a neural network, the "raw material" is a weight in layer 1, and "profit" is the loss. The chain rule multiplies the derivative through every layer in between.

---

**4. ASCII diagram**

```
Chain rule as a cascade of functions:

    x ──→ f ──→ u ──→ g ──→ y

    y = g(f(x))
    u = f(x)
    y = g(u)

    dy/dx = (dy/du) × (du/dx)
          = g'(u) × f'(x)

Visualizing error flow in a neural network:

    Input    Layer 1    Layer 2    Layer 3    Loss
      │         │          │          │         │
      x   →   h₁   →   h₂   →   h₃   →   L
      │         │          │          │         │
      │      ∂h₁/∂x    ∂h₂/∂h₁    ∂h₃/∂h₂    ∂L/∂h₃
      │         │          │          │         │
      └─────────┼──────────┼──────────┼─────────┘
                │          │          │
                └──────────┼──────────┘
                           │
    ∂L/∂x = ∂L/∂h₃ × ∂h₃/∂h₂ × ∂h₂/∂h₁ × ∂h₁/∂x

    Error flows backward from loss to input,
    multiplying partial derivatives along the path.
```

---

**5. Mathematical formulation**

**Single-variable chain rule (two functions):**

If y = g(f(x)) and u = f(x), then:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = g'(f(x)) \cdot f'(x)
$$

(Unicode: dy/dx = dy/du × du/dx = g'(f(x)) × f'(x))

**Single-variable chain rule (multiple functions):**

If y = h(g(f(x))), let u = f(x), v = g(u), then:

$$
\frac{dy}{dx} = \frac{dy}{dv} \cdot \frac{dv}{du} \cdot \frac{du}{dx}
$$

(Unicode: dy/dx = dy/dv × dv/du × du/dx)

**Partial derivative chain rule (multivariable):**

If z = f(x,y) and x = g(t), y = h(t), then:

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial z}{\partial y} \cdot \frac{dy}{dt}
$$

(Unicode: dz/dt = ∂z/∂x × dx/dt + ∂z/∂y × dy/dt)

**For backpropagation (error δ at layer L propagating to layer L-1):**

$$
\delta^{(L-1)} = \delta^{(L)} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial a^{(L-1)}} \cdot \frac{\partial a^{(L-1)}}{\partial z^{(L-1)}}
$$

Where a = activation, z = pre-activation (weighted sum).

---

**6. Worked example (step-by-step)**

**Step 1: Define a tiny neural network**

Input x = 2  
Weight w = 3  
Linear layer: z = w × x = 3 × 2 = 6  
Activation (sigmoid): a = σ(z) = 1/(1 + e⁻ᶻ)  
Loss (simple): L = a² (just to keep it minimal)

**Step 2: Forward pass compute values**

z = 6  
a = σ(6) ≈ 0.9975  
L = (0.9975)² ≈ 0.995

**Step 3: Compute derivatives backward using chain rule**

We want dL/dw to update the weight.

First, dL/da = 2a ≈ 1.995

Second, da/dz = σ(z) × (1 - σ(z)) = a(1-a) ≈ 0.9975 × 0.0025 ≈ 0.0025

Third, dz/dw = x = 2

**Step 4: Apply chain rule**

$$
\frac{dL}{dw} = \frac{dL}{da} \times \frac{da}{dz} \times \frac{dz}{dw}
$$

dL/dw ≈ 1.995 × 0.0025 × 2 ≈ 1.995 × 0.005 ≈ 0.009975

**Step 5: Interpret**

A small increase in w (from 3 to 3.001) increases L by about 0.00001 (0.009975 × 0.001). The sigmoid saturated (z=6 is far in the tail), so the derivative is tiny. The network has learned nothing from this example because the activation is already near 1.

**Step 6: Compare if sigmoid were unsaturated**

If z = 0 instead of 6:  
a = 0.5  
da/dz = 0.5 × 0.5 = 0.25  
dL/dw = 2×0.5 × 0.25 × 2 = 1 × 0.5 = 0.5

Much larger gradient. The chain rule reveals that saturated neurons kill learning.

---

**7. How this appears inside neural networks and LLMs**

- **Backpropagation algorithm**: The entire algorithm is the chain rule applied layer by layer from output to input. Each layer receives error from above, multiplies by its local derivative, and passes the result backward.

- **Vanishing gradients**: In deep networks, multiplying many derivatives < 1 (e.g., sigmoid or tanh) makes the product exponentially small. The chain rule reveals why: dL/dw = (small) × (small) × ... × (small).

- **Exploding gradients**: If derivatives > 1 (e.g., large weights in ReLU regions), the product explodes. The chain rule exposes this as instability.

- **Automatic differentiation**: Every deep learning framework (PyTorch, TensorFlow, JAX) builds a computational graph and applies the chain rule automatically to compute gradients.

- **Residual connections**: Adding skip connections creates parallel paths: y = F(x) + x. The derivative becomes dF/dx + 1. The +1 prevents vanishing because error can flow directly through the skip connection without passing through many small derivatives.

- **Gradient clipping**: When the chain rule produces exploding gradients, you cap the product to a maximum value.

- **Time backpropagation (BPTT)** : For recurrent networks, the chain rule unrolls through time. dL/dw = sum over t of (∂L/∂h_t × (product of derivatives from t to 0)).

---

**8. Brain-like connection (error propagation)**

The brain does not literally compute the chain rule, but it solves the same credit assignment problem. When you reach for a coffee cup and miss, how does your brain adjust the motor commands from shoulder, elbow, wrist, and fingers? Each joint contributed to the error. Theories of cerebellar learning propose that climbing fibers carry error signals that multiply with parallel fiber activities—a biological approximation of the chain rule. The brain's ability to learn sequences (speech, music, dance) requires propagating error through time, just like backpropagation through time in RNNs. Evolution discovered the chain rule in the form of synaptic plasticity rules that correlate pre-synaptic activity with post-synaptic error.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "The chain rule is just canceling fractions. dy/dx = (dy/du) × (du/dx) and the du cancels like (dy/~~du~~) × (~~du~~/dx)."

_Why it is wrong:_ The "cancellation" is a helpful mnemonic, not a mathematical proof. Derivatives are limits, not fractions. For multivariable chain rules, cancellation fails completely. For example, if z = f(x,y) with x=g(t), y=h(t), then dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt). You cannot "cancel" anything because partial derivatives track different independent variables. The single-variable notational trick works only in the simplest case. Relying on cancellation will lead to errors in backpropagation where sums appear, not just products.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  A deep neural network is a chain of transformations.     |
|  The chain rule is how blame travels backward through     |
|  every link. It answers: "When I make a mistake at the    |
|  end, who caused it and by how much?" Without the chain   |
|  rule, the first layer of a 100-layer network could never |
|  learn. With it, error flows backward like a rumor        |
|  spreading through a chain of messengers.                 |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

You have three functions:  
f(x) = x²  
g(u) = u + 3  
h(v) = 2v

Let y = h(g(f(x))).

**Question:** Using the chain rule, find dy/dx. Evaluate at x = 2.

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Step by step:  
u = f(x) = x²  
v = g(u) = u + 3 = x² + 3  
y = h(v) = 2v = 2(x² + 3) = 2x² + 6

Derivative directly: dy/dx = 4x

Chain rule:  
dy/dv = 2  
dv/du = 1  
du/dx = 2x

Product: dy/dx = 2 × 1 × 2x = 4x

At x=2: dy/dx = 8
