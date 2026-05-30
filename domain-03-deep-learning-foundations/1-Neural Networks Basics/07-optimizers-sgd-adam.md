# Optimizers (SGD, Adam)

**DOMAIN: DEEP LEARNING FOUNDATIONS | Sub domain: Neural Networks Basics**

---

**1. Why this concept matters for building intelligent systems**

Gradients tell you which direction is downhill. But how far should you step? What if the terrain is rocky, with steep cliffs and flat valleys? What if some directions need smaller steps than others? Optimizers answer these questions. They convert raw gradients into parameter updates, shaping the trajectory through the loss landscape. Standard SGD takes naive steps. Momentum accelerates through flat regions. Adam adapts the step size for each parameter individually. The choice of optimizer determines whether your network trains in hours, days, or never. Understanding optimizers means understanding how to navigate the complex terrain of high-dimensional learning.

---

**2. Core idea**

**Optimizers are algorithms that update neural network parameters using gradients, with different strategies for step size, momentum, and per-parameter adaptation to accelerate convergence and escape poor local minima.**

---

**3. Concrete analogy**

Imagine you are descending a mountain in thick fog. You can feel the slope at your feet (gradient), but you cannot see the terrain ahead.

- **Vanilla SGD**: Take a small step in the steepest downhill direction. Repeat. It works, but you might zigzag down narrow valleys, wasting time.

- **SGD with Momentum**: Keep a running average of recent steps. If you have been going in the same direction, take a larger step. Like a heavy ball rolling down the mountain—it builds speed, cuts through noise, and rolls over small bumps.

- **Adam**: Maintain not just momentum (first moment) but also a running estimate of how variable the terrain is (second moment). Steep, consistent slopes get large steps. Bumpy, inconsistent areas get small steps. Each direction (parameter) gets its own customized step size.

In deep learning, the "mountain" has millions of dimensions. Momentum and adaptive methods are essential for navigating this landscape efficiently.

---

**4. ASCII diagram**

```
SGD without momentum (erratic, slow):

    Loss ↓
          │   ╱╲    ╱╲
          │  ╱  ╲  ╱  ╲
          │ ╱    ╲╱    ╲
          │╱            ╲
          └────────────────→ Steps

    Path zigzags down the valley.


SGD with momentum (smoother, faster):

    Loss ↓
          │   ┌───┐
          │  ╱     ╲
          │ ╱       ╲
          │╱         └────
          └────────────────→ Steps

    Momentum smooths the path, ignores small fluctuations.


Learning rate comparison:

    Loss ↓
          │
          │     ╱╲
          │    ╱  ╲
          │   ╱    ╲
          │  ╱      ╲
          │ ╱        ╲
          │╱          ╲
          └────────────────→ Steps

    Too high (η=0.1): ──●══●══●══ (oscillates, diverges)
    Too low (η=0.001): ──●─────●─────● (slow, gets stuck)
    Just right (η=0.01): ──●──●──●──● (steady descent)


Adaptive per-parameter learning rates (Adam):

    Parameter 1 (steep, consistent): large steps
    Parameter 2 (flat, noisy): small steps

    w₁ ────●────●────●────●──── (fast)
    w₂ ──●──●──●──●──●── (slow, cautious)
```

---

**5. Mathematical formulation**

**SGD (Stochastic Gradient Descent):**

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

Where η = learning rate

**SGD with Momentum:**

$$
v_t = \beta v_{t-1} + \nabla L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

Where β (e.g., 0.9) controls momentum decay.

**Nesterov Accelerated Gradient (NAG):**

$$
v_t = \beta v_{t-1} + \nabla L(\theta_t - \eta \beta v_{t-1})
$$

Look ahead before computing gradient.

**AdaGrad (adaptive per-parameter):**

$$
g_t = \nabla L(\theta_t)
$$

$$
G_t = G_{t-1} + g_t^2 \quad \text{(elementwise sum of squares)}
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t
$$

Problems: G_t grows forever, learning rate → 0.

**RMSProp (fixes AdaGrad's decay):**

$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

**Adam (Adaptive Moment Estimation):**

First moment (mean, momentum):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

Second moment (uncentered variance):

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

**Bias correction** (since m₀, v₀ start at 0):

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Parameter update:**

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Typical defaults: β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, η = 0.001

**AdamW (Adam with decoupled weight decay):**

Adds L2 regularization directly to update, not through gradient.

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

---

**6. Worked example (step-by-step)**

**Step 1: Simple 1D problem**

Minimize f(w) = w². Minimum at w=0.
Start w=2. Gradient = 2w = 4.

**Step 2: SGD with η=0.1**

w₁ = 2 - 0.1×4 = 2 - 0.4 = 1.6
w₂ = 1.6 - 0.1×3.2 = 1.6 - 0.32 = 1.28
Converges to 0 but slowly.

**Step 3: Momentum with β=0.9, η=0.1**

v₀ = 0
v₁ = 0.9×0 + 4 = 4, w₁ = 2 - 0.1×4 = 1.6
v₂ = 0.9×4 + (2×1.6=3.2) = 3.6 + 3.2 = 6.8, w₂ = 1.6 - 0.1×6.8 = 1.6 - 0.68 = 0.92
v₃ = 0.9×6.8 + (2×0.92=1.84) = 6.12 + 1.84 = 7.96, w₃ = 0.92 - 0.796 = 0.124

Momentum converges faster (0.124 vs 1.28 after 2 steps).

**Step 4: Adam on same problem**

β₁=0.9, β₂=0.999, η=0.1, ε=1e-8

t=1: m₁ = 0.9×0 + 0.1×4 = 0.4, v₁ = 0.999×0 + 0.001×16 = 0.016
m̂₁ = 0.4/(1-0.9)=0.4/0.1=4, v̂₁ = 0.016/(1-0.999)=0.016/0.001=16
w₁ = 2 - 0.1 × 4/(√16+1e-8)=2 - 0.1×4/4=2 - 0.1=1.9

Adam took a smaller first step than SGD (1.9 vs 1.6) because second moment estimate scaled down the effective learning rate. Over time, Adam adapts.

---

**7. How this appears inside neural networks and LLMs**

- **AdamW as default for LLMs:** Most LLMs (GPT, Llama, BERT) use AdamW (Adam with decoupled weight decay). Handles the scale of billions of parameters.

- **Learning rate warmup:** Start with small learning rate, increase linearly to target over first few thousand steps. Essential for transformer training stability.

- **Cosine decay schedule:** After warmup, decay learning rate following cosine curve. Improves final convergence.

- **Gradient clipping:** Clip gradients if their norm exceeds threshold (e.g., 1.0). Prevents exploding gradients in LLMs.

- **Batch size scaling:** Linear scaling rule: when increasing batch size by k, increase learning rate by √k or k (empirically tuned).

- **Layer-wise adaptive learning rates:** LARS/LAMB optimizers adjust learning rate per layer based on parameter norm. Used for large-batch training.

- **ZeRO optimizer (DeepSpeed):** Distributes optimizer states across GPUs, enabling training of trillion-parameter models.

- **8-bit optimizers (bitsandbytes):** Compresses optimizer states to 8-bit integers, reducing memory by 75%.

---

**8. Brain-like connection (synaptic plasticity and momentum)**

The brain's learning rules incorporate momentum-like effects. Synaptic changes persist over time (short-term plasticity). Recent firing patterns influence current plasticity, smoothing out noisy updates. Dopamine signals (reward prediction errors) are integrated over time, behaving like a momentum term in reinforcement learning. The brain's ability to learn stable representations despite noisy neural activity suggests biological optimizers have built-in momentum and adaptive properties. While the brain's exact algorithm differs from Adam, the functional requirements—stability, noise robustness, and efficient use of limited information—are identical.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Adam always converges faster and to a lower loss than SGD. It is strictly better."

_Why it is wrong:_ Adam often converges faster in early training but can generalize worse than SGD with momentum on some tasks (especially image classification). Adam's adaptive learning rates can lead to solutions that are sharper minima, which generalize poorly. Many state-of-the-art results still use SGD with momentum (e.g., ResNets on ImageNet) after Adam warmup. The choice is task-dependent. For LLMs, AdamW dominates, but for computer vision, SGD with momentum is competitive or better. Never assume one optimizer is universally superior.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Gradients tell you where to go. Optimizers tell you how. |
|  SGD walks. Momentum runs. Adam dances—adapting each      |
|  step to the terrain. Choose the wrong optimizer and      |
|  your network stalls in flat regions or oscillates into   |
|  divergence. Choose the right one and you glide down the  |
|  loss landscape smoothly. In the race to the minimum,     |
|  your optimizer is your pace, your rhythm, your strategy. |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

You have a loss landscape with two parameters: w₁ has large, consistent gradients; w₂ has small, noisy gradients.

**Question:** Would vanilla SGD (same learning rate for both) or Adam (per-parameter learning rates) be better? Why?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Adam would be better. Vanilla SGD uses the same learning rate for both parameters. To make progress on w₂ (small gradients), you might need a large learning rate, which would cause w₁ (large gradients) to oscillate or diverge. Adam adapts: w₁ gets small steps (large gradients → large second moment → small effective step), w₂ gets large steps (small gradients → small second moment → large effective step). Adam handles heterogeneous gradient scales automatically.
