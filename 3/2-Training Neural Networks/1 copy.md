# Learning rates

**DOMAIN: DEEP LEARNING FOUNDATIONS | Sub domain: Training Neural Networks**

---

**1. Why this concept matters for building intelligent systems**

Gradients tell you which direction to move. But how far? That is the learning rate. It is the single most important hyperparameter in deep learning. Too large and your network diverges into nonsense. Too small and training takes forever or gets stuck in poor local minima. The right learning rate balances speed and stability. Modern training often uses schedules that start high (fast progress) and decay over time (fine-tune at the end). Understanding learning rates means understanding the throttle of neural network training—push too hard and you crash, too soft and you never arrive.

---

**2. Core idea**

**The learning rate controls the step size during gradient descent, determining how much the parameters change in response to the estimated error at each training iteration.**

---

**3. Concrete analogy**

Imagine you are trying to find the lowest point in a dark valley. You can feel the slope at your feet and take a step.

- **Learning rate too high (η=1.0)**: You take giant leaps. You overshoot the valley, bounce back and forth, maybe even launch yourself out of the valley entirely. You never settle.

- **Learning rate too low (η=0.00001)**: You inch forward microscopically. After hours of walking, you have barely moved. You might get stuck in a small dip that is not the true minimum.

- **Learning rate just right (η=0.01)**: You take confident steps that descend steadily. When you get near the bottom, you step carefully, settling into the minimum.

Now imagine you have a schedule. Start with large steps to descend quickly, then automatically shrink your step size as you approach the bottom. This is learning rate decay or a cosine schedule—the smart hiker's strategy.

---

**4. ASCII diagram**

```
Learning rate effects on convergence:

    Loss →
         │
         │      η too high (diverges)
         │    ╱╲    ╱╲
         │   ╱  ╲  ╱  ╲
         │  ╱    ╲╱    ╲
         │ ╱            ╲
         │╱              ╲___
         │
         │   η just right (smooth descent)
         │      ●────●───●
         │    ╱         ╲
         │   ╱           ╲
         │  ╱             ●
         │ ╱
         │╱
         │
         │   η too low (slow)
         │ ●─────●─────●─────●─────●
         └──────────────────────────→ Iteration


Learning rate schedules:

    η ↑
      │
    1e-3┤           ╱────
      │          ╱
      │        ╱
      │      ╱
      │    ╱  Constant
    1e-4┤  ╱
      │  ╱
      │ ╱
      │╱
      └──────────────────→ Step


    η ↑
      │
    1e-3┤ ╱╲
      │╱  ╲
      │    ╲   Step decay
      │     ╲╱╲
      │       ╲
      │        ╲╱╲
      └──────────────────→ Step


    η ↑
      │        ┌─────┐
      │       ╱       ╲
      │      ╱         ╲      Cosine
      │     ╱           ╲
      │    ╱             ╲
      │   ╱               ╲
      └──────────────────────→ Step


    Warmup (critical for transformers):

    η ↑
      │         ┌─────
      │        ╱
      │       ╱
      │      ╱
      │     ╱          Linear warmup
      │    ╱            then constant
      │   ╱
      │  ╱
      └──────────────────────→ Step
```

---

**5. Mathematical formulation**

**SGD update with learning rate η:**

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

**Learning rate schedules:**

- **Constant:** η_t = η₀

- **Step decay:** η_t = η₀ × γ^{floor(t / step_size)}

- **Exponential decay:** η_t = η₀ × e^{-kt}

- **Inverse time decay:** η_t = η₀ / (1 + kt)

- **Polynomial decay:** η_t = η₀ × (1 - t/T)^p

**Cosine annealing:**

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

Where T = total number of steps.

**Warmup (linear):**

For t < t_warmup:

$$
\eta_t = \eta_{\max} \times \frac{t}{t_{\text{warmup}}}
$$

After warmup: use chosen schedule.

**Cyclical learning rates:** Oscillate between η_min and η_max. Often improves generalization.

**Learning rate finder (Leslie Smith):** Run for a few epochs, increasing η exponentially. Plot loss vs η. Choose η where loss decreases fastest (before explosion).

**Batch size and learning rate relationship:**

Linear scaling rule: when increasing batch size by factor k, increase η by √k (or k, empirically tuned).

For Adam, typical default: η = 0.001 (often works without tuning).

---

**6. Worked example (step-by-step)**

**Step 1: Simple quadratic function**

Minimize f(w) = w², minimum at w=0. Start w₀ = 10.

**Step 2: Learning rate η=0.1**

w₁ = 10 - 0.1×(20) = 10 - 2 = 8
w₂ = 8 - 0.1×(16) = 8 - 1.6 = 6.4
w₃ = 6.4 - 0.1×(12.8) = 6.4 - 1.28 = 5.12

After 10 steps: w ≈ 1.07, after 20: w ≈ 0.12 (steady convergence)

**Step 3: Learning rate η=0.01 (too low)**

w₁ = 10 - 0.01×20 = 10 - 0.2 = 9.8
After 10 steps: w ≈ 8.1, after 50: w ≈ 3.7, after 200: w ≈ 0.5 (very slow)

**Step 4: Learning rate η=1.0 (too high)**

w₁ = 10 - 1.0×20 = 10 - 20 = -10
w₂ = -10 - 1.0×(-20) = -10 + 20 = 10
Oscillates between 10 and -10 forever. Never converges.

**Step 5: Cosine schedule example**

T=100 steps, η_max=0.1, η_min=0.001

At t=0: η = 0.001 + 0.5×(0.099)×(1+cos(0)) = 0.001 + 0.0495×2 = 0.1
At t=50 (halfway): cos(π/2)=0 → η = 0.001 + 0.0495×1 = 0.0505
At t=100: cos(π)=-1 → η = 0.001 + 0.0495×0 = 0.001

**Step 6: Warmup example**

η_max=0.001, warmup_steps=1000

At t=500: η = 0.001 × (500/1000) = 0.0005
At t=1000: η = 0.001
After warmup: decay to 0.0001 over remaining steps.

---

**7. How this appears inside neural networks and LLMs**

- **LLM training (GPT, Llama, BERT):** Use AdamW with learning rate warmup (typically 2000-5000 steps) followed by cosine decay. Starts at 0, peaks at ~1e-4 to 6e-4, decays to ~1e-5.

- **Batch size scaling:** LLMs use large batches (e.g., 0.5M tokens). Learning rate scales roughly linearly with batch size.

- **Layer-wise learning rates (LARS/LAMB):** For large-batch training, learning rate is scaled per layer based on parameter norm.

- **Learning rate finder:** Run a small experiment before full training to identify a safe range.

- **Gradient clipping + learning rate:** For LLMs, gradient norm is clipped (e.g., 1.0) and learning rate is set independently. Both control update magnitude.

- **Stable training:** Warmup prevents early instability from large gradients. The first few steps with very small η allow the optimizer to "settle" before full updates.

- **Fine-tuning:** When fine-tuning a pretrained LLM, use much smaller learning rate (e.g., 1e-5 to 5e-5) than pretraining (1e-4), and often shorter warmup.

- **One-cycle policy:** Increase learning rate, then decrease. Used in many computer vision models (fast.ai).

---

**8. Brain-like connection (neuromodulation of plasticity)**

The brain controls its own "learning rate" through neuromodulators. Acetylcholine and norepinephrine regulate synaptic plasticity, effectively scaling how much neural connections change in response to experience. During novel situations, the brain increases its effective learning rate (higher plasticity). During familiar routines, learning rate decreases (consolidation). Sleep may further consolidate memories with very low effective learning rate, preventing overwriting. The brain's ability to adapt its learning rate is far more sophisticated than current schedules, but the principle—change how much you learn based on context—is deeply analogous to learning rate scheduling in deep learning.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "The learning rate should be as large as possible without diverging. Bigger steps mean faster training."

_Why it is wrong:_ The optimal learning rate for fast convergence is typically smaller than the divergence threshold. Too large a learning rate causes oscillation and can land in sharper minima that generalize poorly. Also, a single learning rate throughout training is suboptimal: you want large steps early (fast progress) and small steps later (fine-tuning). This is why schedules exist. The largest stable learning rate is not the fastest to low loss—medium rates with decay often win. You must tune, not just maximize.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  The learning rate is your stride length across the loss  |
|  landscape. Stride too long and you trip. Stride too      |
|  short and you crawl. The perfect stride changes over     |
|  time: long strides when you are far from the goal,       |
|  short steps when you are close. Learning rate schedules  |
|  automate this wisdom. Master the learning rate and you   |
|  control the rhythm of learning itself—the pace at which  |
|  your network transforms from random to remarkable.       |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

You are training a neural network. At iteration 100, loss = 2.0. You increase the learning rate by 10×. At iteration 101, loss becomes 5.0 (higher). At iteration 102, loss becomes 12.0 (much higher).

**Question:** What is happening? Should you increase the learning rate further, decrease it, or keep it the same?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** The network is diverging. After increasing the learning rate, loss increased sharply and continues to increase (trend is upward). This indicates the learning rate is too high. You should decrease the learning rate—likely back to the original value or even lower. Continuing with a high learning rate will cause loss to go to infinity. If you decrease the learning rate and loss still increases, something else is wrong (bug, gradient explosion, data issue). In this case, revert and reduce η.
