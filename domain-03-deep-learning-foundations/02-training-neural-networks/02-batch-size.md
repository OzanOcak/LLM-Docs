# Batch size

## **DOMAIN: DEEP LEARNING FOUNDATIONS | Sub domain: Training Neural Networks**

---

### **1. Why this concept matters for building intelligent systems**

How many examples should you look at before updating your weights? One? A hundred? The entire dataset? Batch size is this choice. It is a critical lever balancing speed, stability, and generalization. Small batches are noisy (each update is erratic) but regularize the model. Large batches are stable and fast on hardware but can overfit and get stuck in sharp minima. Finding the right batch size is essential for efficient training. Understanding batch size means understanding the tradeoff between gradient estimate quality, computational efficiency, and final model performance.

---

### **2. Core idea**

**Batch size determines how many training examples are used to compute each gradient update, balancing between the noisy but fast updates of stochastic gradient descent and the accurate but expensive updates of full-batch gradient descent.**

---

### **3. Concrete analogy**

Imagine you are learning to throw darts at a bullseye. You can learn in three ways:

- **Stochastic (batch size 1)**: Throw one dart, see where it lands, adjust your aim. You adjust after every throw. Fast to adjust, but each single dart is noisy (wind, twitch). You might overreact to bad luck.

- **Mini-batch (batch size 32)**: Throw 32 darts, look at the average landing position, adjust your aim once. Smooths out noise. You still adjust frequently, but each adjustment is more reliable.

- **Full-batch (batch size = all data)**: Throw 1000 darts, compute the exact center of all landings, adjust once. Very accurate gradient, but you only adjust once per full dataset. Takes forever to collect all darts, and you might miss opportunities to adjust sooner.

In deep learning, mini-batches (32-512 examples) are the sweet spot. They give reliable gradient estimates, leverage GPU parallelism, and update frequently enough to make rapid progress.

---

### **4. ASCII diagram**

```
Batch size effects on gradient noise and convergence:

    True gradient direction (downhill):
                ↓

    Batch size 1 (noisy):
        ↙   ↘   ↙   ↓   ↘   ↙   ↓   ↙   ↘
    Path zigzags but moves downhill eventually.

    Batch size 32 (moderate noise):
        ↙   ↓   ↘   ↓   ↙   ↓
    Smoother path, faster progress per epoch.

    Batch size 1024 (low noise):
        ↓   ↓   ↓   ↓
    Stable, but updates less frequent.

Training time tradeoff:

    Batch size →    1       32      128     1024    Full
    Updates/sec:    High    High    Med     Low     Very Low
    Noise:          High    Med     Low     Very Low None
    GPU utilization: Low    High    High    High    High
    Generalization: Best    Good    Med     Poor    Poor

Loss landscape and batch size:

    Small batch:               Large batch:
    Finds wide minima          Finds sharp minima
    (generalizes well)         (generalizes poorly)

    Loss
      ↑                 ╲
      │                  ╲    ┌─────┐
      │                   ╲   │     │
      │                    ╲  │     │
      │                     ╲ │     │
      │                      ╲│     │
      └──────────────────────────→ Parameter

    Wide minima (small batch) more robust to shift.
```

---

### **5. Mathematical formulation**

**Full-batch gradient descent (batch size = n):**

$$
\nabla L_{\text{full}}(\theta) = \frac{1}{n} \sum_{i=1}^n \nabla L_i(\theta)
$$

**Stochastic gradient descent (batch size = 1):**

$$
\nabla L_{\text{SGD}}(\theta) = \nabla L_i(\theta) \quad \text{for random } i
$$

**Mini-batch gradient descent (batch size = B):**

$$
\nabla L_{\text{mb}}(\theta) = \frac{1}{B} \sum_{i \in \text{batch}} \nabla L_i(\theta)
$$

**Gradient variance:** For batch size B, gradient variance scales as 1/B (for independent samples). Larger batches = less noise.

**Update rule (same for all, only gradient changes):**

$$
\theta_{t+1} = \theta_t - \eta \nabla L_{\text{batch}}(\theta_t)
$$

**Epoch:** One complete pass through the entire dataset.

Number of updates per epoch = n / B

**Generalization gap:** Empirical observation: for the same number of epochs, smaller batches generalize better (test accuracy higher) because the noise helps escape sharp minima.

**Linear scaling rule:** When increasing batch size by factor k, increase learning rate by √k (or k, empirically tuned) to maintain similar training dynamics.

**Critical batch size:** Point beyond which increasing batch size gives diminishing returns in training speed. Usually around 32-512 for vision tasks, larger for LLMs (0.5M tokens per batch).

---

### **6. Worked example (step-by-step)**

#### **Step 1: Setup**

Dataset: 1000 examples. Loss: MSE. True gradient direction optimal.

#### **Step 2: Batch size 1 (SGD)**

Each update uses 1 example. Updates per epoch = 1000.
Gradient estimate is noisy: sometimes points slightly off true direction.
Path zigzags but moves downhill. Good generalization.

#### **Step 3: Batch size 100**

Each update uses 100 examples. Updates per epoch = 10.
Gradient averages 100 examples, much smoother.
Path is straighter, but updates are 100× less frequent.

#### **Step 4: Compute variance reduction**

Assume each example's gradient has variance σ² = 1.0.

Batch size 1: gradient variance = 1.0
Batch size 100: gradient variance = 1.0/100 = 0.01

Standard deviation ratio: √0.01 = 0.1. Noise reduced by 10×.

#### **Step 5: Compare convergence speed in steps vs time**

In steps: Batch size 100 needs fewer steps (smoother, more direct).
In time: Batch size 100 uses GPU parallelism (matrix-matrix instead of matrix-vector). Often faster per step as well.

#### **Step 6: Numerical example**

Simple quadratic: f(w) = w², minimum at 0. Start w₀ = 10.
True gradient = 2w.

SGD (batch=1), η=0.1, one gradient sample might be 2w + noise.

Example: w=10, true gradient=20. Noisy gradient sample = 18.
Update: w=10 - 0.1×18 = 8.2

Batch of 10, average gradient = 19.2.
Update: w=10 - 1.92 = 8.08

Batch update is closer to optimal (8.08 vs 8.2). More stable.

---

### **7. How this appears inside neural networks and LLMs**

- **GPU/TPU utilization:** Larger batches leverage parallelism. Matrix multiplication is faster when matrices are larger. On modern hardware, batch sizes of 32-512 are typical for efficient GPU utilization.

- **LLM batch sizes:** Measured in tokens, not examples. GPT-3 used batch size of 3.2M tokens (about 1,500 documents). Extremely large batches require careful learning rate tuning (linear scaling rule).

- **Gradient accumulation:** Simulate large batch on limited memory by accumulating gradients over multiple small batches before updating. Effective batch size = mini-batch size × accumulation steps.

- **Batch normalization:** Behaves differently with batch size. Small batches (<8) give poor statistics. Large batches (>64) give stable estimates. BN requires tuning batch size.

- **Distributed training (data parallelism):** Each GPU processes a subset of batch. Gradients are all-reduced (averaged) across GPUs. Global batch size = per-GPU batch × number of GPUs.

- **Memory constraints:** Larger batches require more memory (store intermediate activations). Tradeoff: larger batch vs larger model.

- **Generalization:** Small batches (32-128) often generalize better than large batches (512+). The noise in small batches acts as implicit regularization.

- **Critical batch size for LLMs:** Around 0.5M tokens. Beyond that, training speed per token saturates (diminishing returns).

---

### **8. Brain-like connection (online vs batch learning)**

The brain learns online (batch size 1 for each experience). Each sensory event triggers immediate synaptic updates. This is stochastic gradient descent in biology. Yet the brain also consolidates memories during sleep, replaying batches of experiences—effectively performing batch updates offline. This replay strengthens important patterns and weakens noise. The combination of online learning (day) and batch replay (sleep) gives the brain both adaptability and stability. Modern machine learning uses similar ideas: online SGD for fast adaptation, replay buffers for stability in reinforcement learning.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Larger batches are always better because they give more accurate gradients and train faster in wall-clock time."

_Why it is wrong:_ Larger batches often generalize worse. The noise in small batches helps escape sharp minima and find flatter regions that generalize better. Also, the linear scaling rule for learning rate breaks down beyond the critical batch size—gradient estimates are so accurate that the effective learning rate per step becomes too large, causing overshooting. There is a U-curve: very small batches are noisy and slow; very large batches generalize poorly; the sweet spot is in the middle (typically 32-256 for vision, larger for LLMs but still not maximal hardware batch size). Always tune batch size for generalization, not just speed.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Batch size is your learning rhythm. One example at a     |
|  time: reactive, noisy, but explores widely. Many at      |
|  once: stable, smooth, but may settle too quickly. The    |
|  right batch size balances noise and accuracy, speed and  |
|  stability. It is not just a hardware parameter—it        |
|  shapes the loss landscape your network traverses.        |
|  Choose batch size wisely; it influences not just how     |
|  fast you learn, but what you learn.                      |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a dataset with 10,000 examples. On your GPU, batch size 32 gives 200 iterations per second. Batch size 256 gives 400 iterations per second.

**Question:** Which batch size processes more examples per second? Which batch size will likely achieve better generalization? Why?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Examples per second:
Batch 32: 32 × 200 = 6400 examples/sec
Batch 256: 256 × 400 = 102,400 examples/sec
Batch 256 processes 16× more examples per second.

Generalization: Batch 32 likely generalizes better. The smaller batch adds noise that helps escape sharp minima. Batch 256 may overfit to training data. However, for LLMs with very large datasets, the generalization gap diminishes, and larger batches are used for throughput. For this moderate-sized dataset (10K), smaller batch is likely better for test accuracy despite lower throughput.
