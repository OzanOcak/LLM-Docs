# Regularization (dropout, L1/L2)

## **DOMAIN: DEEP LEARNING FOUNDATIONS | Sub domain: Training Neural Networks**

---

### **1. Why this concept matters for building intelligent systems**

A powerful network can memorize the training data perfectly. But memorization is not intelligence—it is overfitting. Regularization is the set of techniques that prevent memorization, encouraging the network to learn general patterns instead of noise. L1 and L2 regularization penalize large weights, forcing the network to distribute importance across many features. Dropout randomly turns off neurons during training, forcing each neuron to be useful on its own. Without regularization, deep networks fail on new data. With it, they generalize. Understanding regularization means understanding how to keep your network humble and robust.

---

### **2. Core idea**

**Regularization adds constraints or noise to the training process to reduce overfitting, encouraging simpler models that generalize better to unseen data.**

---

### **3. Concrete analogy**

Imagine two students preparing for an exam with 100 practice questions.

- **No regularization**: The student memorizes every practice question, including typos and odd phrasing. On the real exam (new questions), the student performs poorly because the actual concepts were never learned.

- **L2 regularization**: The student is penalized for relying too heavily on any single study source. They must use all materials, distributing their attention. This prevents overconfidence in any one source.

- **L1 regularization**: The student is forced to choose only the most important study sources, ignoring irrelevant ones entirely. Creates a sparse study plan.

- **Dropout**: Before each study session, the student is randomly forbidden from using 50% of their notes. They must learn to answer questions using only the remaining notes. The next session, a different 50% is forbidden. By the end, every fact has been practiced without relying on any single crutch.

The regularized student understands concepts. The unregularized student memorizes answers.

---

### **4. ASCII diagram**

```text
L1 vs L2 regularization geometry:

    L1 (Lasso) - creates sparsity        L2 (Ridge) - shrinks weights

        w₂ ↑                                 w₂ ↑
          │   ◇                                 │
          │  ╱ ╲                                │    ○
          │ ╱   ╲                               │   ╱ ╲
          │╱     ╲                              │  ╱   ╲
          │       ╲                             │ ╱     ╲
          │        ◇                            │╱       ╲
          │                                     └──────────→ w₁
          └──────────→ w₁                    L2 ball (circle)
    L1 ball (diamond)

    L1 encourages weights on axes (some zero). L2 encourages all weights small but nonzero.


Dropout during training vs inference:

    Training (dropout mask)          Inference (full network)

    Input ──●──●──●──●           Input ──●──●──●──●
           │  │  │  │                  │  │  │  │
        ┌──┴──┴──┴──┴──┐           ┌──┴──┴──┴──┴──┐
        │ ✗  ✓  ✗  ✓   │           │ ✓  ✓  ✓  ✓   │  (all active)
        │  Dropout mask│           │              │
        │  (random 0/1)│           │  Scale by    │
        └──┬──┬──┬──┬──┘           │  keep_prob   │
           │  │  │  │              └──┬──┬──┬──┬──┘
          h₁ h₂ h₃ h₄                 h₁ h₂ h₃ h₄

    At inference, multiply by keep_prob (e.g., 0.5)
    to match expected training activation.
```

---

### **5. Mathematical formulation**

**L2 regularization (Ridge, weight decay):**

Adds penalty proportional to squared weights.

$$
L_{\text{total}}(\theta) = L_{\text{data}}(\theta) + \frac{\lambda}{2} \|\theta\|_2^2 = L_{\text{data}}(\theta) + \frac{\lambda}{2} \sum_j \theta_j^2
$$

Gradient: ∇L_total = ∇L_data + λθ

Weight decay interpretation: θ ← θ - η∇L_data - ηλθ = (1 - ηλ)θ - η∇L_data

**L1 regularization (Lasso):**

Adds penalty proportional to absolute weights.

$$
L_{\text{total}}(\theta) = L_{\text{data}}(\theta) + \lambda \|\theta\|_1 = L_{\text{data}}(\theta) + \lambda \sum_j |\theta_j|
$$

Gradient: ∇L_total = ∇L_data + λ × sign(θ)

L1 produces sparse solutions (many weights exactly zero).

**Elastic Net (combines L1 and L2):**

$$
L_{\text{total}} = L_{\text{data}} + \lambda_1 \|\theta\|_1 + \frac{\lambda_2}{2} \|\theta\|_2^2
$$

**Dropout:**

During training, for each neuron with probability p (dropout rate), set output to 0:

$$
h_{\text{dropout}} = h \odot \mathbf{m}, \quad m_j \sim \text{Bernoulli}(1-p)
$$

During inference, use all neurons but scale outputs by (1-p):

$$
h_{\text{test}} = (1-p) \times h
$$

Or equivalently, scale training activations by 1/(1-p) (inverted dropout).

**Dropout as ensemble:** Equivalent to training exponentially many subnetworks and averaging at test time.

**Early stopping (implicit regularization):** Stop training before convergence. Limits model complexity.

**Batch normalization (has regularizing effect):** Adds noise via batch statistics, similar to dropout.

**Data augmentation (regularization via data):** Create modified copies of training data (rotate, flip, crop, add noise).

---

### **6. Worked example (step-by-step)**

#### **Step 1: Without regularization**

Linear regression on 5 points: True function y = 2x. But we fit degree 10 polynomial (overparameterized).

Training loss: 0.0 (perfect fit to noise)
Test loss: 5.2 (terrible generalization)

#### **Step 2: With L2 regularization (λ=0.01)**

Minimize L = MSE + 0.01 × Σ w_i²

Large weights penalized. Forces all coefficients to be small.

Training loss: 0.5 (not perfect, good)
Test loss: 1.2 (much better)

#### **Step 3: With L1 regularization (λ=0.01)**

Minimize L = MSE + 0.01 × Σ |w_i|

Many coefficients become exactly zero. Only the most important features survive.

Result: 7 out of 10 polynomial coefficients = 0. Sparse model.

Training loss: 0.8, Test loss: 1.5 (sparse, interpretable)

#### **Step 4: Dropout example**

Hidden layer with 4 neurons, dropout rate p=0.5.

Training forward pass:
h = [2.0, 3.0, 1.5, 0.5]
Dropout mask m (Bernoulli with p=0.5): [0, 1, 1, 0]
h_dropout = [0, 3.0, 1.5, 0]

Inference forward pass (scale by 1-p=0.5):
h_test = [2.0×0.5, 3.0×0.5, 1.5×0.5, 0.5×0.5] = [1.0, 1.5, 0.75, 0.25]

Expected training activation (before dropout) = [2.0, 3.0, 1.5, 0.5] × 0.5 = same as inference.

---

### **7. How this appears inside neural networks and LLMs**

- **Weight decay (L2) in AdamW:** Standard practice for LLMs. Default λ = 0.01 to 0.1. Prevents weights from growing too large.

- **Dropout in transformers:** Applied after attention and feed-forward layers. Typical rates: 0.1 (BERT), 0.2 (GPT-2). Prevents co-adaptation of attention heads.

- **Attention dropout:** Dropout applied to attention weights (after softmax). Forces model to distribute attention across multiple positions.

- **Embedding dropout:** Randomly zero out token embeddings during training. Acts as noise in the input space.

- **Stochastic depth (dropout on layers):** Randomly skip entire transformer blocks during training. Forces each block to be independently useful.

- **L1 regularization for pruning:** Not common in LLMs (sparsity not well-supported on GPUs), but used for model compression in deployment.

- **Label smoothing (regularization on targets):** Instead of one-hot labels (0 or 1), use (1-ε, ε/(K-1)). Prevents overconfidence. Common in LLM training.

- **Gradient clipping (regularization on optimization):** Caps gradient norm. Prevents exploding gradients, acts as implicit regularization.

---

### **8. Brain-like connection (synaptic homeostasis)**

The brain regulates connection strengths to prevent runaway excitation (L2-like regularization). Synaptic scaling keeps total excitatory input balanced. The brain also performs dropout-like computation: not all neurons fire on every stimulus. Sparse coding—only a small fraction of neurons active at any time—is a form of L1 regularization that saves energy. During sleep, the brain down-scales synaptic strengths (weight decay), preventing unlimited growth. These homeostatic mechanisms keep neural circuits stable and generalizable, just as regularization keeps artificial networks from overfitting.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Dropout and L2 regularization both prevent overfitting, so I should use maximum regularization possible."

_Why it is wrong:_ Too much regularization causes underfitting. If λ is too large, L2 forces all weights near zero—model becomes constant (high bias). If dropout rate is too high (e.g., 0.9), the network cannot learn—too much noise, gradients vanish. Regularization strength must be tuned via validation set. The goal is to reduce overfitting without harming the model's capacity to learn the true signal. Regularization is a dial, not a switch. Start with small λ (e.g., 1e-5) and increase until validation loss stops improving.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Unregularized networks are savants—they memorize but     |
|  do not understand. Regularization is the humility        |
|  constraint. L2 says: don't trust any single weight too   |
|  much. L1 says: focus only on what matters, ignore the    |
|  rest. Dropout says: learn to function even when parts    |
|  of you are missing. Together, they force your network    |
|  to learn robust, generalizable patterns instead of       |
|  brittle, specific ones. Regularization is not a luxury.  |
|  It is the difference between memorization and wisdom.    |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a neural network with L2 regularization (λ=0.01) and dropout (p=0.5). After training, the model works well on validation data. A colleague says: "Turn off dropout during final training to get better performance."

**Question:** Is your colleague correct? What happens if you train with dropout off but keep L2?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Your colleague is likely incorrect. Training with dropout off changes the training dynamics. Dropout is not just a final step—it fundamentally changes the learning process. Without dropout during training, the network will co-adapt and likely overfit. Even with L2, the network may not generalize as well because dropout forces redundant representations.

If you want to train without dropout, you should re-train from scratch with dropout off and the same L2. But for best results, keep dropout on during training, then use the trained weights (scaled) for inference. Many practitioners mistakenly think dropout is just a "regularization add-on" that can be removed after training. It cannot. The network learned under dropout; removing it changes expected activations. Use inverted dropout (which scales at training time) to make inference match.
