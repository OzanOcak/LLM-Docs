# Batch normalization

## **DOMAIN: DEEP LEARNING FOUNDATIONS | Sub domain: Training Neural Networks**

---

### **1. Why this concept matters for building intelligent systems**

Deep networks are hard to train. Activations can explode or vanish. Gradients can become tiny. The distribution of inputs to each layer shifts constantly as earlier layers change—a problem called internal covariate shift. Batch normalization solves this by normalizing layer inputs to have zero mean and unit variance before each activation. This stabilizes training, allows much higher learning rates, and acts as a regularizer. Without batch norm, deep networks required careful initialization and low learning rates. With it, you can train networks 10x deeper and 10x faster. Understanding batch norm means understanding one of the key innovations that made deep learning practical.

---

### **2. Core idea**

**Batch normalization normalizes the inputs of each layer across a mini-batch to have zero mean and unit variance, then applies learned scale and shift parameters, stabilizing training and enabling much higher learning rates.**

---

### **3. Concrete analogy**

Imagine you are a piano teacher with students at different skill levels. Each student's fingers move at different speeds and ranges. Teaching them with the same instructions is chaotic.

Your solution: before each lesson, you normalize every student's hand position to a standard reference. You measure their current position, subtract the average (center them), and divide by their variation (scale them). Now all students have the same posture baseline. Then you apply personalized adjustments (learned shift and scale) to account for individual anatomy.

This normalization happens before every lesson, not just once. As students improve, their raw positions change, but you re-normalize each time. The lessons become stable and efficient.

Batch normalization does this for neural networks. Each layer's inputs are normalized to a standard distribution before the activation function. Then the network learns two parameters per neuron: a scale (γ) and shift (β) to allow the layer to express any distribution if needed.

---

### **4. ASCII diagram**

```text
Batch normalization applied to one layer:

    Input batch (B examples, d features)
    ┌─────────────────────────────┐
    │ x₁₁  x₁₂  x₁₃  ...  x₁d    │
    │ x₂₁  x₂₂  x₂₃  ...  x₂d    │
    │ ...                        │
    │ xB₁  xB₂  xB₃  ...  xBd    │
    └─────────────────────────────┘
           │
           ▼
    Normalize across batch (per feature)

    For each feature j:
    μ_j = (1/B) Σ_i x_ij
    σ_j² = (1/B) Σ_i (x_ij - μ_j)²
    x̂_ij = (x_ij - μ_j) / √(σ_j² + ε)
           │
           ▼
    Scale and shift (learned parameters)

    y_ij = γ_j × x̂_ij + β_j
           │
           ▼
    Pass to activation function


Batch norm forward pass:

    Input x (B×d)
       │
       ├──→ Compute μ (d), σ² (d) across batch dimension
       │
       ├──→ Normalize: x̂ = (x - μ) / √(σ² + ε)
       │
       └──→ Scale & shift: y = γ ⊙ x̂ + β

       Output y (B×d)


At inference time (using running averages):

    Use μ_running and σ²_running (exponential moving average from training)
    No recomputation per batch
    y = γ ⊙ (x - μ_running) / √(σ²_running + ε) + β
```

---

### **5. Mathematical formulation**

**Batch normalization forward pass (training):**

For a batch of size B, for each feature dimension j:

Batch mean:

$$
\mu_j = \frac{1}{B} \sum_{i=1}^B x_{ij}
$$

Batch variance:

$$
\sigma_j^2 = \frac{1}{B} \sum_{i=1}^B (x_{ij} - \mu_j)^2
$$

Normalize:

$$
\hat{x}_{ij} = \frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

Scale and shift (learnable parameters γ_j, β_j):

$$
y_{ij} = \gamma_j \hat{x}_{ij} + \beta_j
$$

Where ε is a small constant (e.g., 1e-5) for numerical stability.

**Backpropagation through batch norm:**

Gradients flow through μ, σ², x̂, and γ,β. Requires computing ∂L/∂μ and ∂L/∂σ², then propagating to ∂L/∂x.

**Running statistics for inference:**

During training, maintain exponential moving average:

$$
\mu_{\text{running}} = \alpha \mu_{\text{running}} + (1-\alpha) \mu_{\text{batch}}
$$

$$
\sigma_{\text{running}}^2 = \alpha \sigma_{\text{running}}^2 + (1-\alpha) \sigma_{\text{batch}}^2
$$

α typical = 0.9 or 0.99.

At inference, use μ_running and σ_running² (no batch statistics).

**Effect on gradient flow:**

Batch norm reduces internal covariate shift, allowing higher learning rates. Also has a regularizing effect (adds noise via batch statistics).

**Placement in network (pre-activation vs post-activation):**

Original BN paper: applied before activation (x after Wx+b). Later work (Pre-activation ResNet) showed BN after activation sometimes better.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Input batch**

Batch size B=3, one feature dimension (simplified). Input values: x = [2.0, 4.0, 6.0]

#### **Step 2: Compute batch mean**

μ = (2 + 4 + 6)/3 = 12/3 = 4.0

#### **Step 3: Compute batch variance**

σ² = [(2-4)² + (4-4)² + (6-4)²]/3 = [4 + 0 + 4]/3 = 8/3 = 2.6667

#### **Step 4: Normalize (ε=0 for simplicity)**

x̂₁ = (2 - 4)/√2.6667 = (-2)/1.633 = -1.225
x̂₂ = (4 - 4)/1.633 = 0
x̂₃ = (6 - 4)/1.633 = 2/1.633 = 1.225

Check: mean of x̂ = (-1.225 + 0 + 1.225)/3 = 0, variance = 1.0 ✓

#### **Step 5: Learnable scale and shift**

Assume γ = 0.5, β = 1.0

y₁ = 0.5 × (-1.225) + 1.0 = -0.6125 + 1.0 = 0.3875
y₂ = 0.5 × 0 + 1.0 = 1.0
y₃ = 0.5 × 1.225 + 1.0 = 0.6125 + 1.0 = 1.6125

#### **Step 6: Without batch norm (no normalization)**

Raw x = [2, 4, 6] would go directly to activation. Distribution is uncentered with variance 2.67.

With batch norm, distribution centered at 0 (before γ,β) and unit variance. After γ,β, network controls mean and variance.

#### **Step 7: Inference with running statistics**

After training, running mean = 4.2, running variance = 2.8.

New test input x=5.0:
x̂ = (5.0 - 4.2)/√2.8 = 0.8/1.673 = 0.478
ŷ = 0.5 × 0.478 + 1.0 = 0.239 + 1.0 = 1.239

No batch statistics computed during inference.

---

### **7. How this appears inside neural networks and LLMs**

- **CNN architectures:** Batch norm is standard in ResNet, Inception, and most CNNs. Placed after convolution, before activation (Conv → BN → ReLU).

- **Transformers use Layer Norm, not Batch Norm:** Layer norm normalizes across features (per example) not across batch. Why? Batch norm fails with variable sequence lengths and small batches (common in LLMs). Layer norm is the default in transformers.

- **Batch size limitation:** Batch norm requires sufficiently large batch (e.g., ≥16) for stable statistics. For small batches or distributed training with tiny per-GPU batches, batch norm degrades. Use sync BN (sync across GPUs) or layer norm.

- **Group normalization:** Alternative between batch and layer norm. Divides channels into groups, normalizes within group. Works well for small batches.

- **Instance normalization:** Normalizes per example, per channel (used in style transfer).

- **Regularization effect:** Batch norm adds noise (batch statistics vary with sampling). This acts like dropout, sometimes allowing removal of dropout.

- **Frozen batch norm during fine-tuning:** When fine-tuning, sometimes freeze batch norm running statistics (use pretrained μ,σ). Prevents statistics shift on small fine-tuning dataset.

---

### **8. Brain-like connection (gain control)**

The brain performs normalization continuously. Retinal ganglion cells adapt to mean light level (subtract mean) and contrast (divide by standard deviation). This is batch normalization in early vision. Cortical neurons also normalize their inputs: the firing rate of a neuron is often divided by the sum of activity in surrounding neurons (divisive normalization). This keeps neural activity within a dynamic range, prevents saturation, and enhances contrast. The brain's normalization is local (not across a batch of images) but serves the same purpose: stable, efficient processing across varying input conditions.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Batch normalization eliminates the need for careful initialization. I can use any initialization and it will work."

_Why it is wrong:_ Batch norm greatly reduces sensitivity to initialization but does not eliminate it. Extremely poor initialization (e.g., all weights zero) still breaks batch norm (all activations same → μ=x, σ=0, division by zero). Also, batch norm does not fix exploding/vanishing gradients from very deep networks—skip connections (ResNets) are still needed for extreme depth. Use good initialization (e.g., He or Xavier) even with batch norm.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Before batch norm, deep networks were fragile. Learning  |
|  rates had to be tiny. Initialization had to be perfect.  |
|  Batch norm changed everything. It stabilized training,   |
|  enabled 10x higher learning rates, and made deep         |
|  networks practical. Every modern CNN uses it. Even       |
|  transformers, which prefer layer norm, owe their         |
|  existence to the normalization revolution batch norm     |
|  started. Normalize your layers—your gradients will thank  |
|  you, your convergence will accelerate, and your network  |
|  will finally become stable.                              |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are training a ResNet on ImageNet with batch size 32. A colleague suggests switching to batch size 4 to fit a larger model on the same GPU memory.

**Question:** What problem might occur with batch size 4 when using batch normalization? What alternative could you use instead?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Batch size 4 is very small for batch norm. The batch statistics (μ and σ²) will be noisy—poor estimates of the true population statistics. This noise can destabilize training and hurt convergence.

Alternatives:

- Use sync batch norm (aggregate statistics across multiple GPUs) to get effective batch size 4×num_gpus.
- Use layer normalization (normalizes across features, not batch) instead of batch norm.
- Use group normalization (compromise between batch and layer norm).
- Use larger batch size by gradient accumulation (simulate larger batch without increasing memory).

For ImageNet, batch size 4 is generally too small for batch norm. Many practitioners use batch size 32-256 for stable batch norm statistics.
