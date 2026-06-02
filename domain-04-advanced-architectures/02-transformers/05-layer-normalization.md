# Layer normalization

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Transformers (The LLM Revolution)**

---

### **1. Why this concept matters**

Transformers are deep—dozens or hundreds of layers. Without normalization, activations grow or shrink uncontrollably, gradients vanish or explode, and training collapses. Batch normalization works for CNNs but fails for transformers because sequence lengths vary and batch sizes are often small. Layer normalization is the solution: it normalizes across features for each token independently, not across the batch. This stabilizes training, enables much deeper networks, and is essential to every modern LLM. Without layer norm, there is no transformer.

---

### **2. Core idea**

**Layer normalization normalizes the activations of a single example across all features, computing mean and variance per token, then applying learned scale and shift parameters, making training stable regardless of batch size or sequence length.**

---

### **3. Concrete analogy**

Imagine you are a teacher grading essays. Each student (token) submits an essay with 100 dimensions (grammar, vocabulary, argument strength, etc.). The raw scores vary wildly: one student scores 90 in grammar but 10 in creativity; another scores 50 in everything.

Grading is hard because the scales are inconsistent. Your solution: for each student individually, you rescale their scores to have average 0 and standard deviation 1, then apply a personalized curve (learned scale and shift). Now you can compare fairly across dimensions.

This is layer norm. Each token (student) is normalized independently using its own statistics (mean and variance across its feature dimensions). This works for any sequence length, any batch size, and does not require other tokens for statistics.

---

### **4. ASCII diagram**

```
Layer normalization (applied to one token):

    Input token x (d_model dimensions)

    x = [x₁, x₂, x₃, ..., x_d]

    Step 1: Compute mean μ across dimensions
    μ = (x₁ + x₂ + ... + x_d) / d

    Step 2: Compute variance σ² across dimensions
    σ² = ((x₁-μ)² + (x₂-μ)² + ... + (x_d-μ)²) / d

    Step 3: Normalize
    x̂ᵢ = (xᵢ - μ) / √(σ² + ε)  for all i

    Step 4: Scale and shift (learned parameters γ, β)
    yᵢ = γᵢ × x̂ᵢ + βᵢ  for all i

    Output: normalized token y (same shape as x)


Comparison of normalization types:

    Batch Norm (CNNs)          Layer Norm (Transformers)

    Normalize across batch     Normalize across features
    for each feature           for each example

    x: [batch, features]       x: [batch, features]
    μ over batch dim           μ over features dim
    Requires large batch       Works with batch size 1
    Broken for variable length Works for any sequence length


Placement in transformer block (two variants):

    Post-LN (original):        Pre-LN (modern, more stable):

    x → Attn → Add → Norm      x → Norm → Attn → Add
         ↓                            ↓
         Norm → FFN → Add             x (residual)
              ↓                         ↓
              Norm                      Norm → FFN → Add
                                       (repeat pattern)

    Pre-LN has better gradient flow for deep transformers.
```

---

### **5. Mathematical formulation**

**Layer normalization for a single token x ∈ ℝ^d:**

Mean:

$$
\mu = \frac{1}{d} \sum_{i=1}^d x_i
$$

(Unicode: μ = (1/d) Σ x_i)

Variance:

$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2
$$

(Unicode: σ² = (1/d) Σ (x_i - μ)²)

Normalize:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

(Unicode: x̂_i = (x_i - μ) / √(σ² + ε))

Scale and shift (learnable per-dimension parameters):

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

(Unicode: y_i = γ_i × x̂_i + β_i)

Where:

- γ, β ∈ ℝ^d are learned (different per feature dimension)
- ε is a small constant (e.g., 1e-5) for numerical stability
- No batch statistics. Statistics computed per token.

**For a batch of tokens (shape: n × d):**

Compute independently for each token. No interaction across tokens.

**Pre-LN vs Post-LN residual connection:**

Post-LN (original transformer):

$$
\mathbf{x} = \text{LayerNorm}(\mathbf{x} + \text{Attention}(\mathbf{x}))
$$

Pre-LN (modern, more stable):

$$
\mathbf{x} = \mathbf{x} + \text{Attention}(\text{LayerNorm}(\mathbf{x}))
$$

Pre-LN ensures gradients flow through the skip connection without passing through normalization, enabling much deeper transformers.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Input token**

A token with d=4 dimensions:
x = [2.0, -1.0, 3.0, 0.5]

#### **Step 2: Compute mean**

μ = (2.0 + (-1.0) + 3.0 + 0.5) / 4 = (4.5) / 4 = 1.125

#### **Step 3: Compute variance**

(x - μ): [0.875, -2.125, 1.875, -0.625]
Squares: [0.7656, 4.5156, 3.5156, 0.3906]
Sum = 0.7656 + 4.5156 + 3.5156 + 0.3906 = 9.1874
σ² = 9.1874 / 4 = 2.29685
σ = √2.29685 ≈ 1.5155

#### **Step 4: Normalize (ε=0 for simplicity)**

x̂₁ = (2.0 - 1.125) / 1.5155 = 0.875 / 1.5155 = 0.577
x̂₂ = (-1.0 - 1.125) / 1.5155 = -2.125 / 1.5155 = -1.402
x̂₃ = (3.0 - 1.125) / 1.5155 = 1.875 / 1.5155 = 1.237
x̂₄ = (0.5 - 1.125) / 1.5155 = -0.625 / 1.5155 = -0.412

x̂ = [0.577, -1.402, 1.237, -0.412]

Check: mean ≈ 0, variance ≈ 1 ✓

#### **Step 5: Scale and shift (learned γ, β)**

Assume learned: γ = [1.0, 0.5, 2.0, 0.8], β = [0.1, 0.0, -0.2, 0.05]

y₁ = 1.0×0.577 + 0.1 = 0.677
y₂ = 0.5×(-1.402) + 0.0 = -0.701
y₃ = 2.0×1.237 + (-0.2) = 2.474 - 0.2 = 2.274
y₄ = 0.8×(-0.412) + 0.05 = -0.330 + 0.05 = -0.280

Final normalized token: y = [0.677, -0.701, 2.274, -0.280]

#### **Step 6: Interpret**

The token's activations now have controlled scale (~range -0.7 to 2.3) instead of original range (-1 to 3). The γ parameters have learned to emphasize dimension 3 (γ=2.0) and de-emphasize dimension 2 (γ=0.5). The network learns which features matter.

---

### **7. How this appears inside neural networks and LLMs**

- **Every transformer block:** Contains at least two layer norms (one after attention, one after feed-forward). Pre-LN transformers (most modern) have norm before each sub-layer.

- **Pre-LN vs Post-LN:** Original BERT used Post-LN (norm after residual addition). Modern LLMs (GPT-3, Llama, PaLM) use Pre-LN (norm before). Pre-LN has better gradient flow, enables deeper networks, and requires no learning rate warmup.

- **RMSNorm (Root Mean Square Normalization):** Simplified variant used in Llama, PaLM. Removes mean centering (only scales by RMS). Computationally cheaper and performs similarly.

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}} \odot \gamma
$$

- **No bias after layer norm:** Since layer norm already centers activations, the subsequent linear layers often omit bias terms (saves parameters).

- **Stable training:** Layer norm keeps activations in a reasonable range regardless of initialization or depth. Without it, 100-layer transformers would be impossible.

- **Position independence:** Each token normalized independently. Same token at different positions gets different normalization (because other features differ), preserving positional information.

---

### **8. Brain-like connection (homeostatic plasticity)**

The brain maintains stable firing rates through homeostatic plasticity. When a neuron fires too frequently, synaptic strengths are scaled down; when too rarely, scaled up. This keeps neural activity within a dynamic range despite varying inputs. Layer normalization is the artificial analogue: it rescales activations to have stable mean and variance, preventing saturation or vanishing. The brain's homeostatic mechanisms operate on longer timescales (hours to days), while layer norm operates instantly per token. Both solve the same problem: maintaining stable information flow through a processing system with many layers.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Layer normalization and batch normalization are interchangeable. Either works for transformers."

_Why it is wrong:_ Batch normalization fails for transformers. Reasons:

- Sequence lengths vary; batch norm statistics become noisy or invalid.
- Small batches (common in LLM training due to memory constraints) produce unstable statistics.
- During autoregressive generation (decoding), tokens are processed one at a time—batch norm would have no batch to compute statistics over.
- Layer norm works with any batch size (including 1) and any sequence length.

Layer norm is not just as good—it is the _only_ choice that works robustly for transformers. This is why every LLM uses layer norm or its variants (RMSNorm), never batch norm.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Deep networks are unstable by nature. Activations drift, |
|  gradients vanish, training collapses. Layer normalization|
|  is the stabilizer. It anchors each token's activations   |
|  to a consistent scale, independent of other tokens.      |
|  Without it, transformers would be too fragile to train.  |
|  With it, we can stack 100 layers, 1 trillion parameters, |
|  and train on the entire internet. Layer norm is the      |
|  silent enabler of the LLM revolution.                    |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a transformer with 32 layers using Post-LN (norm after residual addition). Training is unstable—gradients vanish in early layers. A colleague suggests switching to Pre-LN. Why would this help?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** In Post-LN, the residual path (x + Attention(x)) flows through layer normalization before the next residual. The gradient must pass through the normalization's derivative, which can be small. In Pre-LN, the normalization is applied _before_ the attention/FFN, and the residual path is x + Attention(Norm(x)). The skip connection preserves the gradient from higher layers without passing through normalization. This creates a direct gradient highway. In very deep transformers, Pre-LN ensures gradients from the loss can reach early layers without vanishing. Modern LLMs (GPT-3, Llama) use Pre-LN exclusively for this reason.
