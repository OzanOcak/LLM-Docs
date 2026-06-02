# Feed-forward layers

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Transformers (The LLM Revolution)**

---

### **1. Why this concept matters**

Self-attention is the star of the transformer. But attention alone is not enough. Attention mixes information across positions—it tells each token what to look at. But after looking, the token needs to _process_ that information: transform it, combine it with its own state, and prepare for the next layer. That is the feed-forward network's job. It is a simple two-layer MLP applied identically to every position. Without it, transformers would just be fancy weighted averages. Feed-forward layers add the nonlinear computation that turns attended information into higher-level representations—the "thinking" after the "looking."

---

### **2. Core idea**

**The feed-forward layer in a transformer is a position-wise two-layer MLP that applies the same nonlinear transformation independently to each token's representation, expanding and then contracting the dimension to enable complex feature interactions.**

---

### **3. Concrete analogy**

Imagine a team of analysts at a meeting. Each analyst has a notepad (their token representation). During the meeting (attention), they look around the room, gather information from colleagues, and jot down notes. But raw notes are not a final decision.

After the meeting, each analyst retires to their office (feed-forward layer) and processes their notes alone. They expand their thinking (project to a higher dimension: 4x the space), consider many possible interpretations, then distill down (project back) to a refined conclusion. Crucially, each analyst works independently—no cross-communication during this phase. The same transformation (same weights) applies to every analyst, but each starts from different attended information.

This pattern—attend (mix across positions), then think (transform per position)—repeats in every transformer block. The feed-forward layer is where the computation happens.

---

### **4. ASCII diagram**

```
Transformer block with feed-forward layer:

    Input (from previous layer)    X (n × d_model)
                │
                ▼
        ┌───────────────┐
        │   Multi-Head  │   ← Attention (mixes positions)
        │   Attention   │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │   Add & Norm  │   ← Residual + LayerNorm
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │  Feed-Forward │   ← Position-wise MLP
        │    Network    │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │   Add & Norm  │   ← Residual + LayerNorm
        └───────────────┘
                │
                ▼
          Output to next block


Feed-forward layer internals:

    Input token x (d_model)
           │
           ▼
    Linear 1: x × W₁ + b₁    (d_model → d_ff, typically 4×d_model)
           │
           ▼
    Activation: GELU(x) or ReLU(x)
           │
           ▼
    Linear 2: x × W₂ + b₂    (d_ff → d_model)
           │
           ▼
         Output token (d_model)


Position-wise: same weights W₁, W₂ applied to every token independently.
```

---

### **5. Mathematical formulation**

**Feed-forward layer (position-wise):**

For each token i in the sequence:

$$
\text{FFN}(\mathbf{x}_i) = \mathbf{W}_2 \cdot \text{Activation}(\mathbf{W}_1 \mathbf{x}_i + \mathbf{b}_1) + \mathbf{b}_2
$$

(Unicode: FFN(x_i) = W₂ × activation(W₁ × x_i + b₁) + b₂)

Where:

- x_i ∈ ℝ^{d_model} (token representation)
- W₁ ∈ ℝ^{d_ff × d_model}, W₂ ∈ ℝ^{d_model × d_ff}
- d_ff is typically 4×d_model (e.g., 3072 for d_model=768, 49152 for d_model=12288)
- b₁ ∈ ℝ^{d_ff}, b₂ ∈ ℝ^{d_model}

**Activation functions (choices):**

ReLU (original transformer):

$$
\text{ReLU}(x) = \max(0, x)
$$

GELU (BERT, GPT-2, GPT-3, modern LLMs):

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))
$$

Swish / SiLU (some recent models):

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
\]

**GLU variants (Gated Linear Units, used in T5, PaLM, Llama):**


$$

\text{FFN}\_{\text{GLU}}(\mathbf{x}) = (\mathbf{W}\_2 \cdot \text{Activation}(\mathbf{W}\_1 \mathbf{x} + \mathbf{b}\_1)) \odot (\mathbf{W}\_3 \mathbf{x} + \mathbf{b}\_3)

$$

**Parameter count for FFN (excluding GLU variants):**


$$

\text{params} = (d*{\text{model}} \times d*{\text{ff}}) + d*{\text{ff}} + (d*{\text{ff}} \times d*{\text{model}}) + d*{\text{model}} \approx 2 \times d*{\text{model}} \times d*{\text{ff}}

$$

For d_ff = 4×d_model: ≈ 8 × d_model² parameters. In BERT-base (d_model=768, d_ff=3072), each FFN has about 2×768×3072 ≈ 4.7M parameters—dominating the transformer's parameter count.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Setup**

d_model = 4, d_ff = 8 (2× expansion for illustration; real transformers use 4×)

Token embedding x = [1.0, -0.5, 0.2, 0.8]

#### **Step 2: Linear layer 1 (W₁: 8×4, b₁: 8)**

W₁ (simplified, showing first 2 rows):
Row1: [0.5, 0.2, -0.1, 0.3] → [0.5×1 + 0.2×(-0.5) + (-0.1)×0.2 + 0.3×0.8] = [0.5 -0.1 -0.02 +0.24] = 0.62
Row2: [0.1, 0.4, 0.3, -0.2] → [0.1 -0.2 +0.06 -0.16] = -0.20
(Continue for rows 3-8)

z₁ = W₁·x + b₁ = [0.62, -0.20, 0.35, -0.05, 0.18, 0.45, -0.30, 0.12] (8-dimensional)

#### **Step 3: Activation (GELU)**

GELU approximation for each element:
GELU(0.62) ≈ 0.5×0.62×(1 + tanh(0.797×(0.62+0.0447×0.62³))) ≈ 0.45
GELU(-0.20) ≈ -0.08
(Compute for all 8 dims)

h = [0.45, -0.08, 0.25, -0.02, 0.13, 0.33, -0.12, 0.08]

#### **Step 4: Linear layer 2 (W₂: 4×8, b₂: 4)**

W₂ (4×8 matrix, each row maps 8-dim to 1-dim):
Simplified: first row [0.6, 0.3, 0.1, 0.2, 0.4, 0.5, 0.1, 0.3]

Output dim1 = 0.6×0.45 + 0.3×(-0.08) + 0.1×0.25 + 0.2×(-0.02) + 0.4×0.13 + 0.5×0.33 + 0.1×(-0.12) + 0.3×0.08 = 0.27 -0.024 +0.025 -0.004 +0.052 +0.165 -0.012 +0.024 = 0.496

Similarly for remaining 3 output dimensions.

Output = [0.50, 0.12, -0.08, 0.33] (4-dimensional, back to d_model)

#### **Step 5: Interpret**

Input x = [1.0, -0.5, 0.2, 0.8] → FFN output = [0.50, 0.12, -0.08, 0.33]

The token representation has been transformed. The large hidden dimension (8 vs 4) allowed complex nonlinear mixing of features. The output is now ready for the next transformer block.

---

### **7. How this appears inside neural networks and LLMs**

- **Every transformer block:** Each block has exactly one FFN after the attention. BERT-base has 12 FFNs, GPT-3 has 96 FFNs.

- **Parameter dominance:** FFNs account for 2/3 to 3/4 of total parameters in transformers. Attention heads are parameter-light; FFNs are parameter-heavy.

- **Sparsity (ReLU/GELU):** Many activations become zero (ReLU) or near-zero (GELU). This sparsity can be exploited for efficient inference (MoE).

- **Mixture of Experts (MoE):** Replace a single FFN with multiple parallel FFNs (experts), plus a router that selects top-k experts per token. Enables massive parameter counts (e.g., Mixtral 8×7B, GPT-4 rumored MoE).

- **GLU variants:** Llama, PaLM, and many modern LLMs use GLU (Gated Linear Unit) variants (e.g., SwiGLU) instead of simple two-layer MLP. More parameters but better performance.

- **Position-wise independence:** The FFN applies the same weights to every token. This is a form of weight sharing across positions, reducing parameters compared to per-position MLPs.

- **What does the FFN learn?** Interpretability research suggests FFN neurons specialize: some detect specific patterns (e.g., "negation," "past tense"), others act as key-value memories.

---

### **8. Brain-like connection (cortical columns and local computation)**

In the brain's cortex, information flows through layers. Superficial layers handle local computation (like FFNs), while deeper layers integrate across columns (like attention). The separation of attention (mixing) and feed-forward (computation) mirrors the brain's columnar architecture. Each cortical column performs local nonlinear transformations on its input—analogous to a FFN—while long-range connections (attention-like) allow columns to communicate. The FFN's expansion to 4× dimension resembles the brain's expansion from thalamic input (few fibers) to cortical columns (many neurons) back to output. Evolution discovered that local computation requires overrepresentation before compression; transformers independently arrived at the same principle.

---

### **9. Common misunderstanding and why it is wrong**

*Misunderstanding:* "The feed-forward layer is just a simple MLP. It is not as important as attention."

*Why it is wrong:* While simpler in concept, FFNs dominate transformer parameters and compute. In BERT-base, FFNs account for ~66% of parameters (about 70M of 110M total). Removing FFNs collapses transformer performance to near-random. Attention is for communication; FFNs are for computation. Neither works without the other. The transformer's power comes from the interplay: attention gathers relevant context, then FFNs transform each token's representation based on that context. Skimp on FFN dimension (e.g., d_ff = 2×d_model instead of 4×) and performance drops significantly.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Attention connects. Feed-forward computes. Every         |
|  transformer block alternates between looking around      |
|  (attention) and thinking alone (FFN). The FFN is where   |
|  the neural network does its nonlinear work—expanding     |
|  into a high-dimensional space, activating sparse         |
|  features, and compressing back. Without it, attention    |
|  is just a fancy averager. Without attention, the FFN     |
|  has no context to compute on. Together, they form the    |
|  complete transformer.                                    |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

BERT-base has d_model=768 and d_ff=3072. GPT-3 has d_model=12288 and d_ff=49152.

**Question:** Calculate the number of parameters in a single feed-forward layer (excluding biases, using approximate formula) for both models. How many times larger is GPT-3's FFN compared to BERT-base's?

*(Answer hidden below)*

---

.

.

.

.

.

**Answer:**

Single FFN parameters ≈ 2 × d_model × d_ff

BERT-base: 2 × 768 × 3072 = 2 × 2,359,296 = 4,718,592 ≈ 4.7M parameters

GPT-3: 2 × 12288 × 49152 = 2 × 603,979,776 = 1,207,959,552 ≈ 1.21B parameters

Ratio: 1.21B / 4.7M = 257× larger

Each GPT-3 FFN has more parameters than the entire BERT-base model (110M). GPT-3 has 96 such FFNs (one per layer), so total FFN parameters ≈ 96 × 1.21B ≈ 116B—most of GPT-3's 175B total parameters. This scaling is why GPT-3 requires massive compute.
$$
