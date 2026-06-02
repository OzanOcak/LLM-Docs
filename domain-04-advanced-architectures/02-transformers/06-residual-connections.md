# Residual connections

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Transformers (The LLM Revolution)**

---

### **1. Why this concept matters**

Deep networks should be more powerful than shallow ones. But in practice, adding layers often makes training harder, not better. Gradients vanish, activations degrade, and performance saturates or even drops. Residual connections solve this by adding a shortcut: the input to a layer is added to its output. This simple addition creates a gradient highway, allowing signals to flow directly through the network without passing through every nonlinearity. With residual connections, you can train networks with hundreds of layers. Without them, even 20 layers are difficult. Residual connections are the reason modern LLMs can be 100+ layers deep.

---

### **2. Core idea**

**A residual connection adds the input of a layer to its output, creating a direct path for gradients to skip over the layer's transformations, allowing much deeper networks by preventing vanishing gradients and enabling identity learning.**

---

### **3. Concrete analogy**

Imagine a tall office building with 100 floors. To reach the top, you could take the stairs (passing through every floor). But each floor has a heavy door that slows you down. After 50 floors, you are exhausted. This is a deep network without residuals.

Now imagine the building has an express elevator that bypasses all floors, plus stairs on each floor. You can take the elevator to any floor, then walk a short distance. This is a residual network. The elevator is the skip connection; the stairs are the layer's transformation.

Even better: if a floor's stairs are blocked (a layer learns nothing), you still have the elevator. The network can learn that some layers should do nothing (by learning weights near zero) and rely on the skip connection. This "identity shortcut" ensures that adding more layers never hurts performance—the network can always ignore them.

---

### **4. ASCII diagram**

```
Residual connection in a transformer block:

    Input x
       │
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 │
    ┌─────────────────────┐              │
    │   Multi-Head         │              │
    │   Attention          │              │
    └─────────────────────┘              │
       │                                 │
       │  output = Attention(x)          │
       │                                 │
       ▼                                 │
       ⊕ ←─────────────── skip ──────────┘
       │
       ▼
    LayerNorm (or directly to next layer)
       │
       ▼
       x + Attention(x)   ← Residual output


Residual block general form:

    y = F(x) + x

    where:
    - x = input
    - F(x) = learned transformation (attention, FFN, convolution)
    - + = elementwise addition


Comparison: With vs without residuals

    Without residuals:           With residuals:

    x → Layer1 → Layer2 → ...    x → Layer1 → ⊕ → Layer2 → ⊕ → ...
    ↓    ↓       ↓               ↓     ↓      ↓      ↓      ↓
    No direct path              Direct skip connections
    Gradients must pass         Gradients flow through
    through every layer         skip connection (multiply by 1)

Residual variants:

    Original:  y = F(x) + x
    Pre-activation (ResNet v2): y = x + F(Norm(x))
    Transformer (Pre-LN): y = x + Attention(Norm(x))
```

---

### **5. Mathematical formulation**

**Basic residual block:**

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

(Unicode: y = F(x, {W_i}) + x)

Where:

- x = input to the block
- F = learned transformation (e.g., attention, feed-forward network, convolution)
- - is elementwise addition (requires x and F(x) to have same shape)

**Gradient flow through residual connection:**

Backpropagation through a residual block:

$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right)
$$

(Unicode: ∂L/∂x = (∂L/∂y) × (∂F/∂x + I))

The identity matrix I ensures that even if ∂F/∂x → 0, the gradient still flows through:

$$
\frac{\partial L}{\partial \mathbf{x}} \approx \frac{\partial L}{\partial \mathbf{y}} \quad \text{when} \quad \frac{\partial \mathcal{F}}{\partial \mathbf{x}} \to 0
$$

**Transformer with Pre-LN (residual connections before normalization):**

$$
\mathbf{x}' = \mathbf{x} + \text{Attention}(\text{LayerNorm}(\mathbf{x}))
$$

$$
\mathbf{x}'' = \mathbf{x}' + \text{FFN}(\text{LayerNorm}(\mathbf{x}'))
$$

**Transformer with Post-LN (original transformer, residuals after normalization):**

$$
\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{Attention}(\mathbf{x}))
$$

$$
\mathbf{x}'' = \text{LayerNorm}(\mathbf{x}' + \text{FFN}(\mathbf{x}'))
$$

**Projected residuals (when dimensions change):**

If F changes dimensions, the skip connection needs a projection:

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{W}_{\text{skip}} \mathbf{x}
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Simple residual block**

Input x = [2.0, -1.0, 3.0]

Learned transformation F(x) = [0.5, 0.2, -0.3] (e.g., attention output)

#### **Step 2: Without residual**

Output y = F(x) = [0.5, 0.2, -0.3]

The input [2, -1, 3] is lost entirely. The layer must learn to preserve useful information.

#### **Step 3: With residual**

y = F(x) + x = [0.5+2.0, 0.2+(-1.0), -0.3+3.0] = [2.5, -0.8, 2.7]

The output still contains the original input (dominates: 2.5 vs 0.5 from F). The layer only needs to learn the _residual_ (small change) rather than the full mapping.

#### **Step 4: Gradient flow example**

Assume loss L = y² (simplified). ∂L/∂y = 2y = [5.0, -1.6, 5.4]

∂L/∂x (without residual) = ∂L/∂y × ∂F/∂x. If ∂F/∂x is small (e.g., 0.1), gradient = [0.5, -0.16, 0.54]

∂L/∂x (with residual) = ∂L/∂y × (∂F/∂x + I) = ∂L/∂y × (0.1I + I) = 1.1 × ∂L/∂y = [5.5, -1.76, 5.94]

The gradient is 11× larger than the non-residual case. Early layers receive meaningful learning signals.

#### **Step 5: Deep network example**

100 layers with residuals. Each layer has ∂F/∂x ≈ 0.1. Without residuals, gradient after 100 layers = (0.1)¹⁰⁰ ≈ 1e-100 (vanished). With residuals, minimum gradient = 1 (from identity path) plus contributions from paths that pass through some F's. Gradients survive regardless of depth.

---

### **7. How this appears inside neural networks and LLMs**

- **Every transformer block:** Contains two residual connections (one around attention, one around feed-forward). These are essential for training transformers beyond a few layers.

- **ResNet (Residual Networks for vision):** The original residual network. Showed that 152-layer networks train successfully, winning ImageNet 2015. Transformers adopted the same principle.

- **Pre-LN vs Post-LN:** Both have residuals, but Pre-LN places layer norm _before_ the transformation on the residual branch. This creates an even cleaner gradient path.

- **Weight decay and residuals:** Residual connections interact with weight decay. If F(x) learns near-zero weights, the block defaults to identity. This is a form of implicit regularization.

- **Gradient clipping + residuals:** Even with residuals, deep transformers need gradient clipping (e.g., max norm 1.0) to prevent explosion. Residuals prevent vanishing, not exploding.

- **MLP-Mixer and alternative architectures:** Some architectures replace attention with MLPs but keep residuals. Residuals are universal for deep networks, not specific to attention.

- **Training stability:** Residual connections allow much higher learning rates. Without residuals, learning rate must be tiny to avoid divergence.

---

### **8. Brain-like connection (parallel processing pathways)**

The brain has multiple parallel pathways for information. In the visual system, the "what" pathway (temporal lobe) and "where" pathway (parietal lobe) process the same input in parallel, then recombine. This is like multiple residual streams. More directly, cortical layers have both feedforward connections (through the layer) and skip connections (apical dendrites that bypass layers). Neurons integrate inputs from both local processing (F(x)) and direct bypass (x). This circuitry may serve the same function as artificial residual connections: preserving information across processing stages and ensuring stable gradient-like learning signals (spike-timing-dependent plasticity) through multiple synapses.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Residual connections just add the input to the output. That's trivial. Why is this considered a breakthrough?"

_Why it is wrong:_ The triviality is the breakthrough. Before ResNet (2015), the deep learning community believed that adding layers inevitably caused degradation. The insight that "just adding the input back" solves vanishing gradients was not obvious—it took years to discover. The mathematical consequence is profound: the network can learn identity mappings (F(x)=0) without any cost, so adding layers never hurts. With batch normalization, ResNets enabled 1000-layer networks. What seems obvious in retrospect was a major breakthrough. The simplicity is the beauty.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Depth is power. But depth kills gradients. Residual      |
|  connections resurrect them. They create a gradient       |
|  highway that bypasses every nonlinearity, ensuring       |
|  that early layers always receive a learning signal.      |
|  Without residuals, you cannot train a 100-layer network. |
|  With them, you can train GPT-3 with 96 layers, ResNet    |
|  with 152 layers, or any architecture that needs depth.   |
|  Residuals are the foundation upon which deep learning    |
|  stands.                                                   |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a 50-layer network without residual connections. Training loss stops decreasing after 20 layers—the network cannot learn. You add residual connections but do not change anything else.

**Question:** Why does this help? What happens to the gradient of layer 1 compared to before?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Without residuals, the gradient at layer 1 is the product of 49 Jacobians (∂F/∂x) from layers 2-50. If each Jacobian's norm is ≤ 1 (common with tanh, sigmoid, or weight initialization), the product decays to near zero. With residuals, the gradient at layer 1 receives contributions from paths that skip any subset of layers. The path that skips all layers (pure identity) has gradient = 1 regardless of other layers. Even if all F's have ∂F/∂x=0, the gradient is still 1. The gradient is the sum of 2⁴⁹ paths (each layer either passes through F or skips). The shortest paths (skip most layers) dominate, keeping gradient magnitudes reasonable. Layer 1 now learns.
