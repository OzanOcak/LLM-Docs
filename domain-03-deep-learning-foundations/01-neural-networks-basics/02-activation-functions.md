# Activation functions (ReLU, sigmoid, tanh)

## **DOMAIN: DEEP LEARNING FOUNDATIONS | Sub domain: Neural Networks Basics**

---

### **1. Why this concept matters for building intelligent systems**

A weighted sum plus bias is just a linear function. Stack linear functions and you still get a linear function. No matter how many layers, without nonlinearity, a neural network collapses to a single linear transformation. Activation functions break this linearity. They are the "decision gates" that let neural networks model curves, edges, boundaries, and eventually, language. Choosing the right activation function determines whether your network trains quickly, suffers vanishing gradients, or produces sparse representations. Understanding activation functions means understanding how neurons "decide" to fire—and how deep networks become more than just complicated linear regression.

---

### **2. Core idea**

**Activation functions introduce nonlinearity into neural networks, transforming the weighted sum of inputs into an output that determines whether a neuron "fires" and with what strength, enabling the network to learn complex patterns.**

---

### **3. Concrete analogy**

Imagine three types of volume knobs:

- **Sigmoid knob**: Turns smoothly from 0 to 10. Very sensitive in the middle (5), but at the extremes (0 or 10), turning the knob does almost nothing. The response is S-shaped. Good for probability-like outputs.

- **Tanh knob**: Same smooth S-shape, but ranges from -10 to +10 instead of 0 to 10. Centered at zero. Good when you need positive and negative outputs.

- **ReLU knob**: Simple: if the knob is turned left of zero, output is zero. If turned right, output equals the turn amount. No sensitivity near zero—just a sharp kink. Surprisingly effective and computationally cheap.

In a neural network, each neuron has such a knob. The activation function determines how the neuron responds to its inputs. Without these knobs, all neurons would respond linearly, and the network could never learn to make decisions.

---

### **4. ASCII diagram**

```text
Activation functions compared:

    Sigmoid: σ(z) = 1/(1+e^{-z})

    f(z) ↑
      1 ┤         ▄▄▄▄▄
        │       ▄▄
      0 ┼──────▄▄
        └────────────────→ z
        -4  -2   0   2   4
    Output range: (0, 1)
    Problem: Gradients vanish outside [-3,3]


    Tanh: tanh(z) = (e^z - e^{-z})/(e^z + e^{-z})

    f(z) ↑
      1 ┤         ▄▄▄▄▄
        │       ▄▄
      0 ┼──────▄▄
        │    ▄▄
     -1 ┼─▄▄▄
        └────────────────→ z
        -4  -2   0   2   4
    Output range: (-1, 1)
    Problem: Still saturates, but zero-centered (better than sigmoid)


    ReLU: ReLU(z) = max(0, z)

    f(z) ↑
        │     /
        │    /
        │   /
        │  /
        │ /
      0 ┼/──────────────→ z
        │
        └────────────────→ z
    Output range: [0, ∞)
    Problem: "Dying ReLU" (neurons stuck at 0 forever)


    Leaky ReLU: f(z) = max(0.01z, z)

    f(z) ↑
        │     /
        │    /
        │   /
        │  /
      0 ┼╱──────────────→ z
        │╱
    Small negative slope prevents dying neurons.
```

---

### **5. Mathematical formulation**

**Sigmoid:**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Derivative: σ'(z) = σ(z)(1 - σ(z))

Range: (0, 1), centered at σ(0)=0.5

**Tanh (Hyperbolic Tangent):**

$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = 2\sigma(2z) - 1
$$

Derivative: tanh'(z) = 1 - tanh²(z)

Range: (-1, 1), centered at tanh(0)=0 (zero-centered, often preferred over sigmoid)

**ReLU (Rectified Linear Unit):**

$$
\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
$$

Derivative: ReLU'(z) = 1 if z > 0, 0 if z ≤ 0 (undefined at 0, but subgradient 0 is used)

Range: [0, ∞)

**Leaky ReLU (fix for dying ReLU):**

$$
\text{LeakyReLU}(z) = \max(\alpha z, z), \quad \alpha \text{ small (e.g., 0.01)}
$$

**ELU (Exponential Linear Unit):**

$$
\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^{z} - 1) & \text{if } z \leq 0 \end{cases}
$$

**Swish (Google's discovery, used in some LLMs):**

$$
\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

**GELU (Gaussian Error Linear Unit, used in BERT, GPT):**

$$
\text{GELU}(z) = z \cdot \Phi(z)
$$

Where Φ is standard normal CDF. Approximated as 0.5z(1 + tanh(√(2/π)(z + 0.044715z³)))

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define inputs**

A single neuron receives weighted sum z = -2, 0, and 2.

#### **Step 2: Compute sigmoid outputs**

σ(-2) = 1/(1 + e²) = 1/(1 + 7.389) = 1/8.389 = 0.119
σ(0) = 1/(1 + 1) = 0.5
σ(2) = 1/(1 + e⁻²) = 1/(1 + 0.135) = 1/1.135 = 0.881

#### **Step 3: Compute tanh outputs**

tanh(-2) = (e⁻² - e²)/(e⁻² + e²) = (0.135 - 7.389)/(0.135 + 7.389) = (-7.254)/7.524 = -0.964
tanh(0) = 0
tanh(2) = 0.964

#### **Step 4: Compute ReLU outputs**

ReLU(-2) = max(0, -2) = 0
ReLU(0) = max(0, 0) = 0
ReLU(2) = max(0, 2) = 2

#### **Step 5: Compare gradients (how learning signal propagates)**

For sigmoid, gradient at z=2 is σ'(2) = 0.881 × 0.119 = 0.105 (small)
For tanh, gradient at z=2 is 1 - 0.964² = 1 - 0.929 = 0.071 (very small)
For ReLU, gradient at z=2 is 1 (full signal passes through)

ReLU does not saturate for positive inputs, allowing deep networks to train.

#### **Step 6: Show dying ReLU**

If a neuron always receives negative inputs (e.g., due to bad initialization), its output is always 0. Gradient is 0, so it never recovers. It is "dead forever."

Example: z = -1 always → ReLU = 0 → gradient = 0 → weights never update.

---

### **7. How this appears inside neural networks and LLMs**

- **ReLU in early deep learning:** Standard for CNNs and fully connected networks. Fast, simple, avoids vanishing gradients.

- **GELU in modern LLMs:** BERT, GPT, Llama, and most transformers use GELU (or approximations). It is smoother than ReLU and works better for language.

- **Swish in some architectures:** Used in MobileNetV3 and some Google models. Discovered by automated search.

- **Sigmoid for output layer:** Binary classification outputs use sigmoid to produce probabilities between 0 and 1.

- **Softmax (sigmoid generalization):** Multi-class output layer uses softmax, which exponentiates and normalizes across classes.

- **Tanh in RNNs (historical):** Early RNNs used tanh to keep activations centered near zero, helping with vanishing gradients (though not solving fully).

- **GLU (Gated Linear Unit):** Variant used in some transformers (e.g., Google's T5). GLU(x) = x ⊙ σ(Wx + b).

- **Layer Normalization + Activation:** In transformers, layer norm often comes before or after activation (Pre-LN vs Post-LN). Pre-LN helps training stability.

---

### **8. Brain-like connection (neural firing rates)**

Biological neurons have activation functions—they are not linear. The firing rate of a neuron as a function of input current follows a sigmoid-like curve: low input produces few spikes, high input saturates at maximum firing rate. The sharpness of the curve is adaptive. ReLU approximates a neuron with a threshold but no upper bound—not biologically realistic but computationally efficient. The brain's activation functions are analog, continuous, and plastic—they change with experience. The diversity of artificial activation functions reflects different tradeoffs between biological plausibility, computational efficiency, and trainability.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Sigmoid and tanh are obsolete. ReLU is always better."

_Why it is wrong:_ ReLU has the dying ReLU problem—neurons can get stuck at zero forever. Sigmoid and tanh saturate but do not die. For output layers requiring probabilities (0 to 1), sigmoid is still the standard. For autoencoders needing bounded outputs, tanh works well. In certain architectures (e.g., small networks, some RNNs), tanh still outperforms ReLU. The best activation depends on the task, architecture, and initialization. Modern LLMs use GELU, not ReLU. There is no universal winner.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Without activation functions, deep learning is just      |
|  matrix multiplication—linear algebra, not intelligence.  |
|  Activation functions give neural networks the power to   |
|  decide, to fire or not, to represent yes/no, maybe,      |
|  and everything between. Sigmoid gives probabilities.     |
|  Tanh gives balanced signals. ReLU gives sparsity and     |
|  speed. GELU gives smooth language understanding.         |
|  Choose your nonlinearity wisely—it shapes the mind.      |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

A neuron has z = -1, 0, 3. For each activation (sigmoid, tanh, ReLU), compute the output and the gradient (derivative) at that point.

**Question:** At which activation and input does the gradient vanish (become very small)? Which activation maintains a gradient of 1 for positive inputs?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

At z=-1:

- Sigmoid: σ≈0.269, σ'≈0.269×0.731=0.197 (small)
- Tanh: tanh≈-0.762, derivative=1-0.581=0.419
- ReLU: 0, derivative=0 (dead neuron)

At z=0:

- Sigmoid: 0.5, derivative=0.5×0.5=0.25
- Tanh: 0, derivative=1
- ReLU: 0, derivative=0 at exactly 0 (subgradient 0)

At z=3:

- Sigmoid: 0.953, derivative≈0.953×0.047=0.045 (very small—saturation)
- Tanh: 0.995, derivative≈1-0.99=0.01 (very small—saturation)
- ReLU: 3, derivative=1 (maintains gradient)

ReLU maintains gradient of 1 for all positive inputs. Sigmoid and tanh saturate at large |z|, causing vanishing gradients. ReLU at negative inputs dies (gradient 0).
