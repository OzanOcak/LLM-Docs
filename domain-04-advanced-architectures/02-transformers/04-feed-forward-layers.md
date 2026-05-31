# Feed-Forward Layers

## The Personal Assistant Analogy

Imagine you're a busy executive. Your assistant receives information, thinks about it deeply, and then presents you with a refined summary. They don't just pass messages along—they process, analyze, and extract the key points. That's what feed-forward layers do in transformers: after words have attended to each other, each word passes through its own personal processing unit that thinks about its new context and extracts useful patterns.

In LLMs, feed-forward layers are where most of the "thinking" happens. While attention mixes information between words, feed-forward layers process each word individually, applying complex transformations. In fact, about two-thirds of a transformer's parameters live in these layers—they're the workhorses of the model.

---

## What Are Feed-Forward Layers?

### The Core Idea

A feed-forward layer is a simple neural network that processes each position independently and identically.

```text

After attention:            Feed-forward layer:
                            ┌─────────────────┐
    "cat"* ───────────────────►  Expand      │
                              │  (4× size)    │
    "sat"* ───────────────────►  Activate     │
                              │  (ReLU/GELU)  │
    "on"*  ───────────────────►  Compress     │
                              │  (back to size)│
                              └─────────────────┘
                                    ↓
                            New representations
```

```python

def ff_intro():
    """
    The basic concept of feed-forward layers
    """
    print("Feed-Forward Layers: Individual Processing")
    print("=" * 60)

    print("""
    Each position gets the same treatment:

    Input: x (e.g., 768-dim vector for "cat")
           ↓
    Step 1: Expand to larger dimension (e.g., 3072)
           ↓
    Step 2: Apply non-linear activation (GELU)
           ↓
    Step 3: Compress back to original dimension (768)
           ↓
    Output: New x, now with deeper processing

    Same weights, applied to every position!
    """)

    print("\nThis is where the model 'thinks' about each word individually.")

ff_intro()
```

---

## The Two-Layer Design

### Architecture

```python

def ff_architecture():
    """
    The structure of a feed-forward layer
    """
    print("Feed-Forward Layer Architecture")
    print("=" * 60)

    d_model = 768  # Typical transformer size
    d_ff = 3072    # Usually 4× d_model

    print(f"Model dimension (d_model): {d_model}")
    print(f"Feed-forward dimension (d_ff): {d_ff} (4×)")

    print("""
    Formula: FF(x) = W₂ · GELU(W₁ · x + b₁) + b₂

    Layer 1 (Expansion):
        W₁: (d_ff × d_model) matrix
        b₁: (d_ff) bias
        Expands from 768 → 3072

    Activation (GELU):
        Non-linearity (smooth version of ReLU)

    Layer 2 (Compression):
        W₂: (d_model × d_ff) matrix
        b₂: (d_model) bias
        Compresses from 3072 → 768
    """)

ff_architecture()
```

### Visualizing the Transformation

```text

Dimension:     768                   3072                   768
              ┌───┐                ┌───┐                ┌───┐
              │   │                │   │                │   │
Input vector  │   │ ───[W₁,b₁]───> │   │ ──[GELU]─────> │   │ ───[W₂,b₂]───> Output
              │   │                │   │                │   │
              └───┘                └───┘                └───┘

Each rectangle is a dimension of the vector.
Expansion allows more representational capacity.
```

---

## Step by Step Through a Feed-Forward Layer

```python

import numpy as np

def ff_step_by_step():
    """
    Detailed walkthrough of feed-forward computation
    """
    print("Feed-Forward Layer: Step by Step")
    print("=" = 60)

    # Small dimensions for illustration
    d_model = 4
    d_ff = 8  # 2× for this example

    # Input vector for one word
    x = np.array([0.5, -0.3, 0.8, 0.2])

    print(f"Input vector x (dim={d_model}): {x}")

    # Layer 1 weights and bias (random for demo)
    np.random.seed(42)
    W1 = np.random.randn(d_ff, d_model) * 0.1
    b1 = np.zeros(d_ff)

    print("\nStep 1: Linear expansion")
    print(f"  W1 shape: {W1.shape} ({d_ff}×{d_model})")

    z1 = W1 @ x + b1
    print(f"  z1 = W1·x + b1")
    print(f"  z1 shape: {z1.shape}")
    print(f"  z1 values: {[round(v,3) for v in z1]}")

    print("\nStep 2: Activation (GELU)")
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    a = gelu(z1)
    print(f"  a = GELU(z1)")
    print(f"  a values: {[round(v,3) for v in a]}")

    # Layer 2 weights and bias
    W2 = np.random.randn(d_model, d_ff) * 0.1
    b2 = np.zeros(d_model)

    print("\nStep 3: Linear compression")
    print(f"  W2 shape: {W2.shape} ({d_model}×{d_ff})")

    y = W2 @ a + b2
    print(f"  y = W2·a + b2")
    print(f"  y shape: {y.shape}")
    print(f"  y values: {[round(v,3) for v in y]}")

    print("\nFinal output has same dimension as input (4)")
    print("But now it's been processed and transformed!")

ff_step_by_step()
```

---

## Why Expand Then Compress?

### The Intuition

```python

def why_expand_compress():
    """
    Why feed-forward layers use expansion
    """
    print("Why Expand Then Compress?")
    print("=" = 60)

    print("""
    Analogy: Thinking about a problem

    Step 1 (Expand): Consider many possibilities
        "What could this mean? Let me think of all
        the implications, connections, and interpretations..."

    Step 2 (Activate): Decide which are relevant
        "These ideas make sense, those don't"

    Step 3 (Compress): Synthesize into conclusion
        "Based on all that, here's my refined understanding"
    """)

    print("\nMathematical intuition:")
    print("• Expansion creates more 'room' for computation")
    print("• Each dimension can represent different features")
    print("• Non-linearity lets them interact")
    print("• Compression distills the result")

why_expand_compress()
```

### Universal Approximation Theorem

```python

def universal_approx():
    """
    Why two layers with non-linearity are powerful
    """
    print("Universal Approximation Property")
    print("=" = 60)

    print("""
    A feed-forward network with:
    • One hidden layer
    • Non-linear activation
    • Enough hidden units

    Can approximate ANY continuous function!

    This means the feed-forward layer can learn
    any transformation of the input representation.
    """)

    print("\nIn practice, transformers use this to:")
    print("• Detect patterns in the attended information")
    print("• Apply knowledge stored in weights")
    print("• Transform representations for next layer")

universal_approx()
```

---

## Activation Functions in Feed-Forward Layers

### GELU (Gaussian Error Linear Unit)

Modern LLMs use GELU instead of ReLU.

```python

def activation_comparison():
    """
    Comparing activation functions
    """
    import math

    print("Activation Functions in Feed-Forward Layers")
    print("=" = 60)

    def gelu(x):
        return 0.5 * x * (1 + math.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

    def relu(x):
        return max(0, x)

    test_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]

    print("x    | ReLU | GELU")
    print("-" * 25)

    for x in test_values:
        r = relu(x)
        g = gelu(x)
        print(f"{x:3.1f} |  {r:3.1f}  | {g:5.3f}")

    print("\nWhy GELU?")
    print("• Smooth (differentiable everywhere)")
    print("• Negative values get small negative output (not zero)")
    print("• Works better for transformers than ReLU")

activation_comparison()
```

---

## A Complete Feed-Forward Implementation

```python

class FeedForward:
    """
    Complete feed-forward layer implementation
    """
    def __init__(self, d_model, d_ff, activation='gelu'):
        self.d_model = d_model
        self.d_ff = d_ff

        # Initialize weights
        self.W1 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)

        self.activation = activation

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x):
        """
        x: (seq_len, d_model) or (d_model,)
        """
        # Store input shape
        single_input = len(x.shape) == 1

        if single_input:
            x = x.reshape(1, -1)

        seq_len = x.shape[0]

        print(f"\nProcessing {seq_len} positions in parallel")

        # Step 1: Expansion
        z1 = x @ self.W1.T + self.b1
        print(f"  After expansion: {z1.shape}")

        # Step 2: Activation
        if self.activation == 'gelu':
            a = self.gelu(z1)
        else:  # relu
            a = np.maximum(0, z1)
        print(f"  After activation: {a.shape}")

        # Step 3: Compression
        y = a @ self.W2.T + self.b2
        print(f"  After compression: {y.shape}")

        if single_input:
            y = y[0]

        return y

def ff_demo():
    """
    Demonstrate feed-forward layer
    """
    print("Complete Feed-Forward Layer Demo")
    print("=" = 60)

    # Create feed-forward layer
    ff = FeedForward(d_model=4, d_ff=8)

    # Single position
    x_single = np.array([0.5, -0.3, 0.8, 0.2])
    print("\nSingle position input:")
    y_single = ff.forward(x_single)
    print(f"  Output: {[round(v,3) for v in y_single]}")

    # Multiple positions in parallel
    x_batch = np.array([
        [0.5, -0.3, 0.8, 0.2],  # word 1
        [0.1, 0.7, -0.2, 0.4],  # word 2
        [0.3, 0.2, 0.9, -0.1]   # word 3
    ])

    print("\nMultiple positions (3 words):")
    y_batch = ff.forward(x_batch)
    print(f"  Output shape: {y_batch.shape}")

ff_demo()
```

---

## Feed-Forward vs Attention

### Different Roles

| Aspect        | Attention                  | Feed-Forward               |
| ------------- | -------------------------- | -------------------------- |
| Communication | Between words              | Within a word              |
| Operation     | Mixes information          | Processes individually     |
| Parameters    | Shared across positions    | Shared across positions    |
| Computation   | O(n²) with sequence length | O(n) with sequence length  |
| Role          | "Look at other words"      | "Think about what you saw" |

### The Transformer Dance

```python

def transformer_dance():
    """
    How attention and feed-forward work together
    """
    print("Attention + Feed-Forward: The Transformer Dance")
    print("=" = 60)

    print("""
    Each transformer layer does:

    1. Multi-Head Attention
       └── Words share information
       └── "What should I pay attention to?"

    2. Add & LayerNorm
       └── Residual connection + normalize

    3. Feed-Forward Network
       └── Each word processes individually
       └── "Now that I have this info, what do I think?"

    4. Add & LayerNorm
       └── Another residual + normalize

    Then next layer...
    """)

    print("\nThis combination is incredibly powerful!")

transformer_dance()
```

---

## Why This Matters for LLMs

### 1. Most Parameters Are in Feed-Forward Layers

```python

def parameter_count():
    """
    Where the parameters are
    """
    print("Parameter Distribution in Transformers")
    print("=" = 60)

    d_model = 768
    d_ff = 3072
    n_layers = 12
    n_heads = 12
    d_head = d_model // n_heads

    # Attention parameters: Q,K,V,O projections
    attn_params = 4 * d_model * d_model * n_layers

    # Feed-forward parameters: two layers
    ff_params = 2 * d_model * d_ff * n_layers

    total = attn_params + ff_params

    print(f"BERT-base style model:")
    print(f"  d_model={d_model}, d_ff={d_ff}, layers={n_layers}")
    print(f"\nAttention parameters: {attn_params:,}")
    print(f"Feed-forward parameters: {ff_params:,}")
    print(f"Total: {total:,}")
    print(f"\nFeed-forward percentage: {ff_params/total*100:.1f}%")

parameter_count()
```

### 2. GLU Variants in Modern LLMs

```python

def glu_variants():
    """
    Gated Linear Unit variants
    """
    print("GLU Variants in Modern LLMs")
    print("=" = 60)

    print("""
    Many modern LLMs use Gated Linear Units instead of simple FF:

    FF = GELU(xW₁)W₂                 (standard)

    GLU variants:

    SwiGLU (Llama): Swish(xW₁) ⊗ (xV)
    GeGLU:         GELU(xW₁) ⊗ (xV)
    GLU:           σ(xW₁) ⊗ (xV)

    Where ⊗ is element-wise multiplication.

    These add a "gate" that controls information flow.
    """)

    print("\nUsed in: Llama, PaLM, many modern LLMs")

glu_variants()
```

### 3. Computational Cost

```python

def computational_cost():
    """
    Cost of feed-forward layers
    """
    print("Computational Cost of Feed-Forward Layers")
    print("=" = 60)

    seq_len = 2048
    d_model = 768
    d_ff = 3072

    # FLOPs for one feed-forward layer
    flops_per_word = 2 * d_model * d_ff  # approximate
    total_flops = seq_len * flops_per_word

    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"FF dimension: {d_ff}")
    print(f"\nFLOPs per layer: {total_flops:,}")
    print(f"For 12 layers: {total_flops * 12:,}")

    print("\nThis is why GPUs need to be so powerful!")

computational_cost()
```

### 4. Knowledge Storage

```python

def knowledge_storage():
    """
    Where models store knowledge
    """
    print("Feed-Forward Layers as Knowledge Storage")
    print("=" = 60)

    print("""
    Research suggests feed-forward layers act as
    key-value memories:

    • W₁ weights detect patterns
    • Activation selects relevant memories
    • W₂ weights retrieve associated knowledge

    For example:
    Input: "Capital of France is"
    FF layer might activate "Paris" pattern

    This is where factual knowledge is stored!
    """)

    print("\nAttention finds what to look up,")
    print("feed-forward layers store the knowledge itself.")

knowledge_storage()
```

---

## Feed-Forward Cheat Sheet

| Aspect                   | Value                                    |
| ------------------------ | ---------------------------------------- |
| Typical expansion factor | 4× d_model                               |
| Activation               | GELU (modern) or ReLU (older)            |
| Parameters               | 2 × d_model × d_ff × layers              |
| Computation              | O(seq_len × d_model × d_ff)              |
| Role                     | Individual processing, knowledge storage |

---

## Quick Recap

• Feed-forward layers process each word independently after attention—like a personal assistant who takes the information gathered from others and thinks deeply about what it means

• They expand to a larger dimension, apply non-linearity, then compress back—this expansion creates "thinking room" to explore possibilities, then distills the result

• Feed-forward layers contain most of a transformer's parameters and store factual knowledge—about two-thirds of the model lives here, and research suggests they act as key-value memories for information learned during training

---

## Mental Hook

> "Feed-forward layers are where each word goes to think after the meeting—attention lets everyone share information, but feed-forward is the private office where each word processes what it learned and updates its own understanding."
