# Sparse attention

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Model Architecture Variants**

---

### **1. Why this concept matters**

Standard self-attention has a hidden cost: it scales quadratically with sequence length. For 1,000 tokens, attention computes 1 million pairs. For 100,000 tokens, it computes 10 billion pairs. This makes processing long documents, books, or codebases prohibitively expensive. Sparse attention solves this by restricting each token to attend only to a subset of other tokens—not all. The result is linear or near-linear scaling, enabling context windows of 100,000+ tokens. Sparse attention is why models like Longformer, BigBird, and Mistral can handle entire books, not just paragraphs.

---

### **2. Core idea**

**Sparse attention restricts the attention pattern to a subset of token pairs (e.g., local windows, global tokens, or learned patterns), reducing computational complexity from O(n²) to O(n) or O(n√n) while preserving model performance for long sequences.**

---

### **3. Concrete analogy**

Imagine a large conference with 1,000 people. Full attention would require every person to talk to every other person. That is 500,000 conversations—chaotic and slow.

Now imagine a more efficient communication protocol:

- **Local attention:** You only talk to people at your table (neighbors). This covers most local information.
- **Global attention:** A few moderators (important tokens) talk to everyone and broadcast summaries.
- **Sliding window:** You talk to the 5 people to your left and 5 to your right. As you move through the room, you cover the whole group without pairwise communication.

Sparse attention uses these patterns. Each token only attends to a subset of other tokens: local neighbors (window), a few global tokens, or random samples. Total conversations drop from 500,000 to ~5,000. The model still learns the important relationships—most information is local, and global information is handled by specialized mechanisms.

---

### **4. ASCII diagram**

```
Full attention vs Sparse attention patterns (n=12):

Full attention (O(n²)):               Sparse (sliding window, w=3):

    Tokens 1-12                           Tokens 1-12
    ┌─────────────────┐                   ┌─────────────────┐
    │ ████████████████│                   │ ██               │
    │ ████████████████│                   │ ████             │
    │ ████████████████│                   │ ██████           │
    │ ████████████████│                   │  ██████          │
    │      ...        │                   │   ██████         │
    └─────────────────┘                   │    ██████        │
    Every pair attended                  │     ██████       │
                                          │      ██████      │
    Sparse patterns:                      │       ██████     │
                                          └─────────────────┘

    Sliding window: each token attends to w neighbors on each side
    Complexity: O(n × w) = O(n) if w constant


    Global + Local (Longformer):

    ┌─────────────────────────────────────┐
    │ G ██████████████████████████████████│  G = global token (attends to all)
    │ 1 ██                                │  Local tokens attend to neighbors
    │ 2 ████                              │      and global tokens
    │ 3 ██████                            │
    │ 4   ██████                          │
    │ 5     ██████                        │
    │ 6       ██████                      │
    │ 7         ██████                    │
    │ 8           ██████                  │
    │ 9             ██████                │
    │10               ██████              │
    │11                 ██████            │
    │12                   ██████          │
    └─────────────────────────────────────┘
```

---

### **5. Mathematical formulation**

**Full attention complexity:**

$$
\text{FLOPs} = O(n^2 \cdot d)
$$

Where n = sequence length, d = hidden dimension.

**Sliding window attention (local):**

Each token i attends to tokens in window [i-w, i+w]:

$$
\Omega(i) = \{j : |i-j| \leq w\}
$$

Complexity: O(n × (2w+1) × d) = O(n) if w is constant.

**Longformer (global + local):**

- Some tokens (e.g., [CLS]) are global: attend to all tokens
- Other tokens attend to local window + global tokens

Complexity: O(n × (w + g)) where g = number of global tokens.

**BigBird (window + global + random):**

Three attention patterns:

1. Local window (w)
2. Global tokens (g)
3. Random tokens (r)

Complexity: O(n × (w + g + r)) = O(n).

**Block-sparse attention (Sparse Transformer):**

Divide sequence into blocks of size b. Each token attends to:

- Tokens in its own block
- Tokens in neighboring blocks
- A few random blocks

Complexity: O(n × b).

---

### **6. Worked example (step-by-step)**

#### **Step 1: Sequence length n=10,000**

Full attention: 10,000² = 100 million pairs. At 100ns per pair (optimized), 10 seconds per layer. For 32 layers: 320 seconds per forward pass. Impossible.

#### **Step 2: Sliding window attention (w=512)**

Each token attends to 1024 neighbors (512 left + 512 right + self). Total pairs = 10,000 × 1024 ≈ 10 million pairs. 10× reduction.

#### **Step 3: Longformer (global tokens = 10, local window = 512)**

Global tokens attend to all (10 × 10,000 = 100,000 pairs)
Local tokens attend to window (10,000 × 1024 ≈ 10 million pairs)
Total ≈ 10.1 million pairs. Similar to sliding window.

#### **Step 4: Block-sparse (block size=64)**

10,000 / 64 ≈ 156 blocks.

Each token attends to:

- Its own block (64 tokens)
- Adjacent blocks (128 tokens)
- 2 random blocks (128 tokens)

Total per token ≈ 320 tokens. Total pairs = 10,000 × 320 = 3.2 million pairs. 30× reduction from sliding window, 300× from full attention.

#### **Step 5: Impact on training**

Full attention on 10,000 tokens: requires 100B pairs per layer, OOM on 80GB GPU.
Sparse attention (window=512): 10M pairs per layer, fits on 24GB GPU.
Enables training on 10,000-100,000 token sequences.

---

### **7. How this appears inside neural networks and LLMs**

- **Longformer (2020):** Encoder-only model with sliding window + global attention. Handles 4,096 tokens (vs BERT's 512). Used for long document classification.

- **BigBird (2020):** Window + global + random attention. Theoretical equivalence to full attention (universal approximator) with linear complexity. Handles 4,096+ tokens.

- **Sparse Transformer (2019):** Block-sparse attention for image generation. Enabled modeling of high-resolution images (64×64 pixels).

- **Mistral (2023):** Uses sliding window attention (window=4096). Claims effective context of 32K with 4K window via attention sinks. Much faster than full attention.

- **GPT-4 (rumored):** Uses mixture of full and sparse attention across layers. Early layers (local patterns) use sparse; later layers (global reasoning) use full.

- **Routing Transformer (2020):** Learnable sparse patterns—model learns which positions to attend to.

- **Linformer (2020):** Linear attention via low-rank projection (not sparse, but similar motivation). Projects n×n matrix to n×k.

- **FlashAttention (2022-2024):** Not sparse, but optimizes full attention to be memory-efficient. Enables 128K context on A100 without sparsity.

- **Tradeoff:** Sparse attention reduces compute but may lose long-range dependencies. Hybrid approaches (sparse in early layers, full in later) balance efficiency and expressivity.

---

### **8. Brain-like connection (foveated attention)**

The human visual system does not process every pixel equally. Foveal vision (center) has high resolution—like local attention. Peripheral vision (edges) is coarse—like global attention. The brain shifts the fovea (focus) based on task demands—like dynamic sparse attention. This is foveated attention: computational efficiency through selective processing. Sparse attention in transformers mimics this biological principle: most information is local, global information is handled by specialized mechanisms, and the model learns where to allocate attention. Without foveation, human vision would require 100 million photoreceptors processing 100× per second—comparable to the quadratic complexity problem solved by sparse attention.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Sparse attention is just a memory optimization. Models with sparse attention are less powerful because they cannot see long-range dependencies."

_Why it is wrong:_ Sparse attention can maintain theoretical equivalence to full attention with the right pattern. BigBird proves that window + global + random attention is a universal approximator (can approximate any full attention pattern). In practice, most long-range dependencies are not needed—language has locality. The attention span beyond 1000 tokens is rarely required for individual word relationships (though global context matters). Moreover, modern models combine sparse attention with deep layers: early layers capture local patterns with sparsity, later layers capture global patterns with fewer tokens (pooling). The tradeoff is compute vs capability, but for many tasks, sparse attention is sufficient and enables much longer contexts. Mistral's sliding window (4096) yields 32K effective context—still covering long documents.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Full attention is powerful but O(n²) kills long          |
|  sequences. A book has 100,000 tokens—10 billion          |
|  attention pairs. Sparse attention brings complexity      |
|  back to O(n), making long documents, entire codebases,   |
|  and hours of audio feasible. Mistral's 32K context,      |
|  Longformer's 4K documents, Sparse Transformer's high-    |
|  resolution images—all enabled by sparsity. Without it,   |
|  LLMs would be stuck summarizing paragraphs, not books.   |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are designing a model to process a 50,000-token document (e.g., an entire research paper). Full attention would require 2.5 billion pairs per layer—impossible on your GPU.

**Question:** Which sparse attention pattern would you choose? What are the tradeoffs between sliding window, global + local, and block-sparse? Which would you pick for a research paper summarization task?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** For research paper summarization (50,000 tokens), I would choose Longformer-style global + local attention.

**Tradeoffs:**

- **Sliding window (w=512):** Each token attends to 1024 neighbors. Complexity O(n). Loses global structure (introduction and conclusion distant). May miss cross-section relationships.

- **Block-sparse (blocks of 256):** Each token attends to own block + adjacent + random. Efficient but may miss systematic relationships.

- **Global + local (Longformer):** Add global tokens (e.g., [CLS], section headers) that attend to all tokens. These capture document-level structure. Local tokens (window) capture paragraph-level detail.

For summarization, the model needs both:

- Local details (within sections) → sliding window
- Global coherence (across sections) → global tokens

Thus, global + local is optimal. Set window size to 512-1024 (covers a paragraph). Add 10-20 global tokens (e.g., [CLS] per section). Complexity: O(n × w + g × n) ≈ O(n). Achievable on 24-48GB GPU.

Mistral's sliding window alone (no global tokens) might work but would rely on attention sinks to propagate information—riskier for very long documents. Longformer is proven for this use case.
