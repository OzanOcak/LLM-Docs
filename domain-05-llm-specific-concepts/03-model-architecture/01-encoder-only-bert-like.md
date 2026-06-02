# Encoder-only (BERT-like)

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Model Architecture Variants**

---

### **1. Why this concept matters**

Not every language task requires generation. Sometimes you just need to understand. Classify an email as spam or not. Extract names and dates from a document. Determine if two sentences are paraphrases. For these tasks, you do not need a decoder that produces text—you need a model that builds rich, bidirectional representations. Encoder-only models (BERT, RoBERTa, ALBERT) excel at this. They see the entire input at once, attending to left and right context simultaneously. They are smaller, faster, and often more accurate than decoder-only models for understanding tasks. Before ChatGPT, BERT was the dominant LLM architecture.

---

### **2. Core idea**

**An encoder-only model uses a stack of transformer encoder blocks with bidirectional self-attention to build contextual representations of an input sequence, typically using a [CLS] token or per-token outputs for classification tasks.**

---

### **3. Concrete analogy**

Imagine a detective at a crime scene. They do not need to generate a story—they need to understand the evidence. They look at the entire scene simultaneously: footprint here, broken window there, fingerprint on the glass. They consider all clues in context, no sequential order. The detective's job is understanding, not storytelling.

Encoder-only models are detectives. They take a full input (a sentence, a document) and process every token in context of all others. The output is a rich representation: at each position, a vector that captures the token's meaning given both left and right context. For classification, they use a special [CLS] token whose final representation aggregates the entire input.

This is different from decoder-only models (storytellers), which read left-to-right and generate new text. Encoder-only models do not generate—they understand.

---

### **4. ASCII diagram**

```
Encoder-only architecture (BERT):

    Input: "The cat sat on the mat"

    Tokenization: [CLS] The cat sat on the mat [SEP]

    ┌─────────────────────────────────────────────────────────┐
    │                    Encoder Stack                        │
    │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐         │
    │  │ Layer │ → │ Layer │ → │ Layer │ → │ Layer │  (12×)  │
    │  │   1   │   │   2   │   │   3   │   │  12   │         │
    │  └───────┘   └───────┘   └───────┘   └───────┘         │
    │       ↑           ↑           ↑           ↑             │
    │  Bidirectional self-attention at every layer           │
    │  (Each token attends to all tokens left and right)     │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    Output representations (same shape as input):

    [CLS]  The   cat   sat   on    the   mat   [SEP]
     ↓      ↓     ↓     ↓     ↓     ↓     ↓      ↓
    h_CLS  h_1   h_2   h_3   h_4   h_5   h_6    h_SEP

    For sentiment: use h_CLS → classifier → Positive/Negative
    For NER: use per-token h_i → classifier → Person/Org/Location


Bidirectional attention (encoder-only) vs Causal (decoder-only):

    Encoder (BERT):               Decoder (GPT):

    "cat" attends to:            "cat" attends to:
      - The (left)                 - The (left only)
      - cat (self)                 - cat (self)
      - sat (right)                - (cannot see right)
      - mat (right)

    Full bidirectional context    Left-only causal mask
```

---

### **5. Mathematical formulation**

**Encoder architecture:**

Stack of L identical transformer blocks. Each block contains:

1. Multi-head self-attention (bidirectional, no mask)
2. Feed-forward network
3. Residual connections + layer normalization

**Bidirectional self-attention (no causal mask):**

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

No masking—every token can attend to every other token, including future positions.

**Input representation:**

For token sequence t₁...t_n, input embeddings:

$$
\mathbf{e}_i = \text{TokenEmbed}(t_i) + \text{PositionEmbed}(i) + \text{SegmentEmbed}(s_i)
$$

Where s_i indicates which sentence (for NSP tasks).

**[CLS] token:**

Special token prepended to every input. Its output representation h_CLS is used for classification tasks.

**Pre-training objectives:**

1. **Masked Language Modeling (MLM):** Predict randomly masked tokens (15% of positions)
2. **Next Sentence Prediction (NSP, optional):** Predict whether sentence B follows sentence A (used in BERT, dropped in RoBERTa)

**Fine-tuning for classification:**

Add a classification head on top of h_CLS:

$$
P(y | x) = \text{softmax}(\mathbf{W} \mathbf{h}_{\text{CLS}} + \mathbf{b})
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Input sentence**

Text: "The cat sat"

Tokenization: [CLS] The cat sat [SEP]

#### **Step 2: Token embedding (simplified 4-dim)**

[CLS] → [0.1, 0.2, 0.1, 0.3]
The → [0.4, 0.1, 0.2, 0.1]
cat → [0.2, 0.5, 0.1, 0.2]
sat → [0.3, 0.2, 0.4, 0.1]
[SEP] → [0.0, 0.1, 0.3, 0.2]

#### **Step 3: Position embeddings (same shape, added)**

[CLS] + pos0 → [0.15, 0.25, 0.15, 0.35]
...

#### **Step 4: Bidirectional self-attention (first layer)**

For "cat" position: attends to "The" (left), "cat" (self), "sat" (right), "[SEP]" (right). Produces a representation that integrates both left and right context.

#### **Step 5: After 12 layers**

Final representation h_CLS captures the meaning of the entire sentence.

#### **Step 6: Sentiment classification**

h_CLS → linear layer → sigmoid → score (0.85 positive, 0.15 negative) → Positive.

---

### **7. How this appears inside neural networks and LLMs**

- **BERT (2018):** The original encoder-only model. 110M parameters (base) or 340M (large). Pre-trained on BooksCorpus + English Wikipedia. Set state-of-the-art on GLUE benchmark.

- **RoBERTa (2019):** BERT but better. Trained longer, more data (160GB vs 16GB), no NSP, dynamic masking. Outperformed BERT significantly.

- **ALBERT (2019):** Parameter-efficient encoder. Shared parameters across layers, smaller embedding size. Reduced memory while maintaining performance.

- **DistilBERT (2019):** Distilled version of BERT (40% smaller, 60% faster, 97% performance). Common for edge deployment.

- **Modern use cases:** Encoder-only models are still state-of-the-art for classification, NER, information extraction, and semantic similarity (embedding models like BGE, E5).

- **Embedding models:** Many text embedding models (for retrieval) use encoder-only architecture with pooling (mean or [CLS]) to produce fixed-size vectors.

- **Why not decoder-only for understanding:** Decoder-only models have causal mask (cannot see right context), which is suboptimal for tasks requiring full bidirectional understanding. Encoder-only models are smaller, faster, and more accurate for pure understanding tasks.

---

### **8. Brain-like connection (global comprehension)**

The brain's language system does not read left-to-right like a decoder. It processes sentences holistically. When you read "The cat...", your brain does not commit to an interpretation until the end of the sentence. This is bidirectional processing—the brain maintains multiple hypotheses and resolves them when full context is available. Encoder-only models mirror this: they see the entire input and build representations that integrate both directions. Patients with left-hemisphere damage (Broca's aphasia) can still comprehend language globally even if production is impaired—like an encoder-only model that understands but cannot generate.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Encoder-only models are obsolete. Decoder-only models (GPT) can do everything BERT can do, and more."

_Why it is wrong:_ Decoder-only models are autoregressive—they predict the next token. For classification, they must prompt-engineer (e.g., "Sentiment: Positive/Negative") which is slower and often less accurate than encoder-only. For extracting embeddings, decoder-only models are suboptimal because causal masking limits left context for early tokens. On GLUE benchmark (classification tasks), encoder-only models still outperform decoder-only models of similar size. For pure understanding tasks (no generation), encoder-only models are smaller, faster, and more accurate. They are not obsolete—they are the right tool for many jobs. Use BERT for classification, GPT for generation.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Not every AI task requires text generation. Often,      |
|  you just need to understand: Is this email spam? Does   |
|  this review mention a refund? What companies are in     |
|  this article? For these tasks, encoder-only models are  |
|  faster, smaller, and more accurate than GPT. BERT       |
|  revolutionized NLP in 2018, and its descendants remain  |
|  state-of-the-art for understanding. Before ChatGPT,     |
|  there was BERT—and BERT is still the better choice      |
|  when you need comprehension, not creation.              |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are building a system to classify customer support emails into 10 categories (refund, technical issue, account question, etc.). You have 50,000 labeled examples.

**Question:** Would you use an encoder-only model (BERT) or a decoder-only model (GPT)? Why? If you chose decoder-only, how would you adapt it for classification?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Use encoder-only (BERT). Classification is a pure understanding task—no generation required. BERT processes the entire email bidirectionally and outputs a [CLS] representation that can be fed into a 10-class classifier. It is smaller, faster, and more accurate.

If you insist on decoder-only (GPT), you would need to prompt-engineer: "Classify this email into one of: refund, technical issue, account question, ... Email: [content]\nCategory:" and hope the model outputs the correct category. This is slower (generates multiple tokens), less reliable (may output "refund" or "The category is refund" or "I think it's refund"), and requires more compute. GPT may also have a smaller effective context window due to causal masking. For classification, encoder-only is the superior choice. This is why production classification systems use BERT, not GPT.
