# Encoder vs decoder architecture

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Transformers (The LLM Revolution)**

---

### **1. Why this concept matters**

Not all sequence tasks are the same. Sometimes you need to understand a complete input (classification, NER). That is an encoder task. Sometimes you need to generate an output from scratch (story writing, code generation). That is a decoder task. Sometimes you need to transform one sequence into another (translation, summarization). That requires both: an encoder to read the input, a decoder to write the output. The transformer architecture offers three variants: encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5, BART). Understanding the difference is crucial for choosing the right model for your task.

---

### **2. Core idea**

**Encoder-only models use bidirectional self-attention to build rich representations of an input sequence; decoder-only models use masked self-attention to generate sequences autoregressively; encoder-decoder models combine both, using an encoder to process the input and a decoder to generate the output with cross-attention to the encoder.**

---

### **3. Concrete analogy**

Three types of translators:

- **Encoder-only (BERT):** Listens to a sentence in Spanish and tells you the sentiment (happy/sad). Does not speak. This is understanding, not generation.

- **Decoder-only (GPT):** Listens to a prompt and continues speaking. It never sees the full future—it can only look at words it has already written. Great for creative writing, bad for translation because it cannot see the full input before starting.

- **Encoder-decoder (T5):** Listens to the full Spanish sentence, understands it completely (encoder), then speaks the English translation (decoder). It can look at the whole input before generating any output. Best for translation, summarization, and any task where output depends on a complete input.

The decoder-only model (GPT) became dominant for LLMs because the "input" is just a prompt, and the "output" is the continuation. But for tasks like translation, the encoder-decoder remains superior.

---

### **4. ASCII diagram**

```
Three transformer architectures:

1. Encoder-only (BERT, RoBERTa)

   Input: "The cat sat"
       │
   ┌───┴───┐
   │Encoder│ (bidirectional attention)
   └───┬───┘
       │
   Output: [cls] vector for classification

   • Sees full context left and right
   • No masking
   • Used for: sentiment, NER, embedding


2. Decoder-only (GPT, Llama)

   Input: "The cat" → predicts "sat"
       │
   ┌───┴───┐
   │Decoder│ (causal/masked attention)
   └───┬───┘
       │
   Output: "sat" (next token)

   • Sees only past tokens (causal mask)
   • Autoregressive generation
   • Used for: language modeling, chat, code generation


3. Encoder-Decoder (T5, BART)

   Input: "The cat sat" (German? No, English)
       │
   ┌───┴───┐
   │Encoder│ (bidirectional)
   └───┬───┘
       │
   (cross-attention)
       │
   ┌───┴───┐
   │Decoder│ (causal + cross-attention)
   └───┬───┘
       │
   Output: "Die Katze saß" (German translation)

   • Encoder sees full input
   • Decoder generates output step by step
   • Used for: translation, summarization, text-to-text
```

---

### **5. Mathematical formulation**

**Encoder-only (bidirectional self-attention):**

For all positions i, j in input sequence (no mask):

$$
\mathbf{X}_{\text{out}} = \text{Attention}(\mathbf{X}, \mathbf{X}, \mathbf{X})
$$

(Query, Key, Value all from input X. Every token attends to every token.)

**Decoder-only (causal self-attention):**

Masked so position i cannot attend to j > i:

$$
\mathbf{X}_{\text{out}}^{(i)} = \text{Attention}(\mathbf{X}^{(i)}, \mathbf{X}^{(\leq i)}, \mathbf{X}^{(\leq i)})
$$

Where the mask sets attention scores to -∞ for future positions:

$$
\text{Mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

**Encoder-decoder:**

Encoder (bidirectional):

$$
\mathbf{H}_{\text{enc}} = \text{Encoder}(\mathbf{X}_{\text{input}})
$$

Decoder (causal + cross-attention):
At each step t, decoder attends to:

- Previous decoder outputs (causal self-attention)
- All encoder hidden states (cross-attention)

$$
\mathbf{s}_t = \text{Decoder}(\mathbf{y}_{<t}, \mathbf{H}_{\text{enc}})
$$

Cross-attention:

$$
\text{CrossAttn}(\mathbf{s}_t, \mathbf{H}_{\text{enc}}, \mathbf{H}_{\text{enc}})
$$

**Summary of attention patterns:**

| Architecture    | Self-attention mask    | Cross-attention to encoder    |
| --------------- | ---------------------- | ----------------------------- |
| Encoder-only    | None (bidirectional)   | No                            |
| Decoder-only    | Causal (future masked) | No                            |
| Encoder-decoder | Causal (in decoder)    | Yes (decoder queries encoder) |

---

### **6. Worked example (step-by-step)**

#### **Step 1: Encoder-only (BERT)**

Input: "cat sat"

Position 1 ("cat") attends to: "cat" (itself), "sat" (future)
Position 2 ("sat") attends to: "cat" (past), "sat" (itself)

Result: "sat" knows about "cat" before outputting anything. Great for understanding.

#### **Step 2: Decoder-only (GPT) for next token prediction**

Input so far: "cat"

Position 1 ("cat") attends only to itself (no future)
Model predicts next token: "sat"

After appending "sat":
Position 2 ("sat") attends to: "cat", "sat" (past and self, but not future beyond)

Result: "cat" never sees "sat" before predicting it. Great for generation.

#### **Step 3: Encoder-decoder for translation**

Input (English): "cat sat"

Encoder processes "cat" and "sat" bidirectionally → produces hidden states H₁, H₂

Decoder starts with start token. At step 1, queries H (all encoder positions) via cross-attention → attends to "cat" strongly → outputs "Katze" (German for cat)

At step 2, decoder queries H again → may attend to "sat" strongly → outputs "saß"

Result: Decoder can look at the entire English sentence before generating each German word. Best for translation.

---

### **7. How this appears inside neural networks and LLMs**

- **BERT (2018):** Encoder-only. State-of-the-art for classification, NER, question answering. Not for generation.

- **GPT (2018-2024):** Decoder-only. Dominant for chat, code, creative writing. Scales better than encoder-decoder because attention is simpler (no cross-attention).

- **T5 (2019):** Encoder-decoder. Converts every NLP task to text-to-text. Still state-of-the-art for translation and summarization.

- **BART (2019):** Encoder-decoder pre-trained with denoising (corrupt input, reconstruct). Good for generation tasks.

- **Why decoder-only won for LLMs:** Language modeling (predict next token) is naturally decoder-only. Scaling laws show decoder-only trains efficiently. Cross-attention in encoder-decoder adds parameters and compute without clear benefit for pure generation.

- **Encoder-decoder for reasoning:** Some tasks (math word problems, code generation from spec) may benefit from bidirectional encoding of the problem before generation.

- **PrefixLM (hybrid):** Models like GLM (ChatGLM) and some versions of PaLM use a prefix: some tokens see full bidirectional attention (encoder-like), others are causal (decoder-like). Blends both paradigms.

---

### **8. Brain-like connection (reading vs speaking)**

The brain separates comprehension (encoder) from production (decoder). Wernicke's area in the temporal lobe processes heard language (bidirectional context). Broca's area in the frontal lobe generates spoken language (sequential, causal). These areas are connected by the arcuate fasciculus—the brain's "cross-attention" pathway, allowing the production system to query the comprehension system. Damage to Wernicke's causes fluent but meaningless speech (can produce words, cannot understand). Damage to Broca's causes halting, effortful speech (understands but cannot generate properly). This is the biological analog of encoder-decoder separation: understanding and generation are distinct but connected.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Decoder-only models (GPT) can do everything encoder-decoder models can do. Encoder-decoder is obsolete."

_Why it is wrong:_ For pure generation (given a prompt, continue), decoder-only is perfect. But for tasks where the output depends on a complete input (translation, summarization), encoder-decoder still outperforms. The bidirectional encoder captures context that a causal decoder cannot. GPT-4 can translate because it has seen many translation examples during training, but for the same parameter count, an encoder-decoder (e.g., T5-XXL) often achieves better translation BLEU scores. The choice is task-dependent. For chat, decoder-only wins. For translation, encoder-decoder wins. Neither is obsolete.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Not every task is next-token prediction. Sometimes you   |
|  need to understand the whole input before producing      |
|  output (encoder-decoder). Sometimes you just need to     |
|  generate (decoder-only). Sometimes you only need to      |
|  understand (encoder-only). GPT dominates the popular     |
|  imagination, but BERT and T5 are still the right tools   |
|  for many jobs. Know the difference. Choose the           |
|  architecture that matches your task, not the hype.       |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You need to build a model for:
A) Predicting the next word in a sentence
B) Classifying movie reviews as positive/negative
C) Translating English to Japanese

**Question:** Which architecture (encoder-only, decoder-only, or encoder-decoder) would you choose for each task? Why?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

A) Decoder-only (GPT). Next word prediction is autoregressive generation by definition. Causal masking ensures the model cannot cheat by seeing the future.

B) Encoder-only (BERT). Classification requires understanding the entire input. Bidirectional attention captures full context. No generation needed. Take the [CLS] token or pool over all token representations.

C) Encoder-decoder (T5). Translation requires reading the entire source sentence (bidirectional encoder) and generating the target sentence step by step (causal decoder with cross-attention). Decoder-only models can translate as a special case of next-token prediction (prefix: "English: ... Japanese: ..."), but encoder-decoder is more parameter-efficient and often more accurate for this task.
