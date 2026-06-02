# Encoder-decoder (T5-like)

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Model Architecture Variants**

---

### **1. Why this concept matters**

Decoder-only models (GPT) generate text from a prompt. Encoder-only models (BERT) understand text but cannot generate. But what about tasks where the input and output have different structures—translation, summarization, text-to-SQL? For these, you want an encoder to read the full input (bidirectional) and a decoder to generate the output (causal). Encoder-decoder models combine the best of both worlds. T5, BART, and the original transformer are encoder-decoder. While decoder-only dominates chat, encoder-decoder remains state-of-the-art for translation and summarization. Understanding this architecture is essential for tasks where input and output differ.

---

### **2. Core idea**

**An encoder-decoder model uses a bidirectional encoder to process the input sequence, a causal decoder to generate the output sequence autoregressively, and cross-attention between decoder and encoder to allow the decoder to access the full input at each generation step.**

---

### **3. Concrete analogy**

Imagine a simultaneous interpreter at the UN. They listen to the entire Spanish sentence (encoder—bidirectional), understanding every word in context. Then they speak the English translation (decoder—causal, one word at a time). While speaking, they can still look back at the original Spanish sentence (cross-attention) to check details.

This is different from:

- **Decoder-only (GPT):** Would have to translate on the fly, hearing one word at a time, like a tourist with a phrasebook.
- **Encoder-only (BERT):** Could understand the Spanish sentence but could not produce the English translation at all.

Encoder-decoder is purpose-built for sequence-to-sequence tasks: input sequence (source language, document, query) → output sequence (target language, summary, SQL query). This is why T5 (Text-to-Text Transfer Transformer) frames every NLP task as text-to-text: classification becomes "sentiment: positive" output, translation becomes "Hola" output, etc.

---

### **4. ASCII diagram**

```
Encoder-decoder architecture (T5, BART, original Transformer):

    Input sequence: "The cat sat on the mat"

    ┌─────────────────────────────────────────────────────────┐
    │                    Encoder (bidirectional)              │
    │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐         │
    │  │ Block │ → │ Block │ → │ Block │ → │ Block │  (L×)   │
    │  │   1   │   │   2   │   │   3   │   │   L   │         │
    │  └───────┘   └───────┘   └───────┘   └───────┘         │
    │       ↑           ↑           ↑           ↑             │
    │  Bidirectional self-attention (no mask)                │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼ (encoder outputs H)
    ┌─────────────────────────────────────────────────────────┐
    │                    Decoder (causal)                     │
    │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐         │
    │  │ Block │ → │ Block │ → │ Block │ → │ Block │  (L×)   │
    │  │   1   │   │   2   │   │   3   │   │   L   │         │
    │  └───────┘   └───────┘   └───────┘   └───────┘         │
    │       ↑           ↑           ↑           ↑             │
    │  Causal self-attention (future masked)                 │
    │  + Cross-attention to encoder outputs H                │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    Output sequence: "Le chat était assis sur le tapis"


Three attention mechanisms in encoder-decoder:

    1. Encoder self-attention (bidirectional)
    2. Decoder self-attention (causal)
    3. Cross-attention: decoder queries encoder keys/values


T5's text-to-text framework:

    Task prefix → Input → Output

    "translate English to German: The cat sat" → "Die Katze saß"
    "summarize: [long article]" → "short summary"
    "sentiment: This movie was great" → "positive"
    "colossus: 2 + 2 = 4" → "true"
```

---

### **5. Mathematical formulation**

**Encoder (bidirectional):**

For input sequence x₁...x_m:

$$
\mathbf{H}_{\text{enc}} = \text{Encoder}(\mathbf{x})
$$

Each encoder layer has bidirectional self-attention (no mask).

**Decoder (causal + cross-attention):**

For output sequence y₁...y_n (shifted right, start with <s>):

$$
\mathbf{s}_t = \text{Decoder}(\mathbf{y}_{<t}, \mathbf{H}_{\text{enc}})
$$

Each decoder layer has:

1. Causal self-attention (masked, future not visible)
2. Cross-attention: queries from decoder, keys/values from encoder

**Cross-attention:**

$$
\text{CrossAttn}(\mathbf{s}_t, \mathbf{H}_{\text{enc}}) = \text{softmax}\left(\frac{\mathbf{s}_t \mathbf{H}_{\text{enc}}^T}{\sqrt{d_k}}\right) \mathbf{H}_{\text{enc}}
$$

**Training objective:**

For sequence-to-sequence tasks:

$$
L_{\text{seq2seq}} = -\sum_{t=1}^{n} \log P(y_t | y_{<t}, \mathbf{x})
$$

**T5's pre-training (span corruption):**

Input: "Thank you for [X] me to your [Y] party."
Output: "inviting" (X), "birthday" (Y)

Corrupt input by replacing random spans with sentinel tokens, train decoder to reconstruct original text. This teaches the model to generate missing text conditioned on context.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Input (translation task)**

Prompt: "translate English to French: The cat sat"

#### **Step 2: Encoder processes input**

Input tokens: ["translate", "English", "to", "French", ":", "The", "cat", "sat"]

Encoder (bidirectional) produces H_enc: 8 vectors, each capturing full context.

For "cat", representation includes left context ("translate", "English", "to", "French", ":", "The") and right context ("sat").

#### **Step 3: Decoder starts generation**

Initial decoder input: <s> (start token)

First decoder step:

- Causal self-attention: only <s> (nothing else)
- Cross-attention: queries <s> against all encoder states H_enc
- Output: most attention on "translate", "The", "cat" → predicts "Le"

#### **Step 4: Decoder step 2**

Input: ["<s>", "Le"]

Causal self-attention: "Le" attends to "<s>" and itself.
Cross-attention: "Le" queries H_enc → focuses on "cat" → predicts "chat"

#### **Step 5: Decoder step 3**

Input: ["<s>", "Le", "chat"]

Cross-attention: "chat" queries H_enc → focuses on "sat" → predicts "était"

#### **Step 6: Continue**

Generates "assis" (sat), then "sur" (on), then "le" (the), then "tapis" (mat), then </s>.

Final output: "Le chat était assis sur le tapis"

---

### **7. How this appears inside neural networks and LLMs**

- **Original Transformer (Vaswani et al., 2017):** Encoder-decoder architecture. Designed for machine translation.

- **T5 (Google, 2019):** Text-to-Text Transfer Transformer. Encoder-decoder with 11B parameters. Framed all NLP tasks as text-to-text. State-of-the-art on GLUE, SuperGLUE, and many benchmarks.

- **BART (Facebook, 2019):** Encoder-decoder pre-trained with denoising (corrupt input, reconstruct). Excellent for summarization and generation.

- **T0 (Hugging Face, 2021):** Encoder-decoder instruction-tuned on 62 tasks. Strong zero-shot performance.

- **LongT5 (Google, 2021):** Encoder-decoder with efficient attention for long documents (summarization).

- **Modern usage:** Encoder-decoder still dominates translation, summarization, and text-to-SQL. Decoder-only models (GPT) are catching up via few-shot prompting, but encoder-decoder remains more efficient for these tasks.

- **Parameter efficiency:** Encoder-decoder has separate parameters for encoder and decoder (2×). Decoder-only has only decoder (1×). For large-scale general-purpose LLMs, decoder-only is more parameter-efficient. For specialized seq2seq tasks, encoder-decoder is optimal.

- **T5's "colossus" task:** T5 was trained on the "Colossal Clean Crawled Corpus" (C4)—750GB of cleaned web text. The span corruption objective (replace random spans with sentinel tokens) teaches the model to perform any text-to-text task.

---

### **8. Brain-like connection (Wernicke-Geschwind model)**

The brain's language system separates comprehension (Wernicke's area, temporal lobe) from production (Broca's area, frontal lobe). Wernicke's area processes input bidirectionally, building understanding. The arcuate fasciculus connects Wernicke's to Broca's. Broca's area generates output sequentially, one word at a time. This is encoder-decoder: Wernicke's is the encoder (bidirectional understanding), Broca's is the decoder (causal generation), and the arcuate fasciculus is the cross-attention pathway. Damage to Wernicke's causes fluent but meaningless speech (decoder works, encoder broken). Damage to Broca's causes halting but meaningful speech (encoder works, decoder broken). The brain evolved encoder-decoder architecture for language.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Encoder-decoder models are obsolete. Decoder-only models (GPT) can do translation and summarization just as well, with fewer parameters."

_Why it is wrong:_ For translation and summarization, encoder-decoder models are still more efficient and often more accurate at equal parameter counts. T5-11B outperformed GPT-3 175B on translation benchmarks when GPT-3 was released. Decoder-only models can do translation via prompting ("translate English to French: ..."), but they need to learn the task through examples (few-shot) or instruction tuning. Encoder-decoder is purpose-built for seq2seq, with bidirectional encoder providing full context and separate decoder parameters. For chat and open-ended generation, decoder-only wins. For translation, encoder-decoder still holds advantages. Neither is obsolete—they are tools for different jobs.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Translation, summarization, text-to-SQL—these tasks     |
|  require reading the full input before generating.        |
|  Encoder-decoder is designed for this. It separates       |
|  understanding (bidirectional encoder) from production    |
|  (causal decoder with cross-attention). T5 showed that    |
|  all NLP tasks can be framed as text-to-text. While       |
|  decoder-only models have taken the spotlight, encoder-   |
|  decoder remains the best choice when input and output    |
|  have different structures. Know the architecture,        |
|  choose the right tool.                                   |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are building a medical text summarization system. Input: 10,000-character clinical note. Output: 500-character summary. The input and output are in the same language (English) but have very different structures (narrative vs bullet points).

**Question:** Would you choose encoder-decoder (T5) or decoder-only (GPT)? Why? What advantage does encoder-decoder provide for this specific task?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Choose encoder-decoder (T5). For summarization, the encoder can read the entire clinical note bidirectionally, capturing dependencies across the full document. The decoder then generates the summary step by step, with cross-attention allowing it to refer back to relevant parts of the original note.

Decoder-only (GPT) would need to read the note left-to-right (causal), which is suboptimal for understanding long documents. It would also need to generate the summary autoregressively without a separate encoder, forcing it to store all information in a single hidden state. For long inputs (10,000 characters), decoder-only may struggle with the context window.

Encoder-decoder advantages for this task:

1. Bidirectional encoder captures full document context
2. Separate decoder parameters specialized for generation
3. Cross-attention allows focusing on relevant sections during summary generation
4. T5 was specifically pre-trained on span corruption, which mimics summarization (corrupt long input, reconstruct short output)

For summarization, encoder-decoder remains the established best practice, though decoder-only models (with sufficient scale) are competitive.
