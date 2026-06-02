# Decoder-only (GPT-like)

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Model Architecture Variants**

---

### **1. Why this concept matters**

If you have used ChatGPT, you have used a decoder-only model. This architecture dominates modern LLMs: GPT-3, GPT-4, Llama, Mistral, Claude, Gemini—all decoder-only. Why? Because language is inherently generative. The task of predicting the next token (autoregressive language modeling) is both the pre-training objective and the final use case. Decoder-only models are simpler than encoder-decoder (no cross-attention) and more flexible than encoder-only (they can generate). They scale with data and compute better than any other architecture. Understanding decoder-only means understanding the architecture behind the LLM revolution.

---

### **2. Core idea**

**A decoder-only model uses a stack of transformer decoder blocks with causal (masked) self-attention, predicting the next token autoregressively given previous tokens, and is trained via next-token prediction on massive text corpora.**

---

### **3. Concrete analogy**

Imagine a writer composing a story. They start with the first word, then the second, then the third. Each new word depends on everything written so far. They cannot see the future—only the past. The writer is autoregressive.

Now imagine the writer has an exceptional memory. They remember every word they have written and use that context to decide what comes next. That is a decoder-only model.

Unlike a translator (encoder-decoder) who sees the full source sentence before starting, or a classifier (encoder-only) who just understands, the decoder-only model's job is to continue. It is a storyteller, not an analyst. This makes it ideal for chat, code generation, creative writing, and any task where the output is a continuation of the input.

---

### **4. ASCII diagram**

```
Decoder-only architecture (GPT):

    Input tokens (prompt): "The cat sat"

    ┌─────────────────────────────────────────────────────────┐
    │                    Decoder Stack                        │
    │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐         │
    │  │ Layer │ → │ Layer │ → │ Layer │ → │ Layer │  (L×)   │
    │  │   1   │   │   2   │   │   3   │   │   L   │         │
    │  └───────┘   └───────┘   └───────┘   └───────┘         │
    │       ↑           ↑           ↑           ↑             │
    │  Causal self-attention (masked, future not visible)    │
    │  + Feed-forward network (per token)                    │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    Output logits (vocabulary size, e.g., 50,257)
    Predict next token: "on" or "down" or "."
                              │
                              ▼
    Sample/greedy decode → "on" → append to input → repeat


Causal attention mask (sequence length 4):

    Tokens:    t₁    t₂    t₃    t₄
    t₁:        ✓     ✗     ✗     ✗    (can only see itself)
    t₂:        ✓     ✓     ✗     ✗    (can see t₁ and t₂)
    t₃:        ✓     ✓     ✓     ✗    (can see t₁,t₂,t₃)
    t₄:        ✓     ✓     ✓     ✓    (can see all up to t₄)

    Row = query token, Column = key token
    ✓ = allowed to attend, ✗ = masked (-∞ in softmax)


Training objective (causal language modeling):

    Input:  "The cat sat on"
    Target: "cat sat on the"

    At each position, predict next token given previous tokens only.
```

---

### **5. Mathematical formulation**

**Causal self-attention:**

For position i, attention only to positions j ≤ i:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where M is the causal mask:

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

**Training objective (causal language modeling):**

Given sequence x₁...x_T, maximize:

$$
L_{\text{CLM}} = \sum_{t=1}^{T-1} \log P(x_{t+1} | x_1, ..., x_t)
$$

**Inference (autoregressive generation):**

Starting with prompt x₁...x_k, for t = k+1 to max_len:

$$
x_t \sim P(\cdot | x_1, ..., x_{t-1})
$$

**Positional encodings:**

Decoder-only models use absolute (GPT-2, GPT-3, Llama 1) or rotary (RoPE, Llama 2/3, GPT-NeoX) positional encodings.

**No cross-attention:**

Unlike encoder-decoder, decoder-only has no cross-attention blocks. All attention is self-attention within the sequence.

**Key architectural differences from encoder-only:**

| Feature             | Encoder-only (BERT)  | Decoder-only (GPT)     |
| ------------------- | -------------------- | ---------------------- |
| Attention mask      | None (bidirectional) | Causal (future masked) |
| Pre-training        | MLM + NSP            | Causal LM (next token) |
| Use case            | Understanding        | Generation             |
| Positional encoding | Absolute             | Absolute or RoPE       |

---

### **6. Worked example (step-by-step)**

#### **Step 1: Input prompt**

Prompt: "The capital of France is"

#### **Step 2: Tokenization (simplified)**

Tokens: ["The", "capital", "of", "France", "is"] (5 tokens)

#### **Step 3: Causal attention computation (simplified, first layer)**

For token "is" (position 5):

- Attends to tokens 1-5 (The, capital, of, France, is)
- Cannot attend to anything beyond (none yet)

For token "France" (position 4):

- Attends to tokens 1-4 (The, capital, of, France)
- Does NOT attend to "is" (future token, masked)

#### **Step 4: Generate next token**

Model output logits over vocabulary (50,000 words). Highest logits: "Paris" (0.85), "located" (0.10), "a" (0.03), etc.

Sample (or greedy): select "Paris"

#### **Step 5: Append and continue**

New sequence: ["The", "capital", "of", "France", "is", "Paris"]

Repeat: predict next token. Could be "." (end of sentence) or "and" etc.

#### **Step 6: Stop condition**

Model generates end-of-sequence token <|endoftext|> or reaches max length.

---

### **7. How this appears inside neural networks and LLMs**

- **GPT series:** GPT-1 (117M), GPT-2 (1.5B), GPT-3 (175B), GPT-4 (estimated >1T). All decoder-only. GPT-3 showed emergent few-shot learning at scale.

- **Llama (Meta):** Llama 1 (7B-65B), Llama 2 (7B-70B), Llama 3 (8B-70B-400B+). Decoder-only with RoPE positional encodings and SwiGLU activation.

- **Mistral (7B-12B):** Decoder-only with sliding window attention (more efficient for long contexts).

- **Gemini (Google):** Decoder-only (reportedly). Multi-modal (text + images + audio) but decoder-only architecture.

- **Claude (Anthropic):** Decoder-only with RLHF for safety.

- **Why decoder-only dominates:** Simpler architecture than encoder-decoder (no cross-attention), scales well with data, and the pre-training objective (next token prediction) is exactly the inference task. You can also use decoder-only for understanding tasks via prompt engineering (e.g., "Classify this sentiment: ...").

- **Scaling laws:** Decoder-only models exhibit predictable power-law scaling: loss decreases as compute, data, and parameters increase. This predictability is why companies invest billions in scaling.

---

### **8. Brain-like connection (left-to-right language production)**

Human language production is autoregressive. When you speak, you do not plan the entire sentence before starting. You produce one word at a time, with each word conditioned on what you have already said. Broca's area (frontal lobe) is responsible for this sequential production. Damage to Broca's area causes halting, effortful speech—you know what you want to say but cannot produce it fluently. This is analogous to a decoder-only model with a limited context window or faulty attention. The brain's language system is not bidirectional—comprehension is (Wernicke's area), but production is strictly causal. Decoder-only models mirror this: they generate left-to-right, with no future information.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Decoder-only models cannot do bidirectional understanding. They are worse than BERT for all NLP tasks."

_Why it is wrong:_ Decoder-only models can do understanding tasks via prompting. For sentiment analysis: "Classify this review as positive or negative: 'I loved this movie.' Sentiment:" The model generates "positive." With few-shot examples, decoder-only models match or exceed BERT on many understanding benchmarks. More importantly, decoder-only models can do _both_ understanding and generation, while BERT can only understand. This flexibility, combined with scaling laws, is why the industry has converged on decoder-only. For pure understanding tasks with no generation, BERT is still more efficient. But for general-purpose LLMs, decoder-only is the standard.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Decoder-only models are the engine of the generative AI  |
|  revolution. ChatGPT, Llama, Claude, Gemini—all decoder-  |
|  only. Their power comes from scaling: more data, more    |
|  parameters, more compute. The simple objective of next-  |
|  token prediction, when scaled to trillions of tokens,    |
|  produces models that can write code, answer questions,   |
|  and reason. Decoder-only is not just an architecture.    |
|  It is the architecture that taught machines to speak.    |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a prompt: "The first three prime numbers are 2, 3,"

**Question:** A decoder-only model continues this prompt. What is the next token likely to be? What information does the model need to have learned during pre-training to answer correctly?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** The next token is likely "5" (the third prime number).

The model needs to have learned during pre-training:

1. The definition of prime numbers (divisible only by 1 and itself)
2. The sequence of primes: 2, 3, 5, 7, 11, ...
3. Counting: "first, second, third" → position in sequence
4. The pattern: after "2, 3," the next is "5"

This is not memorization of a specific sentence (unlikely to appear exactly in training). The model generalizes from multiple examples of prime number sequences in textbooks, articles, and code. Decoder-only models learn these patterns through next-token prediction across trillions of tokens. This is why they can complete sequences correctly without explicit fine-tuning on prime numbers.
