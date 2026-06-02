# Seq2Seq models

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Sequence Models (Pre-Transformer)**

---

### **1. Why this concept matters**

An RNN can map a sequence to a single vector (sentiment) or a vector to a sequence (image captioning). But what about mapping a sequence to another sequence of different length? Translation: "The cat sat" (3 words) → "Le chat était assis" (4 words). Summarization: 500 words → 50 words. This requires a new architecture: one RNN reads the input sequence into a fixed vector (encoder), and another RNN writes out the output sequence from that vector (decoder). Seq2Seq models were the breakthrough that made neural machine translation practical and directly led to the encoder-decoder architecture used in modern transformers.

---

### **2. Core idea**

**A Seq2Seq model uses two RNNs: an encoder that compresses the input sequence into a fixed-size context vector, and a decoder that generates the output sequence from that context vector, one token at a time.**

---

### **3. Concrete analogy**

Imagine a simultaneous interpreter at the UN. Someone speaks a sentence in Spanish. The interpreter listens to the entire sentence (encoder), compresses it into understanding (context vector), then speaks the English translation (decoder). The interpreter cannot start translating until the speaker finishes the sentence—because the full meaning may only be clear at the end.

Now imagine the interpreter has a terrible memory. After hearing a long sentence, they forget the beginning. The translation becomes inaccurate. This is the bottleneck of basic Seq2Seq: the context vector must compress everything, and it fails for long sentences.

The solution (attention, covered next) was to let the decoder look back at all encoder hidden states, not just the final one. But the core architecture—encoder + decoder—remains the foundation of modern translation and summarization systems.

---

### **4. ASCII diagram**

```
Seq2Seq architecture for translation:

    INPUT: "The cat sat" (English)

    Encoder (RNN):
    "The"    "cat"    "sat"
      ↓        ↓        ↓
    h₁  →    h₂  →    h₃
                      │
                      ↓
                 Context vector
                 (final encoder state)
                      │
                      ↓
    Decoder (RNN):    │
                      │
    Start token ──→ h'_1 ──→ "Le"
                      │
    "Le" ──────────→ h'_2 ──→ "chat"
                      │
    "chat" ────────→ h'_3 ──→ "était"
                      │
    "était" ───────→ h'_4 ──→ "assis"
                      │
    "assis" ───────→ h'_5 ──→ End token

    OUTPUT: "Le chat était assis" (French)


Detailed data flow:

    Input:    x₁    x₂    x₃
              ↓     ↓     ↓
    Encoder:  RNN → RNN → RNN → context = h₃
                           │
                           ↓
    Decoder:  <START> → RNN → y₁ (Le)
                      │
                      ↓
              y₁    → RNN → y₂ (chat)
                      │
                      ↓
              y₂    → RNN → y₃ (était)
                      │
                      ↓
              y₃    → RNN → y₄ (assis)
```

---

### **5. Mathematical formulation**

**Encoder:** Processes input sequence x₁,...,x_T

$$
\mathbf{h}_t = \text{RNN}_{\text{enc}}(\mathbf{x}_t, \mathbf{h}_{t-1})
$$

**Context vector:** Final encoder hidden state (or concatenation/sum of multiple)

$$
\mathbf{c} = \mathbf{h}_T
$$

(Unicode: c = h_T)

**Decoder initial state:** Usually context vector (or transformed version)

$$
\mathbf{s}_0 = \mathbf{c}
$$

**Decoder:** Generates output sequence y₁,...,y_T' one step at a time

$$
\mathbf{s}_t = \text{RNN}_{\text{dec}}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1})
$$

**Output probabilities:**

$$
P(y_t | y_{<t}, \mathbf{c}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t + \mathbf{b}_o)
$$

**Training (teacher forcing):** Feed ground-truth previous token y\_{t-1} (not the model's own prediction)

**Inference (decoding):** Feed the model's own previous prediction. Common strategies:

- Greedy: pick highest probability each step
- Beam search: keep top K candidates at each step

**Loss (cross-entropy):**

$$
L = -\sum_{t=1}^{T'} \log P(y_t | y_{<t}, \mathbf{c})
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define a tiny Seq2Seq**

Encoder and decoder: 2D hidden states (simplified)
Input: "A B" (2 words) → Output: "X Y" (2 words)

#### **Step 2: Encoder processes input**

h₀ = [0, 0]
h₁ = tanh(W·[x_A, h₀]) = [0.6, 0.2] (after seeing "A")
h₂ = tanh(W·[x_B, h₁]) = [0.8, 0.5] (after seeing "B")
Context c = h₂ = [0.8, 0.5]

#### **Step 3: Decoder initial state**

s₀ = c = [0.8, 0.5]

#### **Step 4: Decoder step 1 (generate first output)**

Input to decoder: <START> token (vector [0,0])
s₁ = tanh(W_dec·[<START>, s₀]) = [0.7, 0.3]
Output logits: W_o × s₁ = [2.1, 1.2, 0.5] for vocabulary {X, Y, END}
Softmax probabilities: P(X)=0.65, P(Y)=0.25, P(END)=0.10
Pick X (greedy). y₁ = "X"

#### **Step 5: Decoder step 2 (generate second output)**

Input: y₁ = "X" (vector [1,0])
s₂ = tanh(W_dec·[x_X, s₁]) = [0.5, 0.9]
Output logits: [1.2, 2.5, 0.3] → P(X)=0.20, P(Y)=0.75, P(END)=0.05
Pick Y. y₂ = "Y"

#### **Step 6: Decoder step 3 (generate END or continue)**

Input: y₂ = "Y"
s₃ = tanh(...) = [0.2, 0.4]
Output: P(END)=0.85, stop.

Final output: "X Y"

#### **Step 7: Compare to greedy vs beam search**

Greedy gave "X Y". Beam search (width=2) might explore "X END" or "Y X" as alternatives, potentially finding better translations. For short sequences, greedy works fine.

---

### **7. How this appears inside neural networks and LLMs**

- **Neural machine translation:** Seq2Seq with attention was Google Translate's backbone from 2016 to ~2020 (before transformers).

- **Text summarization:** Encoder reads full document, decoder generates summary. Long sequences → attention essential.

- **Speech recognition:** Encoder processes audio frames, decoder generates text transcript.

- **Image captioning:** CNN encoder (not RNN) processes image, RNN decoder generates caption. Same encoder-decoder principle.

- **Transformer is a Seq2Seq:** The original transformer paper used encoder-decoder architecture for translation. GPT (decoder-only) dropped the encoder. T5 and BART kept encoder-decoder.

- **Teacher forcing vs scheduled sampling:** Teacher forcing (using ground-truth during training) causes exposure bias—model never sees its own mistakes during training, so errors compound at inference. Scheduled sampling gradually mixes in model's own predictions.

- **Beam search in LLMs:** Modern LLMs use beam search (or its variants) for generation, not just greedy decoding.

---

### **8. Brain-like connection (delayed production)**

Seq2Seq models mirror how humans produce language. You listen to a full sentence (encoding), formulate a mental representation (context), then produce a response (decoding). You do not start speaking the translation after the first word—you wait for the complete input. The bottleneck of the fixed context vector mirrors human memory limitations: long sentences are harder to translate because you forget the beginning. Professional interpreters use note-taking (external memory) to compensate—analogous to attention, which lets the decoder revisit the input. Seq2Seq with attention is a computational model of human translation with memory aids.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Seq2Seq models are obsolete. Transformers replaced them completely."

_Why it is wrong:_ The encoder-decoder architecture is not obsolete—it is the foundation of many modern models. T5, BART, and the original transformer are all Seq2Seq models. What changed is the RNN inside the encoder and decoder was replaced by attention. The high-level architecture (encode input → compress → decode output) is still the standard for translation, summarization, and any task where input and output sequences have different lengths. Seq2Seq is a pattern, not a specific RNN implementation. Modern LLMs still use Seq2Seq when they encode a prompt and generate a response.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Seq2Seq taught us that mapping between sequences of      |
|  different lengths requires two separate roles: one to    |
|  read and understand (encoder), one to write and express  |
|  (decoder). This separation is now baked into every LLM   |
|  that takes a prompt and generates a response. The        |
|  encoder-decoder pattern is not a historical footnote.    |
|  It is how machines translate, summarize, and answer.     |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are training a Seq2Seq model for translation. During training, you always feed the correct previous word to the decoder (teacher forcing). At inference, you feed the model's own previous predictions.

**Question:** What problem does this create? Name one technique to mitigate it.

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Exposure bias. The model learns to correct its mistakes when given the ground truth, but during inference it receives its own (potentially incorrect) predictions, leading to error accumulation. Early mistakes cause more mistakes later.

Mitigation techniques:

- Scheduled sampling: gradually replace ground truth with model's own predictions during training.
- Professor forcing: use adversarial training to make training and inference distributions similar.
- Beam search: explore multiple candidates instead of greedy decoding.
- Reinforcement learning: directly optimize for final translation quality, not per-step cross-entropy.
