# LSTMs (Long Short-Term Memory)

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Sequence Models (Pre-Transformer)**

---

### **1. Why this concept matters**

RNNs had a fatal flaw: they forgot. After 10-20 steps, the hidden state degraded into noise—early words in a sentence left no trace. This made RNNs useless for understanding long documents, processing audio, or any task where information must travel far. Long Short-Term Memory (LSTM) fixed this by introducing a separate "cell state" that flows like a river through the network, with gates that control what to remember, forget, and output. LSTMs were the dominant sequence model for over a decade and remain the standard for many time-series tasks today.

---

### **2. Core idea**

**An LSTM maintains two hidden vectors: a cell state (long-term memory) that flows through time with minimal modification, and a hidden state (short-term working memory) that is filtered from the cell state; three learned gates control when to forget, store, and output information.**

---

### **3. Concrete analogy**

Imagine you are an office assistant managing a filing cabinet.

- **Cell state (long-term memory):** The actual filing cabinet. Information stays until you explicitly remove it. It flows unchanged through time.

- **Hidden state (short-term memory):** Your desk. What you are actively working with right now. Cleared and updated at each step.

- **Forget gate:** Decides what to throw away from the filing cabinet. "I no longer need last quarter's reports."

- **Input gate:** Decides what new information to add. "This new email about Q4 should be filed."

- **Output gate:** Decides what to pull from the cabinet onto your desk for the current task. "For answering this question, I need the client's contact info."

In an RNN, the filing cabinet and the desk are the same thing. Every new input overwrites previous information. In an LSTM, the cabinet persists. Information can be stored for hundreds of steps and retrieved exactly when needed.

---

### **4. ASCII diagram**

```
LSTM cell at time step t:

        h_{t-1} (previous hidden)    x_t (current input)
                │                      │
                └──────────┬───────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      ▼                      │
    │   ┌─────────────────────────────────────┐  │
    │   │  Forget Gate    Input Gate    Output│  │
    │   │    σ             σ             σ    │  │
    │   └─────────────────────────────────────┘  │
    │                      │                      │
    │   C_{t-1} ──────┬───×───┐                  │
    │                 │       │                  │
    │                 ▼       ▼                  │
    │               Forget   +    tanh          │
    │                         │                  │
    │                         ▼                  │
    │                     C_t (new cell) ────┬──→ C_t
    │                                        │
    │                                        ▼
    │                                    × ← Output gate
    │                                        │
    └────────────────────────────────────────┼──→ h_t
                                              │
                                        (to next step)

Legend:
σ = sigmoid (outputs 0-1, controls flow)
× = elementwise multiplication
+ = elementwise addition
C = cell state (long-term memory)
h = hidden state (short-term memory)
```

---

### **5. Mathematical formulation**

**Forget gate:** Decides what to discard from cell state.

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

(Unicode: f*t = σ(W_f·[h*{t-1}, x_t] + b_f))

**Input gate:** Decides what new information to store.

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**Candidate cell state:** The new information to consider adding.

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

**Update cell state:** Forget old, add new.

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

(Unicode: C*t = f_t ⊙ C*{t-1} + i_t ⊙ \tilde{C}\_t)

**Output gate:** Decides what to output as hidden state.

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

**Hidden state (output of LSTM):**

$$
h_t = o_t \odot \tanh(C_t)
$$

Where:

- σ = sigmoid (outputs 0 to 1)
- ⊙ = elementwise multiplication
- [h_{t-1}, x_t] = concatenation of previous hidden and current input
- All W and b are learned parameters

---

### **6. Worked example (step-by-step)**

#### **Step 1: Initialize**

Cell state C₀ = [0, 0], hidden state h₀ = [0, 0] (2-dimensional LSTM for simplicity)

#### **Step 2: First input x₁ = [1.0, 0.5]**

Simplified: forget gate f₁ = [0.1, 0.2] (mostly keep old)
input gate i₁ = [0.9, 0.8] (mostly add new)
candidate C̃₁ = [0.7, -0.3]
output gate o₁ = [0.8, 0.6]

C₁ = f₁⊙C₀ + i₁⊙C̃₁ = [0,0] + [0.9×0.7, 0.8×(-0.3)] = [0.63, -0.24]

h₁ = o₁ ⊙ tanh(C₁) = [0.8×tanh(0.63), 0.6×tanh(-0.24)] ≈ [0.8×0.56, 0.6×(-0.23)] = [0.45, -0.14]

#### **Step 3: Second input x₂ = [0.2, 0.8]**

Forget gate f₂ = [0.05, 0.1] (forget almost nothing)
Input gate i₂ = [0.7, 0.9]
Candidate C̃₂ = [0.3, 0.6]
Output gate o₂ = [0.9, 0.7]

C₂ = f₂⊙C₁ + i₂⊙C̃₂ = [0.05×0.63, 0.1×(-0.24)] + [0.7×0.3, 0.9×0.6] = [0.0315, -0.024] + [0.21, 0.54] = [0.2415, 0.516]

h₂ = o₂ ⊙ tanh(C₂) = [0.9×tanh(0.2415), 0.7×tanh(0.516)] ≈ [0.9×0.236, 0.7×0.474] = [0.212, 0.332]

#### **Step 4: Interpret**

C₁ stored information from first input. C₂ added information from second input while preserving most of C₁ (forget gate was near zero). The cell state acts as a cumulative memory. The hidden state is a filtered, activated version for output.

---

### **7. How this appears inside neural networks and LLMs**

- **Pre-transformer era LSTMs:** State-of-the-art for machine translation, speech recognition, text generation, and sentiment analysis from ~1997 to 2017.

- **Language modeling:** LSTMs could generate coherent paragraphs (e.g., Karpathy's "The Unreasonable Effectiveness of RNNs"). They remembered subject-verb agreement across dozens of words.

- **Encoder-decoder (Seq2Seq):** One LSTM encodes input sentence into final cell state; another LSTM decodes into output sentence. Used in Google Translate pre-2017.

- **Why Transformers won:** LSTMs still have sequential processing (can't parallelize) and limited practical memory (~100-200 steps). Attention is O(1) path length and O(n²) compute—faster on GPUs for typical sequence lengths.

- **Current usage:** LSTMs still dominate time series forecasting (stock prices, sensor data), audio processing, and some edge deployments due to smaller memory footprint than transformers.

- **Bidirectional LSTMs:** Process sequence forward and backward, then concatenate hidden states. Captures context from both directions.

---

### **8. Brain-like connection (gated memory)**

The LSTM's gating mechanism mirrors biological memory. The hippocampus uses similar "gates" to encode new memories (input gate), consolidate existing ones (forget gate), and retrieve relevant memories (output gate). Dopamine signals modulate what is remembered versus forgotten. The cell state is analogous to long-term potentiation—synaptic strengths that persist for hours or days. The hidden state is analogous to working memory—transient neural activity that clears after a task. LSTMs were directly inspired by theories of hippocampal function, making them one of the few neural architectures explicitly designed after brain circuitry.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "LSTMs solve vanishing gradients completely. They can remember indefinitely."

_Why it is wrong:_ LSTMs mitigate but do not eliminate vanishing gradients. The cell state is updated additively (C*t = f_t⊙C*{t-1} + i_t⊙C̃_t), so gradients can flow through the cell state without repeated multiplication by small matrices. However, the forget gate can still learn to close, which cuts off gradient flow. In practice, LSTMs remember reliably for ~100-200 steps, not indefinitely. For truly long sequences (books, whole documents), even LSTMs fail. This is why transformers with explicit attention won—they have O(1) gradient paths regardless of sequence length.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  LSTMs taught us that memory needs structure. You cannot  |
|  just pass a hidden state forward and hope it survives.   |
|  You need separate pathways: one for long-term storage,   |
|  one for current work, and gates to control the flow.     |
|  This insight—that memory requires selective updating—    |
|  was the bridge from forgetful RNNs to the sophisticated  |
|  attention mechanisms of modern LLMs. Learn LSTMs and     |
|  you learn the foundations of learnable memory itself.    |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

An LSTM processes a 200-word sentence. The forget gate for a particular cell stays very close to 1.0 for all 200 steps.

**Question:** What happens to the information stored in that cell from word 5? Will it still be present at word 200? Why is this both powerful and potentially problematic?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** The information from word 5 will still be present at word 200 because forget gate near 1.0 means C*t ≈ C*{t-1} + i_t⊙C̃_t. The cell state accumulates information additively without decay.

This is powerful: long-range dependencies are preserved across hundreds of steps.

This is also problematic: the cell state can grow without bound (values can become huge), and the LSTM may never forget irrelevant information. The forget gate must learn when to close. Without forgetting, the cell becomes saturated with old, irrelevant information, drowning out new signals. Memory requires selective forgetting—not just unlimited storage.
