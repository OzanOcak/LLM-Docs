# Bidirectional RNNs

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Sequence Models (Pre-Transformer)**

---

### **1. Why this concept matters**

A standard RNN reads a sentence from left to right. By the time it reaches the word "bank," it has seen "river" but not "account." This is a problem because meaning often depends on future context. In "I went to the bank to deposit money," the word "deposit" disambiguates "bank" from the financial institution, not the river. Bidirectional RNNs solve this by reading the sequence in both directions simultaneously—forward and backward—then combining both perspectives. This simple idea dramatically improved performance on any task where the entire sequence is available at once: sentiment analysis, named entity recognition, and sequence labeling.

---

### **2. Core idea**

**A bidirectional RNN processes a sequence with two separate RNNs: one forward (left to right) and one backward (right to left), then concatenates their hidden states at each position to produce a context-aware representation.**

---

### **3. Concrete analogy**

Imagine you are translating a sentence from a foreign language. You read the entire sentence, then go back and translate each word with full context. That is bidirectional processing.

Now imagine you are in a conversation where someone says: "I'm going to the bank..." You cannot tell if they mean river bank or financial bank until you hear the rest: "...to deposit a check" or "...to fish for trout." Your brain processes language bidirectionally—you hold the ambiguous word in memory, wait for future context, then reinterpret it.

A standard RNN is like reading aloud without looking ahead. You commit to "bank" as river bank immediately, then stumble when you hear "deposit." A bidirectional RNN reads the whole sentence first, then understands each word with both past and future context.

---

### **4. ASCII diagram**

```
Bidirectional RNN processing a 4-word sentence:

Forward RNN (left to right):
    x₁ ──→ h₁_f ──→ h₂_f ──→ h₃_f ──→ h₄_f
           ↑        ↑        ↑        ↑
    (start)   x₁      x₂      x₃      x₄

Backward RNN (right to left):
    x₄ ──→ h₄_b ──→ h₃_b ──→ h₂_b ──→ h₁_b
           ↑        ↑        ↑        ↑
    (start)   x₄      x₃      x₂      x₁

Combine at each position:

Position 1: h₁ = [h₁_f, h₁_b]  (forward + backward)
Position 2: h₂ = [h₂_f, h₂_b]
Position 3: h₃ = [h₃_f, h₃_b]
Position 4: h₄ = [h₄_f, h₄_b]


Visualization:

    Forward:  h₁_f → h₂_f → h₃_f → h₄_f
              │      │      │      │
    Input:    x₁     x₂     x₃     x₄
              │      │      │      │
    Backward: h₁_b ← h₂_b ← h₃_b ← h₄_b

    Output:   y₁     y₂     y₃     y₄
    (each y_t uses both h_t_f and h_t_b)
```

---

### **5. Mathematical formulation**

**Forward RNN (from t = 1 to T):**

$$
\overrightarrow{h}_t = \text{RNN}_{\text{fwd}}(\mathbf{x}_t, \overrightarrow{h}_{t-1})
$$

(Unicode: h*t_fwd = RNN_fwd(x_t, h*{t-1}\_fwd))

**Backward RNN (from t = T down to 1):**

$$
\overleftarrow{h}_t = \text{RNN}_{\text{bwd}}(\mathbf{x}_t, \overleftarrow{h}_{t+1})
$$

(Unicode: h*t_bwd = RNN_bwd(x_t, h*{t+1}\_bwd))

**Concatenated hidden state at position t:**

$$
\mathbf{h}_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t]
$$

(Unicode: h_t = [h_t_fwd, h_t_bwd] concatenated)

**Output at position t (for sequence labeling):**

$$
\mathbf{y}_t = \mathbf{W}_y \mathbf{h}_t + \mathbf{b}_y
$$

**For classification of entire sequence (e.g., sentiment):**

Use final states from both directions:

$$
\mathbf{h}_{\text{final}} = [\overrightarrow{h}_T ; \overleftarrow{h}_1]
$$

**Variants:**

- **Bidirectional LSTM/GRU:** Same idea, but use LSTM or GRU cells instead of vanilla RNN.

- **Deep bidirectional:** Stack multiple bidirectional layers. Requires careful handling (future information leaking into past is not allowed in causal tasks).

---

### **6. Worked example (step-by-step)**

#### **Step 1: Set up the sentence**

Sentence: "The cat sat" (3 words, 2D embeddings for simplicity)

Word vectors:
x₁ = "The" = [1, 0]
x₂ = "cat" = [0, 1]
x₃ = "sat" = [1, 1]

#### **Step 2: Forward RNN (simplified, just for intuition)**

Start with h₀_fwd = [0, 0]

h₁_fwd = tanh(0.5×x₁ + 0.3×h₀_fwd) ≈ [0.46, 0]
h₂_fwd = tanh(0.5×x₂ + 0.3×h₁_fwd) ≈ [0, 0.46] + 0.3×[0.46,0] ≈ [0.14, 0.46]
h₃_fwd = tanh(0.5×x₃ + 0.3×h₂_fwd) ≈ [0.5, 0.5] + 0.3×[0.14,0.46] ≈ [0.54, 0.64]

#### **Step 3: Backward RNN**

Start with h₄_bwd = [0, 0] (after last word)

Process from x₃ to x₁:

h₃_bwd = tanh(0.5×x₃ + 0.3×h₄_bwd) = [0.46, 0.46]
h₂_bwd = tanh(0.5×x₂ + 0.3×h₃_bwd) = [0, 0.46] + 0.3×[0.46,0.46] ≈ [0.14, 0.60]
h₁_bwd = tanh(0.5×x₁ + 0.3×h₂_bwd) = [0.5, 0] + 0.3×[0.14,0.60] ≈ [0.54, 0.18]

#### **Step 4: Concatenate at each position**

h₁ = [h₁_fwd, h₁_bwd] = [[0.46,0], [0.54,0.18]] → 4D vector
h₂ = [h₂_fwd, h₂_bwd] = [[0.14,0.46], [0.14,0.60]]
h₃ = [h₃_fwd, h₃_bwd] = [[0.54,0.64], [0.46,0.46]]

#### **Step 5: Interpret**

For word "cat" (position 2), h₂ contains:

- Forward context: info from "The" + "cat"
- Backward context: info from "sat" + "cat"

The bidirectional representation captures the full sentence context around each word, not just the left side. This is essential for tasks like named entity recognition where "Apple" could be a company or a fruit depending on following words.

---

### **7. How this appears inside neural networks and LLMs**

- **BERT (and all encoder-only models):** The bidirectional transformer is a direct generalization of bidirectional RNNs. BERT reads entire sequences in both directions simultaneously via self-attention.

- **Named Entity Recognition (NER):** Identifying "Apple Inc." vs "apple fruit" requires future context. BiLSTM-CRF was the state-of-the-art pre-BERT.

- **Part-of-speech tagging:** The tag for "bank" depends on surrounding words. Bidirectional context dramatically improves accuracy.

- **Sentiment analysis for entire sentence:** Use final states [h_T_fwd, h_1_bwd] to capture full sentence sentiment.

- **Why not bidirectional in GPT (decoders):** GPT is autoregressive—it predicts the next word. Using future words would be cheating (information leak). Bidirectional only works when the entire sequence is available at once (encoders, not decoders).

- **Bidirectional in speech recognition:** Audio is processed as a sequence of frames. Future frames provide context (e.g., a phoneme depends on following phonemes). Bidirectional RNNs were state-of-the-art pre-transformer.

---

### **8. Brain-like connection (contextual integration)**

The brain processes language bidirectionally. When you read "The man who lives next door is a doctor," the verb "is" depends on "man" (singular), not on "neighbors" or "door." Your brain integrates information from both directions. Functional MRI shows that language processing involves both left-to-right incremental processing and right-to-left reanalysis. Broca's area and the temporal lobe maintain a representation of the entire sentence while you parse. Bidirectional RNNs mirror this: they do not commit to an interpretation until both past and future context have been considered. The brain is not purely autoregressive—it waits for disambiguating information before finalizing meaning.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Bidirectional RNNs are always better than unidirectional RNNs. Why would anyone use a unidirectional RNN?"

_Why it is wrong:_ Bidirectional RNNs cannot be used for real-time or streaming tasks. In speech recognition on a live conversation, you do not have future frames—the future has not happened yet. In language modeling (predicting the next word), using future words would be cheating. In robotics, you cannot see the future sensor readings. Bidirectional processing requires the entire sequence in advance. For offline tasks (sentiment analysis on a complete review, NER on a complete document), bidirectional is superior. For online tasks, unidirectional is the only option. Choose based on your use case, not reflex.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Context is not just about the past. The future also      |
|  disambiguates meaning. Bidirectional RNNs taught us      |
|  that the best representation of a word comes from both   |
|  sides of the sentence. This idea directly inspired       |
|  BERT and modern bidirectional encoders. If you only      |
|  look backward, you see half the picture. Bidirectional   |
|  processing is not an architecture trick—it is how        |
|  understanding actually works.                            |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are building a model to predict the next word in a sentence (language modeling). Can you use a bidirectional RNN?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** No. In language modeling (autoregressive generation), at time t you only have access to words 1 through t-1. Using future words (t+1, t+2, etc.) would leak information from the future into the present—the model would "cheat" by seeing the answer. For generation, you must use a unidirectional (causal) RNN. Bidirectional RNNs are only for tasks where the entire input sequence is available at once, such as classification of a complete sentence, named entity recognition, or sentiment analysis on a full review. This is why GPT (decoder) is unidirectional and BERT (encoder) is bidirectional.
