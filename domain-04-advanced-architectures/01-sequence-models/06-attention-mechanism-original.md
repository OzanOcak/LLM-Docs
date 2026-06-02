# Attention mechanism (original version)

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Sequence Models (Pre-Transformer)**

---

### **1. Why this concept matters**

Seq2Seq models had a fatal flaw: the encoder compressed the entire input into a single fixed vector. For long sentences, information from the beginning was lost—the decoder forgot the first words by the time it finished translating. Attention fixed this by giving the decoder direct access to every encoder hidden state, letting it "look back" at relevant parts of the input when generating each output word. This simple idea transformed machine translation from forgetting after 20 words to handling entire paragraphs. Attention is not just a patch for RNNs—it became the foundation of the transformer and every modern LLM.

---

### **2. Core idea**

**Attention allows a decoder to dynamically weigh different parts of the input sequence when generating each output token, computing a context vector as a weighted sum of encoder hidden states where weights depend on the decoder's current state.**

---

### **3. Concrete analogy**

Imagine you are translating a long document. The old way (no attention): you read the entire document, then close your eyes and translate from memory. By the end, you have forgotten the first page.

The attention way: you keep the document open. When translating a sentence, your eyes scan back over the source text, focusing on relevant parts. For "The cat sat," when generating "chat" (French for cat), you look at "cat" in the source. When generating "assis" (sat), you look at "sat." You consult the source every time you need it.

Now imagine a more sophisticated version: you don't just look at one word. You look at all words, but assign different importance scores. For "Le chat était assis," the word "chat" gets high attention weight from the encoder's "cat," some weight from "sat" (for context), and little from "The." This weighted combination becomes your context for generating each French word. That is attention.

---

### **4. ASCII diagram**

```
Attention mechanism in Seq2Seq:

    Encoder hidden states:    h₁    h₂    h₃    h₄    h₅
    (from input sentence)      │     │     │     │     │
                              │     │     │     │     │
    Decoder state s_t ────────┼─────┼─────┼─────┼─────┘
                              │     │     │     │
    Compute attention scores: e₁   e₂    e₃    e₄    e₅
                              │     │     │     │     │
    Softmax:                  α₁   α₂    α₃    α₄    α₅
                              │     │     │     │     │
                              └─────┴─────┴─────┴─────┘
                                        │
                                        ↓
                           context = Σ α_i × h_i
                                        │
                                        ↓
                        ┌────────────────┴───────────────┐
                        │  Combine with s_t → output y_t │
                        └────────────────────────────────┘


Detailed scoring and weighting:

    s_t (decoder state)
           │
           ├─── dot product ───→ e₁ = score(s_t, h₁)
           ├─── dot product ───→ e₂ = score(s_t, h₂)
           ├─── ...            → e₃, e₄, e₅
           │
           ↓
    α = softmax([e₁, e₂, e₃, e₄, e₅])
           │
           ↓
    context = α₁×h₁ + α₂×h₂ + α₃×h₃ + α₄×h₄ + α₅×h₅
```

---

### **5. Mathematical formulation**

**Given:** Encoder hidden states h₁,...,h_T (from input sequence), decoder hidden state s_t (current step)

**Attention scores (alignment):** How well each input position matches current decoder state.

Additive attention (Bahdanau, 2014):

$$
e_{ti} = \mathbf{v}^T \tanh(\mathbf{W}_a \mathbf{s}_t + \mathbf{U}_a \mathbf{h}_i)
$$

Dot-product attention (Luong, 2015):

$$
e_{ti} = \mathbf{s}_t^T \mathbf{h}_i
$$

Scaled dot-product (Vaswani, 2017):

$$
e_{ti} = \frac{\mathbf{s}_t^T \mathbf{h}_i}{\sqrt{d}}
$$

**Attention weights:** Normalize scores to probabilities.

$$
\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^T \exp(e_{tj})}
$$

(Unicode: α_ti = exp(e_ti) / Σ_j exp(e_tj))

**Context vector:** Weighted sum of encoder states.

$$
\mathbf{c}_t = \sum_{i=1}^T \alpha_{ti} \mathbf{h}_i
$$

(Unicode: c_t = Σ_i α_ti × h_i)

**Combined output (for decoder next step):**

$$
\tilde{\mathbf{s}}_t = \tanh(\mathbf{W}_c [\mathbf{c}_t; \mathbf{s}_t])
$$

(Unicode: s̃_t = tanh(W_c × [c_t; s_t]) concatenation)

---

### **6. Worked example (step-by-step)**

#### **Step 1: Encoder hidden states**

English input "cat sat" → 2 words. Encoder produces:
h₁ = [0.8, 0.3] (for "cat")
h₂ = [0.2, 0.9] (for "sat")

#### **Step 2: Decoder state at time t (generating first French word)**

s_t = [0.6, 0.4] (current decoder state before generating "chat")

#### **Step 3: Compute attention scores (dot product)**

e₁ = s_t · h₁ = 0.6×0.8 + 0.4×0.3 = 0.48 + 0.12 = 0.60
e₂ = s_t · h₂ = 0.6×0.2 + 0.4×0.9 = 0.12 + 0.36 = 0.48

#### **Step 4: Softmax to get weights**

exp(0.60) = 1.822, exp(0.48) = 1.616, sum = 3.438

α₁ = 1.822 / 3.438 = 0.53
α₂ = 1.616 / 3.438 = 0.47

#### **Step 5: Compute context vector**

c_t = 0.53 × [0.8, 0.3] + 0.47 × [0.2, 0.9] = [0.424, 0.159] + [0.094, 0.423] = [0.518, 0.582]

#### **Step 6: Interpret**

"cat" got slightly higher weight (0.53 vs 0.47) for generating "chat." Both words contributed because context matters. The context vector is a blend of both encoder states, weighted by relevance.

#### **Step 7: Compare to no attention (Seq2Seq baseline)**

Without attention, c_t would be just h₂ (final encoder state) = [0.2, 0.9]. "cat" would contribute nothing to generating "chat," leading to worse translation. Attention preserves information from early words.

---

### **7. How this appears inside neural networks and LLMs**

- **Breaking the bottleneck:** Attention solved the fixed-context-vector problem. Decoding can now access any part of the input directly.

- **Alignment visualization:** Attention weights α_ti can be visualized as a heatmap. For translation, you see diagonal alignment: "cat" → "chat," "sat" → "assis." This interpretability was revolutionary.

- **Long sentences:** Attention allowed translation of 50+ word sentences without degradation. RNNs alone failed beyond 20 words.

- **Direct path for gradients:** In backprop, gradients from decoder can flow directly to any encoder state, not just through the final state. This mitigated vanishing gradients.

- **Precursor to self-attention:** The original attention had queries from decoder, keys/values from encoder. Self-attention (next section) uses queries, keys, values all from the same sequence.

- **Attention is everywhere:** Modern LLMs use attention (self-attention, cross-attention) in every block. The original mechanism is cross-attention between encoder and decoder.

---

### **8. Brain-like connection (visual attention)**

When you look at a scene, your eyes do not process every pixel equally. You fixate on relevant regions: when searching for a red cup, your attention is drawn to red objects. This is visual attention—a weighted combination of visual features based on current goals. The brain's attention mechanism computes "relevance scores" between internal goals (like decoder state) and sensory input (like encoder states). Neurons in the parietal cortex encode these attention weights. The original attention mechanism was directly inspired by cognitive neuroscience models of visual attention. Transformers later generalized this to self-attention (attending to different parts of the same sequence).

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "The original attention mechanism is obsolete. Transformers use self-attention, which is completely different."

_Why it is wrong:_ The original attention (Bahdanau, 2014) is cross-attention: queries from decoder, keys/values from encoder. Transformers use both self-attention (queries, keys, values all from same sequence) AND cross-attention (in encoder-decoder models like T5). The mathematical core—softmax of scaled dot-products, weighted sum of values—is identical. The original attention paper introduced the formula that powers every modern LLM. It is not obsolete; it is the foundation.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Before attention, every word in a sentence had to be     |
|  squeezed through a bottleneck—the final encoder state.   |
|  Information leaked away. Attention removed the           |
|  bottleneck entirely. Every encoder state connects        |
|  directly to every decoder state. This simple insight—    |
|  that the model should look back at the input whenever    |
|  needed—transformed machine translation and led directly  |
|  to the transformer. Attention is not just a mechanism.   |
|  It is the reason LLMs can handle long contexts.          |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have encoder states for a 10-word sentence. The decoder is generating the 5th output word. The attention weights α₅,ᵢ are all approximately 0.1 (uniform) for i=1..10.

**Question:** What does this tell you about the decoder's current state relative to the input? Is this likely to produce a good translation? Why or why not?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Uniform attention weights mean the decoder state s_t has roughly equal dot-product similarity to all encoder states. This suggests the decoder is not focusing on any particular input word—it is "undecided" or the input words are all equally relevant to the current output.

This is usually bad for translation. Most output words align strongly with 1-2 input words (diagonal alignment). Uniform attention means the context vector is an average of all encoder states, which dilutes relevant information. The decoder may produce a vague or incorrect word. For example, generating "the" from "the cat sat" might have diffuse attention, but generating "cat" should focus sharply on "cat." Uniform attention at that step would be problematic.
