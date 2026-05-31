# RNNs (Recurrent Neural Networks)

## **DOMAIN: ADVANCED ARCHITECTURES | Sub domain: Sequence Models (Pre-Transformer)**

---

### **1. Why this concept matters for building intelligent systems**

Language is a sequence. Audio is a sequence. Video is a sequence of frames. A standard neural network processes each input independently—it has no memory of what came before. This is fine for recognizing a cat in a single photo, but useless for understanding a sentence where "bank" means something different after "river" than after "money." Recurrent Neural Networks (RNNs) solved this by introducing memory: a hidden state that persists from step to step, carrying information forward through time. Without RNNs, there would be no language models. And without understanding their limitations, you cannot appreciate why Transformers revolutionized the field.

---

### **2. Core idea**

**An RNN processes sequential data by maintaining a hidden state that is updated at each time step based on both the current input and the previous hidden state, allowing information to persist across the sequence.**

---

### **3. Concrete analogy**

Imagine you are reading a mystery novel. You don't process each sentence in isolation—you carry forward your understanding from previous pages. "The butler entered the room" means something completely different if you remember that "the murder weapon was discovered missing three chapters ago." Your brain maintains a hidden state (your evolving understanding of the plot) that updates with each new sentence. This hidden state influences how you interpret future sentences.

Now imagine you have a terrible memory. After each sentence, you forget everything except the most recent sentence. You could still read the book, but you would miss all the connections—the setup in Chapter 1 would have no impact on the payoff in Chapter 10. That is a standard neural network. An RNN is the version with a functioning memory: each sentence updates your understanding, and that understanding shapes how you read everything that follows.

---

### **4. ASCII diagram**

```text
RNN unrolled through time (processing a 3-word sentence):

Time step t=1        t=2        t=3
Input: "The"         "cat"      "sat"
          ↓             ↓          ↓
        ┌───┐         ┌───┐      ┌───┐
   h₀ →│RNN│→ h₁   h₁→│RNN│→ h₂  h₂→│RNN│→ h₃
        └───┘         └───┘      └───┘
          ↓             ↓          ↓
       Output?       Output?     Output?
      (optional)    (optional)   (optional)

Where:
h₀ = initial hidden state (usually zeros)
h₁ = f("The", h₀)   (understanding after reading "The")
h₂ = f("cat", h₁)   (understanding after reading "The cat")
h₃ = f("sat", h₂)   (understanding after reading "The cat sat")

Each step: new_hidden = tanh( W_input × input + W_hidden × old_hidden + bias )
```

---

### **5. Mathematical formulation**

**RNN cell at time step t:**

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)
$$

(Unicode: h*t = tanh(W_xh × x_t + W_hh × h*{t-1} + b_h))

Where:

- x_t = input vector at time t (e.g., word embedding)
- h\_{t-1} = hidden state from previous time step
- h_t = new hidden state (the "memory" after seeing x_t)
- W_xh = weight matrix mapping input to hidden
- W_hh = weight matrix mapping previous hidden to new hidden (the recurrence!)
- b_h = bias vector
- tanh = hyperbolic tangent activation (outputs between -1 and 1)

**Output at each time step (optional):**

$$
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

**For a sequence of length T, the RNN processes:**

$$
\mathbf{h}_1 = f(\mathbf{x}_1, \mathbf{h}_0)
$$

$$
\mathbf{h}_2 = f(\mathbf{x}_2, \mathbf{h}_1)
$$

$$
\vdots
$$

$$
\mathbf{h}_T = f(\mathbf{x}_T, \mathbf{h}_{T-1})
$$

The final hidden state h_T theoretically contains information from the entire sequence.

**Vanishing gradient problem (backpropagation through time):**

$$
\frac{\partial L}{\partial \mathbf{h}_{t-1}} = \frac{\partial L}{\partial \mathbf{h}_t} \cdot \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \frac{\partial L}{\partial \mathbf{h}_t} \cdot \mathbf{W}_{hh}^T \cdot \text{diag}(\tanh'(\cdot))
$$

If ‖W_hh‖ < 1, gradients vanish exponentially with sequence length. If ‖W_hh‖ > 1, gradients explode.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define a tiny RNN**

Input size = 2, hidden size = 2 (simplest possible)

Weights:

$$
\mathbf{W}_{xh} = \begin{bmatrix} 0.5 & -0.2 \\ 0.3 & 0.8 \end{bmatrix}, \quad
\mathbf{W}_{hh} = \begin{bmatrix} 0.6 & 0.1 \\ -0.2 & 0.4 \end{bmatrix}, \quad
\mathbf{b}_h = \begin{bmatrix} 0.1 \\ 0.05 \end{bmatrix}
$$

#### **Step 2: Initial state**

h₀ = [0, 0]ᵀ

#### **Step 3: First input (t=1)**

x₁ = [1.0, 0.5]ᵀ

Compute: W_xh × x₁ = [0.5×1.0 + (-0.2)×0.5, 0.3×1.0 + 0.8×0.5]ᵀ = [0.5 - 0.1, 0.3 + 0.4]ᵀ = [0.4, 0.7]ᵀ

W_hh × h₀ = [0, 0]ᵀ

Add bias: [0.4 + 0.1, 0.7 + 0.05]ᵀ = [0.5, 0.75]ᵀ

Apply tanh: h₁ = [tanh(0.5), tanh(0.75)]ᵀ ≈ [0.462, 0.635]ᵀ

#### **Step 4: Second input (t=2)**

x₂ = [0.2, 0.8]ᵀ

W_xh × x₂ = [0.5×0.2 + (-0.2)×0.8, 0.3×0.2 + 0.8×0.8]ᵀ = [0.1 - 0.16, 0.06 + 0.64]ᵀ = [-0.06, 0.70]ᵀ

W_hh × h₁ = [0.6×0.462 + 0.1×0.635, (-0.2)×0.462 + 0.4×0.635]ᵀ = [0.277 + 0.064, -0.092 + 0.254]ᵀ = [0.341, 0.162]ᵀ

Sum: [-0.06 + 0.341 + 0.1, 0.70 + 0.162 + 0.05]ᵀ = [0.381, 0.912]ᵀ

Apply tanh: h₂ ≈ [0.364, 0.722]ᵀ

#### **Step 5: Interpret**

The hidden state evolved from [0,0] → [0.462, 0.635] → [0.364, 0.722]. Each step combined new input with the compressed memory of all previous inputs.

---

### **7. How this appears inside neural networks and LLMs**

- **Pre-transformer era:** RNNs (and their variants LSTM, GRU) were the standard for language modeling, machine translation, and speech recognition.

- **Character-level language models:** RNNs can predict the next character given previous characters, generating text one character at a time.

- **Encoder-decoder architecture (Seq2Seq):** One RNN encodes the input sentence into a final hidden state (context vector); another RNN decodes that vector into the output sentence. This was state-of-the-art for translation before Transformers.

- **Vanishing gradients killed long-range memory:** In practice, RNNs forget after about 10-20 steps. "The cat that lived in the house that Jack built sat" — the RNN loses the connection between "cat" and "sat." LSTMs (gated RNNs) extended this to ~100 steps, but still far from perfect.

- **Why Transformers won:** RNNs process sequentially (step 2 cannot start until step 1 finishes), so they cannot parallelize on GPUs. Transformers process all steps simultaneously. Also, Transformers have direct connections between any two positions (O(1) path length), while RNNs have path length O(sequence length).

- **Current usage:** RNNs are still used in some edge devices (small footprint), time series forecasting, and audio processing where sequences are short and memory is tight.

---

### **8. Brain-like connection (working memory)**

The brain's working memory maintains information over short time scales—like remembering the beginning of a sentence while processing the end. RNNs were directly inspired by this: neural activity persists through recurrent connections, forming a dynamical system. However, the brain's memory is far more sophisticated: it can selectively maintain some information while discarding others (gating, like LSTMs), and it can recall arbitrary past events (attention, like Transformers). RNNs captured the persistence aspect but missed the direct access that makes human memory powerful. The shift from RNNs to Transformers mirrors the shift from a fading trace to a searchable index.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "RNNs process sequences in parallel, so they are faster than Transformers for long sequences."

_Why it is wrong:_ RNNs are strictly sequential—you cannot compute h₂ until you have computed h₁. This makes them slow on modern parallel hardware (GPUs). Transformers process all positions in parallel using self-attention, which is O(1) sequential depth (just a few operations) but O(n²) in memory. For long sequences, RNNs are memory-efficient but time-inefficient; Transformers are time-efficient but memory-hungry. This is why recent research (Mamba, RWKV) is revisiting recurrent architectures with better parallelism.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  RNNs were the first neural networks that truly           |
|  understood time. They taught us that memory matters—that |
|  language cannot be processed word by word in isolation.  |
|  Their limitations (sequential processing, vanishing      |
|  gradients) directly motivated the invention of attention |
|  and Transformers. You cannot understand why Transformers |
|  are revolutionary without understanding what they        |
|  replaced. RNNs are the foundation. The flaws are the     |
|  lessons.                                                  |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have an RNN processing the sentence "The weather today is sunny." The hidden state after processing "sunny" should reflect the meaning of the entire sentence.

**Question:** If the RNN suffers from vanishing gradients, what happens to the influence of the word "The" on the final hidden state compared to the word "sunny"? Why?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** The influence of "The" becomes exponentially smaller than the influence of "sunny." During backpropagation, the gradient signal from the final loss must multiply through many time steps. Each multiplication by W_hh and the tanh derivative reduces the signal. After 5 steps (from "The" to "sunny"), the gradient for earlier words becomes tiny. The RNN effectively forgets early words, making the final hidden state dominated by recent inputs. This is why RNNs struggle with long-range dependencies.
