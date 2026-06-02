# Pre-training

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Training Paradigms**

---

### **1. Why this concept matters**

You cannot learn a language from a few examples. You need millions of books, billions of web pages, trillions of tokens. Pre-training is the phase where the model consumes this massive corpus, learning grammar, facts, reasoning patterns, and world knowledge—all from the simple objective of predicting the next word (or masked word). This phase costs millions of dollars and weeks of compute. But it produces a base model that already understands language. Pre-training is why GPT-3 knows the capital of France without ever being explicitly taught. It is the foundation upon which all other capabilities are built.

---

### **2. Core idea**

**Pre-training is the initial phase of LLM training where the model learns general language understanding from massive, diverse, unlabeled text data using self-supervised objectives such as next-token prediction or masked language modeling.**

---

### **3. Concrete analogy**

Imagine you want to train a world-class chef. You do not start with teaching specific dishes. Instead, you send them to culinary school for two years. They learn knife skills, flavor combinations, cooking techniques, ingredient properties. They cook thousands of dishes, not for customers, but for practice. They taste everything, learning what works and what does not.

After culinary school, they can cook anything. You give them a new recipe (fine-tuning) and they excel. But without that foundational training, they would struggle.

Pre-training is culinary school for LLMs. The model reads billions of sentences, learning to predict the next word. It learns grammar, vocabulary, common sense, reasoning. This phase requires no human labels—just raw text. After pre-training, the model is a "foundation model": it understands language but does not yet follow instructions or answer questions. Those come later (fine-tuning, instruction tuning). But without pre-training, there is no foundation to build on.

---

### **4. ASCII diagram**

```
Two-stage LLM training pipeline:

    Stage 1: Pre-training (costly, once)

    Massive unlabeled text (trillions of tokens)
    ┌────────────────────────────────────────┐
    │ Wikipedia, books, web pages, code,     │
    │ scientific papers, social media, ...   │
    └────────────────────────────────────────┘
                      │
                      ▼
    Self-supervised objective:
    • Causal LM (GPT): predict next token
    • MLM (BERT): predict masked tokens
                      │
                      ▼
    Base model (understands language,
    but does not follow instructions)


    Stage 2: Fine-tuning / Alignment (cheaper, task-specific)

    Labeled / curated data
    ┌────────────────────────────────────────┐
    │ Question-answer pairs, instructions,   │
    │ human preferences, chat conversations  │
    └────────────────────────────────────────┘
                      │
                      ▼
    Supervised / RL objective
                      │
                      ▼
    Deployable model (can answer questions,
    follow instructions, chat)


Pre-training objectives comparison:

    GPT (Causal Language Modeling):
    Input:  "The cat sat on the"
    Target: "mat"

    Masked LM (BERT):
    Input:  "The [MASK] sat on the mat"
    Target: "cat"

    Prefix LM (UniLM, some T5 variants):
    Input (prefix): "Translate English to French:"
    Output (generation): "The cat sat on the mat"
```

---

### **5. Mathematical formulation**

**Causal Language Modeling (GPT, Llama):**

Given sequence x₁, x₂, ..., x_T, maximize log-likelihood of next token:

$$
L_{\text{CLM}} = \sum_{t=1}^{T-1} \log P(x_{t+1} | x_1, ..., x_t)
$$

(Unicode: L = Σ log P(x\_{t+1} | x_1...x_t))

**Masked Language Modeling (BERT):**

Sample 15% of positions to mask. Replace with [MASK] (80%), random token (10%), unchanged (10%). Predict original token:

$$
L_{\text{MLM}} = \sum_{i \in M} \log P(x_i | x_{\setminus M})
$$

Where M is set of masked positions.

**Pre-training dataset scale:**

| Model             | Tokens | Years (approx) |
| ----------------- | ------ | -------------- |
| GPT-1             | 0.5B   | 2018           |
| BERT              | 3.3B   | 2018           |
| GPT-3             | 300B   | 2020           |
| Llama 1           | 1.4T   | 2023           |
| Llama 3           | 15T    | 2024           |
| GPT-4 (estimated) | 13T+   | 2023           |

**Scaling laws:**

Test loss L scales as power law with compute C, dataset size D, and parameters N:

$$
L(N, D) \propto N^{-0.05} + D^{-0.095}
$$

Optimal training uses compute-optimal ratio (Chinchilla: 20 tokens per parameter). GPT-4 estimated ~25-30 tokens per parameter.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Training example (Causal LM)**

Input sequence: "The capital of France is Paris"

At position 1: input "The" → predict "capital"
At position 2: input "The capital" → predict "of"
At position 3: input "The capital of" → predict "France"
At position 4: input "The capital of France" → predict "is"
At position 5: input "The capital of France is" → predict "Paris"

Loss = sum of cross-entropy over all predictions.

#### **Step 2: What the model learns**

From this single sentence, the model learns:

- Word order (capital follows The)
- Relationship (France and Paris are linked)
- Grammar (is follows France)
- Factual knowledge (Paris is capital of France)

From billions of sentences, these patterns aggregate into robust knowledge.

#### **Step 3: Data source composition (Llama 3 example)**

Llama 3 pre-training data (15T tokens):

- 50% web pages (Common Crawl)
- 25% books and academic papers
- 15% code (GitHub)
- 10% multilingual data (non-English)

The model learns to code from the code portion, to reason from scientific papers, and to converse from web text.

#### **Step 4: Compute cost estimate (GPT-3 scale)**

175B parameters, 300B tokens.

Floating point operations (FLOPs) ≈ 6 × N × D = 6 × 175e9 × 300e9 ≈ 3.15e23 FLOPs.

On 10,000 A100 GPUs (312 TFLOPS each), theoretical time = 3.15e23 / (312e12 × 1e4) ≈ 1e7 seconds ≈ 115 days.

In practice, with overhead, ~30-60 days.

---

### **7. How this appears inside neural networks and LLMs**

- **Base model availability:** Pre-trained models are released as "base" or "foundation" models (e.g., Llama-2-7b, GPT-3-175B). Users then fine-tune on specific tasks.

- **Pre-training data quality matters more than quantity:** Llama 3 used 15T tokens but heavily filtered for quality. Clean data beats more data.

- **De-duplication:** Removing duplicate text prevents memorization and improves generalization.

- **Causal mask for autoregressive models:** During pre-training, GPT uses causal mask (future positions not visible). BERT uses no mask (bidirectional).

- **Mixed precision training:** Pre-training uses FP16 or BF16 to reduce memory. Loss scaling prevents underflow.

- **Gradient accumulation:** Simulates larger batch sizes than GPU memory allows.

- **Checkpointing:** Saves model state every N steps. If training crashes, resume from last checkpoint.

- **Warmup and learning rate schedule:** Linear warmup from 0 to peak (e.g., 2000 steps), then cosine decay.

---

### **8. Brain-like connection (infant language acquisition)**

Human infants pre-train on language before they can speak. For the first 12-18 months, they listen to speech (input) without producing output. They learn phonemes, word boundaries, and simple grammar through statistical learning—detecting which sounds co-occur frequently (like subword tokenization), which sequences predict others (like next-word prediction). This is pre-training. Then, around 12-18 months, they begin to produce words (fine-tuning). The brain's language acquisition follows the same two-stage pattern: massive passive exposure, then active production. Damage to language areas before age 5 can cause permanent deficits—the pre-training window is critical. After that, the brain can still learn new languages but never as natively. LLMs show similar critical periods: pre-training data composition determines final capabilities.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Pre-training is just unsupervised learning. It is cheap because it does not require labels."

_Why it is wrong:_ Pre-training is "self-supervised" (labels come from the data itself, e.g., the next word is the label), but it is not cheap. Pre-training requires massive compute (millions of dollars), terabytes of data, and engineering expertise. It is the most expensive phase of LLM development. The "self-supervised" label is misleading: predicting the next token is a task with a ground-truth label (the actual next token). The model gets immediate feedback on every single token. This is not unsupervised—it is supervised with automatically generated labels. The cost is not in labeling but in computation.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Pre-training is where LLMs learn almost everything:      |
|  grammar, world facts, reasoning, code, multilingual      |
|  understanding. The months of compute, billions of        |
|  parameters, trillions of tokens—all for the simple       |
|  task of predicting the next word. This deceptively       |
|  simple objective forces the model to learn the hidden    |
|  structure of language. Pre-training is the slow,         |
|  expensive miracle that makes LLMs possible. Without it,  |
|  your model cannot even form a grammatical sentence.      |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have 10 billion tokens of high-quality text and 1 billion parameters. According to the Chinchilla scaling laws, should you pre-train a larger model (more parameters) or collect more data? Why?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** According to Chinchilla scaling laws (Hoffmann et al., 2022), the optimal training uses 20 tokens per parameter. For 1 billion parameters, you need 20 billion tokens. You have only 10 billion tokens (half the optimal). You should collect more data (another 10 billion tokens) before increasing model size. If you increase model size without adding data, you will under-train the model—it will memorize but not generalize. The optimal compute point balances parameters and data. With 10B tokens, the optimal model size is 500M parameters (20 tokens/parameter). Your 1B model is over-parameterized for your data. Train a smaller model or collect more data. This is why modern LLMs (Llama 3) are trained on 15T tokens with large models—they balance data and parameters for optimal compute.
