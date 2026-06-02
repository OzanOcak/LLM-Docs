# Fine-tuning

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Training Paradigms**

---

### **1. Why this concept matters**

A pre-trained LLM knows language but does not follow instructions. It will continue a sentence ("The capital of France is...") but will not answer questions ("What is the capital of France?") in a helpful way. Fine-tuning bridges this gap. It takes the base model and trains it further on a small dataset of labeled examples—question-answer pairs, instructions, or chat conversations. This is the process that turns GPT-3 into ChatGPT. Fine-tuning is fast, cheap, and effective. With just thousands of examples, you can adapt a billion-parameter model to answer questions, translate languages, or follow complex instructions.

---

### **2. Core idea**

**Fine-tuning continues training of a pre-trained model on a smaller, task-specific dataset with a supervised objective, adapting the model's behavior to follow instructions, answer questions, or perform specialized tasks while preserving the general knowledge learned during pre-training.**

---

### **3. Concrete analogy**

A medical student graduates from medical school (pre-training). They know anatomy, physiology, pharmacology—general medical knowledge. But they cannot perform surgery yet. They need a surgical residency (fine-tuning): supervised practice on specific procedures, with an expert surgeon providing feedback. After a few years, they are a surgeon.

Now imagine instead of years, fine-tuning takes hours and uses only a few hundred examples. That is the power of LLM fine-tuning. The model already knows language, reasoning, and facts. Fine-tuning just reshapes its behavior—how to respond, what format to use, which tone to adopt. It is not teaching new facts (though it can learn new ones). It is teaching the model how to interact.

---

### **4. ASCII diagram**

```
Pre-training vs Fine-tuning:

    Pre-training                        Fine-tuning
    (weeks, billions of tokens)         (hours, thousands of examples)

    Raw text → Base Model               Base Model + labeled data → Task Model

    "The capital of" → "France"         "Q: capital of France?" → "A: Paris"
    "Translate to French:" → "Bonjour"  "Translate 'Hello' to French" → "Bonjour"

    General language understanding      Task-specific behavior


Fine-tuning data examples (instruction format):

    Input: "What is the capital of France?"
    Output: "The capital of France is Paris."

    Input: "Explain quantum computing to a 5-year-old."
    Output: "Quantum computing is like having a magic coin that can be heads and tails at the same time..."

    Input: "Translate 'Hello' to Spanish."
    Output: "Hola"


Fine-tuning process:

    Pre-trained Model (frozen or not)
              │
              │ Training on labeled data
              ▼
    ┌─────────────────────────────────┐
    │ Input: "Q: What is 2+2?"        │
    │ Target: "A: 2+2 equals 4."      │
    │ Loss = CrossEntropy(pred, target)│
    └─────────────────────────────────┘
              │
              │ Backpropagate (full or LoRA)
              ▼
    Fine-tuned Model (updated weights)


Transfer learning spectrum:

    Pre-trained ──┬── Feature Extraction (frozen backbone)
                  ├── Full Fine-tuning (all weights updated)
                  ├── LoRA (low-rank updates)
                  └── Adapters (small trainable modules)

    (More data / task similarity → more weights unfrozen)
```

---

### **5. Mathematical formulation**

**Full fine-tuning:**

Initialize θ = θ_pretrained. Minimize supervised loss:

$$
L_{\text{FT}}(\theta) = -\sum_{(x,y) \in D_{\text{task}}} \log P_\theta(y | x)
$$

Where D_task is the fine-tuning dataset (e.g., instruction-response pairs).

**Low-Rank Adaptation (LoRA) — most common efficient fine-tuning:**

For a weight matrix W ∈ ℝ^{d×k}, LoRA adds a low-rank update:

$$
W' = W + \Delta W = W + BA
$$

Where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, with r ≪ min(d,k) (e.g., r=8, d=4096, k=4096).

During fine-tuning, only A and B are updated. Original W is frozen.

Parameter reduction: from d×k to r×(d+k).

**Learning rate for fine-tuning:**

Typical learning rate is 10-100× smaller than pre-training:

$$
\eta_{\text{FT}} \approx 10^{-5} \text{ to } 10^{-4}
$$

(Pre-training η ~ 10^{-4} to 10^{-3})

**Epochs for fine-tuning:**

Small dataset (500-10,000 examples) → 2-5 epochs. More epochs cause overfitting.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Pre-trained base model**

Model: Llama 2 7B, pre-trained on 2 trillion tokens. It knows language and facts.

#### **Step 2: Fine-tuning dataset (medical QA)**

500 examples of medical questions with answers:

| Input (Question)        | Output (Answer)                                                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| "What is hypertension?" | "Hypertension, or high blood pressure, is when blood pressure is consistently above 130/80 mmHg."                             |
| "What causes a stroke?" | "A stroke occurs when blood flow to part of the brain is interrupted, either by a clot (ischemic) or bleeding (hemorrhagic)." |

#### **Step 3: Fine-tuning with LoRA (r=16)**

Original attention weight W_q in layer 5: shape 4096×4096 → 16.7M parameters.

LoRA adds B (4096×16) and A (16×4096) → 2×4096×16 = 131,072 trainable parameters.

Total trainable parameters per layer: ~0.5M. For 32 layers: ~16M trainable (vs 7B total).

#### **Step 4: Training**

Loss = cross-entropy between model output and target answer.

Learning rate = 2e-5 (small, to avoid catastrophic forgetting).

Batch size = 8.

Epochs = 3.

#### **Step 5: After fine-tuning**

Model now responds to medical questions accurately. It has learned the format: question → answer, not just next-token prediction. It still knows general facts from pre-training.

#### **Step 6: Compare to no fine-tuning (base model)**

Base model, given "What is hypertension?", might generate:

- "Hypertension is a condition. Hypertension is a condition where..." (repetition)
- "I'm sorry, I don't know that." (refusal)
- Continue the sentence in a different direction.

Fine-tuning teaches it to answer directly and accurately.

---

### **7. How this appears inside neural networks and LLMs**

- **Supervised fine-tuning (SFT):** The standard method. Train on (prompt, response) pairs. Turns base model into instruction follower.

- **Catastrophic forgetting:** Fine-tuning on a small dataset can cause the model to forget pre-trained knowledge. Mitigations: low learning rates, LoRA (updates few parameters), replay (mix original data).

- **LoRA (Low-Rank Adaptation):** Most common efficient fine-tuning method. Adds trainable low-rank matrices. Reduces memory from 7B to 16M trainable parameters. Enables fine-tuning on a single GPU.

- **QLoRA (Quantized LoRA):** 4-bit quantized base model + LoRA. Fine-tune 65B model on a single 24GB GPU.

- **Instruction tuning:** Fine-tuning on thousands of (instruction, response) pairs across diverse tasks. Produces models that generalize to new instructions (e.g., FLAN-T5, Alpaca).

- **Fine-tuning for domain adaptation:** Legal, medical, financial domains. Pre-trained model knows general English; fine-tuning on domain documents teaches terminology.

- **Fine-tuning for formatting:** Teaching model to output JSON, XML, or follow specific templates.

---

### **8. Brain-like connection (skill acquisition)**

The brain pre-trains on massive sensory input during infancy (learning language, object permanence, causal reasoning). Then, fine-tuning happens throughout life: learning to drive, learning a new software tool, adapting to a new job. Each new skill uses the pre-trained foundation and updates only relevant neural pathways. This is efficient learning—you do not relearn language every time you learn to cook. LoRA's low-rank updates mirror brain plasticity: only a subset of connections (synapses) are modified during new learning, preserving previously acquired knowledge. Patients with brain damage can lose fine-tuned skills (e.g., playing piano) while retaining general language—dissociations that resemble freezing parts of a model while fine-tuning others.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Fine-tuning is just training from scratch on a small dataset. I can skip pre-training and directly fine-tune."

_Why it is wrong:_ Fine-tuning without pre-training does not work. A randomly initialized model cannot learn from 500 examples—it would overfit or fail to generalize. The pre-trained model already understands grammar, syntax, and world facts. Fine-tuning adjusts the model's behavior; it does not teach language from zero. Training from scratch on 500 examples would produce a model that memorizes the 500 examples but cannot generalize to new inputs. The pre-trained foundation is essential. Fine-tuning is adaptation, not acquisition.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Pre-training builds the engine. Fine-tuning steers it.   |
|  With just hundreds of labeled examples, you can turn    |
|  a general-purpose LLM into a medical advisor, a code    |
|  generator, a customer service bot, or a translation     |
|  engine. Fine-tuning is accessible—it requires hours,     |
|  not weeks, and can run on a single GPU. It is the        |
|  interface between massive pre-training and practical    |
|  applications. Without fine-tuning, LLMs are impressive   |
|  but useless. With it, they become tools.                 |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a pre-trained 7B parameter model. You want to fine-tune it on a legal document summarization dataset of 500 examples. You have one GPU with 24GB memory. Full fine-tuning (updating all 7B parameters) requires too much memory.

**Question:** What technique can you use to fine-tune efficiently? Approximately how many trainable parameters would it require if you use rank r=8 on 32 layers each with 4096×4096 attention weights?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Use LoRA (Low-Rank Adaptation). For each 4096×4096 weight matrix (W_q, W_k, W_v, W_o in each attention layer, plus FFN weights), add B (4096×8) and A (8×4096). Trainable parameters per matrix = 2 × 4096 × 8 = 65,536.

Assuming 32 layers, with 4 attention matrices per layer = 128 matrices. 128 × 65,536 = 8,388,608 (~8.4M trainable parameters). This is 0.12% of 7B parameters. Plus LoRA on FFN weights (2 per layer) would add more, but still tiny fraction of full model.

LoRA plus 4-bit quantization (QLoRA) would reduce memory further, allowing fine-tuning on 24GB GPU. This is why LoRA is standard for efficient fine-tuning.
