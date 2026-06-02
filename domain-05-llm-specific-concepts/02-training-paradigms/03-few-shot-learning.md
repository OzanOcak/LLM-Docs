# Few-shot learning

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Training Paradigms**

---

### **1. Why this concept matters**

Fine-tuning requires thousands of labeled examples. But what if you only have five? Or you cannot afford to update the model's weights at all? Few-shot learning solves this by teaching the model to learn from examples presented in the prompt itself—no gradient updates, no fine-tuning. The model sees a few input-output pairs (shots) and then answers a new query. This is how GPT-3 demonstrated its remarkable abilities: with just 1-5 examples, it could translate, summarize, or solve math problems. Few-shot learning is the ultimate test of a model's ability to learn from context, and it is the default way most people interact with LLMs today via in-context learning.

---

### **2. Core idea**

**Few-shot learning enables a model to perform a new task by providing a small number of input-output examples (shots) in the prompt, without any parameter updates, relying entirely on the model's ability to learn from context at inference time.**

---

### **3. Concrete analogy**

Imagine you are visiting a foreign country. You do not speak the language, and you have no phrasebook. A local gives you a few examples:

- "Apple → Ringo"
- "Banana → Budo"
- "Cat → Neko"

Then they ask: "What is 'dog'?" You infer the pattern (the word maps to its Japanese equivalent) and answer "Inu." You learned a new task from just three examples, without studying a dictionary. That is few-shot learning.

Now imagine you have no examples—just the question. That is zero-shot. You have a dictionary (thousands of examples) and study it—that is fine-tuning. Few-shot is the middle ground: learn the pattern from a handful of examples presented at test time.

---

### **4. ASCII diagram**

```
Zero-shot vs One-shot vs Few-shot:

Zero-shot (no examples):
    Prompt: "Translate 'Hello' to Spanish:"
    Output: "Hola"
    (Model relies entirely on pre-training)


One-shot (1 example):
    Prompt: "Translate 'Hello' to French: Bonjour
              Translate 'Goodbye' to Spanish:"
    Output: "Adiós"


Few-shot (3 examples):
    Prompt: "English: Apple → Spanish: Manzana
             English: Banana → Spanish: Plátano
             English: Cherry → Spanish: Cereza
             English: Grape → Spanish:"
    Output: "Uva"


In-context learning (no weight updates):

    Examples in prompt (demonstrations)
              │
              ▼
    ┌─────────────────────────────────────┐
    │ "Q: What is 2+2? A: 4              │
    │  Q: What is 5+3? A: 8              │
    │  Q: What is 10-4? A: 6             │
    │  Q: What is 7+2?"                  │
    └─────────────────────────────────────┘
              │
              │ Forward pass only (no backprop)
              ▼
    Model outputs: "A: 9"  (learned the addition task)


Scaling behavior:

    Performance │
                │         ┌─────────────
                │       ┌──┘
                │    ┌──┘
                │ ┌──┘
                │●┘   (Few-shot > One-shot > Zero-shot)
                └──────────────────→ Number of shots
                 0   1   2   3   4   5
```

---

### **5. Mathematical formulation**

**Few-shot inference (no training):**

Given:

- Task description T (optional)
- K examples (x₁, y₁), ..., (x_K, y_K)
- Query input x_new

Construct prompt P:

$$
P = [T, (x_1, y_1), ..., (x_K, y_K), x_{\text{new}}]
$$

Model generates output:

$$
\hat{y} = \arg\max_y P_{\theta}(y | P)
$$

Where θ is fixed (no weight updates).

**The key property: In-context learning**

The model learns from examples purely through the forward pass. The examples prime the model's hidden states, biasing it toward the task distribution. This works because the model was pre-trained on data containing similar patterns (e.g., "Q: X A: Y" formatting).

**Effectiveness depends on:**

- Number of shots (K): More examples → better performance, diminishing returns.
- Order of examples: Random order sometimes hurts performance.
- Format consistency: All examples should follow the same pattern.
- Task complexity: Simple tasks (translation) need few shots; complex reasoning may need more.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Task: Sentiment classification (positive/negative)**

Zero-shot prompt:

```
Classify the sentiment of the movie review as positive or negative.

Review: "This movie was terrible."
Sentiment:
```

Model output might be: "Negative" (if the model understands sentiment).

#### **Step 2: One-shot**

```
Review: "I loved this movie!" → Sentiment: Positive
Review: "This movie was terrible." → Sentiment:
```

Model now has one example of mapping positive review to "Positive". It can analogize: "terrible" is opposite of "loved" → output "Negative".

#### **Step 3: Few-shot (3 examples)**

```
Review: "Amazing acting, great plot!" → Positive
Review: "Boring, waste of time." → Negative
Review: "Beautiful cinematography." → Positive
Review: "This movie was terrible." →
```

With three examples, the model sees two positive patterns and one negative. It learns the association: positive words (amazing, great, beautiful) → Positive; negative words (boring, waste, terrible) → Negative. It outputs "Negative".

#### **Step 4: Why few-shot > one-shot > zero-shot**

Zero-shot: Model must infer task from description alone. Risky if task ambiguous.
One-shot: One example disambiguates the task. Model can pattern-match.
Few-shot: Multiple examples show the range of inputs and outputs, reducing ambiguity. Model learns the decision boundary.

#### **Step 5: Limits of few-shot**

If the task is complex (e.g., solving algebraic equations), few-shot may fail. The model cannot learn truly new functions from 5 examples—it is retrieving patterns from pre-training, not learning from scratch.

---

### **7. How this appears inside neural networks and LLMs**

- **GPT-3 paper (2020):** Introduced few-shot learning as a primary evaluation method. Showed that with 1-5 examples, GPT-3 could perform translation, question answering, and arithmetic competitively with fine-tuned models.

- **In-context learning:** The term for few-shot without weight updates. It works because the model's attention mechanism can relate the query to the examples. The examples act as a short-term "task specification."

- **Few-shot vs fine-tuning:** Few-shot requires no training, no GPU, no data storage. Fine-tuning requires compute but works for tasks where few-shot fails (e.g., specialized domains, formatting rules).

- **Example ordering matters:** For few-shot, the order of examples can change accuracy by 10-20%. Putting the most informative examples first helps.

- **Model scale and few-shot:** Few-shot performance improves with model size. Smaller models (<1B) cannot do few-shot; 175B models excel at it. This emergent property appears only above a certain scale.

- **Meta-learning (learn to learn):** Some models are explicitly trained on few-shot tasks during pre-training (e.g., "few-shot learning" as a training objective). These models can learn from even fewer examples.

---

### **8. Brain-like connection (rapid task inference)**

The human brain performs few-shot learning constantly. You meet a new colleague and learn their name from one introduction (one-shot learning). You see a new type of animal and learn to recognize it from one example. This is not fine-tuning (which would require many repetitions) but rapid task inference. The brain's prefrontal cortex maintains a "task set"—a temporary configuration of neural circuits that biases processing toward the current task. Few-shot examples update this task set, similar to how prompt examples update the LLM's hidden state activations. This is why you can learn a new rule on the fly without changing your brain's synaptic weights.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Few-shot learning means the model learns new knowledge from a few examples. It can acquire new facts that weren't in pre-training."

_Why it is wrong:_ Few-shot does not teach the model new facts. It only shows the model how to format its output or which pattern to apply. The model cannot memorize a new fact (e.g., "the capital of Atlantis is X") from a few-shot example—it will not recall that fact later outside the prompt. Few-shot learning is in-context: the examples are only present in the current prompt. Once the prompt is forgotten, the model does not retain the knowledge. True learning requires weight updates (fine-tuning). Few-shot is a retrieval and pattern-matching mechanism, not long-term memory.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Few-shot learning is how most people use LLMs today.     |
|  You do not fine-tune ChatGPT—you give it examples in     |
|  the prompt. "Here is how I want you to respond. Now      |
|  answer this." This ability—learning from context without |
|  weight updates—is an emergent property of large models.  |
|  It is what makes LLMs flexible, interactive, and useful  |
|  for hundreds of tasks without retraining. Few-shot       |
|  is the superpower of scale: the model learns tasks on    |
|  the fly, just from examples in its input.                |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You want a model to classify customer support emails into "refund request," "technical issue," or "account question." You have 1,000 labeled examples. You need high accuracy (>95%).

**Question:** Should you use few-shot learning (5 examples in the prompt) or fine-tuning? Why? If you use fine-tuning, could you still use few-shot for the final deployment?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** You should use fine-tuning, not few-shot. Few-shot works for simple tasks where the model already understands the categories (e.g., sentiment). For a specific three-class classification with subtle distinctions (refund vs account issue), few-shot with 5 examples likely yields poor accuracy (<50%). Fine-tuning on 1,000 examples will train the model to recognize these specific categories accurately (>90%).

For final deployment, you can still use few-shot on top of fine-tuning: provide 1-2 examples in the prompt as reminders of the format, but the fine-tuned model already knows the task. However, fine-tuning alone should suffice. The key: few-shot is for tasks the model already understands but needs formatting guidance. Fine-tuning is for learning new, specific tasks with labeled data.
