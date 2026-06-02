# Zero-shot learning

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Training Paradigms**

---

### **1. Why this concept matters**

You have no examples. You cannot fine-tune. You cannot even provide a single demonstration. Can the model still perform the task? Zero-shot learning says yes. The model relies entirely on its pre-trained knowledge and the task description in the prompt. No examples, no weight updates—just instructions. This is the purest test of an LLM's ability to understand and follow natural language instructions. Modern LLMs (GPT-4, Llama 3, Claude) excel at zero-shot tasks that would have been impossible for models just a few years ago. Zero-shot is how most users interact with ChatGPT: they give instructions, not examples.

---

### **2. Core idea**

**Zero-shot learning enables a model to perform a new task from a natural language description alone, without any examples or parameter updates, relying solely on the model's pre-trained knowledge and instruction-following ability.**

---

### **3. Concrete analogy**

Imagine you are in a foreign country and meet someone who speaks no English. You have no phrasebook, no examples. You point to an apple and say "apple." They nod. You point to a banana and say "banana." They nod. Then you point to a cat and say "cat." They look confused. You try again: "cat." Nothing.

Now imagine you instead say: "I am going to say an English word, and you say the word in your language." They understand the instruction immediately and respond correctly for every word. This is zero-shot: you explained the task without showing any examples.

Zero-shot works when the model already understands the concept (e.g., "translate") and can map it to its pre-trained knowledge. The instruction activates the relevant capability without demonstrations.

---

### **4. ASCII diagram**

```
Zero-shot vs One-shot vs Few-shot:

Zero-shot (no examples):
    Prompt: "Translate 'Hello' to Spanish."
    Output: "Hola"
    (Model understands the task from description alone)

One-shot (1 example):
    Prompt: "English: Hello → Spanish: Hola
              English: Goodbye → Spanish:"
    Output: "Adiós"

Few-shot (3 examples):
    Prompt: "Apple → Manzana
             Banana → Plátano
             Cherry → Cereza
             Grape →"
    Output: "Uva"


Zero-shot instruction formats:

    Direct instruction:
        "Classify this review as positive or negative: 'Terrible movie.'"

    Role-playing:
        "You are a sentiment classifier. Output only 'Positive' or 'Negative'.
         Review: 'Amazing film!'"

    Task description:
        "Convert the following English sentence to French: 'The cat is black.'"


Why zero-shot works (emergent ability):

    Model scale → Zero-shot accuracy

    Accuracy↑
           │                                    ● (175B)
           │                              ●
           │                         ●
           │                    ●
           │               ●
           │          ●
           │     ●
           │●   (1B)
           └────────────────────────────────→ Model size

    Zero-shot performance emerges only above ~10B parameters.
    Smaller models require at least one example (one-shot).
```

---

### **5. Mathematical formulation**

**Zero-shot inference (no examples):**

Given:

- Task description T (natural language instruction)
- Query input x

Construct prompt P = [T, x]

Model generates output:

$$
\hat{y} = \arg\max_y P_{\theta}(y | P)
$$

Where θ is fixed (no weight updates). No examples are provided.

**The key requirement: Instruction following**

The model must understand the relationship between T and x. This requires pre-training on data that includes instructions, such as:

- "Translate X to Y" patterns in web text
- Instruction-response pairs (e.g., FLAN, T0, SuperNaturalInstructions)
- Chat data (user-assistant conversations)

**Mathematical condition for zero-shot success:**

The model must have learned the mapping from instruction space to task space during pre-training. Formally, there exists a function f such that:

$$
f_{\theta}(T, x) \approx y
$$

For unseen tasks, this holds if the model has learned to generalize instruction patterns.

**Comparison to few-shot:**

Zero-shot: P = [T, x]
Few-shot: P = [T, (x₁,y₁), ..., (x_K,y_K), x]

Few-shot provides additional disambiguation via examples. Zero-shot relies on the instruction being unambiguous.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Zero-shot translation**

Task: Translate English to Spanish

Prompt:

```
Translate the following English sentence to Spanish:

"Hello, how are you?"
```

Model (GPT-4) output:

```
"Hola, ¿cómo estás?"
```

Why it works: The model has seen "translate X to Y" thousands of times during pre-training. It understands the instruction without examples.

#### **Step 2: Zero-shot classification**

Prompt:

```
Classify the sentiment of this movie review as positive, negative, or neutral.

Review: "The acting was superb but the plot was predictable."
```

Model output:

```
Neutral
```

Why it works: The model understands sentiment concepts and can apply them to new text.

#### **Step 3: Zero-shot task that fails for smaller models**

Prompt (complex reasoning):

```
A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?
```

Small model (1B) output:

```
The ball costs $0.10.
```

(Correct answer is $0.05. The small model fails at arithmetic reasoning.)

Large model (GPT-4) output:

```
Let x be the cost of the ball. Then the bat costs x + 1.00.
Total: x + (x + 1.00) = 1.10 → 2x + 1.00 = 1.10 → 2x = 0.10 → x = 0.05.
The ball costs $0.05.
```

Zero-shot reasoning emerges at scale.

#### **Step 4: Zero-shot with formatting**

Prompt:

```
Output only the answer as a number: What is 15 × 7?
```

Model output:

```
105
```

The instruction "output only the answer as a number" overrides the model's default behavior of generating verbose responses. This is zero-shot formatting control.

---

### **7. How this appears inside neural networks and LLMs**

- **Instruction-tuned models:** GPT-3.5, GPT-4, Llama 2/3 Chat, Claude, Mistral Instruct are fine-tuned specifically for zero-shot instruction following. Base models (non-instruct) perform poorly on zero-shot tasks.

- **Emergent ability:** Zero-shot performance appears only above ~10B parameters. Smaller models require few-shot or fine-tuning. This is why GPT-2 (1.5B) could not follow instructions, but GPT-3 (175B) could.

- **Task description matters:** Wording affects zero-shot accuracy. "Classify sentiment" works better than "What is the sentiment?" Instructions should be explicit, concise, and formatted consistently.

- **Role prompting:** "You are a helpful assistant" primes the model for zero-shot instruction following. This is why chat models have system prompts.

- **Zero-shot vs few-shot tradeoff:** Zero-shot is simpler (no examples to write) but less accurate for ambiguous tasks. Few-shot provides disambiguation but consumes tokens.

- **FLAN (Fine-tuned Language Net):** Models specifically fine-tuned on thousands of instructions (zero-shot style) show dramatically improved zero-shot performance. FLAN-T5, FLAN-UL2.

- **Zero-shot reasoning (Chain-of-Thought):** "Let's think step by step" is a zero-shot prompt that induces multi-step reasoning. This is a zero-shot technique (no examples provided).

---

### **8. Brain-like connection (instruction following without demonstration)**

Humans excel at zero-shot learning. If someone says, "Please list all the items in this room starting with the letter 'C'," you understand immediately. You do not need examples. This requires language comprehension, concept knowledge (letters), and task decomposition. Brain regions involved: Broca's area (interpreting instructions), prefrontal cortex (task maintenance), and parietal cortex (visual search). Zero-shot instruction following is a high-level cognitive skill that emerges late in development (age 5-7). Similarly, LLMs only exhibit robust zero-shot performance at large scales—it is an emergent property of size and instruction tuning, not present in smaller models.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Zero-shot learning means the model has never seen anything related to the task before."

_Why it is wrong:_ Zero-shot means no examples _in the prompt_. The model has almost certainly seen related tasks during pre-training. For translation, the model has seen parallel text ("Hello → Hola") billions of times. For classification, it has seen sentiment-labeled data. Zero-shot works because the model has pre-existing knowledge. True "zero-shot" (no prior exposure) is impossible—the model would have no basis to perform the task. The term refers to the inference setting (no provided examples), not the pre-training data. A more accurate term is "zero-example" or "instruction-based" learning.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Zero-shot is the simplest way to use an LLM. No         |
|  examples to write, no fine-tuning to run. Just an       |
|  instruction. This is how ChatGPT works: you tell it     |
|  what to do, and it does it. Zero-shot capabilities have  |
|  skyrocketed with model scale and instruction tuning.    |
|  Modern LLMs can translate, summarize, classify, and     |
|  reason with no demonstrations. Zero-shot is the          |
|  user-friendly face of LLMs—the reason anyone can        |
|  interact with them without machine learning expertise.   |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a 7B parameter instruction-tuned model. You want it to extract dates from text (e.g., "I was born on May 3, 1990" → "May 3, 1990").

**Question:** Would you use zero-shot or few-shot? Why? If zero-shot fails, what would you try next?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Start with zero-shot. Provide a clear instruction: "Extract the date from the following text. Output only the date." If the model works, done. Zero-shot is easiest.

If zero-shot fails (model extracts extra text or format wrong), try few-shot with 1-3 examples:

```
Text: "The meeting is scheduled for June 15, 2024." → Date: June 15, 2024
Text: "I joined the company on 2023-01-10." → Date: 2023-01-10
Text: "Born May 3, 1990" → Date: May 3, 1990
Text: "The deadline is tomorrow." →
```

If few-shot still fails, consider fine-tuning on date extraction examples (requires labeled dataset). For a 7B model, zero-shot may already work if it was instruction-tuned on similar tasks. Try zero-shot first; fall back to few-shot; fine-tune only if necessary.
