# RLHF (Reinforcement Learning from Human Feedback)

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Training Paradigms**

---

### **1. Why this concept matters**

Instruction tuning teaches a model to follow instructions, but it does not teach it what a _good_ response looks like. A model could answer "How do I bake a cake?" with a technically correct but rude or dangerous response. RLHF solves this by using human preferences as a reward signal. Humans rank multiple model outputs, a reward model learns to predict those preferences, and the LLM is optimized via reinforcement learning to produce responses that humans prefer. RLHF is the secret sauce behind ChatGPT's helpfulness, harmlessness, and honesty—transforming a competent instruction-follower into a genuinely helpful assistant.

---

### **2. Core idea**

**RLHF aligns language models with human preferences by training a reward model on human comparisons, then using reinforcement learning (PPO) to optimize the LLM to generate responses that maximize the predicted reward.**

---

### **3. Concrete analogy**

Imagine training a puppy to fetch. You could give explicit instructions ("bring the ball back"), but the puppy might drop the ball halfway. Instead, you reward good behavior with treats. When the puppy brings the ball all the way back, you give a treat. When it drops it, no treat. Over time, the puppy learns what you want.

RLHF works the same way. The model generates responses (like the puppy fetching). Humans rank responses: "Response A is better than B." A reward model learns to predict these rankings (the treat). Then the LLM is trained via reinforcement learning to maximize the predicted reward (get more treats). The model learns what humans prefer without being explicitly told the rules: be helpful, be harmless, be honest.

---

### **4. ASCII diagram**

```
RLHF three-stage pipeline:

    Stage 1: Supervised Fine-Tuning (SFT)
    ┌─────────────────────────────────────────────┐
    │ Instruction-response pairs                  │
    │ "Explain gravity" → "Gravity pulls objects" │
    └─────────────────────────────────────────────┘
                          │
                          ▼
    Base model → SFT model (learns to follow instructions)


    Stage 2: Reward Modeling
    ┌─────────────────────────────────────────────┐
    │ Human preferences: "Response A > Response B"│
    │ Response A: "The answer is 4." (helpful)    │
    │ Response B: "Why ask?" (unhelpful)          │
    └─────────────────────────────────────────────┘
                          │
                          ▼
    SFT model → Reward Model (r(x, y) = score)


    Stage 3: Reinforcement Learning (PPO)
    ┌─────────────────────────────────────────────┐
    │ Prompt x → SFT model → Response y           │
    │ Reward model scores r(x, y)                 │
    │ PPO updates model to maximize r             │
    └─────────────────────────────────────────────┘
                          │
                          ▼
    RLHF model (helpful, harmless, honest)


Detailed RLHF loop:

    Prompt: "What is 2+2?"
         │
         ▼
    LLM generates: "4" (response A)
                  "Why do you ask?" (response B)
         │
         ▼
    Humans rank: A > B
         │
         ▼
    Reward model learns: r(A) = 0.8, r(B) = 0.1
         │
         ▼
    PPO updates LLM weights to increase probability of high-reward responses
```

---

### **5. Mathematical formulation**

**Stage 1: Supervised Fine-Tuning (SFT)**

Standard instruction tuning on demonstration data:

$$
L_{\text{SFT}} = -\sum_{(x,y) \in D_{\text{demo}}} \log \pi_{\text{SFT}}(y | x)
$$

**Stage 2: Reward Modeling (RM)**

Collect human preference dataset: for each prompt x, two responses y*w (preferred) and y_l (dispreferred). Train reward model r*φ(x, y) using Bradley-Terry loss:

$$
L_{\text{RM}} = -\log \sigma(r_φ(x, y_w) - r_φ(x, y_l))
$$

Where σ is sigmoid function.

**Stage 3: Reinforcement Learning (PPO - Proximal Policy Optimization)**

Optimize π_θ to maximize reward while staying close to SFT model:

$$
\text{maximize} \quad E_{x \sim D, y \sim \pi_θ(y|x)} \left[ r_φ(x, y) - \beta \cdot D_{\text{KL}}(\pi_θ \| \pi_{\text{SFT}}) \right]
$$

KL penalty prevents the model from drifting too far and generating gibberish to exploit the reward model.

**PPO update (simplified):**

$$
\theta \leftarrow \theta + \alpha \nabla_\theta E\left[ \frac{\pi_θ(y|x)}{\pi_{\text{old}}(y|x)} A(x,y) \right]
$$

Where A(x,y) is advantage (how much better this response is than average).

---

### **6. Worked example (step-by-step)**

#### **Step 1: SFT (skip—assume we have a base instruction model)**

#### **Step 2: Collect human preferences**

Prompt: "How do I make friends?"

Response A: "Join clubs, be yourself, and don't be afraid to start conversations." (helpful, kind)
Response B: "Just talk to people, it's not hard." (dismissive, unhelpful)

Human label: A > B

#### **Step 3: Train reward model**

The reward model learns to assign higher score to A than B:

r(A) = 0.85, r(B) = 0.30

#### **Step 4: RL training (PPO) loop**

Prompt: "I'm feeling lonely. Any advice?"

LLM generates candidate responses:

- "I'm sorry you're feeling lonely. Try reaching out to a friend or joining a local club." (empathetic)
- "You'll be fine." (dismissive)

Reward model scores: first response 0.9, second 0.2.

PPO updates LLM weights to increase probability of empathetic, helpful responses and decrease dismissive ones.

#### **Step 5: After RLHF**

The model now consistently produces helpful, empathetic responses. It has learned the _pattern_ of what humans prefer, not just the content of training examples.

---

### **7. How this appears inside neural networks and LLMs**

- **ChatGPT (GPT-3.5):** First major model to use RLHF at scale. Collected human preferences on model outputs, trained reward model, optimized with PPO. The result was dramatically more helpful and less toxic than GPT-3.

- **Llama 2 Chat:** Used RLHF with two reward models (helpfulness + safety) and a safety-specific data collection pipeline.

- **Claude (Anthropic):** Used "Constitutional AI"—RLHF with AI-generated feedback instead of human labels for some dimensions (harmlessness). Reduced reliance on human annotators.

- **PPO vs other RL algorithms:** PPO is standard because it is stable and sample-efficient. It clips updates to prevent destructive policy changes.

- **KL penalty (β):** Hyperparameter controlling how much the model can deviate from the SFT model. Typical β = 0.01-0.1. Too low → reward hacking (exploiting reward model bugs). Too high → no improvement.

- **Reward hacking:** The model learns to exploit reward model flaws, e.g., repeating "I am helpful" to get high reward. KL penalty and adversarial training mitigate this.

- **Cost:** RLHF requires tens of thousands to millions of human preference labels. Expensive but essential for alignment.

---

### **8. Brain-like connection (dopamine reward learning)**

The brain's reinforcement learning system uses dopamine as a reward signal. When you do something good (eat, achieve a goal), dopamine neurons fire, reinforcing the preceding actions. RLHF mirrors this: the reward model provides the "dopamine signal" (high reward for good responses), and PPO is the learning rule that updates the model to repeat those behaviors. Humans learn social norms (helpfulness, politeness, honesty) through this same mechanism: positive feedback from others shapes behavior over time. RLHF is a computational model of social learning—the model learns to be helpful because it is "rewarded" for helpful responses.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "RLHF is just supervised fine-tuning on human-ranked data. You could just train on the preferred responses directly."

_Why it is wrong:_ Supervised learning would only train on the single best response per prompt. RLHF learns from _relative_ preferences (A > B, not just "A is correct"). This provides richer signal: the model learns that B is worse than A, not just that A is good. Additionally, RLHF allows the model to explore and generate novel responses during training, then get feedback. SFT only trains on existing examples. RLHF can discover better responses than any in the training data. This exploration-exploitation tradeoff is why RLHF outperforms pure SFT for alignment.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Instruction tuning teaches models to follow commands.    |
|  RLHF teaches them to be helpful, harmless, and honest.   |
|  It aligns models with human values, reducing toxicity,   |
|  bias, and harmful outputs. RLHF is the reason ChatGPT    |
|  feels different from GPT-3. It is the alignment layer    |
|  that turns raw intelligence into safe, helpful           |
|  assistance. Without RLHF, LLMs are powerful but risky.   |
|  With it, they become tools you can trust.                |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are training a customer service chatbot. You have 10,000 labeled examples of good responses. You also have a budget to collect 1,000 preference comparisons (Response A vs Response B).

**Question:** Should you use SFT only (train on the 10,000 examples) or SFT + RLHF (train on examples, then RLHF on preferences)? Why?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Use SFT + RLHF. The 10,000 examples will teach the model the basic format and content of customer service responses (SFT). RLHF on 1,000 preference comparisons will refine the model to learn nuances: which response is more polite, which is more helpful, which de-escalates anger. SFT alone may produce technically correct but socially awkward responses. RLHF aligns the model with customer service norms (empathy, patience, professionalism). The preference comparisons capture this "better than" signal that SFT cannot provide. With limited preference budget, focus RLHF on edge cases or ambiguous scenarios where SFT fails. The combination is more effective than either alone.
