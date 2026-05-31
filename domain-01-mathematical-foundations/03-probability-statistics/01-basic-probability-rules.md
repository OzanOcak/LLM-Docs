# Basic probability rules

## **DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Probability & Statistics**

---

### **1. Why this concept matters for building intelligent systems**

The world is uncertain. Sensors have noise. Language is ambiguous. The future is unpredictable. A system that cannot handle uncertainty is brittle—it will fail the moment reality deviates from its training data. Probability is the mathematics of uncertainty. It gives you the rules for quantifying doubt, combining evidence, and making optimal decisions when you cannot be certain. Every LLM, every vision system, every reinforcement learning agent is probabilistic at its core, even when it pretends to be deterministic.

---

### **2. Core idea**

**Probability rules provide a consistent framework for quantifying uncertainty, combining events, and updating beliefs when new evidence arrives, with all probabilities lying between 0 (impossible) and 1 (certain).**

---

### **3. Concrete analogy**

Imagine a bag containing 3 red marbles and 2 blue marbles. You reach in without looking.

- The probability of drawing a red marble is 3/5 = 0.6 (60%).
- The probability of drawing a blue marble is 2/5 = 0.4 (40%).
- The probability of drawing either red or blue is 1.0 (certainty).

Now draw one marble, then draw another without replacing the first. The probability of the second draw changes based on what happened first. If the first was red, only 2 red and 2 blue remain (probability of second red = 2/4 = 0.5). If the first was blue, 3 red and 1 blue remain (probability of second red = 3/4 = 0.75).

Your AI faces this constantly: given that it saw a "the," what is the probability the next word is "cat" versus "dog"? Probability rules turn this intuition into precise calculations.

---

### **4. ASCII diagram**

```text
Probability as area on a unit square (Venn diagram):

    Total space = 1.0 (certainty)

    ┌─────────────────────────────┐
    │                             │
    │    ┌─────────┐              │
    │    │    A    │              │
    │    │  P(A)   │              │
    │    └─────────┘              │
    │                             │
    │    ┌─────────┐              │
    │    │    B    │              │
    │    │  P(B)   │              │
    │    └─────────┘              │
    │                             │
    └─────────────────────────────┘

    Overlapping events (A ∩ B):

    ┌─────────────────────────────┐
    │    ┌─────────┐              │
    │    │    A    │              │
    │    │   ┌───┐ │              │
    │    │   │A∩B│ │              │
    │    └───┼───┼─┘              │
    │        │ B │                │
    │        └───┘                │
    │                             │
    └─────────────────────────────┘

    P(A or B) = P(A) + P(B) - P(A and B)
    (Subtract overlap to avoid double-counting)
```

---

### **5. Mathematical formulation**

**Probability axioms (Kolmogorov):**

1. Non-negativity: P(A) ≥ 0 for any event A
2. Normalization: P(Ω) = 1 (Ω = all possible outcomes)
3. Additivity: If A and B are mutually exclusive (cannot both happen), then P(A ∪ B) = P(A) + P(B)

**Complement rule (probability of "not A"):**

$$
P(A^c) = 1 - P(A)
$$

(Unicode: P(not A) = 1 - P(A))

**Addition rule (general, for any A and B):**

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

(Unicode: P(A or B) = P(A) + P(B) - P(A and B))

**Multiplication rule (for independent events):**

If A and B are independent, then:

$$
P(A \cap B) = P(A) \times P(B)
$$

(Unicode: P(A and B) = P(A) × P(B))

**Conditional probability (probability of A given B has occurred):**

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

(Unicode: P(A|B) = P(A and B) / P(B))

**General multiplication rule (always true):**

$$
P(A \cap B) = P(A|B) \times P(B) = P(B|A) \times P(A)
$$

**Law of total probability (partitioning into cases):**

If B₁, B₂, ..., Bₙ partition Ω (mutually exclusive and exhaustive), then:

$$
P(A) = \sum_{i=1}^n P(A|B_i) P(B_i)
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define the scenario**

A medical test for a disease has:

- Disease prevalence: P(D) = 0.01 (1% of population has it)
- Test sensitivity: P(Positive | D) = 0.99 (99% of sick test positive)
- Test specificity: P(Negative | not D) = 0.95 (95% of healthy test negative)

#### **Step 2: Compute probability of a positive test**

Using law of total probability:

P(Pos) = P(Pos|D) × P(D) + P(Pos|not D) × P(not D)

P(Pos|not D) = 1 - P(Neg|not D) = 1 - 0.95 = 0.05

P(not D) = 1 - P(D) = 0.99

P(Pos) = (0.99 × 0.01) + (0.05 × 0.99) = 0.0099 + 0.0495 = 0.0594

#### **Step 3: Compute probability you actually have the disease given a positive test**

Using conditional probability (Bayes' rule, though not yet formally introduced):

P(D|Pos) = P(Pos|D) × P(D) / P(Pos) = (0.99 × 0.01) / 0.0594 = 0.0099 / 0.0594 ≈ 0.1667

Even with a 99% accurate test, a positive result means only ~16.7% chance of having the disease because the disease is rare (1%) and false positives (5% of healthy) outnumber true positives.

#### **Step 4: Verify complement rule**

P(not D | Pos) = 1 - P(D|Pos) = 1 - 0.1667 = 0.8333

The vast majority of positive tests are false positives. This counterintuitive result is critical for medical AI and any system making decisions under rare-event conditions.

---

### **7. How this appears inside neural networks and LLMs**

- **Softmax output layer**: Converts logits into probabilities summing to 1. The model's prediction is a probability distribution over possible next words or classes.

- **Cross-entropy loss**: Measures how well predicted probability distribution matches true distribution. Minimizing cross-entropy is equivalent to maximizing likelihood of the data.

- **Sampling in text generation**: LLMs don't just pick the highest-probability word. They sample from the probability distribution (with temperature scaling) to produce diverse, creative outputs.

- **Bayesian neural networks**: Treat weights as probability distributions, not fixed values. Predictions come with uncertainty estimates.

- **Dropout regularization**: Randomly drops neurons with probability p during training. Equivalent to training an ensemble of subnetworks.

- **Attention as probability**: Softmax over attention scores produces a probability distribution over input positions, indicating "how much to attend" to each.

- **Language model perplexity**: exp(average negative log probability) measures how "surprised" the model is by test data. Lower perplexity = better predictions.

- **Reinforcement learning**: Policies map states to probability distributions over actions. Value functions estimate expected future reward.

---

### **8. Brain-like connection (predictive coding)**

The brain continuously makes probabilistic predictions about sensory input. When you reach for a coffee cup, your brain predicts the weight, temperature, and texture. The mismatch between prediction and actual sensation (prediction error) drives learning and attention. This "predictive coding" theory suggests the brain implements something like Bayesian inference: prior beliefs (P(hypothesis)) combined with sensory evidence (P(data|hypothesis)) yield posterior beliefs (P(hypothesis|data)). Dopamine signals encode reward prediction errors—a probabilistic quantity. Your daily experience of surprise is your brain's probability calculation making itself felt.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Probability 0.5 means 'I don't know,' and probability 0.9 means 'I'm pretty sure.'"

_Why it is wrong:_ Probability 0.5 is a precise statement: an event that occurs half the time in the long run. It is not a confession of ignorance. In many contexts (like weather forecasts), P(rain)=0.5 means that in 100 days with identical conditions, it rains on about 50 of them. This is different from "I have no idea" (which would be a uniform distribution over all possibilities, not a point estimate). Confusing epistemic uncertainty (lack of knowledge) with aleatoric uncertainty (inherent randomness) is a common pitfall in AI system design.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Certainty is a lie the universe never promised you.      |
|  Sensors fail. Words have multiple meanings. The future   |
|  is not determined. Probability rules are your defense    |
|  against this uncertainty. They let you quantify doubt,   |
|  combine noisy evidence, and make decisions that are      |
|  optimal in expectation. Build an AI without probability  |
|  and it shatters on first contact with the real world.    |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You roll a fair six-sided die twice.

**Question:** What is P(sum = 7)? What is P(sum = 7 | first roll = 3)? Are the events "first roll = 3" and "sum = 7" independent?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Total outcomes: 6×6 = 36

Pairs summing to 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) → 6 pairs

P(sum=7) = 6/36 = 1/6 ≈ 0.1667

P(sum=7 | first=3) = P(second=4) = 1/6 ≈ 0.1667

Since P(sum=7 | first=3) = P(sum=7), the events are independent. Knowing the first roll does not change probability of sum=7.
