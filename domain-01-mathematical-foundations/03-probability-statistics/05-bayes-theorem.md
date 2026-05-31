# Bayes' theorem

## **DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Probability & Statistics**

---

### **1. Why this concept matters for building intelligent systems**

You have a belief about the world. Then you see evidence. How should your belief change? This is the fundamental question of learning, and Bayes' theorem is the mathematical answer. It tells you how to update probabilities when new data arrives. Every time an LLM predicts the next word, it is implicitly applying Bayesian reasoning over its vocabulary. Every time a spam filter learns from a new email, Bayes is at work. Every time a medical AI updates a diagnosis from a test result, Bayes is the engine. Without Bayes, learning is just heuristics. With Bayes, learning has a rational foundation.

---

### **2. Core idea**

**Bayes' theorem describes the probability of a hypothesis given observed evidence as proportional to the likelihood of the evidence under the hypothesis multiplied by the prior probability of the hypothesis.**

---

### **3. Concrete analogy**

Imagine you hear a noise at night. You have two hypotheses: H₁ = "it's a burglar" and H₂ = "it's the cat knocking something over."

Your prior belief: burglaries are rare (P(burglar)=0.001), cat noises are common (P(cat)=0.1).

You hear a loud crash (evidence E). The likelihood: P(crash|burglar)=0.9 (burglars make noise), P(crash|cat)=0.3 (cats sometimes cause crashes, but not always).

Bayes' theorem combines prior and likelihood to compute posterior probability: given you heard a crash, how likely is a burglar? Even though burglars almost always make noise, the crash is much more likely to be the cat because cats are far more common. The theorem quantifies this intuition.

---

### **4. ASCII diagram**

```text
Bayes' theorem as updating beliefs:

    Prior                     Likelihood                Posterior
    P(H)                      P(E|H)                    P(H|E)

    Before evidence        How probable is the      Updated belief after
        │                 evidence if H is true?       seeing evidence
        ▼                         │                         │
     0.8│  H₁                  0.9│  H₁                  0.7│  H₁
        │   │                     │   │                     │   │
     0.2│  H₂                  0.3│  H₂                  0.3│  H₂
        ▼                         ▼                         ▼
    Belief in H₁=0.8         If H₁ true, E has       After seeing E,
    Belief in H₂=0.2         high probability        H₁ more confident

    The update factor: P(E|H) / P(E)

    Bayes' rule in action:

        Belief before         Evidence strength          Belief after
        (prior)          ×    (likelihood ratio)    =    (posterior)

        P(H)             ×    P(E|H) / P(E)         =    P(H|E)

    If evidence is surprising (low P(E)), it has larger impact.
```

---

### **5. Mathematical formulation**

**Standard form:**

$$
P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}
$$

(Unicode: P(H|E) = P(E|H) × P(H) / P(E))

Where:

- P(H|E) = posterior probability of hypothesis H given evidence E
- P(E|H) = likelihood of evidence E assuming H is true
- P(H) = prior probability of H (before seeing evidence)
- P(E) = marginal probability of evidence (averaged over all hypotheses)

**Expanded form (using law of total probability):**

For a set of mutually exclusive hypotheses H₁, H₂, ..., Hₙ that cover all possibilities:

$$
P(H_i|E) = \frac{P(E|H_i) P(H_i)}{\sum_{j=1}^n P(E|H_j) P(H_j)}
$$

(Unicode: P(H_i|E) = P(E|H_i)P(H_i) / Σ_j P(E|H_j)P(H_j))

**Odds form (for comparing two hypotheses):**

$$
\frac{P(H_1|E)}{P(H_2|E)} = \frac{P(E|H_1)}{P(E|H_2)} \cdot \frac{P(H_1)}{P(H_2)}
$$

(Unicode: Posterior odds = Likelihood ratio × Prior odds)

**Bayesian updating in sequence:** After seeing evidence E₁, the posterior becomes the prior for E₂:

P(H|E₁, E₂) ∝ P(E₂|H) × P(H|E₁)

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define the problem**

A medical test for a disease has:

- Disease prevalence: P(D) = 0.01 (1% of population has it)
- Test sensitivity: P(Pos|D) = 0.99 (99% of sick test positive)
- Test specificity: P(Neg|not D) = 0.95 (95% of healthy test negative)

You test positive. What is P(D|Pos)?

#### **Step 2: Identify quantities**

P(D) = 0.01 (prior)
P(not D) = 0.99
P(Pos|D) = 0.99 (likelihood)
P(Pos|not D) = 1 - 0.95 = 0.05 (false positive rate)

#### **Step 3: Compute P(Pos) using law of total probability**

P(Pos) = P(Pos|D) × P(D) + P(Pos|not D) × P(not D)
P(Pos) = (0.99 × 0.01) + (0.05 × 0.99)
P(Pos) = 0.0099 + 0.0495 = 0.0594

#### **Step 4: Apply Bayes' theorem**

P(D|Pos) = P(Pos|D) × P(D) / P(Pos)
P(D|Pos) = (0.99 × 0.01) / 0.0594
P(D|Pos) = 0.0099 / 0.0594 ≈ 0.1667

#### **Step 5: Interpret**

Despite a 99% accurate test, a positive result means only about 16.7% chance of having the disease. Why? The disease is rare (1%), so false positives (5% of 99% = 4.95% of population) outnumber true positives (0.99% of population).

#### **Step 6: Update with a second independent test**

Now take a second test (same accuracy). Given first test was positive, our prior becomes P(D) = 0.1667.

P(Pos₂|D) = 0.99, P(Pos₂|not D) = 0.05

P(Pos₂) = 0.99×0.1667 + 0.05×0.8333 = 0.165 + 0.0417 = 0.2067

P(D|Pos₁, Pos₂) = (0.99 × 0.1667) / 0.2067 = 0.165 / 0.2067 ≈ 0.798

After two positive tests, probability rises to ~80%.

---

### **7. How this appears inside neural networks and LLMs**

- **Naive Bayes classifiers:** Used for spam detection, sentiment analysis, and text classification. Assumes features are independent given the class.

- **Bayesian neural networks:** Place priors on weights, compute posterior distributions (approximated via variational inference or MCMC). Provides uncertainty estimates.

- **Bayesian optimization:** Efficiently tunes hyperparameters by building a probabilistic surrogate model (Gaussian process) and using Bayes' rule to decide where to sample next.

- **Kalman filters:** Used in robotics and time series. Repeatedly apply Bayes' rule: predict (prior) then update with measurement (posterior).

- **Latent Dirichlet Allocation (LDA):** Topic model for documents that uses Bayesian inference to discover latent topics.

- **Bayesian model selection:** Compares different neural network architectures using marginal likelihood.

- **Uncertainty calibration:** LLM output probabilities often need calibration; Platt scaling and temperature scaling can be viewed as empirical Bayes methods.

- **Thompson sampling:** In reinforcement learning, selects actions by sampling from posterior distribution over rewards.

---

### **8. Brain-like connection (predictive coding)**

The brain continuously performs Bayesian inference. When you catch a ball, your brain predicts its trajectory (prior), integrates visual feedback (likelihood), and updates its prediction (posterior). The predictive coding theory of perception states that the brain maintains hierarchical generative models of sensory input, with higher levels predicting activity at lower levels. Prediction errors (differences between prediction and actual sensory input) propagate upward, updating beliefs via something approximating Bayes' rule. Dopamine signals in the basal ganglia encode reward prediction errors—a form of Bayesian updating. Your moment-to-moment experience of surprise is your brain's Bayesian engine reporting that the evidence did not match the prior.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Bayes' theorem is just a formula for medical tests. I don't need it for deep learning."

_Why it is wrong:_ Bayes' theorem is not a niche formula—it is the foundation of rational learning under uncertainty. In deep learning, you use it every time you:

- Interpret a softmax output as a probability (implicitly assuming a prior over classes)
- Use dropout as approximate Bayesian inference
- Apply early stopping (a form of empirical Bayes)
- Regularize with L2 weight decay (equivalent to a Gaussian prior over weights)
- Calibrate model probabilities

Without understanding Bayes, you are applying these techniques blindly. Understanding Bayes reveals why they work and when they fail.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Learning is belief revision. Evidence arrives. Your      |
|  beliefs should respond. Bayes' theorem is the only       |
|  consistent rule for how to respond. It quantifies how    |
|  surprising evidence should change your mind, how rare    |
|  events affect rare hypotheses, and how multiple pieces   |
|  of evidence combine. Without Bayes, you are guessing.    |
|  With Bayes, you are reasoning.                           |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

A factory produces widgets. Two machines make them: Machine A makes 60% of widgets, Machine B makes 40%. Machine A produces 2% defectives. Machine B produces 5% defectives. You find a defective widget.

**Question:** What is the probability it came from Machine B?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

P(B) = 0.4 (prior)
P(D|B) = 0.05
P(D|A) = 0.02

P(D) = 0.05×0.4 + 0.02×0.6 = 0.02 + 0.012 = 0.032

P(B|D) = (0.05 × 0.4) / 0.032 = 0.02 / 0.032 = 0.625

Even though Machine B produces fewer total widgets, it produces a higher proportion of defectives, so given a defective, probability it came from B is 62.5%.
822
