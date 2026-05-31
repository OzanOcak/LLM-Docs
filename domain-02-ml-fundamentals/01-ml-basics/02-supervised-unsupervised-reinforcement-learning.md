# Supervised vs unsupervised vs reinforcement learning

## **DOMAIN: MACHINE LEARNING FUNDAMENTALS | Sub domain: ML Basics**

---

### **1. Why this concept matters for building intelligent systems**

Not all learning is the same. Sometimes you have a teacher showing you the right answers (supervised). Sometimes you have no answers and must find hidden structure (unsupervised). Sometimes you get only rewards or punishments for your actions, with no one telling you the right move (reinforcement). Each type of learning appears in intelligent systems, and each requires different algorithms and data. Knowing which one to use for which problem is the first design decision in building any mind-like system. Choose wrong and your system cannot learn. Choose right and you match the learning strategy to the nature of the task.

---

### **2. Core idea**

**Supervised learning uses labeled input-output pairs to learn a mapping; unsupervised learning finds hidden structure in unlabeled data; reinforcement learning learns actions by maximizing cumulative rewards through trial and error.**

---

### **3. Concrete analogy**

Three ways to learn chess:

- **Supervised**: You study a database of grandmaster games, where each board position is labeled with the optimal move. You learn to predict the move from any position.

- **Unsupervised**: You look at thousands of board positions without labels. You notice clusters: some positions have pieces concentrated in the center, others have castled kings. You discover patterns without being told they matter.

- **Reinforcement**: You play games against yourself. When you win, you get positive reward. When you lose, negative reward. You try moves, observe outcomes, and gradually discover which sequences lead to checkmate. No one tells you the right move at any step—only final outcomes.

Real intelligence uses all three. A child learns object names from labeled examples (supervised), discovers that dogs and cats form separate groups (unsupervised), and learns to walk by trying and falling (reinforcement).

---

### **4. ASCII diagram**

```text
Three paradigms of machine learning:

    SUPERVISED LEARNING

    Training data:         Learning:         Prediction:
    ┌──────┬──────┐         ┌─────┐          New x → ŷ
    │ x₁   │ y₁   │         │     │
    │ x₂   │ y₂   │─────→   │  f  │─────→
    │ x₃   │ y₃   │         │     │
    │ .    │ .    │         └─────┘
    └──────┴──────┘
    Inputs + Labels       Learned Mapping    Output for new input

    Examples: Classification, Regression, Object Detection


    UNSUPERVISED LEARNING

    Training data:         Learning:         Discovery:
    ┌──────┐               ┌─────┐          Clusters, Patterns
    │ x₁   │               │     │
    │ x₂   │               │  g  │─────→    Compression, Outliers
    │ x₃   │─────→         │     │
    │ .    │               └─────┘          Reduced dimensions
    └──────┘
    Inputs only           Find Structure    Hidden regularities

    Examples: Clustering, Dimensionality Reduction, Anomaly Detection


    REINFORCEMENT LEARNING

    State s₁──→ Action a₁──→ Reward r₁, State s₂
         ↑                         │
         └───────── Policy π ←──────┘

    Agent learns by interacting with environment:

    [State] → Agent → [Action] → Environment → [Reward, Next State]
       ↑                                              │
       └──────────────────────────────────────────────┘

    Examples: Game Playing, Robotics, Self-driving Cars
```

---

### **5. Mathematical formulation**

**Supervised learning:**

Given: Training data D = {(x₁, y₁), ..., (xₙ, yₙ)} with x_i ∈ X, y_i ∈ Y

Goal: Learn function f: X → Y that minimizes expected error on new data

For classification (discrete Y):

$$
\hat{y} = \arg\max_{c} P(Y=c | X=x)
$$

For regression (continuous Y):

$$
\hat{y} = E[Y | X=x]
$$

**Unsupervised learning:**

Given: Data {x₁, ..., xₙ} with x_i ∈ X, no labels

Goal: Find structure in X, such as clusters, low-dimensional representations, or density estimates

Clustering: Partition into K groups minimizing within-group distance

$$
\min_{S_1,...,S_K} \sum_{k=1}^K \sum_{x \in S_k} \|x - \mu_k\|^2
$$

Dimensionality reduction: Find lower-dimensional z such that x ≈ g(z)

**Reinforcement learning:**

Given: Environment defined by states S, actions A, transition dynamics P(s'|s,a), reward function R(s,a)

Goal: Find policy π: S → A that maximizes expected cumulative discounted reward

$$
J(\pi) = E\left[\sum_{t=0}^\infty \gamma^t r_t\right]
$$

Where γ ∈ [0,1) discounts future rewards.

Value function (expected future reward from state s):

$$
V^\pi(s) = E\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, \pi\right]
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define three tasks**

You have a dataset of houses with size (sq ft), bedrooms, price.

#### **Step 2: Supervised task**

Predict price from size and bedrooms. Labeled data: each house has known price.

Goal: Learn f(size, bedrooms) ≈ price

Application: Estimate value of a new house

#### **Step 3: Unsupervised task**

Given only size and bedrooms (no price). Find clusters of similar houses.

Goal: Discover natural groupings (e.g., "small downtown condos," "suburban family homes," "luxury estates")

Application: Market segmentation

#### **Step 4: Reinforcement task**

A robot vacuums a house. Actions: move forward, turn left, turn right. Reward: +1 for cleaning dirt, -0.01 for battery usage, -10 for getting stuck.

Goal: Learn policy mapping sensor readings to actions that maximizes total reward over time.

Application: Autonomous vacuum cleaner

#### **Step 5: Compare data requirements**

Supervised: needs many labeled examples (expensive labels)
Unsupervised: needs only unlabeled data (cheap, abundant)
Reinforcement: needs only reward signal (no explicit labels, but requires interaction)

#### **Step 6: Compare feedback loops**

Supervised: immediate, correct output provided
Unsupervised: no explicit feedback, structure is intrinsic
Reinforcement: delayed, sparse, trial-and-error

---

### **7. How this appears inside neural networks or LLMs**

- **Supervised in LLMs:** Next-token prediction is supervised learning (the next token is the label). Fine-tuning on instruction-response pairs is supervised.

- **Unsupervised in LLMs:** Pre-training on raw text without labels learns grammar, facts, and reasoning through self-supervision (predict masked word—a form of unsupervised with automatic labels).

- **Reinforcement in LLMs:** RLHF (Reinforcement Learning from Human Feedback) trains LLMs to produce preferred responses using reward models learned from human rankings.

- **Self-supervised learning:** A hybrid. Labels come from data itself (e.g., predict missing word). This dominates modern LLM pre-training.

- **Semi-supervised learning:** Combines small labeled dataset with large unlabeled dataset. Used when labeling is expensive.

- **Multi-task learning:** A single model learns multiple supervised tasks simultaneously (e.g., translate, summarize, answer questions).

- **Meta-learning (learning to learn):** The model learns how to adapt quickly to new tasks with few examples.

---

### **8. Brain-like connection (multiple learning systems)**

The brain uses all three paradigms in different circuits:

- **Supervised**: The cerebellum learns motor commands from error signals (climbing fibers provide "labels" for movement correction).

- **Unsupervised**: The visual cortex develops edge detectors and object selectivity from statistical regularities in natural images, without explicit labels.

- **Reinforcement**: The basal ganglia learn action policies from dopamine reward signals. When you get a unexpected reward, dopamine fires, reinforcing preceding actions.

These systems operate in parallel. When you learn to catch a ball, supervised systems adjust muscle commands, unsupervised systems track ball trajectory patterns, and reinforcement systems learn which strategies lead to successful catches.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Unsupervised learning is just supervised learning without labels. Reinforcement learning is just supervised learning with delayed rewards."

_Why it is wrong:_ The differences are fundamental, not superficial. In supervised learning, the loss function provides immediate, per-example gradients. In reinforcement learning, you have a credit assignment problem: which past actions caused the eventual reward? This requires different algorithms (policy gradients, Q-learning, TD learning). In unsupervised learning, there is no error signal at all—only structure to discover. Reducing all learning to "just prediction with different data" misses the algorithmic innovations required for each paradigm (e.g., exploration in RL, clustering in unsupervised, backpropagation in supervised).

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Before you write a line of code, ask: Do I have labels?  |
|  Do I have only rewards? Do I have neither? The answer    |
|  determines your algorithm, your data requirements, and   |
|  your evaluation strategy. Supervised excels when labels  |
|  are cheap. Unsupervised reveals hidden structure.        |
|  Reinforcement handles sequential decisions. Choose       |
|  wrong and your system cannot learn. Choose right and     |
|  you harness the right kind of intelligence for the task. |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

For each scenario, identify the learning paradigm:

1. A system learns to play Go by playing millions of games against itself, receiving only win/loss at the end.

2. A system learns to classify emails as spam or not spam using a dataset where each email is manually labeled.

3. A system analyzes shopping cart data to find groups of customers who buy similar products, without any labels.

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

1. Reinforcement learning (trial and error, delayed reward, sequential decisions)

2. Supervised learning (labeled input-output pairs, independent examples, immediate correct answer provided)

3. Unsupervised learning (no labels, discovering hidden structure through clustering)
