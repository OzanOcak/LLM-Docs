# Logistic regression

## **DOMAIN: MACHINE LEARNING FUNDAMENTALS | Sub domain: Core Algorithms**

---

### **1. Why this concept matters for building intelligent systems**

Linear regression predicts numbers. But what about yes/no questions? Is this email spam? Will this customer buy? Does this image contain a cat? Linear regression fails because its outputs are unbounded, not probabilities between 0 and 1. Logistic regression solves this by wrapping linear regression inside the sigmoid function, transforming any real number into a probability. It is the simplest classification algorithm and the direct ancestor of the output layer in every neural network and LLM. When an LLM predicts the next token, it is doing logistic regression (via softmax, the multi-class version). Understand logistic regression and you understand classification at its core.

---

### **2. Core idea**

**Logistic regression models the probability that an input belongs to a class by passing a linear combination of features through the sigmoid function, then learns weights by maximizing the likelihood of the observed labels.**

---

### **3. Concrete analogy**

Imagine you are a doctor diagnosing a disease based on a single test score. Higher scores indicate higher risk. You have historical data: test scores and whether each patient had the disease (0 = healthy, 1 = sick).

You cannot fit a straight line to 0/1 data—the line would go above 1 and below 0. Instead, you want an S-shaped curve that starts near 0 for low scores, rises smoothly, and approaches 1 for high scores. That S-curve is the sigmoid. For a given test score, the curve tells you the probability of disease.

When a new patient arrives with score 75, you read off the curve: probability ≈ 0.8. You decide: if probability > 0.5, predict sick. Logistic regression gives you both the binary decision and the confidence behind it.

---

### **4. ASCII diagram**

```text
Logistic regression (single feature):

    P(Y=1 | x) ↑
        1.0 ┤                    ●
            │                 ●
        0.8 ┤              ●
            │           ●
        0.6 ┤        ●
            │     ●
        0.5 ┤──●────────────────  Decision threshold
            │  ●
        0.4 ┤ ●
            │●
        0.2 ┤●
            │
        0.0 ┼────────────────────→ x (test score)
           0   20   40   60   80   100

    S-shaped sigmoid curve: σ(z) = 1 / (1 + e^{-z})
    where z = w₁x + w₀ (linear regression output)

    Decision boundary (for multiple features):

         x₂ ↑
            │     ○  ○  ○
            │   ○  ○  ○
            │ ○  ○  ○  ○
            │─────●─────  (separating line)
            │  ●  ●  ●
            │ ●  ●  ●
            │●  ●  ●
            └──────────→ x₁

    Points on one side → P > 0.5 → class 1
    Points on other side → P < 0.5 → class 0
    The separating line is where z = w^T x + b = 0
```

---

### **5. Mathematical formulation**

**Sigmoid function (squashes output to [0,1]):**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

(Unicode: σ(z) = 1 / (1 + e^{-z}))

Properties: σ(0) = 0.5, σ(z) → 1 as z → ∞, σ(z) → 0 as z → -∞

**Logistic regression model (binary classification):**

First compute linear score: z = w₁x₁ + w₂x₂ + ... + w_d x_d + b = w^T x + b

Then convert to probability:

$$
P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

$$
P(y=0 | \mathbf{x}) = 1 - P(y=1 | \mathbf{x})
$$

**Decision rule:** Predict class 1 if P ≥ 0.5 (equivalently, w^T x + b ≥ 0)

**Loss function (cross-entropy / negative log-likelihood):**

For a single example with true label y ∈ {0,1}:

$$
L(\mathbf{w}, b) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

Where ŷ = P(y=1|x)

**Log-likelihood for n examples:**

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
$$

**Gradient of cross-entropy (beautifully simple):**

$$
\nabla_{\mathbf{w}} L = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i) \mathbf{x}_i
$$

$$
\nabla_{b} L = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)
$$

Compare to linear regression gradient: same form! (replace ŷ - y with linear regression's residual)

**Multi-class logistic regression (softmax):**

For K classes, probability of class k:

$$
P(y=k | \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}
$$

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define the data**

Hours studied (x) and pass/fail (y): (1,0), (2,0), (3,1), (4,1), (5,1)

#### **Step 2: Initialize parameters**

Start with w = 0, b = 0

#### **Step 3: Compute predictions for each example**

For x=1: z = 0×1 + 0 = 0, σ(0) = 0.5
x=2: 0.5
x=3: 0.5
x=4: 0.5
x=5: 0.5

All ŷ = 0.5

#### **Step 4: Compute gradient**

ŷ - y for each: (0.5-0)=0.5, (0.5-0)=0.5, (0.5-1)=-0.5, (0.5-1)=-0.5, (0.5-1)=-0.5

Sum of (ŷ-y) = 0.5+0.5-0.5-0.5-0.5 = -0.5

∇b = (-0.5)/5 = -0.1

For ∇w: Σ(ŷ-y)x = 0.5×1 + 0.5×2 + (-0.5)×3 + (-0.5)×4 + (-0.5)×5 = 0.5 + 1.0 -1.5 -2.0 -2.5 = -4.5

∇w = -4.5/5 = -0.9

#### **Step 5: Update parameters with learning rate α=0.1**

w_new = 0 - 0.1×(-0.9) = 0 + 0.09 = 0.09
b_new = 0 - 0.1×(-0.1) = 0 + 0.01 = 0.01

#### **Step 6: After many iterations (simulate convergence)**

W will become positive (more study → higher pass probability). Final model might be w≈1.5, b≈-4.

Check: For x=3, z = 1.5×3 - 4 = 4.5 - 4 = 0.5, σ(0.5) = 0.62 (62% chance of passing)
For x=1, z = 1.5 - 4 = -2.5, σ(-2.5) = 0.076 (7.6% chance)

#### **Step 7: Decision boundary**

Solve w×x + b = 0 → 1.5x - 4 = 0 → x = 4/1.5 ≈ 2.67 hours

Study less than 2.67 hours → predict fail; more → predict pass.

---

### **7. How this appears inside neural networks or LLMs**

- **Output layer for binary classification:** Last layer is logistic regression (sigmoid on linear output).

- **Multi-class output (softmax):** Generalized logistic regression. Every LLM uses softmax over its vocabulary to predict the next token.

- **Attention weights:** Softmax over attention scores is multi-class logistic regression applied to queries and keys.

- **Interpretability:** Logistic regression coefficients can be interpreted as feature importance (how much each feature increases log-odds of class membership).

- **Calibration:** LLM probabilities are often overconfident; logistic regression can be used to calibrate them (Platt scaling).

- **Linear classifiers in embedding space:** After extracting LLM embeddings, a logistic regression classifier can predict properties (sentiment, toxicity, topic) with high accuracy.

- **Click-through rate prediction:** In recommendation systems, logistic regression on user and item features predicts probability of click.

---

### **8. Brain-like connection (spiking neurons as sigmoids)**

A biological neuron's firing rate as a function of its input current follows an S-shaped curve strikingly similar to the sigmoid. For low input current, the neuron rarely fires (P≈0). As current increases, firing rate rises smoothly. At high current, the neuron saturates (P≈1). The threshold where firing rate reaches 50% is analogous to the decision boundary w^T x + b = 0. The brain performs logistic regression at the level of individual neurons. A single neuron cannot compute complex functions, but populations of neurons working together—each a logistic regression unit—create the representational power of the brain.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Logistic regression is a regression algorithm (because of the name). It predicts continuous values."

_Why it is wrong:_ Despite the name, logistic regression is a classification algorithm. The "regression" refers to the fact that it models the probability of class membership as a regression on the log-odds scale: log(P/(1-P)) = w^T x + b. The output is a probability (continuous between 0 and 1), but the decision task is classification. The name is historical and confusing. For actual regression (predicting continuous values), you use linear regression, not logistic regression.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Every time an LLM picks the next word, it is doing       |
|  multi-class logistic regression on a vocabulary of       |
|  50,000+ tokens. The sigmoid and softmax are the          |
|  bridges from linear algebra to probability. Without      |
|  logistic regression, neural networks could only produce  |
|  unbounded numbers, not meaningful probabilities.         |
|  Master logistic regression and you master the art of     |
|  turning scores into decisions with confidence.           |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

A logistic regression model predicts whether a customer will buy a product (y=1) based on age (x). The model is P(buy) = σ(0.1x - 4).

**Question:** For a 30-year-old customer, what is the predicted probability of buying? For a 50-year-old? At what age does P=0.5?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Age 30: z = 0.1×30 - 4 = 3 - 4 = -1, σ(-1) = 1/(1+e¹) = 1/(1+2.718) = 1/3.718 ≈ 0.269 (27% chance)

Age 50: z = 0.1×50 - 4 = 5 - 4 = 1, σ(1) = 1/(1+e⁻¹) = 1/(1+0.368) = 1/1.368 ≈ 0.731 (73% chance)

P=0.5 when z=0: 0.1x - 4 = 0 → 0.1x = 4 → x = 40 years old. At 40, the customer is exactly at the decision boundary.
