# Overfitting and underfitting

## **DOMAIN: MACHINE LEARNING FUNDAMENTALS | Sub domain: ML Basics**

---

### **1. Why this concept matters for building intelligent systems**

A model that fails to learn the pattern in training data is useless. A model that learns the training data perfectly but fails on new data is equally useless. Between these extremes lies the sweet spot of generalization. Overfitting and underfitting are the two failure modes of every learning system. Recognizing them—and knowing how to balance between them—is the central practical challenge of machine learning. No matter how sophisticated your neural network, if it overfits or underfits, it does not generalize. Understanding these concepts means understanding the fundamental tension in all learning: fidelity to data versus flexibility for the unknown.

---

### **2. Core idea**

**Underfitting occurs when a model is too simple to capture the underlying pattern in the data; overfitting occurs when a model is too complex and learns noise instead of the signal, failing to generalize to new data.**

---

### **3. Concrete analogy**

Imagine two students preparing for a geography exam with practice maps.

- **Underfitting student**: Only memorizes "countries are shapes on paper." Cannot identify any specific country on the practice map. Fails practice questions. Clearly not studying enough.

- **Overfitting student**: Memorizes every single practice map exactly, including coffee stains, crease marks, and random scratches. Aces the practice test by recognizing stains, not countries. On the real exam (clean maps with no stains), the student fails completely. Memorized noise, not signal.

- **Just-right student**: Studies the patterns—coastlines, borders, relative positions. Learns what makes a country recognizable. Does well on practice and real exams.

Your machine learning model is a student. The training data is practice problems. The test data is the final exam. The goal is not perfect practice scores—it is real-world performance.

---

### **4. ASCII diagram**

```text
Underfitting vs Overfitting visualized:

    Underfitting (High Bias)         Just Right             Overfitting (High Variance)

    y ↑                             y ↑                         y ↑
      │      x x                      │                          │
      │    x   x                      │    x x                   │  x x x
      │  x     x                      │  x     x                 │ x     x
      │ x       x                     │ x       x                │x       x
      │────────────────→ x            │──────────────→ x         │──────────────→ x
         Simple curve                    Appropriate curve          Wiggly curve
         misses pattern                  captures pattern           captures noise

    Training error: HIGH              Training error: LOW        Training error: VERY LOW
    Test error: HIGH                  Test error: LOW            Test error: HIGH


    Learning curves over time (epochs):

    Error ↑
          │                    ╲
          │                     ╲ Train error
          │                      ╲
          │                       ╲    ┌─────┐
          │                        ╲   │     │
          │                         ╲  │     │
          │                          ╲ │     │ Test error
          │                           ╲│     │
          │                            └─────┘
          └────────────────────────────────────→ Model complexity / Time
                Underfit    Just Right    Overfit

    Gap between train and test error indicates overfitting.
```

---

### **5. Mathematical formulation**

**Bias-variance decomposition of expected prediction error:**

For a model f̂(x) trained on dataset D, predicting target y = f(x) + ε (with E[ε]=0, Var(ε)=σ²):

$$
E[(y - \hat{f}(x))^2] = \underbrace{(E[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{E[(E[\hat{f}(x)] - \hat{f}(x))^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible error}}
$$

- **Bias**: Error from wrong assumptions (underfitting). High bias = model too simple.
- **Variance**: Error from sensitivity to training data (overfitting). High variance = model too complex.
- **Irreducible**: Noise inherent in data (cannot be reduced).

**Underfitting indicators:**

Training error HIGH, Test error HIGH, Train ≈ Test

**Overfitting indicators:**

Training error LOW, Test error HIGH, Train << Test

**Regularization (reduces overfitting):**

Add penalty term to loss function to constrain model complexity:

$$
\min_{\theta} \frac{1}{n} \sum_{i=1}^n L(y_i, f_\theta(x_i)) + \lambda \|\theta\|_p^p
$$

Where λ controls regularization strength (λ=0 = no regularization, λ→∞ = model collapses to constant).

**Model capacity (complexity) continuum:**

Low capacity → High capacity (increasing model complexity)

Linear → Polynomial → Decision Tree → Neural Network → Ensemble

(Simplest) ....................................... (Most complex)

Underfit region ← Optimal → Overfit region

---

### **6. Worked example (step-by-step)**

#### **Step 1: Generate simple data**

True relationship: y = 2x + 1 + noise, where x in [0,10]

Training set: 10 points with noise
Test set: 50 clean points (no noise, for evaluation)

#### **Step 2: Try a linear model (underfitting candidate)**

Model: y = w₁x + w₀

Training error (MSE): 3.2
Test error: 3.5

Both errors high and similar. Model too simple to capture any curvature (though true relationship is linear, noise causes underfit perception? Actually linear is correct here—this example needs different true function.)

**Let's use true relationship: y = 0.5x³ - 3x² + 2x + 10 + noise**

#### **Step 3: Model A (degree 1 polynomial) — UNDERFITTING**

y = w₁x + w₀

Cannot capture cubic shape. Training error: 45.2, Test error: 47.8
High bias. Both errors high.

#### **Step 4: Model B (degree 3 polynomial) — JUST RIGHT**

y = w₃x³ + w₂x² + w₁x + w₀

Matches true cubic shape. Training error: 2.1, Test error: 2.3
Low errors, train ≈ test. Generalizes well.

#### **Step 5: Model C (degree 15 polynomial) — OVERFITTING**

y = w₁₅x¹⁵ + ... + w₀

Enough parameters to fit every training point exactly (error 0.0). But test error: 89.4 (much higher than training). The model learned noise in training data.

#### **Step 6: Compare**

| Model     | Training Error | Test Error | Verdict                 |
| --------- | -------------- | ---------- | ----------------------- |
| Degree 1  | 45.2           | 47.8       | Underfit (high bias)    |
| Degree 3  | 2.1            | 2.3        | Good fit                |
| Degree 15 | 0.0            | 89.4       | Overfit (high variance) |

---

### **7. How this appears inside neural networks or LLMs**

- **Early stopping:** Monitor validation loss during training. When validation loss increases while training loss decreases, you are overfitting. Stop training at that point.

- **Dropout:** Randomly drops neurons during training (typically 10-50%). Forces network to learn redundant representations, reducing overfitting.

- **Weight regularization (L1/L2):** Penalizes large weights. L2 (ridge) shrinks weights toward zero; L1 (lasso) creates sparse weights. Both reduce overfitting.

- **Data augmentation:** Creates modified copies of training data (rotate images, add noise, paraphrase text). Effectively increases dataset size, reducing overfitting.

- **Model size:** LLMs with billions of parameters can overfit if trained on small datasets. Pre-training on massive data (trillions of tokens) matches model capacity to data size.

- **Transfer learning:** Fine-tuning a large pre-trained model on a small dataset can cause overfitting. Use lower learning rates and regularization.

- **Ensemble methods:** Average predictions of multiple models reduces variance (overfitting) while preserving bias.

- **Batch normalization:** Adds noise during training (by using batch statistics) which acts as regularization, reducing overfitting.

---

### **8. Brain-like connection (memorization vs generalization)**

The brain also faces the overfitting-underfitting tradeoff. Memorizing every detail of every past experience is maladaptive—you would be paralyzed by irrelevant specifics. But too much compression loses important patterns. Sleep plays a role in generalization: during slow-wave sleep, the hippocampus replays memories while the neocortex extracts general patterns, effectively "regularizing" memories to prevent overfitting to specific episodes. Patients with highly detailed (photographic) memory often struggle with abstraction—their brains overfit. Conversely, patients with hippocampal damage cannot form specific memories but retain general knowledge—their brains underfit. The healthy brain balances both.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Low training error means my model is good. If training error is high, I need a bigger model."

_Why it is wrong:_ Low training error is necessary but not sufficient. A model with zero training error could be overfitting badly (like the degree-15 polynomial). High training error could indicate underfitting, but could also indicate data noise or a bug. The only reliable diagnostic is the gap between training and validation/test error. Low train + low validation = good. Low train + high validation = overfitting. High train + high validation = underfitting. Always look at both. A bigger model will fix underfitting but will make overfitting worse.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Every learning system walks a knife edge. Too simple    |
|  and it misses the pattern. Too complex and it chases    |
|  noise. The gap between training and test performance    |
|  is your warning light. When it grows, you are           |
|  memorizing, not learning. When it is zero but both      |
|  are high, you are missing something fundamental.        |
|  Balance is not a luxury—it is the entire game.          |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You train three neural networks on the same dataset:

- Model X: Train accuracy 98%, Validation accuracy 97%
- Model Y: Train accuracy 100%, Validation accuracy 85%
- Model Z: Train accuracy 75%, Validation accuracy 74%

**Question:** Which model is underfitting, which is overfitting, and which is well-fitted?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

- Model X: Well-fitted (high train, high validation, small gap)
- Model Y: Overfitting (perfect train, much lower validation → large gap)
- Model Z: Underfitting (low train, low validation, small gap → both poor, model too simple)
