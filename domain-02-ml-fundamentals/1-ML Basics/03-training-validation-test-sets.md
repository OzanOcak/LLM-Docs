# Training vs validation vs test sets

**DOMAIN: MACHINE LEARNING FUNDAMENTALS | Sub domain: ML Basics**

---

**1. Why this concept matters for building intelligent systems**

You cannot judge a student by the questions they have already practiced. They might have memorized the answers without understanding. The same is true for machine learning models. If you evaluate a model on the same data used to train it, you have no idea whether it learned general patterns or simply memorized examples. The solution is to split your data into three distinct sets: one to teach (training), one to tune (validation), and one to truthfully evaluate (test). This separation is the difference between a model that works in the real world and one that fails the moment it sees new data.

---

**2. Core idea**

**The training set is used to learn model parameters; the validation set is used to tune hyperparameters and select models; the test set is used only once to evaluate final performance on unseen data.**

---

**3. Concrete analogy**

Imagine you are a medical student preparing for a licensing exam. You have a bank of 1000 practice questions.

- **Training set (700 questions)**: You study these repeatedly, learn the concepts, memorize facts. This is where learning happens.

- **Validation set (150 questions)**: After each study session, you test yourself on these questions. They help you decide whether to study more, change your approach, or move on. You might adjust your study strategy based on your validation performance.

- **Test set (150 questions)**: On exam day, you see entirely new questions. Your score here is the only honest measure of your true knowledge. If you did well on training and validation but poorly on test, you were memorizing, not learning.

Crucially, you must never look at the test questions before exam day. The moment you see them, they become contaminated. The test set is used exactly once, at the very end.

---

**4. ASCII diagram**

```
Data splitting for supervised learning:

    Complete Dataset (100%)
    ┌────────────────────────────────────────────────────┐
    │                                                    │
    │  Training Set     Validation Set    Test Set       │
    │  (60-80%)          (10-20%)          (10-20%)      │
    │  ┌────────────┐    ┌────────┐        ┌────────┐    │
    │  │            │    │        │        │        │    │
    │  │    Used    │    │  Used  │        │ Used   │    │
    │  │    for     │    │  for   │        │  ONCE  │    │
    │  │   weight   │    │ hyper- │        │  at    │    │
    │  │   updates  │    │parameter│       │  end   │    │
    │  │            │    │  tuning │        │        │    │
    │  └────────────┘    └────────┘        └────────┘    │
    └────────────────────────────────────────────────────┘

    The workflow:

    Training Data ──→ Train model ──→ Evaluate on Validation
         ↑                                  │
         │                                  ↓
         └──────────────────── Adjust hyperparameters
                                              │
                                              ↓
    Test Data (unseen) ──→ Final Evaluation ──→ Report performance

    Typical splits for different dataset sizes:

    Small dataset (100-1000):  60% Train, 20% Val, 20% Test
    Medium dataset (10K-100K): 70% Train, 15% Val, 15% Test
    Large dataset (1M+):       98% Train, 1% Val, 1% Test

    Cross-validation (when data is scarce):

    Fold 1: [Train] [Train] [Train] [Val]   (rotate validation fold)
    Fold 2: [Train] [Train] [Val]  [Train]
    Fold 3: [Train] [Val]  [Train] [Train]
    Fold 4: [Val]  [Train] [Train] [Train]

    Average performance across K folds.
```

---

**5. Mathematical formulation**

Let D = {(x₁, y₁), ..., (xₙ, yₙ)} be the full dataset.

Split D into three disjoint subsets:

D_train ∪ D_val ∪ D_test = D
D_train ∩ D_val = ∅, D_train ∩ D_test = ∅, D_val ∩ D_test = ∅

Typical sizes: |D_train| ≈ 0.7n, |D_val| ≈ 0.15n, |D_test| ≈ 0.15n

**Training:** Minimize empirical risk on training set

$$
\hat{\theta} = \arg\min_{\theta} \frac{1}{|D_{\text{train}}|} \sum_{(x,y) \in D_{\text{train}}} L(y, f_\theta(x))
$$

**Validation:** For each hyperparameter setting H (learning rate, network depth, regularization strength):

$$
\text{Perf}(H) = \frac{1}{|D_{\text{val}}|} \sum_{(x,y) \in D_{\text{val}}} L(y, f_{\hat{\theta}(H)}(x))
$$

Select H\* = argmax Perf(H) (or argmin for loss)

**Testing:** Final evaluation (once, no going back)

$$
\text{TestAcc} = \frac{1}{|D_{\text{test}}|} \sum_{(x,y) \in D_{\text{test}}} \mathbf{1}[y = f_{\hat{\theta}(H^*)}(x)]
$$

**K-fold cross-validation (when validation set is too small):**

Partition data into K equal folds. For each fold k:

- Train on all folds except k
- Validate on fold k

Final performance = average over K folds

$$
\text{CVScore} = \frac{1}{K} \sum_{k=1}^K \text{Score}_k
$$

---

**6. Worked example (step-by-step)**

**Step 1: Define the dataset**

100 labeled examples of handwritten digits (0-9). Want to build a classifier.

**Step 2: Split the data**

- Training: 70 examples (70%)
- Validation: 15 examples (15%)
- Test: 15 examples (15%)

**Step 3: Train models with different hyperparameters**

Model A (simple): 1 hidden layer, 50 neurons  
Training accuracy: 85%  
Validation accuracy: 82%

Model B (complex): 3 hidden layers, 200 neurons each  
Training accuracy: 100% (perfect on training)  
Validation accuracy: 78% (worse than model A!)

Model A has better validation performance (82% > 78%). The complex model memorized training data but does not generalize (overfitting).

**Step 4: Select best model based on validation**

Choose model A (1 layer, 50 neurons).

**Step 5: Final test evaluation**

Test on 15 unseen examples:

Model A test accuracy: 80%

This is the honest estimate of real-world performance.

**Step 6: What not to do**

If you had looked at test set earlier and tuned hyperparameters to improve test accuracy, your reported 80% would be invalid (overly optimistic). The test set must remain untouched until the final moment.

---

**7. How this appears inside neural networks and LLMs**

- **Early stopping:** Monitor validation loss during training. If validation loss stops decreasing (or increases) for N epochs, stop training. Prevents overfitting.

- **Hyperparameter search:** Grid search, random search, or Bayesian optimization on validation set to choose learning rate, batch size, number of layers, dropout rate, etc.

- **Model selection:** Compare different architectures (e.g., GPT-2 vs GPT-3 size, different attention mechanisms) using validation performance.

- **LLM fine-tuning:** Split fine-tuning data into train (update weights) and validation (monitor overfitting, select checkpoint). Test set held out for final evaluation.

- **Cross-validation for small datasets:** When labeled data is scarce (e.g., medical imaging with 500 samples), use k-fold cross-validation instead of a single validation split.

- **Train-test contamination in LLMs:** Major risk during pre-training. If test examples accidentally appear in training data, reported performance is fraudulent. Reputable benchmarks verify no contamination.

- **Data leakage:** Examples from validation or test sets leaking into training (due to duplicates, time series issues, or preprocessing). Catastrophic for model evaluation.

---

**8. Brain-like connection (sleep and memory consolidation)**

The brain separates learning phases. Wakefulness is training: you experience examples, update neural connections. Slow-wave sleep may act as validation: the brain replays memories, strengthens important patterns, and weakens irrelevant ones. The test set is real-world performance the next day—how well did you actually learn? Cramming before an exam (training on test-like questions) gives inflated confidence but poor generalization. Spaced learning with proper separation mirrors the train-val-test paradigm, suggesting the brain evolved similar principles to avoid overfitting to recent experiences.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "If I have a lot of data, I don't need a validation or test set. I can just train on all data and trust the training accuracy."

_Why it is wrong:_ Training accuracy is always optimistic. A model with enough capacity can achieve 100% training accuracy by memorizing (including noise). Without a validation set, you cannot detect overfitting. Without a test set, you have no unbiased estimate of real-world performance. Even with massive datasets, you need held-out data for honest evaluation. Companies that deploy models without proper validation discover failures only after release—at much higher cost.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  A model that performs perfectly on training data is      |
|  suspicious, not impressive. The test set is your only    |
|  honest mirror. Without it, you are lying to yourself     |
|  about how well your system will perform in the wild.     |
|  Training teaches. Validation guides. Test tells truth.   |
|  Never skip any of the three. Never peek at the test set. |
|  Your future users are your final test set. Prepare them  |
|  honestly or fail publicly.                               |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

You have 1000 labeled examples. You split into 700 train, 150 validation, 150 test. You train 20 different hyperparameter combinations, selecting the best based on validation accuracy. You then report that model's test accuracy as 85%.

**Question:** Is it valid to then go back and train a new model using both train AND validation (850 examples) and report its performance on the same test set? Why or why not?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** No, it is not valid. The test set has now been "seen" by you (the model developer) when you reported the first 85% accuracy. If you train a new model using train+validation and evaluate on the same test set, you are effectively tuning hyperparameters based on test performance. The test set is no longer unbiased. You would need a new, fresh test set to honestly evaluate the new model. This is why datasets are split once and test set is locked away until the final, one-time evaluation.
