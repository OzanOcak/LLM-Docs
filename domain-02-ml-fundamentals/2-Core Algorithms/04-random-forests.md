# Random forests

**DOMAIN: MACHINE LEARNING FUNDAMENTALS | Sub domain: Core Algorithms**

---

**1. Why this concept matters for building intelligent systems**

A single decision tree is interpretable but fragile. Change a few data points and the tree changes completely. It overfits easily, creating deep, brittle rules that fail on new data. Random forests solve this by growing many trees—each slightly different—and averaging their predictions. This is ensemble learning: wisdom of the crowds applied to machine learning. The random forest is one of the most successful off-the-shelf algorithms: it handles mixed data types, resists overfitting, provides feature importance, and often outperforms neural networks on tabular data. Understanding random forests means understanding how combining many weak models creates a strong one.

---

**2. Core idea**

**A random forest builds many decision trees on bootstrapped copies of the training data, randomly selecting a subset of features at each split, and aggregates their predictions by majority vote (classification) or average (regression).**

---

**3. Concrete analogy**

Imagine you are a doctor diagnosing a rare disease. Instead of trusting one specialist, you consult 100 specialists. Each specialist:

- Sees a slightly different set of patient records (some records are repeated, some omitted).
- Only considers a random subset of symptoms (e.g., one specialist looks only at fever and cough; another looks only at blood pressure and age).
- Builds their own diagnostic tree based on their limited view.

No single specialist is perfect—each has blind spots. But when you ask all 100 for a diagnosis, the majority vote is remarkably accurate. The specialists' individual errors average out, and their collective wisdom exceeds any single expert.

This is the random forest. Each tree is a "specialist" with limited data and limited features. The forest is the crowd. The majority vote is the diagnosis.

---

**4. ASCII diagram**

```
Random forest architecture:

    Training Data (n examples, d features)
              │
              ├─── Bootstrap sample 1 (n with replacement)
              │         │
              │         └─── Build tree with random feature subset (√d features)
              │                    │
              ├─── Bootstrap sample 2 ──→ Tree 2
              │
              ├─── Bootstrap sample 3 ──→ Tree 3
              │
              └─── ... ──→ Tree K (typically 100-500 trees)

    Prediction for new example:

              Tree 1  ──→ Class A
              Tree 2  ──→ Class A
              Tree 3  ──→ Class B
              Tree 4  ──→ Class A
                ...
              Tree K  ──→ Class A

    Majority vote: Class A wins → Final prediction

    Bootstrap sampling (bagging):

    Original data:   [●, ○, ■, □, ▲, ▼]

    Sample 1:        [●, ●, ■, □, □, ▼]  (● appears twice, ▲ missing)
    Sample 2:        [○, ■, □, ▲, ▼, ▼]  (■ appears once, ● missing)
    Sample 3:        [●, ○, ○, ■, □, ▲]  (○ appears twice)

    Each tree sees about 63% of original data (rest are out-of-bag samples).
```

---

**5. Mathematical formulation**

**Bootstrap aggregating (Bagging):**

For b = 1 to B (number of trees):

1. Draw bootstrap sample D_b of size n (sample with replacement from original data)
2. Train tree T_b on D_b

**Feature randomization:** At each node, consider only mtry randomly selected features (instead of all d features)

Typical mtry:

- Classification: mtry = √d
- Regression: mtry = d/3

**Out-of-bag (OOB) error:** For each example i, predict using only trees where i was NOT in the bootstrap sample (about 37% of trees). OOB error is an unbiased estimate of test error without a separate validation set.

**Prediction for new x:**

Classification (majority vote):

$$
\hat{y} = \text{mode}\{T_1(x), T_2(x), ..., T_B(x)\}
$$

Regression (average):

$$
\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(x)
$$

**Expected error reduction:**

If trees are independent, ensemble variance = (1/B) × individual variance

In practice, trees are correlated, but variance still reduces significantly.

**Feature importance (permutation importance):**

For each feature j:

1. Permute feature j randomly in OOB data (break association with target)
2. Measure increase in OOB error
3. Larger increase = more important feature

**Gini importance (mean decrease impurity):**

Sum over all trees of the total decrease in Gini impurity from splits on feature j, weighted by node size.

---

**6. Worked example (step-by-step)**

**Step 1: Define small dataset**

Predict whether to go out (Yes/No) based on weather:

| Temp | Humidity | Wind   | GoOut |
| ---- | -------- | ------ | ----- |
| 75   | High     | Weak   | Yes   |
| 80   | High     | Weak   | Yes   |
| 85   | High     | Strong | No    |
| 70   | Normal   | Weak   | Yes   |
| 65   | Normal   | Strong | No    |
| 72   | High     | Strong | No    |
| 78   | Normal   | Weak   | Yes   |
| 82   | Normal   | Strong | No    |

n=8 examples, d=3 features

**Step 2: Bootstrap sample 1 (n=8, with replacement)**

Random sample: [75,H,W,Y], [85,H,S,N], [70,N,W,Y], [72,H,S,N], [78,N,W,Y], [82,N,S,N], [75,H,W,Y], [65,N,S,N]

Tree built on this sample using mtry = √3 ≈ 2 features per split.

**Step 3: Bootstrap sample 2**

Different sample: [80,H,W,Y], [85,H,S,N], [78,N,W,Y], [72,H,S,N], [82,N,S,N], [65,N,S,N], [70,N,W,Y], [80,H,W,Y]

Build second tree on this sample.

**Step 4: Continue for B=5 trees**

Each tree sees different data and different feature subsets.

**Step 5: Predict new example**

New: Temp=77, Humidity=Normal, Wind=Weak

Tree 1: Yes
Tree 2: Yes
Tree 3: No
Tree 4: Yes
Tree 5: Yes

Majority: 4 Yes, 1 No → Predict Yes

**Step 6: OOB error estimate**

Example 1 (75,H,W,Y): Which trees did NOT see this example? Suppose trees 2 and 5 did not. Their predictions: Tree 2=Yes, Tree5=Yes → OOB predicts Yes (correct). Track OOB accuracy across all examples.

---

**7. How this appears inside neural networks or LLMs**

- **Feature importance analysis:** Before feeding features into an LLM, random forests identify which features actually matter, reducing input dimensionality.

- **Interpretability layer:** Random forests trained on LLM embeddings produce explainable rules for what the LLM learned.

- **Anomaly detection:** Isolation Forest (random forest variant) isolates anomalies by randomly splitting features—anomalies require fewer splits.

- **Missing value handling:** Random forests handle missing data naturally. Used to impute missing values before LLM processing.

- **Tabular LLMs:** Recent models like TabPFN compete with random forests; random forests remain the baseline to beat.

- **Ensemble diversity:** The principle of random forests—diversity through randomization and averaging—inspired ensemble methods in deep learning (dropout, stochastic depth).

- **Active learning:** Random forests provide uncertainty estimates (disagreement among trees) used to select which examples to label next.

---

**8. Brain-like connection (ensemble coding)**

The brain uses ensemble coding constantly. Orientation tuning in visual cortex: each neuron prefers a specific orientation, but the population's average activity encodes orientation more precisely than any single neuron. Decision-making: multiple brain regions (prefrontal cortex, parietal cortex, basal ganglia) each produce their own "vote," integrated to produce a final action. The random forest principle—diversity + averaging—appears in the brain's redundant, distributed representations. A single neuron can die without impairing function; the population compensates. This robustness mirrors the random forest's resistance to overfitting.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "More trees are always better. I should use as many as my computer can handle."

_Why it is wrong:_ Adding more trees reduces variance, but the benefit diminishes quickly. After about 100-200 trees, the marginal improvement is negligible. More importantly, beyond a point, more trees do not increase overfitting—random forests do not overfit with more trees (they asymptote). But they do waste compute. The law of diminishing returns applies strongly. Start with 100 trees, measure OOB error. If still decreasing, add more, but 500 is almost never needed. Memory and prediction time also scale linearly with number of trees.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  One tree is brittle. Many trees are robust. Random       |
|  forests turn a simple algorithm into a powerful one by   |
|  adding randomness and averaging. No scaling, no          |
|  normalization, no feature engineering—just data in,      |
|  predictions out. For spreadsheets and databases,         |
|  random forests are often the best answer. They teach     |
|  a universal lesson: collect diverse opinions, each       |
|  based on limited evidence, and trust the majority.       |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

You have a dataset with 1000 examples and 20 features. You train a random forest with 100 trees.

**Question:** For each tree, approximately how many examples are in its bootstrap sample? Approximately how many examples are out-of-bag (not seen by that tree)? What is mtry for classification?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Bootstrap sample size = 1000 (same as original, sampled with replacement). About 63.2% of original examples appear at least once (≈632 examples). Out-of-bag = 1000 - 632 = about 368 examples (37% not seen).

For classification, mtry = √d = √20 ≈ 4.47, so each split considers 4 randomly selected features. For regression, mtry = d/3 ≈ 6.67, typically rounded to 6 or 7.
