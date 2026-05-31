# Decision trees

## **DOMAIN: MACHINE LEARNING FUNDAMENTALS | Sub domain: Core Algorithms**

---

### **1. Why this concept matters for building intelligent systems**

Linear models draw straight lines. Logistic regression draws smooth curves. But what if the true decision boundary is a rectangle? A checkerboard? A nested set of if-then rules? Decision trees model exactly this: hierarchical, nonlinear, interpretable rules. A decision tree asks a sequence of yes/no questions about the data, each question splitting the data into purer subsets. The result is a model you can explain to a human: "If age > 30 and income > 50K and credit score > 700, then approve loan." Random forests and gradient boosting—among the most powerful practical algorithms—are just collections of decision trees. Understanding decision trees means understanding the logic of human-readable machine learning.

---

### **2. Core idea**

**A decision tree recursively partitions the feature space into regions, assigning the majority class (or average value) to each region, using questions about individual features at each split to maximize the purity of child nodes.**

---

### **3. Concrete analogy**

Imagine you are a doctor diagnosing a patient. You ask a series of questions:

- "Is fever above 100°F?" Yes → go to next question; No → probably not an infection.
- "Is cough present?" Yes → possible respiratory infection; No → possible other infection.
- "Is shortness of breath present?" Yes → possible pneumonia; No → possible bronchitis.

Each question splits the patients into groups. After enough questions, each group contains mostly patients with the same diagnosis. The set of questions forms a tree. The path from root to leaf is a diagnosis rule.

Decision trees automate this process. They find which questions (which features and which thresholds) best split the data at each step, building the tree automatically from examples.

---

### **4. ASCII diagram**

```text
Decision tree for loan approval:

                        Is income > 50K?
                         /           \
                       Yes             No
                       /                 \
                  Has credit             Is income
                  history?                > 30K?
                   /    \                /    \
                 Yes     No            Yes     No
                 /        \            /        \
            Approve    Check debt   Check      Reject
                       /      \     savings?
                    Low     High     /    \
                    /        \     Yes    No
                Approve    Reject  /        \
                               Approve    Reject

Visual as a tree structure:

                    ┌─────────────┐
                    │ Income>50K? │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              │              ▼
    ┌───────────┐          │      ┌───────────┐
    │ Has credit│          │      │ Income>30K│
    │ history?  │          │      │     ?     │
    └─────┬─────┘          │      └─────┬─────┘
          │                │            │
      ┌───┴───┐            │        ┌───┴───┐
      │       │            │        │       │
      ▼       ▼            │        ▼       ▼
   Approve  Check          │     Check    Reject
            debt           │     savings
            │              │        │
         ┌──┴──┐           │     ┌──┴──┐
         │     │           │     │     │
         ▼     ▼           │     ▼     ▼
     Approve Reject        │  Approve Reject
                           │
                    ┌──────┴──────┐
                    │ Has savings?│
                    └─────────────┘

Feature space partition (2D):

    x₂ (income)
       ↑
   100K│  A   A   A   │   A   A   A
       │  A   A   A   │   A   A   A
   50K─┼──────────────┼──────────────
       │  R   R   R   │   R   R   R
       │  R   R   R   │   R   R   R
    0K └──────────────┴──────────────→ x₁ (credit score)
           0         700
```

---

### **5. Mathematical formulation**

**Setup:** Data with features x and target y (discrete for classification, continuous for regression)

**Splitting criterion for classification (measure impurity):**

**Gini impurity:** Probability of misclassifying a randomly chosen point if labeled randomly according to node distribution

$$
G = 1 - \sum_{k=1}^K p_k^2
$$

Where p_k = proportion of class k in node

(Unicode: G = 1 - Σ p_k²)

G = 0 when node is pure (single class), maximum when classes are balanced.

**Entropy (information gain):**

$$
H = -\sum_{k=1}^K p_k \log_2(p_k)
$$

**Information gain from split:** H_parent - weighted average H_children

**Splitting criterion for regression (variance reduction):**

Variance of node: σ² = (1/n) Σ (y_i - ȳ)²

Variance reduction = σ²_parent - weighted average σ²_children

**Stopping criteria:**

- Maximum depth reached
- Minimum samples per leaf
- Pure node (single class)
- No improvement from further splits

**Prediction for leaf node:**

- Classification: majority class (mode) of training points in leaf
- Regression: average value (mean) of training points in leaf

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define small dataset**

Predict whether to play tennis (Yes/No) based on weather:

| Outlook  | Humidity | Wind   | Play |
| -------- | -------- | ------ | ---- |
| Sunny    | High     | Weak   | No   |
| Sunny    | High     | Strong | No   |
| Overcast | High     | Weak   | Yes  |
| Rainy    | Normal   | Weak   | Yes  |
| Rainy    | Normal   | Strong | No   |
| Rainy    | High     | Weak   | Yes  |
| Overcast | Normal   | Strong | Yes  |
| Sunny    | Normal   | Weak   | Yes  |
| Sunny    | Normal   | Strong | Yes  |
| Rainy    | High     | Strong | No   |

#### **Step 2: Compute root impurity**

Total: 10 examples, 5 Yes (p=0.5), 5 No (p=0.5)

Gini = 1 - (0.5² + 0.5²) = 1 - (0.25+0.25) = 0.5

#### **Step 3: Evaluate candidate splits**

Split on Outlook:

- Sunny: 3 examples (2 No, 1 Yes) → Gini = 1 - ((2/3)²+(1/3)²) = 1 - (4/9+1/9)=1-5/9=4/9≈0.444
- Overcast: 2 examples (2 Yes) → Gini = 0
- Rainy: 5 examples (3 Yes, 2 No) → Gini = 1 - ((3/5)²+(2/5)²) = 1 - (9/25+4/25)=1-13/25=12/25=0.48

Weighted Gini = (3/10)×0.444 + (2/10)×0 + (5/10)×0.48 = 0.1332 + 0 + 0.24 = 0.3732

Information gain = 0.5 - 0.3732 = 0.1268

#### **Step 4: Compare to other splits**

Similar calculation for Humidity, Wind. Outlook gives best gain.

#### **Step 5: Build tree**

Root splits on Outlook. Overcast branch pure (all Yes) → leaf.

Sunny branch (2 No,1 Yes) needs further splitting.

#### **Step 6: Split Sunny branch**

Within Sunny examples:

| Humidity | Play |
| -------- | ---- |
| High     | No   |
| High     | No   |
| Normal   | Yes  |

Split on Humidity: High → both No (pure), Normal → Yes (pure). Perfect split.

#### **Step 7: Split Rainy branch**

Within Rainy examples (3 Yes, 2 No):

Best split is Wind: Weak → 2 Yes (pure), Strong → 2 No (pure). Perfect split.

#### **Step 8: Final tree**

Outlook:

- Overcast → Yes
- Sunny → Humidity: High→No, Normal→Yes
- Rainy → Wind: Weak→Yes, Strong→No

Perfect classification on training data (100% accuracy).

---

### **7. How this appears inside neural networks or LLMs**

- **Random Forests:** Ensemble of many decision trees trained on random subsets of data and features. Reduces variance (overfitting) of individual trees.

- **Gradient Boosting (XGBoost, LightGBM, CatBoost):** Sequentially builds trees where each new tree corrects errors of previous trees. State-of-the-art for tabular data.

- **Interpretability of LLMs:** Decision trees can approximate LLM behavior locally (e.g., LIME uses a decision tree to explain individual predictions).

- **Feature importance:** Tree-based models provide natural feature importance scores (how much each feature reduces impurity). Used to analyze which features LLMs rely on.

- **Rule extraction:** Decision trees from LLM embeddings produce human-readable rules like "If sentiment score > 0.7 and topic = sports, then generate enthusiastic response."

- **Weak learners:** In boosting, shallow decision trees (stumps) serve as weak learners.

- **Decision tree ensembles in recommendation:** Many production recommendation systems use gradient-boosted decision trees (not neural networks) due to speed and interpretability.

---

### **8. Brain-like connection (hierarchical decision making)**

The brain makes decisions hierarchically. Visual processing: edges → shapes → objects → categories. Motor planning: reach → grasp → lift → place. This is a decision tree with hundreds of nodes. Each cortical column performs a binary or multi-way classification based on inputs from lower levels. Neurological damage reveals the tree structure: damage to a specific node (e.g., fusiform face area) causes highly specific deficits (can't recognize faces) while leaving other branches intact (can still recognize objects). The brain's decision tree is not learned by greedy impurity minimization, but the hierarchical, conditional structure is remarkably similar.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Decision trees are obsolete. Neural networks are always better."

_Why it is wrong:_ For tabular data (spreadsheets, databases, feature vectors), gradient-boosted decision trees often outperform neural networks. They handle mixed data types (categorical, numerical), missing values gracefully, and require minimal preprocessing (no scaling). They are faster to train, more interpretable, and use less memory. In Kaggle competitions, XGBoost and LightGBM dominate tabular tasks. Neural networks excel at unstructured data (images, text, audio). The right tool depends on the data type. Decision trees are far from obsolete.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  A decision tree is machine learning you can read aloud.  |
|  No black boxes, no inscrutable matrices. Just questions  |
|  and answers leading to a conclusion. For tabular data,   |
|  ensembles of trees remain state-of-the-art. For any      |
|  system that must explain itself to humans, decision      |
|  trees or their ensembles are the gold standard. In a     |
|  world of billion-parameter mysteries, trees are the      |
|  glass boxes that let you see the logic inside.           |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a node with 10 examples: 7 of class A, 3 of class B.

**Question:** Calculate the Gini impurity of this node. Is it pure, impure, or maximum impurity?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

p_A = 7/10 = 0.7, p_B = 3/10 = 0.3

G = 1 - (p_A² + p_B²) = 1 - (0.49 + 0.09) = 1 - 0.58 = 0.42

The node is impure (not pure, since G > 0). Maximum impurity would be 0.5 (5 of each class). This node (0.42) is moderately pure, closer to maximum than to pure (0).
