# Handling Imbalanced Data

## The Airport Security Analogy

Imagine you're training airport security screeners. On a typical day, 99.9% of passengers are harmless, and 0.1% are threats. If you train screeners on this real-world data, they'll learn that "everyone is harmless" is right 99.9% of the time—and they'll miss every single threat. That's the imbalanced data problem: when one class is much rarer than others, models learn to be lazy and always predict the majority class, achieving high accuracy but failing at the actual task.

In machine learning, imbalanced data is everywhere—fraud detection, disease diagnosis, rare event prediction. A model that's 99% accurate but misses all the fraud cases is useless. Handling imbalance is about forcing the model to pay attention to the rare but important cases.

---

## What Is Imbalanced Data?

### The Core Problem

```python

def imbalance_intro():
    """
    Understanding class imbalance
    """
    print("Imbalanced Data: When Classes Aren't Equal")
    print("=" * 60)

    examples = {
        "Fraud Detection": {"normal": 9990, "fraud": 10},
        "Disease Diagnosis": {"healthy": 9500, "sick": 500},
        "Customer Churn": {"stayed": 9200, "left": 800},
        "Click-through": {"no click": 9900, "click": 100}
    }

    for task, counts in examples.items():
        total = sum(counts.values())
        print(f"\n{task}:")
        for label, count in counts.items():
            print(f"  {label}: {count} ({count/total*100:.1f}%)")

imbalance_intro()
```

### Why Models Fail on Imbalanced Data

```python

def why_models_fail():
    """
    Why standard models struggle with imbalance
    """
    print("Why Models Fail on Imbalanced Data")
    print("=" * 60)

    print("""
    A standard model optimizing for accuracy:

    Dataset: 9900 normal, 100 fraud

    Strategy A: Predict all as normal
    • Accuracy = 9900/10000 = 99% ✓ (looks great!)
    • Recall for fraud = 0% ✗ (useless)

    Strategy B: Actually try to find fraud
    • Might get 80% accuracy on fraud
    • But misclassify some normal
    • Overall accuracy might drop to 98.5%
    • Model "thinks" it's doing worse!

    The model is PUNISHED for trying to find rare cases!
    """)

why_models_fail()
```

---

## Detecting Imbalance

### Visualizing Class Distribution

```python

def detect_imbalance():
    """
    How to detect class imbalance
    """
    print("Detecting Class Imbalance")
    print("=" * 60)

    from collections import Counter

    # Simulated labels
    labels = [0] * 9900 + [1] * 100

    counts = Counter(labels)
    total = len(labels)

    print("Class distribution:")
    for cls, count in sorted(counts.items()):
        percentage = count / total * 100
        bar = "█" * int(percentage * 2)
        print(f"  Class {cls}: {count:5d} ({percentage:5.2f}%) {bar}")

    imbalance_ratio = counts[0] / counts[1]
    print(f"\nImbalance ratio: 1 : {imbalance_ratio:.1f}")
    print("(more than 10:1 is considered imbalanced)")

detect_imbalance()
```

---

## Technique 1: Resampling

### Random Oversampling

Duplicate examples from minority class.

```python

def oversampling():
    """
    Random oversampling
    """
    print("Technique 1: Random Oversampling")
    print("=" * 60)

    import random

    # Original imbalanced dataset
    majority = [f"normal_{i}" for i in range(100)]
    minority = [f"fraud_{i}" for i in range(5)]

    print(f"Original: Majority={len(majority)}, Minority={len(minority)}")

    # Oversample minority
    target_size = len(majority)
    oversampled_minority = random.choices(minority, k=target_size)

    print(f"After oversampling: Minority={len(oversampled_minority)}")
    print(f"  (duplicated {target_size//len(minority)}x each example)")

    print("\n✅ Balances classes")
    print("❌ Risk of overfitting (same examples repeated)")

oversampling()
```

### Random Undersampling

Remove examples from majority class.

```python

def undersampling():
    """
    Random undersampling
    """
    print("Technique 2: Random Undersampling")
    print("=" * 60)

    import random

    # Original imbalanced dataset
    majority = [f"normal_{i}" for i in range(100)]
    minority = [f"fraud_{i}" for i in range(5)]

    print(f"Original: Majority={len(majority)}, Minority={len(minority)}")

    # Undersample majority
    target_size = len(minority)
    undersampled_majority = random.sample(majority, target_size)

    print(f"After undersampling: Majority={len(undersampled_majority)}")
    print(f"  (kept only {target_size} of 100)")

    print("\n✅ Balances classes, less overfitting risk")
    print("❌ Loses potentially useful data")

undersampling()
```

### SMOTE (Synthetic Minority Oversampling)

Create synthetic examples, not just duplicates.

```python

def smote():
    """
    SMOTE: Synthetic Minority Oversampling
    """
    print("Technique 3: SMOTE (Synthetic Minority Oversampling)")
    print("=" * 60)

    print("""
    SMOTE creates NEW synthetic examples:

    1. Pick a minority example
    2. Find its k nearest neighbors (also minority)
    3. Pick one neighbor randomly
    4. Create new point along the line between them

    Visual:

    Original minority:    ●        ●

    Synthetic examples:   ●   ○   ●   ○   ●

    Creates realistic variations instead of duplicates!
    """)

    print("✅ Creates new data, reduces overfitting")
    print("❌ Can create unrealistic examples if data is sparse")

smote()
```

---

## Technique 2: Class Weights

### Penalizing Mistakes on Minority Class

```python

def class_weights():
    """
    Using class weights to handle imbalance
    """
    print("Technique 4: Class Weights")
    print("=" * 60)

    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # Original imbalanced labels
    y = np.array([0]*9900 + [1]*100)

    # Compute balanced weights
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))

    print(f"Class distribution: 0: {np.sum(y==0)}, 1: {np.sum(y==1)}")
    print(f"\nComputed weights:")
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls}: {weight:.3f}")

    print("\nModel now pays {weight:.1f}x more attention to minority class")
    print("during training, effectively balancing the loss.")

class_weights()
```

### How It Works Mathematically

```python

def weight_math():
    """
    The math behind class weights
    """
    print("The Math of Class Weights")
    print("=" * 60)

    print("""
    Standard loss:         Loss = -log(p_pred)

    Weighted loss:         Loss = -weight × log(p_pred)

    Where weight = total_samples / (n_classes × n_samples_in_class)

    Example with 9900 normal, 100 fraud:

    weight_normal = 10000 / (2 × 9900) = 0.505
    weight_fraud = 10000 / (2 × 100) = 50

    Fraud mistakes are penalized 100× more!
    """)

weight_math()
```

---

## Technique 3: Evaluation Metrics That Matter

### Accuracy Is Misleading

```python

def accuracy_problem():
    """
    Why accuracy is misleading with imbalance
    """
    print("Why Accuracy Lies with Imbalanced Data")
    print("=" * 60)

    # Confusion matrix for fraud detector that always says "normal"
    tn = 9900  # true negatives (correctly identified normal)
    fp = 0     # false positives (normal misclassified as fraud)
    fn = 100   # false negatives (fraud missed)
    tp = 0     # true positives (fraud caught)

    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Normal  Fraud")
    print(f"Actual Normal  {tn:4d}    {fp:4d}")
    print(f"       Fraud   {fn:4d}    {tp:4d}")

    print(f"\nAccuracy:  {accuracy:.2%}  ← Looks great!")
    print(f"Precision: {precision:.2%}  ← But can't find any fraud")
    print(f"Recall:    {recall:.2%}  ← Missed all fraud!")
    print(f"F1-score:  {f1:.2%}  ← Reveals the truth")

accuracy_problem()
```

### Better Metrics for Imbalanced Data

```python

def better_metrics():
    """
    Metrics that work with imbalance
    """
    print("Metrics That Handle Imbalance")
    print("=" * 60)

    metrics = {
        "Precision": "Of all positive predictions, how many were correct?",
        "Recall": "Of all actual positives, how many did we catch?",
        "F1-score": "Harmonic mean of precision and recall",
        "AUC-ROC": "Performance across all thresholds",
        "AUC-PR": "Precision-recall curve (better for imbalance)",
        "Cohen's Kappa": "Accuracy compared to random chance"
    }

    for metric, desc in metrics.items():
        print(f"  • {metric}: {desc}")

better_metrics()
```

---

## Code: Complete Imbalanced Data Pipeline

```python

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def imbalance_pipeline():
    """
    Complete imbalanced data handling pipeline
    """
    print("Complete Imbalanced Data Pipeline")
    print("=" * 60)

    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.95, 0.05],  # 95% class 0, 5% class 1
        random_state=42
    )

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain distribution: {np.bincount(y_train)}")
    print(f"Test distribution: {np.bincount(y_test)}")

    # Option 1: Class weights
    print("\n" + "=" * 40)
    print("OPTION 1: Class Weights")
    print("=" * 40)

    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    model_weighted = RandomForestClassifier(class_weight=class_weight, random_state=42)
    model_weighted.fit(X_train, y_train)
    y_pred_weighted = model_weighted.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred_weighted, digits=3))

    # Option 2: SMOTE
    print("\n" + "=" * 40)
    print("OPTION 2: SMOTE")
    print("=" * 40)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE - Train distribution: {np.bincount(y_train_smote)}")

    model_smote = RandomForestClassifier(random_state=42)
    model_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = model_smote.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_smote, digits=3))

imbalance_pipeline()
```

---

## Technique Comparison

| Technique         | How It Works               | Pros                      | Cons               | Best For          |
| ----------------- | -------------------------- | ------------------------- | ------------------ | ----------------- |
| Oversampling      | Duplicate minority         | Simple, keeps all data    | Overfitting        | Small datasets    |
| Undersampling     | Remove majority            | Less overfitting          | Loses data         | Large datasets    |
| SMOTE             | Create synthetic           | New data, less overfit    | Can create noise   | Medium data       |
| Class weights     | Weight loss                | No data changes           | May not be enough  | Any size          |
| Anomaly detection | Treat as outlier detection | Handles extreme imbalance | Different paradigm | Extreme (1:1000+) |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Imbalanced data isn't a niche problem—it's everywhere:     ║
║                                                              ║
║  💳 FRAUD DETECTION: 99.9% normal, 0.1% fraud.              ║
║     Missing fraud costs millions, annoying customers costs   ║
║     reputation. Balance is critical.                         ║
║                                                              ║
║  🏥 MEDICAL DIAGNOSIS: Rare diseases are, by definition,     ║
║     rare. A model that misses them is dangerous.             ║
║                                                              ║
║  🛒 CUSTOMER CHURN: 5-10% churn rate is typical.             ║
║     Identifying who will leave lets you intervene.           ║
║                                                              ║
║  🌍 DISASTER PREDICTION: Earthquakes, floods are rare.       ║
║     False alarms are bad, but missed events are catastrophic.║
║                                                              ║
║  🤖 LLM SAFETY: Harmful content is rare in training          ║
║     data, but we need models to detect it reliably.          ║
║                                                              ║
║  Ignoring imbalance means your model optimizes for           ║
║  the wrong thing—being "mostly right" instead of            ║
║  being right when it matters most.                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Imbalanced data occurs when classes are unevenly distributed—like 99.9% normal transactions vs 0.1% fraud, causing models to be lazy and always predict the majority class

• Three main approaches to handle imbalance: resampling (oversample minority or undersample majority), class weights (penalize mistakes on minority class more), and better metrics (precision, recall, F1 instead of accuracy)

• The right technique depends on your data and problem—SMOTE for creating synthetic data, class weights for simplicity, or treating it as anomaly detection for extreme imbalance

---

## Mental Hook

> "Handling imbalanced data is like training a security guard who sees 10,000 friendly faces and 1 intruder. If you just reward them for being right, they'll let everyone in. You have to make catching that one intruder worth more than 10,000 correct welcomes."
