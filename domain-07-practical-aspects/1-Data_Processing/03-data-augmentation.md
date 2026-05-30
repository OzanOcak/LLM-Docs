# Data Augmentation

## The Basketball Coach Analogy

Imagine a basketball coach preparing for a big game. They only have 5 players, but they need the team to handle any situation. So they run drills where players practice against different defenses, shoot from different angles, and scrimmage in different formations. They're not getting new players—they're augmenting their practice by varying the conditions. That's data augmentation: creating new training examples by modifying existing ones in realistic ways.

In machine learning, data augmentation is a powerful technique to make models more robust and generalizable. For LLMs, this means taking existing text and creating variations—paraphrasing, back-translation, synonym replacement—to teach the model that meaning persists across different expressions. It's like giving the model more practice without collecting more data.

---

## What Is Data Augmentation?

### The Core Idea

Data augmentation creates new training examples by applying transformations to existing data, teaching the model to be invariant to unimportant variations.

```text

Original data:
    "The cat sat on the mat"

Augmented variations:
    "The cat sat on the mat"        (original)
    "The cat sat on the mat."       (punctuation added)
    "A cat sat on the mat"          (synonym)
    "The cat was sitting on the mat" (paraphrase)
    "Le chat s'est assis sur le tapis" (back-translation from French)

All have the same meaning → model learns meaning is invariant!
```

```python

def augmentation_intro():
    """
    The basic concept of data augmentation
    """
    print("Data Augmentation: More Practice Without More Data")
    print("=" * 60)

    print("""
    Why augment?

    • Limited data → create more examples
    • Reduce overfitting → see more variations
    • Improve robustness → handle edge cases
    • Balance classes → fix imbalanced datasets

    Key principle: Apply transformations that preserve the label
    (meaning, sentiment, category, etc.) while changing surface form.
    """)

augmentation_intro()
```

---

## Text Augmentation Techniques

### 1. Synonym Replacement

Replace words with their synonyms while preserving meaning.

```python

def synonym_replacement():
    """
    Replacing words with synonyms
    """
    print("Technique 1: Synonym Replacement")
    print("=" * 60)

    import random

    # Simplified synonym dictionary
    synonyms = {
        "good": ["great", "excellent", "wonderful", "fantastic"],
        "bad": ["terrible", "awful", "horrible", "poor"],
        "happy": ["joyful", "pleased", "delighted", "content"],
        "sad": ["unhappy", "miserable", "down", "gloomy"],
        "big": ["large", "huge", "enormous", "massive"],
        "small": ["tiny", "little", "miniature", "compact"]
    }

    def synonym_replace(text, p=0.3):
        words = text.split()
        new_words = []

        for word in words:
            if word.lower() in synonyms and random.random() < p:
                new_words.append(random.choice(synonyms[word.lower()]))
            else:
                new_words.append(word)

        return ' '.join(new_words)

    original = "The movie was good but the ending was bad"

    print(f"Original: {original}")

    random.seed(42)
    for i in range(3):
        augmented = synonym_replace(original)
        print(f"Augmented {i+1}: {augmented}")

synonym_replacement()
```

### 2. Back-Translation

Translate to another language and back, creating natural paraphrases.

```python

def back_translation():
    """
    Back-translation for paraphrasing
    """
    print("Technique 2: Back-Translation")
    print("=" * 60)

    print("""
    Process:

    English: "The cat sat on the mat"
        ↓ (translate to French)
    French: "Le chat s'est assis sur le tapis"
        ↓ (translate back to English)
    English: "The cat sat on the mat" (original)
    English: "The cat was sitting on the rug" (paraphrase!)

    Why it works: Different languages express the same idea
    in different ways, so back-translation creates natural
    variations without changing meaning.
    """)

    print("Example paraphrases from back-translation:")
    paraphrases = [
        "The cat sat on the mat",
        "A cat was sitting on the mat",
        "The cat sat down on the rug",
        "On the mat sat the cat"
    ]

    for p in paraphrases:
        print(f"  • {p}")

back_translation()
```

### 3. Random Insertion/Deletion

Add or remove words to create variations.

```python

def random_insertion():
    """
    Random word insertion and deletion
    """
    print("Technique 3: Random Insertion/Deletion")
    print("=" * 60)

    import random

    def random_insert(text, n=1):
        words = text.split()
        filler_words = ["very", "really", "quite", "actually", "basically"]

        for _ in range(n):
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(filler_words))

        return ' '.join(words)

    def random_delete(text, p=0.1):
        words = text.split()
        new_words = [w for w in words if random.random() > p]
        return ' '.join(new_words)

    original = "The movie was good"

    print(f"Original: {original}")

    random.seed(42)
    print(f"Insert:   {random_insert(original)}")
    print(f"Delete:   {random_delete(original)}")

    print("\n⚠️ Use carefully—can change meaning!")
    print("Better for sentiment/classification tasks where")

random_insertion()
```

### 4. Text Surface Transformations

Change formatting while preserving meaning.

```python

def surface_transforms():
    """
    Surface-level transformations
    """
    print("Technique 4: Surface Transformations")
    print("=" * 60)

    original = "The cat sat on the mat."

    transformations = [
        ("lowercase", original.lower()),
        ("uppercase", original.upper()),
        ("title case", original.title()),
        ("remove punctuation", original.replace(".", "")),
        ("extra spaces", original.replace(" ", "  ")),
        ("contractions", "The cat's sat on the mat."),
    ]

    print(f"Original: {original}\n")

    for name, transformed in transformations:
        print(f"{name:15}: {transformed}")

surface_transforms()
```

### 5. MixUp (Advanced)

Create new examples by interpolating between existing ones.

```python

def mixup():
    """
    MixUp: interpolating between examples
    """
    print("Technique 5: MixUp (Advanced)")
    print("=" * 60)

    print("""
    MixUp creates new examples by combining two existing ones:

    Text A: "I loved this movie"
    Text B: "This film was terrible"

    Mix (λ = 0.7):
    New text: 70% A + 30% B (in embedding space)
    New label: 70% positive + 30% negative

    This teaches the model to handle ambiguous cases
    and smooth decision boundaries.

    Used in: BERT-MixUp, text classification
    """)

mixup()
```

---

## Augmentation for Different Tasks

### Classification Tasks

```python

def classification_aug():
    """
    Augmentation for classification
    """
    print("Augmentation for Classification")
    print("=" * 60)

    task_examples = {
        "Sentiment Analysis": {
            "original": "This movie is great!",
            "augmented": [
                "This film is great!",
                "This movie is fantastic!",
                "I loved this movie",
                "This movie is really great"
            ]
        },
        "Spam Detection": {
            "original": "Get rich fast!!!",
            "augmented": [
                "Get rich quickly!!!",
                "Become wealthy fast!!!",
                "Make money fast!!!",
                "Get rich now!!!"
            ]
        }
    }

    for task, examples in task_examples.items():
        print(f"\n{task}:")
        print(f"  Original: {examples['original']}")
        print("  Augmented:")
        for aug in examples['augmented'][:2]:
            print(f"    • {aug}")

classification_aug()
```

### NER (Named Entity Recognition)

```python

def ner_aug():
    """
    Augmentation for NER
    """
    print("Augmentation for NER")
    print("=" * 60)

    print("""
    For NER, we must preserve entity boundaries and types:

    Original: "[Steve Jobs]PERSON founded [Apple]ORG"

    Safe augmentations:
    • Replace with other person names: "[Bill Gates]PERSON founded [Microsoft]ORG"
    • Paraphrase around entities: "The founder of [Apple]ORG was [Steve Jobs]PERSON"
    • Add modifiers: "[Steve Jobs]PERSON famously founded [Apple]ORG"

    Must keep entity annotations intact!
    """)

ner_aug()
```

---

## Code: Complete Augmentation Pipeline

```python

import random
import nltk
from nltk.corpus import wordnet

def augmentation_pipeline():
    """
    Complete augmentation example
    """
    print("Complete Augmentation Pipeline")
    print("=" * 60)

    # Download wordnet if needed (uncomment in real code)
    # nltk.download('wordnet')

    def get_synonyms(word):
        """Get synonyms for a word using WordNet"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.add(lemma.name())
        return list(synonyms)

    def augment_text(text, techniques=['synonym', 'backtranslate']):
        """Apply multiple augmentation techniques"""
        augmented = [text]  # original

        # Synonym replacement (simplified)
        if 'synonym' in techniques:
            words = text.split()
            for i, word in enumerate(words):
                if random.random() < 0.2:  # 20% chance
                    syns = get_synonyms(word)
                    if syns:
                        new_words = words.copy()
                        new_words[i] = random.choice(syns)
                        augmented.append(' '.join(new_words))

        # Random word insertion
        if 'insert' in techniques:
            insert_words = ["very", "really", "quite"]
            words = text.split()
            pos = random.randint(0, len(words))
            new_words = words.copy()
            new_words.insert(pos, random.choice(insert_words))
            augmented.append(' '.join(new_words))

        return augmented

    # Test on example
    text = "The movie was good"
    print(f"Original: {text}")

    # Set seed for reproducibility
    random.seed(42)

    augmented = augment_text(text)
    print("\nAugmented versions:")
    for i, aug in enumerate(augmented[1:], 1):
        print(f"  {i}. {aug}")

augmentation_pipeline()
```

---

## When to Use (and Not Use) Augmentation

### Best Practices

| Situation                      | Use Augmentation? | Why                                   |
| ------------------------------ | ----------------- | ------------------------------------- |
| Small dataset (<1000 examples) | ✅ YES            | Need more data                        |
| Imbalanced classes             | ✅ YES            | Create examples for minority class    |
| Overfitting                    | ✅ YES            | Adds diversity, reduces memorization  |
| Text classification            | ✅ YES            | Label-preserving transformations easy |
| Translation                    | ✅ YES            | Back-translation is perfect for this  |
| NER with entities              | ⚠️ CAUTIOUSLY     | Must preserve entity annotations      |
| Grammar correction             | ⚠️ CAUTIOUSLY     | Can introduce errors                  |
| Very large dataset             | 🤔 MAYBE          | Might not help much                   |
| Already high performance       | 🤔 MAYBE          | Test if it improves                   |

### When Augmentation Can Hurt

```python

def augmentation_risks():
    """
    Risks of data augmentation
    """
    print("When Augmentation Can Backfire")
    print("=" * 60)

    risks = [
        "CHANGING MEANING: 'not good' → 'not bad' flips sentiment",
        "INTRODUCING ERRORS: Bad synonyms can create nonsense",
        "OVER-AUGMENTING: Model sees unrealistic variations",
        "ENTITY CONFUSION: In NER, changing names breaks annotations",
        "DOMAIN SHIFT: Augmentations that don't match real data"
    ]

    for risk in risks:
        print(f"  • {risk}")

augmentation_risks()
```

---

## Augmentation Cheat Sheet

| Technique           | How It Works           | Best For          | Risk                  |
| ------------------- | ---------------------- | ----------------- | --------------------- |
| Synonym replacement | Replace with synonyms  | Classification    | Bad synonyms          |
| Back-translation    | Translate and back     | All tasks         | Low (high quality)    |
| Random insertion    | Add filler words       | Sentiment         | Can change meaning    |
| Random deletion     | Remove words           | Robustness        | Loses info            |
| Surface transforms  | Case, punctuation      | Format invariance | Minimal               |
| MixUp               | Interpolate embeddings | Classification    | Hard to interpret     |
| LLM-based           | Use LLM to paraphrase  | All tasks         | Cost, quality control |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Data augmentation is a force multiplier for your data:     ║
║                                                              ║
║  📈 MORE DATA WITHOUT COLLECTING: Turn 1000 examples        ║
║     into 10,000 variations with minimal effort              ║
║                                                              ║
║  🛡️ PREVENTS OVERFITTING: Models see the same              ║
║     meaning in different forms, learning concepts           ║
║     instead of memorizing surface patterns                  ║
║                                                              ║
║  ⚖️ BALANCES IMBALANCED DATASETS: Create more               ║
║     examples for underrepresented classes                   ║
║                                                              ║
║  🌍 IMPROVES ROBUSTNESS: Models handle real-world           ║
║     variations (typos, synonyms, different phrasing)        ║
║                                                              ║
║  💰 COST-EFFECTIVE: Much cheaper than collecting            ║
║     new data, especially for specialized domains            ║
║                                                              ║
║  In production LLMs, augmentation is often the              ║
║  difference between a brittle model and a robust one.       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Data augmentation creates new training examples by modifying existing ones—like a basketball coach running varied drills to prepare players for any situation, without needing new players

• Techniques include synonym replacement, back-translation, random insertion/deletion, and surface transformations—each preserving meaning while changing surface form

• Augmentation is especially valuable for small datasets, imbalanced classes, and improving model robustness—but must be applied carefully to avoid changing the actual label or meaning

---

## Mental Hook

> "Data augmentation is like a musician practicing a song in every key, tempo, and style—they're not learning new songs, but they're becoming so fluent that they can play the original perfectly in any situation."
