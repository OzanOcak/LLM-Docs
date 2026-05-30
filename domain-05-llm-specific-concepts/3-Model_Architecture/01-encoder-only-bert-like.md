# Encoder-Only Models (BERT-like)

## The Detective Analogy

Imagine a detective examining a mystery novel. They read the entire book cover to cover, making notes, connecting clues from Chapter 1 with revelations from Chapter 20. They can look forward and backward, seeing how everything fits together. That's an encoder-only model like BERT: it processes the entire input at once, with full bidirectional context, perfect for understanding and analysis tasks.

In the LLM family, encoder-only models are the specialists in understanding. They don't generate text—they read and comprehend. BERT and its variants power Google Search, sentiment analysis, named entity recognition, and countless other applications where understanding text matters more than generating it.

---

## What Is an Encoder-Only Model?

### The Core Idea

Encoder-only models use only the encoder part of the transformer, processing input bidirectionally to create rich representations.

```text

Decoder-only (GPT):                     Encoder-only (BERT):
    → → → → → → →                          ← → ← → ← → ←
    The cat sat on the                     The cat sat on the
        ↓   ↓   ↓   ↓                           ↓   ↓   ↓   ↓
    Generate next word                    Understand everything

    "Look at past words only"             "Look at ALL words"
    Good for GENERATION                    Good for UNDERSTANDING
```

```python

def encoder_only_intro():
    """
    The basic concept of encoder-only models
    """
    print("Encoder-Only Models: Masters of Understanding")
    print("=" * 60)

    print("""
    Architecture:
    • Only the encoder stack (no decoder)
    • Bidirectional attention (full context)
    • Each token sees ALL other tokens

    Training objective: Masked Language Modeling
    • Predict randomly masked words
    • Uses context from both sides

    Output: Rich representations for each token
    • Can be used for classification, QA, NER
    """)

encoder_only_intro()
```

---

## How Encoder-Only Models Work

### Bidirectional Attention

```python

def bidirectional_attention():
    """
    How encoder-only models see all context
    """
    print("Bidirectional Attention: Seeing Everything")
    print("=" * 60)

    sentence = "The cat sat on the mat"
    words = sentence.split()

    print(f"Sentence: '{sentence}'")
    print("\nFor the word 'sat', encoder-only sees:")
    print("  • Left context: 'The', 'cat'")
    print("  • Right context: 'on', 'the', 'mat'")
    print("  • ALL at once!")

    print("\nAttention pattern for 'sat':")
    print("  The: 0.1")
    print("  cat: 0.3")
    print("  sat: 0.2 (itself)")
    print("  on:  0.15")
    print("  the: 0.1")
    print("  mat: 0.15")
    print("  (sums to 1.0)")

bidirectional_attention()
```

### Masked Language Modeling

```python

def masked_lm():
    """
    How encoder-only models are trained
    """
    print("Training: Masked Language Modeling")
    print("=" * 60)

    sentence = "The cat sat on the mat"

    print(f"Original: '{sentence}'")
    print("\nTraining examples (15% of tokens masked):")

    masked_examples = [
        ("The [MASK] sat on the mat", "cat"),
        ("The cat [MASK] on the mat", "sat"),
        ("The cat sat on [MASK] mat", "the"),
        ("[MASK] cat sat on the mat", "The")
    ]

    for masked, target in masked_examples:
        print(f"  Input: '{masked:30}' → Predict: '{target}'")

    print("\nModel must use BOTH left and right context!")
    print("This creates deep bidirectional understanding.")

masked_lm()
```

---

## BERT Architecture

### BERT Base vs BERT Large

```python

def bert_sizes():
    """
    BERT model sizes
    """
    print("BERT Model Variants")
    print("=" * 60)

    variants = {
        "BERT Base": {
            "layers": 12,
            "hidden_size": 768,
            "attention_heads": 12,
            "parameters": "110M"
        },
        "BERT Large": {
            "layers": 24,
            "hidden_size": 1024,
            "attention_heads": 16,
            "parameters": "340M"
        }
    }

    for name, specs in variants.items():
        print(f"\n{name}:")
        for key, value in specs.items():
            print(f"  • {key}: {value}")

bert_sizes()
```

### BERT Input Format

```python

def bert_input():
    """
    How BERT formats input
    """
    print("BERT Input Format")
    print("=" * 60)

    sentence1 = "I love this movie"
    sentence2 = "It was great"

    print(f"Sentence 1: '{sentence1}'")
    print(f"Sentence 2: '{sentence2}'")

    print("\nBERT input format:")
    print("  [CLS] I love this movie [SEP] It was great [SEP]")

    print("\nToken roles:")
    print("  [CLS] - Classification token")
    print("         Final representation used for classification")
    print("  [SEP] - Separator between sentences")
    print("         Helps model distinguish different segments")

    # Token type IDs
    print("\nToken Type IDs (segment embeddings):")
    print("  [CLS] I love this movie [SEP] It was great [SEP]")
    print("     0  0   0    0    0     0    1   1    1     1")
    print("  (0 = first sentence, 1 = second sentence)")

bert_input()
```

---

## What Encoder-Only Models Can Do

### 1. Text Classification

```python

def classification():
    """
    Using BERT for classification
    """
    print("Text Classification with BERT")
    print("=" * 60)

    print("""
    Process:
    1. Add [CLS] token at start
    2. Pass through BERT
    3. Use [CLS] final representation
    4. Add classification head

    Example: Sentiment Analysis
    Input: "[CLS] I loved this movie [SEP]"
           ↓
    BERT → [CLS] representation (768-dim)
           ↓
    Classifier → positive (90%) / negative (10%)
    """)

classification()
```

### 2. Named Entity Recognition (NER)

```python

def ner():
    """
    Token-level classification with BERT
    """
    print("Named Entity Recognition with BERT")
    print("=" * 60)

    sentence = "Apple was founded by Steve Jobs in Cupertino"

    print(f"Sentence: '{sentence}'")
    print("\nBERT tags each token:")
    print("  Apple     → B-ORG (Beginning of Organization)")
    print("  was       → O (Outside any entity)")
    print("  founded   → O")
    print("  by        → O")
    print("  Steve     → B-PER (Beginning of Person)")
    print("  Jobs      → I-PER (Inside Person)")
    print("  in        → O")
    print("  Cupertino → B-LOC (Beginning of Location)")

    print("\nEach token gets its own classification!")

ner()
```

### 3. Question Answering

```python

def qa():
    """
    Extractive QA with BERT
    """
    print("Question Answering with BERT")
    print("=" * 60)

    context = "The capital of France is Paris. It is known for the Eiffel Tower."
    question = "What is the capital of France?"

    print(f"Context: {context}")
    print(f"Question: {question}")

    print("\nBERT format:")
    print("  [CLS] question [SEP] context [SEP]")

    print("\nBERT predicts start and end positions:")
    print("  Start token: 'Paris' (position 5)")
    print("  End token:   'Paris' (position 5)")
    print("  Answer: Paris")

    print("\nThe model finds the answer span in the context!")

qa()
```

### 4. Sentence Pair Tasks

```python

def sentence_pair():
    """
    Sentence pair tasks (NLI, paraphrase detection)
    """
    print("Sentence Pair Tasks with BERT")
    print("=" * 60)

    tasks = {
        "Natural Language Inference": {
            "premise": "A dog is playing in the park",
            "hypothesis": "An animal is outside",
            "label": "entailment"
        },
        "Paraphrase Detection": {
            "sentence1": "The cat sat on the mat",
            "sentence2": "A cat was sitting on the rug",
            "label": "paraphrase"
        }
    }

    for task_name, data in tasks.items():
        print(f"\n{task_name}:")
        for key, value in data.items():
            print(f"  {key}: {value}")

    print("\nBERT processes both sentences with [SEP]")
    print("[CLS] representation captures relationship")

sentence_pair()
```

---

## Encoder-Only vs Other Architectures

| Aspect        | Encoder-Only (BERT)     | Decoder-Only (GPT)   | Encoder-Decoder (T5)         |
| ------------- | ----------------------- | -------------------- | ---------------------------- |
| Attention     | Bidirectional           | Causal (left only)   | Encoder: bi, Decoder: causal |
| Training      | Masked LM               | Next word prediction | Span corruption              |
| Best at       | Understanding           | Generation           | Transformation               |
| Sees future?  | Yes                     | No                   | Encoder: Yes                 |
| Output        | Representations         | Text                 | Text                         |
| Example tasks | Classification, NER, QA | Chat, completion     | Translation, summary         |

### When to Use Encoder-Only

```python

def when_to_use():
    """
    When to choose encoder-only models
    """
    print("When to Use Encoder-Only Models")
    print("=" * 60)

    use_cases = [
        "Sentiment analysis",
        "Named entity recognition",
        "Text classification",
        "Extractive question answering",
        "Sentence similarity",
        "Natural language inference",
        "Feature extraction for other models"
    ]

    print("Perfect for:")
    for task in use_cases:
        print(f"  • {task}")

    print("\nNot good for:")
    print("  • Text generation")
    print("  • Creative writing")
    print("  • Chatbots")
    print("  • Any task requiring free-form output")

when_to_use()
```

---

## Why This Matters for LLMs

### 1. BERT Revolutionized NLP

```python

def bert_revolution():
    """
    How BERT changed NLP
    """
    print("The BERT Revolution (2018)")
    print("=" * 60)

    print("""
    Before BERT:
    • Task-specific architectures
    • Limited pre-training
    • Less transfer learning

    BERT introduced:
    • Deep bidirectional pre-training
    • One model for many tasks
    • State-of-the-art on 11 tasks
    • Accessible to everyone (open source!)

    This started the "pre-train, fine-tune" paradigm
    that dominates NLP today.
    """)

bert_revolution()
```

### 2. Still Widely Used

```python

def bert_family():
    """
    The BERT family of models
    """
    print("The BERT Family")
    print("=" * 60)

    models = {
        "BERT": "Original",
        "RoBERTa": "Optimized BERT (more data, longer training)",
        "DistilBERT": "Smaller, faster (40% size, 95% performance)",
        "ALBERT": "Even smaller (parameter sharing)",
        "ELECTRA": "More efficient pre-training",
        "DeBERTa": "Enhanced with disentangled attention"
    }

    for model, desc in models.items():
        print(f"  • {model}: {desc}")

bert_family()
```

### 3. Powering Real-World Applications

```python

def real_world():
    """
    Real-world applications of encoder-only models
    """
    print("Real-World Applications")
    print("=" * 60)

    apps = {
        "Google Search": "BERT powers search understanding",
        "Spam detection": "Email classification",
        "Customer support": "Ticket categorization",
        "Healthcare": "Medical record analysis",
        "Legal tech": "Document review",
        "Social media": "Content moderation"
    }

    for app, desc in apps.items():
        print(f"  • {app}: {desc}")

real_world()
```

### 4. Feature Extraction

```python

def feature_extraction():
    """
    Using BERT as a feature extractor
    """
    print("BERT as a Feature Extractor")
    print("=" * 60)

    print("""
    You can use BERT without fine-tuning:

    1. Pass text through BERT
    2. Extract [CLS] or mean pooling
    3. Use as features for other models

    Benefits:
    • No training needed
    • High-quality embeddings
    • Works for many tasks

    Example:
    text → BERT → [0.2, -0.5, 0.8, ...] (768-dim vector)
                ↓
          Your classifier (logistic regression, SVM, etc.)
    """)

feature_extraction()
```

---

## Encoder-Only Cheat Sheet

| Aspect         | Details                      |
| -------------- | ---------------------------- |
| Architecture   | Transformer encoder only     |
| Attention      | Bidirectional (full context) |
| Training       | Masked Language Modeling     |
| Special tokens | [CLS], [SEP], [MASK]         |
| Output         | Representations (not text)   |
| Best for       | Understanding tasks          |
| Examples       | BERT, RoBERTa, DistilBERT    |
| Parameters     | 110M (base) to 340M (large)  |

---

## Quick Recap

• Encoder-only models like BERT process text bidirectionally—like a detective who reads the whole mystery novel before drawing conclusions, they see all context at once

• They're trained on masked language modeling—predicting randomly hidden words using context from both sides, which creates deep bidirectional understanding

• These models excel at understanding tasks—classification, named entity recognition, question answering, and feature extraction—but can't generate text, making them perfect for analysis, not creation

---

## Mental Hook

> "Encoder-only models are like brilliant readers who devour entire books at once, understanding every word in context—they can't write stories, but ask them anything about what they've read and they'll give you the perfect answer."
