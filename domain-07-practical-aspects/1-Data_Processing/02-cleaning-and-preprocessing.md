# Cleaning and Preprocessing

## The Gold Refinery Analogy

Imagine you've just mined a ton of rock. It contains gold—but also dirt, gravel, toxic minerals, and worthless rubble. You can't just melt the whole pile and call it jewelry. You need to crush it, wash it, separate the gold, remove impurities, and refine it into pure metal. That's cleaning and preprocessing for LLMs: taking the raw data (web crawls, books, etc.) and transforming it into clean, usable training material.

Raw internet data is a mess. It contains HTML tags, ads, duplicate content, gibberish, hate speech, and personal information. If you trained on this directly, your model would learn to output HTML, repeat itself, and potentially harm users. Cleaning and preprocessing turns digital garbage into training gold.

---

## What Is Cleaning and Preprocessing?

### The Core Idea

Cleaning and preprocessing is the process of transforming raw collected data into high-quality training examples.

```text

Raw Data (messy)                      Clean Data (ready to train)
─────────────────────────────────────────────────────────────────

<html><body>                          The Industrial Revolution
<div class="ad">Buy now!</div>        began in Britain in the
<p>The Industrial Revolution began     late 18th century...
in Britain...</p></body></html>

Check out this awesome site!!!        [REMOVED - low quality]
Click here!!!

User123: "I hate everything"          [REMOVED - toxic]
Reply: "Me too"

... (duplicate of same article)       [REMOVED - duplicate]
```

```python

def preprocessing_intro():
    """
    The basic concept of cleaning and preprocessing
    """
    print("Cleaning and Preprocessing: From Garbage to Gold")
    print("=" * 60)

    print("""
    What we need to remove or fix:

    • HTML tags, scripts, styling
    • Ads, navigation menus
    • Duplicate content
    • Gibberish (lorem ipsum, spam)
    • Toxic content (hate speech, violence)
    • Personal information (PII)
    • Low-quality machine-generated text
    • Formatting inconsistencies
    """)

preprocessing_intro()
```

---

## Step 1: Text Extraction

### From HTML to Plain Text

```python

def html_extraction():
    """
    Extracting text from HTML
    """
    print("Step 1: Text Extraction")
    print("=" * 60)

    html_doc = """
    <html>
    <head><title>My Page</title></head>
    <body>
        <div class="header">Navigation Links</div>
        <div class="ad">Buy cheap watches!!!</div>
        <article>
            <h1>The Industrial Revolution</h1>
            <p>The Industrial Revolution began in <strong>Britain</strong>
            in the late 18th century.</p>
        </article>
        <div class="footer">Copyright 2024</div>
    </body>
    </html>
    """

    print("Raw HTML:")
    print(html_doc)

    # Simulated extraction
    print("\nAfter extraction (using BeautifulSoup, trafilatura, etc.):")
    print("""
    The Industrial Revolution
    The Industrial Revolution began in Britain in the late 18th century.
    """)

    print("\nTools: BeautifulSoup, trafilatura, jusText, boilerpy3")

html_extraction()
```

### Code Example: Simple HTML Extraction

```python

import re

def simple_extraction():
    """
    Simple HTML tag removal
    """
    print("Simple HTML Tag Removal")
    print("=" * 60)

    html = "<p>The <strong>cat</strong> sat on the <em>mat</em>.</p>"

    # Remove HTML tags with regex (simplified)
    clean = re.sub(r'<[^>]+>', '', html)

    print(f"Original: {html}")
    print(f"Cleaned:  {clean}")

    print("\n⚠️ Real extraction is more complex—handling scripts, CSS, etc.")
    print("Use libraries like BeautifulSoup for production!")

simple_extraction()
```

---

## Step 2: Language Identification

### Keeping the Languages You Want

```python

def language_id():
    """
    Identifying and filtering by language
    """
    print("Step 2: Language Identification")
    print("=" * 60)

    samples = [
        "The cat sat on the mat.",  # English
        "Le chat s'est assis sur le tapis.",  # French
        "Die Katze saß auf der Matte.",  # German
        "猫はマットの上に座った。",  # Japanese
        "qwerty asdfgh zxcvbn",  # Gibberish
    ]

    print("Detecting languages:")
    for text in samples:
        # Simulated language detection
        if "cat" in text:
            lang = "English"
        elif "chat" in text:
            lang = "French"
        elif "Katze" in text:
            lang = "German"
        elif "猫" in text:
            lang = "Japanese"
        else:
            lang = "Unknown (probably gibberish)"

        print(f"  {text[:30]}... → {lang}")

    print("\nTools: fastText, langdetect, CLD2")
    print("Common practice: Keep only top 10-20 languages")

language_id()
```

---

## Step 3: Quality Filtering

### Removing Low-Quality Content

```python

def quality_filtering():
    """
    Filtering out low-quality text
    """
    print("Step 3: Quality Filtering")
    print("=" * 60)

    low_quality_signals = [
        "Excessive punctuation (!!!!!!)",
        "ALL CAPS TEXT",
        "Repetitive content (buy now buy now buy now)",
        "Very short sentences (< 5 words)",
        "Gibberish (asdfghjkl)",
        "Machine-generated spam",
        "Clickbait headlines"
    ]

    print("Signals of low-quality text:")
    for signal in low_quality_signals:
        print(f"  • {signal}")

    print("\nQuality scoring example:")
    texts = [
        ("The Industrial Revolution began in Britain...", 0.95),
        ("Click here!!! Free money!!! Now!!!", 0.15),
        ("Cat cat cat cat cat", 0.30),
    ]

    for text, score in texts:
        bar = "█" * int(score * 20)
        print(f"  {score:.2f} {bar:20} {text[:30]}...")

quality_filtering()
```

### Code: Simple Quality Heuristics

```python

def quality_heuristics():
    """
    Simple quality scoring
    """
    print("Simple Quality Heuristics")
    print("=" * 60)

    def quality_score(text):
        score = 1.0

        # Penalize ALL CAPS
        if text.isupper() and len(text) > 20:
            score -= 0.3

        # Penalize excessive punctuation
        punct_count = text.count('!') + text.count('?') + text.count('.')
        if punct_count > len(text) * 0.3:
            score -= 0.2

        # Penalize repetition
        words = text.split()
        if len(set(words)) < len(words) * 0.3:
            score -= 0.3

        # Penalize very short sentences
        if len(words) < 5:
            score -= 0.4

        return max(0, score)

    test_texts = [
        "The Industrial Revolution began in Britain in the late 18th century.",
        "CLICK HERE FOR FREE MONEY!!! NOW!!! LIMITED TIME!!!",
        "cat cat cat cat cat cat cat cat cat cat",
        "Hello world, how are you today?"
    ]

    for text in test_texts:
        score = quality_score(text)
        bar = "█" * int(score * 20)
        print(f"\nScore: {score:.2f} {bar}")
        print(f"Text: {text}")

quality_heuristics()
```

---

## Step 4: Toxicity and Harmful Content Filtering

### Keeping It Safe

```python

def toxicity_filtering():
    """
    Removing harmful content
    """
    print("Step 4: Toxicity and Harmful Content Filtering")
    print("=" * 60)

    harmful_categories = [
        "Hate speech (racial, religious, etc.)",
        "Violence and gore",
        "Sexual content",
        "Harassment and bullying",
        "Dangerous instructions (how to make weapons)",
        "Personal attacks",
        "Extreme profanity"
    ]

    print("Categories typically filtered out:")
    for cat in harmful_categories:
        print(f"  • {cat}")

    print("\nTools: Perspective API, detoxify, custom classifiers")
    print("This is crucial for safety and alignment!")

toxicity_filtering()
```

---

## Step 5: PII Redaction

### Protecting Privacy

```python

def pii_redaction():
    """
    Removing personal information
    """
    print("Step 5: PII Redaction")
    print("=" * 60)

    pii_types = [
        "Names (John Smith → [NAME])",
        "Email addresses (user@example.com → [EMAIL])",
        "Phone numbers (555-123-4567 → [PHONE])",
        "Physical addresses",
        "Social security numbers",
        "Credit card numbers",
        "IP addresses"
    ]

    print("Personal information to redact:")
    for pii in pii_types:
        print(f"  • {pii}")

    print("\nExample:")
    original = "Contact John at john.smith@email.com or 555-123-4567"
    redacted = "Contact [NAME] at [EMAIL] or [PHONE]"

    print(f"  Original: {original}")
    print(f"  Redacted: {redacted}")

pii_redaction()
```

### Code: Simple PII Detection

```python

import re

def pii_detection():
    """
    Simple PII detection with regex
    """
    print("Simple PII Detection")
    print("=" * 60)

    def redact_pii(text):
        # Email pattern (simplified)
        text = re.sub(r'b[w.-]+@[w.-]+.w+b', '[EMAIL]', text)

        # Phone pattern (simplified)
        text = re.sub(r'bd{3}[-.]?d{3}[-.]?d{4}b', '[PHONE]', text)

        # SSN pattern
        text = re.sub(r'bd{3}-d{2}-d{4}b', '[SSN]', text)

        return text

    test = "Contact: john.doe@email.com, phone 555-123-4567, SSN 123-45-6789"

    print(f"Original: {test}")
    print(f"Redacted: {redact_pii(test)}")

    print("\n⚠️ Real PII detection is much more complex!")
    print("Use specialized libraries like presidio, scrubadub")

pii_detection()
```

---

## Step 6: Deduplication

### Removing Duplicates

```python

def deduplication():
    """
    Finding and removing duplicate content
    """
    print("Step 6: Deduplication")
    print("=" * 60)

    print("""
    Duplicates waste training time and can bias the model.

    Types of duplicates:

    EXACT DUPLICATES: Same article on multiple sites
    → Remove all but one copy

    NEAR-DUPLICATES: Same article with minor edits
    → Use MinHash or similar to find and filter

    TEMPLATE CONTENT: Same navigation menus on every page
    → Extract main content, remove boilerplate

    REPEATED PHRASES: "Click here" thousands of times
    → Document-level scoring, filter outliers
    """)

deduplication()
```

### Code: Simple Near-Duplicate Detection

```python

def near_duplicate():
    """
    Simple near-duplicate detection with MinHash concept
    """
    print("Near-Duplicate Detection Concept")
    print("=" * 60)

    from collections import Counter

    def jaccard_similarity(text1, text2):
        """Simplified Jaccard similarity on words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0

    docs = [
        "The cat sat on the mat. It was comfortable.",
        "The cat sat on the mat. It was very comfortable.",
        "Dogs love to play fetch in the park.",
    ]

    print("Document pairs and their similarity:")
    sim = jaccard_similarity(docs[0], docs[1])
    print(f"  Docs 1 & 2: {sim:.2f} (near-duplicate)")

    sim = jaccard_similarity(docs[0], docs[2])
    print(f"  Docs 1 & 3: {sim:.2f} (different)")

near_duplicate()
```

---

## Step 7: Normalization

### Making Text Consistent

```python

def normalization():
    """
    Text normalization steps
    """
    print("Step 7: Normalization")
    print("=" * 60)

    examples = [
        ("Unicode normalization", "café vs cafeu0301"),
        ("Whitespace normalization", "Multiple   spaces → single"),
        ("Case folding (optional)", "The CAT → the cat (for some tasks)"),
        ("Punctuation standardization", "“smart quotes” → "straight quotes""),
        ("Number formatting", "1,000 vs 1000"),
        ("URL removal", "http://example.com → [URL] or removed")
    ]

    for step, desc in examples:
        print(f"  • {step}: {desc}")

normalization()
```

---

## The Complete Preprocessing Pipeline

```python

def complete_pipeline():
    """
    The full cleaning and preprocessing pipeline
    """
    print("Complete Preprocessing Pipeline")
    print("=" * 60)

    steps = [
        "1. RAW DATA → Extract text from HTML/PDF/etc.",
        "2. EXTRACTED TEXT → Language identification",
        "3. TARGET LANGUAGES → Quality filtering",
        "4. QUALITY TEXT → Toxicity filtering",
        "5. SAFE TEXT → PII redaction",
        "6. CLEAN TEXT → Deduplication (exact and near)",
        "7. UNIQUE TEXT → Normalization",
        "8. NORMALIZED TEXT → Final dataset ready for training"
    ]

    for step in steps:
        print(f"  {step}")

    print("\nData volume reduction:")
    print("  Raw crawl: 100 TB")
    print("  After filtering: 10-30 TB")
    print("  After deduplication: 5-15 TB")
    print("  Final training data: 1-10 TB")

complete_pipeline()
```

---

## Cleaning and Preprocessing Cheat Sheet

| Step               | What It Does            | Why It Matters                  | Tools                        |
| ------------------ | ----------------------- | ------------------------------- | ---------------------------- |
| Extraction         | Get text from HTML/PDF  | Remove non-content              | BeautifulSoup, trafilatura   |
| Language ID        | Identify language       | Keep target languages           | fastText, langdetect         |
| Quality filtering  | Remove low-quality text | Improve model performance       | Custom heuristics, ML models |
| Toxicity filtering | Remove harmful content  | Safety, alignment               | Perspective API, detoxify    |
| PII redaction      | Remove personal info    | Privacy, compliance             | presidio, scrubadub          |
| Deduplication      | Remove duplicates       | Efficient training, reduce bias | MinHash, SimHash             |
| Normalization      | Make text consistent    | Reduce sparsity                 | unicodedata, custom rules    |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Cleaning and preprocessing isn't optional—it's essential:  ║
║                                                              ║
║  🧹 REMOVES GARBAGE: Without cleaning, your model learns    ║
║     to generate HTML, ads, and spam instead of language     ║
║                                                              ║
║  🛡️ PROTECTS PRIVACY: PII in training data can leak         ║
║     through the model—redaction is crucial                   ║
║                                                              ║
║  ⚖️ REDUCES BIAS: Filtering toxic content prevents          ║
║     the model from learning harmful patterns                 ║
║                                                              ║
║  💰 SAVES MONEY: Training on duplicates wastes              ║
║     compute—deduplication can cut costs significantly        ║
║                                                              ║
║  🎯 IMPROVES PERFORMANCE: Clean data = better model         ║
║     Quality matters as much as quantity                      ║
║                                                              ║
║  The difference between a good model and a great model      ║
║  is often in the preprocessing, not the architecture.       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Cleaning and preprocessing transforms raw, messy data into high-quality training material—like refining gold ore into pure metal, removing impurities that would ruin the final product

• Key steps include extraction, language ID, quality filtering, toxicity removal, PII redaction, deduplication, and normalization—each addressing a specific problem in raw data

• This process can reduce data volume by 90% or more—but the remaining 10% is what actually makes a model smart, safe, and useful
