# Tokenization: Word vs Subword vs Character

## The LEGO Analogy

Imagine you have three ways to build with LEGOs. You could use pre-built houses (whole words)—fast but limited because you can't make new combinations. You could use individual bricks (characters)—flexible but slow to build anything complex. Or you could use a mix of bricks and pre-built sections (subwords)—the sweet spot that gives you both speed and flexibility. That's tokenization: deciding how to break text into pieces for the model to process.

In LLMs, tokenization is the crucial first step that converts raw text into numbers the model can understand. The choice of tokenization affects vocabulary size, handling of unknown words, model size, and even performance on different languages. It's the foundation everything else builds on.

---

## What Is Tokenization?

### The Core Idea

Tokenization is the process of splitting text into smaller pieces (tokens) and mapping them to numbers.

```text

Raw text: "The cat sat"
              ↓
    Tokenization (splitting)
              ↓
    ["The", "cat", "sat"]   (word tokens)
              ↓
    Mapping to IDs
              ↓
    [464, 5432, 8912]       (what the model sees)
```

```python

def tokenization_intro():
    """
    The basic concept of tokenization
    """
    print("Tokenization: Turning Text into Numbers")
    print("=" * 60)

    text = "The cat sat on the mat"

    print(f"Raw text: '{text}'")
    print("\nStep 1: Split into tokens")
    print("Step 2: Map each token to an ID")
    print("Step 3: Model processes IDs")

    print("\nAnalogy: Like converting ingredients to")
    print("recipe codes before cooking.")

tokenization_intro()
```

---

## Three Levels of Tokenization

### 1. Character Tokenization

Character tokenization splits text into individual characters.

```python

def character_tokenization():
    """
    Character-level tokenization
    """
    print("Character Tokenization: Individual Letters")
    print("=" * 60)

    text = "cat"

    # Character tokens
    chars = list(text)

    print(f"Text: '{text}'")
    print(f"Character tokens: {chars}")

    # Vocabulary size
    vocab_size = len(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:"))

    print(f"\nVocabulary size: ~{vocab_size} characters")
    print("(plus punctuation, numbers, special tokens)")

    print("\nPros:")
    print("  • Tiny vocabulary (≈ 100-200 tokens)")
    print("  • Can handle any word (even misspellings)")
    print("  • No unknown tokens")

    print("\nCons:")
    print("  • Very long sequences (every letter is a token)")
    print("  • Harder to learn word meanings")
    print("  • Computationally inefficient")

    # Example
    sentence = "The cat sat"
    char_count = len(sentence)
    word_count = len(sentence.split())
    print(f"\nExample: '{sentence}'")
    print(f"  Word count: {word_count} tokens")
    print(f"  Char count: {char_count} tokens ({char_count/word_count:.1f}x longer)")

character_tokenization()
```

### 2. Word Tokenization

Word tokenization splits text into whole words.

```python

def word_tokenization():
    """
    Word-level tokenization
    """
    print("Word Tokenization: Complete Words")
    print("=" * 60)

    text = "The cat sat on the mat"

    # Word tokens (simplified)
    words = text.lower().split()

    print(f"Text: '{text}'")
    print(f"Word tokens: {words}")

    # Vocabulary size (English has many words!)
    print(f"\nVocabulary size needed: 50,000 - 500,000+")
    print("(English has hundreds of thousands of words)")

    print("\nPros:")
    print("  • Short sequences (one token per word)")
    print("  • Semantically meaningful units")
    print("  • Efficient for common words")

    print("\nCons:")
    print("  • Huge vocabulary")
    print("  • Can't handle unknown words (UNK tokens)")
    print("  • Misses relationships between word forms")
    print("  • Different languages need different vocabularies")

    # Unknown word problem
    print("\nUnknown word problem:")
    print("  Training: sees 'cat', 'cats'")
    print("  Inference: sees 'caten' (unknown) → [UNK]")
    print("  Model loses all information!")

word_tokenization()
```

### 3. Subword Tokenization (The Sweet Spot)

Subword tokenization splits words into common pieces—frequent words stay whole, rare words break down.

```python

def subword_intro():
    """
    Subword tokenization: Best of both worlds
    """
    print("Subword Tokenization: The Sweet Spot")
    print("=" * 60)

    print("""
    Common words stay whole:   "the", "cat", "sat"

    Rare words break down:     "caten" → ["cat", "en"]
    "unhappiness" → ["un", "happiness"] or ["un", "happy", "ness"]

    New words can be built from pieces!
    """)

    print("\nVocabulary size: 30,000 - 50,000 (typical for LLMs)")
    print("• Smaller than word vocab")
    print("• Larger than character vocab")

    print("\nPros:")
    print("  • No unknown words (can always break down)")
    print("  • Efficient for common words")
    print("  • Shares pieces across words (learns morphology)")
    print("  • Works across languages")

    print("\nUsed in: All modern LLMs (GPT, BERT, Llama, Claude)")

subword_intro()
```

---

## Subword Tokenization Algorithms

### Byte-Pair Encoding (BPE)

BPE starts with characters and merges the most frequent pairs.

```python

def bpe_demo():
    """
    How Byte-Pair Encoding works
    """
    print("Byte-Pair Encoding (BPE)")
    print("=" * 60)

    # Training corpus
    corpus = ["low", "lower", "lowest", "high", "higher", "highest"]

    print(f"Training corpus: {corpus}")
    print("\nStep 1: Start with characters")

    # Count character frequencies
    from collections import Counter
    import re

    # Initial vocabulary (characters)
    chars = Counter()
    for word in corpus:
        chars.update(list(word))

    print(f"Initial vocab: {sorted(chars.keys())}")

    print("\nStep 2: Find most frequent pair and merge")
    print("  'l' + 'o' appears often → merge to 'lo'")
    print("  'o' + 'w' appears often → merge to 'ow'")
    print("  'er' appears often → merge to 'er'")

    print("\nStep 3: Repeat until desired vocab size")

    # Final subwords might look like:
    final_vocab = ["low", "er", "est", "high", "lo", "ow"]
    print(f"\nFinal subwords: {final_vocab}")

    print("\nTokenizing new word: 'lowest'")
    print("  Greedy algorithm: 'low' + 'est' → ['low', 'est']")

bpe_demo()
```

### WordPiece (BERT)

Similar to BPE but uses likelihood instead of frequency.

```python

def wordpiece_demo():
    """
    WordPiece tokenization (used in BERT)
    """
    print("WordPiece Tokenization")
    print("=" * 60)

    print("""
    WordPiece is like BPE but smarter:

    Instead of just counting frequency,
    it merges pairs that maximize likelihood of training data.

    Special marker: '##' indicates continuation

    "unhappiness" → ["un", "##happiness"]
    "playing"     → ["play", "##ing"]
    """)

    examples = {
        "unhappiness": ["un", "##happiness"],
        "playing": ["play", "##ing"],
        "tokenization": ["token", "##ization"],
        "running": ["run", "##ning"]
    }

    for word, tokens in examples.items():
        print(f"  {word}: {tokens}")

wordpiece_demo()
```

### SentencePiece (Used in Llama, T5)

```python

def sentencepiece_demo():
    """
    SentencePiece tokenization
    """
    print("SentencePiece")
    print("=" * 60)

    print("""
    SentencePiece treats text as raw bytes/Unicode:

    Key features:
    • No need for pre-tokenization (works on raw text)
    • Handles any language
    • Includes whitespace as tokens
    • Used in Llama, T5, many modern models

    Example: "Hello world" might tokenize as:
    ["▁Hello", "▁world"]  (▁ represents space)
    """)

    print("\nAdvantage: Truly language-agnostic")
    print("Works for languages without spaces (Chinese, Japanese)")

sentencepiece_demo()
```

---

## Comparison Table

| Aspect              | Character         | Word             | Subword               |
| ------------------- | ----------------- | ---------------- | --------------------- |
| Vocabulary size     | Tiny (100-200)    | Huge (50k-500k+) | Medium (30k-50k)      |
| Sequence length     | Very long         | Short            | Medium                |
| Unknown words       | Never (can spell) | Frequent [UNK]   | Rare (can break down) |
| Language dependent  | No                | Yes              | Moderate              |
| Semantic meaning    | Lost              | Preserved        | Partial               |
| Training efficiency | Slow              | Fast             | Balanced              |
| Used in modern LLMs | No                | Rarely           | Yes (all)             |

---

## A Complete Tokenization Example

```python

def tokenization_comparison():
    """
    Compare all three approaches on real text
    """
    print("Tokenization Comparison")
    print("=" * 60)

    text = "The cats are playing unhappily"

    print(f"Text: '{text}'\n")

    # Character tokenization
    chars = list(text)
    print(f"Character ({len(chars)} tokens):")
    print(f"  {chars}\n")

    # Word tokenization (simplified)
    words = text.split()
    print(f"Word ({len(words)} tokens):")
    print(f"  {words}\n")

    # Subword tokenization (simulated BPE)
    subwords = ["The", "cat", "s", "are", "play", "ing", "un", "happy", "ly"]
    print(f"Subword ({len(subwords)} tokens):")
    print(f"  {subwords}")

    print("\nSubword advantages:")
    print("  • 'cats' → 'cat' + 's' (learns plural)")
    print("  • 'playing' → 'play' + 'ing' (learns -ing form)")
    print("  • 'unhappily' → 'un' + 'happy' + 'ly' (learns morphology)")

tokenization_comparison()
```

---

## Why This Matters for LLMs

### 1. All Modern LLMs Use Subword Tokenization

```python

def llm_tokenizers():
    """
    Tokenizers used by popular LLMs
    """
    print("Tokenizers in Popular LLMs")
    print("=" * 60)

    models = {
        "GPT-2": "BPE (Byte-Pair Encoding)",
        "GPT-3/4": "BPE",
        "BERT": "WordPiece",
        "RoBERTa": "BPE",
        "T5": "SentencePiece",
        "Llama": "SentencePiece (BPE variant)",
        "Claude": "Likely SentencePiece or BPE",
        "Gemini": "SentencePiece"
    }

    for model, tokenizer in models.items():
        print(f"  • {model}: {tokenizer}")

llm_tokenizers()
```

### 2. Vocabulary Size Matters

```python

def vocab_sizes():
    """
    Vocabulary sizes of popular models
    """
    print("Vocabulary Sizes of Popular Models")
    print("=" * 60)

    vocabs = {
        "GPT-2": "50,257",
        "GPT-3": "50,257",
        "BERT-base": "30,000",
        "BERT-large": "30,000",
        "T5": "32,000",
        "Llama 1": "32,000",
        "Llama 2/3": "32,000",
        "GPT-4": "~50,000 (estimated)"
    }

    for model, vocab in vocabs.items():
        print(f"  • {model}: {vocab} tokens")

vocab_sizes()
```

### 3. Tokenization Affects Model Behavior

```python

def tokenization_effects():
    """
    How tokenization impacts model performance
    """
    print("How Tokenization Affects LLMs")
    print("=" * 60)

    effects = {
        "Vocabulary size": "Larger vocab = more parameters, better coverage",
        "Subword units": "Affects how model handles morphology",
        "Language coverage": "Good tokenizer works for many languages",
        "Sequence length": "Shorter sequences = faster inference",
        "Special tokens": "[CLS], [SEP], etc. enable specific tasks"
    }

    for aspect, effect in effects.items():
        print(f"  • {aspect}: {effect}")

tokenization_effects()
```

### 4. The "Token" Economy

```python

def token_economy():
    """
    Practical implications of tokenization
    """
    print("The Token Economy")
    print("=" * 60)

    print("""
    Tokenization has real-world impact:

    1. Cost: APIs charge by token
    2. Context window: Limited by tokens, not words
    3. Speed: More tokens = slower processing
    4. Language bias: Some languages tokenize inefficiently

    Example (GPT-3 tokenizer):
    English: "Hello" → 1 token
    Spanish: "Hola" → 1 token
    Thai: "สวัสดี" → 4-5 tokens (costs more!)

    This creates bias toward languages with efficient tokenization.
    """)

token_economy()
```

---

## Special Tokens

LLMs use special tokens to understand structure.

```python

def special_tokens():
    """
    Special tokens in LLMs
    """
    print("Special Tokens")
    print("=" * 60)

    special = {
        "[CLS]": "Classification token (BERT)",
        "[SEP]": "Separator between sentences",
        "[MASK]": "Masked token for training (BERT)",
        "<s>": "Start of sequence (Llama, T5)",
        "</s>": "End of sequence",
        "<|endoftext|>": "GPT's end token",
        "[UNK]": "Unknown word (older models)",
        "[PAD]": "Padding for batches"
    }

    for token, purpose in special.items():
        print(f"  • {token}: {purpose}")

special_tokens()
```

---

## Tokenization Cheat Sheet

| Algorithm     | Used In      | How It Works                         |
| ------------- | ------------ | ------------------------------------ |
| BPE           | GPT, RoBERTa | Merge most frequent character pairs  |
| WordPiece     | BERT         | Merge pairs that maximize likelihood |
| SentencePiece | T5, Llama    | Works on raw text, includes spaces   |
| Unigram       | Some models  | Start with big vocab, prune unlikely |

---

## Quick Recap

• Character tokenization uses individual letters—tiny vocabulary but long sequences, can handle any text but inefficient and loses word meaning

• Word tokenization uses whole words—short sequences but huge vocabulary and can't handle unknown words, leading to information loss

• Subword tokenization (used in all modern LLMs) is the sweet spot—common words stay whole, rare words break into pieces, giving both efficiency and the ability to handle any text

---

## Mental Hook

> "Tokenization is like choosing between building with individual bricks (characters), pre-built houses (words), or a mix of bricks and pre-built sections (subwords)—the last gives you both flexibility and efficiency."
