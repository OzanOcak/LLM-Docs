# BPE (Byte Pair Encoding)

## The Word Puzzle Analogy

Imagine you have a huge box of letter tiles and you're trying to build words efficiently. You notice that "th" appears constantly—in "the", "that", "this", "with". So you glue those two tiles together permanently. Then you notice "the" itself appears everywhere, so you glue those three together. Soon you have a collection of common letter groups that let you build words faster. That's Byte Pair Encoding (BPE): finding the most frequent character pairs in your training data and merging them into subwords.

In LLMs, BPE is the most popular tokenization algorithm. It's used by GPT models, RoBERTa, and many others. BPE strikes the perfect balance between character flexibility and word efficiency—common words become single tokens, while rare words break down into familiar pieces.

---

## What Is BPE?

### The Core Idea

BPE starts with individual characters and iteratively merges the most frequent pair of tokens.

```text

Start: characters
['t', 'h', 'e', ' ', 'c', 'a', 't', ' ', 's', 'a', 't']

Step 1: Find most frequent pair → 't' + 'h' appears often
Merge: add 'th' to vocabulary

Step 2: Find next most frequent → 'th' + 'e' appears often
Merge: add 'the' to vocabulary

Continue until desired vocabulary size
```

```python

def bpe_intro():
    """
    The basic concept of BPE
    """
    print("BPE: Building a Vocabulary from Frequent Pairs")
    print("=" * 60)

    print("""
    BPE algorithm:

    1. Start with all characters as tokens
    2. Count frequency of every adjacent pair
    3. Merge the most frequent pair into a new token
    4. Repeat until desired vocabulary size

    Result: Common words become single tokens
            Rare words break into common pieces
    """)

    print("\nAnalogy: Like creating shortcuts for common")
    print("letter combinations in shorthand writing.")

bpe_intro()
```

---

## BPE Step by Step

### Training Phase: Building the Vocabulary

```python

from collections import Counter
import re

def bpe_training_demo():
    """
    Demonstrate BPE training on a small corpus
    """
    print("BPE Training: Building the Vocabulary")
    print("=" * 60)

    # Training corpus
    corpus = ["low", "lower", "lowest", "high", "higher", "highest"]
    print(f"Training corpus: {corpus}")

    # Step 1: Initialize with characters
    print("\nStep 1: Start with character vocabulary")

    # Get all characters
    chars = set()
    for word in corpus:
        chars.update(list(word))

    vocab = {char: i for i, char in enumerate(sorted(chars))}
    print(f"Initial characters: {sorted(chars)}")

    # Step 2: Count character pairs in corpus
    print("\nStep 2: Count frequency of each adjacent pair")

    # Convert corpus to list of character sequences
    sequences = [list(word) for word in corpus]

    # Count pairs
    pair_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1

    print("Pair frequencies:")
    for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {pair[0]}+{pair[1]}: {count}")

    # Step 3: Merge most frequent pair
    print("\nStep 3: Merge most frequent pair")
    most_frequent = max(pair_counts, key=pair_counts.get)
    new_token = most_frequent[0] + most_frequent[1]
    print(f"  Merging '{most_frequent[0]}' + '{most_frequent[1]}' → '{new_token}'")

    # Step 4: Repeat
    print("\nStep 4: Repeat until desired vocab size")
    print("  Next merges might be:")
    print("  'o' + 'w' → 'ow'")
    print("  'l' + 'ow' → 'low'")
    print("  'e' + 'r' → 'er'")
    print("  'low' + 'er' → 'lower'")

bpe_training_demo()
```

### A Complete BPE Implementation

```python

import re
from collections import Counter, defaultdict

def bpe_complete_demo():
    """
    Complete BPE implementation on small corpus
    """
    print("Complete BPE Implementation")
    print("=" * 60)

    class BPE:
        def __init__(self, num_merges=10):
            self.num_merges = num_merges
            self.vocab = {}
            self.merges = {}

        def get_stats(self, words):
            """Count pair frequencies in words"""
            pairs = defaultdict(int)
            for word, freq in words.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i+1]] += freq
            return pairs

        def merge_vocab(self, pair, words):
            """Merge a pair in all words"""
            new_words = {}
            bigram = ' '.join(pair)
            replacement = ''.join(pair)

            for word, freq in words.items():
                new_word = word.replace(bigram, replacement)
                new_words[new_word] = freq
            return new_words

        def train(self, corpus):
            """Train BPE on corpus"""
            # Initialize with characters + end token
            words = {}
            for text in corpus:
                # Add space between characters and end token
                word = ' '.join(list(text)) + ' </w>'
                words[word] = words.get(word, 0) + 1

            print("Initial words:")
            for word, freq in words.items():
                print(f"  {word}: {freq}")

            # Perform merges
            for i in range(self.num_merges):
                pairs = self.get_stats(words)
                if not pairs:
                    break

                best = max(pairs, key=pairs.get)
                self.merges[best] = i

                print(f"\nMerge {i+1}: {'+'.join(best)} → {''.join(best)}")
                print(f"  frequency: {pairs[best]}")

                words = self.merge_vocab(best, words)

            self.vocab = set()
            for word in words:
                self.vocab.update(word.split())

            print(f"\nFinal vocabulary ({len(self.vocab)} tokens):")
            print(f"  {sorted(self.vocab)}")

        def tokenize(self, text):
            """Tokenize new text using learned merges"""
            # Start with characters
            tokens = list(text) + ['</w>']

            # Apply merges in order learned
            for (a, b), _ in sorted(self.merges.items(), key=lambda x: x[1]):
                i = 0
                while i < len(tokens) - 1:
                    if tokens[i] == a and tokens[i+1] == b:
                        tokens = tokens[:i] + [a+b] + tokens[i+2:]
                    else:
                        i += 1

            return tokens

    # Train BPE on small corpus
    corpus = ["low", "lower", "lowest", "high", "higher", "highest"]
    bpe = BPE(num_merges=8)
    bpe.train(corpus)

    # Test on new words
    print("\n" + "=" * 60)
    print("Testing on new words:")
    test_words = ["lowly", "highest", "lowest", "higher"]

    for word in test_words:
        tokens = bpe.tokenize(word)
        print(f"  '{word}' → {tokens}")

bpe_complete_demo()
```

---

## How BPE Tokenizes New Text

### Greedy Tokenization

```python

def bpe_tokenization_demo():
    """
    How BPE tokenizes new text
    """
    print("BPE Tokenization: Greedy Algorithm")
    print("=" * 60)

    # Learned vocabulary (from previous training)
    vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
             'er', 'es', 'est', 'high', 'low', 'lowest', '</w>']

    def tokenize(word, vocab):
        """Simple greedy tokenization"""
        word = word + '</w>'
        tokens = []
        i = 0

        while i < len(word):
            # Try to find longest matching token
            found = False
            for j in range(len(word), i, -1):
                candidate = word[i:j]
                if candidate in vocab:
                    tokens.append(candidate)
                    i = j
                    found = True
                    break
            if not found:
                # Shouldn't happen with proper BPE
                tokens.append(word[i])
                i += 1

        return tokens

    test_cases = ["lowest", "higher", "unhappiness"]

    for word in test_cases:
        tokens = tokenize(word, vocab)
        print(f"\n'{word}' → {tokens}")

bpe_tokenization_demo()
```

---

## Why BPE Works So Well

### Handling Unknown Words

```python

def unknown_words():
    """
    How BPE handles words it hasn't seen
    """
    print("BPE: No Unknown Words")
    print("=" * 60)

    print("""
    With word tokenization:
    Training: sees "cat", "cats"
    Inference: sees "caten" → [UNK] (lost!)

    With BPE:
    Training: learns "cat", "s", "en"
    Inference: "caten" → ["cat", "en"]  (understood!)
    """)

    examples = {
        "unhappiness": ["un", "happiness"] or ["un", "happy", "ness"],
        "tokenization": ["token", "ization"],
        "playing": ["play", "ing"],
        "stronger": ["strong", "er"],
        "strongest": ["strong", "est"]
    }

    print("\nCommon BPE segmentations:")
    for word, tokens in examples.items():
        print(f"  {word}: {tokens}")

unknown_words()
```

### Frequency Distribution

```text

Word frequency in text follows Zipf's law:
    • Few words appear VERY often ("the", "and", "of")
    • Many words appear rarely

BPE adapts perfectly:
    • Common words → single tokens (efficient)
    • Rare words → break into common pieces (flexible)
    • New words → can always be constructed
```

---

## BPE vs Other Tokenizers

| Aspect          | BPE          | WordPiece  | Unigram       |
| --------------- | ------------ | ---------- | ------------- |
| Merge criterion | Frequency    | Likelihood | Likelihood    |
| Direction       | Bottom-up    | Bottom-up  | Top-down      |
| Used in         | GPT, RoBERTa | BERT       | ALBERT, XLNet |
| Complexity      | Simple       | Moderate   | Complex       |

---

## Why This Matters for LLMs

### 1. BPE Powers Most Popular LLMs

```python

def bpe_models():
    """
    Which models use BPE
    """
    print("BPE in Popular LLMs")
    print("=" * 60)

    models = {
        "GPT-2": "BPE",
        "GPT-3": "BPE",
        "GPT-4": "BPE (likely)",
        "RoBERTa": "BPE",
        "Llama": "BPE (SentencePiece variant)",
        "Claude": "Likely BPE-based"
    }

    for model, tokenizer in models.items():
        print(f"  • {model}: {tokenizer}")

bpe_models()
```

### 2. Vocabulary Size Trade-offs

```python

def vocab_tradeoffs():
    """
    Choosing vocabulary size
    """
    print("Vocabulary Size Trade-offs")
    print("=" * 60)

    sizes = {
        "Small (10k)": {
            "pros": "Small model, fast",
            "cons": "Long sequences, rare words over-segmented"
        },
        "Medium (32-50k)": {
            "pros": "Good balance",
            "cons": "Sweet spot for most LLMs"
        },
        "Large (100k+)": {
            "pros": "Short sequences, efficient for common words",
            "cons": "Large embedding matrix, rare words still unknown"
        }
    }

    for size, info in sizes.items():
        print(f"\n{size}:")
        for key, value in info.items():
            print(f"  • {key}: {value}")

vocab_tradeoffs()
```

### 3. The Byte-Level Extension

```python

def byte_level_bpe():
    """
    Byte-level BPE (used in GPT-2/3/4)
    """
    print("Byte-Level BPE")
    print("=" * 60)

    print("""
    GPT-2/3/4 use byte-level BPE:

    • Works on raw bytes (UTF-8), not characters
    • Handles ANY text, any language, any emoji
    • 256 base tokens (one per byte)
    • No [UNK] tokens EVER

    Example: "café" → bytes: c a f é
    BPE might merge: "c" + "a" → "ca"
                     "ca" + "f" → "caf"
                     "é" stays as byte (since less common)
    """)

    print("\nThis is why GPT models handle emojis, math symbols,")
    print("and rare Unicode characters seamlessly.")

byte_level_bpe()
```

### 4. Practical Implications

```python

def practical_bpe():
    """
    Real-world BPE considerations
    """
    print("Practical BPE Considerations")
    print("=" * 60)

    print("""
    When using LLMs, BPE affects:

    1. Token counts: "Hello" = 1 token, "Hola" = 1 token
                     "สวัสดี" (Thai) = 5-8 tokens

    2. Cost: You're charged by tokens, not words

    3. Context window: Longer tokenized sequences
                       = less actual text fits

    4. Language bias: Languages with similar training data
                      tokenize more efficiently
    """)

    # Example with GPT-2 tokenizer (approximate)
    examples = [
        ("Hello", 1),
        ("Hello world", 2),
        ("Hello world! How are you?", 8),
        ("😊", 1),
        ("∫ f(x) dx", 5)
    ]

    print("\nApproximate GPT token counts:")
    for text, tokens in examples:
        print(f"  '{text}': ~{tokens} tokens")

practical_bpe()
```

---

## BPE Cheat Sheet

| Aspect       | Details                                         |
| ------------ | ----------------------------------------------- |
| Algorithm    | Iteratively merge most frequent character pairs |
| Start        | Individual characters + end token               |
| End          | Vocabulary of subwords (30k-50k)                |
| Tokenization | Greedy longest-match                            |
| Pros         | No UNK, efficient, learns morphology            |
| Used in      | GPT, RoBERTa, Llama (variants)                  |

---

## Quick Recap

• BPE starts with characters and iteratively merges the most frequent pairs—like gluing together common letter combinations, building a vocabulary of useful subwords from the bottom up

• Common words become single tokens, rare words break into familiar pieces—this means the model never sees an unknown word; it can always fall back to smaller components

• BPE powers most modern LLMs including GPT, RoBERTa, and Llama—its balance of efficiency and flexibility makes it the tokenization algorithm of choice for today's largest models

---

## Mental Hook

> "BPE is like creating a shorthand where common letter combinations get their own symbols—the more often a group appears, the more likely it becomes its own token, making the language both efficient and flexible."
