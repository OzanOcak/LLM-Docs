# WordPiece Tokenization

## The Crossword Puzzle Builder Analogy

Imagine you're designing crossword puzzles and you want a set of common word pieces to reuse. Instead of just counting how often letter pairs appear (like BPE), you ask: "Does merging these two pieces make the puzzles easier to solve overall?" You want pieces that not only appear frequently but also help you build more words. That's WordPiece: it merges pairs that maximize the likelihood of the training data—a smarter, probability-based approach.

In LLMs, WordPiece is the tokenization algorithm behind BERT and many other models. It's similar to BPE but uses a different merging criterion: instead of frequency, it merges pairs that increase the probability of the training data the most. This often results in more linguistically meaningful subwords.

---

## What Is WordPiece?

### The Core Idea

WordPiece starts with characters and iteratively merges pairs that maximize the likelihood of the training data.

```text

BPE: Merge most FREQUENT pair
    "th" appears 1000 times → merge

WordPiece: Merge pair that most INCREASES LIKELIHOOD
    "th" + "e" → "the" makes corpus MORE PROBABLE
    Measures: Does this merge help predict words better?
```

```python

def wordpiece_intro():
    """
    The basic concept of WordPiece
    """
    print("WordPiece: Probability-Based Merging")
    print("=" * 60)

    print("""
    WordPiece algorithm:

    1. Start with character vocabulary
    2. Build a language model on current tokens
    3. Try merging each possible pair
    4. Merge the pair that increases likelihood the MOST
    5. Repeat until desired vocabulary size

    Unlike BPE (frequency), WordPiece uses probability!
    """)

    print("\nAnalogy: Like choosing which LEGO pieces to")
    print("pre-assemble based on how useful they'll be")
    print("for building many different models.")

wordpiece_intro()
```

---

## BPE vs WordPiece: The Key Difference

### Comparison Side by Side

```python

def bpe_vs_wordpiece():
    """
    Compare BPE and WordPiece approaches
    """
    print("BPE vs WordPiece: The Key Difference")
    print("=" * 60)

    print("""
    BPE (frequency-based):
    ──────────────────────────────────
    Count how often 't' + 'h' appears → 1000
    Count how often 'x' + 'z' appears → 1
    Merge 't'+'h' because it's more frequent

    WordPiece (probability-based):
    ──────────────────────────────────
    Current vocabulary: ['t', 'h', 'e']
    Current model probability: 0.3

    Try merging 't'+'h' → new probability: 0.31 (+0.01)
    Try merging 'h'+'e' → new probability: 0.35 (+0.05)
    Merge 'h'+'e' because it improves the model more!
    """)

    print("\nThe probability gain matters more than raw frequency.")
    print("WordPiece creates more useful subwords.")

bpe_vs_wordpiece()
```

### Why Likelihood Matters

```python

def likelihood_intuition():
    """
    Intuition behind likelihood-based merging
    """
    print("Why Likelihood-Based Merging Is Smarter")
    print("=" * 60)

    print("""
    Consider words: "the", "then", "there", "they"

    BPE would see 't'+'h' appears in ALL these words
    → merge 't'+'h' first

    WordPiece considers:
    - Merging 't'+'h' → get "th" (helps a little)
    - Merging 'th'+'e' → get "the" (helps a LOT)
    - "the" is a very common word on its own

    WordPiece might merge 'th'+'e' first because
    it makes the overall model predict text better!
    """)

likelihood_intuition()
```

---

## How WordPiece Works Step by Step

### Training Phase

```python

import math
from collections import defaultdict, Counter

def wordpiece_training_demo():
    """
    Demonstrate WordPiece training concept
    """
    print("WordPiece Training: Step by Step")
    print("=" * 60)

    # Training corpus
    corpus = ["the", "then", "there", "they", "that", "this"]
    print(f"Training corpus: {corpus}")

    # Step 1: Start with characters
    print("\nStep 1: Initialize with characters")
    chars = set()
    for word in corpus:
        chars.update(list(word))

    # Add special tokens
    vocab = sorted(chars)
    print(f"Initial vocabulary: {vocab}")

    # Step 2: Calculate initial likelihood
    print("\nStep 2: Calculate initial corpus likelihood")
    print("  Count frequencies of each character")
    print("  Compute probability of corpus")

    # Simplified likelihood calculation
    def calculate_likelihood(vocab, corpus):
        """Simplified likelihood score"""
        # Count tokens needed
        total_tokens = 0
        for word in corpus:
            # Simulate tokenization
            tokens = 0
            remaining = word
            while remaining:
                # Find longest match (simplified)
                found = False
                for token in sorted(vocab, key=len, reverse=True):
                    if remaining.startswith(token):
                        tokens += 1
                        remaining = remaining[len(token):]
                        found = True
                        break
                if not found:
                    tokens += 1
                    remaining = remaining[1:]
            total_tokens += tokens

        # Lower tokens = higher likelihood (simplified)
        return 1.0 / total_tokens

    initial_likelihood = calculate_likelihood(vocab, corpus)
    print(f"  Initial likelihood score: {initial_likelihood:.4f}")

    # Step 3: Try all possible merges
    print("\nStep 3: Try merging each possible pair")

    candidates = []
    for i in range(len(vocab)):
        for j in range(i, len(vocab)):
            pair = vocab[i] + vocab[j]
            new_vocab = vocab + [pair]
            likelihood = calculate_likelihood(new_vocab, corpus)
            gain = likelihood - initial_likelihood
            candidates.append((pair, gain))

    # Show top candidates
    candidates.sort(key=lambda x: -x[1])
    print("Top candidate merges:")
    for pair, gain in candidates[:3]:
        print(f"  Merge '{pair}': likelihood gain = {gain:.6f}")

    print("\nStep 4: Merge best pair and repeat")
    print("  Continue until desired vocabulary size")

wordpiece_training_demo()
```

### The ## Notation

WordPiece uses a special notation to mark subwords that are continuations.

```python

def hashhash_notation():
    """
    The ## notation in WordPiece
    """
    print("The ## Notation: Marking Continuations")
    print("=" * 60)

    print("""
    WordPiece uses ## to mark subwords that aren't word-initial:

    "unhappiness" → ["un", "##happiness"]
    "playing"     → ["play", "##ing"]
    "tokenization" → ["token", "##ization"]

    This helps the model know:
    • "un" can start a word
    • "##happiness" must follow something
    • Together they form "unhappiness"
    """)

    examples = {
        "unhappiness": ["un", "##happiness"],
        "unhappy": ["un", "##happy"],
        "happily": ["happy", "##ly"],
        "playing": ["play", "##ing"],
        "played": ["play", "##ed"],
        "plays": ["play", "##s"]
    }

    print("\nCommon WordPiece segmentations:")
    for word, tokens in examples.items():
        print(f"  {word}: {tokens}")

hashhash_notation()
```

---

## A Simplified WordPiece Implementation

```python

import math
from collections import defaultdict

def wordpiece_implementation_demo():
    """
    Simplified WordPiece implementation
    """
    print("Simplified WordPiece Implementation")
    print("=" * 60)

    class WordPiece:
        def __init__(self, vocab_size=20):
            self.vocab_size = vocab_size
            self.vocab = set()
            self.scores = {}

        def train(self, corpus):
            """Train WordPiece on corpus"""
            # Start with characters
            chars = set()
            for word in corpus:
                chars.update(list(word))
            self.vocab = {c for c in chars}

            print(f"Initial vocabulary ({len(self.vocab)} chars):")
            print(f"  {sorted(self.vocab)}")

            # Add end-of-word marker
            self.vocab.add('</w>')

            # Prepare corpus with word boundaries
            words = [list(word) + ['</w>'] for word in corpus]

            # Iteratively add new tokens
            iteration = 0
            while len(self.vocab) < self.vocab_size:
                iteration += 1

                # Count all possible pairs
                pair_counts = defaultdict(int)
                for word_tokens in words:
                    for i in range(len(word_tokens) - 1):
                        pair = (word_tokens[i], word_tokens[i+1])
                        pair_counts[pair] += 1

                if not pair_counts:
                    break

                # Calculate "score" for each pair
                # In real WordPiece, this would be likelihood gain
                # Here we use a simplified score based on frequency
                best_pair = max(pair_counts.items(), key=lambda x: x[1])
                pair, count = best_pair

                new_token = pair[0] + pair[1]
                if pair[1] == '</w>':
                    new_token = pair[0] + pair[1]

                self.vocab.add(new_token)

                print(f"\nIteration {iteration}:")
                print(f"  Adding '{new_token}' (from '{pair[0]}'+'{pair[1]}', count={count})")

                # Update words with new token
                new_words = []
                for word_tokens in words:
                    new_word = []
                    i = 0
                    while i < len(word_tokens):
                        if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == pair:
                            new_word.append(new_token)
                            i += 2
                        else:
                            new_word.append(word_tokens[i])
                            i += 1
                    new_words.append(new_word)
                words = new_words

            print(f"\nFinal vocabulary size: {len(self.vocab)}")
            print("Sample tokens:", sorted(self.vocab)[:10])

        def tokenize(self, text):
            """Tokenize text using longest match"""
            # Add end marker
            text = text + '</w>'
            tokens = []
            i = 0

            while i < len(text):
                # Find longest matching token
                found = False
                for j in range(len(text), i, -1):
                    candidate = text[i:j]
                    if candidate in self.vocab:
                        tokens.append(candidate)
                        i = j
                        found = True
                        break
                if not found:
                    # Shouldn't happen with proper training
                    tokens.append(text[i])
                    i += 1

            return tokens

    # Train on simple corpus
    wp = WordPiece(vocab_size=25)
    corpus = ["the", "then", "there", "they", "that", "this", "with", "without"]
    wp.train(corpus)

    # Test tokenization
    print("\n" + "=" * 60)
    print("Testing tokenization:")
    test_words = ["the", "they", "without", "there"]
    for word in test_words:
        tokens = wp.tokenize(word)
        print(f"  '{word}' → {tokens}")

wordpiece_implementation_demo()
```

---

## WordPiece in BERT

### How BERT Uses WordPiece

```python

def bert_wordpiece():
    """
    WordPiece in BERT
    """
    print("WordPiece in BERT")
    print("=" * 60)

    print("""
    BERT uses WordPiece with:
    • Vocabulary size: 30,000 tokens
    • ## notation for continuations
    • Special tokens: [CLS], [SEP], [MASK]

    Example BERT tokenization:

    Input: "I like natural language processing"

    WordPiece output:
    ["i", "like", "natural", "language", "process", "##ing"]

    Then add special tokens:
    ["[CLS]", "i", "like", "natural", "language", "process", "##ing", "[SEP]"]
    """)

    # More examples
    examples = [
        "unhappiness",
        "tokenization",
        "playing",
        "higher",
        "highest"
    ]

    print("\nTypical BERT WordPiece segmentations:")
    for word in examples:
        # Simulated BERT tokenization
        if word == "unhappiness":
            tokens = ["un", "##happiness"]
        elif word == "tokenization":
            tokens = ["token", "##ization"]
        elif word == "playing":
            tokens = ["play", "##ing"]
        elif word == "higher":
            tokens = ["high", "##er"]
        elif word == "highest":
            tokens = ["high", "##est"]

        print(f"  {word}: {tokens}")

bert_wordpiece()
```

---

## WordPiece vs BPE Comparison

| Aspect          | BPE                      | WordPiece                            |
| --------------- | ------------------------ | ------------------------------------ |
| Merge criterion | Frequency                | Likelihood gain                      |
| Goal            | Minimize sequence length | Maximize data probability            |
| Training        | Simple counting          | Requires language model              |
| Complexity      | Lower                    | Higher                               |
| Result          | Good subwords            | Often more linguistically meaningful |
| Used in         | GPT, RoBERTa             | BERT, DistilBERT                     |

### Example: Different Merging Decisions

```text

Corpus: "the", "then", "there", "they", "that", "this"

BPE merges (by frequency):
    't'+'h' → 'th' (appears in all words)
    'th'+'e' → 'the' (appears in many)
    'th'+'a' → 'tha' (for "that")

WordPiece merges (by likelihood gain):
    'th'+'e' → 'the' (biggest gain, very common word)
    't'+'h' → 'th' (also helpful)
    'the'+'n' → 'then' (common word)
```

---

## Why This Matters for LLMs

### 1. WordPiece Powers BERT and Its Variants

```python

def wordpiece_models():
    """
    Models using WordPiece
    """
    print("Models Using WordPiece")
    print("=" * 60)

    models = {
        "BERT": "Original WordPiece (30k vocab)",
        "DistilBERT": "Same as BERT",
        "RoBERTa": "Uses BPE instead (different!)",
        "ALBERT": "WordPiece",
        "ELECTRA": "WordPiece"
    }

    for model, tokenizer in models.items():
        print(f"  • {model}: {tokenizer}")

wordpiece_models()
```

### 2. The ## Notation Helps Understanding

```python

def hashhash_importance():
    """
    Why ## notation matters
    """
    print("The Importance of ## Notation")
    print("=" * 60)

    print("""
    Without ##: ["play", "ing"] could mean "playing" OR "play ing" (two words)

    With ##: ["play", "##ing"] clearly means one word "playing"
            ["play", "ing"] would mean two words "play ing"

    This helps the model understand:
    • Word boundaries
    • Morphological structure
    • Prefixes vs suffixes
    """)

hashhash_importance()
```

### 3. Special Tokens in BERT

```python

def special_tokens_bert():
    """
    Special tokens in BERT's WordPiece
    """
    print("Special Tokens in BERT's WordPiece")
    print("=" * 60)

    special = {
        "[CLS]": "Classification token (first token)",
        "[SEP]": "Separator between sentences",
        "[MASK]": "Masked token for training",
        "[PAD]": "Padding token",
        "[UNK]": "Unknown token (rare with WordPiece)"
    }

    for token, purpose in special.items():
        print(f"  • {token}: {purpose}")

special_tokens_bert()
```

---

## WordPiece Cheat Sheet

| Aspect         | Details                              |
| -------------- | ------------------------------------ |
| Algorithm      | Merge pairs that maximize likelihood |
| Notation       | ## for continuations                 |
| Vocab size     | 30,000 (BERT) to 50,000              |
| Special tokens | [CLS], [SEP], [MASK], [PAD]          |
| Pros           | Linguistically meaningful subwords   |
| Cons           | More complex training than BPE       |
| Used in        | BERT family, ALBERT, ELECTRA         |

---

## Quick Recap

• WordPiece merges pairs that maximize the likelihood of the training data—like choosing which LEGO pieces to pre-assemble based on how much they'll help build many different models

• It uses ## notation to mark subwords that are continuations—helping the model understand word boundaries and morphological structure

• WordPiece is the tokenization algorithm behind BERT and its variants—its probability-based approach often creates more linguistically meaningful subwords than frequency-based BPE

---

## Mental Hook

> "WordPiece is like a master LEGO builder who doesn't just count which pieces connect most often, but simulates how each new pre-assembled piece will help build the entire collection of models more efficiently."
