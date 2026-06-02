# SentencePiece

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Tokenization**

---

### **1. Why this concept matters**

BPE and WordPiece assume text is tokenizable by splitting on spaces. But many languages (Chinese, Japanese, Thai) do not use spaces. Emojis, mathematical notation, and code also break space-based assumptions. SentencePiece solves this by treating input as raw Unicode bytes, learning subword tokens directly from the stream without relying on whitespace. It is the tokenization library behind T5, Llama, and many multilingual models. Understanding SentencePiece means understanding how LLMs handle languages without spaces—and why modern tokenizers are truly language-agnostic.

---

### **2. Core idea**

**SentencePiece is a tokenization library that treats input as a raw Unicode byte stream, learns subword tokens without requiring whitespace, and supports both BPE and Unigram algorithms in a language-agnostic framework.**

---

### **3. Concrete analogy**

Imagine you have a puzzle where pieces fit together based on shape, not on pre-printed spaces. Chinese text is like that—there are no spaces telling you where words start and end. Space-based tokenizers (BPE, WordPiece) would fail because they assume spaces are boundaries.

SentencePiece is like a puzzle solver that looks at the raw stream of characters—ignoring spaces entirely (or rather, treating spaces as just another character). It learns which sequences of characters (including spaces) are common and makes them tokens. "Hello world" might become ["Hello", "▁world"], where "▁" represents the space character. Now the model can learn that "▁world" (space+world) is a common pattern.

This approach works for any language, any script, any symbols. You do not need to pre-segment Chinese into words. SentencePiece learns the segmentation from data.

---

### **4. ASCII diagram**

```
SentencePiece architecture:

Raw input (any Unicode):
    "Hello world! 你好世界 😊"

Normalize (optional):
    "Hello world! 你好世界 😊"

Treat as byte stream (UTF-8):
    H e l l o space w o r l d ! space 你 好 世 界 space 😊

No space special-casing! Spaces are just byte 0x20.

Train subword tokenizer (BPE or Unigram):

Learn merges:
    "He" + "ll" → "Hell"
    "Hell" + "o" → "Hello"
    "▁" + "w" → "▁w"   (▁ represents space)
    "▁w" + "o" → "▁wo"
    ...

Final vocabulary includes tokens with spaces (represented as ▁):

    ["Hello", "▁world", "!", "▁", "你", "好", "世", "界", "😊", ...]


Encoding process (Unigram or BPE):

Input: "Hello world!"
Normalize (optional, e.g., NFKC normalization)
Encode using learned vocab:
    ["Hello", "▁world", "!"]

Decoding:
    Join tokens and replace "▁" with space
    "Hello" + " " + "world" + "!" = "Hello world!"


Comparison to space-based tokenization:

Space-based (BPE/WordPiece):
    Requires pre-tokenization by splitting on spaces.
    Fails on Chinese (no spaces).

SentencePiece:
    No pre-tokenization. Spaces are characters.
    Works on any language, any script.
```

---

### **5. Mathematical formulation**

**SentencePiece does not define a new tokenization algorithm—it provides a framework that can implement BPE or Unigram.**

**The key innovation: treat input as a raw byte sequence (UTF-8).**

Let the input be a sequence of Unicode characters. Encode as UTF-8 bytes: b₁, b₂, ..., bₙ ∈ {0...255}.

**Normalization options:**

- NFKC (Normalization Form KC): Normalizes Unicode (e.g., "ﬁ" → "fi")
- No normalization (preserve original characters)

**Space handling:**

Space character (U+0020) is encoded as byte 0x20. There is no special "space boundary" concept. SentencePiece can learn tokens that include spaces, such as "▁the".

**Vocabulary learning (Unigram algorithm, default in SentencePiece for T5, Llama):**

Goal: Find vocabulary V that maximizes likelihood of the training corpus.

For each possible subword x, compute loss if removed:

$$
\text{loss}(x) = \frac{\text{log-likelihood}(V) - \text{log-likelihood}(V \setminus \{x\})}{\text{freq}(x)}
$$

Remove tokens with smallest loss until target vocabulary size.

**Training process:**

1. Start with a seed vocabulary (all characters/bytes plus common substrings)
2. Repeat until target size:
   - Compute expected loss for each token
   - Remove token with smallest loss (Unigram) OR merge most frequent pair (BPE mode)
3. Optionally run EM (Expectation-Maximization) to refine token probabilities

**Encoding (segment new text):**

Given vocabulary V and learned token probabilities P(s), find the segmentation that maximizes:

$$
\prod_{s \in \text{seg}(x)} P(s)
$$

Using dynamic programming (Viterbi algorithm).

---

### **6. Worked example (step-by-step)**

#### **Step 1: Input text (Chinese + English)**

Input: "我爱NLP"

No spaces. Without SentencePiece, a space-based tokenizer would see this as a single token (or fallback to characters).

#### **Step 2: SentencePiece normalization (optional)**

Assume no normalization needed.

#### **Step 3: Treat as UTF-8 bytes**

Characters: "我" (U+6211), "爱" (U+7231), "N" (U+004E), "L" (U+004C), "P" (U+0050)

UTF-8 bytes:

- 我: 0xE6 0x88 0x91
- 爱: 0xE7 0x88 0xB1
- N: 0x4E
- L: 0x4C
- P: 0x50

Sequence: E6 88 91 E7 88 B1 4E 4C 50 (9 bytes)

#### **Step 4: Learn subword vocabulary (simplified)**

Assume training on large Chinese + English corpus.

Learned tokens might include:

- 我 (single character, frequent)
- 爱 (single character)
- NLP (merges N, L, P if "NLP" is common)
- 我爱 (merge of 我 and 爱, if common phrase)

After training, vocabulary contains a mix of byte-level, character-level, and multi-character tokens.

#### **Step 5: Encode "我爱NLP"**

Possible segmentation:

- Option A: ["我", "爱", "N", "L", "P"] (5 tokens)
- Option B: ["我", "爱", "NLP"] (3 tokens) ← if "NLP" in vocab
- Option C: ["我爱", "NLP"] (2 tokens) ← if "我爱" in vocab

Unigram chooses segmentation with highest probability product: P(我)×P(爱)×P(NLP) vs P(我爱)×P(NLP). Usually longer tokens have lower probability, but fewer tokens could have higher product. Depends on frequencies.

#### **Step 6: Decode back to text**

Join tokens: "我爱" + "NLP" = "我爱NLP" (correct)

Space handling: If tokens contain "▁", they become spaces during decoding.

---

### **7. How this appears inside neural networks and LLMs**

- **T5 (Google):** SentencePiece with Unigram algorithm. Vocabulary size 32,000. Handles multilingual text naturally.

- **Llama (Meta):** SentencePiece with BPE algorithm (not Unigram). Vocabulary size 32,000. Uses byte-level fallback (all Unicode characters representable).

- **ALBERT:** SentencePiece with Unigram.

- **XLM-R (Facebook multilingual BERT):** SentencePiece with Unigram, trained on 100 languages. Vocabulary 250,000.

- **Advantages over BPE/WordPiece:** Language-agnostic (no space assumption), handles any Unicode, reversible (no information loss), supports normalization (NFKC).

- **Normalization tradeoff:** NFKC normalizes "ﬁ" → "fi". Good for reducing vocabulary, but loses distinction (e.g., "ﬃ" → "ffi"). Some models (Llama) avoid normalization.

- **Byte fallback:** SentencePiece can represent any Unicode character by falling back to UTF-8 bytes. No <UNK> token needed. This is why modern LLMs can handle emojis, rare scripts, and even typos gracefully.

---

### **8. Brain-like connection (language-agnostic speech segmentation)**

Human infants learn to segment speech without relying on spaces (because speech has no spaces). They listen to raw acoustic streams, detect statistical regularities (frequent sound sequences), and hypothesize word boundaries. This is exactly what SentencePiece does with text: it treats the input as a raw stream (bytes), detects frequent subsequences, and learns a vocabulary. Multilingual children exposed to both Chinese (no spaces) and English (spaces) learn both without difficulty because their segmentation mechanism is not space-dependent. SentencePiece's language-agnostic design mirrors this cognitive flexibility.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "SentencePiece is a new tokenization algorithm that replaces BPE and WordPiece."

_Why it is wrong:_ SentencePiece is a **library** that implements existing algorithms (BPE, Unigram) in a language-agnostic way. The algorithms themselves are not new. The innovation is treating input as raw byte streams, removing the need for language-specific pre-tokenization (like splitting on spaces). The tokenization inside SentencePiece can be BPE (used by Llama) or Unigram (used by T5). "SentencePiece BPE" is still BPE, just applied to raw Unicode bytes without assuming whitespace boundaries. Do not confuse the library with the algorithm.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Not all languages use spaces. Chinese, Japanese, Thai    |
|  have no word boundaries. Space-based tokenizers fail.    |
|  SentencePiece fixes this by treating text as raw bytes,  |
|  spaces as just another character. It is language-agnostic|
|  and can handle any Unicode text, from English to emojis  |
|  to mathematical symbols. Llama, T5, and many multilingual|
|  models rely on SentencePiece. It is the tokenizer that   |
|  makes LLMs truly global.                                 |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a SentencePiece model trained on English text. You try to tokenize the Japanese sentence "こんにちは" (hello) with no spaces. The model has never seen Japanese characters during training.

**Question:** Does the tokenizer fail (output <UNK>)? If not, how does it represent the Japanese characters?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** The tokenizer does not fail. SentencePiece can represent any Unicode text using byte fallback. Each Japanese character is encoded as UTF-8 bytes (e.g., "こ" is 0xE3 0x81 0x93). SentencePiece's vocabulary includes individual bytes (0x00-0xFF) as fallback tokens. It will tokenize "こんにちは" as a sequence of bytes (3 bytes per character × 5 characters = 15 tokens).

This is inefficient (15 tokens vs 1 token if the model had seen Japanese), but it works without producing <UNK>. Over time, if the model is fine-tuned on Japanese, the tokenizer can learn merges for Japanese byte sequences, reducing the token count. This byte-fallback mechanism is why modern LLMs can handle any language (and even emojis) out of the box, regardless of training data.
