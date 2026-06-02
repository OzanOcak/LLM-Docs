# Word vs subword vs character tokenization

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Tokenization**

---

### **1. Why this concept matters**

Language models do not see letters or words. They see numbers—integers mapped to tokens. But what counts as a token? A whole word? A single character? Something in between? This decision, seemingly low-level, profoundly affects everything: vocabulary size, handling of unknown words, misspellings, and multilingual support. Word tokenization is intuitive but fails on rare words. Character tokenization handles everything but creates long sequences. Subword tokenization (BPE, WordPiece) is the Goldilocks solution—used by every modern LLM. Understanding tokenization means understanding the very first step that turns raw text into numbers a model can process.

---

### **2. Core idea**

**Tokenization splits text into atomic units: word tokenization uses whole words (large vocabulary, many unknowns), character tokenization uses individual letters (small vocabulary, long sequences), and subword tokenization balances both by splitting rare words into common fragments.**

---

### **3. Concrete analogy**

Imagine you are packing a suitcase for a trip.

- **Word tokenization:** You pack whole outfits—complete shirts, pants, dresses. Efficient if you have exactly the right outfits. But if you encounter an unfamiliar clothing item (a "kilt"), you cannot pack it. You are stuck. Vocabulary size = every word you might ever need (millions).

- **Character tokenization:** You pack individual threads. You can assemble any garment from scratch, but packing 100 threads for one shirt is ridiculous. Your suitcase overflows. Sequence length explodes.

- **Subword tokenization:** You pack LEGO bricks: "un-", "-break-", "-able." You can build "unbreakable" by snapping bricks together. Common bricks are stored; rare words are built. This is what modern LLMs do. The vocabulary size is ~50,000 "bricks"—enough to build most English words efficiently.

---

### **4. ASCII diagram**

```
Three tokenization levels on "unbreakable":

Word tokenization:
    ["unbreakable"] → 1 token

Character tokenization:
    ["u", "n", "b", "r", "e", "a", "k", "a", "b", "l", "e"] → 11 tokens

Subword tokenization (BPE):
    ["un", "break", "able"] → 3 tokens


How subword handles rare vs common words:

Common word "the": stays as ["the"] → 1 token

Rare word "flibbertigibbet":
    Might split as ["flib", "bert", "igib", "bet"] → 4 tokens
    (Still more efficient than 16 character tokens)


Vocabulary coverage:

Word-level (50K words):    covers ~95% of English text
Subword (50K pieces):       covers 100% (can build any word)
Character (26+):             covers 100% (but sequence length 5-10× longer)


Example: "I love NLP!"

Word:     ["I", "love", "NLP", "!"] → 4 tokens
Subword:  ["I", "love", "NL", "P", "!"] → 5 tokens (if "NLP" not in vocab)
Character:["I", " ", "l", "o", "v", "e", " ", "N", "L", "P", "!"] → 11 tokens
```

---

### **5. Mathematical formulation**

**Vocabulary size V and sequence length L tradeoff:**

For a text of N characters:

- Word tokenization: V_word ~ 500,000 (English), L_word ~ N / 5 (words)
- Character tokenization: V_char ~ 128 (ASCII), L_char = N (characters)
- Subword tokenization: V_subword ~ 50,000, L_subword ~ N / 3-4

**Compression ratio:**

$$
\text{Ratio} = \frac{L_{\text{char}}}{L_{\text{subword}}} \approx 3 \text{ to } 5
$$

**Out-of-vocabulary (OOV) rate (word-level):**

$$
\text{OOV} = \frac{\text{Number of unseen words in test}}{\text{Total test words}}
$$

For a 50K word vocabulary on general text, OOV ≈ 3-5%.

**Subword coverage:**

Subword tokenization has OOV = 0% by definition—any text can be represented as a sequence of subword pieces.

**Maximum sequence length for transformer (with fixed context window C):**

Maximum characters the model can process = C × (average characters per token)

For C=2048, subword (3 chars/token) → ~6000 chars. Character (1 char/token) → 2048 chars only.

---

### **6. Worked example (step-by-step)**

#### **Step 1: The sentence**

"Tokenization is cool!"

#### **Step 2: Word tokenization**

Split by spaces and punctuation:
["Tokenization", "is", "cool", "!"]

If vocabulary contains these words: 4 tokens. If "Tokenization" is rare, it becomes <UNK> (unknown token). Model loses information.

Vocabulary size needed: 500,000+ words to cover most English.

#### **Step 3: Character tokenization**

["T", "o", "k", "e", "n", "i", "z", "a", "t", "i", "o", "n", " ", "i", "s", " ", "c", "o", "o", "l", "!"]

21 tokens. Sequence is 5× longer than word tokens. Context window of 2048 tokens can only see ~2048 characters (a short paragraph). Model loses long-range context.

Vocabulary size: 128 (ASCII) or smaller. Compact but inefficient.

#### **Step 4: Subword tokenization (BPE, typical vocabulary ~50K)**

Assume learned merges:
"Token" is common → "Token"
"ization" is common suffix → "ization"
"is" is common → "is"
"cool" is common → "cool"
"!" is punctuation → "!"

But "Tokenization" may split as ["Token", "ization"] (2 tokens)

Result: ["Token", "ization", "is", "cool", "!"] → 5 tokens

Compromise: 5 tokens vs word's 4 (slightly longer), vs character's 21 (much shorter). No unknown tokens. "Tokenization" handled even if not in vocabulary.

#### **Step 5: Compare across languages**

English word: "unhappiness" → subword: ["un", "happiness"] (2 tokens)
German compound word: "Donaudampfschifffahrtsgesellschaftskapitän" (Danube steamship company captain)

Word tokenization: 1 token (if in vocab) or <UNK>
Character: 57 tokens!
Subword: splits into known pieces like ["Donau", "dampf", "schiff", "fahrt", "gesellschafts", "kapitän"] → ~6 tokens

Subword handles long compound words gracefully.

---

### **7. How this appears inside neural networks and LLMs**

- **Word tokenization (pre-2018):** Used in early RNN language models. Required huge vocabularies (200K-500K) and still had OOV issues.

- **Character tokenization (rare in LLMs):** Used in some specialized models (CANINE, ByT5). Context window inefficiency limits adoption.

- **Subword tokenization (all modern LLMs):** GPT uses BPE. BERT uses WordPiece. T5 uses SentencePiece (unicode-aware). All have vocab size ~30K-100K.

- **Bytes as tokens (GPT-4, Llama 3):** Treat each byte (0-255) as a token. Handles any text, any language, even binary data. Vocabulary size 256. But long sequences—mitigated by efficient architectures and large context windows (128K tokens).

- **Multilingual support:** Subword tokenization works across languages because common substrings (e.g., "tion", "ung", "ing") appear in many languages. Character-level also works but inefficiently.

- **Digits and numbers:** Subword tokenization often splits numbers into individual digits ("123" → ["1","2","3"]). This is intentional—helps arithmetic reasoning. Some models (GPT-4) handle numbers specially.

- **Special tokens:** All tokenizers add special tokens: [CLS] (classification), [SEP] (separator), <|im_start|> (chat roles), <|endoftext|> (document boundary).

---

### **8. Brain-like connection (morphemes and lexical access)**

The human brain processes language at multiple levels simultaneously. Morphemes (smallest meaning-carrying units: "un-", "-break-", "-able") are the cognitive equivalent of subword tokens. Children learn common morphemes and combine them to understand novel words ("un-happy-ness"). This is subword tokenization. Patients with certain aphasias can understand morphemes but cannot combine them correctly, or vice versa—suggesting separable neural representations. The brain's mental lexicon organizes words by morphemic structure, exactly as subword tokenization organizes vocabulary by common substrings. Evolution arrived at subword efficiency because storing every possible word is impossible; storing morphemes and rules to combine them is optimal.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Subword tokenization is just a preprocessing detail. It does not affect model quality much."

_Why it is wrong:_ Tokenization dramatically affects model behavior. BPE merges based on frequency—common sequences like "ing" become tokens, while rare sequences split. This creates biases: misspellings ("ingg") split differently, harming robustness. Numbers split into digits impair arithmetic (model sees "12" as ["1","2"] not "12"). Multilingual fairness: languages with more common substrings (English, German) get shorter sequences than morphologically rich languages (Turkish, Finnish), penalizing them. Changing tokenization changes the inductive bias. The choice of tokenizer is a model design decision, not a trivial detail.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Every LLM begins with tokenization. It is the lens       |
|  through which the model sees text. Word tokens are       |
|  too rigid; character tokens are too verbose. Subword     |
|  strikes the balance—common words are single tokens,      |
|  rare words are built from fragments. This tiny           |
|  engineering choice enables 50K vocabularies to cover     |
|  the infinite variety of human language. Tokenization     |
|  is not glamorous, but without it, there is no LLM.       |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

A model uses subword tokenization with vocabulary of 50,000 pieces. It sees the training sentence "The flibbertigibbet jumped." The word "flibbertigibbet" is not in the vocabulary. It is split into ["flib", "bert", "igi", "bbet"].

**Question:** What would happen with word-level tokenization (50K vocab) on this sentence? What about character-level?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Word-level tokenization: "flibbertigibbet" is out-of-vocabulary (OOV). The tokenizer would replace it with a special token like <UNK>. The model loses all information about that word—cannot tell if it's a name, a nonsense word, or a rare noun. The sentence becomes "The <UNK> jumped." Meaning is lost.

Character-level tokenization: Would produce ~18 tokens for "flibbertigibbet" (one per letter). The model would see the full spelling and could theoretically learn patterns, but the sequence length is much longer. For a 2048-token context window, this sentence consumes 18 tokens instead of 1 (word-level) or ~4 (subword). Over many rare words, the effective context window shrinks dramatically. This is why LLMs use subword, not character, tokenization.
