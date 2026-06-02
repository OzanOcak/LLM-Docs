# BPE (Byte Pair Encoding)

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Tokenization**

---

### **1. Why this concept matters**

Subword tokenization is the Goldilocks solution—but how do we actually build the vocabulary of subword pieces? Byte Pair Encoding (BPE) is the algorithm that does it. It starts with individual characters and repeatedly merges the most frequent adjacent pair of tokens. The result is a vocabulary of common substrings: "th," "ing," "tion," "un." Every modern LLM uses BPE or its close cousin WordPiece. Understanding BPE means understanding how your model decides what counts as a token—and why "unbreakable" might be ["un", "break", "able"] instead of 12 characters or 1 word.

---

### **2. Core idea**

**BPE builds a subword vocabulary by iteratively merging the most frequent adjacent pair of symbols in a training corpus, starting from individual characters or bytes, until reaching a target vocabulary size.**

---

### **3. Concrete analogy**

Imagine you are compressing a dictionary by finding common letter pairs.

You start with every letter as a separate symbol: a, b, c, ... space, period.

You scan the entire training text and count every adjacent pair: "th" appears 1 million times, "he" appears 900k, "in" appears 800k, etc. The most frequent pair is "th". You merge "t" and "h" into a new symbol "th". Now your vocabulary has one new token.

You repeat. Now "th" + "e" = "the" might be the most frequent pair. Merge them into "the". Keep going until you have, say, 50,000 tokens.

After merging, common words become single tokens ("the", "and", "ing"). Rare words get split into common fragments ("flibbertigibbet" might become ["fl", "ib", "bert", "ig", "ib", "bet"]). You never see <UNK> again because any word can be built from the available pieces.

---

### **4. ASCII diagram**

```
BPE algorithm step-by-step on a tiny corpus:

Initial corpus (words with frequencies):
    "low" (5 times), "lower" (2 times), "newest" (3 times), "widest" (2 times)

Step 0: Split into characters (plus end-of-word marker)
    l o w </w>   (5×)
    l o w e r </w> (2×)
    n e w e s t </w> (3×)
    w i d e s t </w> (2×)

Count pairs:
    "l o": 5+2=7
    "o w": 5+2=7
    "w e": 2+3=5
    "e r": 2
    "n e": 3
    "e w": 3
    ...

Most frequent: "l o" and "o w" (both 7). Merge "l o" → "lo"

Step 1:
    lo w </w> (5×)
    lo w e r </w> (2×)
    n e w e s t </w> (3×)
    w i d e s t </w> (2×)

Count pairs again:
    "lo w": 5+2=7
    "w e": 2+3=5
    "n e": 3
    ...

Merge "lo w" → "low"

Step 2:
    low </w> (5×)
    low e r </w> (2×)
    n e w e s t </w> (3×)
    w i d e s t </w> (2×)

Continue merging until target vocabulary size (e.g., 10 tokens).

Final vocabulary includes: characters + "lo", "low", "er", "est", "new", etc.


Visualization of merge tree:

    "unbreakable" tokenization:

         unbreakable
        /     |     \
      un    break   able
     / \    /  \    /  \
    u   n  br  eak ab  le
           / \   /\
          b   r e  ak
                  /\
                 a  k

Each node is a merge that occurred during BPE training.
```

---

### **5. Mathematical formulation**

**Let V be vocabulary (set of tokens). Initialization:**

$$
V_0 = \{\text{all characters/bytes in training corpus}\}
$$

**Let frequencies: For each adjacent pair (a,b) in corpus, count C(a,b).**

**Iteration (for t = 1 to target_size - |V₀|):**

1. Find the most frequent pair:

   $$
   (p^*, q^*) = \arg\max_{(p,q)} C(p,q) \quad \text{where } p,q \in V_{t-1}
   $$

2. Merge: create new token r = concatenation(p^_, q^_)

   $$
   V_t = V_{t-1} \cup \{r\}
   $$

3. Update corpus: replace every occurrence of (p^_, q^_) with r

4. Update frequencies C for pairs involving r

**Byte-level BPE (used in GPT-2, GPT-4):**

Start with 256 byte values (0-255) as initial vocabulary, not characters. This handles any text (UTF-8), any language, even binary files. No unknown tokens ever—anything is representable as bytes.

**Vocabulary size:**

Typical LLMs: 30,000 - 100,000 tokens. GPT-2: 50,257. GPT-4 (estimated): 100,000. Llama: 32,000.

**Compression ratio improvement:**

Let L_char = number of characters in text. After BPE with V tokens, tokens/char ≈ L_char / 3-4.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Tiny corpus**

Sentences:

- "cat cat cat" (3×)
- "cat sat mat" (1×)
- "mat mat" (2×)

#### **Step 2: Initial tokenization (characters + space)</w> marker**

Let's use space as explicit token " ".

Corpus as tokens: ["c","a","t"," ","c","a","t"," ","c","a","t","\n"] ×3, etc.

But simpler: assume we have frequencies of adjacent pairs.

#### **Step 3: Count initial pairs**

Assume training data yields these pair frequencies:

- "c a": 1000
- "a t": 1000
- "t ": 800
- " s": 400 (space + s from "sat")
- "a t": already counted
- "m a": 500
- "a t": 1000 again

Most frequent is "c a" (1000) and "a t" (1000). Pick "c a".

#### **Step 4: Merge "c a" → "ca"**

Now tokens include "ca". Update corpus: every "c"+"a" becomes "ca".

New pairs involving "ca":

- "ca t": 1000 (since "ca"+"t" appears where "c a t" was)
- other pairs unchanged.

#### **Step 5: Now most frequent might be "ca t" (1000). Merge → "cat"**

Vocabulary now has "cat". "cat" becomes a single token.

#### **Step 6: Continue**

Next might be "m a" (500) → "ma". Then "ma t" (500) → "mat".

Final vocabulary (small example):
['c','a','t',' ','m','s','\n', 'ca', 'cat', 'ma', 'mat', ...]

#### **Step 7: Tokenize new sentence**

"cat mat"

With learned merges:

- "c a t" → merge "c a" to "ca", then "ca t" to "cat" → ["cat"]
- " m a t" → " m" is space+"m" (not merged yet), "m a" → "ma", "ma t" → "mat" → [" ", "mat"]

Result: ["cat", " ", "mat"] (assuming space token exists separately)

Without BPE: ["c","a","t"," ","m","a","t"] (7 tokens). With BPE: 3 tokens.

---

### **7. How this appears inside neural networks and LLMs**

- **GPT series (GPT-2, GPT-3, GPT-4):** Byte-level BPE. Initial vocabulary is 256 bytes. Handles any UTF-8 text. GPT-2 has 50,257 tokens (256 bytes + 50,000 merges + special tokens). No unknown tokens ever.

- **RoBERTa:** Also uses byte-level BPE (same as GPT-2). No <UNK> token.

- **Llama (Meta):** Uses BPE (SentencePiece wrapper, but BPE algorithm). Vocabulary size 32,000.

- **Advantage over character/word:** BPE balances vocabulary size and sequence length. Rare words split into common pieces; common words remain whole.

- **Disadvantage (non-deterministic):** Different BPE implementations (merge order, handling of spaces, Unicode normalization) produce different vocabularies. GPT-2's BPE is not identical to Llama's BPE.

- **Byte fallback:** If a Unicode character is not in the training corpus, BPE at byte level splits it into multiple bytes (UTF-8 encoding). The model can still process it. Full Unicode coverage.

---

### **8. Brain-like connection (morphological acquisition)**

Children acquire subword units (morphemes) through statistical learning, just like BPE. Infants detect frequent sound patterns ("ma," "da") and merge them into proto-words. Later, they segment words into morphemes ("un-" + "break" + "-able") by noticing that certain substrings recur across different words. This is BPE in the brain: the child's "training corpus" is the speech they hear; the "merges" are the morphemes they acquire; the "vocabulary" is their mental lexicon. Children with language impairment often fail at this statistical learning, producing word-fragmentation errors similar to BPE with insufficient merges.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "BPE just splits on spaces and then does frequency merging. It's basically word tokenization with a fallback."

_Why it is wrong:_ BPE does not split on spaces first. It starts from characters (or bytes) and merges across spaces as well. Spaces are just another character (often represented as "\_" or "Ġ"). BPE can merge "th" across word boundaries if "th" appears frequently across words. The difference is crucial: BPE discovers that "ing" is a common suffix regardless of which word it appears in. Word tokenization would never split "walking" into "walk"+"ing" because it never sees "ing" as a separate token. BPE learns true subword units that generalize, not just word fragments. This is why BPE handles morphology ("run", "running", "runner") while word tokenization treats them as unrelated.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  BPE is the algorithm that turned infinite vocabulary     |
|  into a manageable 50,000 tokens. It learns which         |
|  character sequences are common—"ing," "tion," "un"—      |
|  and makes them atomic. Rare words become chains of       |
|  common pieces. No more <UNK>. No more vocabulary         |
|  bloat. Every modern LLM from GPT to Llama to BERT uses   |
|  BPE or its cousins. It is the quiet workhorse of         |
|  tokenization, the first step in turning text into        |
|  numbers that machines can learn from.                    |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You train BPE on an English corpus with a target vocabulary size of 10,000. You then train the same BPE on a German-English bilingual corpus with the same target vocabulary size.

**Question:** How will the vocabularies differ? Which one will handle the German compound word "Donaudampfschifffahrtsgesellschaftskapitän" more efficiently? Why?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** The bilingual vocabulary will contain German-specific merges (e.g., "sch", "fahrt", "gesellschaft") that the English-only BPE lacks. It will therefore split "Donaudampfschifffahrtsgesellschaftskapitän" into fewer tokens (e.g., 6-8 pieces vs 20+ pieces for English BPE). The English-only BPE would split it into many small pieces because it never learned German subword frequencies.

This highlights a key issue: BPE is corpus-dependent. A tokenizer trained only on English performs poorly on other languages. Modern multilingual models (mBERT, XLM-R) train BPE on hundreds of languages simultaneously, producing vocabularies that balance across languages. This is why Llama 3 was trained on multilingual data—its BPE can tokenize many languages efficiently.
