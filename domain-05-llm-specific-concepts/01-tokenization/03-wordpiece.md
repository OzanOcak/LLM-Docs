# WordPiece

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Tokenization**

---

### **1. Why this concept matters**

BPE is not the only subword tokenization algorithm. WordPiece, developed by Google for BERT, takes a different approach: instead of merging the most frequent pair, it merges the pair that maximizes the likelihood of the training data. This subtle difference produces vocabularies that are often more efficient for downstream tasks. WordPiece is the algorithm behind BERT, DistilBERT, and many other encoder-only models. Understanding WordPiece means understanding the design choices that made BERT the dominant pre-trained model of its era—and why BPE eventually won for decoder-only models.

---

### **2. Core idea**

**WordPiece builds a subword vocabulary by iteratively merging the pair of tokens that most increases the likelihood of the training corpus under a unigram language model, rather than simply the most frequent pair.**

---

### **3. Concrete analogy**

Imagine two chefs trying to decide which ingredients to pre-mix. BPE says: "Combine whatever two ingredients appear together most often, regardless of whether it makes the dish tastier." This is fast and simple.

WordPiece says: "Combine the two ingredients that, when merged, would most increase the overall deliciousness of all dishes." It considers not just frequency, but how well the merge improves the representation of the entire corpus.

In practice, WordPiece tends to create more linguistically meaningful subwords because it optimizes a probabilistic objective. For example, it prefers to merge "un" and "break" into "unbreak" if that helps model the distribution of words better. BPE might merge "th" and "e" into "the" because they are frequent, but WordPiece might prioritize different merges that reduce overall token count more efficiently.

---

### **4. ASCII diagram**

```
WordPiece vs BPE merge criterion:

BPE: merge the most frequent adjacent pair
    (frequency-based)

    Corpus: "low" (5x), "lower" (2x), "newest" (3x)

    Pairs: "l o":7, "o w":7, "w e":5, "n e":3, ...
    Merge "l o" → "lo" (highest frequency)


WordPiece: merge the pair that maximizes likelihood
    (likelihood-based)

    For each candidate pair (a,b), compute:

    Likelihood_gain = log P(corpus | new_vocab) - log P(corpus | old_vocab)

    Where P(corpus) = Π (frequency of each word / total words)
    using current vocabulary segmentation.

    Merge the pair with largest gain.


Why likelihood matters:

WordPiece accounts for:
- How often the pair appears
- How often the individual tokens appear
- How segmentation affects other words

Result: WordPiece tends to create more
information-dense subword tokens.


Tokenization visualization (BERT using WordPiece):

Input: "Tokenization is fascinating"

WordPiece segmentation:
    ["Token", "##ization", "is", "fascinating"]

Note: "##ization" indicates it's a continuation
of a previous token (no space before). This is unique to WordPiece.
```

---

### **5. Mathematical formulation**

**Let the corpus be a sequence of words w₁...wₙ. Each word w is segmented into a sequence of subword tokens s₁...s_k.**

**WordPiece maximizes the likelihood of the corpus:**

$$
L = \prod_{w \in \text{corpus}} \prod_{s \in \text{seg}(w)} P(s)
$$

(Unicode: L = Π over words Π over subwords in segmentation P(s))

Where P(s) is the frequency of subword s in the corpus (count / total tokens).

**Greedy merging algorithm:**

Start with a vocabulary of characters (plus special tokens for word boundaries).

At each iteration, for each adjacent pair (a,b) in the current vocabulary:

1. Temporarily merge (a,b) into a new token ab
2. Re-segment the corpus using the new vocabulary (with a greedy longest-match tokenization)
3. Compute the increase in log-likelihood:

$$
\Delta \log L = \sum_{w} \log \prod_{s \in \text{seg}_{\text{new}}(w)} P_{\text{new}}(s) - \sum_{w} \log \prod_{s \in \text{seg}_{\text{old}}(w)} P_{\text{old}}(s)
$$

4. Choose the pair with largest Δlog L

**Segmentation algorithm (greedy longest-match, aka maximum matching):**

Given a word, scan from left to right, take the longest subword token that exists in the vocabulary.

Example: "fascinating" with vocab {"fas", "f", "a", "scinating", ...}

- Start: longest token starting at 'f' that exists? "fas" exists, length 3.
- Then "cinating"? No, maybe "ci"? This is simplified; WordPiece uses "##" to indicate subword continuity.

**WordPiece's "##" convention:**

Tokens without "##" start a new word. Tokens with "##" are glued to previous token without space.

Example: "tokenization" → ["token", "##ization"]

During detokenization: "token" + "ization" = "tokenization" (no space).

---

### **6. Worked example (step-by-step)**

#### **Step 1: Initialization**

Start with character vocabulary: a, b, c, ..., z, space, punctuation. Plus special tokens [UNK], [CLS], [SEP], [MASK].

#### **Step 2: Training corpus**

Assume a tiny corpus: "hello world" (1 time), "hell" (1 time), "help" (1 time)

Initial segmentation (space is token, but let's simplify: no space in examples):

Word "hello": could be segmented as "h", "e", "l", "l", "o" (5 tokens)

#### **Step 3: Candidate merges and likelihood**

WordPiece evaluates each possible character pair:

Candidate "he":

- Merge "h"+"e" into "he"
- Re-segment "hello" as "he", "l", "l", "o" (4 tokens)
- Re-segment "hell" as "he", "l", "l" (3 tokens)
- "help" unchanged ("h","e","l","p" or maybe now "he","l","p"?)
- Compute new token frequencies and likelihood

Candidate "el":

- Merge "e"+"l" into "el"
- "hello" → "h","el","l","o"
- "hell" → "h","el","l"
- ...

Candidate "ll":

- Merge "l"+"l" into "ll"
- "hello" → "h","e","ll","o"
- "hell" → "h","e","ll"
- "help" unchanged (no "ll")

#### **Step 4: Choose best merge**

Suppose "ll" gives the largest likelihood increase because it appears in both "hello" and "hell", and merges reduce token count significantly.

Vocabulary now includes "ll".

#### **Step 5: Next iteration**

Now candidate "he": After "ll" merge, "hello" is ["h","e","ll","o"]. "he" still appears in "hell" as ["h","e","ll"]? Actually "hell" becomes ["h","e","ll"]—yes, "he" pair exists. Possibly merge "he" next.

Continue until reaching target vocabulary size (e.g., 30,000 tokens).

#### **Step 6: Final vocabulary for BERT**

BERT's WordPiece vocabulary has 30,000 tokens. Examples:

- Whole words: "the", "hello", "world"
- Common subwords: "##ing", "##ed", "##ly", "##tion"
- Characters: "a", "b", "c", ... (for fallback)

---

### **7. How this appears inside neural networks and LLMs**

- **BERT (2018):** WordPiece tokenizer with 30,000 tokens. Used in original BERT, RoBERTa (despite name, RoBERTa uses BPE), DistilBERT, ALBERT.

- **"##" convention:** Unique to WordPiece. Indicates continuation tokens. Helps the model know whether a subword starts a new word (no "##") or continues from previous ("##").

- **VS BPE:** BPE typically uses special characters like "Ġ" (space) or "\_" to indicate word boundaries. WordPiece uses "##". BPE merges based on frequency; WordPiece merges based on likelihood gain.

- **VS Unigram (SentencePiece default):** SentencePiece can implement Unigram (another algorithm) or BPE. WordPiece is distinct from both.

- **Why BERT used WordPiece:** Google's internal infrastructure. WordPiece was already used in Google's neural machine translation system. BERT inherited it.

- **Why GPT uses BPE:** OpenAI's choice. BPE is simpler and works well for decoder-only models. For encoder-only (BERT), WordPiece's bidirectional efficiency may have advantages.

- **Fallback to characters:** Any subword tokenizer (WordPiece, BPE) falls back to characters (or bytes) for unseen Unicode characters. WordPiece uses [UNK] token for completely unknown characters (rare).

---

### **8. Brain-like connection (optimal subword segmentation)**

WordPiece's likelihood maximization mirrors how children learn to segment speech. Infants hear continuous speech streams ("Whereismymommy?") and must infer where word boundaries lie. They do this by maximizing the likelihood of the sequence under a learned vocabulary of phonemes and morphemes. The "likelihood gain" of hypothesizing a new word boundary is evaluated implicitly. WordPiece's algorithm formalizes this: given a corpus, find the segmentation that maximizes probability. The result is a vocabulary that reflects natural morphological boundaries—just as children acquire morphemes ("un-", "-ing") because they increase the predictability of the linguistic input.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "WordPiece and BPE are essentially the same algorithm. The difference is just implementation details."

_Why it is wrong:_ The merge criterion is fundamentally different. BPE merges the most frequent pair, period. WordPiece merges the pair that most increases corpus likelihood. This difference matters: WordPiece tends to produce more efficient vocabularies (fewer tokens per word) because it considers how merges affect segmentation of all words, not just raw pair frequency. Additionally, WordPiece's "##" convention and greedy longest-match tokenization are distinct from BPE's approach. In practice, BERT (WordPiece) and GPT (BPE) tokenize the same text differently. "tokenization" might be ["token", "##ization"] in WordPiece vs ["token", "ization"] in BPE (if "ization" is a merge). The choice affects model behavior.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  WordPiece is the tokenizer that powered BERT, the model  |
|  that kicked off the LLM revolution. Its likelihood-based |
|  merging produces subword vocabularies that balance       |
|  coverage and efficiency. While BPE won the decoder-only  |
|  war, WordPiece remains essential for understanding       |
|  encoder-only models and Google's approach to tokenization|
|  Knowing both algorithms reveals that tokenization is not |
|  a solved problem—it is a design space with tradeoffs.    |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You tokenize the word "unhappiness" using WordPiece. The vocabulary contains "un", "happiness", "##ness", "##iness", "##happy".

**Question:** What is the likely WordPiece segmentation? Why does WordPiece prefer this over other segmentations?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** Likely segmentation: ["un", "happiness"] or ["un", "happy", "##ness"] (depending on vocabulary).

WordPiece's greedy longest-match algorithm would segment "unhappiness" as:

1. Start at 'u': longest token starting with 'u' in vocab? "un" exists (length 2). No longer token starting with 'u'? "unh" probably not in vocab. So take "un".
2. Remaining: "happiness". Start at 'h': "happiness" exists (whole word) → take "happiness".

Result: ["un", "happiness"] (2 tokens). Alternative ["un", "happy", "##ness"] if "happiness" not in vocab but "happy" and "##ness" are.

WordPiece prefers this because it maximizes the likelihood (or, under greedy matching, takes longest tokens). It avoids splitting "happy" and "ness" if "happiness" is already a token, because longer tokens increase probability (frequency of "happiness" > product of frequencies of "happy" and "##ness"? Likelihood calculation determines). The key: WordPiece is optimized for likelihood, which often aligns with linguistically natural segmentations.
