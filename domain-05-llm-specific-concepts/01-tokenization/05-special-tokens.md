# Special tokens

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Tokenization**

---

### **1. Why this concept matters**

A raw text corpus contains only words, numbers, punctuation, and spaces. But language models need more: signals for the start and end of a sequence, separators between sentences, a marker for masked words to predict, and boundaries between system and user messages in a chat. These are special tokens. They are not part of the natural language vocabulary but are inserted during preprocessing to guide the model's behavior. Every LLM has them—[CLS], [SEP], <s>, </s>, <|im_start|>, and many others. Understanding special tokens means understanding how to communicate structure to a model that only sees token IDs.

---

### **2. Core idea**

**Special tokens are artificial tokens added to the vocabulary that carry structural meaning (e.g., sequence start, end, separation, masking) and are never part of the natural text, allowing the model to learn task-specific behaviors.**

---

### **3. Concrete analogy**

Imagine you are sending a telegram in the 1900s. Your message is just words: "SELL STOCK." But the telegraph operator needs signals to know where the message starts and ends, who it is from, and whether it is urgent.

So you use special codes:

- <START> marks the beginning
- <END> marks the end
- <SENDER> identifies you
- <URGENT> flags priority

These codes are not part of the message content. They are meta-instructions to the system.

Special tokens in LLMs work the same way. The model sees [CLS] and knows "this token's output will be used for classification." It sees <|im_end|> and knows "the assistant's response ends here." These tokens shape the model's behavior without appearing in the natural text the model is trained to generate.

---

### **4. ASCII diagram**

```
Common special tokens across different models:

BERT (Encoder-only):
    [CLS] - Classification token (output for aggregate representation)
    [SEP] - Separator token (between sentences, or query vs passage)
    [MASK] - Masked token (for MLM pre-training)
    [UNK] - Unknown token (fallback for out-of-vocabulary)
    [PAD] - Padding token (for batching sequences of different lengths)


GPT (Decoder-only, using <|...|> style):
    <|endoftext|> - End of document / separator
    <|im_start|> - Start of chat message (system/user/assistant)
    <|im_end|> - End of chat message


Llama (using special tokens):
    <s> - Start of sequence (BOS)
    </s> - End of sequence (EOS)
    <unk> - Unknown token


T5 (SentencePiece, uses sentinel tokens):
    <extra_id_0>, <extra_id_1>, ... - For span corruption pre-training


Example: BERT input for sentence pair

    [CLS] "cat sat" [SEP] "the cat slept" [SEP]
       ↓       ↓        ↓         ↓            ↓
    Token IDs: 101   ...   102    ...         102

    Model output: [CLS] representation used for classification


Example: Chat format with special tokens (Llama 3 style)

    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a helpful assistant.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    What is 2+2?
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    2+2 = 4.
    <|eot_id|>
```

---

### **5. Mathematical formulation**

Let V be the regular vocabulary (words, subwords, characters/bytes). Define special tokens S.

The extended vocabulary:

$$
V_{\text{total}} = V \cup S
$$

Special tokens are assigned unique IDs, just like regular tokens.

**Special token roles:**

- **BOS (Beginning of Sequence):** Always inserted at the start of input. For decoder-only models, enables the model to begin generation.

- **EOS (End of Sequence):** Marks termination. During training, loss is not computed after EOS. During inference, generation stops when EOS is produced.

- **PAD (Padding):** For batching sequences of different lengths. Padding tokens are masked in attention (ignored) and ignored in loss computation.

- **SEP (Separator):** Used in BERT to separate sentences. Helps the model learn sentence boundaries.

- **CLS (Classification):** Token whose final hidden state is used as aggregate representation for classification tasks.

- **MASK:** Used in masked language modeling. The model must predict the original token.

- **UNK (Unknown):** Fallback for characters/tokens not in vocabulary. Modern LLMs (byte-level BPE, SentencePiece) rarely use UNK.

**Attention mask for special tokens:**

$$
\text{AttentionMask}(i) = \begin{cases} 0 & \text{if token i is PAD} \\ 1 & \text{otherwise} \end{cases}
$$

**Loss mask for special tokens:**

During training, loss for special tokens (e.g., PAD, BOS, separator tokens) is often masked out (ignored) because the model does not need to learn to predict them.

---

### **6. Worked example (step-by-step)**

#### **Step 1: BERT-style classification**

Task: Sentiment analysis of "I love NLP!"

Preprocessing:

1. Add [CLS] at start, [SEP] at end
2. Tokenize (WordPiece)

Result tokens:
[CLS] I love NLP ! [SEP]
Token IDs: 101 1045 2293 17953 999 102

#### **Step 2: Model forward pass**

Model processes all tokens, produces hidden states. The hidden state corresponding to [CLS] (position 0) is passed to a classification head. The model does not classify each word; it uses the aggregated [CLS] representation.

#### **Step 3: Loss computation**

Loss is computed only on the classification output (binary cross-entropy). The model never has to predict [CLS] or [SEP] as outputs—they are only input tokens.

#### **Step 4: GPT generation**

Prompt: "2+2="

Tokenizer adds BOS if configured: <s> 2 + 2 =

Model generates: "4" then <EOS>

Sequence: <s> 2 + 2 = 4 <EOS>

During training, loss is computed on all tokens except PAD. The model learns to predict the next token, including special tokens (EOS).

#### **Step 5: Chat conversation (Llama 3 format)**

System: "You are a helpful assistant."
User: "What is 2+2?"

Special token sequence:
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is 2+2?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
2+2 = 4.
<|eot_id|>

The model learns that after <|start_header_id|>assistant<|end_header_id|>, it should generate a helpful response until <|eot_id|>. Special tokens encode the conversation structure.

---

### **7. How this appears inside neural networks and LLMs**

- **BERT:** [CLS], [SEP], [MASK], [UNK], [PAD]. [CLS] output used for classification. [SEP] separates sentences in NSP (next sentence prediction, now optional).

- **GPT-2/3/4:** <|endoftext|> as document separator. No BOS (implicit start). GPT-4 chat uses special tokens (less documented, but structure similar to Llama).

- **Llama (Meta):** <s> (BOS), </s> (EOS), <unk>. For chat: <|begin_of_text|>, <|start_header_id|>, <|end_header_id|>, <|eot_id|>. These are explicitly part of the training data format.

- **T5:** No special tokens for BOS/EOS. Uses sentinel tokens <extra_id_0> for span corruption. Relies on relative position embeddings.

- **ChatML (OpenAI format):** <|im_start|>, <|im_end|> for system, user, assistant roles.

- **PAD token and attention masking:** Essential for batching. PAD tokens are ignored in attention (mask = 0) and excluded from loss.

- **Special token IDs are fixed:** Once tokenizer is trained, special tokens get fixed IDs (e.g., 0 for PAD, 1 for UNK, 2 for BOS, 3 for EOS). These are never merged with other tokens.

---

### **8. Brain-like connection (pragmatic markers)**

Human speech uses words that are not part of the propositional content but guide conversation structure: "um," "well," "so," "okay." These are pragmatic markers. They signal turn-taking ("so..."), topic shifts ("now then"), or discourse boundaries. Special tokens in LLMs serve the same function: they are not about the content but about the structure. They tell the model "this is the start," "this is the end," "here comes the user," "here comes the assistant." The brain has dedicated mechanisms (e.g., right-hemisphere language areas) for processing these pragmatic signals. Special tokens are the LLM's pragmatic markers.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Special tokens are just implementation details. I can ignore them and just use raw text."

_Why it is wrong:_ Without special tokens, the model does not know where a sequence starts or ends. Try chatting with a model that never saw <|im_start|>user tokens—it will not know who is speaking. For classification, without [CLS] or equivalent, you would have to pool over all token representations arbitrarily. Special tokens are not optional. They are how the model learns task structure. Removing them collapses the model to pure next-token prediction without any guidance. In chat, special tokens are essential for distinguishing system, user, and assistant messages—without them, the model cannot learn the conversational format.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Text alone has no structure. There is no marker for     |
|  "this is the beginning of the input" or "this is the    |
|  separator between sentences." Special tokens provide     |
|  this structure. They turn a raw string into a formatted  |
|  message the model can parse. [CLS] aggregates for        |
|  classification. <|im_start|> marks speaker turns. </s>    |
|  signals end of generation. Without special tokens,       |
|  your model is just guessing at the task. With them,      |
|  you can communicate intent.                              |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are fine-tuning a pre-trained GPT model for a multi-turn chat application. The base model's vocabulary includes <|endoftext|> but not chat-specific special tokens.

**Question:** How do you modify the tokenizer to support chat format (system, user, assistant turns)? What special tokens would you add? What must you do with the model's embedding matrix?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:** You need to add new special tokens: e.g., <|im_start|>, <|im_end|>, and role tokens (system, user, assistant). Steps:

1. Add these tokens to the tokenizer's vocabulary (reserve new IDs)
2. Expand the model's embedding matrix to include these new token IDs (resize token embeddings). New embeddings can be initialized from existing token embeddings (e.g., copy the embedding of <|endoftext|> for <|im_start|> to start).
3. Fine-tune the model on chat data formatted with these special tokens.

Without these steps, the tokenizer will treat <|im_start|> as unknown (or split into subwords it does know, losing the single-token semantics). The model would not have embedding vectors for these new tokens and could not process the chat format correctly.

Alternatively, use a chat template format that the base tokenizer already handles (e.g., the model's native chat format). Many pre-trained models already include chat special tokens (Llama, Mistral). Check before adding.
