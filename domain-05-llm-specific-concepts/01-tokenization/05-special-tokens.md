# Special Tokens

## The Stage Manager Analogy

Imagine a theater production. Before the play starts, the stage manager calls "Places!" (actors get ready). During the play, there are cues for entrances and exits. At the end, "Curtain!" signals the conclusion. These aren't part of the script, but they're essential for the performance. That's what special tokens do for LLMs—they're markers that tell the model where sentences begin and end, when to separate different parts, and what task to perform.

In LLMs, special tokens are crucial for understanding structure. They mark boundaries, separate different pieces of text, and even tell the model what task to perform. Without them, the model would see a continuous stream of words with no indication of where one sentence ends and another begins, or what it's supposed to do with the text.

---

## What Are Special Tokens?

### The Core Idea

Special tokens are reserved tokens in the vocabulary that have specific meanings and functions, not representing actual words.

```text

Regular tokens: "The" "cat" "sat" "on" "the" "mat"

With special tokens:
[CLS] The cat sat on the mat [SEP]

[CLS] = "This is for classification"
[SEP] = "This is the end of a segment"
```

```python

def special_tokens_intro():
    """
    The basic concept of special tokens
    """
    print("Special Tokens: The Stage Directions of LLMs")
    print("=" * 60)

    print("""
    Special tokens are like markup language for text:

    • They're not real words
    • They have special meanings
    • They guide model behavior
    • They're part of the vocabulary
    """)

    print("\nAnalogy: Like HTML tags that tell the browser")
    print("how to display content, not the content itself.")

special_tokens_intro()
```

---

## Types of Special Tokens

### 1. Structure Tokens

These tokens mark the beginning, end, and boundaries of text.

```python

def structure_tokens():
    """
    Tokens that define text structure
    """
    print("Structure Tokens: Marking Boundaries")
    print("=" * 60)

    tokens = {
        "[CLS]": "Classification token (always first in BERT)",
        "[SEP]": "Separator between sentences/segments",
        "[EOS]": "End of sequence (in generation models)",
        "[BOS]": "Beginning of sequence",
        "<s>": "Start token (Llama, T5)",
        "</s>": "End token (Llama, T5)",
        "<|endoftext|>": "GPT's end token"
    }

    for token, purpose in tokens.items():
        print(f"  • {token}: {purpose}")

    print("\nExample with BERT:")
    print("  [CLS] I love this movie [SEP] It was great [SEP]")

structure_tokens()
```

### 2. Control Tokens

These tokens tell the model what task to perform.

```python

def control_tokens():
    """
    Tokens that control model behavior
    """
    print("Control Tokens: Telling the Model What to Do")
    print("=" * 60)

    print("""
    T5-style task prefixes:

    "translate English to German: Hello world"
    "summarize: The long article text..."
    "question: What is the capital of France? context: France is a country..."

    These aren't special tokens in vocabulary,
    but they act like instructions.
    """)

    print("\nSpecial control tokens in some models:")
    print("  • <pad>: Padding token")
    print("  • <unk>: Unknown token")
    print("  • <mask>: Masked token (BERT training)")
    print("  • <sep>: Separator (some models)")
    print("  • <cls>: Classification token (some models)")

control_tokens()
```

### 3. Training Tokens

These tokens are used during training but may not appear in inference.

```python

def training_tokens():
    """
    Tokens used during training
    """
    print("Training Tokens: The Learning Helpers")
    print("=" * 60)

    tokens = {
        "[MASK]": "Masked token for BERT training",
        "<mask>": "Mask token in some models",
        "<pad>": "Padding to make batches uniform",
        "[UNK]": "Unknown word (older models)"
    }

    for token, purpose in tokens.items():
        print(f"  • {token}: {purpose}")

    print("\nBERT's masked language modeling:")
    print("  Input: The [MASK] sat on the mat")
    print("  Target: cat")
    print("  Model learns to predict masked words")

training_tokens()
```

---

## Special Tokens by Model Family

### BERT Family

```python

def bert_special_tokens():
    """
    Special tokens in BERT
    """
    print("BERT Family Special Tokens")
    print("=" * 60)

    print("""
    BERT's vocabulary includes:

    [CLS] - First token always
          • Final representation used for classification
          • Aggregates whole sequence information

    [SEP] - Separator token
          • Marks end of sentence/segment
          • Helps model distinguish different parts

    [MASK] - Masked token
          • Used only during training
          • Model learns to predict original word

    [PAD] - Padding token
          • Makes all sequences same length in batch
          • Attention masks ignore these

    [UNK] - Unknown token
          • Rare with subword tokenization
          • Fallback for truly unknown chars
    """)

    print("Example BERT input:")
    print("  [CLS] I love this movie [SEP] It was great [SEP]")

bert_special_tokens()
```

### GPT Family

```python

def gpt_special_tokens():
    """
    Special tokens in GPT
    """
    print("GPT Family Special Tokens")
    print("=" * 60)

    print("""
    GPT uses simpler special tokens:

    <|endoftext|> - End of sequence
                  • Marks end of document
                  • Separates different examples
                  • Used as default separator

    No [CLS], [SEP], or [MASK] - GPT generates, doesn't classify

    GPT-3/ChatGPT added special tokens for chat:
    <|im_start|>user
    <|im_start|>assistant
    <|im_end|>
    """)

    print("\nChat format example:")
    print("  <|im_start|>user")
    print("  What's the capital of France?")
    print("  <|im_end|>")
    print("  <|im_start|>assistant")
    print("  Paris")
    print("  <|im_end|>")

gpt_special_tokens()
```

### T5 Family

```python

def t5_special_tokens():
    """
    Special tokens in T5
    """
    print("T5 Family Special Tokens")
    print("=" * 60)

    print("""
    T5 uses SentencePiece with special tokens:

    <s> - Start token
        • Beginning of sequence

    </s> - End token
        • End of sequence
        • Also used as separator

    <pad> - Padding token
        • Makes batches uniform

    T5 also uses task prefixes as regular text:
    "translate English to German: "
    "summarize: "
    "question: "
    """)

    print("\nExample T5 input:")
    print("  <s> translate English to German: Hello world </s>")

t5_special_tokens()
```

### Llama Family

```python

def llama_special_tokens():
    """
    Special tokens in Llama
    """
    print("Llama Family Special Tokens")
    print("=" * 60)

    print("""
    Llama uses SentencePiece with:

    <s> - Beginning of sequence
    </s> - End of sequence
    <unk> - Unknown (rarely used)

    Llama 2 chat format:
    [INST] Instruction [/INST] Response </s>
    [INST] Another question [/INST] Answer </s>

    The [INST] and [/INST] are special tokens
    that mark instruction boundaries.
    """)

    print("\nLlama chat example:")
    print("  <s>[INST] What's 2+2? [/INST] 4 </s>")

llama_special_tokens()
```

---

## Special Tokens in Action

### BERT-Style Processing

```python

def bert_processing():
    """
    How BERT uses special tokens
    """
    print("BERT Processing with Special Tokens")
    print("=" * 60)

    sentence1 = "I love this movie"
    sentence2 = "It was great"

    print(f"Sentence 1: '{sentence1}'")
    print(f"Sentence 2: '{sentence2}'")

    # Tokenize (simulated)
    tokens1 = ["I", "love", "this", "movie"]
    tokens2 = ["It", "was", "great"]

    # Add special tokens
    bert_input = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]

    print("\nBERT input with special tokens:")
    print(f"  {bert_input}")

    print("\nWhat each token does:")
    print("  [CLS] - Final representation used for classification")
    print("  [SEP] - Separates the two sentences")
    print("  [SEP] - Marks end of second sentence")

    # Attention mask (1 = real token, 0 = padding)
    attention_mask = [1] * len(bert_input)

    print(f"\nAttention mask: {attention_mask}")
    print("  (1 = attend to this token)")

bert_processing()
```

### Generation with Special Tokens

```python

def generation_with_special():
    """
    How generation models use special tokens
    """
    print("Generation with Special Tokens")
    print("=" * 60)

    prompt = "The capital of France is"

    print(f"Prompt: '{prompt}'")

    # Simulated generation
    print("\nGeneration process:")
    print("  Step 1: Model sees prompt")
    print("  Step 2: Generates ' Paris'")
    print("  Step 3: Generates '</s>' (end token)")
    print("  Step 4: Stops because end token generated")

    print("\nFull sequence with special tokens:")
    print("  <s> The capital of France is Paris </s>")

    print("\nWithout end token, model would generate forever!")

generation_with_special()
```

### Padding and Attention Masks

```python

def padding_demo():
    """
    How padding works in batches
    """
    print("Padding and Attention Masks")
    print("=" * 60)

    # Three sequences of different lengths
    sequences = [
        ["[CLS]", "I", "love", "it", "[SEP]"],
        ["[CLS]", "Great", "movie", "[SEP]"],
        ["[CLS]", "Wow", "[SEP]"]
    ]

    print("Original sequences:")
    for i, seq in enumerate(sequences):
        print(f"  Seq {i+1}: {seq}")

    # Pad to same length
    max_len = max(len(s) for s in sequences)
    padded = []
    attention_masks = []

    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seq = seq + ["[PAD]"] * pad_len
        mask = [1] * len(seq) + [0] * pad_len

        padded.append(padded_seq)
        attention_masks.append(mask)

    print("\nPadded sequences:")
    for i, seq in enumerate(padded):
        print(f"  Seq {i+1}: {seq}")

    print("\nAttention masks (1=real, 0=padding):")
    for i, mask in enumerate(attention_masks):
        print(f"  Seq {i+1}: {mask}")

    print("\nModel ignores [PAD] tokens during attention!")

padding_demo()
```

---

## Special Tokens Comparison Table

| Token   | BERT      | GPT | T5  | Llama | Purpose           |
| ------- | --------- | --- | --- | ----- | ----------------- | --- | ---------- |
| [CLS]   | ✓         | -   | -   | -     | Classification    |
| [SEP]   | ✓         | -   | -   | -     | Separator         |
| [MASK]  | ✓         | -   | -   | -     | Training          |
| <s>     | -         | -   | ✓   | ✓     | Start             |
| </s>    | -         | -   | ✓   | ✓     | End               |
| <       | endoftext | >   | -   | ✓     | -                 | -   | End        |
| [PAD]   | ✓         | ✓   | ✓   | ✓     | Padding           |
| [INST]  | -         | -   | -   | ✓     | Instruction start |
| [/INST] | -         | -   | -   | ✓     | Instruction end   |
| <       | im_start  | >   | -   | ✓     | -                 | -   | Chat start |
| <       | im_end    | >   | -   | ✓     | -                 | -   | Chat end   |

---

## Why This Matters for LLMs

### 1. They Enable Task Specification

```python

def task_specification():
    """
    How special tokens specify tasks
    """
    print("Special Tokens Enable Task Specification")
    print("=" * 60)

    examples = {
        "Classification (BERT)": "[CLS] This movie is great [SEP]",
        "Sentence pair (BERT)": "[CLS] First sentence [SEP] Second [SEP]",
        "Translation (T5)": "<s> translate English to German: Hello </s>",
        "Chat (GPT)": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>",
        "Instruction (Llama)": "<s>[INST] Tell me a joke [/INST]"
    }

    for task, example in examples.items():
        print(f"  • {task}: {example}")

task_specification()
```

### 2. They Control Generation

```python

def generation_control():
    """
    How special tokens control generation
    """
    print("Special Tokens Control Generation")
    print("=" * 60)

    print("""
    Generation loop:

    1. Start with <s> or prompt
    2. Generate next token
    3. If token == </s> or <|endoftext|>: STOP
    4. Otherwise, append and continue

    This is why models know when to stop!
    """)

    print("\nWithout end tokens:")
    print("  Model: 'The capital of France is Paris the capital of...'")
    print("  (continues forever!)")

generation_control()
```

### 3. They Enable Multi-Task Learning

```python

def multi_task():
    """
    Special tokens enable multiple tasks in one model
    """
    print("Special Tokens Enable Multi-Task Learning")
    print("=" * 60)

    print("""
    T5 can do many tasks with the same model:

    Task specified by prefix:
    • "summarize: article text" → summary
    • "translate English to German: Hello" → Hallo
    • "cola sentence: This is grammatical." → acceptable

    No need for different model heads!
    """)

multi_task()
```

### 4. They Structure Conversations

```python

def conversation():
    """
    Special tokens for chat
    """
    print("Special Tokens Structure Conversations")
    print("=" * 60)

    print("""
    Modern chat models use special tokens to track dialogue:

    <|im_start|>user
    What's the weather?<|im_end|>
    <|im_start|>assistant
    It's sunny!<|im_end|>
    <|im_start|>user
    Great!<|im_end|>

    The model learns the pattern:
    • Who is speaking
    • When turns change
    • Conversation history
    """)

conversation()
```

---

## Special Tokens Cheat Sheet

| Model Family | Start | End   | Separator       | Special       |
| ------------ | ----- | ----- | --------------- | ------------- | --- | --- | -------- | ---- | ------ | --- |
| BERT         | [CLS] | [SEP] | [SEP]           | [MASK]        |
| GPT          | -     | <     | endoftext       | >             | -   | <   | im_start | >, < | im_end | >   |
| T5           | <s>   | </s>  | </s>            | Task prefixes |
| Llama        | <s>   | </s>  | [INST], [/INST] | <unk>         |
| Common       | -     | -     | -               | [PAD]         |

---

## Quick Recap

• Special tokens are like stage directions for LLMs—they mark beginnings ([CLS], <s>), endings ([SEP], </s>), and structure how the model should interpret and generate text

• Different model families use different special tokens—BERT uses [CLS] and [SEP], GPT uses <|endoftext|>, T5 and Llama use <s> and </s>, each designed for their specific architecture

• Special tokens enable critical functionality—they tell the model when to stop generating, separate different parts of input, specify tasks, and structure multi-turn conversations

---

## Mental Hook

> "Special tokens are the stage directions of language models—they're not part of the dialogue, but without them, the actors wouldn't know when to enter, when to exit, or who should speak next."
