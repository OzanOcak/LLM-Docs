# Decoder-Only Models (GPT-like)

## The Storyteller Analogy

Imagine a master storyteller spinning a tale. They start with "Once upon a time..." and then continue word by word, each new word building on everything they've said before. They can't jump ahead or peek at the ending—they must create sequentially, one word at a time. That's a decoder-only model like GPT: it generates text left to right, with each new token conditioned on all previous tokens.

In the LLM family, decoder-only models are the stars of generation. GPT, Llama, Mistral, and virtually all modern chat models use this architecture. They've proven that with enough scale, a simple decoder trained only to predict the next word can develop reasoning, follow instructions, and even write poetry.

---

## What Is a Decoder-Only Model?

### The Core Idea

Decoder-only models use only the decoder part of the transformer, generating text autoregressively from left to right.

```text

Decoder-only (GPT):
    The → cat → sat → on → the → mat
     ↓     ↓     ↓     ↓     ↓     ↓
    [ ] → [ ] → [ ] → [ ] → [ ] → [ ]
     ↑     ↑     ↑     ↑     ↑     ↑
    Each step can only see past words!

    "The" predicts "cat"
    "The cat" predicts "sat"
    "The cat sat" predicts "on"
    And so on...
```

```python

def decoder_only_intro():
    """
    The basic concept of decoder-only models
    """
    print("Decoder-Only Models: Masters of Generation")
    print("=" * 60)

    print("""
    Architecture:
    • Only the decoder stack (no encoder)
    • Causal (masked) attention
    • Each token can only see previous tokens

    Training objective: Next Token Prediction
    • Given previous tokens, predict next one
    • Simple but surprisingly powerful

    Output: Generated text, token by token
    • Can continue prompts, answer questions, write stories
    """)

decoder_only_intro()
```

---

## How Decoder-Only Models Work

### Causal Attention Mask

```python

def causal_attention():
    """
    How causal attention prevents looking at future
    """
    print("Causal Attention: Can't See the Future")
    print("=" * 60)

    print("""
    Attention mask for "The cat sat":

    Token:    The  cat  sat
    The  → [1,   0,   0]  (sees only itself)
    cat  → [1,   1,   0]  (sees The, cat)
    sat  → [1,   1,   1]  (sees The, cat, sat)

    During training, when predicting "sat":
    • Can attend to "The", "cat" ✓
    • Cannot attend to future words ✗

    This forces true left-to-right generation!
    """)

causal_attention()
```

### Next Token Prediction

```python

def next_token_prediction():
    """
    How decoder-only models are trained
    """
    print("Training: Next Token Prediction")
    print("=" * 60)

    sentence = "The cat sat on the mat"
    words = sentence.split()

    print(f"Training sentence: '{sentence}'")
    print("\nTraining examples (predict next word):")

    for i in range(len(words) - 1):
        context = ' '.join(words[:i+1])
        target = words[i+1]
        print(f"  Context: '{context:20}' → Predict: '{target}'")

    print("\nThe model learns to:")
    print("• Build language model left-to-right")
    print("• Capture long-range dependencies")
    print("• Develop world knowledge and reasoning")

next_token_prediction()
```

---

## GPT Architecture

### Evolution of GPT Models

```python

def gpt_evolution():
    """
    The GPT family evolution
    """
    print("The GPT Family Evolution")
    print("=" * 60)

    models = {
        "GPT-1 (2018)": {
            "parameters": "117M",
            "layers": 12,
            "innovation": "First generative pre-trained transformer"
        },
        "GPT-2 (2019)": {
            "parameters": "1.5B",
            "layers": 48,
            "innovation": "Showed zero-shot learning, too dangerous to release fully"
        },
        "GPT-3 (2020)": {
            "parameters": "175B",
            "layers": 96,
            "innovation": "Few-shot learning, in-context learning emerges"
        },
        "GPT-3.5/ChatGPT (2022)": {
            "parameters": "~175B",
            "layers": 96,
            "innovation": "RLHF, instruction tuning"
        },
        "GPT-4 (2023)": {
            "parameters": "~1.8T (estimated)",
            "layers": "~120+",
            "innovation": "Multimodal, better reasoning"
        }
    }

    for model, specs in models.items():
        print(f"\n{model}:")
        for key, value in specs.items():
            print(f"  • {key}: {value}")

gpt_evolution()
```

### Autoregressive Generation

```python

def autoregressive():
    """
    How generation works step by step
    """
    print("Autoregressive Generation")
    print("=" * 60)

    prompt = "The capital of France is"

    print(f"Prompt: '{prompt}'")
    print("\nGeneration process:")

    steps = [
        ("Step 1", "The capital of France is", " Paris"),
        ("Step 2", "The capital of France is Paris", ","),
        ("Step 3", "The capital of France is Paris,", " the"),
        ("Step 4", "The capital of France is Paris, the", " city"),
        ("Step 5", "The capital of France is Paris, the city", " of"),
        ("Step 6", "The capital of France is Paris, the city of", " lights"),
        ("Step 7", "The capital of France is Paris, the city of lights", ".")
    ]

    for step, context, next_word in steps:
        print(f"  {step}: '{context}' → '{next_word}'")

    print("\nEach step feeds back into the next!")

autoregressive()
```

---

## What Decoder-Only Models Can Do

### 1. Text Generation

```python

def text_generation():
    """
    Creative text generation
    """
    print("Text Generation with Decoder-Only Models")
    print("=" * 60)

    prompts = [
        "Write a haiku about autumn:",
        "Once upon a time,",
        "The recipe for chocolate chip cookies:",
        "A letter to my future self:"
    ]

    print("Decoder-only models excel at:")
    for prompt in prompts:
        print(f"  • {prompt}")

    print("\nThey can continue in any style they've seen!")

    # Example generation
    print("\nExample output for 'Write a haiku about autumn:':")
    print("  Leaves turn red and gold")
    print("  Crisp air whispers winter's near")
    print("  Harvest moon hangs high")

text_generation()
```

### 2. In-Context Learning

```python

def in_context():
    """
    Learning from examples in the prompt
    """
    print("In-Context Learning")
    print("=" * 60)

    prompt = """
    Translate English to French:
    English: hello
    French: bonjour
    English: cat
    French: chat
    English: dog
    French:"""

    print(f"Prompt with examples:\n{prompt}")
    print("\nModel output: chien")

    print("\nNo fine-tuning needed—just examples in context!")
    print("This emerges only in large decoder-only models.")

in_context()
```

### 3. Instruction Following

```python

def instruction_following():
    """
    Following instructions (after RLHF)
    """
    print("Instruction Following")
    print("=" * 60)

    instructions = [
        "Summarize this article in one sentence: ...",
        "Explain quantum physics to a 5-year-old",
        "Write a poem about AI",
        "Give me 3 tips for public speaking"
    ]

    print("After instruction tuning + RLHF, decoder-only models can:")
    for inst in instructions:
        print(f"  • {inst}")

    print("\nThis is what makes ChatGPT so useful!")

instruction_following()
```

### 4. Code Generation

```python

def code_gen():
    """
    Generating code
    """
    print("Code Generation")
    print("=" * 60)

    prompt = """
    Write a Python function that calculates the factorial of a number:
    """

    print(f"Prompt: {prompt}")
    print("""
    def factorial(n):
        if n <= 1:
            return 1
        else:
            return n * factorial(n-1)
    """)

    print("\nDecoder-only models trained on code (like Codex)")
    print("can generate working code from descriptions!")

code_gen()
```

---

## Decoder-Only vs Other Architectures

| Aspect       | Decoder-Only (GPT)   | Encoder-Only (BERT) | Encoder-Decoder (T5)         |
| ------------ | -------------------- | ------------------- | ---------------------------- |
| Attention    | Causal (left only)   | Bidirectional       | Encoder: bi, Decoder: causal |
| Training     | Next word prediction | Masked LM           | Span corruption              |
| Best at      | Generation           | Understanding       | Transformation               |
| Sees future? | No                   | Yes                 | Encoder: Yes                 |
| Output       | Generated text       | Representations     | Generated text               |
| Scale        | Up to trillions      | 100M-1B             | 100M-10B                     |

### Why Decoder-Only Won

```python

def why_winner():
    """
    Why decoder-only architecture dominates
    """
    print("Why Decoder-Now Models Dominate")
    print("=" * 60)

    reasons = [
        "Simplicity (one stack, one objective)",
        "Scales better (proven to trillions)",
        "Natural for chat/completion",
        "In-context learning emerges",
        "Can do understanding tasks too (with prompting)",
        "Better for RLHF alignment"
    ]

    print("Reasons decoder-only became dominant:")
    for r in reasons:
        print(f"  • {r}")

why_winner()
```

---

## Why This Matters for LLMs

### 1. Powering Modern Chat Models

```python

def chat_models():
    """
    Popular decoder-only models
    """
    print("Popular Decoder-Only Models")
    print("=" * 60)

    models = {
        "GPT-3.5/GPT-4": "OpenAI",
        "Llama 2/3": "Meta (open weights)",
        "Mistral/Mixtral": "Mistral AI",
        "Claude": "Anthropic",
        "Gemini": "Google",
        "Qwen": "Alibaba",
        "DeepSeek": "DeepSeek"
    }

    for model, company in models.items():
        print(f"  • {model}: {company}")

    print("\nAll are decoder-only architectures!")

chat_models()
```

### 2. The Scaling Laws

```python

def scaling():
    """
    How decoder-only models scale
    """
    print("Scaling Laws for Decoder-Only Models")
    print("=" * 60)

    print("""
    Key findings from scaling research:

    • Performance improves predictably with scale
    • Compute-optimal: scale model AND data together
    • Emergent abilities appear at certain thresholds
    • No clear ceiling yet—still scaling!

    Chinchilla optimal for 175B model: ~3T tokens
    GPT-4 likely trained on ~13T tokens

    Simple architecture + massive scale = intelligence
    """)

scaling()
```

### 3. Emergent Abilities

```python

def emergent():
    """
    Abilities that appear at scale
    """
    print("Emergent Abilities in Large Decoders")
    print("=" * 60)

    abilities = [
        "Few-shot learning",
        "Instruction following",
        "Chain-of-thought reasoning",
        "Arithmetic",
        "Code generation",
        "Translation (without explicit training)",
        "Theory of mind"
    ]

    print("These abilities aren't programmed—they EMERGE:")
    for ability in abilities:
        print(f"  • {ability}")

    print("\nThey only appear in very large models!")

emergent()
```

### 4. The Future: Mixture of Experts

```python

def moe():
    """
    Mixture of Experts variant
    """
    print("Mixture of Experts (MoE) Decoders")
    print("=" * 60)

    print("""
    MoE enhances decoder-only architecture:

    Instead of one huge feed-forward layer,
    have multiple "expert" layers with router.

    Each token activates only top experts:
    • More parameters (more knowledge)
    • Same compute (only activate few)
    • Best of both worlds

    Examples: Mixtral 8x7B, GPT-4 (rumored)
    """)

moe()
```

---

## Decoder-Only Cheat Sheet

| Aspect        | Details                       |
| ------------- | ----------------------------- |
| Architecture  | Transformer decoder only      |
| Attention     | Causal (masked)               |
| Training      | Next token prediction         |
| Key property  | Autoregressive generation     |
| Best for      | Generation, chat, completion  |
| Examples      | GPT, Llama, Mistral, Claude   |
| Scale         | Up to trillions of parameters |
| Special sauce | In-context learning emerges   |

---

## Quick Recap

• Decoder-only models generate text left-to-right, one token at a time—like a storyteller who builds a tale word by word, each new word based on everything said before

• They're trained on the simple objective of next token prediction—yet with enough scale, this simple task leads to emergent abilities like reasoning, translation, and code generation

• Decoder-only architecture powers virtually all modern chat models—GPT, Llama, Claude, Mistral—and has proven to scale better than other architectures, making it the dominant choice for large language models

---

## Mental Hook

> "Decoder-only models are like master storytellers who've read millions of books—give them any opening line, and they'll continue the tale seamlessly, never peeking ahead, just building on what's already been said."
