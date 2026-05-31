# Reinforcement Learning from Human Feedback (RLHF)

## The Art Critic Analogy

Imagine an aspiring painter who's learned technique but doesn't know what makes art "good." They paint many pieces, and you, the critic, simply point at the ones you like and say "more like this" and the ones you dislike and say "less like this." Over time, the painter learns your preferences and creates art you'll enjoy. That's RLHF: using human preferences to train models what kind of responses we want—helpful, honest, and harmless.

In LLMs, RLHF is the secret sauce that turned GPT-3 into ChatGPT. It aligns model behavior with human values, making responses more helpful, reducing harmful outputs, and generally making AI assistants that people actually want to use.

---

## What Is RLHF?

### The Core Idea

RLHF uses human preferences to train a reward model, then uses reinforcement learning to optimize the language model.

```text

Stage 1: Collect human preferences
Response A (preferred) vs Response B (not preferred)

Stage 2: Train reward model to predict preferences

Stage 3: Use RL to optimize language model
Maximize reward from reward model
```

```python

def rlhf_intro():
    """
    The basic concept of RLHF
    """
    print("RLHF: Learning from Human Preferences")
    print("=" * 60)

    print("""
    Problem: How do we teach models what's "good"?

    • Next-word prediction → knows language, not values
    • Instruction tuning → follows instructions, but...
    • What about being helpful? Honest? Harmless?

    RLHF solution:
    1. Humans compare different responses
    2. Model learns what humans prefer
    3. Model optimizes for those preferences
    """)

rlhf_intro()
```

---

## The Three Stages of RLHF

### Stage 1: Supervised Fine-Tuning (SFT)

First, we do instruction tuning to get a baseline model.

```python

def stage1_sft():
    """
    Stage 1: Supervised Fine-Tuning
    """
    print("Stage 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    print("""
    Start with pre-trained model (e.g., GPT-3)

    Collect demonstration data:
    Human writes ideal responses to prompts

    Example:
    Prompt: "Explain quantum physics simply"
    Human response: "It's like magic that really exists..."

    Fine-tune model on these demonstrations
    → Model learns basic instruction following
    """)

stage1_sft()
```

### Stage 2: Reward Model Training

This is where RLHF gets its name—we train a model to predict human preferences.

```python

def stage2_reward():
    """
    Stage 2: Train a Reward Model
    """
    print("Stage 2: Reward Model Training")
    print("=" * 60)

    print("""
    For the same prompt, generate multiple responses:

    Prompt: "Tell me a joke"

    Response A: "Why did the chicken cross the road?"
    Response B: "Knock knock jokes are the best..."
    Response C: "I don't know any jokes."

    Humans rank these: A > C > B

    Train reward model to predict:
    reward(A) > reward(C) > reward(B)

    Reward model learns what humans like!
    """)

stage2_reward()
```

### Stage 3: Reinforcement Learning

Now we use the reward model to fine-tune the language model.

```python

def stage3_rl():
    """
    Stage 3: Reinforcement Learning Optimization
    """
    print("Stage 3: Reinforcement Learning")
    print("=" * 60)

    print("""
    RL loop:

    1. Sample prompt
    2. Model generates response
    3. Reward model scores response
    4. Update model to maximize reward

    Also add KL penalty:
    Don't drift too far from original model
    (prevents reward hacking)

    Repeat millions of times
    → Model learns to generate high-reward responses
    """)

stage3_rl()
```

---

## The Complete RLHF Pipeline

```python

def rlhf_pipeline():
    """
    The full RLHF process
    """
    print("The Complete RLHF Pipeline")
    print("=" * 60)

    print("""
    ┌─────────────────────────────────────────────────┐
    │           RLHF PIPELINE                         │
    └─────────────────────────────────────────────────┘

    Step 1: Supervised Fine-Tuning
    ┌──────────────┐    ┌──────────────┐
    │   Prompts    │───>│   Human      │
    │              │    │  Responses   │
    └──────────────┘    └──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   SFT Model         │
                    └─────────────────────┘

    Step 2: Reward Model Training
    ┌──────────────┐    ┌──────────────┐
    │   Prompts    │───>│    Model      │
    │              │    │  Generates    │
    └──────────────┘    │  Responses    │
                        └──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Human Rankings    │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Reward Model      │
                    └─────────────────────┘

    Step 3: RL Optimization
    ┌──────────────┐    ┌──────────────┐
    │   Prompts    │───>│   RL-updated │
    │              │    │    Model     │
    └──────────────┘    └──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Reward Model      │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Update Model      │
                    └─────────────────────┘
    """)

rlhf_pipeline()
```

---

## Concrete Example

### Collecting Preferences

```python

def preference_example():
    """
    Example of human preference data
    """
    print("Collecting Human Preferences")
    print("=" * 60)

    prompt = "What's the capital of France?"

    responses = [
        "Paris is the capital of France.",
        "The capital of France is Paris. It's known for the Eiffel Tower.",
        "idk lol",
        "Well, historically, many cities have served as capitals..."
    ]

    print(f"Prompt: '{prompt}'\n")
    print("Responses:")
    for i, resp in enumerate(responses, 1):
        print(f"\n  {i}. {resp}")

    print("\nHuman rankings (preferred to least):")
    print("  Best:   #2 (informative but concise)")
    print("  Good:   #1 (correct but minimal)")
    print("  OK:     #4 (rambling)")
    print("  Worst:  #3 (unhelpful)")

    print("\nReward model learns these patterns!")

preference_example()
```

### Training the Reward Model

```python

def reward_model_training():
    """
    How the reward model is trained
    """
    print("Training the Reward Model")
    print("=" * 60)

    print("""
    Reward model loss function:

    Loss = -log( σ( reward(chosen) - reward(rejected) ) )

    Plain English:
    Make the difference between chosen and rejected
    as large as possible (in log-odds space).

    Example:
    chosen score: 2.5
    rejected score: 0.5
    difference: 2.0 → high probability model learned

    chosen score: 1.2
    rejected score: 1.1
    difference: 0.1 → model not learning well
    """)

reward_model_training()
```

---

## Why RLHF Works

### Capturing What's Hard to Specify

```python

def hard_to_specify():
    """
    Why we need RLHF instead of rules
    """
    print("What RLHF Captures That Rules Can't")
    print("=" * 60)

    qualities = [
        "Helpfulness (vs just correctness)",
        "Tone (friendly vs robotic)",
        "Conciseness (vs rambling)",
        "Creativity (vs boring)",
        "Safety (avoiding harmful content)",
        "Honesty (admitting uncertainty)"
    ]

    print("These qualities are hard to write rules for:")
    for q in qualities:
        print(f"  • {q}")

    print("\nBut humans can recognize them instantly!")
    print("RLHF lets us use human judgment as training signal.")

hard_to_specify()
```

### Alignment with Human Values

```python

def alignment():
    """
    Aligning models with human values
    """
    print("RLHF for Value Alignment")
    print("=" * 60)

    print("""
    The alignment problem:
    Models optimize for next-word prediction,
    not for being helpful, honest, and harmless.

    RLHF solution:
    • Humans prefer helpful responses
    • Humans prefer honest responses (not making things up)
    • Humans prefer harmless responses (not harmful)

    Reward model learns these preferences
    RL optimizes for them

    Result: Aligned models like ChatGPT!
    """)

alignment()
```

---

## RLHF vs Other Methods

| Aspect       | Supervised Learning    | Instruction Tuning     | RLHF                      |
| ------------ | ---------------------- | ---------------------- | ------------------------- |
| Data         | Correct outputs        | Instructions + outputs | Preferences (comparisons) |
| Signal       | Exact words            | Exact words            | Relative quality          |
| Objective    | Match target           | Follow instruction     | Maximize preference       |
| Flexibility  | Low                    | Medium                 | High                      |
| Human effort | High (write responses) | High (write responses) | Lower (just compare)      |
| Cost         | Medium                 | Medium                 | High (RL training)        |

### Why RLHF Is More Efficient

```python

def efficiency():
    """
    Why comparisons are more efficient
    """
    print("The Efficiency of Comparisons")
    print("=" * 60)

    print("""
    Writing a perfect response is hard.
    Comparing two responses is easy.

    Example:
    Response A: "Paris is the capital of France."
    Response B: "The capital of France is Paris, a beautiful city."

    Which is better? Most people can quickly decide.

    With 100 response pairs, you can train a reward model
    that captures preferences across many dimensions.
    """)

efficiency()
```

---

## Why This Matters for LLMs

### 1. ChatGPT Was Made with RLHF

```python

def chatgpt_rlhf():
    """
    ChatGPT's RLHF training
    """
    print("ChatGPT: Product of RLHF")
    print("=" * 60)

    print("""
    InstructGPT / ChatGPT training:

    1. SFT: 13,000 demonstration pairs
       Human writes ideal responses

    2. Reward Model: 33,000 preference comparisons
       Humans rank different responses

    3. RL: PPO optimization against reward model
       Millions of RL steps

    Result: Models that are:
    • More helpful
    • More truthful
    • Less harmful
    """)

chatgpt_rlhf()
```

### 2. Reducing Harmful Outputs

```python

def safety():
    """
    RLHF for safety
    """
    print("RLHF Makes Models Safer")
    "=" * 60

    print("""
    Without RLHF, models might:
    • Generate toxic content
    • Give dangerous advice
    • Be easily jailbroken

    RLHF helps by:
    • Learning what humans consider harmful
    • Optimizing to avoid those responses
    • Rejecting inappropriate requests politely

    Example:
    Prompt: "How to make a bomb?"
    Without RLHF: [gives instructions]
    With RLHF: "I can't provide instructions for that..."
    """)

safety()
```

### 3. The Scalability Challenge

```python

def scalability():
    """
    Challenges in scaling RLHF
    """
    print("Scaling RLHF: Challenges")
    "=" * 60

    challenges = [
        "Collecting high-quality preferences is expensive",
        "Reward hacking (model finds loopholes)",
        "Preference inconsistency across humans",
        "RL training is unstable and complex",
        "Hard to evaluate if it's working"
    ]

    print("Current challenges:")
    for c in challenges:
        print(f"  • {c}")

scalability()
```

### 4. Beyond RLHF: Constitutional AI

```python

def constitutional():
    """
    Constitutional AI: RLHF without human feedback
    """
    print("Constitutional AI (Claude)")
    "=" * 60

    print("""
    Anthropic's Constitutional AI:

    1. Model generates responses
    2. Model critiques itself using principles
    3. Model revises based on critique
    4. Train on self-critique pairs

    Principles include:
    • Be helpful
    • Be harmless
    • Be ethical
    • Respect privacy

    This reduces need for human feedback!
    """)

constitutional()
```

---

## RLHF Cheat Sheet

| Aspect     | Details                             |
| ---------- | ----------------------------------- |
| Purpose    | Align models with human preferences |
| Stages     | SFT → Reward Model → RL             |
| Data       | Human comparisons (preferences)     |
| Objective  | Maximize reward from reward model   |
| Output     | Helpful, honest, harmless models    |
| Used in    | ChatGPT, Claude, Llama 2 Chat       |
| Challenges | Expensive, complex, reward hacking  |

---

## Quick Recap

• RLHF uses human preferences to train models what we want—like an art critic who just points at paintings they like, the model learns to generate responses humans prefer

• It works in three stages—supervised fine-tuning on demonstrations, training a reward model on comparisons, then using reinforcement learning to optimize against that reward model

• RLHF is what made ChatGPT so good—it aligned GPT-3's raw capabilities with human values of helpfulness, honesty, and harmlessness, creating an assistant people actually want to use

---

## Mental Hook

> "RLHF is like training a chef by just saying 'more like this' and 'less like that'—you don't need to explain recipes or techniques, just point at dishes you like, and eventually the chef learns your taste perfectly."
