# Speculative Decoding

## The Chess Grandmaster Analogy

Imagine a chess grandmaster playing a rapid game. Instead of calculating every move deeply, they quickly consider a few obvious candidate moves, evaluate them roughly, and if one seems promising, they play it without full analysis. If it turns out bad, they revert and think deeper. That's speculative decoding: using a fast, smaller "draft" model to guess multiple future tokens, then having the large "target" model verify them in parallel. When the drafts are correct, we get massive speedups.

In LLM inference, speculative decoding is a game-changer for latency. Instead of generating one token at a time (which is bottlenecked by memory bandwidth), we generate multiple tokens in parallel when the draft model's guesses are good. This can speed up generation by 2-3x with no loss in quality.

---

## The Problem: Autoregressive Generation Is Slow

### One Token at a Time

```python

def autoregressive_problem():
    """
    The fundamental bottleneck of autoregressive generation
    """
    print("The Problem: Generating One Token at a Time")
    print("=" * 60)

    print("""
    Traditional generation:

    Step 1: Input: "The cat"
            Compute → output: " sat"  (1 token, 1 forward pass)

    Step 2: Input: "The cat sat"
            Compute → output: " on"   (1 token, 1 forward pass)

    Step 3: Input: "The cat sat on"
            Compute → output: " the"  (1 token, 1 forward pass)

    Step 4: Input: "The cat sat on the"
            Compute → output: " mat"  (1 token, 1 forward pass)

    For N tokens: N sequential forward passes!
    Each pass is bottlenecked by memory bandwidth, not compute.
    """)

autoregressive_problem()
```

### The Memory Bandwidth Bottleneck

```python

def memory_bottleneck():
    """
    Why generation is memory-bound, not compute-bound
    """
    print("The Memory Bandwidth Bottleneck")
    "=" * 60

    print("""
    For each token generated:

    1. Load model weights from GPU memory → ~seconds (slow!)
    2. Do a tiny bit of compute (one forward pass)
    3. Store results back to memory

    Modern GPUs: Compute is fast, memory is slow.

    Analogy: Like a chef who must walk to the pantry
             for every single ingredient (memory-bound)
             instead of carrying multiple ingredients at once.

    Result: GPU is idle most of the time, waiting for memory!
    """)

memory_bottleneck()
```

---

## The Solution: Speculative Decoding

### How Speculative Decoding Works

```python

def speculative_intro():
    """
    The basic concept of speculative decoding
    """
    print("Speculative Decoding: Draft + Verify")
    "=" * 60

    print("""
    Two models work together:

    1. DRAFT MODEL (small, fast)
       • 100-1000x faster than target
       • Less accurate, but good enough for guesses

    2. TARGET MODEL (large, accurate)
       • The real model we want to use
       • Slow but correct

    Process:

    Step 1: Draft model quickly guesses K future tokens
    Step 2: Target model verifies ALL K tokens in ONE forward pass
    Step 3: Keep longest prefix that matches, reject the rest
    Step 4: Repeat

    If drafts are good: Generate K tokens in 2 passes instead of K passes!
    """)

speculative_intro()
```

### Step-by-Step Example

```python

def speculative_step_by_step():
    """
    Detailed walkthrough of speculative decoding
    """
    print("Speculative Decoding: Step by Step")
    "=" * 60

    prompt = "The capital of France is"
    print(f"Prompt: '{prompt}'\n")

    # Step 1: Draft model guesses
    print("Step 1: Draft model (fast) guesses 4 tokens")
    draft_guesses = [" Paris", " is", " a", " beautiful"]
    print(f"  Draft guesses: {draft_guesses}")

    # Step 2: Target model verifies
    print("\nStep 2: Target model (slow) verifies ALL in one pass")
    print("  Target model computes probabilities for each position:")

    # Simulated verification results
    verification = [
        {"token": " Paris", "correct": True, "prob": 0.95},
        {"token": " is", "correct": True, "prob": 0.92},
        {"token": " a", "correct": False, "prob": 0.30},  # Wrong! Should be "the"
        {"token": " beautiful", "correct": False, "prob": 0.10}
    ]

    for v in verification:
        status = "✓" if v["correct"] else "✗"
        print(f"    {v['token']:12} {status} (prob: {v['prob']:.2f})")

    # Step 3: Accept correct prefix
    print("\nStep 3: Accept longest correct prefix")
    print("  Accepted: ' Paris is' (2 tokens)")
    print("  Rejected: ' a beautiful'")

    # Step 4: Repeat with corrected prefix
    print("\nStep 4: Continue from last accepted token")
    print("  New prompt: 'The capital of France is Paris is'")
    print("  Next round of speculation...")

speculative_step_by_step()
```

---

## Why Speculative Decoding Works

### The Acceptance Rate

```python

def acceptance_rate():
    """
    How acceptance rate affects speedup
    """
    print("Acceptance Rate: Key to Speedup")
    "=" * 60

    import math

    # Simulate different acceptance rates
    gamma = 5  # Number of speculative tokens
    acceptance_rates = [0.5, 0.7, 0.8, 0.9, 0.95]

    print(f"Speculating {gamma} tokens at a time")
    print("Acceptance | Expected tokens | Speedup")
    print("-" * 45)

    for acc in acceptance_rates:
        # Expected tokens per verification pass
        expected = sum(acc ** i for i in range(gamma))
        # Speedup vs sequential (1 token per pass)
        speedup = expected / 2  # 2 passes (draft + verify)
        print(f"   {acc:.0%}    |      {expected:.2f}       |   {speedup:.2f}x")

acceptance_rate()
```

### When Draft Model Is Good

```python

def draft_quality():
    """
    Why a good draft model matters
    """
    print("Draft Model Quality Matters")
    "=" * 60

    scenarios = [
        {"draft": "Random", "acc": 0.1, "speedup": "0.5x (slower!)"},
        {"draft": "Poor", "acc": 0.3, "speedup": "1.2x"},
        {"draft": "Decent", "acc": 0.6, "speedup": "1.8x"},
        {"draft": "Good", "acc": 0.8, "speedup": "2.5x"},
        {"draft": "Excellent", "acc": 0.95, "speedup": "3.0x"}
    ]

    for s in scenarios:
        bar = "█" * int(s["acc"] * 20)
        print(f"  {s['draft']:10} {bar} acc={s['acc']:.0%} → {s['speedup']}")

draft_quality()
```

---

## Implementation Details

### Draft Model Selection

```python

def draft_selection():
    """
    Choosing a draft model
    """
    print("Choosing a Draft Model")
    "=" * 60

    options = [
        {"type": "Same architecture, smaller",
         "example": "Llama 68M vs Llama 7B",
         "pros": "Good alignment, easy to train",
         "cons": "Still relatively slow"},

        {"type": "Different architecture",
         "example": "Transformer with 2 layers vs 32 layers",
         "pros": "Very fast",
         "cons": "May have different output distribution"},

        {"type": "N-gram/statistical",
         "example": "Lookup table based on training data",
         "pros": "Extremely fast, no GPU needed",
         "cons": "Lower accuracy"}
    ]

    for opt in options:
        print(f"\n  • {opt['type']}:")
        print(f"    Example: {opt['example']}")
        print(f"    Pros: {opt['pros']}")
        print(f"    Cons: {opt['cons']}")

draft_selection()
```

### Verification Mechanism

```python

def verification():
    """
    How verification works
    """
    print("Verification: Rejection Sampling")
    "=" * 60

    print("""
    For each speculated token, we check:

    q(x) = draft model probability
    p(x) = target model probability

    Acceptance criterion:

    Accept if: p(x) / q(x) ≥ 1 (draft was confident and correct)
    Otherwise: Accept with probability min(1, p(x)/q(x))

    This ensures:
    • The final distribution matches the target model EXACTLY
    • No quality loss, only speedup
    • Mathematically proven correctness

    When draft is good (q(x) ≈ p(x)), acceptance rate is high.
    When draft is wrong, we reject and sample from target.
    """)

verification()
```

---

## Performance Impact

### Speedup Results

```python

def speedup_results():
    """
    Real-world speedup numbers
    """
    print("Real-World Speedup Results")
    "=" * 60

    results = [
        {"model": "Llama 7B", "draft": "Llama 68M", "speedup": "2.3x"},
        {"model": "GPT-3 175B", "draft": "GPT-3 350M", "speedup": "2.8x"},
        {"model": "CodeLlama", "draft": "N-gram", "speedup": "1.8x"},
        {"model": "T5-XXL", "draft": "T5-small", "speedup": "2.1x"}
    ]

    print("Speedups reported in research:")
    for r in results:
        print(f"  • {r['model']:15} + {r['draft']:15} → {r['speedup']}")

speedup_results()
```

### Cost-Benefit Analysis

```python

def cost_benefit():
    """
    When speculative decoding makes sense
    """
    print("When to Use Speculative Decoding")
    "=" * 60

    factors = [
        {"condition": "Large target model (7B+)", "benefit": "High", "reason": "More to gain from parallelization"},
        {"condition": "Small draft available", "benefit": "High", "reason": "Good acceptance rate"},
        {"condition": "Latency-critical applications", "benefit": "High", "reason": "2-3x faster responses"},
        {"condition": "Batch size = 1", "benefit": "High", "reason": "Can't batch across requests"},
        {"condition": "Long generations", "benefit": "High", "reason": "More tokens to speculate"},
        {"condition": "Small target model", "benefit": "Low", "reason": "Overhead may outweigh gains"}
    ]

    for f in factors:
        print(f"  • {f['condition']:30} → {f['benefit']:6} | {f['reason']}")

cost_benefit()
```

---

## Speculative Decoding in Practice

### Using with Hugging Face

```python

def hf_implementation():
    """
    Implementing speculative decoding
    """
    print("Speculative Decoding with Hugging Face")
    "=" * 60

    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load models
    target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    draft = AutoModelForCausalLM.from_pretrained("facebook/llama-68m-draft")

    # Generate with speculation
    outputs = target.generate(
        input_ids,
        draft_model=draft,
        num_assistant_tokens=5,  # K speculative tokens
        do_sample=True,
        temperature=0.7
    )

    # Or use assisted decoding (built-in)
    outputs = target.generate(
        input_ids,
        assistant_model=draft,  # Hugging Face built-in!
        max_new_tokens=100
    )
    """)

hf_implementation()
```

### With vLLM

```python

def vllm_speculative():
    """
    Speculative decoding in vLLM
    """
    print("Speculative Decoding in vLLM")
    "=" * 60

    print("""
    from vllm import LLM, SamplingParams

    # vLLM supports speculative decoding natively
    llm = LLM(
        model="meta-llama/Llama-2-70b-hf",
        speculative_model="facebook/llama-68m-draft",
        num_speculative_tokens=5
    )

    # Generate - automatically uses speculation
    outputs = llm.generate(prompts, sampling_params)

    # Benefits:
    # • Integrated with PagedAttention
    # • Works with continuous batching
    # • Minimal overhead
    """)

vllm_speculative()
```

---

## Speculative Decoding Cheat Sheet

| Component              | Role                     | Speed                             | Accuracy                      |
| ---------------------- | ------------------------ | --------------------------------- | ----------------------------- |
| Draft model            | Fast guesses             | 100-1000x faster                  | Lower                         |
| Target model           | Verification             | 1x                                | High                          |
| Speculation length (K) | How many tokens to guess | Higher K = more potential speedup | Diminishing returns after K=5 |
| Acceptance rate        | % of guesses correct     | Key metric                        | Depends on draft quality      |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Speculative decoding is a free lunch for LLM inference:    ║
║                                                              ║
║  🚀 2-3x SPEEDUP: Generate tokens 2-3x faster with         ║
║     no loss in quality. Same hardware, better experience.   ║
║                                                              ║
║  🎯 PERFECT QUALITY: Mathematical guarantee that            ║
║     output distribution matches the target model exactly.   ║
║                                                              ║
║  💰 COST SAVINGS: 2-3x faster means 2-3x lower            ║
║     latency or 2-3x more users on same hardware.            ║
║                                                              ║
║  🔧 COMPLEMENTARY: Works with quantization, caching,       ║
║     and batching—stack the optimizations!                   ║
║                                                              ║
║  📱 BETTER UX: Faster responses make applications           ║
║     feel more responsive and natural.                        ║
║                                                              ║
║  Speculative decoding is one of the few techniques          ║
║  that gives you something for nothing—speed without         ║
║  sacrificing quality.                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Speculative decoding uses a fast draft model to guess multiple future tokens—like a chess player quickly considering obvious moves before deep analysis

• The target model verifies all guesses in one parallel forward pass—accepting correct guesses and rejecting wrong ones, with mathematical guarantees of exact output distribution

• With a good draft model, this speeds up generation by 2-3x—addressing the memory bandwidth bottleneck of autoregressive generation without any quality loss

---

## Mental Hook

> "Speculative decoding is like having a quick-thinking assistant who suggests the next few words while you're speaking—you can accept the good suggestions instantly and only pause to correct the bad ones, making the whole conversation 2-3x faster."
