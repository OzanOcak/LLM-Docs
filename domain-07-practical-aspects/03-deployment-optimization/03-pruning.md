# Pruning

## The Bonsai Tree Analogy

Imagine a wild, overgrown tree with thousands of branches. A bonsai master carefully prunes away the weak, crossing, or unnecessary branches, shaping the tree while keeping its essential structure and beauty. The tree becomes smaller, more elegant, and easier to maintain—but it's still recognizably the same tree. That's pruning for neural networks: removing less important connections (weights) or even entire neurons while preserving the model's capabilities.

In LLM optimization, pruning creates smaller, faster models by eliminating redundant or unimportant parameters. Unlike quantization (which shrinks numbers), pruning removes them entirely. The result is a sparse network—many weights become zero—that can be stored more efficiently and sometimes run faster with specialized hardware.

---

## What Is Pruning?

### The Core Idea

Pruning removes parameters from a neural network that contribute little to its output.

```text

Before Pruning (Dense):              After Pruning (Sparse):
┌─────────────────────┐              ┌─────────────────────┐
│ 0.8  0.1  -0.3  0.0 │              │ 0.8  0.0  -0.3  0.0 │
│ 0.0  0.6   0.2 -0.1 │    prune     │ 0.0  0.6   0.0  0.0 │
│-0.2  0.0   0.9  0.0 │    ─────→    │-0.2  0.0   0.9  0.0 │
│ 0.1 -0.4   0.0  0.7 │              │ 0.0 -0.4   0.0  0.7 │
└─────────────────────┘              └─────────────────────┘
   16 non-zero weights                  10 non-zero weights
   100% density                         62.5% density
```

```python

def pruning_intro():
    """
    The basic concept of pruning
    """
    print("Pruning: Removing Unnecessary Connections")
    print("=" * 60)

    print("""
    Why prune?

    • SMALLER MODELS: Fewer parameters to store
    • FASTER INFERENCE: Less computation (if sparsity is exploited)
    • LOWER MEMORY: Can store in sparse formats
    • BETTER GENERALIZATION: Sometimes even improves performance!

    Key insight: Neural networks are massively overparameterized.
    Many weights are redundant and can be removed.
    """)

pruning_intro()
```

---

## Types of Pruning

### 1. Magnitude-Based Pruning

The simplest approach: remove weights with small absolute values.

```python

def magnitude_pruning():
    """
    Pruning based on weight magnitude
    """
    print("Magnitude-Based Pruning")
    print("=" * 60)

    import numpy as np

    # Example weight matrix
    weights = np.array([
        [0.8, 0.02, -0.7, 0.01],
        [-0.6, 0.001, 0.9, -0.03],
        [0.03, 0.5, -0.008, 0.7],
        [0.009, -0.4, 0.02, 0.6]
    ])

    print("Original weights:")
    print(weights)
    print(f"Non-zero: {np.count_nonzero(weights)}/{weights.size}")

    # Prune small weights (threshold 0.05)
    threshold = 0.05
    pruned = weights.copy()
    pruned[np.abs(pruned) < threshold] = 0

    print(f"\nAfter pruning (|weight| < {threshold}):")
    print(pruned)
    print(f"Non-zero: {np.count_nonzero(pruned)}/{pruned.size}")
    print(f"Sparsity: {1 - np.count_nonzero(pruned)/pruned.size:.1%}")

magnitude_pruning()
```

### 2. Structured Pruning

Remove entire neurons, channels, or attention heads at once.

```python

def structured_pruning():
    """
    Structured pruning removes whole structures
    """
    print("Structured Pruning")
    "=" * 60

    print("""
    Unstructured pruning:           Structured pruning:
    ┌───┬───┬───┐                  ┌───┬───┬───┐
    │ ✓ │ ✗ │ ✓ │                  │ ✓ │ ✗ │ ✓ │
    ├───┼───┼───┤                  ├───┼───┼───┤
    │ ✗ │ ✓ │ ✗ │                  │ ✓ │ ✗ │ ✓ │
    ├───┼───┼───┤                  ├───┼───┼───┤
    │ ✓ │ ✗ │ ✓ │                  │ ✓ │ ✗ │ ✓ │
    └───┴───┴───┘                  └───┴───┴───┘
    Random zeros                    Entire column removed

    Structured pruning removes:
    • Entire neurons
    • Attention heads
    • Convolutional channels
    • Layers

    Benefits: Actually speeds up inference on standard hardware!
    """)

structured_pruning()
```

### 3. Iterative Pruning (Lottery Ticket Hypothesis)

Prune, retrain, repeat—sometimes finding "winning tickets."

```python

def lottery_ticket():
    """
    The Lottery Ticket Hypothesis
    """
    print("The Lottery Ticket Hypothesis")
    "=" * 60

    print("""
    Key insight: Within a large network, there's a smaller "winning ticket"
    subnetwork that, if trained in isolation, can match the full network's performance.

    Process:

    1. Train full network
    2. Prune smallest % of weights
    3. Reset remaining weights to original initialization
    4. Retrain the sparse network
    5. Repeat

    Surprisingly, the sparse network can sometimes perform BETTER
    than the original dense network!

    This suggests overparameterization helps training but not inference.
    """)

lottery_ticket()
```

---

## Pruning in Practice

### Pruning a Simple Neural Network

```python

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def pruning_demo():
    """
    Practical pruning example with PyTorch
    """
    print("Pruning in PyTorch")
    "=" * 60

    # Create a simple linear layer
    linear = nn.Linear(10, 5)

    print(f"Original layer: {linear.weight.shape}")
    print(f"Non-zero weights: {torch.count_nonzero(linear.weight).item()}/{linear.weight.numel()}")

    # Apply L1 unstructured pruning (remove 50% of weights)
    prune.l1_unstructured(linear, name='weight', amount=0.5)

    print(f"\nAfter pruning 50% of weights:")
    print(f"Non-zero weights: {torch.count_nonzero(linear.weight).item()}/{linear.weight.numel()}")

    # The weights are still there but masked
    print(f"\nPruned weights (some zeros):")
    print(linear.weight)

    # Make pruning permanent
    prune.remove(linear, 'weight')
    print(f"\nAfter removing pruning mask (weights are now zero):")
    print(linear.weight)

pruning_demo()
```

### Pruning Attention Heads in Transformers

```python

def head_pruning():
    """
    Pruning attention heads in transformers
    """
    print("Pruning Attention Heads")
    "=" * 60

    print("""
    Transformers have multiple attention heads (e.g., 12 in BERT-base).
    Research shows many heads are redundant!

    Head importance can be measured by:

    1. MAGNITUDE: Norm of head output projections
    2. GRADIENT: Sensitivity to removal
    3. CONFIDENCE: How much the head contributes

    Example pruning strategy:

    Original: 12 heads
    ↓
    Measure importance scores
    ↓
    Remove bottom 4 heads (keep 8)
    ↓
    Fine-tune briefly to recover
    ↓
    Model is 33% smaller in attention, minimal quality loss!
    """)

head_pruning()
```

---

## Pruning vs Other Compression Methods

| Method                 | What It Does              | Size Reduction | Speedup     | Quality Impact |
| ---------------------- | ------------------------- | -------------- | ----------- | -------------- |
| Pruning (unstructured) | Remove individual weights | 50-90%         | Minimal\*   | Low            |
| Pruning (structured)   | Remove whole structures   | 30-50%         | Significant | Moderate       |
| Quantization           | Reduce bit precision      | 50-87%         | 2-4x        | Low            |
| Distillation           | Train smaller model       | 40-90%         | 2-10x       | Low-Moderate   |

- Unstructured pruning needs sparse hardware for speedup

---

## Challenges with Pruning

### The Sparsity Problem

```python

def sparsity_challenge():
    """
    Why unstructured pruning doesn't always speed things up
    """
    print("The Sparsity Challenge")
    "=" * 60

    print("""
    Unstructured pruning creates random zeros:

    [0.8, 0.0, 0.0, -0.3, 0.0, 0.6, 0.0, 0.0, 0.9, 0.0]

    Standard hardware (GPUs, CPUs) is optimized for DENSE matrices.
    Random zeros don't help—you still have to load the zeros!

    Structured pruning creates regular patterns:

    [0.8, 0.0, -0.3, 0.6, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
                      ↑                              ↑
    Can skip whole blocks!                      Real speedup!

    Specialized hardware (like NVIDIA's AMPERE) can exploit
    2:4 sparsity (2 non-zero out of every 4 weights) for real speedup.
    """)

sparsity_challenge()
```

### The Pruning-Finetuning Cycle

```python

def prune_finetune():
    """
    The prune-then-finetune cycle
    """
    print("The Prune-Finetune Cycle")
    "=" * 60

    print("""
    Pruning usually follows this pattern:

    Iteration 1:
    ┌─────────────────────────────────────────────┐
    │ 1. Train dense model    → 95% accuracy      │
    │ 2. Prune 30% weights    → 92% accuracy (drops)│
    │ 3. Fine-tune briefly    → 94% accuracy      │
    └─────────────────────────────────────────────┘

    Iteration 2:
    ┌─────────────────────────────────────────────┐
    │ 4. Prune another 30%    → 90% accuracy      │
    │ 5. Fine-tune            → 93% accuracy      │
    └─────────────────────────────────────────────┘

    Final: 60% sparsity, 93% accuracy (only 2% loss)!
    """)

prune_finetune()
```

---

## Pruning in Modern LLMs

### Real-World Examples

```python

def llm_pruning():
    """
    Pruning applied to LLMs
    """
    print("Pruning in Modern LLMs")
    "=" * 60

    examples = [
        ("BERT-Pruner", "Removes up to 40% of attention heads with minimal loss"),
        ("SparseGPT", "One-shot pruning of GPT models without retraining"),
        ("Wanda", "Pruning by weight and activation magnitudes"),
        ("LLM-Pruner", "Task-agnostic pruning for LLMs"),
        ("SliceGPT", "Removes entire layers while preserving structure")
    ]

    for technique, desc in examples:
        print(f"  • {technique}: {desc}")

llm_pruning()
```

### Results from Research

```python

def pruning_results():
    """
    Typical pruning results
    """
    print("Typical Pruning Results")
    "=" * 60

    results = [
        {"model": "BERT-base", "sparsity": "40%", "accuracy_loss": "0.5%"},
        {"model": "GPT-2", "sparsity": "50%", "perplexity_increase": "0.5"},
        {"model": "OPT-175B", "sparsity": "30%", "speedup": "1.5x"},
        {"model": "LLaMA-7B", "sparsity": "50%", "size": "7GB → 3.5GB"}
    ]

    print("Research shows impressive results:")
    for r in results:
        print(f"  • {r['model']}: {r['sparsity']} sparsity, {r.get('accuracy_loss', r.get('perplexity_increase', r.get('size', '')))}")

pruning_results()
```

---

## Pruning Cheat Sheet

| Pruning Type               | What's Removed  | Speedup   | Quality   | Best For                  |
| -------------------------- | --------------- | --------- | --------- | ------------------------- |
| Magnitude (unstructured)   | Small weights   | Minimal\* | Very good | Research, sparse hardware |
| Structured (heads/neurons) | Whole units     | Good      | Good      | CPU/GPU deployment        |
| Layer pruning              | Entire layers   | Excellent | Moderate  | Extreme compression       |
| Iterative pruning          | Gradual removal | Good      | Excellent | Maximum compression       |
| One-shot pruning           | All at once     | Fast      | Moderate  | Quick deployment          |

- Requires sparse-aware hardware

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Pruning reveals something profound about neural networks:  ║
║                                                              ║
║  🧠 OVERPARAMETERIZATION: Most of the network is             ║
║     redundant. We train big, then shrink to what matters.    ║
║                                                              ║
║  🎯 THE LOTTERY TICKET: Winning tickets exist—smaller       ║
║     subnetworks that can match full performance.             ║
║                                                              ║
║  🚀 PRACTICAL DEPLOYMENT: 50% smaller models with           ║
║     minimal quality loss. Run on devices you already have.   ║
║                                                              ║
║  ⚡ SPEUP POTENTIAL: With structured pruning, real          ║
║     speedups on existing hardware.                           ║
║                                                              ║
║  🔬 SCIENTIFIC INSIGHT: Understanding which parts           ║
║     matter helps us design better architectures.             ║
║                                                              ║
║  Pruning isn't just about making models smaller—it's        ║
║  about understanding what they actually need to work.        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Pruning removes unimportant weights or structures from neural networks—like a bonsai artist trimming away unnecessary branches, preserving the essential form while reducing size

• Magnitude pruning removes small weights, structured pruning removes entire neurons or heads—each with different tradeoffs for speedup and hardware compatibility

• The Lottery Ticket Hypothesis suggests "winning ticket" subnetworks exist—smaller networks that, if found, can match or even exceed full network performance

---

## Mental Hook

> "Pruning is like sculpting a model from a block of marble—the masterpiece was always inside; you're just chipping away everything that isn't needed. And sometimes, like the lottery ticket hypothesis suggests, the small sculpture was always meant to be."
