# Model Compression

## The Encyclopedia to Pocket Dictionary Analogy

Imagine you have a 30-volume encyclopedia set containing all human knowledge. It's comprehensive but impossible to carry around. Now imagine compressing it into a pocket dictionary—you lose some nuance and rare words, but you keep the essential information in a form you can actually use day-to-day. That's model compression: taking a massive LLM (like GPT-4) and making it smaller, faster, and more deployable while preserving as much capability as possible.

In real-world applications, model compression is essential. A 175B parameter model can't run on your phone, can't respond in milliseconds, and costs a fortune to serve. Compression techniques like quantization, pruning, and distillation make LLMs practical for real-world use.

---

## What Is Model Compression?

### The Core Idea

Model compression reduces the size and computational cost of a model while trying to maintain its performance.

```text

Original Model (175B)                 Compressed Model (7B)
┌─────────────────────┐               ┌─────────────────────┐
│ • 175B parameters   │    compress   │ • 7B parameters     │
│ • 350GB memory      │    ────────→   │ • 14GB memory      │
│ • 100ms inference   │               │ • 10ms inference    │
│ • 95% accuracy      │               │ • 92% accuracy      │
└─────────────────────┘               └─────────────────────┘
```

Trade-off: Slightly lower quality for MUCH better efficiency!

```python

def compression_intro():
    """
    The basic concept of model compression
    """
    print("Model Compression: Smaller, Faster, Cheaper")
    print("=" * 60)

    print("""
    Why compress?

    • DEPLOYMENT: Run on phones, edge devices, GPUs with limited memory
    • SPEED: Faster inference = better user experience
    • COST: Cheaper to serve (pay-per-token adds up!)
    • ENERGY: Lower power consumption, greener AI
    • LATENCY: Real-time applications need quick responses
    """)

compression_intro()
```

---

## Technique 1: Quantization

### What Is Quantization?

Quantization reduces the precision of the numbers used to store model weights.

```python

def quantization_intro():
    """
    Quantization: Using fewer bits
    """
    print("Quantization: Using Fewer Bits")
    print("=" * 60)

    print("""
    Number precision levels:

    FP32 (32-bit):  -3.4028235e+38  (full precision)
    FP16 (16-bit):  -65504 to 65504  (half precision)
    INT8  (8-bit):  -128 to 127       (integer)
    INT4  (4-bit):  -8 to 7           (very low precision)

    Memory savings:
    • FP32 → FP16: 50% reduction
    • FP32 → INT8: 75% reduction
    • FP32 → INT4: 87.5% reduction
    """)

quantization_intro()
```

### Quantization Example

```python

def quantization_example():
    """
    Concrete quantization example
    """
    print("Quantization in Action")
    print("=" * 60)

    import numpy as np

    # Original FP32 weights (simplified)
    original = np.array([0.1234, -0.5678, 0.9101, -0.1121, 0.3141], dtype=np.float32)

    print(f"Original FP32 weights: {original}")
    print(f"Memory: {original.nbytes} bytes\n")

    # Find min/max for quantization
    min_val, max_val = original.min(), original.max()

    # Quantize to INT8
    scale = (max_val - min_val) / 255
    zero_point = -min_val / scale

    quantized = np.round((original - min_val) / scale).astype(np.int8)

    print(f"Quantized INT8: {quantized}")
    print(f"Memory: {quantized.nbytes} bytes ({quantized.nbytes/original.nbytes*100:.0f}%)")

    # Dequantize (approximate)
    dequantized = min_val + quantized * scale
    print(f"\nDequantized (approximate): {dequantized}")
    print(f"Error: {np.abs(original - dequantized).mean():.6f}")

quantization_example()
```

### Quantization Formats

```python

def quantization_formats():
    """
    Popular quantization formats
    """
    print("Popular Quantization Formats")
    "=" * 60

    formats = {
        "GPTQ": "4-bit quantization, popular for GPU inference",
        "AWQ": "Activation-aware quantization, preserves accuracy",
        "GGUF": "llama.cpp format, CPU-friendly, multiple bit levels",
        "Bitsandbytes": "8-bit and 4-bit for Hugging Face models",
        "NF4": "NormalFloat 4-bit, used in QLoRA"
    }

    print("Different formats for different needs:")
    for fmt, desc in formats.items():
        print(f"  • {fmt}: {desc}")

quantization_formats()
```

---

## Technique 2: Pruning

### What Is Pruning?

Pruning removes less important weights or neurons from the network.

```python

def pruning_intro():
    """
    Pruning: Removing Unimportant Parts
    """
    print("Pruning: Cutting the Dead Wood")
    "=" * 60

    print("""
    Magnitude-based pruning:

    Original weights: [0.5, -0.02, 0.8, 0.001, -0.6, 0.0003]
                         ↓      ↓      ↓      ↓       ↓       ↓
    Keep if |weight| > 0.01?  ✓      ✗      ✓      ✗       ✓      ✗

    Result: [0.5, 0, 0.8, 0, -0.6, 0]

    Pruning can be:
    • Unstructured: Remove individual weights (sparse matrix)
    • Structured: Remove entire neurons/channels (dense matrix)

    Sparse matrices are harder to accelerate, but save memory!
    """)

pruning_intro()
```

### Pruning Example

```python

def pruning_example():
    """
    Simple pruning demonstration
    """
    print("Pruning in Action")
    "=" * 60

    import numpy as np

    # Simulated weight matrix
    weights = np.array([
        [0.8, 0.01, -0.7, 0.02],
        [-0.6, 0.001, 0.9, -0.03],
        [0.02, 0.5, -0.008, 0.7],
        [0.009, -0.4, 0.01, 0.6]
    ])

    print("Original weights:")
    print(weights)
    print(f"Non-zero: {np.count_nonzero(weights)}/{weights.size}")

    # Prune weights below threshold
    threshold = 0.05
    pruned = weights.copy()
    pruned[np.abs(pruned) < threshold] = 0

    print(f"\nAfter pruning (threshold={threshold}):")
    print(pruned)
    print(f"Non-zero: {np.count_nonzero(pruned)}/{pruned.size} ({np.count_nonzero(pruned)/pruned.size*100:.0f}%)")

pruning_example()
```

---

## Technique 3: Distillation

### What Is Distillation?

Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" model.

```python

def distillation_intro():
    """
    Distillation: Learning from the Master
    """
    print("Distillation: Student Learning from Teacher")
    "=" * 60

    print("""
    Traditional training:           Distillation:
    ┌──────────────┐               ┌──────────────┐
    │   Labels     │               │   Teacher     │
    │   (one-hot)  │               │   Model       │
    └──────┬───────┘               └──────┬───────┘
           ↓                              ↓
    ┌──────────────┐               ┌──────────────┐
    │   Student    │               │   Student    │
    │    Model     │               │    Model     │
    └──────────────┘               └──────────────┘

    Learn from hard labels          Learn from soft probabilities
                                    (richer information!)

    Student learns not just the answer,
    but the teacher's reasoning patterns.
    """)

distillation_intro()
```

### Distillation Example

```python

def distillation_example():
    """
    How distillation works
    """
    print("Distillation in Action")
    "=" * 60

    import numpy as np

    # Teacher's soft probabilities for "What animal is this?"
    teacher_probs = np.array([0.7, 0.2, 0.05, 0.03, 0.02])  # cat, dog, rabbit, etc.

    # Hard label (traditional training)
    hard_label = np.array([1, 0, 0, 0, 0])  # one-hot "cat"

    # Temperature scaling (makes soft labels softer)
    temperature = 2.0
    soft_labels = np.exp(np.log(teacher_probs + 1e-10) / temperature)
    soft_labels = soft_labels / soft_labels.sum()

    print(f"Teacher probabilities: {teacher_probs}")
    print(f"Hard label:            {hard_label}")
    print(f"Soft labels (T={temperature}): {np.round(soft_labels, 3)}")

    print("\nSoft labels tell student:")
    print("• It's definitely a cat (0.7)")
    print("• Could be a dog (0.2)")
    print("• Might be other animals (0.1)")
    print("\nStudent learns the relationships between classes!")

distillation_example()
```

### Famous Distilled Models

```python

def distilled_models():
    """
    Popular distilled models
    """
    print("Famous Distilled Models")
    "=" * 60

    models = {
        "DistilBERT": "40% smaller, 95% of BERT performance, 60% faster",
        "TinyBERT": "7.5x smaller, 9.4x faster, 96% of BERT",
        "MobileBERT": "4.3x smaller, 5.5x faster",
        "MiniLM": "Microsoft's distillation, used in many applications",
        "LLM distillation": "Training smaller models (e.g., Phi-3) on GPT-4 outputs"
    }

    for model, desc in models.items():
        print(f"  • {model}: {desc}")

distilled_models()
```

---

## Compression Technique Comparison

| Technique              | Size Reduction | Speedup | Quality Loss      | Best For           |
| ---------------------- | -------------- | ------- | ----------------- | ------------------ |
| Quantization (FP16)    | 50%            | 1.5-2x  | Negligible        | General purpose    |
| Quantization (INT8)    | 75%            | 2-3x    | Small             | GPU inference      |
| Quantization (INT4)    | 87.5%          | 3-4x    | Moderate          | Edge devices       |
| Pruning (unstructured) | Variable       | Limited | Small             | Research           |
| Pruning (structured)   | 30-50%         | 1.5-2x  | Moderate          | CPU deployment     |
| Distillation           | 40-90%         | 2-10x   | Small to moderate | New smaller models |

---

## Putting It All Together

### Compression Pipeline

```python

def compression_pipeline():
    """
    Complete compression workflow
    """
    print("Model Compression Pipeline")
    "=" * 60

    steps = [
        ("1. Start with trained model", "GPT-4, Llama 70B, etc."),
        ("2. Distillation (optional)", "Train smaller student on teacher outputs"),
        ("3. Pruning", "Remove unimportant weights"),
        ("4. Quantization", "Reduce precision (FP16 → INT8 → INT4)"),
        ("5. Format conversion", "Convert to GGUF, GPTQ, etc."),
        ("6. Deployment", "Run on target hardware")
    ]

    for step, desc in steps:
        print(f"\n{step}")
        print(f"  {desc}")

compression_pipeline()
```

### Real-World Example: Llama.cpp

```python

def llama_cpp():
    """
    Llama.cpp: Practical compression
    """
    print("Llama.cpp: Making LLMs Run Anywhere")
    "=" * 60

    print("""
    Llama.cpp quantizes models to run on consumer hardware:

    Original Llama 70B (FP16):
    • 140 GB memory
    • Needs multiple GPUs
    • $$$$

    Quantized to Q4_K_M (4-bit):
    • 40 GB memory
    • Runs on single 48GB GPU or multiple MacBooks
    • Actually usable!

    Quality loss: ~2-5% on benchmarks, but runs anywhere!
    """)

llama_cpp()
```

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Model compression isn't optional—it's essential:           ║
║                                                              ║
║  📱 EDGE DEPLOYMENT: Your phone can't run 175B models.      ║
║     Compression brings AI to devices, enabling privacy.      ║
║                                                              ║
║  ⚡ REAL-TIME APPS: 100ms vs 10ms latency makes or          ║
║     breaks user experience. Compression enables speed.       ║
║                                                              ║
║  💰 COST SAVINGS: Serving a compressed model costs          ║
║     10-100x less. At scale, this is millions of dollars.    ║
║                                                              ║
║  🌱 ENVIRONMENT: Smaller models use less energy.            ║
║     Green AI matters for the planet.                        ║
║                                                              ║
║  🚀 ACCESSIBILITY: Not everyone has 8× A100 GPUs.           ║
║     Compression democratizes AI.                            ║
║                                                              ║
║  The trade-off is small—a few percent quality loss          ║
║  for 10x better efficiency. Worth it every time.            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Quantization reduces numerical precision—from 32-bit to 16-bit, 8-bit, or even 4-bit, cutting memory usage by 50-87% with minimal quality loss

• Pruning removes unimportant weights—based on magnitude or other criteria, creating sparse networks that are smaller but harder to accelerate

• Distillation trains smaller "student" models to mimic larger "teachers"—creating entirely new, efficient models that retain most of the teacher's capabilities

---

## Mental Hook

> "Model compression is like turning a 30-volume encyclopedia into a pocket dictionary—you lose some rare words and nuance, but what you gain is something you can actually carry with you and use every day."
