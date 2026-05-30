# Quantization (GPTQ, AWQ, GGUF)

## The Master Photographer Analogy

Imagine a master photographer who shoots in RAW format—every image contains millions of colors and incredible detail, but each photo is 50MB. For sharing online, they convert to JPEG—they lose some detail you'd never notice, but the file becomes 5MB. For thumbnails, they might use GIF with just 256 colors—blocky but instantly recognizable. That's quantization: trading precision for size, with different formats (GPTQ, AWQ, GGUF) optimized for different "viewing conditions" (hardware).

In LLM deployment, quantization is what makes it possible to run powerful models on consumer hardware. A 70B model in full precision needs 140GB of memory—impossible for most people. Quantized to 4-bit, it needs just 35GB, fitting on a single consumer GPU. Different quantization methods make different tradeoffs between size, speed, and quality.

---

## What Is Quantization?

### The Core Idea

Quantization reduces the number of bits used to represent each parameter in a neural network.

```text

FP32 (32-bit):   3.141592653589793  (full precision)
     ↓
FP16 (16-bit):   3.14159             (half precision)
     ↓
INT8 (8-bit):    3                    (integer, -128 to 127)
     ↓
INT4 (4-bit):    3                    (very coarse, -8 to 7)

Memory savings:
• FP32 → FP16: 50% smaller
• FP32 → INT8: 75% smaller
• FP32 → INT4: 87.5% smaller
```

```python

def quantization_intro():
    """
    The basic concept of quantization
    """
    print("Quantization: Trading Precision for Size")
    print("=" * 60)

    model_sizes = {
        "7B model FP32": "28 GB",
        "7B model FP16": "14 GB",
        "7B model INT8": "7 GB",
        "7B model INT4": "3.5 GB",
        "70B model FP16": "140 GB",
        "70B model INT4": "35 GB"
    }

    print("Memory requirements by precision:")
    for model, size in model_sizes.items():
        print(f"  • {model:15} → {size}")

quantization_intro()
```

---

## Quantization Formats Compared

### GPTQ: GPU-Optimized Quantization

GPTQ is designed for GPU inference, offering excellent performance on NVIDIA GPUs.

```python

def gptq_intro():
    """
    GPTQ: GPU-First Quantization
    """
    print("GPTQ: GPU-Optimized Quantization")
    print("=" * 60)

    print("""
    GPTQ features:

    • 4-bit quantization (mostly)
    • Optimized for GPU inference
    • Layer-wise quantization (one layer at a time)
    • Uses Hessian information to preserve important weights
    • Excellent quality retention

    Best for:
    • Running on NVIDIA GPUs
    • Maximum speed with good quality
    • When you have enough GPU memory for 4-bit

    Supported by: AutoGPTQ, ExLlama, Text Generation WebUI
    """)

gptq_intro()
```

### AWQ: Activation-Aware Quantization

AWQ protects important weights based on activation patterns, not just weight values.

```python

def awq_intro():
    """
    AWQ: Activation-Aware Quantization
    """
    print("AWQ: Activation-Aware Quantization")
    "=" * 60

    print("""
    AWQ key insight:

    Not all weights are equally important!

    Traditional quantization: Protect weights by magnitude
    AWQ: Protect weights based on activation patterns

    How it works:
    1. Run a few calibration samples through the model
    2. See which weights cause big changes when quantized
    3. Protect those weights (keep higher precision)
    4. Quantize the rest more aggressively

    Result: Better quality than GPTQ for same bit rate!

    Best for: Maximum quality at 4-bit on GPUs
    """)

awq_intro()
```

### GGUF: CPU-First Universal Format

GGUF (GPT-Generated Unified Format) is designed for CPU inference, especially with llama.cpp.

```python

def gguf_intro():
    """
    GGUF: CPU-First Universal Format
    """
    print("GGUF: The Universal CPU Format")
    "=" * 60

    print("""
    GGUF features:

    • Multiple quantization levels (2-bit to 8-bit)
    • Optimized for CPU inference (llama.cpp)
    • Runs on anything—laptops, phones, Raspberry Pi
    • Excellent memory mapping (mmap) support
    • Contains all model metadata

    Quantization types:
    • Q2_K: 2-bit (smallest, lowest quality)
    • Q3_K: 3-bit
    • Q4_K: 4-bit (sweet spot)
    • Q5_K: 5-bit (higher quality)
    • Q6_K: 6-bit
    • Q8_0: 8-bit (near lossless)

    Best for: CPU inference, edge devices, maximum compatibility
    """)

gguf_intro()
```

---

## Quality Comparison

```python

def quality_comparison():
    """
    Comparing quantization methods
    """
    print("Quantization Methods: Quality Comparison")
    "=" * 60

    methods = {
        "FP16 (baseline)": {"quality": "100%", "size": "100%", "hardware": "Any GPU"},
        "GPTQ 4-bit": {"quality": "98-99%", "size": "25%", "hardware": "NVIDIA GPU"},
        "AWQ 4-bit": {"quality": "99%", "size": "25%", "hardware": "NVIDIA GPU"},
        "GGUF Q4_K_M": {"quality": "97-98%", "size": "25%", "hardware": "CPU/Any"},
        "GGUF Q2_K": {"quality": "85-90%", "size": "15%", "hardware": "CPU/Any"},
        "INT8": {"quality": "99.5%", "size": "50%", "hardware": "Limited support"}
    }

    print("Quality retention vs FP16:")
    for method, specs in methods.items():
        print(f"\n  {method:15}")
        for key, value in specs.items():
            print(f"    • {key}: {value}")

quality_comparison()
```

---

## When to Use Each Format

```python

def when_to_use():
    """
    Choosing the right quantization format
    """
    print("Choosing the Right Quantization Format")
    "=" * 60

    scenarios = [
        {
            "use_case": "You have an NVIDIA GPU, want maximum speed",
            "choice": "GPTQ 4-bit",
            "why": "Optimized for GPU inference, excellent performance"
        },
        {
            "use_case": "You need the best possible quality at 4-bit",
            "choice": "AWQ 4-bit",
            "why": "Activation-aware, preserves important weights better"
        },
        {
            "use_case": "You're running on CPU (laptop, desktop)",
            "choice": "GGUF Q4_K_M",
            "why": "llama.cpp is fastest on CPU, great quality/size balance"
        },
        {
            "use_case": "You have very limited RAM (phone, Raspberry Pi)",
            "choice": "GGUF Q2_K or Q3_K",
            "why": "Smallest possible size, runs anywhere"
        },
        {
            "use_case": "You need maximum quality, have GPU memory",
            "choice": "FP16 or GGUF Q8_0",
            "why": "Near-lossless, best performance"
        }
    ]

    for scenario in scenarios:
        print(f"\n  • {scenario['use_case']}")
        print(f"    → Choose: {scenario['choice']}")
        print(f"    → Why: {scenario['why']}")

when_to_use()
```

---

## Practical Examples

### Loading a GPTQ Model

```python

def gptq_example():
    """
    Loading a GPTQ model with AutoGPTQ
    """
    print("Loading a GPTQ Model")
    "=" * 60

    print("""
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer

    # Load 4-bit GPTQ model
    model = AutoGPTQForCausalLM.from_quantized(
        "TheBloke/Llama-2-7B-GPTQ",
        device="cuda:0",
        use_triton=False,
        use_safetensors=True,
        disable_exllama=False
    )

    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")

    # Use normally
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))
    """)

gptq_example()
```

### Loading a GGUF Model with llama.cpp

```python

def gguf_example():
    """
    Loading a GGUF model with llama-cpp-python
    """
    print("Loading a GGUF Model")
    "=" * 60

    print("""
    from llama_cpp import Llama

    # Load Q4_K_M GGUF model
    llm = Llama(
        model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx=4096,           # Context window
        n_threads=8,           # CPU threads
        n_gpu_layers=-1,       # Use GPU if available (-1 = all layers)
        verbose=False
    )

    # Generate
    output = llm(
        "Q: What is the capital of France? A:",
        max_tokens=100,
        temperature=0.7,
        echo=True
    )

    print(output["choices"][0]["text"])
    """)

gguf_example()
```

### Loading an AWQ Model

```python

def awq_example():
    """
    Loading an AWQ model
    """
    print("Loading an AWQ Model")
    "=" * 60

    print("""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    # Load AWQ model
    model = AutoAWQForCausalLM.from_quantized(
        "TheBloke/Llama-2-7B-AWQ",
        fuse_layers=True,
        device="cuda:0"
    )

    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-AWQ")

    # Generate
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))
    """)

awq_example()
```

---

## Quantization Levels in GGUF

```python

def gguf_levels():
    """
    GGUF quantization levels explained
    """
    print("GGUF Quantization Levels")
    "=" * 60

    levels = [
        ("Q2_K", "2-bit", "~2.2 bpw", "Smallest, for extreme memory constraints"),
        ("Q3_K_S", "3-bit small", "~3.0 bpw", "Very small, low quality"),
        ("Q3_K_M", "3-bit medium", "~3.3 bpw", "Balance for 3-bit"),
        ("Q3_K_L", "3-bit large", "~3.6 bpw", "Best 3-bit quality"),
        ("Q4_K_S", "4-bit small", "~4.0 bpw", "Good quality, smaller size"),
        ("Q4_K_M", "4-bit medium", "~4.5 bpw", "SWEET SPOT - best balance"),
        ("Q5_K_S", "5-bit small", "~5.0 bpw", "High quality"),
        ("Q5_K_M", "5-bit medium", "~5.3 bpw", "Very high quality"),
        ("Q6_K", "6-bit", "~6.0 bpw", "Near lossless"),
        ("Q8_0", "8-bit", "~8.0 bpw", "Essentially lossless")
    ]

    print("GGUF quantization levels (lower = smaller but worse):")
    for level, bits, bpw, desc in levels:
        bar = "█" * int((8 - float(bpw.split('~')[1])) * 3)
        print(f"  {level:8} ({bits:6}) {bpw:6} {bar} {desc}")

gguf_levels()
```

---

## Quantization Cheat Sheet

| Format      | Target Hardware | Typical Bits | Quality | Speed  | Best Use Case            |
| ----------- | --------------- | ------------ | ------- | ------ | ------------------------ |
| GPTQ        | NVIDIA GPU      | 4-bit        | 98-99%  | ⭐⭐⭐ | GPU inference            |
| AWQ         | NVIDIA GPU      | 4-bit        | 99%     | ⭐⭐⭐ | Maximum quality on GPU   |
| GGUF Q4_K_M | CPU/Any         | 4-bit        | 97-98%  | ⭐⭐   | Universal, balanced      |
| GGUF Q2_K   | CPU/Any         | 2-bit        | 85-90%  | ⭐⭐   | Extreme memory limits    |
| FP16        | GPU             | 16-bit       | 100%    | ⭐⭐⭐ | Reference, high-end GPUs |
| INT8        | Specialized     | 8-bit        | 99.5%   | ⭐⭐   | Specific hardware        |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Quantization is what makes LLMs accessible to everyone:    ║
║                                                              ║
║  💻 CONSUMER HARDWARE: Run 70B models on a single           ║
║     consumer GPU (RTX 3090/4090) instead of needing         ║
║     $40,000 server racks.                                   ║
║                                                              ║
║  📱 EDGE DEPLOYMENT: Run models on phones, laptops,         ║
║     Raspberry Pis. AI anywhere, anytime.                    ║
║                                                              ║
║  🚀 SPEED: Smaller models = faster inference. 4-bit         ║
║     models can be 3-4x faster than FP16.                    ║
║                                                              ║
║  💰 COST: 4-bit models cost 4x less to serve. At           ║
║     scale, this is millions in savings.                      ║
║                                                              ║
║  🔄 CHOICE: Different formats let you optimize for          ║
║     your specific hardware and quality needs.                ║
║                                                              ║
║  Without quantization, LLMs would remain in the cloud,      ║
║  accessible only to big tech. With it, anyone can run       ║
║  state-of-the-art AI on their own machine.                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• GPTQ is optimized for NVIDIA GPUs—offering excellent 4-bit performance with 98-99% quality retention, the go-to for GPU inference

• AWQ uses activation patterns to protect important weights—achieving slightly better quality than GPTQ at the same bit rate, ideal when quality is paramount

• GGUF is the universal CPU-friendly format—with multiple quantization levels (Q2_K to Q8_0), it runs anywhere and offers the best compatibility

---

## Mental Hook

> "Choosing a quantization format is like choosing a file format for your photos—GPTQ is like high-quality JPEG for sharing online (GPU), AWQ is like carefully retouched images preserving important details, and GGUF is like a universal format that plays on any device (CPU)."
