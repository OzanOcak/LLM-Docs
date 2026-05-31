# PagedAttention / vLLM

## The Library Stack Analogy

Imagine a library where each reader gets their own personal copy of every book they want to read. If 100 people are reading different books, you need 100 copies—wasting enormous space. A smarter library uses a shared stack: readers have a catalog telling them where to find pages, and multiple people can read the same physical copy simultaneously. That's PagedAttention: instead of storing separate KV caches for each request, it uses virtual memory techniques to share physical memory blocks efficiently.

vLLM is the system that implements PagedAttention, and it's revolutionized LLM serving. It increases throughput by 10-24x compared to traditional systems, making it possible to serve millions of users with reasonable hardware. It's now the standard for production LLM deployment.

---

## The Problem: KV Cache Fragmentation

### Traditional KV Cache Wastes Memory

```python

def fragmentation_problem():
    """
    How traditional KV cache wastes memory
    """
    print("The Problem: KV Cache Fragmentation")
    print("=" * 60)

    print("""
    Traditional approach: Each request gets contiguous memory

    Request A (100 tokens): [KA1,KA2,...,KA100][VA1,VA2,...,VA100]
    Request B (100 tokens): [KB1,KB2,...,KB100][VB1,VB2,...,VB100]
    Request C (100 tokens): [KC1,KC2,...,KC100][VC1,VC2,...,VC100]

    PROBLEMS:

    1. INTERNAL FRAGMENTATION: Each request pre-allocates maximum space
       • If request only uses 50 tokens, the other 50 slots are wasted

    2. EXTERNAL FRAGMENTATION: When requests finish, they leave holes
       • Memory becomes checkerboard of used/free blocks
       • New requests may not fit in holes

    3. NO SHARING: Even if requests read same prefix, each has its own copy
       • "The capital of France is" repeated in thousands of requests
    """)

fragmentation_problem()
```

### Visualizing Fragmentation

```text

Memory after several requests:

[Request1: ████████████░░░░░░]  (finished, 50% empty)
[Request2: ██████████████████]  (active, full)
[Request3: ████████░░░░░░░░░░]  (finished, 60% empty)
[Request4: ██████████████████]  (active, full)
[Free:    ░░░░░░░░░░░░░░░░░░]

Problem: New request needs 100 contiguous blocks
          Can't fit in scattered holes!
          Must wait or waste memory.
```

---

## The Solution: PagedAttention

### Virtual Memory for KV Cache

```python

def pagedattention_intro():
    """
    The basic concept of PagedAttention
    """
    print("PagedAttention: Virtual Memory for KV Cache")
    "=" * 60

    print("""
    Like OS virtual memory, PagedAttention:

    1. Divides KV cache into fixed-size blocks (pages)
    2. Each request has a logical table mapping positions to physical blocks
    3. Blocks can be shared across requests
    4. Physical memory is allocated on demand

    ┌─────────────────────────────────────────────────────┐
    │ Logical view (per request):                         │
    │ [Token1]─┐                                          │
    │ [Token2]─┼──► Maps to physical blocks               │
    │ [Token3]─┘                                          │
    └─────────────────────────────────────────────────────┘
                           ↓
    ┌─────────────────────────────────────────────────────┐
    │ Physical memory (shared):                           │
    │ Block A: [K1,V1,K2,V2]  ← used by Request1, Request2│
    │ Block B: [K3,V3,K4,V4]  ← used by Request1          │
    │ Block C: [K5,V5,K6,V6]  ← used by Request2, Request3│
    └─────────────────────────────────────────────────────┘
    """)

pagedattention_intro()
```

### How PagedAttention Works

```python

def pagedattention_mechanism():
    """
    Step-by-step PagedAttention
    """
    print("PagedAttention: Step by Step")
    "=" * 60

    print("""
    Step 1: Request arrives
    ─────────────────────────────────────────────────
    Logical block table: empty
    Physical blocks allocated: none

    Step 2: First tokens generated
    ─────────────────────────────────────────────────
    Need space for tokens 1-4 → allocate Block 17
    Logical table: [Token1-4] → Block 17

    Step 3: More tokens generated
    ─────────────────────────────────────────────────
    Tokens 5-8 need space → allocate Block 42
    Logical table: [Token1-4] → Block 17
                   [Token5-8] → Block 42

    Step 4: Another request shares prefix
    ─────────────────────────────────────────────────
    New request also needs first 4 tokens
    Logical table: [Token1-4] → Block 17 (shared!)
                   [Token5-8] → Block 56 (new)

    Memory saved: 4 tokens worth of cache shared!
    """)

pagedattention_mechanism()
```

---

## Benefits of PagedAttention

### 1. Zero Fragmentation

```python

def no_fragmentation():
    """
    How PagedAttention eliminates fragmentation
    """
    print("Benefit 1: Zero Fragmentation")
    "=" * 60

    print("""
    Traditional memory (contiguous):
    [Req1: ████████████████] [Free: ░░░░] [Req2: ██████████]

    Can't fit new 15-block request in 4-block hole!

    PagedAttention (block-based):
    Physical blocks: [A][B][C][D][E][F][G][H][I][J][K][L]
                     ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑

    New request takes ANY free blocks: A, C, F, H, J, L
    └── No need for contiguous space!

    Fragmentation: 0%
    """)

no_fragmentation()
```

### 2. Block Sharing

```python

def block_sharing():
    """
    How blocks can be shared across requests
    """
    print("Benefit 2: Block Sharing")
    "=" * 60

    print("""
    Common prefixes are shared automatically!

    Request 1: "The capital of France is Paris"
    Request 2: "The capital of France is a beautiful city"

    Both share: "The capital of France is"

    Traditional KV cache:
    Req1: [K1,V1][K2,V2][K3,V3][K4,V4][K5,V5][K6,V6]
    Req2: [K1,V1][K2,V2][K3,V3][K4,V4][K5,V5][K6,V6][K7,V7][K8,V8]
           ↑      ↑      ↑      ↑      ↑      ↑
           Duplicate! All 6 blocks duplicated

    PagedAttention:
    Req1: Block A (K1-2), Block B (K3-4), Block C (K5-6)
    Req2: Block A (shared!), Block B (shared!), Block C (shared!), Block D (K7-8)

    Memory saved: 6 blocks → 4 blocks (33% savings)
    """)

block_sharing()
```

### 3. Copy-on-Write

```python

def copy_on_write():
    """
    How copy-on-write enables efficient modification
    """
    print("Benefit 3: Copy-on-Write")
    "=" * 60

    print("""
    When sharing blocks, what if one request diverges?

    Request 1: "The capital of France is Paris"
    Request 2: "The capital of France is Lyon"

    They share first 6 tokens, then differ at token 7.

    Copy-on-write:
    1. Initially, both share all blocks
    2. When Request 2 needs a different token 7:
       - Create new block for tokens 7-8
       - Copy original values (if needed)
       - Update logical mapping

    Original blocks remain shared for prefix!

    Memory efficient + no recomputation of shared prefix!
    """)

copy_on_write()
```

---

## vLLM: The System

### What Is vLLM?

```python

def vllm_intro():
    """
    vLLM: The implementation
    """
    print("vLLM: PagedAttention in Practice")
    "=" * 60

    print("""
    vLLM is a high-throughput serving system that combines:

    1. PagedAttention for memory efficiency
    2. Continuous batching for throughput
    3. Efficient scheduler
    4. CUDA kernels optimized for attention

    Performance gains:
    • 10-24x higher throughput than HuggingFace
    • 2-4x higher than traditional continuous batching
    • 90%+ GPU utilization

    Used by: OpenAI, Microsoft, Anthropic, and thousands of companies
    """)

vllm_intro()
```

### Performance Comparison

```python

def vllm_performance():
    """
    vLLM vs traditional systems
    """
    print("vLLM Performance Comparison")
    "=" * 60

    import matplotlib.pyplot as plt
    import numpy as np

    systems = [
        {"name": "HuggingFace (naive)", "throughput": 1},
        {"name": "Continuous batching", "throughput": 5},
        {"name": "vLLM", "throughput": 15}
    ]

    print("Relative throughput (higher is better):")
    for sys in systems:
        bar = "█" * sys["throughput"]
        print(f"  {sys['name']:25} {sys['throughput']:2}x {bar}")

    print("\nReal numbers (Llama 7B, A100):")
    print("  • HuggingFace: 100 tokens/second")
    print("  • Continuous batching: 500 tokens/second")
    print("  • vLLM: 1,500 tokens/second")

vllm_performance()
```

---

## Using vLLM

### Basic Usage

```python

def vllm_usage():
    """
    How to use vLLM
    """
    print("Using vLLM")
    "=" * 60

    print("""
    from vllm import LLM, SamplingParams

    # Load model
    llm = LLM(model="meta-llama/Llama-2-7b-hf")

    # Generate
    prompts = [
        "The capital of France is",
        "The meaning of life is",
        "Write a poem about"
    ]

    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(output.prompt + output.outputs[0].text)

    # vLLM automatically handles:
    # • PagedAttention memory management
    # • Continuous batching
    # • KV cache optimization
    # • Parallel generation
    """)

vllm_usage()
```

### Advanced Features

```python

def vllm_advanced():
    """
    Advanced vLLM features
    """
    print("Advanced vLLM Features")
    "=" * 60

    features = [
        "Parallel sampling (multiple outputs per prompt)",
        "Beam search",
        "Prompt caching",
        "Prefix caching (across requests)",
        "Tensor parallelism (multiple GPUs)",
        "Quantization support (GPTQ, AWQ)",
        "OpenAI-compatible API server"
    ]

    for f in features:
        print(f"  • {f}")

vllm_advanced()
```

---

## Memory Efficiency Comparison

| Metric               | Traditional       | vLLM      | Improvement              |
| -------------------- | ----------------- | --------- | ------------------------ |
| Memory fragmentation | 20-40%            | <1%       | 40x better               |
| Prefix sharing       | No                | Yes       | Up to 50% memory savings |
| KV cache size        | Pre-allocated max | On-demand | 2-4x smaller             |
| Throughput           | 1x                | 10-24x    | 24x better               |
| Batch size           | Limited           | Large     | 5x larger                |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  vLLM and PagedAttention have transformed LLM serving:      ║
║                                                              ║
║  🚀 THROUGHPUT: 24x more requests on the same hardware.    ║
║     One vLLM server = 24 traditional servers.               ║
║                                                              ║
║  💰 COST: 24x lower cost per request. At scale,           ║
║     this saves millions in GPU costs.                       ║
║                                                              ║
║  📈 SCALABILITY: Serve millions of users with              ║
║     reasonable infrastructure.                              ║
║                                                              ║
║  🧠 MEMORY EFFICIENCY: PagedAttention eliminates            ║
║     fragmentation and enables sharing, doing more with less.║
║                                                              ║
║  🔧 ADOPTION: Used by OpenAI, Microsoft, Anthropic—        ║
║     it's the industry standard for production LLMs.         ║
║                                                              ║
║  Before vLLM, serving LLMs at scale was prohibitively       ║
║  expensive. After vLLM, it's economical and practical.      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• PagedAttention applies virtual memory techniques to KV cache management—dividing cache into fixed-size blocks, eliminating fragmentation and enabling sharing across requests

• vLLM implements PagedAttention with continuous batching—achieving 10-24x higher throughput than traditional systems through better memory utilization

• Key benefits include zero fragmentation, block sharing for common prefixes, copy-on-write for divergent sequences, and dramatically lower memory usage—making large-scale LLM serving economically viable

---

## Mental Hook

> "PagedAttention is like a library where instead of giving every reader their own copy of a book, you give them a map showing where the shared physical copies are—suddenly you can serve 24x more readers with the same number of books."
