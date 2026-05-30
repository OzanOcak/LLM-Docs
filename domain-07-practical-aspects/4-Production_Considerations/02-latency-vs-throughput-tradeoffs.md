# Latency vs Throughput Tradeoffs

## The Highway Analogy

Imagine a highway connecting two cities. You can design it for speed (latency) or for capacity (throughput). A racetrack with no speed limits gets individual cars to their destination fast—but only one car at a time. A 10-lane highway moves thousands of cars per hour, but each individual trip takes longer due to merging and traffic. That's the latency vs throughput tradeoff: you can optimize for fast individual responses or for handling many requests simultaneously, but rarely both.

In LLM serving, this tradeoff is everywhere. A single user wants their response instantly (low latency). A business serving millions of users needs to handle high volume (high throughput). The art of system design is balancing these competing goals based on your priorities.

---

## What Are Latency and Throughput?

### The Core Concepts

```python

def latency_throughput_intro():
    """
    Defining latency and throughput
    """
    print("Latency vs Throughput: The Fundamental Tradeoff")
    print("=" * 60)

    print("""
    LATENCY: Time for ONE request to complete
    ─────────────────────────────────────────────
    • Measured in milliseconds or seconds
    • What the user experiences
    • "How long do I wait for my response?"

    THROUGHPUT: Number of requests processed per unit time
    ─────────────────────────────────────────────────────
    • Measured in requests per second (RPS) or tokens per second
    • What the business cares about
    • "How many users can we serve simultaneously?"

    ┌─────────────────────────────────────────────────────┐
    │  Latency = 100ms per request                         │
    │  Throughput = 10 requests/second                     │
    │  → Can handle 10 users at once, each waits 100ms    │
    └─────────────────────────────────────────────────────┘
    """)

latency_throughput_intro()
```

### The Relationship

```python

def relationship():
    """
    How latency and throughput relate
    """
    print("The Relationship: You Can't Have Both")
    "=" * 60

    import math

    # Theoretical relationship
    latencies = [10, 20, 50, 100, 200, 500, 1000]  # ms

    print("Lower latency usually means lower throughput:")
    print("Latency (ms) | Max Throughput (RPS) | Tradeoff")
    print("-" * 55)

    for lat in latencies:
        # Simplified: throughput limited by latency + overhead
        max_throughput = 1000 / lat  # theoretical max
        if lat < 50:
            tradeoff = "🚀 Low latency, low throughput"
        elif lat < 200:
            tradeoff = "⚖️ Balanced"
        else:
            tradeoff = "🐢 High latency, high throughput possible with batching"

        print(f"     {lat:4d}     |       {max_throughput:5.1f}        | {tradeoff}")

relationship()
```

---

## The Tradeoff in Practice

### Scenario 1: Optimizing for Latency

```python

def latency_focused():
    """
    When low latency is critical
    """
    print("Scenario 1: Latency-Optimized System")
    "=" * 60

    print("""
    USE CASES:
    • Chat applications (real-time conversation)
    • Voice assistants (must respond quickly)
    • Interactive coding assistants
    • Any user-facing feature

    OPTIMIZATIONS:
    • Small batch sizes (often batch size = 1)
    • No waiting to fill batches
    • Faster but more expensive hardware
    • Speculative decoding
    • Streaming responses

    EXAMPLE: ChatGPT
    • Latency target: < 500ms first token
    • Throughput: Handles millions of users via horizontal scaling
    • Cost: Higher per request, but better UX
    """)

latency_focused()
```

### Scenario 2: Optimizing for Throughput

```python

def throughput_focused():
    """
    When high throughput is critical
    """
    print("Scenario 2: Throughput-Optimized System")
    "=" * 60

    print("""
    USE CASES:
    • Batch processing (summarize 10,000 documents)
    • Data labeling/annotation
    • Offline jobs
    • Internal batch jobs

    OPTIMIZATIONS:
    • Large batch sizes (32-256 requests)
    • Wait to fill batches (increased latency)
    • Continuous batching
    • Quantization for memory efficiency
    • PagedAttention for larger batches

    EXAMPLE: Overnight batch processing
    • Latency: Hours or days (acceptable)
    • Throughput: 1000x higher than real-time
    • Cost: 10-100x cheaper per request
    """)

throughput_focused()
```

---

## The Math of Batching

### How Batch Size Affects Latency and Throughput

```python

def batch_math():
    """
    Calculating the tradeoff
    """
    print("The Math of Batching")
    "=" * 60

    # Fixed parameters
    time_per_token = 0.01  # 10ms per token
    avg_tokens = 100  # average response length
    base_overhead = 0.05  # 50ms fixed overhead

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print("Batch | Latency per req | Throughput | Efficiency")
    print("-" * 55)

    for bs in batch_sizes:
        # Time to process batch: overhead + (tokens × time_per_token × batch_size)
        batch_time = base_overhead + (avg_tokens * time_per_token * bs)
        latency_per_req = batch_time  # all requests wait for full batch
        throughput = bs / batch_time  # requests per second

        print(f"  {bs:2d}   |    {latency_per_req*1000:5.0f}ms    |   {throughput:5.1f} RPS  |", end=" ")

        if bs <= 4:
            print("🚀 Low latency")
        elif bs <= 16:
            print("⚖️ Balanced")
        else:
            print("📦 High throughput")

batch_math()
```

### Visualizing the Tradeoff

```text

Latency vs Throughput by Batch Size
─────────────────────────────────────────────────

Latency (ms)        Throughput (RPS)
    ↑                    ↑
 2000┤        📦        60┤              📦
     │        64          │              64
 1500┤       📦          45┤            📦
     │       32           │            32
 1000┤     📦            30┤         📦
     │     16              │         16
 500┤   📦               15┤      📦
     │   8                 │      8
   0┼───┼───┼───┼───┼─── 0┼───┼───┼───┼───┼───
       1   2   4   8  16      1   2   4   8  16
       Batch Size              Batch Size

As batch size increases:
• Latency goes UP (each request waits longer)
• Throughput goes UP (more requests processed per second)
• Choose based on your priority!
```

---

## Real-World Examples

### API Tiers and Their Tradeoffs

```python

def api_tiers():
    """
    Different service tiers for different needs
    """
    print("API Service Tiers")
    "=" * 60

    tiers = [
        {
            "tier": "Real-time",
            "latency": "< 500ms",
            "throughput": "Moderate",
            "price": "$$$",
            "use": "Chat, interactive apps"
        },
        {
            "tier": "Standard",
            "latency": "< 2s",
            "throughput": "High",
            "price": "$$",
            "use": "General purpose"
        },
        {
            "tier": "Batch",
            "latency": "Hours",
            "throughput": "Very High",
            "price": "$",
            "use": "Offline processing"
        }
    ]

    for t in tiers:
        print(f"\n  • {t['tier']:10} | Latency: {t['latency']:10} | Throughput: {t['throughput']:12} | {t['price']} | {t['use']}")

api_tiers()
```

### Cloud Provider Options

```python

def cloud_options():
    """
    Different instance types for different tradeoffs
    """
    print("Cloud Instance Choices")
    "=" * 60

    instances = [
        {
            "type": "A100-80GB",
            "latency": "Excellent",
            "throughput": "High",
            "cost": "$40/hour",
            "best": "Real-time, large models"
        },
        {
            "type": "L4",
            "latency": "Good",
            "throughput": "Good",
            "cost": "$10/hour",
            "best": "Balanced workloads"
        },
        {
            "type": "T4",
            "latency": "Moderate",
            "throughput": "Moderate",
            "cost": "$3/hour",
            "best": "Batch, small models"
        },
        {
            "type": "CPU only",
            "latency": "Poor",
            "throughput": "Low",
            "cost": "$1/hour",
            "best": "Development, tiny models"
        }
    ]

    for i in instances:
        print(f"\n  • {i['type']:12} | Latency: {i['latency']:10} | Throughput: {i['throughput']:10} | {i['cost']:10} | {i['best']}")

cloud_options()
```

---

## Measuring and Monitoring

### Key Metrics to Track

```python

def key_metrics():
    """
    What to measure
    """
    print("Key Metrics for Latency and Throughput")
    "=" * 60

    metrics = [
        {
            "name": "p50 latency",
            "what": "Median response time",
            "target": "< 200ms for real-time"
        },
        {
            "name": "p95 latency",
            "what": "95th percentile (worst experience)",
            "target": "< 500ms"
        },
        {
            "name": "p99 latency",
            "what": "99th percentile (outliers)",
            "target": "< 1s"
        },
        {
            "name": "Time to First Token (TTFT)",
            "what": "How long until first token arrives",
            "target": "< 100ms for streaming"
        },
        {
            "name": "Inter-token latency",
            "what": "Time between subsequent tokens",
            "target": "< 10ms/token"
        },
        {
            "name": "Requests per second (RPS)",
            "what": "Current throughput",
            "target": "Varies by capacity"
        },
        {
            "name": "Queue depth",
            "what": "Waiting requests",
            "target": "Near zero"
        }
    ]

    for m in metrics:
        print(f"\n  • {m['name']}: {m['what']}")
        print(f"    Target: {m['target']}")

key_metrics()
```

---

## Optimization Techniques by Goal

```python

def optimization_by_goal():
    """
    Different techniques for different priorities
    """
    print("Optimization Techniques by Goal")
    "=" * 60

    print("""
    TO REDUCE LATENCY:
    • Smaller batch sizes (often 1)
    • Faster hardware (A100 > T4)
    • Speculative decoding
    • Streaming responses
    • KV caching
    • Model distillation

    TO INCREASE THROUGHPUT:
    • Larger batch sizes
    • Continuous batching
    • PagedAttention (vLLM)
    • Quantization (INT8/INT4)
    • Model pruning
    • Horizontal scaling

    TO BALANCE BOTH:
    • Dynamic batching (adjust based on load)
    • Auto-scaling
    • Request prioritization
    • Multiple model sizes
    """)

optimization_by_goal()
```

---

## The SLO/SLA Decision

### Setting Service Level Objectives

```python

def slo_decision():
    """
    How to choose your targets
    """
    print("Setting SLOs: What Do Your Users Need?")
    "=" * 60

    print("""
    Ask these questions:

    1. IS THIS USER-FACING?
       • Yes → Optimize for latency (p95 < 500ms)
       • No → Optimize for throughput/cost

    2. IS THIS SYNCHRONOUS?
       • Yes → User waits, so low latency critical
       • No → Can be async, throughput matters more

    3. WHAT'S THE BUSINESS IMPACT?
       • Each 100ms delay reduces conversion by 1% → prioritize latency
       • Each $0.01 cost saves $10k/month → prioritize throughput

    4. WHAT'S THE WORKLOAD PATTERN?
       • Steady load → Optimize for throughput
       • Spiky load → Need low latency during peaks

    Example SLO:
    • p95 latency < 500ms for 99% of requests over 1 minute
    • Throughput >= 100 RPS sustained
    """)

slo_decision()
```

---

## Latency vs Throughput Cheat Sheet

| Priority        | Batch Size     | Hardware    | Techniques                        | Example     |
| --------------- | -------------- | ----------- | --------------------------------- | ----------- |
| Low latency     | Small (1-4)    | Fastest GPU | Speculative decoding, streaming   | Chat, voice |
| Balanced        | Medium (8-32)  | Mid-range   | Dynamic batching                  | General API |
| High throughput | Large (32-256) | Many GPUs   | Continuous batching, quantization | Batch jobs  |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Getting this tradeoff wrong can kill your product:         ║
║                                                              ║
║  📱 USER EXPERIENCE: 500ms vs 2s latency can be the        ║
║     difference between users staying or leaving.             ║
║                                                              ║
║  💰 COST: Optimizing for low latency can cost 10x          ║
║     more per request. At scale, that's millions.             ║
║                                                              ║
║  📈 SCALABILITY: Wrong throughput assumptions mean          ║
║     your service crashes under load or sits idle.            ║
║                                                              ║
║  🏢 BUSINESS MODEL: Real-time features enable new           ║
║     products; batch processing enables low prices.           ║
║                                                              ║
║  🔧 ARCHITECTURE: The choice affects everything—           ║
║     hardware, software, caching, load balancing.             ║
║                                                              ║
║  There's no "right" answer—only the right answer            ║
║  for YOUR users, YOUR budget, and YOUR use case.            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Latency is how fast individual requests complete; throughput is how many requests you handle per second—they're fundamentally in tension

• Small batches = low latency, low throughput; large batches = high latency, high throughput—batch size is the primary lever

• Choose your optimization based on use case—real-time user-facing apps need low latency; batch jobs need high throughput

---

## Mental Hook

> "Latency vs throughput is like choosing between a Ferrari and a bus. The Ferrari gets you there fast (low latency) but carries one person. The bus carries 50 people (high throughput) but takes longer and makes frequent stops. You need both in your fleet, just for different trips."
