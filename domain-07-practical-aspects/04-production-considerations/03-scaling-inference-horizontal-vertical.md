# Scaling Inference (Horizontal/Vertical)

## The Restaurant Expansion Analogy

Imagine a popular restaurant that's always full. They have two ways to serve more customers: build a bigger kitchen with more stoves (vertical scaling) or open more locations across town (horizontal scaling). A bigger kitchen can cook more food but has limits—you can only fit so many stoves. More locations can serve unlimited customers but require duplicating everything: kitchens, staff, ingredients. That's scaling inference for LLMs: increasing capacity to handle more users.

In production LLM serving, scaling is inevitable. Your startup succeeds, users flood in, and your single GPU instance is melting. Do you buy a bigger GPU (vertical) or add more GPU instances (horizontal)? Each has tradeoffs in cost, complexity, and capability.

---

## What Are Scaling Strategies?

### The Two Dimensions

```python

def scaling_intro():
    """
    Vertical vs Horizontal scaling
    """
    print("Scaling Inference: Two Paths to More Capacity")
    print("=" * 60)

    print("""
    VERTICAL SCALING (Scale Up)
    ─────────────────────────────────────────────────
    Make your existing machine more powerful:
    • T4 → A100 → H100
    • More GPU memory
    • Faster processors
    • More RAM

    HORIZONTAL SCALING (Scale Out)
    ─────────────────────────────────────────────────
    Add more machines:
    • 1 GPU → 10 GPUs → 100 GPUs
    • Load balancer distributes traffic
    • Each instance handles subset of users
    """)

scaling_intro()
```

### Visual Comparison

```text

VERTICAL SCALING:                 HORIZONTAL SCALING:
    ┌────────┐                       ┌────────┐
    │   T4   │                       │  LB    │  Load Balancer
    └────────┘                       └────────┘
        ↓                                  ↓
    ┌────────┐            ┌────────┐ ┌────────┐ ┌────────┐
    │  A100  │            │ GPU 1  │ │ GPU 2  │ │ GPU 3  │
    └────────┘            └────────┘ └────────┘ └────────┘
        ↓
    ┌────────┐
    │  H100  │
    └────────┘

Bigger machine =              More machines =
limited ceiling               theoretically unlimited
```

---

## Vertical Scaling (Scale Up)

### Going Bigger

```python

def vertical_scaling():
    """
    Making one machine more powerful
    """
    print("Vertical Scaling: The Bigger Engine")
    "=" * 60

    gpus = [
        {
            "model": "T4",
            "memory": "16GB",
            "throughput": "1x",
            "cost": "$2/hour",
            "can_run": "7B models (quantized)"
        },
        {
            "model": "L4",
            "memory": "24GB",
            "throughput": "2x",
            "cost": "$3/hour",
            "can_run": "13B models"
        },
        {
            "model": "A10G",
            "memory": "24GB",
            "throughput": "3x",
            "cost": "$4/hour",
            "can_run": "13B models"
        },
        {
            "model": "A100",
            "memory": "80GB",
            "throughput": "5x",
            "cost": "$10/hour",
            "can_run": "70B models (quantized)"
        },
        {
            "model": "H100",
            "memory": "80GB",
            "throughput": "10x",
            "cost": "$30/hour",
            "can_run": "70B models (full precision)"
        }
    ]

    print("GPU progression:")
    for g in gpus:
        bar = "█" * int(g['throughput'].replace('x', '')) * 2
        print(f"  • {g['model']:6} | Mem: {g['memory']:6} | {g['throughput']:4} {bar:10} | {g['cost']:10} | {g['can_run']}")

vertical_scaling()
```

### When to Scale Vertically

```python

def when_vertical():
    """
    Good times for vertical scaling
    """
    print("When Vertical Scaling Makes Sense")
    "=" * 60

    pros = [
        "Simpler architecture (one machine)",
        "No distributed system complexity",
        "Faster communication (within one machine)",
        "Easier to manage (no load balancer)",
        "Good for models that don't fit on smaller GPUs"
    ]

    cons = [
        "Hardware limits (can only go so big)",
        "Single point of failure",
        "Downtime required to upgrade",
        "Diminishing returns (H100 is expensive!)",
        "Can't scale beyond biggest available GPU"
    ]

    print("✅ PROS:")
    for p in pros:
        print(f"  • {p}")

    print("\n❌ CONS:")
    for c in cons:
        print(f"  • {c}")

when_vertical()
```

---

## Horizontal Scaling (Scale Out)

### Adding More Machines

```python

def horizontal_scaling():
    """
    Adding more instances
    """
    print("Horizontal Scaling: More Cooks in More Kitchens")
    "=" * 60

    print("""
    ┌─────────────────────────────────────────────────────┐
    │                    LOAD BALANCER                     │
    │  (distributes requests across instances)             │
    └─────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Instance 1 │ │  Instance 2 │ │  Instance 3 │
    │   GPU:T4    │ │   GPU:T4    │ │   GPU:T4    │
    │   Model:A   │ │   Model:A   │ │   Model:A   │
    └─────────────┘ └─────────────┘ └─────────────┘
          ↓               ↓               ↓
    ┌─────────────────────────────────────────────────────┐
    │              Shared Database / Cache                 │
    └─────────────────────────────────────────────────────┘
    """)

horizontal_scaling()
```

### When to Scale Horizontally

```python

def when_horizontal():
    """
    Good times for horizontal scaling
    """
    print("When Horizontal Scaling Makes Sense")
    "=" * 60

    pros = [
        "Theoretically unlimited scale",
        "No downtime to add capacity",
        "Fault tolerance (one dies, others keep going)",
        "Can use cheaper, smaller GPUs",
        "Geographic distribution (serve users closer)"
    ]

    cons = [
        "Complex architecture (load balancer, coordination)",
        "Inter-instance communication overhead",
        "Data consistency challenges",
        "More moving parts to manage",
        "Cost of duplicated infrastructure"
    ]

    print("✅ PROS:")
    for p in pros:
        print(f"  • {p}")

    print("\n❌ CONS:")
    for c in cons:
        print(f"  • {c}")

when_horizontal()
```

---

## Scaling Strategies Compared

| Aspect               | Vertical Scaling               | Horizontal Scaling         |
| -------------------- | ------------------------------ | -------------------------- |
| Maximum scale        | Limited by hardware            | Theoretically unlimited    |
| Complexity           | Low                            | High                       |
| Cost at small scale  | Lower                          | Higher (multiple machines) |
| Cost at large scale  | Very high (H100)               | Economies of scale         |
| Fault tolerance      | None                           | High                       |
| Downtime for scaling | Yes                            | No                         |
| Inter-instance comms | N/A                            | Needed for consistency     |
| Best for             | Single large model, simplicity | High traffic, reliability  |

---

## Hybrid Approaches

### Sharding: Splitting the Model

```python

def sharding():
    """
    Model parallelism: splitting a model across GPUs
    """
    print("Hybrid Approach: Model Sharding")
    "=" * 60

    print("""
    When one GPU can't fit the whole model:

    GPU 1: Layers 1-20      GPU 2: Layers 21-40     GPU 3: Layers 41-60
    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │   Forward   │ ──────→ │   Forward   │ ──────→ │   Forward   │
    │   Pass 1    │         │   Pass 2    │         │   Pass 3    │
    └─────────────┘         └─────────────┘         └─────────────┘

    This is VERTICAL scaling (more GPUs per request)
    combined with HORIZONTAL concepts.

    Used for: models too big for one GPU (Llama 405B)
    """)

sharding()
```

### Tensor Parallelism

```python

def tensor_parallel():
    """
    Splitting individual layers across GPUs
    """
    print("Tensor Parallelism: Splitting Layers")
    "=" * 60

    print("""
    Even a single layer can be split:

    Attention Head: [Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8]
                        ↓
    GPU 1: [Q1,Q2]   GPU 2: [Q3,Q4]   GPU 3: [Q5,Q6]   GPU 4: [Q7,Q8]
        ↓               ↓               ↓               ↓
    ┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
    │Compute │      │Compute │      │Compute │      │Compute │
    └────────┘      └────────┘      └────────┘      └────────┘
        ↓               ↓               ↓               ↓
    ┌─────────────────────────────────────────────────────┐
    │                   All-reduce                         │
    └─────────────────────────────────────────────────────┘

    Used in: training, large-scale inference
    """)

tensor_parallel()
```

---

## Auto-scaling

### Scaling Based on Demand

```python

def auto_scaling():
    """
    Dynamic scaling based on load
    """
    print("Auto-scaling: Responding to Traffic")
    "=" * 60

    print("""
    ┌─────────────────────────────────────────────────────┐
    │                    TRAFFIC                            │
    │  ████████░░░░░░████████████░░░░░░░░░░████████████    │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │                 AUTO-SCALER                           │
    │  Monitors: CPU, queue depth, latency                 │
    │  Rules: If metric > threshold for 5 min, scale up    │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │              INSTANCE COUNT                           │
    │  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  │
    │  │ 1 │  │ 2 │  │ 3 │  │ 4 │  │ 5 │  │ 6 │  │ 7 │  │
    │  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘  │
    │  (scales from 2 to 7 based on load)                  │
    └─────────────────────────────────────────────────────┘
    """)

auto_scaling()
```

### Auto-scaling Configuration

```python

def auto_scaling_config():
    """
    Example auto-scaling rules
    """
    print("Auto-scaling Configuration Example")
    "=" * 60

    print("""
    # AWS Auto-scaling config example
    {
        "MinSize": 2,
        "MaxSize": 20,
        "ScaleOutPolicy": {
            "Metric": "GPUUtilization",
            "Threshold": 80,
            "Duration": 300,  # 5 minutes
            "Adjustment": "+2 instances"
        },
        "ScaleInPolicy": {
            "Metric": "GPUUtilization",
            "Threshold": 30,
            "Duration": 600,  # 10 minutes
            "Adjustment": "-1 instance"
        },
        "CoolDown": 300  # Wait 5 min between scaling actions
    }

    # Kubernetes HPA (Horizontal Pod Autoscaler)
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: llm-inference
    spec:
      minReplicas: 2
      maxReplicas: 20
      metrics:
      - type: Resource
        resource:
          name: gpu
          target:
            type: Utilization
            averageUtilization: 70
    """)

auto_scaling_config()
```

---

## Scaling Decision Framework

```python

def scaling_framework():
    """
    How to choose your scaling strategy
    """
    print("Scaling Decision Framework")
    "=" * 60

    print("""
    START HERE
        ↓
    Does model fit on one GPU?
    ├─► NO → Use model parallelism (sharding/tensor parallel)
    └─► YES → What's your primary constraint?
              ├─► Model too slow? → Vertical scaling (faster GPU)
              ├─► Too many users? → Horizontal scaling (more instances)
              └─► Both? → Hybrid (faster GPUs + more of them)
        ↓
    Consider auto-scaling based on load patterns
        ↓
    Monitor and adjust
    """)

scaling_framework()
```

---

## Cost Analysis

```python

def cost_analysis():
    """
    Comparing costs of different strategies
    """
    print("Cost Analysis: Vertical vs Horizontal")
    "=" * 60

    # Scenario: Need 1000 requests/second
    req_per_sec = 1000
    throughput_per_t4 = 50  # requests/second per T4
    throughput_per_h100 = 500  # requests/second per H100

    t4_needed = req_per_sec / throughput_per_t4
    h100_needed = req_per_sec / throughput_per_h100

    t4_cost = t4_needed * 2  # $2/hour for T4 (spot)
    h100_cost = h100_needed * 30  # $30/hour for H100

    print(f"Target throughput: {req_per_sec} RPS")
    print(f"\nHORIZONTAL (T4 instances):")
    print(f"  Need {t4_needed:.0f} T4 instances")
    print(f"  Hourly cost: ${t4_cost:.0f}")
    print(f"  Annual cost: ${t4_cost * 24 * 365:,.0f}")

    print(f"\nVERTICAL (H100 instances):")
    print(f"  Need {h100_needed:.0f} H100 instance")
    print(f"  Hourly cost: ${h100_cost:.0f}")
    print(f"  Annual cost: ${h100_cost * 24 * 365:,.0f}")

    print("\nBUT consider:")
    print("• H100 may have better latency")
    print("• T4 gives better fault tolerance")
    print("• T4 scaling is more flexible")

cost_analysis()
```

---

## Scaling Cheat Sheet

| Scenario                         | Recommended Strategy              | Why                        |
| -------------------------------- | --------------------------------- | -------------------------- |
| Model too big for one GPU        | Vertical (bigger GPU) or sharding | Can't fit otherwise        |
| High traffic, steady load        | Horizontal (many instances)       | Scale out predictably      |
| Unpredictable traffic spikes     | Auto-scaling                      | Save money during low load |
| Low latency critical             | Vertical (faster GPU)             | Faster single-thread perf  |
| Cost sensitive                   | Horizontal with smaller GPUs      | Cheaper per unit compute   |
| Global users                     | Horizontal + geo-distribution     | Serve users closer         |
| Single point of failure concerns | Horizontal                        | Redundancy                 |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Getting scaling wrong can break your business:             ║
║                                                              ║
║  📈 GROWTH: Your service crashes when you succeed.          ║
║     Users leave, never come back.                           ║
║                                                              ║
║  💰 COST: Over-provisioning burns cash. Under-provisioning  ║
║     loses users. The right balance is critical.             ║
║                                                              ║
║  🏢 COMPETITION: If your competitor scales better,          ║
║     they can offer lower prices or faster responses.        ║
║                                                              ║
║  🌍 GLOBAL REACH: Horizontal scaling lets you serve         ║
║     users worldwide with low latency.                       ║
║                                                              ║
║  🔧 FUTURE-PROOFING: The right architecture today           ║
║     prevents painful rewrites tomorrow.                     ║
║                                                              ║
║  Scaling isn't just about handling more users—it's          ║
║  about handling success without burning out.                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Vertical scaling (bigger GPUs) is simpler but hits hardware limits—good for single large models and when low latency is critical

• Horizontal scaling (more instances) is complex but theoretically unlimited—provides fault tolerance and geographic distribution

• Modern systems often use hybrid approaches—model parallelism for giant models, auto-scaling for traffic spikes, and a mix of both for optimal cost/performance

---

## Mental Hook

> "Vertical scaling is like upgrading from a food truck to a restaurant—more kitchen space, but still one location. Horizontal scaling is like opening a franchise—many locations, each serving customers, with a central kitchen for prep. You need both strategies at different stages."
