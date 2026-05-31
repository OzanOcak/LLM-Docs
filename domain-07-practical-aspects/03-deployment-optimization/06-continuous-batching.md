# Continuous Batching

## The Restaurant Kitchen Analogy

Imagine a restaurant kitchen that only cooks one table's entire meal at a time—appetizer, main course, dessert—before starting the next table's order. Table 2 would wait forever while Table 1 finishes their three-course meal. A smarter kitchen interleaves tasks: while Table 1's main course is cooking, they start Table 2's appetizer. That's continuous batching: instead of waiting for entire sequences to finish, we process tokens from multiple requests simultaneously, dramatically improving throughput.

In LLM serving, continuous batching is the secret sauce that lets companies serve millions of users with reasonable hardware. Without it, GPUs would sit idle most of the time, waiting for slow requests to finish. With continuous batching, we keep the GPU busy 100% of the time, increasing throughput by 10-20x.

---

## The Problem: Static Batching Wastes GPU

### How Traditional Batching Works

```python

def static_batching():
    """
    The inefficiency of static batching
    """
    print("Static Batching: Wait for Everyone")
    "=" * 60

    print("""
    Traditional approach:

    Request 1 (short): "Hello" → 1 token generation
    Request 2 (long):  "Write a 500-word essay about..." → 500 tokens

    Step 1: Batch both requests
    Step 2: Wait for Request 2 to finish ALL 500 tokens
    Step 3: Return both responses

    PROBLEM: Request 1 finishes in 10ms, but waits 5 seconds
             for Request 2. GPU is idle while waiting!

    GPU Utilization: ██░░░░░░░░░░░░░░░░░░  (10-20%)
    """)

static_batching()
```

### The Idle Time Problem

```python

def idle_time():
    """
    Visualizing GPU idle time
    """
    print("GPU Idle Time with Static Batching")
    "=" * 60

    import matplotlib.pyplot as plt
    import numpy as np

    # Simulate 4 requests of different lengths
    requests = [
        {"id": 1, "tokens": 10, "color": "fast"},
        {"id": 2, "tokens": 100, "color": "medium"},
        {"id": 3, "tokens": 500, "color": "slow"},
        {"id": 4, "tokens": 20, "color": "fast"}
    ]

    print("Timeline (static batching):")
    print("Time →")
    print("Req1: [██████████] (10 tokens) waiting...")
    print("Req2: [████████████████████████████████████████] (100 tokens)")
    print("Req3: [████████████████████████████████████████████████...] (500 tokens)")
    print("Req4: [████████████████████] (20 tokens) waiting...")
    print("\n" + "="*50)
    print("Fast requests wait for slow requests → GPU idle!")

    # Calculate utilization
    total_time = 500  # longest request
    active_time = 10 + 100 + 500 + 20
    utilization = active_time / (4 * total_time) * 100
    print(f"\nGPU Utilization: {utilization:.1f}%")

idle_time()
```

---

## The Solution: Continuous Batching

### How Continuous Batching Works

```python

def continuous_batching_intro():
    """
    The basic concept of continuous batching
    """
    print("Continuous Batching: Interleaving Tokens")
    "=" * 60

    print("""
    Instead of waiting for whole sequences:

    Time Step 1: Process token 1 from ALL active requests
    Time Step 2: Process token 2 from ALL active requests
    Time Step 3: Process token 3 from ALL active requests
    ...

    When a request finishes, immediately add a new one!

    ┌─────────────────────────────────────────────────┐
    │ Step: 1   2   3   4   5   6   7   8   9   10   │
    ├─────────────────────────────────────────────────┤
    │ Req1: █   █   █   █   █   █   █   █   █   █   │ (finished at 10)
    │ Req2: █   █   █   █   █   █   █   █   █   █   │ (still going)
    │ Req3: █   █   █   █   █   █   █   █   █   █   │ (still going)
    │ Req4:     █   █   █   █   █   █   █   █   █   │ (started at 2)
    │ Req5:         █   █   █   █   █   █   █   █   │ (started at 3)
    └─────────────────────────────────────────────────┘

    GPU always busy! Utilization near 100%
    """)

continuous_batching_intro()
```

### Step-by-Step Example

```python

def continuous_batching_example():
    """
    Detailed continuous batching example
    """
    print("Continuous Batching: Step by Step")
    "=" * 60

    # Initial requests
    requests = [
        {"id": 1, "remaining": 3, "status": "active"},
        {"id": 2, "remaining": 5, "status": "active"},
        {"id": 3, "remaining": 2, "status": "active"}
    ]

    print("Initial batch: 3 requests\n")

    step = 1
    while any(r["remaining"] > 0 for r in requests):
        print(f"Step {step}: Processing one token from each active request")

        # Process one token from each request
        for r in requests:
            if r["remaining"] > 0:
                r["remaining"] -= 1
                print(f"  Request {r['id']}: generated token, {r['remaining']} left")

        # Check for finished requests
        for r in requests:
            if r["remaining"] == 0 and r["status"] == "active":
                r["status"] = "finished"
                print(f"  🎉 Request {r['id']} COMPLETE!")

        # Add new request when one finishes
        if step == 3:  # Simulate new request at step 3
            requests.append({"id": 4, "remaining": 4, "status": "active"})
            print(f"  ➕ New Request 4 added!")

        step += 1
        print()


continuous_batching_example()

```

---

## Continuous Batching vs Static Batching

### Comparison Table

| Aspect                  | Static Batching         | Continuous Batching         |
| ----------------------- | ----------------------- | --------------------------- |
| Batch formation         | Wait for full sequences | Interleave tokens           |
| GPU utilization         | 10-30%                  | 90-100%                     |
| Latency (fast requests) | High (wait for slow)    | Low (processed immediately) |
| Throughput              | Baseline                | 10-20x higher               |
| Memory usage            | Lower                   | Higher (KV cache)           |
| Implementation          | Simple                  | Complex                     |

### Visual Comparison

```text

Static Batching:
Time →
Req1: [████]....................
Req2: [████████████████████]....
Req3: [██████]..................
      ↑ GPU idle most of the time!

Continuous Batching:
Time →
Req1: █ █ █ █ . . . . . . . . .
Req2: █ █ █ █ █ █ █ █ █ █ █ █ █
Req3: █ █ █ █ █ █ . . . . . . .
Req4: . . █ █ █ █ █ █ █ █ . . .
Req5: . . . . █ █ █ █ █ █ █ █ █
      ↑ GPU always busy!
```

---

## Implementation: Iteration-Level Scheduling

### How It Works Under the Hood

```python

def iteration_level_scheduling():
    """
    The key algorithm behind continuous batching
    """
    print("Iteration-Level Scheduling")
    "=" * 60

    print("""
    At each iteration (forward pass):

    1. Look at all ongoing requests
    2. For each request, compute the next token
    3. Update KV cache for each request
    4. Check if any requests finished
    5. Add new waiting requests to the batch

    Pseudo-code:

    def continuous_batching_step(active_requests, waiting_queue):
        # Prepare batch
        batch = []
        for req in active_requests:
            if req.has_more_tokens():
                batch.append(req)

        # Add new requests if space
        while len(batch) < max_batch_size and waiting_queue:
            batch.append(waiting_queue.pop())

        # Run one forward pass for all requests in batch
        outputs = model.forward(batch)

        # Update each request with its new token
        for req, output in zip(batch, outputs):
            req.append_token(output)
            if req.is_finished():
                return_response(req)

        # Next iteration...
    """)

iteration_level_scheduling()
```

### Simple Python Simulation

```python

import time
import random

def continuous_batching_sim():
    """
    Simulate continuous batching
    """
    print("Continuous Batching Simulation")
    "=" * 60

    class Request:
        def __init__(self, id, total_tokens):
            self.id = id
            self.total = total_tokens
            self.generated = 0
            self.active = True

        def step(self):
            if self.generated < self.total:
                self.generated += 1
                return self.generated == self.total
            return False

    # Initial requests
    active = [
        Request(1, 3),
        Request(2, 5),
        Request(3, 2)
    ]
    waiting = []

    # Simulate new requests arriving
    new_requests = [
        Request(4, 4),
        Request(5, 6),
        Request(6, 2)
    ]

    step = 1
    while active or waiting:
        print(f"\nStep {step}: {len(active)} active requests")

        # Process one token for each active request
        finished = []
        for req in active[:]:  # Copy list to avoid modification during iteration
            done = req.step()
            print(f"  Request {req.id}: generated token {req.generated}/{req.total}")
            if done:
                print(f"    ✅ Request {req.id} FINISHED!")
                active.remove(req)

        # Add new requests if space
        while len(active) < 4 and new_requests:
            new_req = new_requests.pop(0)
            active.append(new_req)
            print(f"  ➕ New Request {new_req.id} added ({new_req.total} tokens)")

        step += 1
        # time.sleep(0.5)  # Uncomment to see step-by-step

continuous_batching_sim()
```

---

## Performance Impact

### Throughput Improvement

```python

def throughput_improvement():
    """
    How much faster continuous batching is
    """
    print("Throughput Improvement")
    "=" * 60

    import math

    # Simulate mix of requests
    request_lengths = [10, 100, 500, 20, 30, 200, 50, 800, 15, 400]
    n_requests = len(request_lengths)

    # Static batching: must wait for longest
    static_time = max(request_lengths) * n_requests

    # Continuous batching: sum of all tokens
    continuous_time = sum(request_lengths)

    speedup = static_time / continuous_time

    print(f"Request lengths: {request_lengths}")
    print(f"Number of requests: {n_requests}")
    print(f"Static batching time: {static_time} token-steps")
    print(f"Continuous batching time: {continuous_time} token-steps")
    print(f"Speedup: {speedup:.1f}x")

    print("\nReal-world numbers:")
    print("  • Without continuous batching: 100 requests/second")
    print("  • With continuous batching:    1,500 requests/second")
    print("  • 15x throughput improvement!")

throughput_improvement()
```

### Latency for Fast Requests

```python

def latency_improvement():
    """
    How fast requests benefit
    """
    print("Latency Improvement for Fast Requests")
    "=" * 60

    fast_request_tokens = 10
    slow_request_tokens = 500

    # Static batching: fast waits for slow
    static_latency = slow_request_tokens * 2  # both in same batch

    # Continuous batching: fast processed interleaved
    continuous_latency = fast_request_tokens + 5  # minimal queue delay

    print(f"Fast request: {fast_request_tokens} tokens")
    print(f"Slow request: {slow_request_tokens} tokens")
    print(f"\nStatic batching latency: {static_latency} steps")
    print(f"Continuous batching latency: {continuous_latency} steps")
    print(f"Speedup for fast request: {static_latency/continuous_latency:.1f}x")

latency_improvement()
```

---

## Real-World Implementations

### vLLM

```python

def vllm():
    """
    vLLM's continuous batching
    """
    print("vLLM: Continuous Batching + PagedAttention")
    "=" * 60

    print("""
    vLLM combines continuous batching with PagedAttention:

    ┌─────────────────────────────────────────────────┐
    │              vLLM Scheduler                      │
    ├─────────────────────────────────────────────────┤
    │ 1. Maintains iteration-level batch               │
    │ 2. Dynamically adds/removes requests             │
    │ 3. Manages KV cache with PagedAttention          │
    │ 4. Prioritizes based on policy                   │
    └─────────────────────────────────────────────────┘

    Performance:
    • Up to 24x higher throughput than HuggingFace
    • 2-4x higher than traditional continuous batching
    • Handles 1000+ requests/second on single GPU

    Used by: OpenAI, Microsoft, many production systems
    """)

vllm()
```

### TensorRT-LLM

```python

def tensorrt():
    """
    NVIDIA's TensorRT-LLM
    """
    print("TensorRT-LLM: NVIDIA's Solution")
    "=" * 60

    print("""
    TensorRT-LLM features:

    • In-flight batching (continuous batching)
    • Paged KV cache
    • FP8 quantization
    • Multiple GPU optimizations

    Performance on H100:
    • 10,000+ tokens/second
    • 100x faster than naive implementation
    • Industry standard for production
    """)

tensorrt()
```

---

## Continuous Batching Cheat Sheet

| Aspect                    | Static Batching | Continuous Batching          |
| ------------------------- | --------------- | ---------------------------- |
| Batch at                  | Sequence level  | Token level                  |
| GPU utilization           | 10-30%          | 90-100%                      |
| Throughput                | 1x              | 10-20x                       |
| Fast request latency      | High            | Low                          |
| Memory efficiency         | Lower           | Higher (with PagedAttention) |
| Implementation complexity | Low             | High                         |
| Used in production        | No              | Yes (vLLM, TensorRT-LLM)     |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Continuous batching is what makes LLM serving economical:  ║
║                                                              ║
║  💰 COST: 10-20x higher throughput means 10-20x lower      ║
║     cost per request. At scale, this saves millions.        ║
║                                                              ║
║  🚀 SCALE: Serve millions of users with the same           ║
║     hardware that would otherwise handle thousands.         ║
║                                                              ║
║  ⚡ LATENCY: Fast requests don't wait for slow ones.       ║
║     Users get responses instantly.                          ║
║                                                              ║
║  🌍 DEMOCRATIZATION: Lower costs mean AI becomes           ║
║     accessible to more people and businesses.               ║
║                                                              ║
║  🔧 EFFICIENCY: 90%+ GPU utilization vs 20%—we're         ║
║     actually using our expensive hardware.                  ║
║                                                              ║
║  Every time you use a popular LLM API, continuous          ║
║  batching is why it's fast AND cheap.                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Continuous batching processes tokens from multiple requests interleaved—like a restaurant cooking multiple tables' dishes in parallel instead of finishing one table's entire meal before starting the next

• This increases GPU utilization from 10-30% to 90-100%—eliminating idle time and improving throughput by 10-20x

• Modern systems like vLLM combine continuous batching with PagedAttention—achieving even higher efficiency and making large-scale LLM serving economically viable

---

## Mental Hook

> "Continuous batching is like a short-order cook who works on multiple orders simultaneously—while one steak is grilling, they're plating another, starting a salad, and checking on soup. The kitchen never stops, and everyone gets their food faster."
