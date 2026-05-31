# Cost Optimization Strategies

## The Personal Finance Analogy

Imagine you're making $10,000 a month but spending $9,500 without tracking where it goes. You might be wasting money on subscriptions you don't use, eating out every day, and paying premium prices for everything. A financial advisor would help you track expenses, cut waste, and optimize spending. That's cost optimization for LLM services: understanding where your money goes and systematically reducing costs without sacrificing quality.

In production LLM deployment, costs can spiral out of control quickly. A single model call might cost fractions of a penny, but multiply by millions of requests and you're looking at six-figure monthly bills. Cost optimization isn't optional—it's essential for sustainable business.

---

## The Cost Equation

### Understanding Where Money Goes

```python

def cost_equation():
    """
    Breaking down LLM costs
    """
    print("The LLM Cost Equation")
    print("=" * 60)

    print("""
    Total Cost = (Prompt Tokens × Price_in) + (Completion Tokens × Price_out)

    But that's just the API cost. Real costs include:

    ┌─────────────────────────────────────────────────────┐
    │  API COSTS (if using third-party)                   │
    │  • $0.01-0.03 per 1K prompt tokens                  │
    │  • $0.03-0.06 per 1K completion tokens              │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │  INFRASTRUCTURE COSTS (if self-hosting)             │
    │  • GPU instances ($2-30/hour)                       │
    │  • Storage ($0.02-0.20/GB/month)                    │
    │  • Networking ($0.01-0.15/GB)                       │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │  HIDDEN COSTS                                        │
    │  • Retries on failure                               │
    │  • Long prompts with little value                    │
    │  • Excessive logging and monitoring                  │
    │  • Development time optimizing                       │
    └─────────────────────────────────────────────────────┘
    """)

cost_equation()
```

### Cost Calculation Example

```python

def cost_example():
    """
    Real-world cost calculation
    """
    print("Cost Calculation Example")
    "=" * 60

    # Scenario: 1M requests/month
    avg_prompt_tokens = 500
    avg_completion_tokens = 200
    price_prompt = 0.01 / 1000  # $0.01 per 1K
    price_completion = 0.03 / 1000  # $0.03 per 1K

    monthly_prompt_tokens = 1_000_000 * avg_prompt_tokens
    monthly_completion_tokens = 1_000_000 * avg_completion_tokens

    prompt_cost = monthly_prompt_tokens * price_prompt
    completion_cost = monthly_completion_tokens * price_completion
    total_cost = prompt_cost + completion_cost

    print(f"Monthly requests: 1,000,000")
    print(f"Avg prompt tokens: {avg_prompt_tokens}")
    print(f"Avg completion tokens: {avg_completion_tokens}")
    print(f"\nPrompt tokens: {monthly_prompt_tokens:,} → ${prompt_cost:,.2f}")
    print(f"Completion tokens: {monthly_completion_tokens:,} → ${completion_cost:,.2f}")
    print(f"\n💰 TOTAL MONTHLY COST: ${total_cost:,.2f}")
    print(f"💰 ANNUAL COST: ${total_cost * 12:,.2f}")

cost_example()
```

---

## Strategy 1: Model Selection

### Choosing the Right Model for the Job

```python

def model_selection():
    """
    Different models for different tasks
    """
    print("Strategy 1: Model Selection")
    "=" * 60

    models = [
        {
            "tier": "Premium",
            "model": "GPT-4, Claude-3 Opus",
            "cost": "$$$ (10-30x base)",
            "use": "Complex reasoning, creative writing, critical tasks"
        },
        {
            "tier": "Standard",
            "model": "GPT-3.5, Claude-3 Sonnet",
            "cost": "$$ (1x baseline)",
            "use": "General purpose, most tasks"
        },
        {
            "tier": "Budget",
            "model": "Llama, Mistral, Phi-3",
            "cost": "$ (0.1-0.5x)",
            "use": "High-volume, simpler tasks, self-hosted"
        },
        {
            "tier": "Tiny",
            "model": "Distilled models, Phi-3-mini",
            "cost": "¢ (0.01-0.05x)",
            "use": "Classification, extraction, edge devices"
        }
    ]

    print("Choose the smallest model that works:")
    for m in models:
        print(f"\n  • {m['tier']:10} | {m['model']:20} | {m['cost']:10} | {m['use']}")

model_selection()
```

### Model Routing

```python

def model_routing():
    """
    Intelligent routing to appropriate models
    """
    print("Model Routing: Smart Dispatch")
    "=" * 60

    print("""
    Instead of always using the most expensive model:

                    ┌─────────────────────┐
                    │   Classifier        │
                    │  "Is this complex?" │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ↓                  ↓                  ↓
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │  Simple query │  │ Medium query  │  │ Complex query │
    │  → Tiny model │  │ → GPT-3.5     │  │ → GPT-4       │
    └───────────────┘  └───────────────┘  └───────────────┘

    Savings: 50-80% on costs with minimal quality impact!
    """)

model_routing()
```

---

## Strategy 2: Prompt Optimization

### Shorter Prompts = Lower Costs

```python

def prompt_optimization():
    """
    Reducing token count in prompts
    """
    print("Strategy 2: Prompt Optimization")
    "=" * 60

    print("""
    Every token costs money. Optimize your prompts:

    ❌ BEFORE (250 tokens):
    "I would like you to act as a helpful assistant and provide
    a detailed explanation of the concept of gravity. Please
    include information about Isaac Newton, the apple story,
    and how gravity works on different planets. Also explain
    the difference between mass and weight, and give some
    examples of gravity in everyday life."

    ✅ AFTER (80 tokens):
    "Explain gravity: Newton's discovery, how it varies by planet,
    mass vs weight, and everyday examples. Concise please."

    SAVINGS: 68% fewer tokens → 68% lower cost!
    """)

prompt_optimization()
```

### System Prompts and Templates

```python

def system_prompts():
    """
    Reusing system prompts efficiently
    """
    print("System Prompts: Reuse, Don't Repeat")
    "=" * 60

    print("""
    Instead of sending the same system prompt with every request:

    ❌ NAIVE APPROACH:
    Request 1: [500 token system prompt] + [50 token user query]
    Request 2: [500 token system prompt] + [50 token user query]
    Request 3: [500 token system prompt] + [50 token user query]

    Total tokens: 3 × 550 = 1,650 tokens

    ✅ OPTIMIZED (with prompt caching):
    Request 1: [500 token system prompt] + [50 token user query]
    Request 2: [50 token user query] (cached system prompt)
    Request 3: [50 token user query] (cached system prompt)

    Total tokens: 500 + 3×50 = 650 tokens
    SAVINGS: 61%

    Some APIs (Anthropic) support prompt caching natively!
    """)

system_prompts()
```

---

## Strategy 3: Caching

### Response Caching

```python

def response_caching():
    """
    Cache identical or similar requests
    """
    print("Strategy 3: Response Caching")
    "=" * 60

    print("""
    Many user queries are repeated:

    EXACT CACHE (Redis/Memcached):
    ┌─────────────────────────────────────────────────────┐
    │ Query: "What is the capital of France?"              │
    │ Cache key: hash("capital of France")                 │
    │ → Return cached response "Paris"                     │
    └─────────────────────────────────────────────────────┘

    SEMANTIC CACHE (Vector DB):
    ┌─────────────────────────────────────────────────────┐
    │ Query 1: "Capital of France?"                        │
    │ Query 2: "What's the French capital?"                │
    │ Vector similarity: 0.95 → Same cached response!     │
    └─────────────────────────────────────────────────────┘

    Cache hit rates of 30-50% are common → huge savings!
    """)

response_caching()
```

### Cache Implementation

```python

def cache_code():
    """
    Simple cache implementation
    """
    print("Simple Cache Implementation")
    "=" * 60

    print("""
    import hashlib
    import redis
    import json

    cache = redis.Redis(host='localhost', port=6379)

    def get_cached_or_generate(prompt, model, ttl=3600):
        # Create cache key
        key = hashlib.md5(f"{prompt}:{model}".encode()).hexdigest()

        # Check cache
        cached = cache.get(key)
        if cached:
            print("🎯 Cache hit!")
            return json.loads(cached)

        # Generate response (costly!)
        print("❌ Cache miss, calling API...")
        response = call_llm_api(prompt, model)

        # Store in cache
        cache.setex(key, ttl, json.dumps(response))

        return response
    """)

cache_code()
```

---

## Strategy 4: Batch Processing

### Combining Requests

```python

def batch_processing():
    """
    Process multiple requests together
    """
    print("Strategy 4: Batch Processing")
    "=" * 60

    print("""
    Individual requests:           Batched request:
    ┌─────────────────────┐       ┌─────────────────────┐
    │ Req1: "Summarize X" │       │ Batch:              │
    │ Req2: "Summarize Y" │       │ • Summarize X       │
    │ Req3: "Summarize Z" │       │ • Summarize Y       │
    │ Req4: "Summarize W" │       │ • Summarize Z       │
    └─────────────────────┘       │ • Summarize W       │
        4 API calls               └─────────────────────┘
        4× overhead                    1 API call

    Savings:
    • 4x fewer API calls
    • Lower latency per request (amortized)
    • Better throughput

    Best for: Bulk processing, offline jobs, non-real-time
    """)

batch_processing()
```

---

## Strategy 5: Streaming vs Non-Streaming

```python

def streaming_optimization():
    """
    Streaming doesn't save tokens but improves UX
    """
    print("Strategy 5: Streaming Tradeoffs")
    "=" * 60

    print("""
    STREAMING:
    • User sees tokens as they're generated
    • Better UX, feels faster
    • Same token count, same cost
    • Can stop early if user satisfied

    NON-STREAMING:
    • Wait for complete response
    • Worse UX, feels slower
    • Same token count, same cost
    • Can't stop early

    COST OPTIMIZATION: Implement "stop generation" button
    • User reads first few tokens, gets answer, clicks stop
    • Saves 30-70% of completion tokens!
    • Best of both worlds: good UX + lower cost
    """)

streaming_optimization()
```

---

## Strategy 6: Fine-tuning and Distillation

### Custom Models for Your Use Case

```python

def fine_tuning():
    """
    Train smaller models for specific tasks
    """
    print("Strategy 6: Fine-tuning and Distillation")
    "=" * 60

    print("""
    Instead of using GPT-4 for everything:

    STEP 1: Collect examples of your task
    ┌─────────────────────────────────────────────┐
    │ Input: "Classify this email as spam or not" │
    │ Output: "spam"                              │
    │ ... 1,000 more examples                      │
    └─────────────────────────────────────────────┘

    STEP 2: Fine-tune a smaller model (Llama, Phi-3)
    • One-time cost: $50-500
    • Result: Custom model specialized for your task

    STEP 3: Use fine-tuned model for inference
    • Cost: 10-100x cheaper than GPT-4
    • Quality: Often better for your specific task

    Break-even: After 5,000-50,000 requests, you save money!
    """)

fine_tuning()
```

### Cost Comparison

```python

def cost_comparison():
    """
    Compare costs of different approaches
    """
    print("Cost Comparison: Generic vs Specialized")
    "=" * 60

    monthly_requests = 100_000
    cost_gpt4 = monthly_requests * 0.03  # $0.03 per request
    cost_finetuned = monthly_requests * 0.001  # $0.001 per request
    finetune_cost = 500  # One-time

    monthly_savings = cost_gpt4 - cost_finetuned
    breakeven_months = finetune_cost / monthly_savings

    print(f"Monthly requests: {monthly_requests:,}")
    print(f"GPT-4 monthly cost: ${cost_gpt4:,.2f}")
    print(f"Fine-tuned monthly cost: ${cost_finetuned:,.2f}")
    print(f"Monthly savings: ${monthly_savings:,.2f}")
    print(f"Fine-tuning cost: ${finetune_cost}")
    print(f"Breakeven: {breakeven_months:.1f} months")

    print("\nAfter 12 months:")
    print(f"  GPT-4: ${cost_gpt4 * 12:,.2f}")
    print(f"  Fine-tuned: ${cost_finetuned * 12 + finetune_cost:,.2f}")
    print(f"  Total savings: ${(cost_gpt4 - cost_finetuned) * 12 - finetune_cost:,.2f}")

cost_comparison()
```

---

## Strategy 7: Monitoring and Optimization Loop

### Continuous Improvement

```python

def optimization_loop():
    """
    The cost optimization cycle
    """
    print("Strategy 7: The Optimization Loop")
    "=" * 60

    print("""
    ┌─────────────────────────────────────────────────────┐
    │                     MONITOR                          │
    │  • Track token usage per endpoint                    │
    │  • Identify expensive patterns                       │
    │  • Set up cost alerts                                │
    └─────────────────────┬───────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────┐
    │                    ANALYZE                           │
    │  • Which queries use most tokens?                   │
    │  • Are there cheaper alternatives?                   │
    │  • Where is the waste?                               │
    └─────────────────────┬───────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────┐
    │                    OPTIMIZE                          │
    │  • Implement caching                                 │
    │  • Shorten prompts                                   │
    │  • Route to cheaper models                           │
    └─────────────────────┬───────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────┐
    │                    REPEAT                            │
    │  • Cost optimization is never done                   │
    │  • New patterns emerge, new optimizations possible   │
    └─────────────────────────────────────────────────────┘
    """)

optimization_loop()
```

---

## Cost Optimization Cheat Sheet

| Strategy            | Savings | Effort | Best For          |
| ------------------- | ------- | ------ | ----------------- |
| Model selection     | 10-100x | Low    | All tasks         |
| Prompt optimization | 20-50%  | Medium | Chat, completion  |
| Caching             | 30-70%  | Medium | Repeated queries  |
| Batch processing    | 50-80%  | Low    | Offline jobs      |
| Fine-tuning         | 10-100x | High   | Specialized tasks |
| Model routing       | 50-80%  | Medium | Mixed workloads   |
| Streaming + stop    | 30-70%  | Low    | Interactive apps  |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Cost optimization can be the difference between profit and  ║
║  bankruptcy, especially at scale:                           ║
║                                                              ║
║  💰 SCALE MATTERS: A $0.001 savings per request becomes     ║
║     $1,000/month at 1M requests, $100,000/month at 100M.    ║
║                                                              ║
║  🚀 GROWTH ENABLER: Lower costs mean you can afford         ║
║     more users, more features, more experimentation.        ║
║                                                              ║
║  🏢 COMPETITIVE ADVANTAGE: If your competitor spends        ║
║     10x more per request, you can undercut their prices     ║
║     or invest more in product development.                  ║
║                                                              ║
║  🌱 SUSTAINABILITY: 10x lower costs means 10x less         ║
║     energy consumption and environmental impact.            ║
║                                                              ║
║  🔧 PRODUCT ITERATION: Cheap inference means you can        ║
║     afford to experiment, fail, and try again.              ║
║                                                              ║
║  The difference between a startup that burns through        ║
║  funding and one that reaches profitability is often        ║
║  10-100x cost optimization.                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Model selection is the biggest lever—use the smallest, cheapest model that works for each task, and route intelligently between them

• Caching and prompt optimization reduce token count—cache identical or similar requests, shorten prompts, and reuse system prompts

• Fine-tuning creates custom models 10-100x cheaper than general APIs—with a one-time investment that pays off at scale

---

## Mental Hook

> "Cost optimization is like weight loss for your LLM service—every gram (token) you shed reduces the load (cost). A balanced diet (model selection), portion control (prompt length), and meal prep (caching) can transform an expensive habit into a sustainable lifestyle."
