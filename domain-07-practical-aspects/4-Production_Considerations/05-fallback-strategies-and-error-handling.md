# Fallback Strategies and Error Handling

## The Airline's Backup Plan Analogy

Imagine you're flying from New York to London. Your primary flight gets canceled due to mechanical issues. A good airline immediately rebooks you on a partner airline (fallback), puts you on a flight through Paris (alternate route), and if all else fails, gives you a hotel voucher and meal credits (graceful degradation). That's fallback strategies and error handling for LLM services: planning for when things go wrong so users still get a good experience.

In production LLM systems, things WILL go wrong. APIs rate-limit you, models timeout, GPUs fail, costs spike. Without fallbacks, your application crashes and users see 500 errors. With good error handling, they might get a slightly slower response, a simpler model, or a helpful message—but they don't abandon your product.

---

## Types of Failures

### What Can Go Wrong

```python

def failure_types():
    """
    Common failure modes in LLM applications
    """
    print("What Can Go Wrong: Failure Modes")
    print("=" * 60)

    failures = [
        {
            "type": "API failures",
            "examples": [
                "Rate limit exceeded (429)",
                "Service unavailable (503)",
                "Authentication error (401)",
                "Timeout (request too long)"
            ]
        },
        {
            "type": "Model failures",
            "examples": [
                "Hallucination (factually wrong)",
                "Toxic content generation",
                "Repetition loops",
                "Context window exceeded"
            ]
        },
        {
            "type": "Infrastructure failures",
            "examples": [
                "GPU out of memory",
                "Network partition",
                "Database unavailable",
                "Load balancer down"
            ]
        },
        {
            "type": "Cost failures",
            "examples": [
                "Budget exceeded",
                "Unexpected usage spike",
                "Expensive model accidentally called"
            ]
        }
    ]

    for f in failures:
        print(f"\n  • {f['type']}:")
        for ex in f['examples']:
            print(f"    - {ex}")

failure_types()
```

---

## Fallback Strategies

### Strategy 1: Model Fallbacks

```python

def model_fallbacks():
    """
    Cascading through models of decreasing capability
    """
    print("Fallback Strategy 1: Model Cascading")
    "=" * 60

    print("""
    ┌─────────────────────────────────────────────────────┐
    │                    USER REQUEST                       │
    └─────────────────────┬───────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────┐
    │  TRY: GPT-4 (best quality, expensive)                │
    │  ↓                                                   │
    │  IF rate limited or timeout →                        │
    └─────────────────────┬───────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────┐
    │  TRY: GPT-3.5 (good quality, cheaper)                │
    │  ↓                                                   │
    │  IF also failing →                                   │
    └─────────────────────┬───────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────┐
    │  TRY: Llama (open source, self-hosted)               │
    │  ↓                                                   │
    │  IF all APIs failing →                               │
    └─────────────────────┬───────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────┐
    │  FALLBACK: Cached response or graceful message       │
    │  "Our AI is busy. Please try again in a moment."     │
    └─────────────────────────────────────────────────────┘
    """)

model_fallbacks()
```

### Code Example: Model Cascading

```python

import asyncio
import time

def model_cascading():
    """
    Implementing model fallbacks
    """
    print("Model Cascading Implementation")
    "=" * 60

    print("""
    async def generate_with_fallback(prompt, max_retries=3):
        '''Try multiple models in sequence'''

        # Define models in priority order
        models = [
            {"name": "gpt-4", "client": openai.ChatCompletion, "timeout": 5},
            {"name": "gpt-3.5-turbo", "client": openai.ChatCompletion, "timeout": 3},
            {"name": "claude-instant", "client": anthropic.Completion, "timeout": 3},
            {"name": "self-hosted-llama", "client": local_client, "timeout": 10}
        ]

        last_error = None

        for model in models:
            try:
                # Try this model with timeout
                response = await asyncio.wait_for(
                    model['client'].create(prompt=prompt),
                    timeout=model['timeout']
                )

                # Log success for monitoring
                logger.info(f"Success with {model['name']}")
                return response

            except asyncio.TimeoutError:
                logger.warning(f"{model['name']} timeout")
                last_error = "timeout"
                continue

            except RateLimitError:
                logger.warning(f"{model['name']} rate limited")
                await asyncio.sleep(1)  # Wait before next try
                continue

            except Exception as e:
                logger.error(f"{model['name']} failed: {e}")
                last_error = str(e)
                continue

        # All models failed
        return {
            "error": True,
            "message": "Our AI service is temporarily unavailable",
            "fallback": "Please try again in a few minutes"
        }
    """)

model_cascading()
```

---

## Strategy 2: Degraded Mode

### When You Can't Serve the Full Response

```python

def degraded_mode():
    """
    Serving simpler responses when full service is unavailable
    """
    print("Fallback Strategy 2: Degraded Mode")
    "=" * 60

    scenarios = [
        {
            "normal": "Full chat with memory and context",
            "degraded": "Single-turn responses, no memory",
            "when": "Database for conversation history is down"
        },
        {
            "normal": "RAG with vector search",
            "degraded": "Keyword search only",
            "when": "Vector database unavailable"
        },
        {
            "normal": "GPT-4 with 32K context",
            "degraded": "GPT-3.5 with 4K context",
            "when": "High load or budget exceeded"
        },
        {
            "normal": "Real-time streaming",
            "degraded": "Batch response (non-streaming)",
            "when": "Streaming infrastructure issues"
        }
    ]

    print("Degraded mode examples:")
    for s in scenarios:
        print(f"\n  • Normal: {s['normal']}")
        print(f"    Degraded: {s['degraded']}")
        print(f"    When: {s['when']}")

degraded_mode()
```

---

## Strategy 3: Caching as Fallback

### When You Can't Generate, Use Stored Responses

```python

def caching_fallback():
    """
    Using cached responses when generation fails
    """
    print("Fallback Strategy 3: Cache as Safety Net")
    "=" * 60

    print("""
    Multi-level cache strategy:

    ┌─────────────────────────────────────────────────────┐
    │                   USER REQUEST                        │
    └─────────────────────┬───────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────┐
    │  LEVEL 1: Redis Cache (hot)                          │
    │  • Frequently asked questions                        │
    │  • 1-5 minute TTL                                    │
    │  • Millisecond latency                               │
    └─────────────────────┬───────────────────────────────┘
                          ↓ (miss)
    ┌─────────────────────────────────────────────────────┐
    │  LEVEL 2: PostgreSQL Cache (warm)                    │
    │  • Historical responses                              │
    │  • 30-day retention                                  │
    │  • 10-50ms latency                                   │
    └─────────────────────┬───────────────────────────────┘
                          ↓ (miss)
    ┌─────────────────────────────────────────────────────┐
    │  LEVEL 3: Generate new response                       │
    │  • Call LLM API                                      │
    │  • 500ms-5s latency                                  │
    │  • Store result in Level 1 and 2                     │
    └─────────────────────────────────────────────────────┘

    If Level 3 fails, Level 2 can still serve slightly stale responses!
    """)

caching_fallback()
```

---

## Strategy 4: Graceful Degradation

### Failing Without Breaking

```python

def graceful_degradation():
    """
    How to fail gracefully
    """
    print("Fallback Strategy 4: Graceful Degradation")
    "=" * 60

    print("""
    BAD ERROR HANDLING:
    ┌─────────────────────────────────────────────────────┐
    │  500 Internal Server Error                           │
    │  <html><body><h1>Something broke</h1></body></html> │
    └─────────────────────────────────────────────────────┘

    GOOD ERROR HANDLING:
    ┌─────────────────────────────────────────────────────┐
    │  {                                                     │
    │    "error": true,                                     │
    │    "message": "Our AI service is temporarily busy",   │
    │    "suggestion": "Please try again in 30 seconds",    │
    │    "fallback_response": "I'm having trouble right     │
    │     now. Would you like me to summarize our help      │
    │     center articles instead?",                        │
    │    "request_id": "abc-123-def"                        │
    │  }                                                     │
    └─────────────────────────────────────────────────────┘

    Principles:
    • Be honest (don't pretend nothing happened)
    • Be helpful (offer alternatives)
    • Be trackable (include request ID)
    • Be temporary (indicate when to retry)
    """)

graceful_degradation()
```

---

## Strategy 5: Circuit Breaker

### Preventing Cascading Failures

```python

def circuit_breaker():
    """
    The circuit breaker pattern
    """
    print("Fallback Strategy 5: Circuit Breaker")
    "=" * 60

    print("""
    ┌─────────────────────────────────────────────────────┐
    │                    CIRCUIT BREAKER                    │
    ├─────────────────────────────────────────────────────┤
    │                                                    │
    │  CLOSED (normal operation)                           │
    │  • Requests pass through                             │
    │  • Track failure rate                                │
    │  • If failures > threshold → OPEN                    │
    │                                                    │
    │         ↓                                            │
    │                                                    │
    │  OPEN (failing fast)                                 │
    │  • All requests fail immediately                     │
    │  • No attempts to call failing service               │
    │  • After timeout → HALF-OPEN                         │
    │                                                    │
    │         ↓                                            │
    │                                                    │
    │  HALF-OPEN (testing the waters)                      │
    │  • Allow limited requests through                    │
    │  • If they succeed → CLOSED                          │
    │  • If they fail → OPEN again                         │
    │                                                    │
    └─────────────────────────────────────────────────────┘
    """)

circuit_breaker()
```

### Circuit Breaker Implementation

```python

def circuit_breaker_code():
    """
    Simple circuit breaker implementation
    """
    print("Circuit Breaker Code Example")
    "=" * 60

    print("""
    class CircuitBreaker:
        def __init__(self, failure_threshold=5, timeout=60):
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.failure_count = 0
            self.last_failure_time = None
            self.state = 'CLOSED'

        def call(self, func, *args, **kwargs):
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)

                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0

                return result

            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'

                raise e

    # Usage
    breaker = CircuitBreaker(failure_threshold=3, timeout=30)

    def call_llm_with_protection(prompt):
        try:
            return breaker.call(llm_api, prompt)
        except Exception:
            # Fallback to cache or degraded mode
            return get_cached_response(prompt)
    """)

circuit_breaker_code()

```

---

## Strategy 6: Retry Logic

### Smart Retrying

```python

def retry_logic():
    """
    Intelligent retry strategies
    """
    print("Fallback Strategy 6: Smart Retries")
    "=" * 60

    print("""
    RETRY STRATEGIES:

    1. IMMEDIATE RETRY
       • For transient errors (network glitch)
       • 1 retry, no delay

    2. EXPONENTIAL BACKOFF
       • Wait 1s, then 2s, then 4s, then 8s
       • Prevents overwhelming recovering service
       • Add jitter to avoid thundering herd

    3. RETRY WITH DIFFERENT MODEL
       • Try GPT-4, then GPT-3.5, then Claude
       • Different providers have different failure modes

    4. RETRY WITH DIFFERENT PARAMETERS
       • Shorter prompts on timeout
       • Lower temperature on repetition
       • Fewer max tokens

    EXAMPLE:
    async def call_with_retry(prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await call_llm(prompt)
            except RateLimitError:
                wait = (2 ** attempt) + random.random()
                await asyncio.sleep(wait)
            except TimeoutError:
                prompt = truncate_prompt(prompt)  # Try shorter version

        return fallback_response()
    """)

retry_logic()
```

---

## Error Handling Matrix

| Error Type              | Retry?                   | Fallback                | User Message                         |
| ----------------------- | ------------------------ | ----------------------- | ------------------------------------ |
| Rate limit (429)        | Yes, with backoff        | Cheaper model           | "High traffic, please wait"          |
| Timeout                 | Yes, with shorter prompt | Cache                   | "Taking longer than expected"        |
| Model unavailable (503) | Yes, different model     | Degraded mode           | "AI service temporarily unavailable" |
| Budget exceeded         | No                       | Cheaper model           | "Upgrading our service"              |
| Content moderation      | No                       | Refusal + explanation   | "I can't answer that"                |
| Hallucination detected  | No                       | Fact-check + correction | "Let me verify that..."              |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  In production, failures are inevitable. Your response      ║
║  to failures determines user trust:                         ║
║                                                              ║
║  🚨 WITHOUT FALLBACKS:                                       ║
║     • User sees 500 error                                    ║
║     • User gets frustrated                                   ║
║     • User leaves, maybe never comes back                    ║
║     • You lose revenue, reputation, trust                    ║
║                                                              ║
║  ✅ WITH FALLBACKS:                                           ║
║     • User sees "Please try again" with smile                ║
║     • User gets cached response (slightly stale but helpful) ║
║     • User gets degraded but working service                 ║
║     • User understands and appreciates transparency          ║
║                                                              ║
║  📊 STATISTICS:                                               ║
║     • 40% of users abandon a site after a single error       ║
║     • 60% expect errors to be handled gracefully             ║
║     • Good error handling can recover 30% of lost users      ║
║                                                              ║
║  Your error handling is the difference between a             ║
║  frustrating outage and a "sorry, we're working on it"       ║
║  moment that builds trust.                                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Fallback strategies are planned responses to failures—model cascades, degraded modes, cached responses, and circuit breakers

• Error handling should be graceful and helpful—clear messages, alternative suggestions, and transparency about what went wrong

• Retry logic needs to be smart—exponential backoff, jitter, and different strategies for different error types prevent cascading failures

---

## Mental Hook

> "Fallback strategies are like having a spare tire in your car. You hope you never need it, but when you get a flat on the highway at midnight, you're incredibly grateful it's there. Users feel the same way when your AI service has problems."
