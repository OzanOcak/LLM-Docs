# API Design

## The Restaurant Menu Analogy

Imagine a restaurant with a terrible menu—dishes are poorly described, prices are hidden, and you have to know secret codes to order. No one wants to eat there. A great restaurant has a clear menu, explains dishes, offers options, and makes ordering easy. That's API design for LLMs: how you present your model to users (developers). A well-designed API makes developers happy, your service successful, and your infrastructure manageable.

In LLM deployment, your API is the face of your product. It determines how easily developers can integrate your model, how efficiently you can serve requests, and how well you can control costs and usage. Good API design balances flexibility, simplicity, reliability, and economics.

---

## What Makes a Good LLM API?

### The Core Principles

```python

def api_principles():
    """
    Key principles of LLM API design
    """
    print("Principles of Good LLM API Design")
    print("=" * 60)

    principles = [
        {"principle": "Simplicity", "why": "Developers should understand it in minutes"},
        {"principle": "Flexibility", "why": "Support different use cases (chat, completion, etc.)"},
        {"principle": "Reliability", "why": "Consistent responses, clear error messages"},
        {"principle": "Efficiency", "why": "Minimize latency, maximize throughput"},
        {"principle": "Observability", "why": "See what's happening, debug issues"},
        {"principle": "Cost control", "why": "Users should understand what they'll pay"}
    ]

    for p in principles:
        print(f"  • {p['principle']:15} → {p['why']}")

api_principles()
```

---

## API Endpoints

### Common LLM API Patterns

```python

def api_endpoints():
    """
    Standard LLM API endpoints
    """
    print("Common LLM API Endpoints")
    "=" * 60

    endpoints = [
        {
            "name": "/completions",
            "description": "Simple text completion",
            "example": "POST /completions with {prompt: 'The capital of France is'}"
        },
        {
            "name": "/chat/completions",
            "description": "Multi-turn conversation",
            "example": "POST /chat/completions with [{role: 'user', content: 'Hi'}]"
        },
        {
            "name": "/embeddings",
            "description": "Get vector embeddings",
            "example": "POST /embeddings with {input: 'The cat sat on the mat'}"
        },
        {
            "name": "/moderations",
            "description": "Check content safety",
            "example": "POST /moderations with {input: 'Some text to check'}"
        },
        {
            "name": "/stream",
            "description": "Streaming responses",
            "example": "GET /stream with same parameters as completions"
        }
    ]

    for e in endpoints:
        print(f"\n  • {e['name']}:")
        print(f"    {e['description']}")
        print(f"    {e['example']}")

api_endpoints()
```

---

## Request Parameters

### Designing Flexible Inputs

```python

def request_params():
    """
    Common request parameters
    """
    print("Request Parameters: Giving Users Control")
    "=" * 60

    params = [
        {
            "name": "model",
            "description": "Which model to use",
            "example": "gpt-4, llama-3-70b, claude-3-opus"
        },
        {
            "name": "messages/prompt",
            "description": "The input text",
            "example": "[{'role': 'user', 'content': 'Hello'}]"
        },
        {
            "name": "max_tokens",
            "description": "Maximum response length",
            "example": "100",
            "tradeoff": "Longer = more cost, more latency"
        },
        {
            "name": "temperature",
            "description": "Randomness (0-2)",
            "example": "0.7",
            "tradeoff": "Higher = more creative, less focused"
        },
        {
            "name": "top_p",
            "description": "Nucleus sampling",
            "example": "0.9",
            "tradeoff": "Alternative to temperature"
        },
        {
            "name": "n",
            "description": "Number of completions",
            "example": "1",
            "tradeoff": "More = higher cost"
        },
        {
            "name": "stream",
            "description": "Stream tokens incrementally",
            "example": "false",
            "tradeoff": "True = better UX, more complex"
        },
        {
            "name": "stop",
            "description": "Stop sequences",
            "example": "['\n', 'Human:']",
            "tradeoff": "Control response ending"
        }
    ]

    for p in params:
        print(f"\n  • {p['name']}:")
        print(f"    {p['description']}")
        print(f"    Example: {p['example']}")
        if 'tradeoff' in p:
            print(f"    Tradeoff: {p['tradeoff']}")

request_params()
```

---

## Response Format

### Consistent Output Structure

```python

def response_format():
    """
    Well-structured API responses
    """
    print("Response Format: Predictable Output")
    "=" * 60

    response_example = """
    {
        "id": "cmpl-123abc",
        "object": "chat.completion",
        "created": 1678901234,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Paris is the capital of France."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 8,
            "total_tokens": 14
        }
    }
    """

    print("Example response:")
    print(response_example)

    print("\nKey elements:")
    print("  • id: Unique identifier for tracking/debugging")
    print("  • choices: Array of completions (if n > 1)")
    print("  • finish_reason: Why generation stopped")
    print("  • usage: Token counts for billing/monitoring")

response_format()
```

---

## Streaming vs Non-Streaming

### The Tradeoff

```python

def streaming():
    """
    Streaming responses for better UX
    """
    print("Streaming vs Non-Streaming APIs")
    "=" * 60

    print("""
    NON-STREAMING (batch):
    ┌─────────────────────────────────────────────┐
    │ Request ──► [Wait 2 seconds] ──► Full Response│
    └─────────────────────────────────────────────┘
    • Simple to implement
    • One response, one parse
    • User waits, may think it's broken

    STREAMING (SSE):
    ┌─────────────────────────────────────────────┐
    │ Request ──► Token1 ──► Token2 ──► ... ──► Done│
    └─────────────────────────────────────────────┘
    • Complex implementation
    • Real-time feedback
    • Better user experience
    • Can start displaying immediately
    """)

streaming()
```

### Streaming Implementation

```python

def streaming_code():
    """
    Server-Sent Events (SSE) example
    """
    print("Streaming with Server-Sent Events")
    "=" * 60

    print("""
    # Server-side (Python with FastAPI)
    from fastapi import FastAPI
    from sse_starlette.sse import EventSourceResponse

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        async def generate():
            async for token in model.stream(prompt):
                yield {
                    "event": "completion",
                    "data": json.dumps({"token": token})
                }
        return EventSourceResponse(generate())

    # Client-side (JavaScript)
    const eventSource = new EventSource('/stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        displayToken(data.token);
    };
    """)

streaming_code()
```

---

## Authentication and Rate Limiting

### Keeping Your API Secure

```python

def auth_rate_limiting():
    """
    Authentication and rate limiting
    """
    print("Authentication & Rate Limiting")
    "=" * 60

    auth_methods = [
        {"method": "API Keys", "pros": "Simple", "cons": "Less secure, can't rotate easily"},
        {"method": "JWT Tokens", "pros": "Stateless, scoped", "cons": "More complex"},
        {"method": "OAuth2", "pros": "Industry standard", "cons": "Overkill for simple APIs"},
        {"method": "API Key + IP whitelist", "pros": "More secure", "cons": "Less flexible"}
    ]

    print("Authentication options:")
    for a in auth_methods:
        print(f"  • {a['method']:20} | Pros: {a['pros']} | Cons: {a['cons']}")

    print("\n" + "=" * 40)
    print("Rate Limiting Strategies:")
    print("=" * 40)

    rate_limits = [
        {"type": "Requests per minute", "example": "60 RPM", "use": "General purpose"},
        {"type": "Tokens per minute", "example": "90k TPM", "use": "Better for LLMs"},
        {"type": "Concurrent requests", "example": "5 concurrent", "use": "Prevent overload"},
        {"type": "Cost-based", "example": "$100/month", "use": "Budget control"}
    ]

    for r in rate_limits:
        print(f"  • {r['type']:25} | {r['example']:15} | {r['use']}")

auth_rate_limiting()
```

---

## Error Handling

### Helping Developers Debug

```python

def error_handling():
    """
    Clear error messages
    """
    print("Error Handling: Help, Not Frustration")
    "=" * 60

    good_errors = [
        {
            "code": 400,
            "error": "Invalid parameter: temperature must be between 0 and 2",
            "help": "Clear what's wrong and how to fix it"
        },
        {
            "code": 401,
            "error": "Invalid API key. Check your key at https://dashboard.example.com",
            "help": "Tells user where to fix the problem"
        },
        {
            "code": 429,
            "error": "Rate limit exceeded. Limit: 60 RPM. Reset in 30 seconds",
            "help": "Actionable information about limits"
        },
        {
            "code": 500,
            "error": "Internal server error. Request ID: abc-123-def",
            "help": "Traceable for support"
        }
    ]

    bad_errors = [
        "Error 500",
        "Something went wrong",
        "Invalid request",
        "Access denied"
    ]

    print("❌ BAD ERRORS (unhelpful):")
    for e in bad_errors:
        print(f"  • {e}")

    print("\n✅ GOOD ERRORS (helpful):")
    for e in good_errors:
        print(f"\n  • {e['code']}: {e['error']}")
        print(f"    → {e['help']}")

error_handling()
```

---

## Usage Tracking and Billing

### Token Counting and Pricing

```python

def usage_tracking():
    """
    Tracking usage for billing
    """
    print("Usage Tracking & Billing")
    "=" * 60

    print("""
    Every response should include token usage:

    {
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 28,
            "total_tokens": 43
        }
    }

    Pricing models:

    1. PAY-PER-TOKEN (most common)
       • $0.01 per 1K prompt tokens
       • $0.03 per 1K completion tokens
       • Fair, scales with usage

    2. SUBSCRIPTION TIERS
       • $20/month for 1M tokens
       • $100/month for 10M tokens
       • Predictable for customers

    3. FREE TIER
       • First 100K tokens free
       • Great for onboarding
    """)

usage_tracking()
```

---

## Versioning

### Managing Change

```python

def versioning():
    """
    API versioning strategies
    """
    print("API Versioning: Evolving Without Breaking")
    "=" * 60

    strategies = [
        {
            "method": "URL versioning",
            "example": "/v1/chat/completions, /v2/chat/completions",
            "pros": "Clear, cacheable",
            "cons": "URLs change"
        },
        {
            "method": "Header versioning",
            "example": "Accept: application/vnd.example.v1+json",
            "pros": "URLs stay clean",
            "cons": "Harder to test"
        },
        {
            "method": "Query parameter",
            "example": "/chat/completions?version=1",
            "pros": "Simple",
            "cons": "Can be forgotten"
        },
        {
            "method": "No versioning (evolve)",
            "example": "Add fields, don't remove",
            "pros": "Simple for users",
            "cons": "Can get messy"
        }
    ]

    print("Versioning strategies:")
    for s in strategies:
        print(f"\n  • {s['method']}:")
        print(f"    Example: {s['example']}")
        print(f"    Pros: {s['pros']} | Cons: {s['cons']}")

versioning()
```

---

## Example: OpenAI-Compatible API

```python

def openai_compatible():
    """
    OpenAI API specification (industry standard)
    """
    print("OpenAI-Compatible API (Industry Standard)")
    "=" * 60

    print("""
    POST /v1/chat/completions
    {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": false
    }

    Response:
    {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 16,
            "completion_tokens": 7,
            "total_tokens": 23
        }
    }

    Why this matters: Many tools (LangChain, AutoGPT) expect this format!
    """)

openai_compatible()
```

---

## API Design Cheat Sheet

| Component      | Best Practice                           | Why              |
| -------------- | --------------------------------------- | ---------------- |
| Endpoints      | /v1/chat/completions, /v1/completions   | Clear, versioned |
| Authentication | API keys in headers                     | Simple, standard |
| Parameters     | Descriptive names, sensible defaults    | Easy to use      |
| Responses      | Consistent structure, include usage     | Predictable      |
| Errors         | Clear messages, HTTP status codes       | Debuggable       |
| Rate limits    | Return headers with limits              | Transparent      |
| Streaming      | Server-Sent Events (SSE)                | Real-time UX     |
| Documentation  | OpenAPI/Swagger, interactive playground | Onboarding       |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Your API is your product's interface to the world:         ║
║                                                              ║
║  🚀 ADOPTION: A well-designed API gets integrated           ║
║     faster. Developers choose APIs that are easy to use.    ║
║                                                              ║
║  💰 REVENUE: Clear usage tracking enables fair              ║
║     billing. Hidden costs frustrate users.                  ║
║                                                              ║
║  🔧 MAINTAINABILITY: Versioning lets you improve            ║
║     without breaking existing users.                        ║
║                                                              ║
║  🛡️ SECURITY: Proper authentication prevents abuse.        ║
║     Rate limiting protects your infrastructure.              ║
║                                                              ║
║  📈 SCALABILITY: Good API design works with                  ║
║     caching, CDNs, and load balancers.                       ║
║                                                              ║
║  🌍 ECOSYSTEM: Following standards (OpenAI format)          ║
║     makes your API work with thousands of existing tools.    ║
║                                                              ║
║  A great model behind a terrible API will fail.             ║
║  A good model with a great API will succeed.                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Good API design balances simplicity, flexibility, and reliability—clear endpoints, intuitive parameters, consistent responses, and helpful errors

• Key decisions include streaming vs batch, authentication methods, rate limiting, versioning, and usage tracking—each affecting user experience and operational costs

• Following industry standards (like OpenAI's API format) maximizes compatibility—allowing your API to work with existing tools and libraries

---

## Mental Hook

> "API design is like designing a restaurant menu—the food (model) might be amazing, but if the menu is confusing, prices are hidden, and ordering is complicated, no one will eat there. A great menu makes customers want to order more."
