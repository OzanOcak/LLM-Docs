# Monitoring and Observability

## The Airplane Cockpit Analogy

Imagine flying a modern airplane without any instruments—no altimeter, no speed indicator, no fuel gauge, no warning lights. You'd have no idea if you're about to stall, run out of fuel, or fly into a mountain. That's what running an LLM service without monitoring is like. A proper cockpit gives pilots real-time data, alerts, and insights to fly safely. That's monitoring and observability for LLM services: collecting metrics, logs, and traces to understand what's happening, debug issues, and ensure reliability.

In production LLM deployment, monitoring isn't optional—it's essential. You need to know if your service is slow, if it's costing too much, if users are getting bad responses, or if something is about to break. Good observability turns a black box into a transparent, manageable system.

---

## What Is Monitoring and Observability?

### The Core Concepts

```python

def observability_intro():
    """
    The three pillars of observability
    """
    print("The Three Pillars of Observability")
    print("=" * 60)

    print("""
    ┌─────────────────────────────────────────────────────┐
    │                      METRICS                         │
    │  • Numerical measurements over time                  │
    │  • Examples: requests/sec, latency, error rate       │
    │  • Tells you WHAT is happening                       │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │                       LOGS                            │
    │  • Detailed records of events                        │
    │  • Examples: request/response pairs, errors, warnings│
    │  • Tells you WHY something happened                  │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │                      TRACES                           │
    │  • End-to-end request journey                        │
    │  • Examples: API gateway → model → database → response│
    │  • Tells you WHERE something happened                │
    └─────────────────────────────────────────────────────┘
    """)

observability_intro()
```

---

## Key Metrics to Monitor

### Performance Metrics

```python

def performance_metrics():
    """
    Metrics that tell you how your service is performing
    """
    print("Performance Metrics")
    "=" * 60

    metrics = [
        {
            "name": "Requests per second (RPS)",
            "what": "How many requests are coming in",
            "alert": "Sudden spike or drop",
            "target": "Depends on capacity"
        },
        {
            "name": "Latency (p50, p95, p99)",
            "what": "Response time in milliseconds",
            "alert": "p95 > 500ms",
            "target": "p95 < 200ms"
        },
        {
            "name": "Error rate",
            "what": "Percentage of failed requests",
            "alert": "> 1%",
            "target": "< 0.1%"
        },
        {
            "name": "Concurrent requests",
            "what": "How many requests are processing simultaneously",
            "alert": "Near capacity limit",
            "target": "< 80% of max"
        },
        {
            "name": "Queue depth",
            "what": "How many requests are waiting",
            "alert": "Growing queue",
            "target": "Near zero"
        }
    ]

    print("Essential performance metrics:")
    for m in metrics:
        print(f"\n  • {m['name']}:")
        print(f"    What: {m['what']}")
        print(f"    Alert if: {m['alert']}")
        print(f"    Target: {m['target']}")

performance_metrics()
```

### Resource Metrics

```python

def resource_metrics():
    """
    Metrics about your infrastructure
    """
    print("Resource Metrics")
    "=" * 60

    metrics = [
        {
            "name": "GPU utilization",
            "what": "Percentage of GPU compute used",
            "alert": "> 95% for extended periods",
            "target": "70-90%"
        },
        {
            "name": "GPU memory",
            "what": "VRAM usage",
            "alert": "> 90%",
            "target": "< 80%"
        },
        {
            "name": "CPU utilization",
            "what": "CPU usage for preprocessing/postprocessing",
            "alert": "> 80%",
            "target": "< 60%"
        },
        {
            "name": "RAM usage",
            "what": "System memory",
            "alert": "> 90%",
            "target": "< 75%"
        },
        {
            "name": "Disk I/O",
            "what": "Reading/writing to disk",
            "alert": "Sustained high I/O",
            "target": "Low"
        },
        {
            "name": "Network I/O",
            "what": "Data in/out",
            "alert": "Near bandwidth limit",
            "target": "< 50%"
        }
    ]

    print("Essential resource metrics:")
    for m in metrics:
        print(f"\n  • {m['name']}:")
        print(f"    What: {m['what']}")
        print(f"    Alert if: {m['alert']}")
        print(f"    Target: {m['target']}")

resource_metrics()
```

### Business Metrics

```python

def business_metrics():
    """
    Metrics that matter to the business
    """
    print("Business Metrics")
    "=" * 60

    metrics = [
        {
            "name": "Token count (prompt + completion)",
            "what": "Total tokens processed",
            "why": "Cost tracking, usage patterns"
        },
        {
            "name": "Cost per request",
            "what": "Average cost to serve a request",
            "why": "Profitability, pricing decisions"
        },
        {
            "name": "User satisfaction",
            "what": "Ratings, feedback, thumbs up/down",
            "why": "Product quality, user retention"
        },
        {
            "name": "Cache hit rate",
            "what": "Percentage of requests served from cache",
            "why": "Cost savings, latency improvement"
        },
        {
            "name": "Unique users",
            "what": "Number of distinct users",
            "why": "Growth tracking, engagement"
        },
        {
            "name": "Retention",
            "what": "Users returning after first use",
            "why": "Product-market fit"
        }
    ]

    print("Business metrics to track:")
    for m in metrics:
        print(f"\n  • {m['name']}:")
        print(f"    What: {m['what']}")
        print(f"    Why: {m['why']}")

business_metrics()
```

---

## Logging

### What to Log

```python

def logging_guide():
    """
    Effective logging practices
    """
    print("Logging: The Detailed Record")
    "=" * 60

    print("""
    LOG EVERY REQUEST:
    ┌─────────────────────────────────────────────────────┐
    │ {                                                      │
    │   "timestamp": "2024-01-15T10:30:00Z",               │
    │   "request_id": "req_123abc",                         │
    │   "user_id": "user_456",                              │
    │   "model": "gpt-4",                                    │
    │   "prompt_tokens": 150,                                │
    │   "completion_tokens": 50,                             │
    │   "latency_ms": 450,                                   │
    │   "success": true,                                     │
    │   "error": null                                        │
    │ }                                                      │
    └─────────────────────────────────────────────────────┘

    LOG ERRORS WITH CONTEXT:
    ┌─────────────────────────────────────────────────────┐
    │ {                                                      │
    │   "timestamp": "2024-01-15T10:30:05Z",               │
    │   "request_id": "req_789xyz",                         │
    │   "error_type": "rate_limit_exceeded",                │
    │   "error_message": "User exceeded 60 RPM limit",      │
    │   "stack_trace": "...",                                │
    │   "user_tier": "free"                                  │
    │ }                                                      │
    └─────────────────────────────────────────────────────┘
    """)

logging_guide()
```

### Log Levels

```python

def log_levels():
    """
    Different log levels for different purposes
    """
    print("Log Levels")
    "=" * 60

    levels = [
        {"level": "DEBUG", "use": "Development, troubleshooting", "example": "Loaded model weights in 2.3s"},
        {"level": "INFO", "use": "Normal operations", "example": "Request processed successfully"},
        {"level": "WARNING", "use": "Potential issues", "example": "High latency detected (950ms)"},
        {"level": "ERROR", "use": "Failed operations", "example": "Model inference failed, retrying"},
        {"level": "CRITICAL", "use": "System failures", "example": "GPU out of memory, service degraded"}
    ]

    for l in levels:
        print(f"  • {l['level']:10} | {l['use']:25} | {l['example']}")

log_levels()
```

---

## Tracing

### End-to-End Request Tracking

```python

def tracing():
    """
    Distributed tracing for complex systems
    """
    print("Tracing: Following the Request Journey")
    "=" * 60

    print("""
    A single request might travel through multiple services:

    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │   API      │────▶│  Auth      │────▶│  Rate      │
    │  Gateway   │     │  Service   │     │   Limiter  │
    └────────────┘     └────────────┘     └────────────┘
                                                │
                                                ▼
    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │ Response   │◀────│   Model    │◀────│   Cache    │
    │   to User  │     │  Service   │     │   Check    │
    └────────────┘     └────────────┘     └────────────┘

    Tracing gives you:
    • Time spent in each service
    • Bottlenecks in the chain
    • Error propagation
    • Dependency visualization
    """)

tracing()
```

### Trace Example

```python

def trace_example():
    """
    Example trace structure
    """
    print("Example Trace (OpenTelemetry format)")
    "=" * 60

    print("""
    {
        "trace_id": "abc123def456",
        "spans": [
            {
                "name": "POST /chat/completions",
                "start_time": "10:30:00.000",
                "end_time": "10:30:00.450",
                "attributes": {
                    "http.method": "POST",
                    "http.status_code": 200
                },
                "child_spans": [
                    {
                        "name": "authenticate",
                        "start_time": "10:30:00.001",
                        "end_time": "10:30:00.005",
                        "attributes": {"user_id": "user_123"}
                    },
                    {
                        "name": "rate_limit_check",
                        "start_time": "10:30:00.006",
                        "end_time": "10:30:00.007",
                        "attributes": {"tokens_used": 150}
                    },
                    {
                        "name": "model_inference",
                        "start_time": "10:30:00.008",
                        "end_time": "10:30:00.445",
                        "attributes": {
                            "model": "gpt-4",
                            "prompt_tokens": 150,
                            "completion_tokens": 50
                        }
                    }
                ]
            }
        ]
    }
    """)

trace_example()
```

---

## Alerting

### Setting Up Alerts

```python

def alerting():
    """
    When to notify humans
    """
    print("Alerting: Wake Me Up When Something Breaks")
    "=" * 60

    alerts = [
        {"condition": "Error rate > 1% for 5 minutes", "severity": "P1", "action": "Page on-call engineer"},
        {"condition": "p95 latency > 1s for 10 minutes", "severity": "P2", "action": "Investigate next business day"},
        {"condition": "GPU memory > 90%", "severity": "P2", "action": "Check for memory leak"},
        {"condition": "Cost > $1000/day", "severity": "P3", "action": "Review usage patterns"},
        {"condition": "Cache hit rate < 20%", "severity": "P3", "action": "Review caching strategy"},
        {"condition": "Queue depth > 100", "severity": "P1", "action": "Scale up immediately"}
    ]

    print("Alert rules example:")
    for a in alerts:
        print(f"\n  • {a['condition']}")
        print(f"    Severity: {a['severity']} | Action: {a['action']}")

alerting()
```

### Alert Severities

```python

def severities():
    """
    P0-P4 severity levels
    """
    print("Alert Severity Levels")
    "=" * 60

    severities = [
        {"level": "P0", "meaning": "Complete service outage", "response": "Immediate, 24/7"},
        {"level": "P1", "meaning": "Severe degradation", "response": "Within 1 hour"},
        {"level": "P2", "meaning": "Partial degradation", "response": "Within 8 hours"},
        {"level": "P3", "meaning": "Minor issue", "response": "Next business day"},
        {"level": "P4", "meaning": "Cosmetic issue", "response": "Next release"}
    ]

    for s in severities:
        print(f"  • {s['level']}: {s['meaning']:25} | Response: {s['response']}")

severities()
```

---

## Dashboards

### Visualizing the Data

```python

def dashboards():
    """
    Creating effective dashboards
    """
    print("Dashboards: At-a-Glance Status")
    "=" * 60

    print("""
    ┌─────────────────────────────────────────────────────┐
    │                 LLM SERVICE DASHBOARD                │
    ├───────────────┬─────────────────────────────────────┤
    │               │                                     │
    │ Current RPS:  │  Latency (ms):                      │
    │   245 req/s   │  p50: 120  [████████░░░░░░░░░░]    │
    │               │  p95: 350  [██████████████░░░░]    │
    │ Error rate:   │  p99: 580  [██████████████████░░]  │
    │   0.2%        │                                     │
    │               │  GPU Memory:                        │
    │ Active users: │  ████████████████░░░░  78%         │
    │   1,234       │                                     │
    ├───────────────┼─────────────────────────────────────┤
    │               │                                     │
    │ Cost today:   │  Top errors:                        │
    │   $234.50     │  • rate_limit (45%)                 │
    │               │  • timeout (30%)                     │
    │ Monthly cost: │  • invalid_key (15%)                 │
    │   $4,567      │  • model_error (10%)                 │
    └───────────────┴─────────────────────────────────────┘
    """)

dashboards()
```

### Dashboard Types

```python

def dashboard_types():
    """
    Different dashboards for different audiences
    """
    print("Dashboards for Different Audiences")
    "=" * 60

    types = [
        {"audience": "Engineering", "focus": "Latency, errors, resource usage", "purpose": "Debugging, capacity planning"},
        {"audience": "Product", "focus": "User counts, retention, satisfaction", "purpose": "Feature decisions"},
        {"audience": "Business", "focus": "Cost, revenue, ROI", "purpose": "Pricing, budgeting"},
        {"audience": "Executive", "focus": "High-level KPIs, trends", "purpose": "Strategy"}
    ]

    for t in types:
        print(f"\n  • {t['audience']:12} | {t['focus']:40} | {t['purpose']}")

dashboard_types()
```

---

## Tools of the Trade

```python

def monitoring_tools():
    """
    Popular monitoring tools
    """
    print("Popular Monitoring Tools")
    "=" * 60

    tools = [
        {"category": "Metrics", "tools": "Prometheus, Grafana, Datadog, New Relic"},
        {"category": "Logs", "tools": "ELK Stack (Elasticsearch, Logstash, Kibana), Loki, Splunk"},
        {"category": "Traces", "tools": "Jaeger, Zipkin, OpenTelemetry"},
        {"category": "APM", "tools": "Datadog, New Relic, Dynatrace"},
        {"category": "Uptime", "tools": "Pingdom, Uptime Robot, Statuspage"},
        {"category": "LLM-specific", "tools": "LangSmith, Helicone, Lunary"}
    ]

    for t in tools:
        print(f"  • {t['category']:15} | {t['tools']}")

monitoring_tools()
```

---

## Implementing Monitoring

### Code Example: Metrics with Prometheus

```python

def prometheus_example():
    """
    Instrumenting an LLM API with Prometheus
    """
    print("Implementing Metrics with Prometheus")
    "=" * 60

    print("""
    from prometheus_client import Counter, Histogram, Gauge
    import time

    # Define metrics
    requests_total = Counter('requests_total', 'Total requests', ['model', 'endpoint'])
    latency_histogram = Histogram('request_duration_seconds', 'Request latency', ['model'])
    tokens_counter = Counter('tokens_total', 'Total tokens', ['type'])
    gpu_memory_gauge = Gauge('gpu_memory_used_bytes', 'GPU memory used')

    # In your request handler
    @app.post("/chat/completions")
    async def chat_completion(request: Request):
        start_time = time.time()

        # Increment request counter
        requests_total.labels(model=request.model, endpoint="/chat").inc()


        try:
            # Process request
            result = await model.generate(request.messages)

            # Record latency
            latency = time.time() - start_time
            latency_histogram.labels(model=request.model).observe(latency)

            # Count tokens
            tokens_counter.labels(type='prompt').inc(request.prompt_tokens)
            tokens_counter.labels(type='completion').inc(request.completion_tokens)

            return result

        except Exception as e:
            # Track errors
            requests_total.labels(model=request.model, endpoint="/chat", error=str(e)).inc()
            raise
    """)

prometheus_example()
```

---

## Monitoring Cheat Sheet

| What to Monitor   | Why               | Alert Threshold | Tool                 |
| ----------------- | ----------------- | --------------- | -------------------- |
| Request rate      | Capacity planning | Sudden 2x spike | Prometheus           |
| Latency (p95)     | User experience   | > 500ms         | Prometheus           |
| Error rate        | Service health    | > 1%            | Prometheus           |
| GPU memory        | OOM prevention    | > 90%           | nvidia-smi + Grafana |
| Token usage       | Cost tracking     | > budget        | Custom counter       |
| User satisfaction | Product quality   | < 4 stars       | Database             |
| Cache hit rate    | Efficiency        | < 20%           | Redis metrics        |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Without monitoring, you're flying blind:                   ║
║                                                              ║
║  🚨 DETECT PROBLEMS BEFORE USERS DO: Latency spikes,        ║
║     error bursts, memory leaks—catch them early.             ║
║                                                              ║
║  🔍 DEBUG FAST: When something breaks, logs and traces      ║
║     tell you exactly what happened and why.                  ║
║                                                              ║
║  📈 CAPACITY PLANNING: Know when you need to scale          ║
║     up before you run out of resources.                      ║
║                                                              ║
║  💰 COST OPTIMIZATION: See where money is going and         ║
║     optimize expensive patterns.                             ║
║                                                              ║
║  🎯 BUSINESS INSIGHTS: Understand how users are using       ║
║     your product and what they value.                        ║
║                                                              ║
║  🤝 SLA ENFORCEMENT: Prove to customers you're meeting      ║
║     your service level agreements.                           ║
║                                                              ║
║  In production, if you can't measure it, you can't          ║
║  improve it. And if you can't see it breaking,              ║
║  you can't fix it before your users notice.                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Monitoring has three pillars: metrics (what), logs (why), and traces (where)—each giving a different view of your system's health

• Key metrics include request rate, latency (p95/p99), error rate, resource usage, and business metrics like cost and satisfaction—each serving a different purpose

• Alerting turns data into action—notifying the right people at the right time with the right severity to prevent or mitigate issues

---

## Mental Hook

> "Monitoring is like the instrument panel in a airplane—you wouldn't fly without knowing your altitude, speed, and fuel level. Running an LLM service without monitoring is like flying through a storm with a blindfold on."
