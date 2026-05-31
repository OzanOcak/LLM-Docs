# A/B Testing for LLM Applications

## The Restaurant Recipe Experiment Analogy

Imagine you're a chef with a popular pasta dish. You think a new recipe might be even better. You can't just change the recipe for everyone—if it's worse, you'll lose customers. Instead, you make both versions, serve the new one to every 10th customer, and ask for feedback. After a week, you compare ratings and choose the winner. That's A/B testing: systematically comparing two versions of something to see which performs better.

In LLM applications, A/B testing is essential for making data-driven decisions. Should you use GPT-4 or GPT-3.5? A longer prompt or a shorter one? Temperature 0.7 or 0.9? Instead of guessing, you run experiments, collect data, and let the numbers decide.

---

## What Is A/B Testing?

### The Core Concept

```python

def ab_testing_intro():
    """
    The basic idea of A/B testing
    """
    print("A/B Testing: Let Data Decide")
    print("=" * 60)

    print("""
    ┌─────────────────────────────────────────────────────┐
    │                    USERS                              │
    │  (all traffic to your application)                    │
    └─────────────────────┬───────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
    ┌─────────────────────┐  ┌─────────────────────┐
    │    VARIANT A        │  │    VARIANT B        │
    │  (Control)          │  │  (Treatment)        │
    │ • Current model     │  │ • New model         │
    │ • Old prompt        │  │ • New prompt        │
    │ • Temperature 0.7   │  │ • Temperature 0.9   │
    └─────────────────────┘  └─────────────────────┘
              ↓                       ↓
    ┌─────────────────────┐  ┌─────────────────────┐
    │   Collect metrics   │  │   Collect metrics   │
    │   • User ratings    │  │   • User ratings    │
    │   • Completion rate │  │   • Completion rate │
    │   • Latency         │  │   • Latency         │
    └─────────────────────┘  └─────────────────────┘
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
        ┌─────────────────────────────────────────────┐
        │         STATISTICAL ANALYSIS                 │
        │  Is B significantly better than A?           │
        │  If yes → roll out B to everyone             │
        │  If no → keep A or try something else        │
        └─────────────────────────────────────────────┘
    """)

ab_testing_intro()
```

---

## What Can You A/B Test in LLM Apps?

### Testable Elements

```python

def testable_elements():
    """
    Different things you can test
    """
    print("What to Test in LLM Applications")
    "=" * 60

    test_categories = [
        {
            "category": "Model selection",
            "examples": [
                "GPT-4 vs GPT-3.5",
                "Claude vs Llama",
                "Fine-tuned vs base model"
            ]
        },
        {
            "category": "Prompt engineering",
            "examples": [
                "System prompt variations",
                "Few-shot examples count",
                "Instruction wording"
            ]
        },
        {
            "category": "Parameters",
            "examples": [
                "Temperature (0.7 vs 0.9)",
                "Top-p (0.9 vs 0.95)",
                "Max tokens (500 vs 1000)"
            ]
        },
        {
            "category": "Features",
            "examples": [
                "With RAG vs without",
                "Streaming vs non-streaming",
                "With citations vs without"
            ]
        },
        {
            "category": "UI/UX",
            "examples": [
                "Response format (bullets vs paragraphs)",
                "Tone (formal vs casual)",
                "Length (concise vs detailed)"
            ]
        }
    ]

    for cat in test_categories:
        print(f"\n  • {cat['category']}:")
        for ex in cat['examples']:
            print(f"    - {ex}")

testable_elements()
```

---

## Setting Up an A/B Test

### Step-by-Step Process

```python

def ab_test_process():
    """
    The complete A/B testing workflow
    """
    print("A/B Testing Workflow")
    "=" * 60

    steps = [
        {
            "step": "1. Define hypothesis",
            "example": '"Using GPT-4 will increase user satisfaction by 10%"'
        },
        {
            "step": "2. Choose metrics",
            "example": "Primary: thumbs up/down rate. Secondary: completion rate, latency"
        },
        {
            "step": "3. Determine sample size",
            "example": "Need 1000 users per variant for statistical significance"
        },
        {
            "step": "4. Split traffic",
            "example": "50% to A (control), 50% to B (treatment)"
        },
        {
            "step": "5. Run experiment",
            "example": "Collect data for 1 week or until target sample reached"
        },
        {
            "step": "6. Analyze results",
            "example": "Calculate p-value, confidence intervals"
        },
        {
            "step": "7. Make decision",
            "example": "Roll out winner, or iterate with new hypothesis"
        }
    ]

    for s in steps:
        print(f"\n{s['step']}:")
        print(f"  {s['example']}")

ab_test_process()
```

### Traffic Splitting Code

```python

import random
import hashlib

def traffic_split_demo():
    """
    How to split traffic consistently
    """
    print("Consistent Traffic Splitting")
    "=" * 60

    def get_variant(user_id, variants=['A', 'B'], weights=[0.5, 0.5]):
        """Deterministic variant assignment based on user_id"""
        # Hash the user_id to get a consistent value
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        # Map to 0-1 range
        r = (hash_val % 1000) / 1000

        # Assign based on cumulative weights
        cumsum = 0
        for variant, weight in zip(variants, weights):
            cumsum += weight
            if r < cumsum:
                return variant
        return variants[-1]

    # Simulate users
    users = [f"user_{i}" for i in range(20)]

    print("User assignments (consistent across sessions):")
    counts = {'A': 0, 'B': 0}

    for user in users:
        variant = get_variant(user)
        counts[variant] += 1
        print(f"  {user}: {variant}")

    print(f"\nDistribution: A={counts['A']}, B={counts['B']}")

traffic_split_demo()
```

---

## Metrics to Track

### Primary and Secondary Metrics

```python

def ab_metrics():
    """
    What to measure in LLM A/B tests
    """
    print("Key Metrics for LLM A/B Tests")
    "=" * 60

    metrics = [
        {
            "name": "User satisfaction",
            "how": "Thumbs up/down, star ratings",
            "importance": "⭐⭐⭐⭐⭐",
            "note": "The ultimate measure"
        },
        {
            "name": "Task completion rate",
            "how": "Did user get what they wanted?",
            "importance": "⭐⭐⭐⭐",
            "note": "Especially for specific tasks"
        },
        {
            "name": "Retention",
            "how": "Do users come back?",
            "importance": "⭐⭐⭐⭐",
            "note": "Long-term success metric"
        },
        {
            "name": "Latency",
            "how": "Time to first token, total time",
            "importance": "⭐⭐⭐",
            "note": "Affects user experience"
        },
        {
            "name": "Cost per request",
            "how": "API costs, compute costs",
            "importance": "⭐⭐⭐",
            "note": "Business metric"
        },
        {
            "name": "Factual accuracy",
            "how": "Human evaluation, automated checks",
            "importance": "⭐⭐⭐⭐",
            "note": "Critical for some apps"
        },
        {
            "name": "Toxicity rate",
            "how": "Automated moderation scores",
            "importance": "⭐⭐⭐⭐",
            "note": "Safety metric"
        }
    ]

    for m in metrics:
        print(f"\n  • {m['name']}:")
        print(f"    How: {m['how']}")
        print(f"    Importance: {m['importance']} | {m['note']}")

ab_metrics()
```

---

## Statistical Significance

### Avoiding False Conclusions

```python

def statistical_significance():
    """
    Making sure your results aren't random
    """
    print("Statistical Significance: Don't Be Fooled")
    "=" * 60

    print("""
    You ran an experiment:
    • Variant A: 55% satisfaction (1000 users)
    • Variant B: 58% satisfaction (1000 users)

    Is B really better, or just random chance?

    STATISTICAL TESTS:

    p-value = probability that the difference is due to chance
    • p < 0.05 → Statistically significant (95% confident)
    • p < 0.01 → Highly significant (99% confident)
    • p > 0.05 → Not significant (keep testing)

    CONFIDENCE INTERVALS:
    • A: 55% ± 3% (52-58%)
    • B: 58% ± 3% (55-61%)

    If intervals overlap, difference may not be real.
    """)

statistical_significance()
```

### Sample Size Calculator

```python

def sample_size():
    """
    How many users you need
    """
    print("Sample Size Matters")
    "=" * 60

    import math

    def sample_size_needed(baseline, minimum_detectable_effect, significance=0.05, power=0.8):
        """Simplified sample size calculation"""
        # This is a simplified version - real calculation is more complex
        z_significance = 1.96  # for 95% confidence
        z_power = 0.84  # for 80% power

        p1 = baseline
        p2 = baseline + minimum_detectable_effect

        pooled_p = (p1 + p2) / 2
        effect_size = abs(p1 - p2) / math.sqrt(2 * pooled_p * (1 - pooled_p))

        n = 2 * ((z_significance + z_power) / effect_size) ** 2
        return int(math.ceil(n))

    baseline = 0.5  # 50% satisfaction currently
    effects = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10% improvement

    print(f"Baseline satisfaction: {baseline*100:.0f}%")
    print("\nUsers needed per variant to detect:")

    for effect in effects:
        n = sample_size_needed(baseline, effect)
        print(f"  {effect*100:.0f}% improvement → {n} users")

sample_size()
```

---

## A/B Testing Platform

### Simple Implementation

```python

def ab_platform():
    """
    A simple A/B testing framework
    """
    print("Simple A/B Testing Framework")
    "=" * 60

    print("""
    class ABTest:
        def __init__(self, name, variants, metrics):
            self.name = name
            self.variants = variants
            self.metrics = metrics
            self.results = {v: {m: [] for m in metrics} for v in variants}

        def assign_variant(self, user_id):
            # Consistent assignment
            return get_variant(user_id, self.variants)

        def track(self, user_id, variant, metric, value):
            self.results[variant][metric].append(value)

        def analyze(self):
            report = {}
            for metric in self.metrics:
                a_values = self.results['A'][metric]
                b_values = self.results['B'][metric]

                # Calculate statistics
                report[metric] = {
                    'A_mean': sum(a_values)/len(a_values),
                    'B_mean': sum(b_values)/len(b_values),
                    'lift': (sum(b_values)/len(b_values) - sum(a_values)/len(a_values)) / (sum(a_values)/len(a_values)),
                    'p_value': calculate_p_value(a_values, b_values)
                }
            return report

    # Usage
    test = ABTest(
        name="model_comparison",
        variants=['A', 'B'],
        metrics=['satisfaction', 'latency']
    )

    # In your API endpoint
    user_id = request.user_id
    variant = test.assign_variant(user_id)

    # Use appropriate model for this variant
    if variant == 'A':
        response = model_a.generate(prompt)
    else:
        response = model_b.generate(prompt)

    # Track results
    test.track(user_id, variant, 'latency', response_time)
    test.track(user_id, variant, 'satisfaction', user_rating)
    """)

ab_platform()
```

---

## Common Pitfalls

```python

def pitfalls():
    """
    Mistakes to avoid in A/B testing
    """
    print("Common A/B Testing Pitfalls")
    "=" * 60

    pitfalls = [
        {
            "name": "Stopping too early",
            "problem": "Results look significant after 100 users, but reverse after 1000",
            "solution": "Pre-determine sample size, don't peek"
        },
        {
            "name": "Multiple testing",
            "problem": "Testing 20 variants increases chance of false positive",
            "solution": "Correct for multiple comparisons (Bonferroni)"
        },
        {
            "name": "Novelty effect",
            "problem": "Users rate new variant higher just because it's new",
            "solution": "Run test longer (2-4 weeks)"
        },
        {
            "name": "Selection bias",
            "problem": "Different user segments get different variants",
            "solution": "Randomize properly, check segment balance"
        },
        {
            "name": "Ignoring external factors",
            "problem": "Holiday, news events affect results",
            "solution": "Run control group simultaneously"
        }
    ]

    for p in pitfalls:
        print(f"\n  • {p['name']}:")
        print(f"    Problem: {p['problem']}")
        print(f"    Solution: {p['solution']}")

pitfalls()
```

---

## A/B Testing Cheat Sheet

| Phase          | What to Do                              | Common Mistake                       |
| -------------- | --------------------------------------- | ------------------------------------ |
| Design         | Define hypothesis, metrics, sample size | Vague goals, too many metrics        |
| Implementation | Split traffic consistently              | Leaky assignment (user sees both)    |
| Execution      | Run to completion                       | Stopping early                       |
| Analysis       | Check statistical significance          | Ignoring practical significance      |
| Decision       | Roll out winner or iterate              | Falling in love with your hypothesis |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  A/B testing replaces opinions with data:                   ║
║                                                              ║
║  🧠 AGAINST GUT FEELINGS: "I think users prefer..."        ║
║     is replaced by "Data shows users prefer..."             ║
║                                                              ║
║  💰 ROI JUSTIFICATION: Prove that upgrading to              ║
║     GPT-4 actually increases user satisfaction enough       ║
║     to justify 10x cost.                                    ║
║                                                              ║
║  🚀 CONTINUOUS IMPROVEMENT: Small wins compound.            ║
║     A 1% improvement per week is 67% better in a year.      ║
║                                                              ║
║  🛡️ RISK REDUCTION: Catch bad changes before                ║
║     rolling out to everyone.                                ║
║                                                              ║
║  📈 PRODUCT-MARKET FIT: Learn what users actually           ║
║     value, not what you think they value.                   ║
║                                                              ║
║  Companies that A/B test systematically outperform          ║
║  those that rely on intuition. It's not magic—it's math.    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• A/B testing systematically compares two versions of something to see which performs better—replacing guesswork with data-driven decisions

• Key elements include hypothesis, metrics, sample size, traffic splitting, and statistical analysis—each crucial for valid results

• Common pitfalls like stopping early, multiple testing, and novelty effects can invalidate tests—understanding statistics helps avoid them

---

## Mental Hook

> "A/B testing is like being a scientist in your own product lab. Instead of asking 'what do users want?', you run experiments and let them tell you. Every change becomes a hypothesis, every launch becomes a test, and every metric becomes a clue."
