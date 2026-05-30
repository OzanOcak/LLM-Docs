# Data Collection

## The Master Chef's Pantry Analogy

Imagine a master chef preparing to cook for a year. Before they can create amazing dishes, they need to stock their pantry—fresh vegetables from farms, spices imported from around the world, meats from trusted butchers. The quality of their ingredients determines the quality of their meals. That's data collection for LLMs: gathering the raw text that will become the model's knowledge. A model trained on Wikipedia alone is like a chef with only vegetables—good for salads, terrible for steak.

In LLM development, data collection is the foundation everything else builds on. The size, quality, and diversity of your training data directly determines what your model will know and how well it will perform. This is why companies spend millions curating datasets before training begins.

---

## What Is Data Collection?

### The Core Idea

Data collection is the process of gathering and selecting the text that will be used to train an LLM.

```text

Raw sources (internet, books, etc.)
    ↓
Data Collection (gathering, filtering, deduplicating)
    ↓
Training Dataset
    ↓
[LLM Training]
    ↓
Trained Model
```

The model's knowledge = what was in its training data!

```python

def data_collection_intro():
    """
    The basic concept of data collection
    """
    print("Data Collection: Stocking the Knowledge Pantry")
    print("=" * 60)

    print("""
    What we're collecting:

    • Text data (websites, books, articles)
    • Code (GitHub, documentation)
    • Conversations (social media, forums)
    • Academic papers (arXiv, PubMed)
    • Structured data (tables, JSON)

    Scale:
    • GPT-3: 570 GB of text (300B tokens)
    • Llama 3: 15 trillion tokens
    • This would take a human 100,000+ years to read!
    """)

data_collection_intro()
```

---

## Where Do We Get Data?

### Common Data Sources

```python

def data_sources():
    """
    Major sources of training data
    """
    print("Major Data Sources for LLMs")
    print("=" * 60)

    sources = {
        "Common Crawl": "Petabytes of web data, crawled since 2008",
        "Wikipedia": "High-quality, structured encyclopedia",
        "Books": "Project Gutenberg, publishers, scanned books",
        "arXiv": "2+ million academic papers",
        "GitHub": "Public code repositories",
        "Reddit": "Discussions, conversations",
        "Stack Exchange": "Q&A, technical content",
        "News articles": "Current events, journalism",
        "Social media": "Twitter, Facebook (less common now)"
    }

    for source, desc in sources.items():
        print(f"  • {source}: {desc}")

data_sources()
```

### The Web Crawl Backbone

```python

def common_crawl():
    """
    Common Crawl: The foundation of most LLMs
    """
    print("Common Crawl: The Internet in a Box")
    print("=" * 60)

    print("""
    Common Crawl is a non-profit that crawls the web
    and makes the data freely available.

    One monthly crawl contains:
    • Billions of web pages
    • Petabytes of raw data
    • Hundreds of languages

    GPT-3 used:
    • 410 billion tokens from Common Crawl
    • Filtered and cleaned
    • 60% of total training data

    Problem: It's messy! Contains:
    • Ads, navigation menus
    • Duplicate content
    • Low-quality pages
    • Spam, porn, hate speech
    """)

common_crawl()
```

---

## Data Quality Considerations

### Not All Data Is Equal

```python

def data_quality():
    """
    Quality dimensions of training data
    """
    print("Data Quality: Good Data, Bad Data")
    print("=" * 60)

    quality_factors = {
        "Accuracy": "Is the information correct?",
        "Recency": "Is it up to date?",
        "Diversity": "Does it cover different topics?",
        "Coherence": "Does it make sense?",
        "Toxicity": "Does it contain harmful content?",
        "Duplication": "Is it unique or repeated?"
    }

    print("Quality dimensions to consider:")
    for factor, question in quality_factors.items():
        print(f"  • {factor}: {question}")

    print("\nExample of low-quality text:")
    print('  "Click here for free money!!! Visit our site now!!! " * 10')

    print("\nExample of high-quality text:")
    print('  "The Industrial Revolution was a period of major mechanization...')

data_quality()
```

### The Pareto Principle in Data

```python

def pareto():
    """
    80/20 rule in data collection
    """
    print("The Pareto Principle: Not All Data Is Equal")
    print("=" * 60)

    print("""
    Often, 20% of your data provides 80% of the value:

    HIGH-VALUE DATA (20%):
    • Wikipedia
    • Books
    • Academic papers
    • High-quality news

    LOW-VALUE DATA (80%):
    • Random blog comments
    • Spam
    • Duplicate content
    • Gibberish

    This is why data filtering is crucial—
    removing the 80% of low-value data
    makes training faster and models better!
    """)

pareto()
```

---

## The Data Collection Pipeline

### Step-by-Step Process

```python

def collection_pipeline():
    """
    The complete data collection pipeline
    """
    print("The Data Collection Pipeline")
    print("=" * 60)

    steps = [
        ("1. Source Identification",
         "Choose where to get data from (web, books, etc.)"),

        ("2. Crawling/Scraping",
         "Download the data (respecting robots.txt, rate limits)"),

        ("3. Extraction",
         "Pull useful text from HTML, PDFs, etc."),

        ("4. Language Identification",
         "Detect and separate languages"),

        ("5. Filtering",
         "Remove low-quality, spam, toxic content"),

        ("6. Deduplication",
         "Remove near-duplicate documents"),

        ("7. Privacy Redaction",
         "Remove PII (personal information)"),

        ("8. Sampling",
         "Ensure diversity, balance categories")
    ]

    for step, desc in steps:
        print(f"\n{step}:")
        print(f"  {desc}")

collection_pipeline()
```

### Code: Simple Web Scraper

```python

import requests
from bs4 import BeautifulSoup
import time

def simple_scraper_demo():
    """
    Simple web scraping example
    """
    print("Simple Web Scraper Example")
    print("=" * 60)

    print("""
    import requests
    from bs4 import BeautifulSoup

    def scrape_article(url):
        # Respect robots.txt and rate limits
        time.sleep(1)  # Be polite

        # Fetch page
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract main content (simplified)
        article = soup.find('article')
        if article:
            text = article.get_text()
            return text

        return None

    # This would collect data for training!
    """)

    print("\n⚠️ Always check robots.txt and terms of service!")
    print("⚠️ Respect rate limits to avoid being blocked.")

simple_scraper_demo()
```

---

## Data Diversity: Why It Matters

### The Importance of Variety

```python

def diversity():
    """
    Why diverse data matters
    """
    print("Data Diversity: Teaching the Full Spectrum")
    print("=" * 60)

    print("""
    A model trained only on:

    📚 BOOKS → Formal, literary language (sounds like a professor)
    💬 REDDIT → Informal, conversational (sounds like a friend)
    📰 NEWS → Factual, journalistic (sounds like a reporter)
    💻 GITHUB → Code, technical (sounds like a programmer)

    REAL LLMs NEED ALL OF THESE:

    • Wikipedia: Facts and formal knowledge
    • Books: Narrative, long-form reasoning
    • Web: Diverse opinions, current language
    • Code: Logical structure, programming
    • Social media: Conversational patterns
    """)

diversity()
```

### Domain Coverage

```python

def domain_coverage():
    """
    Ensuring coverage of different domains
    """
    print("Domain Coverage: No Blind Spots")
    print("=" * 60)

    domains = [
        "Science (physics, chemistry, biology)",
        "History (ancient, modern, regional)",
        "Arts (literature, music, painting)",
        "Technology (computers, engineering)",
        "Medicine (health, diseases, treatments)",
        "Law (legal documents, cases)",
        "Business (finance, economics)",
        "Sports (rules, history, teams)",
        "Entertainment (movies, games, celebrities)"
    ]

    print("A well-trained model needs ALL these domains:")
    for domain in domains:
        print(f"  • {domain}")

    print("\nMissing a domain = model is blind in that area!")

domain_coverage()
```

---

## Ethical Considerations

### Privacy and Consent

```python

def privacy():
    """
    Privacy concerns in data collection
    """
    print("Privacy: The Elephant in the Room")
    print("=" * 60)

    issues = [
        "PERSONAL INFORMATION: Names, addresses, emails in training data",
        "PRIVATE CONVERSATIONS: Were people's chats used without consent?",
        "COPYRIGHTED MATERIAL: Books, articles used without permission",
        "BIAS: Data reflects societal biases",
        "DARK PATTERNS: Model learns harmful behaviors from bad data"
    ]

    print("Major ethical concerns:")
    for issue in issues:
        print(f"  • {issue}")

    print("\nSolutions being explored:")
    print("  • Opt-out mechanisms")
    print("  • PII redaction")
    print("  • Synthetic data generation")
    print("  • Better filtering")

privacy()
```

### The Data Paradox

```python

def data_paradox():
    """
    The paradox of data collection
    """
    print("The Data Paradox")
    print("=" * 60)

    print("""
    WE WANT:                        BUT:

    Current information      ←→   May violate copyright
    Diverse perspectives     ←→   May include harmful views
    Natural conversations    ←→   May contain private data
    Large scale              ←→   Harder to filter
    Free access              ←→   Data owners want payment

    This is why data collection is as much
    about ethics and law as it is about technology!
    """)

data_paradox()
```

---

## Real-World Examples

### GPT-3's Data Mix

```python

def gpt3_data():
    """
    What went into GPT-3
    """
    print("GPT-3's Training Data")
    print("=" * 60)

    data_mix = {
        "Common Crawl (filtered)": "410B tokens (60%)",
        "WebText2": "19B tokens (22%)",
        "Books1": "12B tokens (8%)",
        "Books2": "55B tokens (8%)",
        "Wikipedia": "3B tokens (3%)"
    }

    total = 0
    for source, amount in data_mix.items():
        print(f"  • {source}: {amount}")

    print("\nTotal: ~500B tokens")
    print("After filtering: ~300B tokens used")

gpt3_data()
```

### Llama 3's Approach

```python

def llama3_data():
    """
    Llama 3's data strategy
    """
    print("Llama 3: Quality Over Quantity")
    print("=" * 60)

    print("""
    Llama 3 took a different approach:

    • 15 TRILLION tokens (huge!)
    • Heavy filtering for quality
    • 10× more data than Llama 2
    • Focus on multilingual data
    • Code and reasoning data emphasized

    Key insight: More data + better filtering
    = better performance at same model size
    """)

llama3_data()
```

---

## Data Collection Cheat Sheet

| Data Source  | Quality | Quantity | Cost      | Best For         |
| ------------ | ------- | -------- | --------- | ---------------- |
| Common Crawl | Low     | Massive  | Free      | Base training    |
| Wikipedia    | High    | Medium   | Free      | Facts, structure |
| Books        | High    | Large    | Expensive | Narrative, depth |
| GitHub       | Medium  | Large    | Free      | Code, logic      |
| News         | High    | Medium   | Varies    | Current events   |
| Social media | Low     | Massive  | Varies    | Conversations    |

---

## Why This Matters (The Callout Box)

```text

╔══════════════════════════════════════════════════════════════╗
║                   WHY THIS MATTERS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Data collection isn't just "getting text"—it's:            ║
║                                                              ║
║  🧠 DETERMINING WHAT THE MODEL KNOWS                         ║
║     If it's not in the data, the model can't learn it.      ║
║     Want your model to know about recent events?            ║
║     Need recent data in the training mix.                   ║
║                                                              ║
║  🎭 SHAPING THE MODEL'S PERSONALITY                          ║
║     Books make it formal, Reddit makes it casual,           ║
║     GitHub makes it good at code. You are what you eat.     ║
║                                                              ║
║  ⚖️ INTRODUCING (OR REMOVING) BIAS                          ║
║     Data reflects society—including its biases.             ║
║     Careful collection can mitigate harmful patterns.       ║
║                                                              ║
║  📜 RAISING LEGAL AND ETHICAL QUESTIONS                      ║
║     Is it okay to train on copyrighted books?               ║
║     What about private conversations?                        ║
║                                                              ║
║  The quality of your data is the ceiling on                 ║
║  your model's performance. Better data = better AI.         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Quick Recap

• Data collection gathers the raw text that will become the model's knowledge—like a chef stocking a pantry, the ingredients determine what dishes can be made

• Sources range from high-quality (Wikipedia, books) to messy (web crawls)—each contributes different strengths, and filtering is essential to remove low-quality content

• Ethical considerations are huge—privacy, copyright, bias, and consent are ongoing challenges that make data collection as much a legal and moral endeavor as a technical one

---

## Mental Hook

> "Data collection is like raising a child—everything they read, watch, and experience shapes who they become. Garbage in, garbage out. Quality in, quality out. Your model is the sum of its training data."
