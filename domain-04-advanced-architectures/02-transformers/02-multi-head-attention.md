# Multi-Head Attention

## The Panel of Experts Analogy

Imagine you're trying to understand a complex document. Instead of asking one person, you assemble a panel of experts: a grammar expert, a subject-matter expert, a sentiment analyst, and a fact-checker. Each reads the document and gives you their perspective. Then you combine all their insights to form a complete understanding. That's multi-head attention: multiple attention mechanisms running in parallel, each learning to focus on different types of relationships.

In modern LLMs, multi-head attention is what makes transformers so powerful. Instead of a single attention pattern, models like GPT use multiple "heads" that specialize—one might focus on subject-verb relationships, another on adjective-noun pairs, another on coreference resolution. Together, they build a rich, multi-faceted understanding of language.

---

## What Is Multi-Head Attention?

### The Core Idea

Multi-head attention runs multiple attention mechanisms in parallel, each with its own set of Query, Key, Value transformations.

```text

Single attention head:              Multi-head attention:
    One perspective                    Multiple perspectives

    Q K V                              Head 1: Q₁ K₁ V₁
       ↓                                    ↓
    Attention                            Attention
       ↓                                    ↓
    Output₁                              Head 2: Q₂ K₂ V₂
                                              ↓
                                           Attention
                                              ↓
                                         Head 3: Q₃ K₃ V₃
                                              ↓
                                           Attention
                                              ↓
                                    Concatenate + Project
                                              ↓
                                         Final Output
```

```python

def multi_head_intro():
    """
    The basic concept of multi-head attention
    """
    print("Multi-Head Attention: Multiple Perspectives")
    print("=" * 60)

    print("""
    Instead of one attention mechanism:

    Single head: "The cat sat" → one relationship pattern

    Multi-head (8 heads):
    Head 1: Subject-verb relationships
    Head 2: Adjective-noun relationships
    Head 3: Coreference (it → cat)
    Head 4: Position/distance
    Head 5: Semantic similarity
    Head 6: Syntactic structure
    Head 7: Sentiment
    Head 8: Discourse connections

    Then combine all perspectives!
    """)

    print("\nEach head learns something different about language.")

multi_head_intro()
```

---

## How Multi-Head Attention Works

### Step 1: Project into Multiple Subspaces

```python

def multi_head_step1():
    """
    Step 1: Create multiple Q,K,V sets
    """
    print("Step 1: Projecting into Multiple Subspaces")
    print("=" * 60)

    # Input: 3 words, each with 4-dim embedding
    embeddings = {
        "The": [1.0, 0.0, 1.0, 0.0],
        "cat": [0.0, 1.0, 1.0, 1.0],
        "sat": [1.0, 1.0, 0.0, 1.0]
    }

    n_heads = 3
    d_model = 4
    d_head = d_model // n_heads  # 4/3 ≈ 1.33, but we'll simplify

    print(f"Input dimension (d_model): {d_model}")
    print(f"Number of heads: {n_heads}")
    print(f"Dimension per head (d_k): {d_model // n_heads} (simplified)")

    print("\nFor each head, we have separate weight matrices:")
    for head in range(1, n_heads+1):
        print(f"\n  Head {head}:")
        print(f"    W_Q{head}: projects input to Query subspace {head}")
        print(f"    W_K{head}: projects input to Key subspace {head}")
        print(f"    W_V{head}: projects input to Value subspace {head}")

    print("\nEach word gets different Q,K,V for each head!")

multi_head_step1()
```

### Step 2: Compute Attention per Head

```python

def multi_head_step2():
    """
    Step 2: Each head computes attention independently
    """
    print("Step 2: Parallel Attention Computation")
    print("=" * 60)

    # Simplified representations for 3 heads
    heads = {
        "Head 1 (Syntax)": {
            "The": {"Q": 0.2, "K": 0.1, "V": 0.3},
            "cat": {"Q": 0.8, "K": 0.7, "V": 0.9},
            "sat": {"Q": 0.3, "K": 0.4, "V": 0.5}
        },
        "Head 2 (Semantics)": {
            "The": {"Q": 0.1, "K": 0.2, "V": 0.2},
            "cat": {"Q": 0.9, "K": 0.8, "V": 0.8},
            "sat": {"Q": 0.4, "K": 0.3, "V": 0.4}
        },
        "Head 3 (Position)": {
            "The": {"Q": 0.5, "K": 0.5, "V": 0.1},
            "cat": {"Q": 0.5, "K": 0.5, "V": 0.7},
            "sat": {"Q": 0.5, "K": 0.5, "V": 0.6}
        }
    }

    print("Each head computes attention independently:")

    for head_name, word_data in heads.items():
        print(f"\n{head_name}:")
        # For "cat" looking at all words
        q_cat = word_data["cat"]["Q"]
        print(f"  cat's Q = {q_cat}")

        for target in ["The", "cat", "sat"]:
            k = word_data[target]["K"]
            v = word_data[target]["V"]
            score = q_cat * k
            print(f"    Attention to '{target}': score={score:.2f}, value={v}")

multi_head_step2()
```

### Step 3: Concatenate and Project

```python

def multi_head_step3():
    """
    Step 3: Combine all heads
    """
    print("Step 3: Concatenation and Final Projection")
    print("=" * 60)

    # Outputs from 3 heads for the word "cat" (simplified)
    head_outputs = {
        "Head 1": [0.7, 0.2],
        "Head 2": [0.3, 0.8],
        "Head 3": [0.5, 0.5]
    }

    print("Outputs from each head for 'cat':")
    for head, output in head_outputs.items():
        print(f"  {head}: {output}")

    # Concatenate
    concatenated = []
    for head, output in head_outputs.items():
        concatenated.extend(output)

    print(f"\nConcatenated: {concatenated}")
    print(f"Concatenated dimension: {len(concatenated)}")

    # Final projection matrix W_O (simplified)
    W_O = [
        [0.2, 0.1, 0.3, 0.1, 0.2, 0.1],
        [0.1, 0.3, 0.1, 0.2, 0.1, 0.2]
    ]

    # Project back to original dimension
    final = [0, 0]
    for i in range(len(concatenated)):
        final[0] += W_O[0][i] * concatenated[i]
        final[1] += W_O[1][i] * concatenated[i]

    print(f"\nFinal output after projection: [{final[0]:.2f}, {final[1]:.2f}]")
    print("This combines insights from all heads!")

multi_head_step3()
```

---

## The Complete Multi-Head Formula

```python

def multi_head_formula():
    """
    The mathematical formulation
    """
    print("Multi-Head Attention: The Formula")
    print("=" * 60)

    print("""
    MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O

    where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)

    Step by step:

    1. For each head i:
       • Project Q, K, V with learned matrices Wᵢ^Q, Wᵢ^K, Wᵢ^V
       • Compute attention: softmax(QᵢKᵢ^T/√d_k)Vᵢ

    2. Concatenate all head outputs

    3. Project concatenated result with W^O
    """)

    print("\nPlain English:")
    print("• Each head learns different projections")
    print("• Each computes attention in its own subspace")
    print("• Combine all views and project to final representation")

multi_head_formula()
```

---

## A Tiny Multi-Head Implementation

```python

import numpy as np

def multi_head_implementation():
    """
    Minimal multi-head attention implementation
    """
    print("Tiny Multi-Head Attention Implementation")
    print("=" * 60)

    class MultiHeadAttention:
        def __init__(self, d_model, n_heads):
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_head = d_model // n_heads

            # Create weight matrices for all heads
            self.W_Q = np.random.randn(n_heads, d_model, self.d_head) * 0.1
            self.W_K = np.random.randn(n_heads, d_model, self.d_head) * 0.1
            self.W_V = np.random.randn(n_heads, d_model, self.d_head) * 0.1
            self.W_O = np.random.randn(n_heads * self.d_head, d_model) * 0.1

        def attention(self, Q, K, V):
            """Single attention head"""
            scores = Q @ K.T / np.sqrt(self.d_head)
            weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            weights = weights / np.sum(weights, axis=-1, keepdims=True)
            return weights @ V

        def forward(self, x):
            seq_len = x.shape[0]
            head_outputs = []

            print(f"Input shape: {x.shape} ({seq_len} words, {self.d_model}-dim)")
            print(f"Number of heads: {self.n_heads}")
            print(f"Dimension per head: {self.d_head}")

            # Process each head
            for h in range(self.n_heads):
                # Project Q, K, V for this head
                Q = x @ self.W_Q[h]  # (seq_len, d_head)
                K = x @ self.W_K[h]
                V = x @ self.W_V[h]

                # Compute attention
                head_out = self.attention(Q, K, V)
                head_outputs.append(head_out)

                print(f"\nHead {h+1} output shape: {head_out.shape}")

            # Concatenate all heads
            concat = np.concatenate(head_outputs, axis=-1)
            print(f"\nConcatenated shape: {concat.shape}")

            # Final projection
            output = concat @ self.W_O
            print(f"Final output shape: {output.shape}")

            return output

    # Create multi-head attention
    mha = MultiHeadAttention(d_model=8, n_heads=4)

    # Example input: 3 words, 8-dim embeddings
    x = np.random.randn(3, 8)

    # Forward pass
    output = mha.forward(x)

multi_head_implementation()
```

---

## What Different Heads Learn

### Specialization in Practice

```python

def head_specialization():
    """
    What different attention heads learn
    """
    print("What Different Heads Learn")
    print("=" * 60)

    print("""
    Research has found that different heads specialize:

    Syntactic heads:
    • Head 3: Subject-verb agreement
    • Head 5: Determiner-noun relationships (the cat)
    • Head 7: Prepositional phrase attachments

    Semantic heads:
    • Head 1: Coreference (it → cat)
    • Head 4: Synonym relationships
    • Head 6: Hypernym/hyponym (animal → cat)

    Positional heads:
    • Head 2: Nearby words
    • Head 8: Long-range dependencies

    Some heads are redundant—that's okay!
    """)

    print("\nThis specialization emerges naturally from training!")
    print("No one tells heads what to learn.")

head_specialization()
```

### Visualizing Different Heads

```text

Sentence: "The cat sat on the mat because it was comfortable"

Head 1 (Coreference):
    "it" → "cat" (0.7), "mat" (0.2), "comfortable" (0.1)

Head 3 (Subject-verb):
    "cat" → "sat" (0.8)
    "sat" → "cat" (0.7)

Head 5 (Determiner):
    "The" → "cat" (0.9)
    "the" → "mat" (0.8)

Head 7 (Preposition):
    "on" → "mat" (0.8)
    "on" → "sat" (0.6)
```

Each head shows a different pattern!

---

## Why Multi-Head Is Better Than Single Head

### Comparison Table

| Aspect               | Single Head               | Multi-Head                    |
| -------------------- | ------------------------- | ----------------------------- |
| Perspectives         | One view of relationships | Multiple views                |
| Specialization       | Must do everything        | Heads can specialize          |
| Redundancy           | Single point of failure   | Robust (heads can compensate) |
| Representation power | Limited subspace          | Richer combined space         |
| Interpretability     | One pattern to analyze    | Multiple patterns to study    |

### The Ensemble Effect

```python

def ensemble_effect():
    """
    Why multiple heads work better
    """
    print("The Ensemble Effect")
    print("=" * 60)

    print("""
    Single head: Must capture ALL relationships

    Subject-verb   ~~~~~> [ONE HEAD] ~~~~~> ?????
    Coreference    ~~~~~> [ONE HEAD] ~~~~~> Overwhelmed!
    Prepositions   ~~~~~> [ONE HEAD] ~~~~~>

    Multi-head: Divide and conquer

    Subject-verb   ~~~~~> [Head 1] ~~~~~> Specialized!
    Coreference    ~~~~~> [Head 2] ~~~~~> Specialized!
    Prepositions   ~~~~~> [Head 3] ~~~~~> Specialized!
    Position       ~~~~~> [Head 4] ~~~~~> Specialized!

    Then combine: Rich, multi-faceted understanding
    """)

    print("\nEach head becomes an expert in one type of relationship.")

ensemble_effect()
```

---

## Why This Matters for LLMs

### 1. All Modern LLMs Use Multi-Head Attention

```python

def llm_heads():
    """
    Multi-head attention in popular models
    """
    print("Multi-Head Attention in Popular LLMs")
    print("=" * 60)

    models = {
        "BERT Base": "12 layers, 12 heads (144 total attention heads)",
        "BERT Large": "24 layers, 16 heads (384 heads)",
        "GPT-3": "96 layers, 96 heads (9,216 heads!)",
        "Llama 3 70B": "80 layers, 64 heads (5,120 heads)",
        "T5": "12-24 layers, 12-16 heads per layer"
    }

    for model, desc in models.items():
        print(f"  • {model}: {desc}")

    print("\nEach head in each layer learns something different!")
    print("That's millions of learned attention patterns.")

llm_heads()
```

### 2. Head Pruning and Interpretability

```python

def head_pruning():
    """
    Some heads are more important than others
    """
    print("Head Pruning: Not All Heads Are Equal")
    print("=" * 60)

    print("""
    Research shows:

    • Some heads are crucial (remove them → performance drops)
    • Some heads are redundant (remove them → no change)
    • Some heads are even harmful (remove them → performance IMPROVES!)

    This has led to "head pruning" techniques:
    Remove unimportant heads to make models faster and smaller.
    """)

    print("\nExample: In BERT, removing 30% of heads sometimes")
    print("has little effect on performance!")

head_pruning()
```

### 3. Multi-Head vs. Multi-Layer

```python

def multi_head_vs_multi_layer():
    """
    Different dimensions of attention
    """
    print("Multi-Head vs. Multi-Layer")
    print("=" * 60)

    print("""
    Multi-head: Different perspectives WITHIN the same layer
                (parallel specialization)

    Multi-layer: Progressive refinement ACROSS layers
                 (hierarchical processing)

    Layer 1 heads: Basic syntax, nearby words
    Layer 2 heads: Phrase structure
    Layer 3 heads: Semantic roles
    ...
    Layer 12 heads: Abstract concepts, discourse

    They work together!
    """)

multi_head_vs_multi_layer()
```

### 4. Computational Considerations

```python

def computational_aspects():
    """
    Efficiency of multi-head attention
    """
    print("Computational Aspects of Multi-Head Attention")
    print("=" * 60)

    d_model = 768  # BERT size
    n_heads_options = [1, 4, 8, 12, 16, 32]

    print(f"Model dimension (d_model): {d_model}")
    print("\nNumber of heads | Dim per head | Parameter count")
    print("-" * 50)

    for n_heads in n_heads_options:
        d_head = d_model // n_heads
        params = 4 * d_model * d_model  # Rough estimate for Q,K,V,O
        print(f"       {n_heads:2d}        |     {d_head:3d}      |   {params:,}")

computational_aspects()
```

---

## Multi-Head Attention Variants

| Variant                 | Description                     | Used In          |
| ----------------------- | ------------------------------- | ---------------- |
| Multi-Query Attention   | Share keys/values across heads  | Faster inference |
| Grouped-Query Attention | Groups of heads share           | Llama 2, 3       |
| Flash Attention         | Memory-efficient implementation | Modern training  |
| Sparse Multi-Head       | Only attend to local windows    | Long-form models |

---

## Quick Recap

• Multi-head attention runs multiple attention mechanisms in parallel—like assembling a panel of experts where each specializes in different aspects of language (syntax, semantics, coreference)

• Each head learns different projections and can focus on different relationships—they divide the work of understanding language, then their insights are combined into a rich, multi-faceted representation

• All modern LLMs use multi-head attention with dozens of heads per layer—from BERT's 12 heads to GPT-3's 96 heads per layer, this specialization is key to their powerful understanding of language

---

## Mental Hook

> "Multi-head attention is like having a team of specialists read a document simultaneously—the grammar expert, the fact-checker, the sentiment analyst, and the summarizer all work in parallel, then combine their insights into a complete understanding that no single expert could achieve alone."
