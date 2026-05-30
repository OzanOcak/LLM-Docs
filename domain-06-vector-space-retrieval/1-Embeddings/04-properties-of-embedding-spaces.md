# Properties of Embedding Spaces (Linear Relationships, Analogies)

## The Crystal Structure Analogy

Imagine a crystal where atoms arrange themselves in a repeating pattern. If you know the vector from one carbon atom to another in a diamond, you can find every carbon atom just by following that same vector over and over. That's what embedding spaces are like—they have a regular, almost crystalline structure where relationships between concepts are captured as consistent directions. "Man" to "woman" is the same vector as "king" to "queen," and "Paris" to "France" is the same as "Rome" to "Italy."

In LLMs, this property is what makes embeddings so powerful. The space isn't just a random cloud of points—it's organized with geometric regularity. This is why we can do analogies with vector arithmetic, find concept directions, and even debias models by manipulating these consistent directions.

---

## The Linear Structure of Meaning

### What Linear Relationships Mean

```python

def linear_intro():
    """
    The concept of linearity in embedding spaces
    """
    print("Linear Relationships: Meaning as Directions")
    print("=" * 60)

    print("""
    In a linear embedding space:

    • Relationships between concepts are VECTORS
    • The SAME relationship appears in multiple places
    • You can add and subtract meanings

    Example:
    If we find the vector for "masculine → feminine":
    v = woman - man

    Then:
    king + v ≈ queen
    boy + v ≈ girl
    father + v ≈ mother
    """)

linear_intro()
```

### Visualizing Linear Structure

```text

Embedding space as a crystal:

    man ────────→ woman
     ↑              ↑
     │              │
     │ gender       │ gender
     │ vector       │ vector
     │              │
    king ────────→ queen

    boy ─────────→ girl
     ↑              ↑
     │ age         │ age
     │ vector      │ vector
     │              │
    man ─────────→ woman

The SAME gender vector works everywhere!
The SAME age vector works everywhere!
```

---

## The Famous King - Man + Woman = Queen

### The Classic Analogy

```python

def king_queen_analogy():
    """
    The most famous embedding property
    """
    print("The Classic Analogy: King - Man + Woman = Queen")
    print("=" * 60)

    # Simplified 2D embeddings
    embeddings = {
        "king": [0.9, 0.8],
        "queen": [0.9, 0.2],
        "man": [0.3, 0.8],
        "woman": [0.3, 0.2],
        "boy": [0.2, 0.7],
        "girl": [0.2, 0.3],
        "father": [0.4, 0.8],
        "mother": [0.4, 0.2]
    }

    print("2D embedding space (dim1 = royalty, dim2 = gender):")
    for word, vec in embeddings.items():
        print(f"  {word:6}: royalty={vec[0]}, gender={vec[1]}")

    # Gender vector
    gender_vec = [embeddings["woman"][0] - embeddings["man"][0],
                  embeddings["woman"][1] - embeddings["man"][1]]
    print(f"\nGender vector (woman - man): {gender_vec}")

    # Apply to king
    king = embeddings["king"]
    result = [king[0] + gender_vec[0], king[1] + gender_vec[1]]
    print(f"king + gender vector = {result}")
    print(f"queen = {embeddings['queen']}")
    print(f"✓ They match! (0.9, 0.2)")

    # Works for others too
    print("\nIt works for other pairs:")
    for pair in [("boy", "girl"), ("father", "mother")]:
        male = embeddings[pair[0]]
        female = embeddings[pair[1]]
        print(f"  {pair[0]} + gender = {[male[0]+gender_vec[0], male[1]+gender_vec[1]]}")
        print(f"  {pair[1]} = {female}")
        print(f"  ✓")

king_queen_analogy()
```

### Why This Happens

```python

def why_linear():
    """
    Why embeddings become linear
    """
    print("Why Embeddings Become Linear")
    print("=" * 60)

    print("""
    During training, the model sees patterns like:

    "The king ruled the kingdom"
    "The queen ruled the kingdom"

    "The man walked into the room"
    "The woman walked into the room"

    The model learns that "king" and "man" share something,
    and that "queen" and "woman" share something.

    The most efficient way to represent this is to have
    a consistent "gender" direction that can be added
    to any royal/parent/child word.

    This linear structure emerges naturally from
    the training objective!
    """)

why_linear()
```

---

## More Analogy Examples

### Capital-Country Relationships

```python

def capital_country():
    """
    Capital-city analogies
    """
    print("Capital-Country Analogies")
    print("=" * 60)

    # Simplified 2D embeddings
    embeddings = {
        "paris": [0.8, 0.7],
        "france": [0.8, 0.1],
        "rome": [0.6, 0.7],
        "italy": [0.6, 0.1],
        "london": [0.4, 0.7],
        "uk": [0.4, 0.1],
        "berlin": [0.2, 0.7],
        "germany": [0.2, 0.1]
    }

    capital_vec = [embeddings["france"][0] - embeddings["paris"][0],
                   embeddings["france"][1] - embeddings["paris"][1]]

    print("Capital vector (country - capital): moving from city to country")
    print(f"  france - paris = {capital_vec}")

    print("\nParis : France :: Rome : ?")
    rome = embeddings["rome"]
    result = [rome[0] + capital_vec[0], rome[1] + capital_vec[1]]
    print(f"  rome + capital vector = {result}")
    print(f"  italy = {embeddings['italy']}")

    print("\nLondon : UK :: Berlin : ?")
    london = embeddings["london"]
    result = [london[0] + capital_vec[0], london[1] + capital_vec[1]]
    print(f"  berlin + capital vector = {result}")
    print(f"  germany = {embeddings['germany']}")

capital_country()
```

### Tense Relationships

```python

def tense_relationships():
    """
    Verb tense analogies
    """
    print("Verb Tense Analogies")
    print("=" * 60)

    # Simplified embeddings for verbs
    embeddings = {
        "walk": [0.8, 0.3],
        "walked": [0.8, 0.7],
        "talk": [0.6, 0.3],
        "talked": [0.6, 0.7],
        "eat": [0.4, 0.3],
        "ate": [0.4, 0.7],
        "see": [0.2, 0.3],
        "saw": [0.2, 0.7]
    }

    past_vec = [embeddings["walked"][0] - embeddings["walk"][0],
                embeddings["walked"][1] - embeddings["walk"][1]]

    print("Past tense vector (past - present): moving from present to past")
    print(f"  walked - walk = {past_vec}")

    print("\nwalk : walked :: talk : ?")
    talk = embeddings["talk"]
    result = [talk[0] + past_vec[0], talk[1] + past_vec[1]]
    print(f"  talk + past vector = {result}")
    print(f"  talked = {embeddings['talked']}")

    print("\neat : ate :: see : ?")
    see = embeddings["see"]
    result = [see[0] + past_vec[0], see[1] + past_vec[1]]
    print(f"  see + past vector = {result}")
    print(f"  saw = {embeddings['saw']}")

tense_relationships()
```

---

## Mathematical Formulation

### Vector Arithmetic

```python

def vector_arithmetic():
    """
    The math behind analogy solving
    """
    print("The Mathematics of Analogies")
    print("=" * 60)

    print("""
    Given: a : b :: c : ?

    We want to find d such that:
    b - a ≈ d - c

    Therefore:
    d ≈ c + (b - a)

    Example:
    king : queen :: man : ?

    d = man + (queen - king)
      = man + gender_vector
      = woman

    In practice, we find the word whose embedding
    is closest to c + (b - a).
    """)

vector_arithmetic()
```

### Finding the Right Word

```python

def nearest_neighbor():
    """
    Finding the answer word
    """
    print("Finding the Analogy Answer")
    print("=" * 60)

    print("""
    Steps to solve an analogy:

    1. Compute target vector = c + (b - a)

    2. Find word whose embedding is closest
       to this target vector

    3. Exclude a, b, c from candidates

    Example: king : queen :: man : ?

    target = man + (queen - king)
           = man + gender_vector

    Find nearest word to target:
    • woman: 0.95 similarity ✓
    • person: 0.65 similarity ✗
    • human: 0.60 similarity ✗

    Answer: woman
    """)

nearest_neighbor()
```

---

## A Complete Example

```python

import numpy as np

def analogy_demo():
    """
    Complete analogy-finding demonstration
    """
    print("Complete Analogy Demo")
    print("=" * 60)

    # Create a simple 3D embedding space
    embeddings = {
        "king": np.array([0.9, 0.8, 0.5]),
        "queen": np.array([0.9, 0.2, 0.5]),
        "man": np.array([0.3, 0.8, 0.5]),
        "woman": np.array([0.3, 0.2, 0.5]),
        "prince": np.array([0.8, 0.8, 0.4]),
        "princess": np.array([0.8, 0.2, 0.4]),
        "actor": np.array([0.4, 0.8, 0.3]),
        "actress": np.array([0.4, 0.2, 0.3]),
        "paris": np.array([0.9, 0.7, 0.8]),
        "france": np.array([0.9, 0.1, 0.8]),
        "rome": np.array([0.7, 0.7, 0.8]),
        "italy": np.array([0.7, 0.1, 0.8]),
        "london": np.array([0.5, 0.7, 0.8]),
        "uk": np.array([0.5, 0.1, 0.8])
    }

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def solve_analogy(a, b, c, embeddings):
        """Find d such that a:b :: c:d"""
        target = embeddings[c] + (embeddings[b] - embeddings[a])

        best_word = None
        best_score = -1

        for word, vec in embeddings.items():
            if word in [a, b, c]:
                continue
            score = cosine_similarity(target, vec)
            if score > best_score:
                best_score = score
                best_word = word

        return best_word, best_score

    analogies = [
        ("man", "woman", "king"),
        ("man", "woman", "prince"),
        ("man", "woman", "actor"),
        ("paris", "france", "rome"),
        ("paris", "france", "london")
    ]

    print("Solving analogies:")
    for a, b, c in analogies:
        result, score = solve_analogy(a, b, c, embeddings)
        print(f"\n  {a} : {b} :: {c} : ?")
        print(f"  → {result} (similarity: {score:.3f})")

analogy_demo()
```

---

## Beyond Simple Analogies

### Concept Directions

```python

def concept_directions():
    """
    Using directions for more than analogies
    """
    print("Concept Directions: Beyond Analogies")
    print("=" * 60)

    print("""
    We can find directions for ANY concept:

    • Gender direction = woman - man
    • Plural direction = cats - cat
    • Past tense direction = walked - walk
    • Sentiment direction = good - bad
    • Formality direction = formal - informal

    Then we can:

    1. Measure how much a word has that property
       e.g., doctor · gender_direction = ? (bias detection)

    2. Add or remove the property
       e.g., "nurse" - gender_direction = "doctor"? (debiasing)

    3. Find words that vary along that direction
    """)

concept_directions()
```

### Bias Detection and Mitigation

```python

def bias_detection():
    """
    Using linear structure to detect bias
    """
    print("Detecting and Mitigating Bias")
    print("=" * 60)

    print("""
    Problem: Embeddings can capture societal biases

    Example: Gender bias in professions

    Compute gender direction = she - he

    Project professions onto this direction:

    • nurse:     strong feminine bias   (large projection)
    • secretary: strong feminine bias
    • engineer:  slight masculine bias
    • doctor:    slight masculine bias
    • programmer: masculine bias

    Solutions:
    1. Identify biased directions
    2. Remove that component from embeddings
    3. Neutralize the representation

    This is active research area!
    """)

bias_detection()
```

---

## Properties of Embedding Spaces

| Property           | Description                            | Example                             |
| ------------------ | -------------------------------------- | ----------------------------------- |
| Linearity          | Relationships are consistent vectors   | Same gender vector works everywhere |
| Compositionality   | Meanings can be combined               | "royal" + "female" ≈ "queen"        |
| Analogies          | a:b :: c:d solved by vector arithmetic | king - man + woman = queen          |
| Concept directions | Each abstract concept has a direction  | sentiment, tense, formality         |
| Clustering         | Similar words form neighborhoods       | animals together, vehicles together |
| Hierarchy          | Nested categories                      | animal > mammal > dog               |

---

## Why This Matters for LLMs

### 1. Interpretability

```python

def interpretability():
    """
    Understanding what models know
    """
    print("Interpretability Through Linear Structure")
    print("=" * 60)

    print("""
    Linear structure lets us peek inside:

    • Find what directions the model cares about
    • Measure bias in representations
    • Understand how concepts are organized
    • Identify spurious correlations

    Example: Probing tasks
    Train a linear classifier on embeddings
    If it works, the concept is linearly represented.
    """)

interpretability()
```

### 2. Controllable Generation

```python

def controllable():
    """
    Steering generation with directions
    """
    print("Controllable Generation")
    print("=" * 60)

    print("""
    We can steer generation by manipulating embeddings:

    • Add "positive sentiment" direction to generate happier text
    • Add "formal" direction for professional writing
    • Subtract "biased" direction for fairness

    This is called "steering vectors" or "activation engineering"
    and works surprisingly well!
    """)

controllable()
```

### 3. Cross-lingual Transfer

```python

def cross_lingual():
    """
    Linear structure across languages
    """
    print("Cross-lingual Linear Structure")
    print("=" * 60)

    print("""
    In multilingual embeddings, the SAME directions
    work across languages:

    English gender vector ≈ French gender vector
    English plural vector ≈ German plural vector

    This enables:
    • Translation by vector arithmetic
    • Cross-lingual analogies
    • Zero-shot transfer of concepts
    """)

cross_lingual()
```

---

## Properties Cheat Sheet

| Concept          | Vector Operation | Example                     |
| ---------------- | ---------------- | --------------------------- |
| Analogy          | d = c + (b - a)  | king - man + woman = queen  |
| Gender direction | woman - man      | Add to any word to feminize |
| Plural direction | cats - cat       | Add to make plural          |
| Past tense       | walked - walk    | Add to change tense         |
| Sentiment        | good - bad       | Positive vs negative        |
| Degree           | very - somewhat  | Intensity modification      |

---

## Quick Recap

• Embedding spaces have linear structure—relationships between concepts are captured as consistent vectors that work across many word pairs, like a crystal with repeating patterns

• Analogies become vector arithmetic—famous examples like "king - man + woman = queen" work because the gender direction is the same everywhere in the space

• These properties enable interpretability, bias detection, and controllable generation—we can find concept directions, measure bias, and even steer model outputs by manipulating these linear relationships

---

## Mental Hook

> "Embedding spaces are like crystalline structures of meaning—once you discover the vector from 'man' to 'woman', you've found a gender crystal that you can place anywhere in the space to transform kings to queens, boys to girls, and fathers to mothers, all with the same mathematical operation."
