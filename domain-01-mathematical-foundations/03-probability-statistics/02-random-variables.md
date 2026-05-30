# Random variables

**DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Probability & Statistics**

---

**1. Why this concept matters for building intelligent systems**

Probability tells you how likely events are. But intelligent systems need more than event probabilities—they need to work with quantities that vary unpredictably: tomorrow's stock price, the next word in a sentence, the brightness of a pixel. These quantities are random variables. A random variable is a bridge between abstract probability theory and concrete numbers. It lets you compute averages, measure spread, and model relationships between uncertain quantities. Every prediction your AI makes is a random variable. Every evaluation metric (accuracy, loss) is a function of random variables.

---

**2. Core idea**

**A random variable is a function that assigns a numerical value to each outcome of a random process, transforming events into numbers that can be averaged, compared, and modeled mathematically.**

---

**3. Concrete analogy**

Imagine you run a carnival booth with a spinning wheel divided into 8 equal slices. The wheel is fair—each slice has probability 1/8 of stopping under the pointer.

The outcome of the spin is not a number yet. It could be "red," "blue," "green," etc. A random variable turns this outcome into a number:

- Let X = the cash prize you win: $0 for red, $5 for blue, $10 for green, $1 for others.
- Let Y = the angle of the pointer (0° to 360°).

X is a discrete random variable (takes specific values: 0, 1, 5, 10). Y is a continuous random variable (takes any value in an interval).

Now you can ask: What is the average prize? That is E[X] (expected value). How much do prizes vary? That is Var(X) (variance). What is the probability of winning more than $5? That is P(X > 5). The random variable turns the messy physical wheel into clean mathematics.

---

**4. ASCII diagram**

```
Random variable as a function from sample space to numbers:

    Sample Space Ω          Random Variable X          Real Numbers ℝ
    (all outcomes)               (mapping)               (values)

        ω₁ ──────────────────→ X(ω₁) = 0

        ω₂ ──────────────────→ X(ω₂) = 5

        ω₃ ──────────────────→ X(ω₃) = 10

        ω₄ ──────────────────→ X(ω₄) = 1

        ω₅ ──────────────────→ X(ω₅) = 0

        ω₆ ──────────────────→ X(ω₆) = 1

        ω₇ ──────────────────→ X(ω₇) = 5

        ω₈ ──────────────────→ X(ω₈) = 0

    Discrete random variable (finite or countable values):

        Probability mass function (PMF):

        P(X=0) = 3/8 = 0.375
        P(X=1) = 2/8 = 0.250
        P(X=5) = 2/8 = 0.250
        P(X=10)= 1/8 = 0.125

    Continuous random variable (infinite uncountable values):

        Probability density function (PDF) - area = probability

        f(x) ↑
              │    ┌─────┐
              │    │     │
              │    │     │
              │    │     │
              └────┴─────┴────→ x
              Area under curve between a and b = P(a ≤ X ≤ b)
```

---

**5. Mathematical formulation**

**Definition:** A random variable X is a function X: Ω → ℝ, where Ω is the sample space.

**Discrete random variable:** Takes values in a countable set {x₁, x₂, x₃, ...}

Probability mass function (PMF):

$$
p_X(x) = P(X = x)
$$

Properties:

- p_X(x) ≥ 0 for all x
- Σ_x p_X(x) = 1

**Continuous random variable:** Takes values in an uncountable set (interval of ℝ)

Probability density function (PDF): f_X(x)

Properties:

- f_X(x) ≥ 0 for all x
- ∫\_{-∞}^{∞} f_X(x) dx = 1
- P(a ≤ X ≤ b) = ∫_a^b f_X(x) dx

**Cumulative distribution function (CDF):** (works for both discrete and continuous)

$$
F_X(x) = P(X \leq x)
$$

(Unicode: F_X(x) = P(X ≤ x))

Properties:

- F_X(-∞) = 0, F_X(∞) = 1
- F_X is non-decreasing
- For continuous: f_X(x) = d/dx F_X(x)

**Expected value (mean):**

Discrete: E[X] = Σ_x x × p_X(x)

Continuous: E[X] = ∫\_{-∞}^{∞} x × f_X(x) dx

(Unicode: E[X] = Σ x·p(x) or ∫ x·f(x) dx)

**Variance (spread):**

$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$

(Unicode: Var(X) = E[(X-μ)²] = E[X²] - μ²)

**Standard deviation:** σ_X = √Var(X)

---

**6. Worked example (step-by-step)**

**Step 1: Define the random variable**

Roll two fair six-sided dice. Let X = the sum of the two dice.

**Step 2: Determine possible values and probabilities**

Possible sums: 2 through 12

Number of outcomes: 6×6 = 36

| Sum | Pairs                               | Count | P(X=sum)      |
| --- | ----------------------------------- | ----- | ------------- |
| 2   | (1,1)                               | 1     | 1/36 ≈ 0.0278 |
| 3   | (1,2),(2,1)                         | 2     | 2/36 ≈ 0.0556 |
| 4   | (1,3),(2,2),(3,1)                   | 3     | 3/36 ≈ 0.0833 |
| 5   | (1,4),(2,3),(3,2),(4,1)             | 4     | 4/36 ≈ 0.1111 |
| 6   | (1,5),(2,4),(3,3),(4,2),(5,1)       | 5     | 5/36 ≈ 0.1389 |
| 7   | (1,6),(2,5),(3,4),(4,3),(5,2),(6,1) | 6     | 6/36 ≈ 0.1667 |
| 8   | (2,6),(3,5),(4,4),(5,3),(6,2)       | 5     | 5/36 ≈ 0.1389 |
| 9   | (3,6),(4,5),(5,4),(6,3)             | 4     | 4/36 ≈ 0.1111 |
| 10  | (4,6),(5,5),(6,4)                   | 3     | 3/36 ≈ 0.0833 |
| 11  | (5,6),(6,5)                         | 2     | 2/36 ≈ 0.0556 |
| 12  | (6,6)                               | 1     | 1/36 ≈ 0.0278 |

**Step 3: Compute expected value**

E[X] = Σ [sum × P(sum)]

= 2×(1/36) + 3×(2/36) + 4×(3/36) + 5×(4/36) + 6×(5/36) + 7×(6/36) + 8×(5/36) + 9×(4/36) + 10×(3/36) + 11×(2/36) + 12×(1/36)

= (2 + 6 + 12 + 20 + 30 + 42 + 40 + 36 + 30 + 22 + 12) / 36

= 252 / 36 = 7

Intuition: The average sum of two dice is 7.

**Step 4: Compute variance**

E[X²] = Σ [sum² × P(sum)]

= 4×1/36 + 9×2/36 + 16×3/36 + 25×4/36 + 36×5/36 + 49×6/36 + 64×5/36 + 81×4/36 + 100×3/36 + 121×2/36 + 144×1/36

= (4 + 18 + 48 + 100 + 180 + 294 + 320 + 324 + 300 + 242 + 144) / 36

= 1974 / 36 ≈ 54.833

Var(X) = E[X²] - (E[X])² = 54.833 - 49 = 5.833

σ_X = √5.833 ≈ 2.415

Typical sum is 7 ± about 2.4.

---

**7. How this appears inside neural networks and LLMs**

- **Input embeddings**: Each word is mapped to a vector—but from the model's perspective, the next word is a random variable with a probability distribution over the vocabulary.

- **Output distribution**: The final softmax layer produces P(word_i | context). Sampling from this random variable generates text.

- **Dropout**: During training, each neuron's output is multiplied by a Bernoulli random variable (0 or 1) with probability p. This regularizes the model.

- **Stochastic gradient descent**: The gradient computed on a random mini-batch is a random variable whose expectation equals the true gradient.

- **Latent variable models**: Variational autoencoders assume observed data is generated from unobserved latent random variables.

- **Attention as expectation**: Attention output is a weighted average of value vectors, where weights are probabilities. This is E[value | attention distribution].

- **Uncertainty quantification**: Bayesian neural networks treat weights as random variables, producing prediction distributions, not point estimates.

- **Reinforcement learning**: Returns, values, and advantages are all expectations of random variables.

---

**8. Brain-like connection (neural variability)**

Individual neurons do not fire deterministically. In response to the same stimulus, a neuron's spike count varies randomly. This variability is not just noise—it may be a computational feature. The brain represents uncertainty through the variability of neural firing rates. A neuron's firing rate can be interpreted as the mean of an underlying Poisson random variable. Populations of neurons encode probability distributions over stimuli, actions, and rewards. Your brain's ability to make decisions under uncertainty relies on representing and manipulating random variables at the neural level.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "A random variable is a variable whose value is completely unpredictable—like a coin flip before you look."

_Why it is wrong:_ A random variable has a precise probability distribution. You may not know the outcome of a specific trial, but you know exactly the long-run behavior. For a fair coin, P(Heads)=0.5 is not uncertainty—it is certainty about the probability. The randomness is in the outcome, not in your knowledge of the distribution. Confusing "random" with "unknown" leads to mistakes like treating a parameter as random when it is fixed but unknown (the Bayesian/frequentist distinction). In machine learning, the training data is a random variable; the learned weights are not (they are fixed given the data).

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  The world gives you numbers that vary: tomorrow's        |
|  temperature, the next word in a sentence, the error      |
|  on a test example. Random variables are how you model    |
|  this variation. They let you compute averages, measure   |
|  spread, and connect probability to the real numbers      |
|  your AI actually processes. Without random variables,    |
|  you cannot predict, only describe what already happened. |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

A bag contains 3 red marbles and 2 blue marbles. You draw two marbles without replacement. Define random variable X = number of red marbles drawn.

**Question:** What are the possible values of X? Compute P(X=1) and E[X].

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Possible values: X = 0, 1, 2

P(X=1) = (number of ways to draw 1 red and 1 blue) / total ways

Ways to choose 1 red from 3: C(3,1)=3  
Ways to choose 1 blue from 2: C(2,1)=2  
Total ordered draws: 5×4=20 (or combinations: C(5,2)=10)

P(X=1) = (3×2×2)/20 = 12/20 = 0.6 (if ordered)  
Or (3×2)/10 = 6/10 = 0.6 (same)

E[X] = 0×P(0) + 1×P(1) + 2×P(2)  
P(0) = (2×1)/(5×4)=2/20=0.1  
P(2) = (3×2)/(5×4)=6/20=0.3  
E[X] = 0×0.1 + 1×0.6 + 2×0.3 = 0 + 0.6 + 0.6 = 1.2

(The expected number of red marbles when drawing 2 from 3 red, 2 blue is 1.2)
