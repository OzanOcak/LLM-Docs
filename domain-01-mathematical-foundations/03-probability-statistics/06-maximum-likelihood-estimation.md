# Maximum likelihood estimation

## **DOMAIN: MATHEMATICAL FOUNDATIONS | Sub domain: Probability & Statistics**

---

### **1. Why this concept matters for building intelligent systems**

You have data. You have a model. Which parameters make the model most consistent with the data? Maximum likelihood estimation (MLE) answers this question: choose the parameters that maximize the probability of observing the data you actually saw. This simple principle underlies almost all supervised learning. Linear regression, logistic regression, and even neural networks trained with cross-entropy loss are performing maximum likelihood estimation. Understanding MLE means understanding why we minimize negative log likelihood, why the loss function has that form, and how to derive new models from first principles.

---

### **2. Core idea**

**Maximum likelihood estimation finds the parameter values that make the observed data most probable, by maximizing the likelihood function over the parameter space.**

---

### **3. Concrete analogy**

Imagine you are a detective investigating a biased coin. You flip it 10 times and get 7 heads, 3 tails. You do not know the probability p of heads. What is your best guess?

The likelihood of p is the probability of observing exactly 7 heads given that p: L(p) = C(10,7) × p⁷ × (1-p)³.

Try p=0.5: L(0.5) = 120 × 0.5⁷ × 0.5³ = 120 × 0.5¹⁰ ≈ 0.117
Try p=0.7: L(0.7) = 120 × 0.7⁷ × 0.3³ ≈ 120 × 0.0824 × 0.027 ≈ 0.267
Try p=0.8: L(0.8) = 120 × 0.8⁷ × 0.2³ ≈ 120 × 0.210 × 0.008 ≈ 0.202

p=0.7 gives the highest likelihood. That is the maximum likelihood estimate. Intuitively, the best guess for the bias is the observed proportion (7/10 = 0.7), which is no accident—MLE often produces intuitive estimates.

---

### **4. ASCII diagram**

```text
Maximum likelihood estimation visualized:

    Likelihood function L(θ) for coin flip data (7 heads, 3 tails)

    L(θ) ↑
          │
      0.3 ┤     ●
          │    ╱ ╲
      0.2 ┤   ╱   ╲
          │  ╱     ╲
      0.1 ┤ ╱       ╲
          │╱         ╲
       0 ┼─────┬─────┬─────┬─────→ θ (p)
          0.5  0.6  0.7  0.8  0.9

    Maximum at θ = 0.7

    For a Gaussian (normal) distribution:

    f(x) ↑
          │   *   *   *
          │ *       *   *   *
          │*           *   *   *
          │               *   *   *
          └──────────────────────→ x

    MLE for mean μ = sample average (x̄)
    MLE for variance σ² = average squared deviation

    For multiple parameters, we maximize over θ space:

         L(θ)
           ↑
           │      .---.
           │     /     \
           │    /       \
           │   /    ●    \   ← Peak (MLE)
           │  /           \
           └──┴─────┴─────┴──→ θ₁
          ╱
         ╱
        ↓
       θ₂
```

---

### **5. Mathematical formulation**

**Likelihood function:** For independent and identically distributed data {x₁, ..., xₙ}:

$$
L(\theta) = \prod_{i=1}^n p(x_i | \theta)
$$

(Unicode: L(θ) = Π_i p(x_i|θ))

Where p(x_i|θ) is probability mass function (discrete) or probability density (continuous).

**Log-likelihood:** Easier to work with sums than products:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log p(x_i | \theta)
$$

(Unicode: ℓ(θ) = Σ_i log p(x_i|θ))

**Maximum likelihood estimator:**

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} L(\theta) = \arg\max_{\theta} \ell(\theta)
$$

(Unicode: θ̂ = argmax L(θ) = argmax ℓ(θ))

**Common MLE results:**

- Bernoulli (coin flip): p̂ = (number of successes)/n
- Gaussian (normal): μ̂ = (1/n) Σ x_i, σ̂² = (1/n) Σ (x_i - μ̂)²
- Poisson: λ̂ = (1/n) Σ x_i
- Exponential: λ̂ = 1 / ((1/n) Σ x_i)

**Connection to loss functions:**

Minimizing negative log likelihood is equivalent to maximizing likelihood:

$$
\hat{\theta}_{\text{MLE}} = \arg\min_{\theta} \left( -\sum_{i=1}^n \log p(x_i | \theta) \right)
$$

For Gaussian with fixed variance, this gives mean squared error. For Bernoulli, this gives cross-entropy.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Define the problem**

You measure waiting times at a bus stop: [3, 5, 2, 8, 4] minutes. You assume waiting times follow an exponential distribution with rate λ (mean = 1/λ). Find λ̂ (MLE).

#### **Step 2: Write the exponential PDF**

f(x|λ) = λ e^{-λx}, x ≥ 0

#### **Step 3: Write the likelihood**

L(λ) = ∏\_{i=1}^5 λ e^{-λ x_i} = λ⁵ e^{-λ Σ x_i}

#### **Step 4: Write the log-likelihood**

ℓ(λ) = 5 log λ - λ Σ x_i

Σ x_i = 3+5+2+8+4 = 22

ℓ(λ) = 5 log λ - 22λ

#### **Step 5: Take derivative and set to zero**

dℓ/dλ = 5/λ - 22 = 0

5/λ = 22

λ = 5/22 ≈ 0.2273

#### **Step 6: Verify it is a maximum**

Second derivative: d²ℓ/dλ² = -5/λ² < 0 (concave down, so maximum)

#### **Step 7: Interpret**

λ̂ = 0.2273. The mean waiting time is 1/λ̂ = 22/5 = 4.4 minutes. This matches the sample mean (22/5 = 4.4). For exponential distribution, MLE of the mean is just the sample mean.

#### **Step 8: Check with another method**

If you guessed λ=0.2, ℓ = 5 log(0.2) - 22×0.2 = 5×(-1.609) - 4.4 = -8.045 - 4.4 = -12.445
If λ=0.2273, ℓ = 5 log(0.2273) - 22×0.2273 = 5×(-1.481) - 5.0 = -7.405 - 5.0 = -12.405 (higher)

---

### **7. How this appears inside neural networks and LLMs**

- **Cross-entropy loss:** For classification, minimizing cross-entropy is equivalent to MLE for a multinomial distribution. This is the standard loss for LLMs (next token prediction).

- **Mean squared error loss:** For regression with Gaussian noise (fixed variance), minimizing MSE is equivalent to MLE for the mean.

- **Maximum a posteriori (MAP):** Extends MLE by including a prior: θ̂_MAP = argmax log p(data|θ) + log p(θ). L2 regularization = Gaussian prior; L1 regularization = Laplace prior.

- **Variational autoencoders:** Maximize evidence lower bound (ELBO), which approximates maximum likelihood for latent variable models.

- **Generative models:** GANs implicitly maximize likelihood through adversarial training; diffusion models explicitly maximize likelihood of the data distribution.

- **Pretraining objectives:** LLMs are pretrained by maximizing likelihood of text tokens given previous tokens (autoregressive language modeling).

- **Parameter estimation in probabilistic models:** MLE is the default method for estimating parameters in mixture models, factor analysis, and state-space models.

- **Likelihood ratio testing:** Compares nested models (e.g., with/without a layer) using likelihood ratios derived from MLE.

---

### **8. Brain-like connection (efficient coding)**

The efficient coding hypothesis proposes that sensory neurons maximize the likelihood of representing natural stimuli given limited metabolic resources. The retina's center-surround receptive fields emerge from maximizing mutual information (closely related to likelihood) between neural responses and visual input. In the auditory system, neurons tune their response properties to maximize the likelihood of encoding species-specific sounds. The brain may not perform explicit MLE calculations, but evolution has shaped neural circuits to behave as if they are maximizing likelihood under biological constraints.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "MLE gives the true parameter values. If I have enough data, MLE finds the truth."

_Why it is wrong:_ MLE is consistent (converges to true value as n → ∞) but for finite data, it is biased in many cases. For Gaussian variance, the MLE σ̂² = (1/n)Σ(x_i-μ̂)² is biased downward (expected value is σ² × (n-1)/n). This is why we often use the unbiased estimator with denominator n-1. Also, MLE can overfit—without regularization, a sufficiently complex model can achieve arbitrarily high likelihood by memorizing the data. This is why modern deep learning uses regularization (implicitly or explicitly) even though cross-entropy is an MLE objective.

---

### **10. Why This Matters**

```text
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  When you train a neural network with cross-entropy or    |
|  mean squared error, you are doing maximum likelihood     |
|  estimation. Every gradient descent step moves parameters |
|  toward the peak of the likelihood landscape. MLE is the  |
|  bridge between probability theory and loss functions.    |
|  Master MLE and you understand why your loss function     |
|  takes its shape and what it truly means to "fit" a model.|
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You have a Poisson distribution with unknown mean λ. You observe data: [2, 3, 1, 4, 2].

**Question:** Derive the MLE for λ. What is its numerical value for this data?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Poisson PMF: P(X=k) = e^{-λ} λ^k / k!

Likelihood: L(λ) = ∏ e^{-λ} λ^{x_i} / (x_i!) = e^{-nλ} λ^{Σ x_i} / (∏ x_i!)

Log-likelihood: ℓ(λ) = -nλ + (Σ x_i) log λ - Σ log(x_i!)

Derivative: dℓ/dλ = -n + (Σ x_i)/λ = 0 → λ = (Σ x_i)/n = sample mean

For data: Σ x_i = 2+3+1+4+2 = 12, n=5, λ̂ = 12/5 = 2.4

MLE for Poisson mean is always the sample mean.
