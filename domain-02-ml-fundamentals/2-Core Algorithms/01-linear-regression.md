# Linear regression

**DOMAIN: MACHINE LEARNING FUNDAMENTALS | Sub domain: Core Algorithms**

---

**1. Why this concept matters for building intelligent systems**

Before neural networks, before random forests, before SVMs, there was linear regression. It is the simplest learning algorithm, yet it contains all the essential ideas: a model, a loss function, an optimization procedure, and generalization to new data. More importantly, linear regression is the foundation upon which everything else is built. A neural network layer is linear regression followed by an activation function. The last layer of most LLMs is linear regression (plus softmax). Understanding linear regression means understanding the atomic unit of deep learning.

---

**2. Core idea**

**Linear regression models the relationship between input variables and an output variable as a weighted sum plus an intercept, finding the weights that minimize the sum of squared prediction errors.**

---

**3. Concrete analogy**

Imagine you are a real estate agent trying to predict house prices. You notice that larger houses tend to cost more. You collect data: sizes and prices of 10 recent sales. You draw a scatter plot and try to draw the best straight line through the points. That line is your linear regression model.

When a new house comes on the market, you find its size, go up to the line, and read off the predicted price. The line is not perfect—actual prices scatter around it—but it captures the general trend.

Now add more features: number of bedrooms, lot size, age of house. Instead of a 2D line, you now have a hyperplane in higher dimensions. The principle is the same: weighted sum of features equals predicted price.

---

**4. ASCII diagram**

```
Linear regression in 1D (one input feature):

    Price ($100k)
        ↑
      500│                                    ●
          │
      400│                        ●
          │
      300│            ●
          │                    ╱
      200│        ● ╱
          │      ╱
      100│  ●  ╱
          │ ╱
        0┼───────╱────────────────→ Size (sq ft)
          0   1000  2000  3000

    Points: actual sales
    Line: ŷ = w₁x + w₀

    Residual (error): vertical distance from point to line

    Multiple linear regression (2 features):

    Price = w₁ × Size + w₂ × Bedrooms + w₀

    Can visualize as a plane in 3D:

              Price ↑
                 │    ╱
                 │   ╱
                 │  ╱
                 │ ╱
                ╱│
               ╱ │
            Size  ┼──→ Bedrooms

    The coefficients w₁, w₂ tell you how much price changes
    when each feature increases by 1 unit (holding others fixed).
```

---

**5. Mathematical formulation**

**Simple linear regression (one input):**

Model: ŷ = w₁x + w₀

Where:

- x = input feature
- ŷ = predicted output
- w₁ = slope (weight)
- w₀ = intercept (bias)

**Multiple linear regression (d inputs):**

Model: ŷ = w₁x₁ + w₂x₂ + ... + w_d x_d + w₀

In vector form:

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b
$$

Where w ∈ ℝᵈ, x ∈ ℝᵈ, b ∈ ℝ (bias)

**Matrix form for n examples:**

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}
$$

Where X is n×d design matrix (each row is an example)

**Loss function (Mean Squared Error - MSE):**

$$
L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^n (y_i - (\mathbf{w}^T \mathbf{x}_i + b))^2
$$

(Unicode: L(w,b) = (1/n) Σ (y_i - (w^T x_i + b))²)

**Closed-form solution (ordinary least squares):**

Add column of 1s to X to absorb bias. Then:

$$
\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

(Unicode: ŵ = (X^T X)^{-1} X^T y)

**Gradient descent update:**

$$
\nabla_{\mathbf{w}} L = -\frac{2}{n} \sum_{i=1}^n (y_i - (\mathbf{w}^T \mathbf{x}_i + b)) \mathbf{x}_i
$$

$$
\nabla_{b} L = -\frac{2}{n} \sum_{i=1}^n (y_i - (\mathbf{w}^T \mathbf{x}_i + b))
$$

**R² coefficient of determination (goodness of fit):**

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

Where ȳ = mean of observed y. R² = 1 means perfect fit, 0 means no better than predicting the mean.

---

**6. Worked example (step-by-step)**

**Step 1: Define the data**

House sizes (x) in 1000 sq ft and prices (y) in $100k:

| Size (x) | Price (y) |
| -------- | --------- |
| 1.0      | 2.0       |
| 1.5      | 2.5       |
| 2.0      | 3.0       |
| 2.5      | 3.5       |
| 3.0      | 4.0       |

**Step 2: Guess a line and compute errors**

Guess w₁ = 1, w₀ = 1 → ŷ = 1×x + 1

| x   | y   | ŷ   | error (y-ŷ) | squared error |
| --- | --- | --- | ----------- | ------------- |
| 1.0 | 2.0 | 2.0 | 0.0         | 0.0           |
| 1.5 | 2.5 | 2.5 | 0.0         | 0.0           |
| 2.0 | 3.0 | 3.0 | 0.0         | 0.0           |
| 2.5 | 3.5 | 3.5 | 0.0         | 0.0           |
| 3.0 | 4.0 | 4.0 | 0.0         | 0.0           |

MSE = 0. Perfect fit! But this is contrived—real data has noise.

**Step 3: Noisy data example**

Data: (1, 2.1), (2, 2.9), (3, 4.2), (4, 3.8), (5, 5.0)

**Step 4: Compute means**

x̄ = (1+2+3+4+5)/5 = 15/5 = 3

ȳ = (2.1+2.9+4.2+3.8+5.0)/5 = 18/5 = 3.6

**Step 5: Compute w₁ (slope) using formula**

w₁ = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²

| x_i-x̄ | y_i-ȳ | product | (x_i-x̄)² |
| ----- | ----- | ------- | -------- |
| -2    | -1.5  | 3.0     | 4        |
| -1    | -0.7  | 0.7     | 1        |
| 0     | 0.6   | 0.0     | 0        |
| 1     | 0.2   | 0.2     | 1        |
| 2     | 1.4   | 2.8     | 4        |

Sum of products = 3.0 + 0.7 + 0.0 + 0.2 + 2.8 = 6.7

Sum of squares = 4 + 1 + 0 + 1 + 4 = 10

w₁ = 6.7 / 10 = 0.67

**Step 6: Compute w₀**

w₀ = ȳ - w₁x̄ = 3.6 - 0.67×3 = 3.6 - 2.01 = 1.59

**Step 7: Final model**

ŷ = 0.67x + 1.59

**Step 8: Make prediction**

New house size x = 2.5 → predicted price = 0.67×2.5 + 1.59 = 1.675 + 1.59 = 3.265 ($326,500)

---

**7. How this appears inside neural networks or LLMs**

- **Linear layer:** Every fully connected layer in a neural network is y = Wx + b—linear regression on features.

- **Last layer of classification networks:** Softmax is applied to linear regression outputs: logits = Wx + b.

- **Embedding layers:** Lookup table is linear transformation of one-hot vector.

- **Linear probing:** Used to interpret LLMs—train a linear regression on top of frozen hidden states to predict properties (e.g., sentiment, tense). High accuracy indicates the representation encodes that property.

- **Positional encoding in transformers:** Adds linear functions of position to embeddings.

- **Residual connections:** y = F(x) + x. The + x is linear (identity).

- **Layer normalization:** Linear operation (subtract mean, divide by std) applied per example.

- **Value and output projections:** In attention, value vectors are transformed by linear layers before being summed.

- **Low-rank adaptation (LoRA):** Fine-tunes LLMs by adding low-rank linear updates to weight matrices.

---

**8. Brain-like connection (linear receptive fields)**

Neurons in the primary visual cortex have linear receptive fields. Each neuron computes a weighted sum of light intensities in a small region of the retina (plus a bias, the spontaneous firing rate). The weights form a Gabor filter tuned to a specific orientation and frequency. This is linear regression performed by biology. The neuron's output is approximately ŷ = w^T x + b, where x are pixel intensities. The brain learns these weights through experience, just as linear regression learns from data. The simplicity of linear regression is not a limitation—it is a building block reused throughout biological and artificial neural systems.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "Linear regression can only fit straight lines. Real-world relationships are rarely linear, so linear regression is useless."

_Why it is wrong:_ Linear regression fits linear combinations of features—but features can be transformations of raw inputs. You can include x², log(x), sin(x), or interactions (x₁×x₂) as features. The model remains linear in the parameters (weights) but can capture highly nonlinear relationships. For example, ŷ = w₀ + w₁x + w₂x² fits a parabola. This is still linear regression (polynomial regression). The "linear" refers to linearity in weights, not inputs. With basis functions and kernels, linear regression can model almost any smooth function.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  Linear regression is the "hello world" of machine        |
|  learning, but it is also the hidden engine of deep       |
|  networks. Every neural network layer does linear         |
|  regression plus a nonlinearity. The last layer of        |
|  GPT-4 is linear regression. Master linear regression     |
|  and you understand the core operation that, repeated     |
|  billions of times, produces intelligence.                |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

A linear regression model is trained to predict exam score (y) from hours studied (x). The model is ŷ = 5x + 40.

**Question:** What is the predicted score for 4 hours of study? What does the slope (5) mean? What does the intercept (40) mean?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

ŷ = 5×4 + 40 = 20 + 40 = 60

Slope (5): Each additional hour of study increases predicted exam score by 5 points.

Intercept (40): A student who studies 0 hours is predicted to score 40 points. This is the baseline (could represent guessing or prior knowledge).
