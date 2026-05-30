# PCA (Principal Component Analysis)

**DOMAIN: MACHINE LEARNING FUNDAMENTALS | Sub domain: Core Algorithms**

---

**1. Why this concept matters for building intelligent systems**

High-dimensional data is everywhere: images with millions of pixels, text with hundred-thousand-word vocabularies, sensor streams with hundreds of channels. But most dimensions are redundant or noisy. Principal Component Analysis finds the directions of maximum variance—the "axes" that matter most—and projects data onto them, reducing dimensions while preserving information. This is the most widely used unsupervised dimensionality reduction technique. It reveals hidden structure, speeds up algorithms, removes noise, and visualizes high-dimensional data in 2D or 3D. Understanding PCA means understanding how to cut through the curse of dimensionality and find what truly matters in your data.

---

**2. Core idea**

**PCA finds orthogonal directions (principal components) that maximize the variance of projected data, ordered by amount of variance explained, allowing dimensionality reduction by keeping only the top components.**

---

**3. Concrete analogy**

Imagine you are a food critic rating restaurants on 100 different attributes: food quality, service speed, decor, price, location, etc. But many attributes are correlated. Restaurants with good food often have higher prices. Service speed might correlate with how busy the restaurant is.

You want to reduce these 100 measurements to just 2 or 3 "super-attributes" that capture most of the variation between restaurants.

PCA discovers these super-attributes automatically. The first principal component might be "overall quality" (combining food, service, ambiance). The second might be "value" (price relative to quality). The third might be "atmosphere type" (formal vs casual).

Each restaurant gets a score on each component. You can now compare restaurants in 2D space instead of 100D, visualize clusters, and identify outliers.

---

**4. ASCII diagram**

```
PCA finds directions of maximum variance:

    Original 2D data with correlation          After PCA rotation
    (x and y are correlated)

    y ↑                                       PC2 ↑
      │    x  x  x                              │    x
      │  x     x  x                             │  x
      │ x        x                              │ x
      │x         x                              │x
      │           x                             │   x
      │            x                            │     x
      │             x                           │       x
      └────────────→ x                          └────────────→ PC1
                                                (aligned with max variance)

    Explained variance ratio:

    Variance explained ↑
          │
      0.8 ┤████████████████████  PC1
          │
      0.6 ┤
          │
      0.4 ┤████  PC2
          │
      0.2 ┤█  PC3
          │
      0.0 ┼──┬──┬──┬──┬──→ Principal Components
             1  2  3  4  5

    Scree plot: Elbow at PC2 → keep first 2 components

    Data projection onto principal components:

    Original space (3D)          Projected onto PC1 and PC2 (2D)

         z ↑                           PC2 ↑
           │  ●                            │  ●
           │ ●  ●                           │ ●  ●
           │●    ●                          │●    ●
          ╱                               └────────→ PC1
         ╱
        x → y

    (Preserves structure while reducing dimensions)
```

---

**5. Mathematical formulation**

**Given:** Data matrix X of size n × d (n samples, d features)

**Step 1: Standardize (center and optionally scale):**

$$
\tilde{x}_{ij} = x_{ij} - \mu_j
$$

(Subtract mean μ_j from each feature j)

**Step 2: Compute covariance matrix:**

$$
\mathbf{C} = \frac{1}{n-1} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}
$$

C is d × d, where C_jk = covariance between feature j and k.

**Step 3: Compute eigenvectors and eigenvalues:**

Solve C v = λ v

Eigenvectors v₁, v₂, ..., v_d = principal components (directions)
Eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λ_d = variance along each component

**Step 4: Explained variance ratio:**

$$
\frac{\lambda_i}{\sum_{j=1}^d \lambda_j}
$$

(Unicode: λ_i / Σ λ_j)

**Step 5: Project data onto top k components:**

Let V_k be d × k matrix with top k eigenvectors as columns.

Projected data: Z = \tilde{X} V_k (n × k matrix)

**Reconstruction (inverse transform):**

$$
\hat{\mathbf{X}} = \mathbf{Z} \mathbf{V}_k^T + \boldsymbol{\mu}
$$

**Choosing k (cumulative explained variance):**

Choose smallest k such that:

$$
\frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} \geq 0.95
$$

(95% variance retained)

**Alternative formulation (SVD):**

\tilde{X} = U Σ V^T

Then V contains principal components, Σ² contains eigenvalues.

---

**6. Worked example (step-by-step)**

**Step 1: Define 2D data**

Points: (1,2), (3,6), (4,5), (5,7), (7,8)

n=5, d=2

**Step 2: Center the data**

Mean of x: (1+3+4+5+7)/5 = 20/5 = 4
Mean of y: (2+6+5+7+8)/5 = 28/5 = 5.6

Centered data:
(1-4, 2-5.6) = (-3, -3.6)
(3-4, 6-5.6) = (-1, 0.4)
(4-4, 5-5.6) = (0, -0.6)
(5-4, 7-5.6) = (1, 1.4)
(7-4, 8-5.6) = (3, 2.4)

**Step 3: Compute covariance matrix**

C_xx = Σ(x_i - x̄)² / (n-1) = (9+1+0+1+9)/4 = 20/4 = 5.0
C_yy = Σ(y_i - ȳ)² / (n-1) = (12.96+0.16+0.36+1.96+5.76)/4 = 21.2/4 = 5.3
C_xy = Σ(x_i - x̄)(y_i - ȳ) / (n-1) = (10.8 -0.4 +0 +1.4 +7.2)/4 = 19.0/4 = 4.75

C = [[5.0, 4.75], [4.75, 5.3]]

**Step 4: Find eigenvalues**

Eigenvalues satisfy: det(C - λI) = 0

(5.0-λ)(5.3-λ) - (4.75)² = 0
(5.0-λ)(5.3-λ) - 22.5625 = 0
26.5 - 5.3λ - 5.0λ + λ² - 22.5625 = 0
λ² - 10.3λ + 3.9375 = 0

λ = [10.3 ± √(10.3² - 4×3.9375)] / 2 = [10.3 ± √(106.09 - 15.75)]/2 = [10.3 ± √90.34]/2 = [10.3 ± 9.505]/2

λ₁ = (10.3 + 9.505)/2 = 19.805/2 = 9.9025
λ₂ = (10.3 - 9.505)/2 = 0.795/2 = 0.3975

**Step 5: Explained variance**

Total variance = 9.9025 + 0.3975 = 10.3
PC1 explains: 9.9025/10.3 = 0.961 (96.1%)
PC2 explains: 0.3975/10.3 = 0.039 (3.9%)

**Step 6: Find eigenvectors (PC directions)**

For λ₁ = 9.9025:
[5.0-9.9025, 4.75; 4.75, 5.3-9.9025] v = 0
[-4.9025, 4.75; 4.75, -4.6025] v = 0

First equation: -4.9025v₁ + 4.75v₂ = 0 → v₂ = (4.9025/4.75)v₁ ≈ 1.032v₁

Normalized: v₁ ≈ [0.696, 0.718] (direction up-right, correlation with x and y)

**Step 7: Project data onto PC1**

Each point's projection onto first principal component gives a 1D representation that captures 96% of the variance.

---

**7. How this appears inside neural networks and LLMs**

- **Dimensionality reduction for embeddings:** Word embeddings (300-4096 dims) are often PCA-reduced to 2D or 3D for visualization, revealing semantic clusters.

- **Preprocessing for speed:** PCA on high-dimensional features (e.g., bag-of-words with 100K terms) reduces input dimension before training.

- **Noise reduction:** Keeping only top principal components removes directions dominated by noise, acting as a filter.

- **Whitening transform:** PCA followed by scaling (dividing by √λ) decorrelates features and sets variance to 1. Used in some normalization schemes.

- **Latent space analysis:** In autoencoders and GANs, PCA on latent vectors reveals which directions correspond to meaningful attributes (e.g., "smiling" in face generation).

- **Explaining model decisions:** PCA on activations of LLM layers helps understand what each layer encodes.

- **Kernel PCA:** Extension for nonlinear dimensionality reduction using kernel trick (same as in SVMs).

- **Incremental PCA:** Online version for streaming data. Used when datasets are too large for standard PCA.

---

**8. Brain-like connection (receptive field formation)**

The primary visual cortex's receptive fields—Gabor filters tuned to edges and orientations—emerge from PCA-like principles applied to natural images. When you compute principal components of patches of natural scenes, the top components look remarkably like V1 simple cell receptive fields. Evolution and development seem to have discovered that PCA is an efficient coding strategy: decorrelating visual inputs and representing variance along the most informative directions. The brain reduces high-dimensional sensory data to a lower-dimensional representation that preserves behaviorally relevant information, much like PCA.

---

**9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "PCA finds the most important features for predicting my target variable."

_Why it is wrong:_ PCA is unsupervised—it ignores the target variable entirely. It finds directions of maximum variance in the inputs, not directions that best predict the output. For prediction, you might want directions that maximize covariance with the target (partial least squares) or that separate classes (linear discriminant analysis). A feature with huge variance could be pure noise with no predictive power. Always validate that PCA components are useful for your specific prediction task, or use supervised dimensionality reduction instead.

---

**10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  High-dimensional data is overwhelming. Most dimensions   |
|  are correlated, redundant, or noisy. PCA cuts through    |
|  the clutter. It finds the hidden axes that matter,       |
|  reveals clusters you could not see, and compresses       |
|  your data with minimal information loss. Every data      |
|  scientist should know PCA—it is the Swiss Army knife     |
|  of dimensionality reduction. When your data feels too    |
|  big, PCA makes it small without breaking it.             |
-------------------------------------------------------------
```

---

**11. Quick self-check question**

You perform PCA on a dataset with 10 features. The eigenvalues (variances) of the principal components are: [8.2, 3.1, 1.5, 0.9, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01].

**Question:** How many principal components should you keep to retain 90% of the variance? What percentage of variance does PC1 alone explain?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Total variance = 8.2 + 3.1 + 1.5 + 0.9 + 0.5 + 0.3 + 0.2 + 0.1 + 0.05 + 0.01 = 14.91

Cumulative variance:

PC1: 8.2/14.91 = 55.0% alone
PC1+PC2: (8.2+3.1)/14.91 = 11.3/14.91 = 75.8%
+PC3: (11.3+1.5)=12.8/14.91 = 85.8%
+PC4: (12.8+0.9)=13.7/14.91 = 91.9% (exceeds 90%)

Keep first 4 components. PC1 explains 55% of variance alone.
