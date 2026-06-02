# Mixture of Experts (MoE)

## **DOMAIN: LLM-SPECIFIC CONCEPTS | Sub domain: Model Architecture Variants**

---

### **1. Why this concept matters**

Scaling LLMs has a problem: doubling parameters doubles FLOPs for every token. What if you could have 1 trillion parameters but only activate 100 billion per token? That is Mixture of Experts (MoE). Instead of one giant feed-forward network, MoE uses many smaller "expert" networks and a router that selects only the top-k experts for each token. This decouples parameter count from inference cost. MoE is why models like Mixtral (8×7B = 47B total parameters) run as fast as a 12B model while performing like a 30B+ model. It is the most practical path to trillion-parameter models that still fit on GPUs.

---

### **2. Core idea**

**A Mixture of Experts layer replaces a dense feed-forward network with multiple parallel expert networks and a gating (router) network that selects the top-k experts for each token, activating only a subset of parameters per forward pass.**

---

### **3. Concrete analogy**

Imagine a large hospital with many specialized doctors:

- Cardiologists (heart experts)
- Neurologists (brain experts)
- Orthopedists (bone experts)

When a patient arrives, a triage nurse (router) quickly assesses symptoms and sends the patient to the 2 most relevant specialists, not all 100 doctors. The patient receives expert care without paying for 98 unused doctors.

MoE works the same way. For each token, the router computes which experts (specialists) are most relevant and routes the token to them. The other experts remain idle. Result: the model has many parameters (many doctors) but each token uses only a few (only the relevant specialists).

---

### **4. ASCII diagram**

```
Standard FFN layer vs MoE layer:

Standard FFN (dense):
    Input token x
         │
    ┌────┴────┐
    │   FFN   │ (one large network, all parameters active)
    └────┬────┘
         │
    Output y


MoE layer (with 4 experts, top-2 routing):

    Input token x
         │
         ▼
    Router (gating network)
         │
    ┌────┼──────────────────────────┐
    │    │                          │
    ▼    ▼                          ▼
    Expert 1   Expert 2   Expert 3   Expert 4
    (active)    (active)   (idle)     (idle)
    │           │
    └─────┬─────┘
          │
    Combine (weighted sum)
          │
          ▼
    Output y


Router decision for "cat" token:

    Expert scores: E1=0.8, E2=0.6, E3=0.2, E4=0.1
    Top-2: E1 and E2
    Output = 0.8*E1(x) + 0.6*E2(x)


Mixtral 8×7B architecture:

    8 experts, each 7B parameters (56B total)
    Top-2 routing per token
    Active parameters per token: 2×7B = 14B
    Inference speed: ~12B dense model
    Performance: ~30B+ dense model
```

---

### **5. Mathematical formulation**

**MoE layer with N experts {E₁...Eₙ} and router R:**

For input token x (or hidden state h):

Router computes scores:

$$
\mathbf{s} = \text{softmax}(W_r \mathbf{x})
$$

Select top-k experts by score:

$$
\mathcal{T} = \text{top-k}(\mathbf{s})
$$

Output is weighted sum of selected experts:

$$
\mathbf{y} = \sum_{i \in \mathcal{T}} s_i E_i(\mathbf{x})
$$

**Sparsity:** Only k experts are active per token. Total parameters = N × |E_i|. FLOPs per token = k × |E_i|.

**Load balancing loss (auxiliary loss):**

To prevent all tokens routing to same expert:

$$
L_{\text{balance}} = \alpha \sum_{i=1}^N f_i \cdot P_i
$$

Where f_i = fraction of tokens routed to expert i, P_i = average router probability for expert i. Penalizes imbalance.

**Capacity factor:** Limits tokens per expert. If expert exceeds capacity, extra tokens are dropped (treated as zeros). Prevents one expert from being overloaded.

**Switch Transformer (Google):** Top-1 routing (only one expert per token). Simpler, more sparse.

---

### **6. Worked example (step-by-step)**

#### **Step 1: Setup**

MoE layer with 4 experts. Each expert is a small FFN. Router is a linear layer.

#### **Step 2: Input token**

Token embedding x = [2.0, 0.5] (2-dim for simplicity)

#### **Step 3: Router computes scores**

W_r is 4×2 matrix (learned). Example:

W_r · x = [1.5, 0.8, 0.3, 0.1] (raw scores)

Softmax: s = [0.43, 0.28, 0.18, 0.11]

#### **Step 4: Top-2 selection**

Top scores: expert 1 (0.43), expert 2 (0.28)

#### **Step 5: Compute expert outputs**

E₁(x) = [0.9, -0.2] (example output)
E₂(x) = [0.3, 0.7]

#### **Step 6: Weighted combination**

y = 0.43×[0.9, -0.2] + 0.28×[0.3, 0.7] = [0.387, -0.086] + [0.084, 0.196] = [0.471, 0.110]

Experts 3 and 4 are not computed (saved compute).

#### **Step 7: Batch processing (multiple tokens)**

Batch of 512 tokens. Each token may route to different experts. GPUs handle efficiently by grouping tokens by expert.

---

### **7. How this appears inside neural networks and LLMs**

- **Mixtral (Mistral AI, 2024):** 8 experts × 7B = 56B total parameters. Top-2 routing. Performs like 30B+ dense model, runs as fast as 12B dense. State-of-the-art MoE LLM.

- **Switch Transformer (Google, 2021):** Top-1 routing (simpler). 1.6T parameters (but sparse). Showed MoE scales better than dense.

- **GShard (Google, 2020):** First large-scale MoE for translation. Top-2 routing.

- **GPT-4 (rumored):** 16 experts × ~111B = 1.76T total parameters. Top-2 routing. Explains why GPT-4 is large but inference cost is manageable.

- **Qwen 1.5 MoE (Alibaba):** 2.7B active, 14B total parameters.

- **DeepSeek-V2:** 21B active, 236B total. Uses MoE for both attention and FFN.

- **Router challenges:** Load balancing is critical. Without auxiliary loss, all tokens route to same expert (collapse). With proper balancing, MoE works beautifully.

- **Expert specialization:** After training, experts specialize: some on code, some on math, some on multiple languages, some on reasoning. This is emergent, not programmed.

- **Training instability:** MoE training is harder than dense. Small router changes cause large expert load shifts. Requires careful tuning of load balancing loss and capacity factor.

---

### **8. Brain-like connection (functional specialization)**

The brain has functional specialization: different brain regions handle different tasks (vision in occipital lobe, language in temporal/frontal lobes, motor control in cerebellum). Yet the brain does not activate all neurons for every task. MoE is the artificial analogue: specialized "experts" are activated only when relevant. The router is like the brain's task-switching mechanisms (anterior cingulate cortex, basal ganglia) that route information to appropriate processing regions. Patients with brain lesions show deficits in specific domains (e.g., face blindness) but intact performance elsewhere—like disabling an expert in MoE. The brain evolved sparsity for efficiency; MoE is engineering rediscovering the same principle.

---

### **9. Common misunderstanding and why it is wrong**

_Misunderstanding:_ "MoE models have 50B parameters, so they need 50B parameter memory. They are not practical."

_Why it is wrong:_ MoE parameters are sparse—only a fraction are active per token. For inference, you only need to load the experts that will be used. With top-2 routing and 8 experts, only 2 experts are active per token. You can store all parameters on disk (or distributed memory) and load active experts on demand. Training requires all parameters in memory for gradient updates (though ZeRO can shard them). For inference, Mixtral 8×7B runs on a single 24GB GPU because only 14B parameters are active. The "56B parameter" number is total storage, not runtime memory. This is the key: MoE decouples total parameters from inference cost.

---

### **10. Why This Matters**

```
-------------------------------------------------------------
|  WHY THIS MATTERS                                         |
|                                                           |
|  MoE is the secret to trillion-parameter models that     |
|  still run on GPUs. By activating only a subset of        |
|  experts per token, MoE decouples model size from         |
|  inference cost. Mixtral (56B total) runs as fast as      |
|  12B dense but performs like 30B+. GPT-4 is almost        |
|  certainly MoE. The future of LLM scaling is not denser   |
|  models—it is sparser models with more experts. MoE       |
|  is how we get smarter models without slower inference.   |
-------------------------------------------------------------
```

---

### **11. Quick self-check question**

You are designing a 100B parameter dense model (single FFN). It requires 200GB of GPU memory for inference and is too slow. You propose switching to MoE with 16 experts (each 6.25B), top-2 routing.

**Question:** What is the total parameter count? How many parameters are active per token? What is the expected inference memory compared to the dense model? What new training challenge must you address?

_(Answer hidden below)_

---

.

.

.

.

.

**Answer:**

Total parameters: 16 × 6.25B = 100B (same as dense model).

Active parameters per token: top-2 experts = 2 × 6.25B = 12.5B. Inference memory (for weights) ≈ 12.5B × 2 bytes (FP16) = 25GB (plus KV cache, activations). Dense model required 200GB. MoE reduces memory by ~8× at same parameter count.

Expected inference speed: MoE with 12.5B active parameters should run similarly to a 12.5B dense model (not 100B dense). This is the win: same total parameters, much faster inference.

New training challenge: Load balancing. The router must learn to distribute tokens evenly across experts; otherwise, some experts are never trained (wasted capacity) and others are overloaded. Use auxiliary load balancing loss and capacity factor. Also, training MoE is slower because experts are updated sparsely (each expert sees only ~1/8 of batch per step). Requires careful tuning of learning rate, batch size, and router temperature.
