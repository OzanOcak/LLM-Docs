# 🧠 ML/LLM Documentation

Complete guide to Machine Learning, Deep Learning, and Large Language Models.

---

## 📚 Domain 01: Mathematical Foundations

_The language ML speaks_ (16 articles)

### [→ Linear Algebra](domain-01-mathematical-foundations/01-linear-algebra/)

- Vectors and vector operations
- Matrices and matrix operations
- Eigenvalues and eigenvectors
- Linear transformations
- Vector spaces and dimensionality

### [→ Calculus](domain-01-mathematical-foundations/02-calculus/)

- Derivatives and gradients
- Partial derivatives
- Chain rule
- Gradient descent concept
- Multivariable calculus basics

### [→ Probability & Statistics](domain-01-mathematical-foundations/03-probability-statistics/)

- Basic probability rules
- Random variables
- Distributions
- Mean, variance, standard deviation
- Bayes' theorem
- Maximum likelihood estimation

---

## 📚 Domain 02: Machine Learning Fundamentals

_The core concepts_ (13 articles)

### [→ ML Basics](domain-02-ml-fundamentals/01-ml-basics/)

- What is machine learning?
- Supervised vs unsupervised vs reinforcement learning
- Training vs validation vs test sets
- Overfitting and underfitting
- Bias-variance tradeoff
- Cross-validation

### [→ Core Algorithms](domain-02-ml-fundamentals/02-core-algorithms/)

- Linear regression
- Logistic regression
- Decision trees
- Random forests
- SVM (Support Vector Machines)
- K-means clustering
- PCA (Principal Component Analysis)

---

## 📚 Domain 03: Deep Learning Foundations

_The neural network layer_ (13 articles)

### [→ Neural Networks Basics](domain-03-deep-learning-foundations/01-neural-networks-basics/)

- Perceptron and neurons
- Activation functions (ReLU, sigmoid, tanh)
- Layers (input, hidden, output)
- Forward propagation
- Backpropagation
- Loss functions
- Optimizers (SGD, Adam)

### [→ Training Neural Networks](domain-03-deep-learning-foundations/02-training-neural-networks/)

- Learning rates
- Batch size
- Epochs
- Regularization (dropout, L1/L2)
- Batch normalization
- Vanishing/exploding gradients

---

## 📚 Domain 04: Advanced Architectures

_Where LLMs emerge_ (13 articles)

### [→ Sequence Models (Pre-Transformer)](domain-04-advanced-architectures/01-sequence-models/)

- RNNs (Recurrent Neural Networks)
- LSTMs (Long Short-Term Memory)
- GRUs (Gated Recurrent Units)
- Bidirectional RNNs
- Seq2Seq models
- Attention mechanism (original version)

### [→ Transformers (The LLM Revolution)](domain-04-advanced-architectures/02-transformers/)

- Self-attention mechanism
- Multi-head attention
- Positional encodings
- Feed-forward layers
- Layer normalization
- Residual connections
- Encoder vs decoder architecture

---

## 📚 Domain 05: LLM-Specific Concepts

_What makes LLMs special_ (16 articles)

### [→ Tokenization](domain-05-llm-specific-concepts/01-tokenization/)

- Word vs subword vs character tokenization
- BPE (Byte Pair Encoding)
- WordPiece
- SentencePiece
- Special tokens

### [→ Training Paradigms](domain-05-llm-specific-concepts/02-training-paradigms/)

- Pre-training
- Fine-tuning
- Few-shot learning
- Zero-shot learning
- Instruction tuning
- RLHF (Reinforcement Learning from Human Feedback)

### [→ Model Architecture Variants](domain-05-llm-specific-concepts/03-architecture-variants/)

- Encoder-only (BERT-like)
- Decoder-only (GPT-like)
- Encoder-decoder (T5-like)
- Sparse attention
- Mixture of Experts (MoE)

---

## 📚 Domain 06: Vector Space & Retrieval

_The LLM's memory and knowledge access_ (19 articles)

### [→ Embeddings](domain-06-vector-space-retrieval/01-embeddings/)

- What are embeddings?
- How embeddings capture semantics
- Dimensionality reduction (t-SNE, UMAP)
- Properties of embedding spaces

### [→ Distance Metrics](domain-06-vector-space-retrieval/02-distance-metrics/)

- Cosine similarity
- Euclidean distance
- Dot product
- Comparing distance metrics

### [→ Vector Databases](domain-06-vector-space-retrieval/03-vector-databases/)

- What are vector databases?
- Indexing algorithms (HNSW, IVF)
- Approximate Nearest Neighbor (ANN)
- Similarity search basics
- Popular vector databases

### [→ RAG (Retrieval-Augmented Generation)](domain-06-vector-space-retrieval/04-rag/)

- The problem: LLMs have knowledge cutoff
- The solution: Retrieve relevant documents
- How RAG works step-by-step
- Chunking strategies
- Hybrid search
- Real-world applications

---

## 📚 Domain 07: Practical Aspects

_Making it work in reality_ (31 articles)

### [→ Data Processing](domain-07-practical-aspects/01-data-processing/)

- Data collection
- Cleaning and preprocessing
- Data augmentation
- Dataset splits
- Handling imbalanced data
- Data labeling strategies

### [→ Evaluation & Metrics](domain-07-practical-aspects/02-evaluation-metrics/)

- Perplexity
- BLEU score
- ROUGE score
- BERTScore
- Human evaluation
- Benchmark datasets (GLUE, SuperGLUE, MMLU, HumanEval, HELM)
- LLM-as-a-judge

### [→ Deployment & Optimization](domain-07-practical-aspects/03-deployment-optimization/)

- Model compression
- Quantization (GPTQ, AWQ, GGUF)
- Pruning
- Distillation
- KV caching
- Continuous batching
- PagedAttention / vLLM
- Speculative decoding
- Inference optimization
- API design
- Monitoring

### [→ Production Considerations](domain-07-practical-aspects/04-production/)

- Cost optimization
- Latency vs throughput
- Scaling inference
- A/B testing
- Fallback strategies

---

## 🚀 Quick Stats

| Domain                        | Subdomains | Articles |
| ----------------------------- | ---------- | -------- |
| 01. Mathematical Foundations  | 3          | 16       |
| 02. ML Fundamentals           | 2          | 13       |
| 03. Deep Learning Foundations | 2          | 13       |
| 04. Advanced Architectures    | 2          | 13       |
| 05. LLM-Specific Concepts     | 3          | 16       |
| 06. Vector Space & Retrieval  | 4          | 19       |
| 07. Practical Aspects         | 4          | 31       |
| **Total**                     | **20**     | **121**  |

---

## 📖 Learning Path

1. **Domain 01** → Math foundations
2. **Domain 02** → ML basics
3. **Domain 03** → Deep learning
4. **Domain 04** → Modern architectures
5. **Domain 05** → LLM specifics
6. **Domain 06** → Retrieval systems
7. **Domain 07** → Production deployment
