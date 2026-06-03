# ML/LLM Documentation

Unlike traditional machine learning resources that separate pure math from engineering, this guide provides a direct, uninterrupted path from foundational linear algebra straight to production-grade LLM optimization. Every article strips away academic jargon to give you the exact intuition needed to build and scale modern AI models.

---

**Stats:** 6 Domains | 16 Subdomains | 90 Deep-Dive Articles

---

## 11-Step Learning Framework

Every single one of the 90 articles in this repository follows a strict, predictable framework designed to target every learning style (visual, mathematical, and practical):

1. **Why This Concept Matters**: The macro-perspective and real-world importance.
2. **Core Idea**: A razor-sharp, one-sentence summary.
3. **Concrete Analogy**: Relatable real-world or biological comparisons (e.g., visual cortex parallel columns for Multi-Head Attention, or the hippocampus for Vector Spaces).
4. **ASCII Diagram**: Clean visual mapping of data flow and tensor shapes.
5. **Mathematical Formulation**: Clear equations written out with explicit parameter dimensions.
6. **Worked Example**: A step-by-step mathematical walk-through using minimal, simplified numbers.
7. **Production Context**: How this concept actually manifests inside modern LLMs (e.g., MQA, GQA, vLLM).
8. **Brain-Like Connection**: The fascinating biological equivalent of the algorithm.
9. **Common Misunderstanding**: Pitfalls, myths, and why they are wrong.
10. **The Bottom Line**: A summary of why the architecture would fail without this specific piece.
11. **Quick Self-Check**: A conceptual question to test your retention.

## Domain 01: Mathematical Foundations

_The language ML speaks_ (16 articles)

### Linear Algebra

- [Vectors and vector operations](domain-01-mathematical-foundations/01-linear-algebra/01-vectors-and-vector-operations.md)
- [Matrices and matrix operations](domain-01-mathematical-foundations/01-linear-algebra/02-matrices-and-matrix-operations.md)
- [Eigenvalues and eigenvectors](domain-01-mathematical-foundations/01-linear-algebra/03-eigenvalues-and-eigenvectors.md)
- [Linear transformations](domain-01-mathematical-foundations/01-linear-algebra/04-linear-transformations.md)
- [Vector spaces and dimensionality](domain-01-mathematical-foundations/01-linear-algebra/05-vector-spaces-and-dimensionality.md)

### Calculus

- [Derivatives and gradients](domain-01-mathematical-foundations/02-calculus/01-derivatives-and-gradients.md)
- [Partial derivatives](domain-01-mathematical-foundations/02-calculus/02-partial-derivatives.md)
- [Chain rule](domain-01-mathematical-foundations/02-calculus/03-chain-rule.md)
- [Gradient descent concept](domain-01-mathematical-foundations/02-calculus/04-gradient-descent-concept.md)
- [Multivariable calculus basics](domain-01-mathematical-foundations/02-calculus/05-multivariable-calculus-basics.md)

### Probability & Statistics

- [Basic probability rules](domain-01-mathematical-foundations/03-probability-statistics/01-basic-probability-rules.md)
- [Random variables](domain-01-mathematical-foundations/03-probability-statistics/02-random-variables.md)
- [Distributions](domain-01-mathematical-foundations/03-probability-statistics/03-distributions.md)
- [Mean, variance, standard deviation](domain-01-mathematical-foundations/03-probability-statistics/04-mean-variance-standard-deviation.md)
- [Bayes' theorem](domain-01-mathematical-foundations/03-probability-statistics/05-bayes-theorem.md)
- [Maximum likelihood estimation](domain-01-mathematical-foundations/03-probability-statistics/06-maximum-likelihood-estimation.md)

---

## Domain 02: Machine Learning Fundamentals

_The core concepts_ (13 articles)

### ML Basics

- [What is machine learning?](domain-02-ml-fundamentals/01-ml-basics/01-what-is-machine-learning.md)
- [Supervised vs unsupervised vs reinforcement learning](domain-02-ml-fundamentals/01-ml-basics/02-supervised-unsupervised-reinforcement-learning.md)
- [Training vs validation vs test sets](domain-02-ml-fundamentals/01-ml-basics/03-training-validation-test-sets.md)
- [Overfitting and underfitting](domain-02-ml-fundamentals/01-ml-basics/04-overfitting-underfitting.md)
- [Bias-variance tradeoff](domain-02-ml-fundamentals/01-ml-basics/05-bias-variance-tradeoff.md)
- [Cross-validation](domain-02-ml-fundamentals/01-ml-basics/06-cross-validation.md)

### Core Algorithms

- [Linear regression](domain-02-ml-fundamentals/02-core-algorithms/01-linear-regression.md)
- [Logistic regression](domain-02-ml-fundamentals/02-core-algorithms/02-logistic-regression.md)
- [Decision trees](domain-02-ml-fundamentals/02-core-algorithms/03-decision-trees.md)
- [Random forests](domain-02-ml-fundamentals/02-core-algorithms/04-random-forests.md)
- [SVM (Support Vector Machines)](domain-02-ml-fundamentals/02-core-algorithms/05-support-vector-machines.md)
- [K-means clustering](domain-02-ml-fundamentals/02-core-algorithms/06-k-means-clustering.md)
- [PCA (Principal Component Analysis)](domain-02-ml-fundamentals/02-core-algorithms/07-principal-component-analysis.md)

---

## Domain 03: Deep Learning Foundations

_The neural network layer_ (13 articles)

### Neural Networks Basics

- [Perceptron and neurons](domain-03-deep-learning-foundations/01-neural-networks-basics/01-perceptron-and-neurons.md)
- [Activation functions](domain-03-deep-learning-foundations/01-neural-networks-basics/02-activation-functions.md)
- [Layers (input, hidden, output)](domain-03-deep-learning-foundations/01-neural-networks-basics/03-layers-input-hidden-output.md)
- [Forward propagation](domain-03-deep-learning-foundations/01-neural-networks-basics/04-forward-propagation.md)
- [Backpropagation](domain-03-deep-learning-foundations/01-neural-networks-basics/05-backpropagation.md)
- [Loss functions](domain-03-deep-learning-foundations/01-neural-networks-basics/06-loss-functions.md)
- [Optimizers (SGD, Adam)](domain-03-deep-learning-foundations/01-neural-networks-basics/07-optimizers-sgd-adam.md)

### Training Neural Networks

- [Learning rates](domain-03-deep-learning-foundations/02-training-neural-networks/01-learning-rates.md)
- [Batch size](domain-03-deep-learning-foundations/02-training-neural-networks/02-batch-size.md)
- [Epochs](domain-03-deep-learning-foundations/02-training-neural-networks/03-epochs.md)
- [Regularization (dropout, L1/L2)](domain-03-deep-learning-foundations/02-training-neural-networks/04-regularization-dropout-l1-l2.md)
- [Batch normalization](domain-03-deep-learning-foundations/02-training-neural-networks/05-batch-normalization.md)
- [Vanishing/exploding gradients](domain-03-deep-learning-foundations/02-training-neural-networks/06-vanishing-exploding-gradients.md)

---

## Domain 04: Advanced Architectures

_Where LLMs emerge_ (13 articles)

### Sequence Models (Pre-Transformer)

- [RNNs (Recurrent Neural Networks)](domain-04-advanced-architectures/01-sequence-models/01-recurrent-neural-networks-rnn.md)
- [LSTMs (Long Short-Term Memory)](domain-04-advanced-architectures/01-sequence-models/02-long-short-term-memory-lstm.md)
- [GRUs (Gated Recurrent Units)](domain-04-advanced-architectures/01-sequence-models/03-gated-recurrent-units-gru.md)
- [Bidirectional RNNs](domain-04-advanced-architectures/01-sequence-models/04-bidirectional-rnns.md)
- [Seq2Seq models](domain-04-advanced-architectures/01-sequence-models/05-seq2seq-models.md)
- [Attention mechanism (original version)](domain-04-advanced-architectures/01-sequence-models/06-attention-mechanism-original.md)

### Transformers (The LLM Revolution)

- [Self-attention mechanism](domain-04-advanced-architectures/02-transformers/01-self-attention-mechanism.md)
- [Multi-head attention](domain-04-advanced-architectures/02-transformers/02-multi-head-attention.md)
- [Positional encodings](domain-04-advanced-architectures/02-transformers/03-positional-encodings.md)
- [Feed-forward layers](domain-04-advanced-architectures/02-transformers/04-feed-forward-layers.md)
- [Layer normalization](domain-04-advanced-architectures/02-transformers/05-layer-normalization.md)
- [Residual connections](domain-04-advanced-architectures/02-transformers/06-residual-connections.md)
- [Encoder vs decoder architecture](domain-04-advanced-architectures/02-transformers/07-encoder-vs-decoder-architecture.md)

---

## Domain 05: LLM-Specific Concepts

_What makes LLMs special_ (16 articles)

### Tokenization

- [Word vs subword vs character tokenization](domain-05-llm-specific-concepts/01-tokenization/01-word-subword-character-tokenization.md)
- [BPE (Byte Pair Encoding)](domain-05-llm-specific-concepts/01-tokenization/02-byte-pair-encoding-bpe.md)
- [WordPiece](domain-05-llm-specific-concepts/01-tokenization/03-wordpiece.md)
- [SentencePiece](domain-05-llm-specific-concepts/01-tokenization/04-sentencepiece.md)
- [Special tokens](domain-05-llm-specific-concepts/01-tokenization/05-special-tokens.md)

### Training Paradigms

- [Pre-training](domain-05-llm-specific-concepts/02-training-paradigms/01-pre-training.md)
- [Fine-tuning](domain-05-llm-specific-concepts/02-training-paradigms/02-fine-tuning.md)
- [Few-shot learning](domain-05-llm-specific-concepts/02-training-paradigms/03-few-shot-learning.md)
- [Zero-shot learning](domain-05-llm-specific-concepts/02-training-paradigms/04-zero-shot-learning.md)
- [Instruction tuning](domain-05-llm-specific-concepts/02-training-paradigms/05-instruction-tuning.md)
- [RLHF (Reinforcement Learning from Human Feedback)](domain-05-llm-specific-concepts/02-training-paradigms/06-rlhf-reinforcement-learning-human-feedback.md)

### Model Architecture Variants

- [Encoder-only (BERT-like)](domain-05-llm-specific-concepts/03-architecture-variants/01-encoder-only-bert-like.md)
- [Decoder-only (GPT-like)](domain-05-llm-specific-concepts/03-architecture-variants/02-decoder-only-gpt-like.md)
- [Encoder-decoder (T5-like)](domain-05-llm-specific-concepts/03-architecture-variants/03-encoder-decoder-t5-like.md)
- [Sparse attention](domain-05-llm-specific-concepts/03-architecture-variants/04-sparse-attention.md)
- [Mixture of Experts (MoE)](domain-05-llm-specific-concepts/03-architecture-variants/05-mixture-of-experts-moe.md)

---

## Domain 06: Vector Space & Retrieval

_The LLM's memory and knowledge access_ (19 articles)

### Embeddings

- [What are embeddings?](domain-06-vector-space-retrieval/01-embeddings/01-what-are-embeddings.md)
- [How embeddings capture semantics](domain-06-vector-space-retrieval/01-embeddings/02-how-embeddings-capture-semantics.md)
- [Dimensionality reduction (t-SNE, UMAP)](domain-06-vector-space-retrieval/01-embeddings/03-dimensionality-reduction-tsne-umap.md)
- [Properties of embedding spaces](domain-06-vector-space-retrieval/01-embeddings/04-properties-of-embedding-spaces.md)

### Distance Metrics

- [Cosine similarity](domain-06-vector-space-retrieval/02-distance-metrics/01-cosine-similarity.md)
- [Euclidean distance](domain-06-vector-space-retrieval/02-distance-metrics/02-euclidean-distance.md)
- [Dot product](domain-06-vector-space-retrieval/02-distance-metrics/03-dot-product.md)
- [Comparing distance metrics](domain-06-vector-space-retrieval/02-distance-metrics/04-comparing-distance-metrics.md)

### Vector Databases

- [What are vector databases?](domain-06-vector-space-retrieval/03-vector-databases/01-what-are-vector-databases.md)
- [Indexing algorithms (HNSW, IVF)](domain-06-vector-space-retrieval/03-vector-databases/02-indexing-algorithms-hnsw-ivf.md)
- [Approximate Nearest Neighbor (ANN)](domain-06-vector-space-retrieval/03-vector-databases/03-approximate-nearest-neighbor-ann.md)
- [Similarity search basics](domain-06-vector-space-retrieval/03-vector-databases/04-similarity-search-basics.md)
- [Popular vector databases](domain-06-vector-space-retrieval/03-vector-databases/05-popular-vector-databases-pinecone-weaviate-qdrant-milvus.md)

### RAG (Retrieval-Augmented Generation)

- [The problem: LLMs have knowledge cutoff](domain-06-vector-space-retrieval/04-rag/01-the-problem-llms-have-knowledge-cutoff.md)
- [The solution: Retrieve relevant documents](domain-06-vector-space-retrieval/04-rag/02-retrieve-relevant-documents-first.md)
- [How RAG works step-by-step](domain-06-vector-space-retrieval/04-rag/03-how-rag-works-step-by-step.md)
- [Chunking strategies](domain-06-vector-space-retrieval/04-rag/04-chunking-strategies-for-rag.md)
- [Hybrid search](domain-06-vector-space-retrieval/04-rag/05-hybrid-search-vector-keyword.md)
- [Real-world applications](domain-06-vector-space-retrieval/04-rag/06-real-world-applications.md)

---

### Evaluation & Metrics

- [Perplexity](domain-07-practical-aspects/02-evaluation-metrics/01-perplexity.md)
- [BLEU score](domain-07-practical-aspects/02-evaluation-metrics/02-bleu-score.md)
- [ROUGE score](domain-07-practical-aspects/02-evaluation-metrics/03-rouge-score.md)
- [BERTScore](domain-07-practical-aspects/02-evaluation-metrics/04-bertscore.md)
- [Human evaluation](domain-07-practical-aspects/02-evaluation-metrics/05-human-evaluation.md)
- [Benchmark datasets](domain-07-practical-aspects/02-evaluation-metrics/06-benchmark-datasets-glue-superglue-mmlu-humaneval-helm.md)

### Deployment & Optimization

- [Model compression](domain-07-practical-aspects/03-deployment-optimization/01-model-compression.md)
- [Quantization (GPTQ, AWQ, GGUF)](domain-07-practical-aspects/03-deployment-optimization/02-quantization-gptq-awq-gguf.md)
- [Pruning](domain-07-practical-aspects/03-deployment-optimization/03-pruning.md)
- [Distillation](domain-07-practical-aspects/03-deployment-optimization/04-distillation.md)
- [KV caching](domain-07-practical-aspects/03-deployment-optimization/05-kv-caching-for-inference.md)
- [Continuous batching](domain-07-practical-aspects/03-deployment-optimization/06-continuous-batching.md)
- [PagedAttention / vLLM](domain-07-practical-aspects/03-deployment-optimization/07-pagedattention-vllm.md)
- [Speculative decoding](domain-07-practical-aspects/03-deployment-optimization/08-speculative-decoding.md)
- [Inference optimization](domain-07-practical-aspects/03-deployment-optimization/09-inference-optimization.md)
- [API design](domain-07-practical-aspects/03-deployment-optimization/10-api-design.md)
- [Monitoring](domain-07-practical-aspects/03-deployment-optimization/11-monitoring-and-observability.md)

### Production Considerations

- [Cost optimization](domain-07-practical-aspects/04-production-considerations/01-cost-optimization-strategies.md)
- [Latency vs throughput](domain-07-practical-aspects/04-production-considerations/02-latency-vs-throughput-tradeoffs.md)
- [Scaling inference](domain-07-practical-aspects/04-production-considerations/03-scaling-inference-horizontal-vertical.md)
- [A/B testing](domain-07-practical-aspects/04-production-considerations/04-ab-testing-for-llm-applications.md)
- [Fallback strategies](domain-07-practical-aspects/04-production-considerations/05-fallback-strategies-and-error-handling.md)

---

## Quick Stats

| Domain                        | Subdomains | Articles |
| ----------------------------- | ---------- | -------- |
| 01. Mathematical Foundations  | 3          | 16       |
| 02. ML Fundamentals           | 2          | 13       |
| 03. Deep Learning Foundations | 2          | 13       |
| 04. Advanced Architectures    | 2          | 13       |
| 05. LLM-Specific Concepts     | 3          | 16       |
| 06. Vector Space & Retrieval  | 4          | 19       |
| **Total**                     | **16**     | **90**   |

---

## Learning Path

1. **Beginner** → Domains 01 → 02 → 03
2. **Intermediate** → Domains 04 → 05
3. **Advanced** → Domains 06

---

_Last updated: 2026_
