# Part 6: Modern AI Architectures (2026 Edition)

> *"There is no single 'best' architecture  only the best architecture for your problem, your budget, and your constraints."*

By 2026, the AI landscape has diversified far beyond the original transformer. While attention is still the foundation, the *systems* built on top of it vary enormously. This part is your field guide to every major architecture pattern, when to use each one, and how they compare.

---

## Table of Contents

1. [The Architecture Landscape](#1-the-architecture-landscape)
2. [Dense Transformer Models](#2-dense-transformer-models)
3. [Mixture of Experts (MoE)](#3-mixture-of-experts-moe)
4. [Retrieval-Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
5. [Tool-Using Agents](#5-tool-using-agents)
6. [Reasoning Models](#6-reasoning-models)
7. [Long-Context Models](#7-long-context-models)
8. [Multimodal Models](#8-multimodal-models)
9. [Architecture Comparison](#9-architecture-comparison)
10. [Quiz](#10-quiz)

---

## 1. The Architecture Landscape

### The Restaurant Analogy

Think of different AI architectures like different types of restaurants:

| Architecture           | Restaurant Analogy                                                       | When You'd Choose It                    |
| ---------------------- | ------------------------------------------------------------------------ | --------------------------------------- |
| **Dense Transformer**  | Classic fine-dining restaurant  one master chef does everything         | Smaller models, predictable workloads   |
| **Mixture of Experts** | Food court  specialized chefs, you visit the right one                  | Large models where efficiency matters   |
| **RAG**                | Chef who looks up recipes from a cookbook mid-cooking                    | Need current or private information     |
| **Tool-Using Agent**   | Chef who can call for pizza delivery, order ingredients, check reviews   | Need to take actions in the world       |
| **Reasoning Model**    | Chef who carefully plans the meal, considering every step before cooking | Complex problems requiring deep thought |
| **Long-Context**       | Chef who can read (and remember) the entire cookbook at once             | Processing massive documents            |
| **Multimodal**         | Chef who works with sight, sound, and taste simultaneously               | Processing images, audio, video, text   |

### The 2026 Architecture Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Modern AI Architecture Map                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                    ┌─────────────────┐                              │
│                    │   Foundation:    │                             │
│                    │   Transformer    │                             │
│                    │   Architecture   │                             │
│                    └────────┬────────┘                              │
│                             │                                       │
│              ┌──────────────┼──────────────┐                        │
│              │              │              │                        │
│         ┌────┴────┐   ┌────┴────┐   ┌────┴────┐                     │
│         │  Dense  │   │   MoE   │   │  Multi- │                     │
│         │ Models  │   │ Models  │   │  modal  │                     │
│         └────┬────┘   └────┬────┘   └────┬────┘                     │
│              │              │              │                        │
│              └──────────────┼──────────────┘                        │
│                             │                                       │
│              ┌──────────────┼──────────────┐                        │
│              │              │              │                        │
│         ┌────┴────┐   ┌────┴────┐   ┌────┴────┐                     │
│         │   RAG   │   │ Agents  │   │Reasoning│                     │
│         │ Systems │   │ & Tools │   │ Models  │                     │
│         └─────────┘   └─────────┘   └─────────┘                     │
│                                                                     │
│   These are COMPOSABLE  a system can be MoE + Multimodal +         │
│   RAG + Tool-using + Reasoning all at once!                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dense Transformer Models

### What Is a Dense Model?

A **dense model** activates **every parameter** for every input token. When you feed "Hello world" into GPT-4o-mini or Llama 3.1 8B, every single weight in the network participates in computing the output.

```
Dense Model (e.g., Llama 3.1 8B):

Input: "Hello world"
         │
         ▼
┌─────────────────────┐
│  Layer 1 (ALL       │  ← Every neuron fires
│  weights active)    │
├─────────────────────┤
│  Layer 2 (ALL       │  ← Every neuron fires
│  weights active)    │
├─────────────────────┤
│       ...           │
├─────────────────────┤
│  Layer 32 (ALL      │  ← Every neuron fires
│  weights active)    │
└─────────────────────┘
         │
         ▼
Output: next token probabilities

Total params: 8B
Active params per token: 8B (100%)
```

### When Dense Models Are Still Best

| Scenario | Why Dense Works Best |
|----------|---------------------|
| **Small models (< 13B)** | MoE overhead not worth it at small scale |
| **Latency-critical apps** | No routing overhead, simpler execution |
| **Edge/mobile deployment** | Simpler to quantize and optimize |
| **Single-GPU inference** | No cross-GPU communication needed |
| **Fine-tuning** | Simpler to fine-tune, no expert balancing |
| **Consistent workloads** | Predictable compute per token |

### Notable Dense Models (2024-2026)

| Model | Parameters | Context | Strengths |
|-------|-----------|---------|-----------|
| Llama 3.1 8B/70B | 8B, 70B | 128K | Strong open-source baseline |
| Gemma 2 (Google) | 2B, 9B, 27B | 8K | Efficient, great for fine-tuning |
| Qwen 2.5 (Alibaba) | 0.5B-72B | 128K | Strong multilingual |
| Phi-4 (Microsoft) | 14B | 16K | Excellent reasoning for size |
| GPT-4o-mini (OpenAI) | ~8B (est.) | 128K | Great cost/performance ratio |
| Mistral Small (Mistral) | 24B | 32K | Strong European open model |
| Llama 4 Scout (Meta) | 17B active (MoE) | 10M | Actually MoE! Shows the trend |

> **Key trend:** Pure dense models above ~70B parameters are becoming rare. The frontier has moved to MoE architectures, while dense models dominate the small/efficient model category.

### Dense Model Architecture Details

A modern dense transformer (decoder-only) for a model like Llama 3.1 8B:

```
┌─────────────────────────────────────────────┐
│              Llama 3.1 8B Architecture       │
├─────────────────────────────────────────────┤
│                                             │
│  Hidden dimension:     4,096                │
│  Number of layers:     32                   │
│  Attention heads:      32                   │
│  KV heads:             8  (Grouped-Query    │
│                            Attention)       │
│  FFN intermediate:     14,336               │
│  Vocabulary size:      128,256              │
│  Context length:       128K tokens          │
│  Positional encoding:  RoPE                 │
│  Normalization:        RMSNorm (pre-norm)   │
│  Activation:           SiLU (Swish)         │
│  Attention variant:    GQA                  │
│                                             │
│  Key modern innovations:                     │
│  • GQA: Fewer KV heads → less memory        │
│  • RoPE: Rotary positions → extendable      │
│  • SiLU: Smoother than ReLU                  │
│  • RMSNorm: Faster than LayerNorm            │
│  • Pre-norm: More stable training            │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 3. Mixture of Experts (MoE)

### The Core Idea

**Mixture of Experts** replaces each dense feed-forward layer with multiple "expert" sub-networks and a **router** that selects which experts process each token. This lets you have a model with enormous total knowledge (many parameters) but fast inference (only a few experts active per token).

### The Office Building Analogy

```
Dense Model = One person who knows everything
  → Slower, but consistent
  → Must be an expert in EVERY topic

MoE Model = An office building full of specialists
  → A receptionist (router) directs you to the right expert
  → Only 2-3 specialists handle your question
  → The building has 100 experts, but you only see 2
  → Faster per query, but the building is huge
```

### How the Router Works

```
┌─────────────────────────────────────────────────────────────┐
│                    MoE Layer Detail                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Token                                                │
│  Embedding: [0.3, -0.1, 0.8, 0.2, ...]                     │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐                                               │
│  │  Router   │  (small neural network, typically linear)    │
│  │  Network  │                                              │
│  └──────────┘                                               │
│       │                                                     │
│       ▼                                                     │
│  Router Scores (softmax):                                   │
│  Expert 1: 0.05  │  Expert 5: 0.02                          │
│  Expert 2: 0.42  ← Selected (top-2)                        │
│  Expert 3: 0.08  │  Expert 6: 0.03                          │
│  Expert 4: 0.35  ← Selected (top-2)                        │
│  Expert 5: 0.02  │  Expert 8: 0.01                          │
│                                                             │
│       │ Only route to Experts 2 and 4                       │
│       ▼                                                     │
│  ┌──────────┐        ┌──────────┐                           │
│  │ Expert 2  │        │ Expert 4  │   (Experts 1,3,5,6,7,8  │
│  │ (FFN)     │        │ (FFN)     │    are NOT computed!)    │
│  └────┬─────┘        └─────┬────┘                           │
│       │                    │                                │
│       ▼                    ▼                                │
│  Output = 0.55 × Expert2(x) + 0.45 × Expert4(x)            │
│          (weighted by router probabilities, renormalized)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Active vs Total Parameters

This is the key distinction for MoE models:

```
┌─────────────────────────────────────────────────────────────┐
│           Active vs Total Parameters                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Mixtral 8x7B:                                              │
│    Total parameters:    46.7B  (stored on disk/GPU memory)  │
│    Active per token:    12.9B  (actually computed)           │
│    Number of experts:   8                                    │
│    Experts active:      2 per token                          │
│                                                             │
│  DeepSeek-V3:                                               │
│    Total parameters:    671B   (enormous knowledge base)     │
│    Active per token:    37B    (fast inference)              │
│    Number of experts:   256 + 1 shared                      │
│    Experts active:      8 per token                         │
│                                                             │
│  Llama 4 Maverick:                                          │
│    Total parameters:    400B                                 │
│    Active per token:    17B                                  │
│    Number of experts:   128                                  │
│    Experts active:      Top-k routing                       │
│                                                             │
│  INSIGHT: DeepSeek-V3 has GPT-4-level knowledge (671B       │
│  total params) but runs almost as fast as a 37B dense       │
│  model!                                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Expert Balancing: The Load Problem

A naive router might send all tokens to the same 2 experts, leaving others unused:

```
Unbalanced routing:              Balanced routing:
Expert 1: ████████████ 40%       Expert 1: ████ 12%
Expert 2: ██████████   33%       Expert 2: ████ 13%
Expert 3: ███          10%       Expert 3: ████ 12%
Expert 4: ██            7%       Expert 4: ████ 13%
Expert 5: █             3%       Expert 5: ████ 12%
Expert 6: █             3%       Expert 6: ████ 13%
Expert 7: █             2%       Expert 7: ████ 12%
Expert 8: █             2%       Expert 8: ████ 13%

Problem: Experts 7-8 never       All experts learn and
learn anything useful!           contribute equally.
```

**Solutions:**

| Technique | How It Works |
|-----------|-------------|
| **Auxiliary load-balancing loss** | Add a penalty term encouraging equal expert usage |
| **Expert capacity** | Cap max tokens per expert; overflow goes to next-best expert |
| **Token dropping** | If an expert is full, drop the lowest-priority tokens |
| **Expert choice routing** | Experts choose their tokens instead of tokens choosing experts |
| **Noise in routing** | Add Gaussian noise to router logits during training |

### Why DeepSeek and Mixtral Use MoE

#### Mixtral (Mistral AI, 2023-2024)

Mixtral was the first widely-adopted open-source MoE model:

```
Mixtral 8x7B (2023):
├── 8 experts per MoE layer (replacing FFN layers)
├── 2 experts active per token
├── 46.7B total, 12.9B active
├── Performance ≈ Llama 2 70B (6× fewer active params!)
├── Inference cost ≈ 13B dense model
└── Set off the open-source MoE revolution

Mixtral 8x22B (2024):
├── 8 experts, each 22B-class
├── 141B total, 39B active  
├── Competitive with GPT-4-turbo on many tasks
└── Pushed MoE to larger scale
```

#### DeepSeek (DeepSeek AI, 2024-2025)

DeepSeek pushed MoE to its extreme with innovative designs:

```
DeepSeek-V2 (2024):
├── Multi-head Latent Attention (MLA)
│   └── Compresses KV cache dramatically
├── 236B total, 21B active
├── Extremely cost-efficient inference
└── Novel: shared + routed expert architecture

DeepSeek-V3 (2024-2025):
├── 671B total, 37B active
├── 256 routed experts + 1 shared expert
├── Auxiliary-loss-free load balancing
│   └── Uses bias terms instead of auxiliary loss
├── Multi-Token Prediction (MTP)
│   └── Predicts multiple future tokens simultaneously
├── FP8 mixed-precision training
├── Trained for ~$5.5M (extremely cost-efficient!)
└── Performance rivaling GPT-4o and Claude 3.5 Sonnet

DeepSeek Innovations:
┌────────────────────────────────────────────────────┐
│  1. Shared Expert: One expert that ALL tokens use  │
│     → Captures common patterns                     │
│                                                    │
│  2. Fine-grained Experts: 256 smaller experts      │
│     instead of 8 large ones                        │
│     → More precise specialization                  │
│                                                    │
│  3. Bias-based load balancing: Add learnable bias  │
│     to router instead of auxiliary loss             │
│     → No interference with main training signal    │
│                                                    │
│  4. MLA (Multi-head Latent Attention):             │
│     → Compress KV cache by 93.3%                   │
│     → Enables much larger batch sizes              │
└────────────────────────────────────────────────────┘
```

### MoE Tradeoffs Summary

| Advantage | Tradeoff |
|-----------|----------|
| More knowledge per FLOP | Higher memory (must store ALL experts) |
| Faster inference than equivalent dense | More complex training infrastructure |
| Better scaling to very large models | Expert load balancing is tricky |
| Lower per-token compute cost | Harder to fine-tune (expert collapse risk) |
| Can specialize experts for different domains | Communication overhead in distributed setup |

---

## 4. Retrieval-Augmented Generation (RAG)

### The Core Problem RAG Solves

LLMs have a **knowledge cutoff**  they only know what was in their training data. They also **hallucinate**  confidently stating false information. RAG solves both by giving the model access to an external knowledge base at inference time.

### The Open-Book Exam Analogy

```
Without RAG (closed-book exam):
  Student must answer from memory only.
  "What was our Q3 2025 revenue?"  
  → "I'm not sure, but based on industry trends..." (hallucination risk)

With RAG (open-book exam):
  Student can look up information in their notes.
  "What was our Q3 2025 revenue?"
  → *looks up financial report* → "According to the Q3 report, revenue was $4.2B"
```

### RAG Architecture Diagram

```
═══════════════════════════════════════════════════════════════════
                     RAG ARCHITECTURE
═══════════════════════════════════════════════════════════════════

OFFLINE PHASE (Index Building):
───────────────────────────────

  ┌──────────────────┐
  │  Document Corpus  │  (PDFs, web pages, databases, etc.)
  │  "Company docs,   │
  │   knowledge base, │
  │   product manuals" │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Chunking         │  Split documents into passages
  │  (512-2048 tokens │  (overlapping chunks)
  │   per chunk)      │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Embedding Model  │  Convert text → dense vectors
  │  (e.g., BGE,      │  "How to reset password" → [0.2, -0.1, 0.8, ...]
  │   E5, Cohere      │
  │   embed)          │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Vector Database  │  Store and index embeddings
  │  (Pinecone, Qdrant│  for fast similarity search
  │   Weaviate, pgvec │
  │   Chroma)         │
  └──────────────────┘


ONLINE PHASE (Query Time):
──────────────────────────

  User Query: "How do I reset my password?"
           │
           ▼
  ┌──────────────────┐
  │  Embed Query      │  Same embedding model as indexing
  │  → [0.3, -0.2,    │
  │     0.7, ...]     │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Vector Search    │  Find top-K most similar chunks
  │  (cosine sim or   │  
  │   ANN search)     │  Top 5 results:
  │                   │  1. "Password Reset Guide" (0.92)
  │                   │  2. "Account Security FAQ" (0.87)
  │                   │  3. "Login Troubleshooting" (0.83)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────┐
  │  Prompt Construction                                  │
  │                                                      │
  │  System: "Answer based on the provided context."     │
  │                                                      │
  │  Context:                                            │
  │  [Chunk 1] "To reset your password, go to           │
  │   Settings > Security > Reset Password..."          │
  │  [Chunk 2] "If you've forgotten your password,      │
  │   click 'Forgot Password' on the login page..."     │
  │                                                      │
  │  User: "How do I reset my password?"                 │
  └────────┬─────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────┐
  │  LLM Generation   │  Generate answer grounded in context
  │                   │
  │  "To reset your   │
  │   password:        │
  │   1. Go to         │
  │   Settings >       │
  │   Security >       │
  │   Reset Password   │
  │   2. Enter your    │
  │   current and new  │
  │   password..."     │
  └──────────────────┘

═══════════════════════════════════════════════════════════════════
```

### Embedding Models

Embedding models convert text into dense vectors where similar meanings are close together:

```
"How to reset password"     → [0.31, -0.12, 0.78, 0.45, ...]
"Password reset instructions" → [0.29, -0.14, 0.81, 0.42, ...]  ← Very similar!
"Best pizza in New York"    → [-0.55, 0.72, 0.11, -0.33, ...]   ← Very different

Cosine similarity:
  reset queries: 0.97 (almost identical meaning)
  reset vs pizza: 0.12 (completely different topics)
```

#### Popular Embedding Models (2025-2026)

| Model | Dimensions | Max Tokens | Strengths |
|-------|-----------|------------|-----------|
| OpenAI text-embedding-3-large | 3,072 | 8,191 | Great quality, easy API |
| Cohere embed-v4 | 1,024 | 512 | Multilingual, efficient |
| BGE-M3 (BAAI) | 1,024 | 8,192 | Open-source, multilingual |
| E5-Mistral-7B | 4,096 | 32K | LLM-based, high quality |
| Jina Embeddings v3 | 1,024 | 8,192 | Open-source, flexible |
| Nomic Embed Text v2 | 768 | 8,192 | Open, Matryoshka support |
| Voyage 3 | 1,024 | 32K | Excellent for code & retrieval |
| Google Gemini Embedding | 3,072 | 8,192 | Integrated with Google ecosystem |



#### Approximate Nearest Neighbor (ANN) Search

ANN algorithms trade slight accuracy for massive speed improvements:

| Algorithm | How It Works | Speed | Accuracy |
|-----------|-------------|-------|----------|
| **HNSW** | Graph-based: navigate linked nodes | Very Fast | Very High |
| **IVF** | Cluster vectors, search only relevant clusters | Fast | High |
| **Product Quantization** | Compress vectors, search compressed space | Fastest | Moderate |
| **ScaNN** (Google) | Hybrid: partition + quantize | Very Fast | Very High |

```
HNSW (Hierarchical Navigable Small World):

Layer 3: o─────────────────────o  (few long-range connections)
          │                     │
Layer 2: o───o─────o───────o───o  (more connections)
          │   │     │       │   │
Layer 1: o─o─o─o─o─o─o─o─o─o─o─o  (many short-range connections)
          │ │ │ │ │ │ │ │ │ │ │ │
Layer 0: oooooooooooooooooooooooo  (all vectors)

Search: Start at top layer, greedily navigate to nearest node,
        then drop to lower layer and repeat.
        
Result: O(log n) search instead of O(n)!
For 10M vectors: ~100 comparisons instead of 10,000,000
```

### Chunking Strategies

How you split documents into chunks **dramatically affects** RAG quality:

#### Common Chunking Approaches

```
FIXED-SIZE CHUNKING:
┌──────────┐┌──────────┐┌──────────┐
│ 512 tokens││ 512 tokens││ 512 tokens│  Simple but may split
└──────────┘└──────────┘└──────────┘  mid-sentence

OVERLAPPING CHUNKS:
┌──────────────┐
│   Chunk 1     │
│          ┌────┴───────────┐         50-100 token overlap
│          │   Chunk 2       │        ensures context continuity
│          │           ┌─────┴──────┐
│          │           │   Chunk 3   │
└──────────┘           └────────────┘

SEMANTIC CHUNKING:
┌─────────────────┐  ┌────────────────────┐  ┌──────────┐
│ Paragraph about  │  │ Paragraph about    │  │ Table of │
│ company history  │  │ product features   │  │ pricing  │
└─────────────────┘  └────────────────────┘  └──────────┘
Split at natural boundaries (paragraphs, sections, topics)

RECURSIVE CHARACTER SPLITTING:
Try splitting by: sections → paragraphs → sentences → words
Keep chunks within size limits while preserving structure.

AGENTIC CHUNKING (2025+):
Use an LLM to decide optimal chunk boundaries!
"Read this document and tell me where the natural
 topic boundaries are."
```

#### Chunking Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| Fixed-size | Simple, predictable | Splits mid-thought | Uniform documents |
| Overlapping | Captures boundary context | More storage, redundancy | General purpose |
| Semantic | Preserves meaning | Uneven sizes | Structured documents |
| Recursive | Respects document structure | More complex | Code, markdown, HTML |
| Agentic/LLM-based | Best quality boundaries | Slow, expensive to index | High-value corpora |

### When to Use RAG vs Fine-Tuning

This is one of the most common architectural decisions:

```
┌─────────────────────────────────────────────────────────────────┐
│              RAG vs Fine-Tuning Decision Matrix                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    USE RAG WHEN:                                 │
│  ✓ Information changes frequently                               │
│  ✓ You need citations/sources                                   │
│  ✓ You have a large document corpus                             │
│  ✓ Factual accuracy is critical                                 │
│  ✓ You need to add new knowledge without retraining             │
│  ✓ You want to control what knowledge is available              │
│                                                                 │
│                    USE FINE-TUNING WHEN:                         │
│  ✓ You need to change model behavior/style                      │
│  ✓ The knowledge is stable and domain-specific                  │
│  ✓ You need faster inference (no retrieval step)                │
│  ✓ The task requires specialized formatting                     │
│  ✓ You want to improve instruction following                    │
│                                                                 │
│                    USE BOTH WHEN:                                │
│  ✓ Enterprise applications (fine-tune for style + RAG           │
│    for knowledge)                                               │
│  ✓ Customer support (fine-tune for tone + RAG for               │
│    product knowledge)                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Factor | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Knowledge freshness** | Always current | Frozen at training time |
| **Hallucination** | Reduced (grounded in docs) | Can still hallucinate |
| **Setup cost** | Moderate (build index) | High (training compute) |
| **Latency** | Higher (retrieval + generation) | Lower (direct generation) |
| **Privacy** | Data stays in your control | Data used in training |
| **Citability** | Can cite sources | Cannot cite sources |
| **Behavior change** | Limited | Significant |
| **Maintenance** | Update index as docs change | Retrain periodically |

### Advanced RAG Techniques (2025-2026)

```
BASIC RAG:
  Query → Retrieve → Generate

ADVANCED RAG:
  ┌─────────────────────────────────────────────┐
  │  1. Query Rewriting                          │
  │     LLM rephrases query for better retrieval │
  │                                              │
  │  2. Hybrid Search                            │
  │     Vector search + keyword search (BM25)    │
  │                                              │
  │  3. Re-ranking                               │
  │     Cross-encoder re-scores top results      │
  │                                              │
  │  4. Multi-hop Retrieval                      │
  │     Follow chains of related documents       │
  │                                              │
  │  5. Self-RAG                                 │
  │     Model decides WHEN to retrieve           │
  │     and evaluates retrieval quality           │
  │                                              │
  │  6. GraphRAG (Microsoft)                     │
  │     Build knowledge graph from documents,    │
  │     use graph structure for retrieval         │
  │                                              │
  │  7. Agentic RAG                              │
  │     Agent decides retrieval strategy,         │
  │     iterates if results insufficient          │
  └─────────────────────────────────────────────┘
```

---

## 5. Tool-Using Agents

### Beyond Text Generation

Standard LLMs can only generate text. **Tool-using agents** can take actions in the world  searching the web, running code, querying databases, calling APIs, and more.

### The Assistant Analogy

```
Standard LLM:
  "What's the weather in Tokyo?"
  → "I don't have access to real-time weather data, but Tokyo 
     typically has..." (guessing from training data)

Tool-using Agent:
  "What's the weather in Tokyo?"
  → [THINKS: I should use the weather API tool]
  → [CALLS: get_weather(city="Tokyo")]
  → [RECEIVES: {"temp": 22, "condition": "partly cloudy"}]
  → "The current weather in Tokyo is 22°C and partly cloudy."
```

### Function Calling

Modern LLMs are trained to output structured **function calls** when they determine a tool is needed:

```
┌─────────────────────────────────────────────────────────────┐
│                   Function Calling Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. User provides available tools:                          │
│     tools = [                                               │
│       {                                                     │
│         "name": "get_weather",                              │
│         "description": "Get current weather for a city",    │
│         "parameters": {                                     │
│           "city": {"type": "string"},                       │
│           "units": {"type": "string", "enum": ["C","F"]}   │
│         }                                                   │
│       },                                                    │
│       {                                                     │
│         "name": "search_web",                               │
│         "description": "Search the internet",               │
│         "parameters": {                                     │
│           "query": {"type": "string"}                       │
│         }                                                   │
│       }                                                     │
│     ]                                                       │
│                                                             │
│  2. User asks: "What's the weather in Paris?"               │
│                                                             │
│  3. LLM outputs a tool call:                                │
│     {                                                       │
│       "tool_call": {                                        │
│         "name": "get_weather",                              │
│         "arguments": {"city": "Paris", "units": "C"}        │
│       }                                                     │
│     }                                                       │
│                                                             │
│  4. Application executes the function, returns result       │
│                                                             │
│  5. LLM receives result and generates final response        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The ReAct Pattern

**ReAct** (Reason + Act) is the dominant pattern for agent behavior, interleaving thinking and acting:

```
User: "What's the population of the country that won 
       the most recent FIFA World Cup?"

Agent:
  THOUGHT: I need to find who won the most recent World Cup,
           then look up that country's population.
  
  ACTION: search_web("most recent FIFA World Cup winner")
  OBSERVATION: Argentina won the 2022 FIFA World Cup in Qatar.
  
  THOUGHT: Argentina won. Now I need Argentina's population.
  
  ACTION: search_web("Argentina population 2025")
  OBSERVATION: Argentina's estimated population in 2025 is 
               approximately 46.7 million.
  
  THOUGHT: I now have all the information needed.
  
  ANSWER: The most recent FIFA World Cup was won by Argentina 
          in 2022. Argentina's current population is approximately 
          46.7 million people.
```


### Model Context Protocol (MCP)

**MCP** (introduced by Anthropic in late 2024, widely adopted by 2025-2026) standardizes how AI models connect to external tools and data sources:

```
┌─────────────────────────────────────────────────────────────┐
│                Model Context Protocol (MCP)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BEFORE MCP:                                                │
│  Every app builds custom integrations                       │
│                                                             │
│  App A ──custom──→ Gmail                                    │
│  App A ──custom──→ Slack                                    │
│  App B ──custom──→ Gmail  (duplicate work!)                 │
│  App B ──custom──→ Slack  (duplicate work!)                 │
│                                                             │
│  WITH MCP:                                                  │
│  Universal protocol  build once, use everywhere            │
│                                                             │
│  ┌─────────────┐    MCP     ┌──────────────────┐           │
│  │  Any AI App  │◄─────────►│  MCP Server:     │           │
│  │  (Client)    │           │  Gmail           │           │
│  └─────────────┘            └──────────────────┘           │
│         │           MCP     ┌──────────────────┐           │
│         └──────────────────►│  MCP Server:     │           │
│                             │  Slack           │           │
│                             └──────────────────┘           │
│                     MCP     ┌──────────────────┐           │
│         └──────────────────►│  MCP Server:     │           │
│                             │  Database        │           │
│                             └──────────────────┘           │
│                                                             │
│  MCP provides three types of capabilities:                  │
│  1. TOOLS:     Functions the model can call                 │
│  2. RESOURCES: Data the model can read                      │
│  3. PROMPTS:   Templates for common interactions            │
│                                                             │
│  Think of MCP as "USB-C for AI"  one standard connector   │
│  that works with everything.                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### MCP Example

```json
// MCP Server exposes tools like this:
{
  "tools": [
    {
      "name": "read_email",
      "description": "Read emails from inbox",
      "inputSchema": {
        "type": "object",
        "properties": {
          "folder": {"type": "string", "default": "inbox"},
          "limit": {"type": "integer", "default": 10}
        }
      }
    },
    {
      "name": "send_email",
      "description": "Send an email",
      "inputSchema": {
        "type": "object",
        "properties": {
          "to": {"type": "string"},
          "subject": {"type": "string"},
          "body": {"type": "string"}
        },
        "required": ["to", "subject", "body"]
      }
    }
  ]
}
```

### Agent Architecture Patterns

```
┌─────────────────────────────────────────────────────────────┐
│               Agent Architecture Patterns                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. SINGLE AGENT (Simple)                                   │
│     ┌───────┐                                               │
│     │ Agent │──→ Tool A, Tool B, Tool C                     │
│     └───────┘                                               │
│     One LLM does everything                                 │
│                                                             │
│  2. ROUTER AGENT                                            │
│     ┌────────┐                                              │
│     │ Router │─┬→ Code Agent                                │
│     └────────┘ ├→ Research Agent                            │
│                └→ Writing Agent                             │
│     Specialized sub-agents for different tasks              │
│                                                             │
│  3. MULTI-AGENT COLLABORATION                               │
│     ┌─────────┐    ┌──────────┐    ┌──────────┐            │
│     │ Planner │───→│ Executor │───→│ Reviewer │            │
│     └─────────┘    └──────────┘    └──────────┘            │
│          ▲                              │                   │
│          └──────────────────────────────┘                   │
│     Agents play different roles, iterate                    │
│                                                             │
│  4. HIERARCHICAL AGENTS                                     │
│     ┌───────────────┐                                       │
│     │ Manager Agent  │                                      │
│     └───────┬───────┘                                       │
│       ┌─────┼─────┐                                         │
│       ▼     ▼     ▼                                         │
│     ┌───┐ ┌───┐ ┌───┐                                      │
│     │W1 │ │W2 │ │W3 │  Worker agents                       │
│     └───┘ └───┘ └───┘                                      │
│     Manager delegates and synthesizes                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Agent Frameworks (2025-2026)

| Framework | Creator | Key Feature |
|-----------|---------|-------------|
| **LangGraph** | LangChain | Graph-based agent workflows |
| **CrewAI** | Open-source | Multi-agent role-playing |
| **AutoGen** | Microsoft | Multi-agent conversations |
| **OpenAI Agents SDK** | OpenAI | Production agent framework |
| **Claude Code / Artifacts** | Anthropic | Agentic coding & tool use |
| **Google ADK** | Google | Agent Development Kit |
| **Mastra** | Open-source | TypeScript-first agent framework |
| **Pydantic AI** | Pydantic team | Type-safe Python agents |

---

## 6. Reasoning Models

### The Breakthrough: Thinking Before Answering

Standard LLMs generate answers **token by token** without planning ahead. Reasoning models take a fundamentally different approach: they **think step by step** before producing an answer, spending extra compute at inference time.

### The Exam Analogy

```
Standard LLM (fast, reactive):
  Reads question → Immediately starts writing answer
  Like a student who answers without thinking
  
Reasoning Model (deliberate, methodical):
  Reads question → Thinks through the problem → Checks work → Writes answer
  Like a student who uses scratch paper before writing the final answer
```

### Chain-of-Thought (CoT) Prompting

The precursor to reasoning models  simply asking the model to "think step by step":

```
WITHOUT Chain-of-Thought:
  Q: "If a store offers 30% off a $80 jacket, and there's 
      an additional 10% student discount, what's the final price?"
  A: "$50.40" (often wrong  might apply discounts incorrectly)

WITH Chain-of-Thought:
  Q: Same question + "Think step by step."
  A: "Let me work through this:
      1. Original price: $80
      2. 30% discount: $80 × 0.30 = $24 off
      3. Price after first discount: $80 - $24 = $56
      4. 10% student discount on $56: $56 × 0.10 = $5.60
      5. Final price: $56 - $5.60 = $50.40
      
      The final price is $50.40."
      
  (More reliable  can verify each step)
```

#### Why CoT Works

```
Standard generation:
  P(answer | question)  one giant leap

CoT generation:
  P(step1 | question) ×
  P(step2 | question, step1) ×
  P(step3 | question, step1, step2) ×
  P(answer | question, step1, step2, step3)
  
  Breaking the problem into smaller steps makes each
  individual step easier and more likely to be correct.
```

### o1/o3-Style Reasoning Models (OpenAI)

OpenAI's o1 (2024) and o3 (2025) introduced **trained chain-of-thought reasoning**:

```
┌─────────────────────────────────────────────────────────────┐
│            o1/o3 Reasoning Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User: "Prove that √2 is irrational"                        │
│                                                             │
│  ┌─── HIDDEN REASONING CHAIN (internal "thinking") ───┐    │
│  │                                                      │   │
│  │  "I need to prove √2 is irrational. The classic      │   │
│  │   approach is proof by contradiction...               │   │
│  │                                                      │   │
│  │   Assume √2 = p/q where p,q are integers with        │   │
│  │   no common factors.                                  │   │
│  │                                                      │   │
│  │   Then 2 = p²/q², so p² = 2q².                       │   │
│  │   This means p² is even, so p must be even.          │   │
│  │   Let p = 2k. Then (2k)² = 2q², so 4k² = 2q²,      │   │
│  │   meaning q² = 2k², so q is also even.               │   │
│  │                                                      │   │
│  │   But if both p and q are even, they share            │   │
│  │   factor 2, contradicting our assumption.             │   │
│  │                                                      │   │
│  │   Wait, let me double-check the logic...              │   │
│  │   Yes, this is correct. Let me also verify            │   │
│  │   that p being even follows from p² being even...     │   │
│  │   If p were odd, p = 2m+1, then p² = 4m²+4m+1       │   │
│  │   which is odd. Contradiction. ✓                      │   │
│  │                                                      │   │
│  │   The proof is solid."                                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Visible Output: [Clean, polished proof]                    │
│                                                             │
│  Time: 15-60 seconds of "thinking"                          │
│  Tokens: 500-10,000 reasoning tokens (hidden)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### How o1/o3 Are Trained

```
Standard LLM training:
  Input → Output (direct)

Reasoning model training:
  1. Generate many chain-of-thought reasoning traces
  2. Verify which traces lead to correct answers
  3. Train the model to produce good reasoning chains
  4. Use reinforcement learning to optimize for 
     correct final answers through better reasoning

Key technique: "Process reward models" that evaluate
each STEP of reasoning, not just the final answer.

┌─────────────────────────────────────────────────┐
│  Outcome Reward Model (ORM):                     │
│    Only checks: Is final answer correct? ✓/✗     │
│                                                  │
│  Process Reward Model (PRM):                     │
│    Checks each step:                             │
│    Step 1: ✓ correct                             │
│    Step 2: ✓ correct                             │
│    Step 3: ✗ error! ← Catches mistake early      │
│    Step 4: (depends on step 3, also wrong)       │
│                                                  │
│  PRM enables much better reasoning training!     │
└─────────────────────────────────────────────────┘
```

### Test-Time Compute Scaling

A paradigm shift: instead of making models bigger (training-time scaling), make them **think longer** (test-time scaling):

```
Traditional scaling:                Test-time scaling:
  Spend more on TRAINING              Spend more on INFERENCE
  Bigger model → Better answers       Same model + more thinking → Better answers

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Performance                                                 │
│      │                                                       │
│      │                    ╱── More thinking time             │
│      │               ╱───╱                                   │
│      │          ╱───╱                                        │
│      │     ╱───╱                                             │
│      │╱───╱                                                  │
│      │                                                       │
│      └──────────────────────── Compute (test-time)           │
│                                                              │
│  Key insight: You can CHOOSE how hard the model thinks       │
│  based on problem difficulty!                                │
│                                                              │
│  Easy question: "What's 2+2?"    → 10 thinking tokens        │
│  Hard question: "Prove P≠NP"     → 100,000 thinking tokens   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### Techniques for Test-Time Compute Scaling

| Technique | Description | Example |
|-----------|-------------|---------|
| **Extended CoT** | Generate longer reasoning chains | o1, o3 |
| **Best-of-N** | Generate N answers, select best | Early reasoning approaches |
| **Tree search** | Explore multiple reasoning paths | AlphaProof, MCTS-based |
| **Self-consistency** | Sample many CoTs, majority vote on answer | SC-CoT |
| **Iterative refinement** | Generate answer, critique, improve | Self-refine |
| **Verification** | Generate proof, verify with checker | Code: run tests; Math: verify |

### DeepSeek-R1: Open-Source Reasoning

DeepSeek-R1 (January 2025) demonstrated that reasoning capabilities can emerge through **pure reinforcement learning**, without supervised reasoning traces:

```
┌─────────────────────────────────────────────────────────────┐
│                  DeepSeek-R1 Approach                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STAGE 1: Cold Start (small amount of CoT data)             │
│    • Fine-tune on a few thousand reasoning examples         │
│    • Gives model basic reasoning format                     │
│                                                             │
│  STAGE 2: Large-scale RL (GRPO)                             │
│    • Model generates reasoning + answer                     │
│    • Reward = correctness of final answer                   │
│    • NO human-written reasoning traces needed!              │
│    • Model learns to reason by trial and error              │
│                                                             │
│  STAGE 3: Rejection Sampling + SFT                          │
│    • Collect best reasoning traces from RL model            │
│    • Fine-tune new model on these traces                    │
│    • Produces cleaner, more readable reasoning              │
│                                                             │
│  STAGE 4: Second round of RL                                │
│    • Further optimize for helpfulness + accuracy            │
│                                                             │
│  Key discovery: "Aha moment"                                │
│  The model spontaneously develops behaviors like:           │
│  - Backtracking ("Wait, that's wrong, let me reconsider")  │
│  - Self-verification ("Let me check this step")            │
│  - Multiple approaches ("Another way to solve this...")     │
│                                                             │
│  These emerged from RL WITHOUT being explicitly taught!     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### GRPO: Group Relative Policy Optimization

DeepSeek-R1 used GRPO instead of PPO:

```
PPO: Needs a separate value model (critic)  expensive!

GRPO:
  1. For each prompt, generate GROUP of K responses
  2. Score all K responses (correctness, format, etc.)
  3. Compute advantage relative to the group mean
  4. Update policy to favor above-average responses
  
  No value model needed! The group itself serves as baseline.
  
  Example:
  Prompt: "What is 7 × 8?"
  
  Response 1: "56" (correct)     → advantage = +1
  Response 2: "54" (wrong)       → advantage = -1
  Response 3: "56" (correct)     → advantage = +1
  Response 4: "58" (wrong)       → advantage = -1
  
  Update: Increase probability of responses like 1 and 3
```

### Reasoning Model Comparison (2025-2026)

| Model | Creator | Approach | Key Strengths |
|-------|---------|----------|--------------|
| o1 | OpenAI | Trained CoT + RL | Strong math, code, science |
| o3 | OpenAI | Extended o1 approach | SOTA on many benchmarks |
| o4-mini | OpenAI | Efficient reasoning | Fast, cost-effective reasoning |
| Claude 3.5 Sonnet (extended thinking) | Anthropic | Extended thinking mode | Balanced reasoning + helpfulness |
| Gemini 2.5 Pro | Google | Built-in reasoning | Multimodal reasoning |
| DeepSeek-R1 | DeepSeek | RL-based open-source | Fully open, reproducible |
| QwQ | Alibaba | Open-source reasoning | Competitive with R1 |
| Phi-4-reasoning | Microsoft | Small reasoning model | Efficient, compact |

---

## 7. Long-Context Models

### The Memory Challenge

Early transformers were limited to 512-2048 tokens. Modern models handle **1 million+ tokens**  equivalent to reading several full-length novels at once.

### Context Length Evolution

```
2017 │ Transformer    │ 512 tokens      │ ≈ 1 page
2020 │ GPT-3          │ 2,048 tokens    │ ≈ 4 pages
2022 │ GPT-3.5        │ 4,096 tokens    │ ≈ 8 pages
2023 │ Claude 2       │ 100K tokens     │ ≈ 200 pages (a novel!)
2023 │ GPT-4 Turbo    │ 128K tokens     │ ≈ 250 pages
2024 │ Gemini 1.5 Pro │ 1M tokens       │ ≈ 2,000 pages (several novels!)
2024 │ Claude 3.5     │ 200K tokens     │ ≈ 400 pages
2025 │ Gemini 2.5 Pro │ 1M tokens       │ ≈ 2,000 pages
2025 │ Llama 4 Scout  │ 10M tokens      │ ≈ 20,000 pages!
     │                │                 │
     ▼                ▼                 ▼
  Year            Context Length       Human Equivalent
```

### The Technical Challenge: Why Long Context Is Hard

The self-attention mechanism scales **quadratically** with sequence length:

```
Attention complexity: O(n²) where n = sequence length

  n = 1K   →     1M operations     (fast)
  n = 8K   →    64M operations     (manageable)
  n = 128K →    16B operations     (expensive)
  n = 1M   → 1,000B operations    (enormous!)

Memory for KV cache (per layer):
  n = 1K   →    ~16 MB
  n = 128K →  ~2,048 MB (2 GB)
  n = 1M   → ~16,384 MB (16 GB)
  
  × 80 layers = 1.28 TB of KV cache for 1M context!
  
Solutions:
  • GQA (Grouped-Query Attention): Reduce KV heads
  • MLA (Multi-head Latent Attention): Compress KV cache
  • Ring Attention: Distribute attention across GPUs
  • Flash Attention: Memory-efficient attention kernel
  • Sparse Attention: Don't attend to everything
```

### RoPE Scaling: Extending Context at Inference Time

**RoPE** (Rotary Position Embeddings) encodes position information using rotation matrices. The key advantage: RoPE can be **scaled** to support longer contexts than the model was trained on.

#### How RoPE Works (Simplified)

```
Standard positional encoding:
  Position 0: add [0.0, 1.0, 0.0, 1.0, ...]
  Position 1: add [0.84, 0.54, 0.01, 0.99, ...]
  Fixed vectors  can't extend beyond training length

RoPE:
  Position 0: rotate by 0°
  Position 1: rotate by θ
  Position 2: rotate by 2θ
  Position n: rotate by nθ
  
  Key insight: The RELATIVE rotation between positions
  i and j only depends on (i-j), not absolute position.
  This makes it naturally extrapolable!
```

#### RoPE Scaling Techniques

| Technique | How It Works | Typical Extension |
|-----------|-------------|-------------------|
| **Position Interpolation** (PI) | Scale positions: pos' = pos × (L_train / L_target) | 2-8× |
| **NTK-aware scaling** | Scale the base frequency θ | 4-16× |
| **YaRN** | Combines NTK + PI + attention scaling | 8-128× |
| **Dynamic NTK** | Adjust scaling dynamically based on sequence length | 4-32× |
| **Long RoPE** (Microsoft) | Non-uniform scaling of different frequency bands | 8-2048× |
| **Continuous pretraining** | Train on longer sequences (expensive but robust) | Any |

```
RoPE Position Interpolation:

Original (trained on 4K tokens):
  Positions: 0, 1, 2, 3, ..., 4095
  Rotations: 0θ, 1θ, 2θ, ..., 4095θ

Interpolated (extend to 32K tokens):
  Positions: 0, 1, 2, 3, ..., 32767
  Scaled:    0, 1/8, 2/8, 3/8, ..., 4095.875
  Rotations: 0θ, θ/8, 2θ/8, ..., 4095.875θ

  All rotations stay within the range the model saw during
  training, so attention patterns still work!
  
  Trade-off: Reduced position resolution (positions are
  "squeezed together"), may hurt fine-grained position tasks
```

### Long Context vs RAG: When to Use Each

```
┌─────────────────────────────────────────────────────────────────┐
│              Long Context vs RAG                                 │
├────────────────────┬────────────────────────────────────────────┤
│                    │                                            │
│  Long Context      │  RAG                                       │
│  ┌────────────┐    │  ┌────────────┐                            │
│  │ Stuff all  │    │  │ Search for │                            │
│  │ documents  │    │  │ relevant   │                            │
│  │ into the   │    │  │ chunks     │                            │
│  │ prompt     │    │  │ only       │                            │
│  └────────────┘    │  └────────────┘                            │
│                    │                                            │
│  PROS:             │  PROS:                                     │
│  • Simple          │  • Scales to millions of documents         │
│  • No pipeline     │  • Cheaper per query                       │
│  • Global context  │  • Deterministic retrieval                 │
│  • Better for      │  • Works with any model                    │
│    cross-document  │  • Updatable without re-processing         │
│    reasoning       │                                            │
│                    │  CONS:                                     │
│  CONS:             │  • Retrieval can miss relevant info        │
│  • Expensive per   │  • Complex pipeline                        │
│    query           │  • Chunk boundary issues                   │
│  • Limited by      │  • Can't reason across chunks              │
│    context window  │    easily                                  │
│  • "Lost in the    │                                            │
│    middle" effect  │                                            │
│  • Higher latency  │                                            │
│                    │                                            │
└────────────────────┴────────────────────────────────────────────┘
```

#### The "Lost in the Middle" Problem

Research showed that LLMs pay more attention to information at the **beginning and end** of the context, and may miss information in the **middle**:

```
Attention Distribution Over Long Context:

High  │ ████                                    ████
      │ ████                                    ████
      │ ████                                    ████
      │ ████ █                                █ ████
      │ ████ ██                              ██ ████
      │ ████ ███                            ███ ████
Low   │ ████ █████ ██ ██ ██ ██ ██ ██ ██ ██ ████ ████
      └──────────────────────────────────────────────
       Start          Middle                    End
       
  The model "forgets" information in the middle!
  
  Mitigations:
  1. Put important info at start and end
  2. Use RAG to surface only the relevant pieces  
  3. Newer models (2025+) are better at this
  4. Explicitly prompt: "Pay attention to ALL context"
```

#### Decision Framework: Context vs RAG

| Scenario | Best Approach | Why |
|----------|--------------|-----|
| Analyzing a single 100-page document | Long Context | Need global understanding |
| Searching across 10,000 documents | RAG | Too much for any context window |
| Comparing 5 contracts | Long Context | Need cross-document reasoning |
| Customer support with large KB | RAG | Knowledge base is huge and dynamic |
| Summarizing a codebase | Long Context | Need full codebase understanding |
| Answering questions about product catalog | RAG | Catalog is large, queries are specific |
| Legal document review | Both | RAG to find relevant docs, long context to analyze them |

---

## 8. Multimodal Models

### Beyond Text: Processing the Full Spectrum

**Multimodal models** process and generate multiple types of data  text, images, audio, video  in a unified architecture.

### The Human Brain Analogy

```
Human brain processes multiple modalities simultaneously:
  👁️ Vision  → See a recipe photo
  👂 Hearing → Listen to cooking instructions  
  📝 Text    → Read ingredient list
  🧠 Understanding → Combine all to cook the meal

Multimodal AI does the same:
  🖼️ Image encoder → Understand photos
  🎵 Audio encoder → Understand speech/music
  📹 Video encoder → Understand video
  📝 Text decoder  → Generate text responses
  🧠 LLM backbone  → Reason across all modalities
```

### Architecture Approaches

There are several ways to build multimodal models:

#### Approach 1: Modality-Specific Encoders + LLM

```
┌─────────────────────────────────────────────────────────────┐
│         Encoder-Fusion Architecture (most common)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Image:  ┌────────────┐   ┌───────────┐                    │
│  🖼️ ───→ │ Vision     │──→│ Projection│──→ Visual tokens    │
│          │ Encoder    │   │ Layer     │    [v1, v2, v3...]  │
│          │(ViT/SigLIP)│   └───────────┘                    │
│          └────────────┘                                     │
│                                                             │
│  Audio:  ┌────────────┐   ┌───────────┐                    │
│  🎵 ───→ │ Audio      │──→│ Projection│──→ Audio tokens     │
│          │ Encoder    │   │ Layer     │    [a1, a2, a3...]  │
│          │(Whisper)   │   └───────────┘                    │
│          └────────────┘                                     │
│                                                             │
│  Text:   ┌────────────┐                                     │
│  📝 ───→ │ Tokenizer  │──────────────────→ Text tokens      │
│          └────────────┘                    [t1, t2, t3...]  │
│                                                             │
│  All tokens fed into LLM:                                   │
│  [v1, v2, v3, ..., a1, a2, ..., t1, t2, t3, ...]          │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────┐                                    │
│  │    LLM Backbone      │                                    │
│  │  (Transformer        │                                    │
│  │   Decoder)           │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│       Text response                                         │
│                                                             │
│  Examples: LLaVA, GPT-4V, Gemini, Claude 3.5               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Approach 2: Early Fusion (Native Multimodal)

```
┌─────────────────────────────────────────────────────────────┐
│         Early Fusion Architecture                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ALL modalities tokenized into a SINGLE token space:        │
│                                                             │
│  Image → patches → discrete tokens: [IMG_42, IMG_187, ...] │
│  Audio → frames → discrete tokens: [AUD_7, AUD_255, ...]   │
│  Text  → BPE → tokens: [THE, CAT, SAT, ...]                │
│                                                             │
│  Interleaved sequence:                                      │
│  [THE, IMG_42, IMG_187, CAT, AUD_7, SAT, ...]              │
│                                                             │
│  ┌─────────────────────────────────────────┐                │
│  │  Single Transformer processes ALL       │                │
│  │  modalities as one unified sequence     │                │
│  │  No separate encoders needed!           │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
│  Advantages: True cross-modal reasoning                     │
│  Challenges: Requires massive training data and compute     │
│                                                             │
│  Examples: Gemini (partially), Chameleon (Meta)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Approach 3: Diffusion + LLM Hybrid

```
┌─────────────────────────────────────────────────────────────┐
│         LLM + Diffusion Model Hybrid                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For GENERATING images/audio (not just understanding):      │
│                                                             │
│  User: "Draw a sunset over mountains"                       │
│       │                                                     │
│       ▼                                                     │
│  ┌────────────┐                                             │
│  │    LLM      │  → Understands intent                      │
│  │             │  → Generates detailed prompt               │
│  └──────┬─────┘     for diffusion model                     │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │  Diffusion    │  → Generates image from LLM's prompt     │
│  │  Model        │                                          │
│  │  (DALL·E,     │                                          │
│  │   Imagen)     │                                          │
│  └──────────────┘                                           │
│                                                             │
│  Examples: GPT-4o (native image gen), Gemini                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Vision-Language Models (VLMs)

VLMs are the most mature multimodal models:

#### How Vision Encoding Works

```
Input Image (224×224 pixels)
         │
         ▼
┌─────────────────────────┐
│  Split into patches      │
│  (16×16 pixels each)     │
│  = 14 × 14 = 196 patches │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Vision Transformer      │
│  (ViT / SigLIP)         │
│                         │
│  Each patch → embedding  │
│  196 patch embeddings    │
│  + self-attention        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Projection Layer        │
│  (Linear or MLP)         │
│                         │
│  Map vision embeddings   │
│  into LLM's embedding   │
│  space                   │
└────────┬────────────────┘
         │
         ▼
  196 visual tokens that the LLM
  can "read" like text tokens
```

#### Key VLM Capabilities (2025-2026)

| Capability | Example | Notable Models |
|-----------|---------|----------------|
| **Image understanding** | "What's in this photo?" | GPT-4o, Claude 3.5, Gemini 2.5 |
| **OCR / document reading** | Read text from images, PDFs | GPT-4o, Gemini, Claude |
| **Visual reasoning** | "Count the red objects" | Gemini 2.5 Pro, GPT-4o |
| **Chart/graph analysis** | Interpret data visualizations | Claude 3.5, GPT-4o |
| **Image generation** | "Create an image of..." | GPT-4o, Gemini Imagen 3 |
| **Image editing** | "Remove the background" | GPT-4o, Gemini |
| **Multi-image comparison** | "How do these two images differ?" | Gemini, Claude 3.5 |
| **Spatial understanding** | "What's to the left of the car?" | GPT-4o, Gemini 2.5 |

### Audio Models

```
┌─────────────────────────────────────────────────────────────┐
│                  Audio Model Capabilities                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SPEECH-TO-TEXT (Understanding):                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Audio    │───→│ Whisper  │───→│  Text    │              │
│  │  Input    │    │ Encoder  │    │  Output  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│  Example: "Hello, how are you?" → transcription             │
│                                                             │
│  TEXT-TO-SPEECH (Generation):                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Text    │───→│  TTS     │───→│  Audio   │              │
│  │  Input   │    │  Model   │    │  Output  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│  Example: text → natural speech (ElevenLabs, VALL-E)        │
│                                                             │
│  NATIVE AUDIO LLMs (Both):                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Audio   │◄──→│   LLM    │◄──→│  Audio   │              │
│  │  Input   │    │ (native) │    │  Output  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│  Example: GPT-4o Advanced Voice  real-time conversation    │
│           with emotion, interruption, multiple voices       │
│                                                             │
│  Key models:                                                │
│  • Whisper v3 (OpenAI): SOTA open-source transcription      │
│  • GPT-4o Voice: Real-time voice conversations              │
│  • Gemini 2.5 Live: Streaming audio understanding           │
│  • Sesame / CSM: Open-source conversational speech model    │
│  • Kokoro: Lightweight open-source TTS                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Video Understanding

Video understanding is the frontier of multimodal AI:

```
Video = Images + Audio + Time

Challenges:
  1. ENORMOUS data: 30 fps × 1080p = massive input
  2. Temporal reasoning: understanding sequences of events
  3. Long videos: a 2-hour movie = millions of visual tokens

Approaches:
  ┌─────────────────────────────────────────────────┐
  │  1. FRAME SAMPLING                               │
  │     Select key frames (1-4 fps instead of 30)    │
  │     Process each as an image                     │
  │     + Temporal position encoding                 │
  │                                                  │
  │  2. VIDEO TOKENS                                 │
  │     3D patches: spatial + temporal               │
  │     Like ViT but extends across time             │
  │                                                  │
  │  3. VIDEO ENCODERS                               │
  │     Specialized architectures: ViViT, TimeSformer│
  │     Compress temporal information                │
  │                                                  │
  │  4. HIERARCHICAL                                 │
  │     Frame-level → clip-level → video-level       │
  │     Progressive abstraction                      │
  └─────────────────────────────────────────────────┘

State of the art (2025-2026):
  • Gemini 2.5 Pro: Process up to hours of video
  • GPT-4o: Video understanding via frame sampling
  • Claude 3.5 Sonnet: Image sequences (frames)
  • Sora (OpenAI): Video generation (separate domain)
  • Veo 2 (Google): High-quality video generation
```

### Multimodal Architecture Comparison

| Approach | Input Modalities | Output Modalities | Key Advantage | Key Limitation |
|----------|-----------------|-------------------|---------------|----------------|
| **Encoder fusion** (LLaVA) | Image + Text | Text | Easy to add modalities | Can't generate images |
| **Early fusion** (Gemini) | Image + Audio + Text | Text + Image | Deep cross-modal reasoning | Huge training cost |
| **LLM + Diffusion** (GPT-4o) | Text + Image | Text + Image | High quality generation | Complex pipeline |
| **Any-to-any** (Meta Chameleon) | Any | Any | Maximum flexibility | Hardest to train |

---

## 9. Architecture Comparison

### The Master Comparison Table

| Architecture | Best For | Scale | Latency | Cost | Complexity |
|-------------|---------|-------|---------|------|-----------|
| **Dense Transformer** | Small/medium models, edge | 1B-70B | Low | Low-Medium | Low |
| **MoE** | Frontier models, efficient large models | 50B-2T+ | Low-Medium | Medium | High |
| **RAG** | Knowledge-intensive, current info | Any model | Medium-High | Medium | Medium |
| **Tool-Using Agents** | Action-oriented tasks | Any model | High | High | High |
| **Reasoning Models** | Complex math, code, logic | 7B-400B+ | Very High | Very High | Medium-High |
| **Long-Context** | Document analysis, code review | 8B-400B+ | High | High | Medium |
| **Multimodal** | Image/audio/video understanding | 3B-400B+ | Medium-High | High | Very High |

### Decision Flowchart

```
START: What do you need?
  │
  ├─ "Understand/generate images/audio/video"
  │   └─→ MULTIMODAL MODEL (GPT-4o, Gemini 2.5 Pro, Claude 3.5)
  │
  ├─ "Solve complex math/reasoning problems"
  │   └─→ REASONING MODEL (o3, DeepSeek-R1, Gemini 2.5 Pro)
  │
  ├─ "Take actions in the world (search, APIs, code execution)"
  │   └─→ TOOL-USING AGENT (with function calling / MCP)
  │
  ├─ "Answer questions about my private/current data"
  │   └─→ RAG SYSTEM (embedding + vector DB + LLM)
  │
  ├─ "Process very long documents (>100 pages)"
  │   └─→ LONG-CONTEXT MODEL (Gemini 2.5 Pro 1M, Llama 4 Scout)
  │
  ├─ "Deploy on edge/mobile with tight latency"
  │   └─→ SMALL DENSE MODEL (Phi-4, Gemma 2, Llama 3.1 8B)
  │
  └─ "Maximum capability, cost is secondary"
      └─→ MoE FRONTIER MODEL (GPT-4o, Claude 3.5 Opus,
                                Gemini 2.5 Pro, DeepSeek-V3)
```

### Real-World System Examples

Most production AI systems **combine** multiple architectures:

```
┌─────────────────────────────────────────────────────────────┐
│  Example: AI Coding Assistant (like Cursor, Copilot)         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Components used:                                           │
│  ✓ Long-Context Model: Read entire codebase                │
│  ✓ RAG: Search documentation and code patterns             │
│  ✓ Tool-Using Agent: Run code, tests, linting              │
│  ✓ Reasoning Model: Complex debugging and architecture     │
│  ✓ Fast Dense Model: Inline autocomplete suggestions       │
│                                                             │
│  Flow:                                                      │
│  1. User asks: "Fix the failing test in auth.py"           │
│  2. RAG retrieves relevant code files and docs             │
│  3. Long-context loads the full module                     │
│  4. Reasoning model analyzes the bug                       │
│  5. Agent runs the test to verify the fix                  │
│  6. Dense model provides autocomplete while editing        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Example: Enterprise Customer Support Bot                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Components used:                                           │
│  ✓ RAG: Search knowledge base for answers                  │
│  ✓ Multimodal: Understand customer screenshots             │
│  ✓ Tool-Using Agent: Look up account info, create tickets  │
│  ✓ Small Dense Model: Quick FAQ responses                  │
│                                                             │
│  Flow:                                                      │
│  1. Customer sends image of error message                  │
│  2. Multimodal model reads the error                       │
│  3. RAG searches knowledge base for solutions              │
│  4. Agent checks customer's account status                 │
│  5. If resolved: Small model generates response            │
│     If not: Agent creates support ticket                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Cost-Performance Trade-offs (2025-2026 Pricing)

| Model/Config | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) | Speed | Quality |
|-------------|---------------------------|----------------------------|-------|---------|
| GPT-4o-mini | ~$0.15 | ~$0.60 | Fast | Good |
| GPT-4o | ~$2.50 | ~$10.00 | Medium | Very Good |
| o3-mini | ~$1.10 | ~$4.40 | Slow | Excellent (reasoning) |
| Claude 3.5 Sonnet | ~$3.00 | ~$15.00 | Medium | Very Good |
| Gemini 2.5 Pro | ~$1.25-2.50 | ~$10.00-15.00 | Medium | Excellent |
| Gemini 2.5 Flash | ~$0.15 | ~$0.60 | Very Fast | Good |
| DeepSeek-V3 (API) | ~$0.27 | ~$1.10 | Fast | Very Good |
| DeepSeek-R1 (API) | ~$0.55 | ~$2.19 | Slow | Excellent (reasoning) |
| Llama 3.1 70B (self-hosted) | ~$0.50* | ~$0.50* | Medium | Good |

*Self-hosted costs include GPU rental but no per-token API fee.

---

## 10. Quiz

### Questions

**Q1:** In a Mixture of Experts model with 256 experts where 8 are active per token, and total parameters are 671B, approximately how many parameters are active per token?

- A) 671B (all of them)
- B) ~21B
- C) ~37B
- D) 256B

**Q2:** What is the primary advantage of RAG over fine-tuning for knowledge-intensive tasks?

- A) RAG is always faster
- B) RAG can use current information and provide citations
- C) RAG produces better-quality text
- D) RAG requires no LLM

**Q3:** In the ReAct pattern, what does the agent alternate between?

- A) Reading and writing
- B) Encoding and decoding
- C) Thinking (reasoning) and acting (using tools)
- D) Training and inference

**Q4:** What is "test-time compute scaling" in reasoning models?

- A) Using bigger GPUs during inference
- B) Spending more compute on "thinking" during inference to improve answer quality
- C) Scaling the model size at test time
- D) Running multiple models simultaneously

**Q5:** Why is the "Lost in the Middle" effect a concern for long-context models?

- A) The model loses parameters in the middle layers
- B) The model pays less attention to information in the middle of long contexts
- C) The middle tokens are dropped during processing
- D) The model runs out of memory for middle tokens

**Q6:** What is MCP (Model Context Protocol)?

- A) A method for compressing model context
- B) A standardized protocol for connecting AI models to external tools and data sources
- C) A type of model architecture
- D) A training technique for multimodal models

**Q7:** How does DeepSeek-R1 develop reasoning capabilities?

- A) By training on millions of human-written reasoning traces
- B) Primarily through reinforcement learning (GRPO), where the model learns to reason by trial and error
- C) By copying OpenAI's o1 architecture
- D) By using a reasoning-specific transformer architecture

**Q8:** What is the key innovation of MLA (Multi-head Latent Attention) used in DeepSeek?

- A) It adds more attention heads
- B) It compresses the KV cache dramatically, enabling larger batch sizes
- C) It removes attention entirely
- D) It uses attention only on the first layer

**Q9:** Which approach to multimodal AI uses separate specialized encoders for each modality (image, audio) and feeds their outputs into a shared LLM?

- A) Early fusion
- B) Encoder-fusion architecture
- C) Any-to-any generation
- D) Diffusion-only architecture

**Q10:** When would you choose a small dense model (e.g., Phi-4 14B) over a large MoE model (e.g., DeepSeek-V3 671B)?

- A) When you need maximum accuracy
- B) When deploying on edge devices, need low latency, or have limited GPU memory
- C) When processing very long documents
- D) When you need multilingual support

### Answers

**A1:** **C) ~37B.** DeepSeek-V3 has 671B total parameters but only activates ~37B per token. This is because only 8 out of 256 experts plus the shared expert and attention layers are active for each token. This gives GPT-4-level knowledge with much faster inference.

**A2:** **B) RAG can use current information and provide citations.** RAG retrieves from an updateable knowledge base, so it always has the latest information. It can also point to specific source documents, which is critical for enterprise and factual applications. Fine-tuning bakes knowledge into model weights, making it static.

**A3:** **C) Thinking (reasoning) and acting (using tools).** ReAct stands for "Reason + Act." The agent generates a thought (THOUGHT), takes an action (ACTION), observes the result (OBSERVATION), and repeats until it has enough information to provide a final answer.

**A4:** **B) Spending more compute on "thinking" during inference to improve answer quality.** Instead of making models bigger (training-time scaling), reasoning models can think for longer on harder problems, using more tokens of chain-of-thought reasoning. This is a paradigm shift from "bigger model = better" to "more thinking = better."

**A5:** **B) The model pays less attention to information in the middle of long contexts.** Research showed that LLMs have a U-shaped attention pattern  they attend well to the beginning and end of the context but may miss crucial information in the middle. This makes context placement strategy important for long-context applications.

**A6:** **B) A standardized protocol for connecting AI models to external tools and data sources.** MCP, introduced by Anthropic and widely adopted by 2025-2026, acts like "USB-C for AI"  providing a universal standard for tools, resources, and prompts that any AI application can connect to.

**A7:** **B) Primarily through reinforcement learning (GRPO), where the model learns to reason by trial and error.** DeepSeek-R1 showed that reasoning can emerge from RL alone (with minimal supervised seeding). The model spontaneously developed behaviors like backtracking, self-verification, and trying multiple approaches  without being explicitly taught these strategies.

**A8:** **B) It compresses the KV cache dramatically, enabling larger batch sizes.** MLA reduces KV cache by 93.3% compared to standard multi-head attention by projecting keys and values into a lower-dimensional latent space. This is crucial for efficient inference at scale, especially for large MoE models.

**A9:** **B) Encoder-fusion architecture.** This approach uses specialized encoders (e.g., ViT for images, Whisper for audio) to process each modality, then projects these representations into the LLM's embedding space. Models like LLaVA, GPT-4V, and Claude 3.5 use variants of this approach.

**A10:** **B) When deploying on edge devices, need low latency, or have limited GPU memory.** Small dense models are simpler to deploy, quantize, and optimize. They fit on a single GPU, have predictable latency, and are easier to fine-tune. MoE models require more memory (all experts must be loaded) even though they compute fewer parameters per token.

---

## Summary

The modern AI landscape is rich and diverse. Here's what to remember:

| Architecture | Key Insight |
|-------------|-------------|
| **Dense** | Simple, reliable, best for small-medium scale |
| **MoE** | More knowledge per FLOP  the frontier architecture |
| **RAG** | External knowledge > baked-in knowledge for current/private data |
| **Agents** | LLMs can take actions, not just generate text |
| **Reasoning** | Thinking longer > thinking faster for hard problems |
| **Long-Context** | Context windows have grown 2000× in 3 years |
| **Multimodal** | The future is models that see, hear, and speak |

The most powerful AI systems of 2026 combine multiple architectures  an MoE backbone with multimodal inputs, RAG for knowledge, tool use for actions, and reasoning for complex problems. Understanding each piece helps you design the right system for your needs.

