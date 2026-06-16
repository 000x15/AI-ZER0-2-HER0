# Part 4: Transformers  The Architecture That Changed Everything

> *"Attention is all you need."  Vaswani et al., 2017*
>
> *This single paper may be the most consequential machine learning publication of the 21st century.*

---

## Table of Contents

1. [Why Transformers Replaced RNNs](#1-why-transformers-replaced-rnns)
2. [The "Attention Is All You Need" Paper  Context and Impact](#2-the-attention-is-all-you-need-paper--context-and-impact)
3. [Embeddings  How Words Become Vectors](#3-embeddings--how-words-become-vectors)
4. [Positional Encoding  Teaching Order to an Orderless Architecture](#4-positional-encoding--teaching-order-to-an-orderless-architecture)
5. [The Attention Mechanism  From Scratch](#5-the-attention-mechanism--from-scratch)
6. [Multi-Head Attention](#6-multi-head-attention)
7. [The Encoder Architecture](#7-the-encoder-architecture)
8. [The Decoder Architecture](#8-the-decoder-architecture)
9. [Encoder-Decoder Architecture (Full Transformer)](#9-encoder-decoder-architecture-full-transformer)
10. [Decoder-Only Architecture (GPT-Style)](#10-decoder-only-architecture-gpt-style)
11. [Step-by-Step Token Processing: How "The cat sat" Predicts "on"](#11-step-by-step-token-processing-how-the-cat-sat-predicts-on)
12. [KV-Cache and Inference Optimization](#12-kv-cache-and-inference-optimization)
13. [Modern Transformer Variants (2024-2026)](#13-modern-transformer-variants-2024-2026)
14. [Quiz](#14-quiz)

---

## 1. Why Transformers Replaced RNNs

Before the Transformer, RNNs (and their variants LSTMs/GRUs) were the go-to architecture for sequence processing  language translation, text generation, speech recognition. They worked, but they had two fundamental problems that became increasingly painful as models scaled up.

### Problem 1: Sequential Processing Kills Parallelism

```
RNN PROCESSING:  (must go one step at a time)

  "The"  →  RNN  →  "cat"  →  RNN  →  "sat"  →  RNN  →  "on"  →  RNN
  Step 1     ↓       Step 2     ↓       Step 3     ↓       Step 4     ↓
             h₁                 h₂                 h₃                 h₄

  Total time = 4 × (time per step)
  GPUs with 10,000 cores sit mostly idle  only 1 step runs at a time!

TRANSFORMER PROCESSING:  (all tokens in parallel!)

  "The"  ──────►  ┌─────────────────┐  ──────►  output₁
  "cat"  ──────►  │   TRANSFORMER   │  ──────►  output₂
  "sat"  ──────►  │   (parallel!)   │  ──────►  output₃
  "on"   ──────►  └─────────────────┘  ──────►  output₄

  Total time = 1 × (time for all tokens together)
  ALL GPU cores working simultaneously!
```

This is not a minor difference. Training a large language model with RNNs would take **months or years** on modern hardware. With Transformers, it takes **weeks**. The parallelism of Transformers is what made models with billions of parameters feasible.

### Problem 2: Long-Range Dependencies

Even LSTMs struggle with very long sequences. Information must pass through hundreds of sequential steps, degrading along the way:

```
LSTM: Information must traverse every intermediate step

  "The  president  of  the  United  States  of  America  ...  signed"
   t=1     t=2    t=3  t=4    t=5     t=6   t=7    t=8  ... t=50

  Distance from "president" to "signed" = 48 steps
  Information about "president" is filtered through 48 gate operations!

TRANSFORMER: Direct connection between any two tokens

  "The  president  of  the  United  States  ...  signed"
         │                                        │
         └──────── DIRECT ATTENTION ──────────────┘

  Distance from "president" to "signed" = 1 attention step!
  No matter how far apart, any two tokens can "talk" directly.
```

### The Paradigm Shift

| Aspect | RNN/LSTM | Transformer |
|--------|----------|-------------|
| Processing | Sequential | Parallel |
| Long-range deps | Degrades with distance | Constant (direct attention) |
| Training speed | Slow (can't parallelize over time) | Fast (full parallelism) |
| Max practical length | ~1000 tokens | 4K-128K+ tokens (2026) |
| GPU utilization | Poor | Excellent |
| Scalability | Hits a wall | Scales with compute (scaling laws) |

---

## 2. The "Attention Is All You Need" Paper  Context and Impact

### The State of NLP in 2017

Before the Transformer paper, the best machine translation systems used **encoder-decoder RNNs with attention** (Bahdanau et al., 2014). The architecture looked like:

```
2014-2017 STATE OF THE ART:

  Source sentence ──► [LSTM Encoder] ──► context ──► [LSTM Decoder] ──► Translation
                          │                              │
                          └─── attention mechanism ──────┘
                               (added in 2014)
```

The attention mechanism (added by Bahdanau) was already a breakthrough  it let the decoder "look back" at specific parts of the encoder's output. But the underlying architecture was still sequential RNNs.

### The Key Question

A team at Google Brain (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Łukasz Kaiser, and Illia Polosukhin) asked a radical question:

> **"What if we got rid of recurrence entirely and built the whole model from just attention?"**

The result  published as "Attention Is All You Need" at NeurIPS 2017  didn't just improve translation. It created a new architecture that would dominate **all of AI** within a few years.

### Impact Timeline

```
2017: "Attention Is All You Need"  Transformer for translation
2018: BERT (encoder-only)  NLP benchmarks destroyed
2018: GPT-1 (decoder-only)  generative language modeling
2019: GPT-2  "too dangerous to release" (1.5B parameters)
2020: GPT-3  few-shot learning emerges (175B parameters)
2020: Vision Transformer (ViT)  Transformers conquer computer vision
2021: DALL-E, Codex  images and code from text
2022: ChatGPT  Transformers go mainstream
2023: GPT-4  multimodal, professional-level reasoning
2024: Claude 3, Gemini, Llama 3  open and closed frontier models
2025: Claude 4, GPT-5, Gemini 2.5  extended reasoning, massive context
2026: Every major AI system is built on Transformers (or descendants)
```

---

## 3. Embeddings  How Words Become Vectors

Neural networks work with numbers, not words. We need a way to convert words (or tokens) into numerical vectors that capture meaning. This is the role of **embeddings**.

### 3.1 The Problem with Naive Representations

**One-hot encoding** assigns each word a unique binary vector:

```
Vocabulary: [cat, dog, kitten, fish, car]

cat    = [1, 0, 0, 0, 0]
dog    = [0, 1, 0, 0, 0]
kitten = [0, 0, 1, 0, 0]
fish   = [0, 0, 0, 1, 0]
car    = [0, 0, 0, 0, 1]

Problem 1: "cat" and "kitten" are just as different as "cat" and "car"!
           dot(cat, kitten) = 0
           dot(cat, car) = 0
           (all pairs are equally distant  no semantic information)

Problem 2: With a vocabulary of 50,000 words, each vector has 50,000 dimensions!
           (sparse and wasteful)
```

### 3.2 Word2Vec Intuition

**Word2Vec** (Mikolov et al., 2013) discovered that you could learn dense, meaningful word vectors from raw text by training on a simple task: **predict a word from its neighbors** (or vice versa).

**The key insight:** Words that appear in similar contexts have similar meanings.

```
Training data (from real text):
  "The cat sat on the mat"     → context of "sat": [cat, on]
  "The dog sat on the rug"     → context of "sat": [dog, on]
  "The cat lay on the mat"     → context of "lay": [cat, on]

Because "cat" and "dog" appear in similar contexts:
  → They get SIMILAR embeddings!

Learned embeddings (hypothetical 4D):
  cat    = [0.7,  0.2, -0.1,  0.8]
  kitten = [0.6,  0.3, -0.2,  0.9]    ← similar to cat!
  dog    = [0.8,  0.1,  0.0,  0.7]    ← similar to cat!
  car    = [-0.3, 0.9,  0.5, -0.2]    ← very different
```

### 3.3 The Famous Arithmetic of Embeddings

Word2Vec revealed that embedding space captures **semantic relationships** as geometric directions:

```
WORD EMBEDDING ARITHMETIC:

  king - man + woman ≈ queen

  ┌──────────────────────────────────────────┐
  │         ▲ "royalty"                      │
  │         │                                │
  │   king  ●─────────────────● queen        │
  │         │    "gender"     │              │
  │         │   direction →   │              │
  │         │                 │              │
  │   man   ●─────────────────● woman        │
  │         │                 │              │
  │         │                                │
  └──────────────────────────────────────────┘

  vector(king) - vector(man) ≈ vector("royalty")
  vector("royalty") + vector(woman) ≈ vector(queen)

Other examples:
  Paris - France + Italy ≈ Rome
  walked - walking + swimming ≈ swam
```

### 3.4 Token Embeddings in Transformers

Modern Transformers don't use Word2Vec directly. Instead, they learn their own embeddings as part of end-to-end training. The embedding is simply a **lookup table** (implemented as a matrix multiplication with one-hot vectors):

```
EMBEDDING LOOKUP:

Vocabulary size: V = 50,000
Embedding dimension: d = 768 (e.g., BERT-base)

Embedding Matrix E: (V × d) = 50,000 × 768

Token "cat" has ID 4523:

  one_hot(4523) × E = E[4523] = [0.12, -0.34, 0.56, ..., 0.78]
                                  └────── 768 dimensions ──────┘

This 768-dimensional vector IS the embedding of "cat."
During training, these values are updated by backpropagation
to capture whatever meanings are useful for the task.
```

### 3.5 Subword Tokenization

Modern models don't embed whole words  they use **subword tokens** (BPE, WordPiece, SentencePiece):

```
Word-level:      "unhappiness" → ["unhappiness"]        (one token)
                 "unforgettable" → ["unforgettable"]     (one token)
                 "supercalifragilistic" → [UNKNOWN]      (not in vocab!)

Subword (BPE):   "unhappiness" → ["un", "happiness"]    (two tokens)
                 "unforgettable" → ["un", "forget", "table"] (three tokens)
                 "supercalifragilistic" → ["super", "cal", "if", "rag", "il", "istic"]

Benefits:
  ✓ Handles any word (no UNKNOWN tokens)
  ✓ Smaller vocabulary (32K-100K tokens covers all text)
  ✓ Shares knowledge between related words ("un-" prefix)
  ✓ Works across languages
```

---

## 4. Positional Encoding  Teaching Order to an Orderless Architecture

### 4.1 The Problem

Attention (which we'll cover next) treats its input as a **set**, not a sequence. It computes relationships between all pairs of tokens simultaneously, with no notion of order. But word order matters enormously:

```
"The dog bit the man"     ← Very different meaning!
"The man bit the dog"     ← Same words, different order!

Without positional information, both sentences produce
IDENTICAL attention patterns  the Transformer can't
distinguish them!
```

### 4.2 The Solution: Add Position Information to Embeddings

We create a **positional encoding** vector for each position and **add** it to the token embedding:

```
INPUT REPRESENTATION:

  Token embedding  +  Positional encoding  =  Input to Transformer

  "The" embedding     Position 0 encoding     Combined
  [0.5, 0.2, -0.1] + [0.0, 1.0, 0.0]       = [0.5, 1.2, -0.1]

  "cat" embedding     Position 1 encoding     Combined
  [0.7, 0.3, 0.4]  + [0.84, 0.54, 0.84]     = [1.54, 0.84, 1.24]

  "sat" embedding     Position 2 encoding     Combined
  [0.1, 0.8, -0.3] + [0.91, -0.42, 0.91]    = [1.01, 0.38, 0.61]
```

### 4.3 Sinusoidal Positional Encoding (Original Transformer)

The original paper used sine and cosine functions of different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
  pos = position in the sequence (0, 1, 2, ...)
  i = dimension index (0, 1, 2, ..., d_model/2 - 1)
  d_model = embedding dimension (e.g., 512)
```

**Why sines and cosines?**

```
INTUITION: Each dimension is a "clock" ticking at a different frequency

Dimension 0-1: Fast clock     ─╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱─   (changes every position)
Dimension 2-3: Medium clock   ──╲──╱──╲──╱──╲──╱──   (changes every ~4 positions)
Dimension 4-5: Slow clock     ────╲────╱────╲────╱─   (changes every ~16 positions)
  ...
Dimension 510-511: Very slow  ──────────╲──────────╱   (changes over thousands)

Position in sequence →

Like binary encoding, but smooth! Position 0 and Position 1 differ in
the fast dimensions. Position 0 and Position 100 differ in the slow ones.
Every position gets a unique combination of values.
```

**Key properties:**
1. **Unique encoding** for every position
2. **Relative positions are linear transformations**  the model can learn to compute "how far apart are positions i and j?" as a simple linear operation on their encodings (PE(pos+k) can be represented as a linear function of PE(pos))
3. **Generalizes to longer sequences**  since sin/cos are defined for all positions, the encoding works for sequences longer than those seen during training (in theory)

### 4.4 Learned Positional Embeddings (Modern Approach)

Many modern Transformers (BERT, GPT-2) just **learn** positional embeddings as trainable parameters:

```
Position 0: [p₀₀, p₀₁, p₀₂, ..., p₀₇₆₇]    ← learned by backpropagation
Position 1: [p₁₀, p₁₁, p₁₂, ..., p₁₇₆₇]    ← learned by backpropagation
  ...
Position 2047: [p₂₀₄₇,₀, ..., p₂₀₄₇,₇₆₇]   ← max position (context length)
```

**Trade-off:** Learned embeddings often perform slightly better within the training range but cannot generalize to positions beyond what was trained on.

### 4.5 RoPE  Rotary Position Embedding (Modern Standard, 2026)

**RoPE** (Su et al., 2021) is the dominant positional encoding in modern LLMs (LLaMA, Mistral, Qwen, Gemma, etc.). Instead of adding position to the embeddings, it **rotates** query and key vectors based on position:

```
ROPE INTUITION:

Instead of:  embedding + position_encoding
RoPE does:   rotate(embedding, angle = position × frequency)

The rotation angle increases with position, at different rates
for different dimension pairs. This naturally encodes relative
position: the dot product between two rotated vectors depends
only on their RELATIVE position difference, not absolute positions.

Key advantage: RoPE naturally extends to longer sequences than
seen during training (with techniques like NTK-aware scaling,
YaRN, etc.)
```

---

## 5. The Attention Mechanism  From Scratch

This is the heart of the Transformer. Take your time with this section  everything else builds on it.

### 5.1 Intuition: The Library Analogy

**Imagine you're in a library doing research on "climate change effects on polar bears."**

```
YOUR RESEARCH PROCESS:

1. You have a QUERY:     "How does climate change affect polar bears?"

2. The library has KEYS:  Every book has a title/description
   - "Arctic Wildlife Biology"           ← Very relevant!
   - "Polar Ice Cap Measurements"        ← Somewhat relevant
   - "Introduction to Jazz Music"        ← Not relevant
   - "Climate Models and Predictions"    ← Relevant!
   - "History of Ancient Rome"           ← Not relevant

3. You MATCH your query against each key:
   Query × Key = Relevance Score
   
   "climate...bears?" × "Arctic Wildlife"    = 0.95 (high match!)
   "climate...bears?" × "Polar Ice Caps"     = 0.70
   "climate...bears?" × "Jazz Music"         = 0.02 (no match)
   "climate...bears?" × "Climate Models"     = 0.80
   "climate...bears?" × "Ancient Rome"       = 0.01

4. You read each book's VALUE (content) weighted by relevance:
   
   Output = 0.95 × [Arctic Wildlife content]
          + 0.70 × [Ice Cap content]
          + 0.02 × [Jazz content]          ← basically ignored
          + 0.80 × [Climate Model content]
          + 0.01 × [Rome content]          ← basically ignored

Result: A blended summary focused on the most relevant sources!
```

This is **exactly** what attention does:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What does each item contain?" (used for matching)
- **Value (V):** "What information does each item actually provide?"

### 5.2 Self-Attention: Where Q, K, V Come From

In **self-attention**, the queries, keys, and values all come from the **same** input sequence. Each token generates its own Q, K, and V by multiplying its embedding with learned weight matrices:

```
SELF-ATTENTION SETUP:

Input tokens: ["The", "cat", "sat"]

Each token's embedding is projected into Q, K, V:

  Token          Embedding (d=4)          Q (via W_Q)      K (via W_K)      V (via W_V)
  ─────          ──────────────           ──────────       ──────────       ──────────
  "The"    →    [1.0, 0.5, 0.3, 0.2] × W_Q = q₁         × W_K = k₁       × W_V = v₁
  "cat"    →    [0.8, 0.9, 0.1, 0.7] × W_Q = q₂         × W_K = k₂       × W_V = v₂
  "sat"    →    [0.2, 0.4, 0.8, 0.5] × W_Q = q₃         × W_K = k₃       × W_V = v₃

  W_Q, W_K, W_V are LEARNED weight matrices (d × d_k)
  These are the main learnable parameters in attention!
```

### 5.3 Dot-Product Attention  Worked Example

Let's work through a complete example with actual numbers. We'll use small dimensions for clarity.

**Setup:** 3 tokens, embedding dimension d=4, attention dimension d_k=3

```
Step 1: Start with token embeddings

  x₁ ("The") = [1.0, 0.5, 0.3, 0.2]
  x₂ ("cat") = [0.8, 0.9, 0.1, 0.7]
  x₃ ("sat") = [0.2, 0.4, 0.8, 0.5]

Step 2: Project to Q, K, V using weight matrices

  Suppose (simplified) W_Q, W_K, W_V project from d=4 to d_k=3:

  Q = X · W_Q:
    q₁ = [0.5,  0.8,  0.3]     (query for "The")
    q₂ = [0.9,  0.2,  0.7]     (query for "cat")
    q₃ = [0.3,  0.6,  0.9]     (query for "sat")

  K = X · W_K:
    k₁ = [0.4,  0.7,  0.2]     (key for "The")
    k₂ = [0.8,  0.3,  0.6]     (key for "cat")
    k₃ = [0.1,  0.9,  0.5]     (key for "sat")

  V = X · W_V:
    v₁ = [1.0,  0.0,  0.5]     (value for "The")
    v₂ = [0.0,  1.0,  0.3]     (value for "cat")
    v₃ = [0.5,  0.5,  1.0]     (value for "sat")
```

```
Step 3: Compute attention scores (Q · K^T)

  Score(i,j) = q_i · k_j  (dot product)

  Scores matrix:
              k₁("The")  k₂("cat")  k₃("sat")
  q₁("The")  [ 0.5×0.4 + 0.8×0.7 + 0.3×0.2 ] = 0.20+0.56+0.06 = 0.82
              [ 0.5×0.8 + 0.8×0.3 + 0.3×0.6 ] = 0.40+0.24+0.18 = 0.82
              [ 0.5×0.1 + 0.8×0.9 + 0.3×0.5 ] = 0.05+0.72+0.15 = 0.92

  q₂("cat")  [ 0.9×0.4 + 0.2×0.7 + 0.7×0.2 ] = 0.36+0.14+0.14 = 0.64
              [ 0.9×0.8 + 0.2×0.3 + 0.7×0.6 ] = 0.72+0.06+0.42 = 1.20
              [ 0.9×0.1 + 0.2×0.9 + 0.7×0.5 ] = 0.09+0.18+0.35 = 0.62

  q₃("sat")  [ 0.3×0.4 + 0.6×0.7 + 0.9×0.2 ] = 0.12+0.42+0.18 = 0.72
              [ 0.3×0.8 + 0.6×0.3 + 0.9×0.6 ] = 0.24+0.18+0.54 = 0.96
              [ 0.3×0.1 + 0.6×0.9 + 0.9×0.5 ] = 0.03+0.54+0.45 = 1.02

  Raw scores matrix:
              The    cat    sat
  The    [   0.82   0.82   0.92  ]
  cat    [   0.64   1.20   0.62  ]
  sat    [   0.72   0.96   1.02  ]
```

```
Step 4: Scale by √d_k

  WHY SCALE?
  ─────────────────────────────────────────────────────────
  Without scaling, as d_k grows large, the dot products grow large too
  (variance ≈ d_k). Large values push softmax into saturation regions
  where gradients vanish. Dividing by √d_k normalizes the variance to ~1.

  d_k = 3, so √d_k = √3 ≈ 1.73

  Scaled scores:
              The    cat    sat
  The    [   0.47   0.47   0.53  ]
  cat    [   0.37   0.69   0.36  ]
  sat    [   0.42   0.55   0.59  ]
```

```
Step 5: Apply Softmax (row-wise)

  Softmax converts scores to probabilities (each row sums to 1):
  softmax(z_i) = exp(z_i) / Σ exp(z_j)

  Row 1 ("The"): exp(0.47)=1.60, exp(0.47)=1.60, exp(0.53)=1.70
    Sum = 4.90
    Attention weights: [1.60/4.90, 1.60/4.90, 1.70/4.90]
                     = [0.327,     0.327,     0.347]

  Row 2 ("cat"): exp(0.37)=1.45, exp(0.69)=1.99, exp(0.36)=1.43
    Sum = 4.87
    Attention weights: [1.45/4.87, 1.99/4.87, 1.43/4.87]
                     = [0.298,     0.409,     0.294]

  Row 3 ("sat"): exp(0.42)=1.52, exp(0.55)=1.73, exp(0.59)=1.80
    Sum = 5.05
    Attention weights: [1.52/5.05, 1.73/5.05, 1.80/5.05]
                     = [0.301,     0.343,     0.356]

  Attention weight matrix:
              The    cat    sat
  The    [   0.327  0.327  0.347 ]  ← "The" attends roughly equally
  cat    [   0.298  0.409  0.294 ]  ← "cat" attends most to itself
  sat    [   0.301  0.343  0.356 ]  ← "sat" attends slightly more to itself
```

```
Step 6: Multiply attention weights by Values

  output_i = Σ_j  attention_weight(i,j) × v_j

  For "The" (row 1):
    = 0.327 × v₁ + 0.327 × v₂ + 0.347 × v₃
    = 0.327 × [1.0, 0.0, 0.5] + 0.327 × [0.0, 1.0, 0.3] + 0.347 × [0.5, 0.5, 1.0]
    = [0.327, 0.000, 0.164] + [0.000, 0.327, 0.098] + [0.174, 0.174, 0.347]
    = [0.501, 0.501, 0.609]

  For "cat" (row 2):
    = 0.298 × [1.0, 0.0, 0.5] + 0.409 × [0.0, 1.0, 0.3] + 0.294 × [0.5, 0.5, 1.0]
    = [0.298, 0.000, 0.149] + [0.000, 0.409, 0.123] + [0.147, 0.147, 0.294]
    = [0.445, 0.556, 0.566]

  For "sat" (row 3):
    = 0.301 × [1.0, 0.0, 0.5] + 0.343 × [0.0, 1.0, 0.3] + 0.356 × [0.5, 0.5, 1.0]
    = [0.301, 0.000, 0.151] + [0.000, 0.343, 0.103] + [0.178, 0.178, 0.356]
    = [0.479, 0.521, 0.610]
```

### 5.4 The Complete Attention Formula

Everything above can be summarized in one elegant equation:

```
                         Q · K^T
Attention(Q, K, V) = softmax( ────── ) · V
                          √d_k

Where:
  Q = queries matrix   (n × d_k)     n = sequence length
  K = keys matrix      (n × d_k)     d_k = key/query dimension
  V = values matrix    (n × d_v)     d_v = value dimension
  Q·K^T = score matrix (n × n)       every token scores against every other
  softmax = row-wise   (n × n)       weights sum to 1 per row
  × V =               (n × d_v)      weighted combination of values
```

```
VISUAL SUMMARY OF ATTENTION:

  Q          K^T                    Scores              Weights        V          Output
(n×d_k)   (d_k×n)                  (n×n)               (n×n)       (n×d_v)      (n×d_v)
┌─────┐   ┌───────┐    ÷√d_k     ┌───────┐  softmax  ┌───────┐   ┌─────┐     ┌─────┐
│     │   │       │      │       │       │    │      │ ·327 ·327│   │     │     │     │
│     │ × │       │  ──► │  ──►  │       │ ──┼────► │ ·298 ·409│ × │     │ ──► │     │
│     │   │       │      │       │       │    │      │ ·301 ·343│   │     │     │     │
└─────┘   └───────┘               └───────┘           └───────┘   └─────┘     └─────┘
  MatMul        Scale         Softmax         MatMul       = Context-aware embeddings!
```

### 5.5 Self-Attention with a Real Sentence

Let's see how attention works on a more meaningful example:

```
Sentence: "The animal didn't cross the street because it was too tired"

Question: What does "it" refer to?  → "the animal" (not "the street")

ATTENTION WEIGHTS for "it" (simplified visualization):

  The    animal  didn't  cross   the    street  because  it    was    too    tired
  0.05   0.35    0.02    0.03   0.04    0.15    0.02    0.12   0.05   0.03   0.14
         ████                           ██                █                  ██
         ▲                              ▲                                    ▲
         │                              │                                    │
    Most attention               Some attention                        "tired" → animate
    (correct!)                   (also plausible)                      (helps resolve)

The model learns that "it" in this context likely refers to "animal"
because "tired" is a property of animate objects, not streets.
With "was too wide" instead: "it" would attend more to "street"!
```

This ability to dynamically determine what each word should attend to  based on the full context  is the magic of self-attention.

### 5.6 Why Scale by √d_k?  Mathematical Intuition

This is a subtle but important detail:

```
Consider two random vectors q and k, each with d_k dimensions,
where each element is drawn from N(0, 1):

  q · k = q₁k₁ + q₂k₂ + ... + q_{d_k}k_{d_k}

Each product q_i·k_i has:
  E[q_i·k_i] = 0        (mean)
  Var[q_i·k_i] = 1      (variance)

The sum of d_k such terms:
  E[q·k] = 0
  Var[q·k] = d_k         ← variance GROWS with dimension!

So for d_k = 512:
  Standard deviation of q·k ≈ √512 ≈ 22.6

This means dot products will be large numbers like ±30 or ±50.

Softmax of [30, 1, -20, 2]:
  ≈ [1.000, 0.000, 0.000, 0.000]    ← completely saturated!
  Gradient ≈ 0 for all but one position. Training stalls.

After scaling by √d_k:
  Var[q·k / √d_k] = d_k / d_k = 1   ← normalized!

Softmax of [1.3, 0.04, -0.88, 0.09]:
  ≈ [0.39, 0.24, 0.10, 0.26]         ← smooth distribution
  Healthy gradients everywhere.
```

---

## 6. Multi-Head Attention

### 6.1 Why Multiple Heads?

A single attention head computes ONE type of relationship between tokens. But language has MANY types of relationships:

```
Sentence: "The fluffy cat that John adopted sat on the warm mat"

Relationships a model needs to capture:
  1. Syntactic:  "cat" ←→ "sat"        (subject-verb)
  2. Adjectival: "cat" ←→ "fluffy"     (modifier)
  3. Relative:   "cat" ←→ "adopted"    (relative clause)
  4. Spatial:    "sat" ←→ "mat"        (location)
  5. Adjectival: "mat" ←→ "warm"       (modifier)

ONE attention head can focus on ONE type of relationship.
MULTIPLE heads can capture ALL of them simultaneously!
```

### 6.2 How Multi-Head Attention Works

```
MULTI-HEAD ATTENTION:

                    ┌─── Head 1: W_Q1, W_K1, W_V1 → Attention → head₁ ───┐
                    │                                                       │
Input ──────────────┼─── Head 2: W_Q2, W_K2, W_V2 → Attention → head₂ ───┤
(n × d_model)       │                                                       ├──► Concat ──► W_O ──► Output
                    │                                                       │                       (n × d_model)
                    ├─── Head 3: W_Q3, W_K3, W_V3 → Attention → head₃ ───┤
                    │                                                       │
                    └─── Head h: W_Qh, W_Kh, W_Vh → Attention → head_h ──┘


DIMENSIONS:

  d_model = 512 (total embedding dimension)
  h = 8 heads
  d_k = d_v = d_model / h = 512 / 8 = 64 per head

  Each head works in a 64-dimensional subspace
  Results are concatenated: 8 × 64 = 512
  Then projected through W_O (512 × 512) back to d_model
```

### 6.3 What Different Heads Learn

Research has shown that different heads in trained Transformers specialize in different patterns:

```
HEAD SPECIALIZATION (observed in BERT):

Head 1: Syntactic - Subject-Verb agreement
  "The cats [that the dog chased] WERE fast"
  Head 1 connects "cats" → "were"

Head 2: Next/Previous token
  Every token attends to its immediate neighbor
  (like a local context window)

Head 3: Positional
  Tokens at similar positions attend to each other

Head 4: Separator tokens
  Attends to [SEP] or punctuation marks

Head 5: Coreference
  "John said HE would..."  Head 5 links "he" → "John"

Head 6: Rare/important words
  Attends to semantically unusual tokens in the sequence
```

### 6.4 The Math

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) · W_O

Where head_i = Attention(Q · W_Qi, K · W_Ki, V · W_Vi)

Parameters:
  W_Qi ∈ ℝ^(d_model × d_k)     for each head i
  W_Ki ∈ ℝ^(d_model × d_k)     for each head i
  W_Vi ∈ ℝ^(d_model × d_v)     for each head i
  W_O  ∈ ℝ^(h·d_v × d_model)   output projection

Total parameters in multi-head attention:
  = h × (d_model × d_k × 3) + (h × d_v) × d_model
  = h × (d_model × (d_model/h) × 3) + d_model²
  = 3 × d_model² + d_model²
  = 4 × d_model²

For d_model=512: 4 × 512² = 1,048,576 ≈ 1M parameters
(just for one attention layer!)
```

**Important:** Multi-head attention with h heads of dimension d_k = d_model/h has roughly the **same computational cost** as single-head attention with full d_model dimensions. We get multiple perspectives "for free" (in terms of FLOPs).

---

## 7. The Encoder Architecture

### 7.1 Encoder Block Structure

The Transformer encoder is a stack of identical layers, each containing two sub-layers:

```
SINGLE ENCODER BLOCK:

  Input (n × d_model)
    │
    ▼
  ┌──────────────────────────────────┐
  │       MULTI-HEAD ATTENTION       │◄──── Each token attends to
  │  (self-attention: Q=K=V=input)   │      ALL other tokens
  └───────────────┬──────────────────┘
                  │
            ┌─────▼─────┐
            │  ADD & NORM │◄──── Residual connection + Layer Norm
            └─────┬─────┘
                  │
  ┌───────────────▼──────────────────┐
  │      FEED-FORWARD NETWORK        │◄──── Position-wise: same MLP
  │  (two linear layers + ReLU)      │      applied to each position
  │  FFN(x) = max(0, xW₁+b₁)W₂+b₂  │      independently
  └───────────────┬──────────────────┘
                  │
            ┌─────▼─────┐
            │  ADD & NORM │◄──── Another residual + Layer Norm
            └─────┬─────┘
                  │
                  ▼
  Output (n × d_model)         ← Same shape as input!
                                  (allows stacking)

  This block is repeated N times (N=6 in original, N=12+ in modern)
```

### 7.2 Residual Connections (Add)

Same concept as ResNet skip connections. Each sub-layer's output is **added** to its input:

```
RESIDUAL CONNECTION:

  x ──────────────────────────────────┐
  │                                   │
  ▼                                   │
  ┌─────────────────┐                 │
  │   Sub-layer     │                 │
  │  (attention or  │                 │
  │   feed-forward) │                 │
  └────────┬────────┘                 │
           │                          │
           ▼                          │
        Sub-layer(x) + x  ◄──────────┘
           │
           ▼
      Layer Norm

Output = LayerNorm(x + SubLayer(x))
```

**Why residual connections?**
- Allow gradients to flow directly through the network (skip the sub-layer if needed)
- Enable much deeper networks (original: 6 layers; modern: 80-100+ layers)
- If a sub-layer has nothing useful to add, it can learn to output ~0, and the input passes through unchanged

### 7.3 Layer Normalization

Layer Norm normalizes across the feature dimension for each token independently:

```
LAYER NORMALIZATION:

  For a single token's representation x = [x₁, x₂, ..., x_d]:

  μ = mean(x₁, ..., x_d)          (mean across features)
  σ = std(x₁, ..., x_d)           (standard deviation)

  LayerNorm(x)_i = γ × (x_i - μ)/σ + β

  Where γ (scale) and β (shift) are learnable parameters

Example:
  x = [4.0, 2.0, 0.0, 2.0]
  μ = 2.0,  σ = 1.41
  
  Normalized: [(4-2)/1.41, (2-2)/1.41, (0-2)/1.41, (2-2)/1.41]
            = [1.41, 0.0, -1.41, 0.0]

WHY NOT BATCH NORM?
  Batch Norm normalizes across the batch dimension  problematic for:
  1. Variable-length sequences (different positions shouldn't share stats)
  2. Small batch sizes during inference
  3. Autoregressive generation (batch size = 1)
  Layer Norm normalizes per-token, avoiding all these issues.
```

### 7.4 The Feed-Forward Network (FFN)

Each position goes through the same two-layer MLP independently:

```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂

Dimensions:
  x:  (d_model)         e.g., 512
  W₁: (d_model × d_ff)  e.g., 512 × 2048
  W₂: (d_ff × d_model)  e.g., 2048 × 512

The inner dimension d_ff is typically 4× d_model

  Input: 512 dims ──► Expand: 2048 dims ──► Contract: 512 dims

  ┌──────┐      ┌──────────────┐      ┌──────┐
  │ 512  │─────►│   ReLU       │─────►│ 512  │
  │ dims │      │   2048 dims  │      │ dims │
  └──────┘      └──────────────┘      └──────┘
               (expansion creates room
                for nonlinear processing)
```

**Why is the FFN needed?** Attention is a **linear** operation on values (it's just weighted averaging). The FFN adds **nonlinearity**, enabling the model to compute complex functions of the attended information. Recent interpretability research suggests FFNs act as **key-value memories**  storing factual knowledge learned during training.

### 7.5 Pre-Norm vs. Post-Norm

The original Transformer used **Post-Norm** (normalize after the residual), but modern Transformers almost universally use **Pre-Norm** (normalize before the sub-layer):

```
POST-NORM (original, 2017):           PRE-NORM (modern, 2026):

  x ────────────────┐                  x ────────────────┐
  │                 │                  │                 │
  ▼                 │                  ▼                 │
  Sub-layer         │                  LayerNorm         │
  │                 │                  │                 │
  ▼                 │                  ▼                 │
  Add ◄─────────────┘                  Sub-layer         │
  │                                    │                 │
  ▼                                    ▼                 │
  LayerNorm                            Add ◄─────────────┘
  │                                    │
  ▼                                    ▼

Pre-Norm is more stable for training deep models because
the residual path is completely unobstructed (no normalization
modifying the gradient flow through skip connections).
```

---

## 8. The Decoder Architecture

### 8.1 What Makes the Decoder Different

The decoder has the same components as the encoder, plus two critical additions:

1. **Masked self-attention**  prevents the model from "cheating" by looking at future tokens
2. **Cross-attention**  allows the decoder to attend to the encoder's output

```
SINGLE DECODER BLOCK:

  Input (target sequence, n × d_model)
    │
    ▼
  ┌──────────────────────────────────┐
  │    MASKED MULTI-HEAD ATTENTION   │◄──── Can only attend to
  │  (self-attention with causal     │      PREVIOUS positions
  │   mask  no peeking ahead!)      │      (and self)
  └───────────────┬──────────────────┘
                  │
            ┌─────▼─────┐
            │  ADD & NORM │
            └─────┬─────┘
                  │
  ┌───────────────▼──────────────────┐
  │    CROSS-ATTENTION               │◄──── Q from decoder,
  │  (Q = decoder, K,V = encoder)    │      K,V from encoder output
  │  This is how the decoder "reads" │
  │  the source sentence!            │
  └───────────────┬──────────────────┘
                  │
            ┌─────▼─────┐
            │  ADD & NORM │
            └─────┬─────┘
                  │
  ┌───────────────▼──────────────────┐
  │      FEED-FORWARD NETWORK        │
  └───────────────┬──────────────────┘
                  │
            ┌─────▼─────┐
            │  ADD & NORM │
            └─────┬─────┘
                  │
                  ▼
  Output (n × d_model)
```

### 8.2 Masked Self-Attention (Causal Mask)

During training, the decoder sees the entire target sequence at once (for parallelism). But at position t, it should only use information from positions 0 to t  not future positions. This is enforced with a **causal mask**:

```
CAUSAL MASKING:

Attention scores BEFORE masking:        AFTER masking (−∞ → 0 after softmax):

         The   cat   sat   on            The   cat   sat   on
The   [  0.8   0.3   0.5   0.2 ]    [  0.8   -∞    -∞    -∞  ]
cat   [  0.4   0.9   0.6   0.1 ]    [  0.4   0.9   -∞    -∞  ]
sat   [  0.2   0.7   0.8   0.3 ]    [  0.2   0.7   0.8   -∞  ]
on    [  0.5   0.3   0.4   0.7 ]    [  0.5   0.3   0.4   0.7 ]

         ✓ = can see    ✗ = masked

After softmax, -∞ becomes 0 (zero attention weight):

The   [  1.00   0.00   0.00   0.00 ]  ← "The" only sees itself
cat   [  0.38   0.62   0.00   0.00 ]  ← "cat" sees "The" and itself
sat   [  0.18   0.42   0.40   0.00 ]  ← "sat" sees first 3 tokens
on    [  0.22   0.18   0.20   0.40 ]  ← "on" sees all 4 tokens

┌───┬───┬───┬───┐
│ ✓ │ ✗ │ ✗ │ ✗ │   This triangular mask ensures the model
├───┼───┼───┼───┤   can only attend to previous (and current)
│ ✓ │ ✓ │ ✗ │ ✗ │   positions  preserving the autoregressive
├───┼───┼───┼───┤   property during training!
│ ✓ │ ✓ │ ✓ │ ✗ │
├───┼───┼───┼───┤
│ ✓ │ ✓ │ ✓ │ ✓ │
└───┴───┴───┴───┘
```

**Why is this necessary?** During generation, the model predicts one token at a time  it hasn't generated future tokens yet! The mask simulates this during training so the model learns the same way it will be used.

### 8.3 Cross-Attention

In cross-attention, the **queries** come from the decoder, but the **keys and values** come from the encoder's output:

```
CROSS-ATTENTION IN TRANSLATION:

Encoder output (source: "Le chat est assis"):
  k₁,v₁ = processed "Le"
  k₂,v₂ = processed "chat"
  k₃,v₃ = processed "est"
  k₄,v₄ = processed "assis"

Decoder generating (target: "The cat is sitting"):
  When generating "cat" (position 2):
    q₂ = decoder's representation at position 2

  Cross-attention:
    q₂ · k₁ = 0.1  (low: "Le" is an article)
    q₂ · k₂ = 0.9  (high: "chat" = "cat"!)
    q₂ · k₃ = 0.05 (low: "est" = "is")
    q₂ · k₄ = 0.15 (low: "assis" = "sitting")

    → Decoder heavily attends to "chat" when producing "cat" ✓

This is how the decoder "reads" the source sentence to produce
the translation! Each target word "looks at" the most relevant
source words.
```

---

## 9. Encoder-Decoder Architecture (Full Transformer)

### 9.1 The Complete Picture

```
FULL TRANSFORMER (Encoder-Decoder):

SOURCE INPUT                                          TARGET INPUT
"Le chat est assis"                                   "<START> The cat is"
      │                                                     │
      ▼                                                     ▼
┌──────────┐                                          ┌──────────┐
│  Token   │                                          │  Token   │
│ Embedding│                                          │ Embedding│
│    +     │                                          │    +     │
│ Position │                                          │ Position │
└────┬─────┘                                          └────┬─────┘
     │                                                     │
     ▼                                                     ▼
┌─────────────┐                                    ┌───────────────┐
│  ENCODER    │                                    │   DECODER     │
│  Block 1    │                                    │   Block 1     │
│ ┌─────────┐ │                                    │ ┌───────────┐ │
│ │Self-Attn│ │                                    │ │Masked Self│ │
│ │+Add&Norm│ │                                    │ │ Attention │ │
│ ├─────────┤ │                                    │ │+Add&Norm  │ │
│ │FFN      │ │                                    │ ├───────────┤ │
│ │+Add&Norm│ │      ┌────────────────────────────►│ │Cross-Attn │ │
│ └─────────┘ │      │                             │ │(K,V=enc)  │ │
├─────────────┤      │                             │ │+Add&Norm  │ │
│  ENCODER    │      │                             │ ├───────────┤ │
│  Block 2    │      │                             │ │FFN        │ │
│    ...      │      │                             │ │+Add&Norm  │ │
├─────────────┤      │                             │ └───────────┘ │
│    ...      │      │                             ├───────────────┤
├─────────────┤      │                             │   DECODER     │
│  ENCODER    │──────┘                             │   Block 2     │
│  Block N    │   encoder output                   │    ...        │
│             │   (used as K,V                     ├───────────────┤
└─────────────┘    in ALL decoder                  │   DECODER     │
                   cross-attention                 │   Block N     │
                   layers)                         └───────┬───────┘
                                                           │
                                                           ▼
                                                    ┌──────────┐
                                                    │  Linear  │
                                                    │    +     │
                                                    │ Softmax  │
                                                    └────┬─────┘
                                                         │
                                                         ▼
                                                   "sitting" (next token)
```

### 9.2 Use Cases for Encoder-Decoder

| Task | Source (Encoder) | Target (Decoder) |
|------|-----------------|-------------------|
| Translation | "Le chat est assis" | "The cat is sitting" |
| Summarization | Long article | Short summary |
| Question Answering | Context + question | Answer |
| Speech-to-Text | Audio features | Transcript |

**Notable encoder-decoder models:** T5, BART, mBART, Whisper (for speech)

---

## 10. Decoder-Only Architecture (GPT-Style)

### 10.1 Why Decoder-Only?

A surprising finding: for many tasks, you don't need a separate encoder at all! You can frame everything as **text generation**:

```
ENCODER-DECODER (traditional):
  Encoder input:  "Translate to French: The cat sat"
  Decoder output: "Le chat était assis"

DECODER-ONLY (GPT-style):
  Input + Output as one sequence:
  "Translate to French: The cat sat → Le chat était assis"

  The model just generates text left-to-right, treating
  EVERYTHING (including the "instruction") as a sequence to continue.
```

### 10.2 Decoder-Only Architecture

```
DECODER-ONLY TRANSFORMER (GPT-Style):

  Input: "The cat sat"
      │
      ▼
  ┌──────────┐
  │  Token   │
  │ Embedding│
  │    +     │
  │ Position │
  └────┬─────┘
       │
       ▼
  ┌───────────────┐
  │   Block 1     │
  │ ┌───────────┐ │
  │ │ MASKED    │ │     ← Causal mask (can only see past)
  │ │ Self-Attn │ │     ← No encoder, no cross-attention!
  │ │+Add&Norm  │ │
  │ ├───────────┤ │
  │ │ FFN       │ │
  │ │+Add&Norm  │ │
  │ └───────────┘ │
  ├───────────────┤
  │   Block 2     │
  │     ...       │
  ├───────────────┤
  │     ...       │
  ├───────────────┤
  │   Block N     │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐     ┌──────────────────────────────────────────┐
  │    Linear     │     │ Output is a probability distribution     │
  │  (d_model →   │────►│ over the entire vocabulary:              │
  │   vocab_size) │     │                                          │
  │    +          │     │ P("on")    = 0.45  ← highest!            │
  │  Softmax      │     │ P("down")  = 0.15                       │
  └───────────────┘     │ P("the")   = 0.08                       │
                        │ P("in")    = 0.07                       │
                        │ P("up")    = 0.05                       │
                        │ P(others)  = 0.20                       │
                        │                                          │
                        │ Next token: "on" (sample or argmax)      │
                        └──────────────────────────────────────────┘
```

### 10.3 Why Decoder-Only Dominates in 2026

Almost all large language models in 2026 are decoder-only:

| Model | Type | Parameters | Architecture |
|-------|------|-----------|-------------|
| GPT-4/4o/4.5 | Decoder-only | Undisclosed (rumored MoE) | Transformer |
| Claude 3.5/4 (Anthropic) | Decoder-only | Undisclosed | Transformer |
| Gemini 2.5 (Google) | Decoder-only | Undisclosed (MoE) | Transformer |
| LLaMA 3 (Meta) | Decoder-only | 8B, 70B, 405B | Transformer |
| Mistral / Mixtral | Decoder-only | 7B, 8x22B (MoE) | Transformer |
| Qwen 2.5 (Alibaba) | Decoder-only | 0.5B-72B | Transformer |
| DeepSeek V3/R1 | Decoder-only | 671B (MoE) | Transformer |

**Why?**
1. **Simplicity:** One architecture for everything (no encoder needed)
2. **Scaling:** Decoder-only scales better  scaling laws favor unified models
3. **Flexibility:** Any task can be framed as text generation
4. **Pretraining efficiency:** Next-token prediction on raw text is simple and scales well
5. **In-context learning:** Decoder-only models naturally support few-shot prompting

---

## 11. Step-by-Step Token Processing: How "The cat sat" Predicts "on"

Let's trace through the **complete** forward pass of a simplified decoder-only Transformer processing "The cat sat" and predicting "on."

### 11.1 Tokenization

```
Input text: "The cat sat"

Tokenizer (BPE):
  "The" → token ID 464
  " cat" → token ID 3797     (note: space is part of the token)
  " sat" → token ID 3332

Token IDs: [464, 3797, 3332]
```

### 11.2 Embedding + Positional Encoding

```
Step 1: Token Embedding Lookup

  E[464]  = [0.12, -0.34, 0.56, ..., 0.78]    (d=768 dimensions)
  E[3797] = [0.45, 0.23, -0.11, ..., 0.34]
  E[3332] = [-0.22, 0.67, 0.33, ..., -0.15]

Step 2: Add Positional Encoding

  x₁ = E[464]  + PE(0) = [0.12+0.00, -0.34+1.00, 0.56+0.00, ..., 0.78+...]
  x₂ = E[3797] + PE(1) = [0.45+0.84, 0.23+0.54, -0.11+0.84, ..., 0.34+...]
  x₃ = E[3332] + PE(2) = [-0.22+0.91, 0.67-0.42, 0.33+0.91, ..., -0.15+...]

Input matrix X: shape (3 × 768)
```

### 11.3 Through the First Transformer Block

```
Step 3: Masked Multi-Head Attention

  For each of 12 heads (d_k = 768/12 = 64 per head):
    Q_h = X · W_Q^h     (3 × 64)
    K_h = X · W_K^h     (3 × 64)
    V_h = X · W_V^h     (3 × 64)

    Scores = Q_h · K_h^T / √64    (3 × 3)

    Apply causal mask:
    ┌──────┬──────┬──────┐
    │ s₁₁  │  -∞  │  -∞  │    Token 1 only sees token 1
    │ s₂₁  │ s₂₂  │  -∞  │    Token 2 sees tokens 1-2
    │ s₃₁  │ s₃₂  │ s₃₃  │    Token 3 sees tokens 1-3
    └──────┴──────┴──────┘

    Weights = softmax(masked scores)     (3 × 3)
    head_h = Weights · V_h              (3 × 64)

  Concatenate all heads: [head₁|head₂|...|head₁₂]  (3 × 768)
  Project: MultiHead_output = Concat · W_O          (3 × 768)

Step 4: Residual + Layer Norm
  X = LayerNorm(X + MultiHead_output)               (3 × 768)

Step 5: Feed-Forward Network
  For each position independently:
    FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
    W₁: 768 → 3072 (expand 4×)
    W₂: 3072 → 768 (contract back)

Step 6: Residual + Layer Norm
  X = LayerNorm(X + FFN_output)                     (3 × 768)
```

### 11.4 Through Remaining Blocks

```
Step 7: Repeat Steps 3-6 for blocks 2 through N

  Block 2:  X → Masked Attention → Add&Norm → FFN → Add&Norm → X
  Block 3:  X → Masked Attention → Add&Norm → FFN → Add&Norm → X
  ...
  Block 12: X → Masked Attention → Add&Norm → FFN → Add&Norm → X

  Each block refines the representations further.
  Early blocks: basic syntax, word relationships
  Middle blocks: semantic composition
  Late blocks: task-specific patterns, output preparation
```

### 11.5 Final Prediction

```
Step 8: Take the LAST token's representation

  After all blocks, X has shape (3 × 768)
  We only need x₃ (the last position) for prediction:
  
  x₃ = [refined 768-dim representation of "sat",
         incorporating context from "The" and "cat"]

Step 9: Project to vocabulary

  logits = x₃ · W_vocab + b_vocab
  W_vocab: (768 × 50,257)  maps to probability over entire vocabulary

  logits = [... -2.3, 4.1, -0.5, ... 6.8, ..., -1.2, ...]
                              │              │
                           "the"           "on"

Step 10: Softmax → Probabilities

  probs = softmax(logits)

  P(" on")     = 0.42   ← HIGHEST  model predicts "on"!
  P(" the")    = 0.11
  P(" down")   = 0.08
  P(" in")     = 0.06
  P(" upon")   = 0.04
  P(" over")   = 0.03
  ...
  (50,257 probabilities summing to 1.0)

Step 11: Sample or Greedy Decode

  Greedy: next_token = argmax(probs) = " on"
  
  Sampling (temperature=1.0): randomly sample according to probabilities
    → Might get "on" (42%), "the" (11%), "down" (8%), etc.

  Sampling (temperature=0.5): probabilities become more peaked
    → "on" becomes ~70%, other tokens shrink
  
  Sampling (temperature=2.0): probabilities flatten
    → "on" becomes ~15%, more diversity


THE COMPLETE SENTENCE SO FAR: "The cat sat on"

To continue generating:
  Feed [464, 3797, 3332, 319]  ("The cat sat on")
  → Model predicts: " the" (most likely)
  Feed [464, 3797, 3332, 319, 262]  ("The cat sat on the")
  → Model predicts: " mat" (likely, given common phrasing)
  ...and so on until an end-of-sequence token is generated.
```

### 11.6 Summary Diagram of the Full Forward Pass

```
"The cat sat"
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TOKENIZE:  [464, 3797, 3332]                                       │
├─────────────────────────────────────────────────────────────────────┤
│ EMBED + POSITION:  3 × 768 matrix                                  │
├─────────────────────────────────────────────────────────────────────┤
│ BLOCK 1:  Masked Self-Attention → Add&Norm → FFN → Add&Norm        │
│ BLOCK 2:  Masked Self-Attention → Add&Norm → FFN → Add&Norm        │
│ ...                                                                 │
│ BLOCK 12: Masked Self-Attention → Add&Norm → FFN → Add&Norm        │
├─────────────────────────────────────────────────────────────────────┤
│ TAKE LAST POSITION: 768-dim vector                                  │
├─────────────────────────────────────────────────────────────────────┤
│ LINEAR: 768 → 50,257 (vocabulary size)                              │
├─────────────────────────────────────────────────────────────────────┤
│ SOFTMAX: probability distribution over vocabulary                   │
├─────────────────────────────────────────────────────────────────────┤
│ OUTPUT: " on" (P=0.42)                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. KV-Cache and Inference Optimization

### 12.1 The Problem: Redundant Computation

When generating text autoregressively (one token at a time), naive implementation recomputes attention for ALL previous tokens at every step:

```
NAIVE AUTOREGRESSIVE GENERATION:

Step 1: Process ["The"]
  → Compute Q, K, V for "The"
  → Predict "cat"

Step 2: Process ["The", "cat"]
  → Compute Q, K, V for "The" AGAIN (wasteful!)
  → Compute Q, K, V for "cat"
  → Predict "sat"

Step 3: Process ["The", "cat", "sat"]
  → Compute Q, K, V for "The" AGAIN (wasteful!)
  → Compute Q, K, V for "cat" AGAIN (wasteful!)
  → Compute Q, K, V for "sat"
  → Predict "on"

Step N: Must recompute K,V for all N-1 previous tokens!

Computational cost grows as O(N²)  gets very expensive for long sequences!
```

### 12.2 The Solution: KV-Cache

**Key insight:** In the causal attention mask, the K and V vectors for previous tokens **never change**  they only depend on the token and its position, not on future tokens. So we can **cache** them!

```
KV-CACHE GENERATION:

Step 1: Process ["The"]
  → Compute Q₁, K₁, V₁ for "The"
  → CACHE: K_cache = [K₁], V_cache = [V₁]
  → Attention: Q₁ · K₁^T / √d → softmax → · V₁
  → Predict "cat"

Step 2: Process ONLY ["cat"]  (just the NEW token!)
  → Compute Q₂, K₂, V₂ for "cat"
  → APPEND to cache: K_cache = [K₁, K₂], V_cache = [V₁, V₂]
  → Attention: Q₂ · [K₁, K₂]^T / √d → softmax → · [V₁, V₂]
  → Predict "sat"

Step 3: Process ONLY ["sat"]
  → Compute Q₃, K₃, V₃ for "sat"
  → APPEND to cache: K_cache = [K₁, K₂, K₃], V_cache = [V₁, V₂, V₃]
  → Attention: Q₃ · [K₁, K₂, K₃]^T / √d → softmax → · [V₁, V₂, V₃]
  → Predict "on"

Only ONE new token is processed at each step!
(Plus the growing cache lookups, but no redundant computation)
```

### 12.3 KV-Cache Memory

```
KV-CACHE MEMORY per token per layer:

  Memory = 2 × d_model × sizeof(float16)
           ↑
       K and V  (no need to cache Q  only used for current token)

For GPT-3 (175B):
  d_model = 12,288
  num_layers = 96
  num_heads = 96
  Memory per token = 2 × 12,288 × 2 bytes × 96 layers = ~4.7 MB per token

For a 4K context:
  KV-cache = 4,096 × 4.7 MB ≈ 19 GB!  (just for the cache!)

For a 128K context (modern models):
  KV-cache = 131,072 × 4.7 MB ≈ 617 GB!  (enormous!)

This is why KV-cache optimization is crucial in 2026.
```

### 12.4 KV-Cache Optimization Techniques (2026)

| Technique | How It Works | Savings |
|-----------|-------------|---------|
| **Multi-Query Attention (MQA)** | All heads share ONE K and V (only Q is per-head) | KV cache ÷ num_heads |
| **Grouped-Query Attention (GQA)** | Groups of heads share K,V (compromise between MQA and MHA) | KV cache ÷ num_groups |
| **Sliding Window Attention** | Only cache last W tokens (Mistral uses W=4096) | Fixed cache size |
| **KV-Cache Quantization** | Store cached K,V in int8/int4 instead of float16 | 2-4× reduction |
| **Paged Attention (vLLM)** | Manage KV-cache like OS virtual memory pages | Reduces fragmentation |
| **Ring Attention** | Distribute KV-cache across multiple GPUs | Enables longer contexts |

```
GROUPED-QUERY ATTENTION (GQA)  Used in LLaMA 3, Gemma, Mistral:

MULTI-HEAD (MHA):                GROUPED-QUERY (GQA):          MULTI-QUERY (MQA):

Q₁ K₁ V₁                         Q₁ ─┐                        Q₁ ─┐
Q₂ K₂ V₂                         Q₂ ─┤─ K₁ V₁                 Q₂ ─┤
Q₃ K₃ V₃                         Q₃ ─┐                        Q₃ ─┤─ K₁ V₁
Q₄ K₄ V₄                         Q₄ ─┤─ K₂ V₂                 Q₄ ─┤
Q₅ K₅ V₅                         Q₅ ─┐                        Q₅ ─┤
Q₆ K₆ V₆                         Q₆ ─┤─ K₃ V₃                 Q₆ ─┤
Q₇ K₇ V₇                         Q₇ ─┐                        Q₇ ─┤
Q₈ K₈ V₈                         Q₈ ─┤─ K₄ V₄                 Q₈ ─┘

8 Q, 8 K, 8 V = 24               8 Q, 4 K, 4 V = 16           8 Q, 1 K, 1 V = 10
Full cache                        50% cache savings             87.5% cache savings
Best quality                      Near-MHA quality              Slight quality loss
```

### 12.5 Prefill vs. Decode Phases

Modern LLM inference has two distinct phases:

```
INFERENCE PHASES:

PHASE 1: PREFILL (process the entire prompt at once)
  ┌──────────────────────────────────────────────────────────────┐
  │ Input: "Write a poem about the ocean"                        │
  │                                                              │
  │ Process all 7 tokens in PARALLEL (like training)             │
  │ Build the initial KV-cache for all prompt tokens             │
  │                                                              │
  │ Compute-bound: lots of matrix multiplications                │
  │ Duration: fast (high GPU utilization)                        │
  └──────────────────────────────────────────────────────────────┘
                               │
                               ▼
PHASE 2: DECODE (generate one token at a time)
  ┌──────────────────────────────────────────────────────────────┐
  │ Token 1: Generate "The"                                      │
  │   → Process only 1 new token, attend to 8 cached tokens      │
  │ Token 2: Generate "waves"                                    │
  │   → Process only 1 new token, attend to 9 cached tokens      │
  │ Token 3: Generate "crash"                                    │
  │   → Process only 1 new token, attend to 10 cached tokens     │
  │ ...                                                          │
  │                                                              │
  │ Memory-bound: small computation, but large KV-cache reads    │
  │ Duration: slow (low GPU utilization, limited by memory BW)   │
  └──────────────────────────────────────────────────────────────┘

This is why LLMs respond slowly  the decode phase is
inherently sequential and memory-bandwidth limited.
```

### 12.6 Speculative Decoding

A clever trick to speed up the decode phase:

```
SPECULATIVE DECODING:

Normal decoding (1 token at a time with BIG model):
  Big model: "The" → "waves" → "crash" → "against" → "the" → ...
  Time: 5 × T_big

Speculative decoding:
  1. Small model (draft): quickly generates 5 candidate tokens
     "The waves crash against the"  (fast, ~0.1 × T_big each)

  2. Big model: VERIFIES all 5 tokens IN PARALLEL (one forward pass!)
     Checks: "The" ✓  "waves" ✓  "crash" ✓  "against" ✓  "the" ✓

  3. If all match: accept all 5 tokens! (5× speedup!)
     If mismatch at position 3: accept first 2, resample from position 3

Typical speedup: 2-3× in practice (not all drafts are accepted)
Used in production by many LLM providers in 2026.
```

---

## 13. Modern Transformer Variants (2024-2026)

### 13.1 Mixture of Experts (MoE)

```
STANDARD TRANSFORMER FFN:          MoE FFN:

  Input → [FFN] → Output           Input → [Router] → Select top-K experts
           ↑                                    │
         (ALL parameters                        ├── Expert 1: [FFN₁] ←── (selected)
          always active)                        ├── Expert 2: [FFN₂]
                                                ├── Expert 3: [FFN₃] ←── (selected)
                                                ├── Expert 4: [FFN₄]
                                                ├── Expert 5: [FFN₅]
                                                ├── Expert 6: [FFN₆]
                                                ├── Expert 7: [FFN₇]
                                                └── Expert 8: [FFN₈]

                                    Only 2 out of 8 experts activated per token!
                                    Total parameters: 8× more
                                    Active parameters: 2× (same as dense model)

Mixtral 8x7B: 46B total params, but only ~13B active per token
DeepSeek V3: 671B total params, ~37B active per token
```

### 13.2 Key Modern Modifications

| Modification | Original (2017) | Modern (2026) | Why |
|---|---|---|---|
| **Positional encoding** | Sinusoidal (absolute) | RoPE (relative/rotary) | Better length generalization |
| **Normalization** | Post-Norm | Pre-Norm (RMSNorm) | Stable training at scale |
| **Activation** | ReLU | SwiGLU / GeGLU | Better performance |
| **Attention** | Multi-Head (MHA) | Grouped-Query (GQA) | Smaller KV-cache |
| **FFN** | Standard | Mixture of Experts (MoE) | More params, same compute |
| **Context length** | 512 tokens | 128K-2M tokens | Long documents |
| **Precision** | FP32 | BF16 / FP8 | Faster, less memory |
| **Vocabulary** | 32K-37K | 100K-150K | Better multilingual, code |

### 13.3 Flash Attention

Standard attention requires storing the full N×N attention matrix in GPU memory  O(N²). **Flash Attention** (Dao et al., 2022) restructures the computation to use GPU SRAM (fast, small) instead of HBM (slow, large):

```
STANDARD ATTENTION:
  1. Compute full S = QK^T      (N×N matrix in HBM  slow!)
  2. Compute P = softmax(S)      (N×N matrix in HBM  slow!)
  3. Compute O = P·V             (read N×N from HBM  slow!)
  
  Memory: O(N²)      HBM reads: O(N² · d)

FLASH ATTENTION:
  Process attention in TILES that fit in SRAM:
  1. Load blocks of Q, K, V into SRAM
  2. Compute partial attention within SRAM (fast!)
  3. Accumulate results using online softmax trick
  4. Write only the final output to HBM
  
  Memory: O(N)       HBM reads: O(N² · d² / SRAM_size)
  
  In practice: 2-4× faster, enables much longer sequences!
```

### 13.4 State Space Models  The Emerging Challenger

While Transformers dominate in 2026, **State Space Models (SSMs)** like **Mamba** are emerging as potential alternatives for certain use cases:

```
TRANSFORMER vs MAMBA:

Transformer:                        Mamba (SSM):
  ✓ Best quality for most tasks      ✓ Linear-time inference (O(N) not O(N²))
  ✓ Mature ecosystem                 ✓ No KV-cache needed
  ✓ Proven at scale                  ✓ Constant memory per step
  ✗ Quadratic attention O(N²)        ✗ Newer, less proven at very large scale
  ✗ Large KV-cache                   ✗ May underperform on tasks requiring
  ✗ Slow decode phase                  precise long-range lookup

Hybrid architectures (Jamba, Zamba) combine both:
  Mamba layers for efficient long-range processing
  + Attention layers for precise token interactions
```

---

## 14. Quiz

### Questions

**Q1:** Why does the Transformer use scaled dot-product attention (dividing by √d_k) instead of plain dot-product attention? What would happen without the scaling?

**Q2:** Explain the purpose of the causal mask in the decoder's self-attention. Why is it needed during training but not at inference time for each new token?

**Q3:** In the KV-Cache optimization, why do we cache K and V but not Q? What property of causal attention makes this possible?

**Q4:** A Transformer has d_model = 1024 and uses 16 attention heads with Grouped-Query Attention (GQA) where 4 heads share each K,V pair. How many distinct K,V pairs are there per layer, and what is the dimension of each?

**Q5:** Consider this sentence: "The trophy doesn't fit in the suitcase because it is too big." What does "it" refer to  the trophy or the suitcase? How would a Transformer's attention mechanism help resolve this ambiguity? What about: "The trophy doesn't fit in the suitcase because it is too small"?

---

### Answers

**A1:** Without scaling, the dot products Q·K^T grow in magnitude proportionally to d_k (variance = d_k for random vectors). With d_k = 64 (typical), dot products would have standard deviation ≈ 8, producing values like ±15 to ±25. When these large values are passed through softmax, the output becomes extremely peaked  almost a one-hot vector  pushing the gradients to near zero (softmax saturation). Dividing by √d_k normalizes the variance back to ~1, keeping softmax in a regime where gradients flow healthily and attention can be distributed across multiple tokens when appropriate.

**A2:** The causal mask prevents position t from attending to positions t+1, t+2, ... (future tokens). During **training**, the entire sequence is processed in parallel for efficiency, so all positions are computed simultaneously  without the mask, position 3 could "see" the answer at position 4, making the task trivially easy (cheating). The mask ensures each position can only use information from previous positions, simulating autoregressive generation. During **inference** with KV-cache, each new token naturally only has access to cached K,V from previous tokens  there are no future tokens to mask because they haven't been generated yet. So the mask is implicitly enforced by the sequential generation process.

**A3:** We cache K and V because these are computed from **previous tokens** and their values **never change**  a token's K and V depend only on that token's embedding and position, not on any future tokens (due to the causal mask). Q, however, is only needed for the **current** token being processed. Each new token generates its own Q, which is used once to compute attention scores against all cached K vectors, then discarded. There's no need to reuse a token's Q in future steps  only its K and V are needed (for other tokens to attend to it).

**A4:** With 16 heads and GQA where 4 heads share each K,V pair, there are **16 / 4 = 4 distinct K,V pairs** per layer (4 groups of 4 heads). Each K and V vector has dimension d_k = d_model / num_heads = 1024 / 16 = 64. So per layer, the KV-cache stores 4 × 2 × 64 = 512 values per token (instead of 16 × 2 × 64 = 2048 for full MHA  a 4× reduction).

**A5:** This is the classic **Winograd Schema Challenge**  designed to test understanding of commonsense reasoning.

- "...because **it is too big**": "it" refers to **the trophy**  a big trophy won't fit in a normally-sized suitcase. The attention mechanism would learn to connect "it" to "trophy" because "too big" relates to the object failing to fit (the thing trying to go inside).

- "...because **it is too small**": "it" refers to **the suitcase**  a small suitcase can't hold a normally-sized trophy. Here, attention shifts to connect "it" to "suitcase" because "too small" relates to the container being insufficient.

The Transformer resolves this through its self-attention mechanism: when processing "it", the model attends to both "trophy" and "suitcase" as candidate referents. The key resolution comes from attending to "big" vs. "small"  the model has learned from training data that the entity described as "too big" in a "doesn't fit" scenario is the object being inserted, while "too small" describes the container. The attention weights for "it" will concentrate more on "trophy" or "suitcase" depending on the adjective, demonstrating how attention dynamically adapts based on the full context. Modern large Transformers (2026) solve these consistently, while this was a major challenge for pre-Transformer NLP systems.

