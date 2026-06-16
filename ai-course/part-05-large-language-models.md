# Part 5: Large Language Models  From Raw Data to Chatbot

> *"The best way to predict the future is to build a language model and let it autocomplete."*

In Part 4, we learned how transformers work  the self-attention mechanism, positional encodings, and the encoder-decoder architecture. Now we zoom out and ask: **how do you take that architecture and turn it into ChatGPT, Claude, or Gemini?**

The answer is a multi-stage pipeline that transforms terabytes of messy internet text into a helpful, harmless, and honest AI assistant. This part covers every stage of that journey.

---

## Table of Contents

1. [The Big Picture: The LLM Pipeline](#1-the-big-picture-the-llm-pipeline)
2. [Stage 1: Pretraining](#2-stage-1-pretraining)
3. [Stage 2: Supervised Fine-Tuning](#3-stage-2-supervised-fine-tuning)
4. [Stage 3: Alignment](#4-stage-3-alignment)
5. [Advanced Techniques](#5-advanced-techniques)
6. [The Complete Pipeline Diagram](#6-the-complete-pipeline-diagram)
7. [Quiz](#7-quiz)

---

## 1. The Big Picture: The LLM Pipeline

### The Cooking Analogy

Think of building an LLM like training a chef:

| Stage | Chef Analogy                       | LLM Stage                  | What Happens                                   |
| ----- | ---------------------------------- | -------------------------- | ---------------------------------------------- |
| 1     | Read every cookbook ever written   | **Pre-training**           | Learn language patterns from billions of pages |
| 2     | Apprentice in a restaurant kitchen | **Supervised Fine-Tuning** | Learn to follow specific instructions          |
| 3     | Get customer feedback and adjust   | **Alignment (RLHF/DPO)**   | Learn human preferences and values             |
| 4     | Specialize in a cuisine            | **Task-specific tuning**   | Optimize for particular use cases              |

A chef who has only read cookbooks (Stage 1) knows a lot about food but might serve you a raw ingredient list when you ask for dinner. Each subsequent stage makes the model more *useful*, *safe*, and *aligned* with what humans actually want.

### The Three Types of Models You'll Encounter

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM Model Types                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BASE MODEL          INSTRUCTION-TUNED        ALIGNED/CHAT          │
│  (GPT-3, Llama       (Llama-Instruct,        (ChatGPT, Claude,      │
│   base weights)       Gemma-IT)               Gemini)               │
│                                                                     │
│  "The quick brown     "Here's how to          "I'd be happy to      │
│   fox jumped over      make pasta:             help you make pasta! │
│   the lazy dog.        1. Boil water           Here's a simple      │
│   The quick brown      2. Add pasta            recipe:              │
│   fox jumped..."       3. Cook 8 min"          1. Boil water..."    │
│                                                                     │
│  Completes text.      Follows instructions.   Helpful, harmless,    │
│  No conversation.     Sometimes robotic.      conversational.       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage 1: Pretraining

Pretraining is the most expensive and foundational stage. It's where the model learns language, facts, reasoning patterns, and code  all from predicting the next word.

### 2.1 Data Collection

#### Where Does the Training Data Come From?

Modern LLMs are trained on **trillions of tokens** from diverse sources:

| Source | Examples | What It Teaches | Approximate Share |
|--------|----------|-----------------|-------------------|
| **Web crawl** | Common Crawl, FineWeb | General knowledge, writing styles | 50-70% |
| **Books** | Books3, Project Gutenberg | Long-form reasoning, narrative | 5-15% |
| **Code** | GitHub, Stack Overflow | Logic, structured thinking, programming | 10-20% |
| **Academic papers** | ArXiv, PubMed, Semantic Scholar | Scientific reasoning, technical depth | 3-8% |
| **Wikipedia** | All languages | Factual knowledge, encyclopedic style | 2-5% |
| **Conversations** | Reddit, forums | Dialogue patterns, Q&A format | 3-10% |
| **Curated/Synthetic** | Textbooks, generated data | High-quality reasoning, targeted skills | 5-15% |

#### Common Crawl: The Internet's Library

Common Crawl is a nonprofit that crawls the web and makes the data freely available. As of 2026:

- **250+ billion** web pages archived
- **Petabytes** of raw data
- Crawled monthly since 2008
- Used as the foundation for most LLM training datasets

But you can't just dump raw Common Crawl into a model. The data requires extensive **cleaning and filtering**:

```
Raw Common Crawl (~250 PB)
         │
         ▼
┌─────────────────────┐
│  Language Detection │  ← Remove non-target languages
│  (fastText/CLD3)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Deduplication      │  ← Remove exact & near-duplicate pages
│  (MinHash/SimHash)  │     (30-50% of web is duplicated)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Quality Filtering   │  ← Remove spam, boilerplate, low-quality
│  (Classifier-based)  │     pages, adult content
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  PII Removal         │  ← Remove personal information
│  (Regex + NER)       │     (emails, phone numbers, SSNs)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Content Extraction  │  ← Strip HTML, navigation, ads
│  (trafilatura)       │     Keep just the article text
└────────┬────────────┘
         │
         ▼
  Clean Dataset (~5-15 TB of text)
```

#### FineWeb and Modern Data Pipelines

HuggingFace's **FineWeb** (2024) demonstrated that careful data curation matters enormously. Their pipeline produced 15 trillion tokens of high-quality English web data that trained better models than datasets 3× larger but less carefully filtered.

Key insight: **Data quality > Data quantity**. A model trained on 1 trillion high-quality tokens often outperforms one trained on 5 trillion low-quality tokens.

#### Data Mixing: The Recipe Matters

Training data isn't just thrown together  the **mixture proportions** are carefully tuned:

```python
# Conceptual data mixture (simplified)
training_mix = {
    "web_text":     0.55,   # General knowledge
    "code":         0.18,   # Programming & logic
    "books":        0.10,   # Long-form reasoning
    "academic":     0.07,   # Scientific knowledge
    "wikipedia":    0.04,   # Factual grounding
    "conversations": 0.04,  # Dialogue patterns
    "math":         0.02,   # Mathematical reasoning
}
# These proportions significantly affect model capabilities
```

Llama 3's technical report revealed they trained on **15 trillion tokens** with a carefully optimized data mix, using more code and math data in later training stages to boost reasoning.

---

### 2.2 Tokenization

Before text enters the model, it must be converted to numbers. This is done by a **tokenizer**.

#### Why Not Just Use Characters or Words?

| Approach | Vocabulary Size | Problem |
|----------|----------------|---------|
| Characters | ~256 | Sequences too long; "hello" = 5 tokens |
| Words | 500,000+ | Too many rare words; can't handle "unforgettable" if unseen |
| **Subwords** | **32K-128K** | **Just right: common words are 1 token, rare words split into pieces** |

#### Byte-Pair Encoding (BPE): The Dominant Algorithm

BPE starts with individual characters and **iteratively merges** the most frequent pairs:

**Step-by-step example:**

Starting text: `"low low low low lowest lowest newer newer wider"` 

**Step 0: Start with characters**
```
Vocabulary: {l, o, w, e, s, t, n, r, i, d, _}
Tokens: l|o|w|_|l|o|w|_|l|o|w|_|l|o|w|_|l|o|w|e|s|t|_|...
```

**Step 1: Most frequent pair is (l, o) → merge to "lo"**
```
Vocabulary: {l, o, w, e, s, t, n, r, i, d, _, lo}
Tokens: lo|w|_|lo|w|_|lo|w|_|lo|w|_|lo|w|e|s|t|_|...
```

**Step 2: Most frequent pair is (lo, w) → merge to "low"**
```
Vocabulary: {l, o, w, e, s, t, n, r, i, d, _, lo, low}
Tokens: low|_|low|_|low|_|low|_|low|e|s|t|_|...
```

**Step 3: Most frequent pair is (e, r) → merge to "er"**
```
Vocabulary: {..., er}
Tokens: low|_|low|_|low|_|low|_|low|e|s|t|_|n|ew|er|_|n|ew|er|_|...
```

This continues for thousands of merges until you reach the desired vocabulary size (typically 32K-128K tokens).

#### SentencePiece: Language-Agnostic Tokenization

While BPE operates on pre-tokenized words, **SentencePiece** (used by Llama, Gemma, T5) treats text as a raw byte stream:

- No language-specific pre-processing needed
- Works for any language (Chinese, Japanese, Arabic, etc.)
- Uses either BPE or **Unigram** algorithm internally
- Handles whitespace as a special character (▁)

```
Input:  "Hello world"
Output: ["▁Hello", "▁world"]        # ▁ represents space

Input:  "unbelievable"
Output: ["▁un", "believ", "able"]    # Subword decomposition

Input:  "こんにちは世界"
Output: ["▁", "こんにちは", "世界"]    # Works for Japanese too
```

#### Tokenization in Practice

```python
# Using the tiktoken library (OpenAI's tokenizer)
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Large language models are fascinating!"
tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Decoded tokens: {[enc.decode([t]) for t in tokens]}")

# Output:
# Text: Large language models are fascinating!
# Tokens: [35353, 4221, 7015, 527, 25555, 0]
# Token count: 6
# Decoded tokens: ['Large', ' language', ' models', ' are', ' fascinating', '!']
```

#### Key Tokenization Concepts

| Concept | Explanation |
|---------|-------------|
| **Vocabulary size** | Total number of unique tokens (32K-128K typical) |
| **Fertility** | Average number of tokens per word (~1.3 for English) |
| **Unknown tokens** | BPE can decompose any text down to bytes  no OOV! |
| **Special tokens** | `<BOS>`, `<EOS>`, `<PAD>`, `<|im_start|>` etc. |
| **Token healing** | Fixing artifacts at token boundaries during generation |

> **Why tokenization matters:** A model with 8K context window and fertility of 1.3 can process ~6,150 English words. But for languages like Chinese, fertility might be 2-3×, effectively shrinking the usable context. Modern tokenizers (GPT-4o, Llama 3) have been optimized for multilingual efficiency.

---

### 2.3 The Training Objective: Next-Token Prediction

The pretraining objective is beautifully simple: **predict the next token**.

#### How It Works

Given a sequence of tokens, the model must predict the probability distribution over all possible next tokens:

```
Input:   "The capital of France is"
Target:  "Paris"

Model output (probability distribution):
  Paris:    0.82  ← highest probability
  Lyon:     0.03
  Berlin:   0.01
  the:      0.02
  a:        0.01
  ...       ...
  (sum = 1.0 across all 128K vocabulary tokens)
```

#### The Loss Function: Cross-Entropy

```
Loss = -log(P(correct_token))

If model predicts P("Paris") = 0.82:
  Loss = -log(0.82) = 0.198  (low loss, good!)

If model predicts P("Paris") = 0.01:
  Loss = -log(0.01) = 4.605  (high loss, bad!)

If model predicts P("Paris") = 1.0:
  Loss = -log(1.0) = 0.0     (perfect, but never happens)
```

#### Why Next-Token Prediction Is So Powerful

This seemingly simple objective forces the model to learn:

1. **Grammar:** To predict "is" after "The cat", you need syntax
2. **Facts:** To predict "Paris" after "capital of France is", you need world knowledge  
3. **Reasoning:** To predict "4" after "2 + 2 =", you need arithmetic
4. **Code:** To predict "}" after a function body, you need program structure
5. **Theory of mind:** To predict dialogue responses, you need to model human intent

```
┌─────────────────────────────────────────────────────────┐
│     "All you need is predict the next token"            │
│                                                         │
│     The ─→ cat ─→ sat ─→ on ─→ the ─→ mat              │
│      │      │      │      │      │      │               │
│      ▼      ▼      ▼      ▼      ▼      ▼              │
│    [cat]  [sat]   [on]  [the]  [mat]   [.]              │
│                                                         │
│     Each position predicts the next token.              │
│     Loss is computed at every position simultaneously.  │
│     This is why training is so parallelizable!          │
└─────────────────────────────────────────────────────────┘
```

#### Autoregressive Generation

At inference time, the model generates text one token at a time:

```
Step 1: Input "Once upon a"     → Predict "time"
Step 2: Input "Once upon a time" → Predict "there"
Step 3: Input "Once upon a time there" → Predict "was"
Step 4: Input "Once upon a time there was" → Predict "a"
...and so on until <EOS> or max length
```

This is called **autoregressive** generation because each output becomes part of the next input.

---

### 2.4 Training Infrastructure

Training a frontier LLM is one of the most compute-intensive tasks humans have ever undertaken.

#### Scale of Modern Training

| Model (Year) | Parameters | Training Tokens | GPUs | Training Time | Estimated Cost |
|--------------|-----------|-----------------|------|--------------|---------------|
| GPT-3 (2020) | 175B | 300B | ~1,000 V100 | ~34 days | $4-5M |
| Llama 2 70B (2023) | 70B | 2T | 2,048 A100 | ~80 days | $3-5M |
| Llama 3 405B (2024) | 405B | 15T | 16,384 H100 | ~54 days | $60-100M |
| GPT-4 (2023) | ~1.8T (MoE) | ~13T | ~25,000 A100 | ~90 days | $80-100M |
| Llama 4 Behemoth (2025) | ~2T (MoE) | ~30T | ~32,000 H100 | ~months | $100M+ |

#### Distributed Training Strategies

No single GPU can hold a frontier model. Training is distributed across thousands of GPUs:

```
┌──────────────────────────────────────────────────────────────┐
│              Distributed Training Strategies                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  DATA PARALLELISM          MODEL PARALLELISM                 │
│  (Same model, different    (Different parts of model         │
│   data on each GPU)         on different GPUs)               │
│                                                              │
│  ┌─────┐ ┌─────┐          ┌─────┐ ┌─────┐ ┌─────┐            │
│  │GPU 0│ │GPU 1│          │GPU 0│→│GPU 1│→│GPU 2│            │
│  │Full │ │Full │          │Layers│ │Layers│ │Layers│         │
│  │Model│ │Model│          │ 1-10 │ │11-20 │ │21-30 │         │
│  │     │ │     │          └─────┘ └─────┘ └─────┘            │    
│  │Batch│ │Batch│          ← Pipeline Parallelism →           │
│  │ A   │ │ B   │                                             │
│  └─────┘ └─────┘          TENSOR PARALLELISM                 │
│                            (Split individual layers)         │
│  Gradients synced          ┌─────┬─────┐                     │
│  after each step           │GPU 0│GPU 1│                     │
│                            │Col A│Col B│ ← One matrix        │
│                            │     │     │   split across      │
│                            └─────┴─────┘   GPUs              │
└──────────────────────────────────────────────────────────────┘
```

**In practice, frontier labs use all three simultaneously**  a technique called **3D parallelism**:

- **Data parallelism (DP):** Replicate model across groups, each processing different data
- **Pipeline parallelism (PP):** Split layers across GPUs in a pipeline
- **Tensor parallelism (TP):** Split individual weight matrices across GPUs

#### Key Training Concepts

| Concept | What It Is | Typical Values |
|---------|-----------|----------------|
| **Learning rate** | Step size for weight updates | 1e-4 to 6e-4, with warmup + cosine decay |
| **Batch size** | Samples per gradient update | Millions of tokens (ramps up during training) |
| **Gradient accumulation** | Simulating larger batches | 4-64 micro-batches per step |
| **Mixed precision** | Using FP16/BF16 for speed | BF16 dominant (better dynamic range) |
| **Gradient checkpointing** | Trade compute for memory | Recompute activations instead of storing |
| **Flash Attention** | Memory-efficient attention | 2-4× speedup, O(N) memory |
| **Context length** | Tokens per training sequence | 4K-8K base, extended later |

#### Training Stability

Training runs costing $100M+ **cannot afford to crash**. Key stability techniques:

```
Loss
 │
 │\
 │ \      ╱── Loss spike! (common)
 │  \    ╱    Usually caused by bad data
 │   \──╱     or numerical instability
 │    \
 │     \───── Gradual decrease
 │          \  over weeks/months
 │           \
 │            \______ Convergence
 │
 └──────────────────── Training Steps

 Recovery strategies:
 1. Roll back to checkpoint before spike
 2. Skip the problematic data batch  
 3. Lower learning rate temporarily
 4. Investigate and filter bad data
```

---

### 2.5 Scaling Laws

One of the most important discoveries in modern AI: model performance follows **predictable mathematical relationships** with scale.

#### The Chinchilla Scaling Laws (2022)

DeepMind's Chinchilla paper established that for a fixed compute budget, you should scale model size and data roughly equally:

```
Optimal tokens ≈ 20 × parameters

Examples:
  1B parameter model  → train on  20B tokens
  7B parameter model  → train on 140B tokens
  70B parameter model → train on  1.4T tokens
```

#### The Power Law

Test loss decreases as a power law with compute, data, and parameters:

```
L(N) = (N_c / N)^α_N    (scaling with parameters)
L(D) = (D_c / D)^α_D    (scaling with data)
L(C) = (C_c / C)^α_C    (scaling with compute)

Where:
  L = test loss
  N = number of parameters
  D = number of training tokens
  C = compute in FLOPs
  α ≈ 0.05-0.1 (the exponent is small  diminishing returns)
```

#### What This Means in Practice

```
Loss │
     │ ●
     │  ●
     │    ●
     │      ●                  Each 10× increase in compute
     │         ●               gives roughly the same
     │            ●            improvement in loss
     │                ●
     │                     ●
     │                           ●
     └──────────────────────────────── Log(Compute)

     10^18    10^20    10^22    10^24    10^26
     FLOPs    FLOPs    FLOPs    FLOPs    FLOPs
     (tiny)  (small)  (medium) (large) (frontier)
```

#### Over-Training: The Modern Approach

Post-Chinchilla, the trend shifted toward **over-training**: training smaller models on much more data than the scaling laws suggest:

| Model | Parameters | Tokens | Tokens/Param Ratio | Why |
|-------|-----------|--------|-------------------|-----|
| Chinchilla | 70B | 1.4T | 20× | Optimal per scaling laws |
| Llama 2 7B | 7B | 2T | 286× | Cheaper inference at scale |
| Llama 3 8B | 8B | 15T | 1,875× | Even more over-trained |
| Phi-3 mini | 3.8B | 3.3T | 868× | Small but capable |
| Gemma 2 2B | 2.6B | 2T | 769× | Efficient deployment |

**The reasoning:** A model that's slightly less loss-optimal but **much smaller** is cheaper to deploy. If you're serving billions of queries, inference cost dominates training cost.

---

## 3. Stage 2: Supervised Fine-Tuning

After pretraining, you have a base model that's excellent at completing text but terrible at following instructions. The model might respond to "What's 2+2?" with "What's 2+3? What's 2+4?..." because it's been trained to *continue* text, not *answer* questions.

### 3.1 Supervised Fine-Tuning (SFT)

SFT transforms a text-completion engine into an instruction-following assistant.

#### The Training Data

SFT uses **(instruction, response)** pairs:

```json
{
  "instruction": "Explain photosynthesis in simple terms.",
  "response": "Photosynthesis is how plants make their own food using sunlight. 
    The plant's leaves contain chlorophyll (the green pigment), which captures 
    light energy from the sun. The plant then combines this energy with water 
    from the soil and carbon dioxide from the air to create glucose (sugar) 
    for energy, releasing oxygen as a byproduct. Think of it like a solar-
    powered kitchen where the plant cooks its own meals!"
}
```

#### Scale of SFT Data

Surprisingly, you don't need much:

| Dataset | Size | Notable For |
|---------|------|-------------|
| InstructGPT (OpenAI) | ~13K | Proved SFT works with small data |
| FLAN Collection | ~1.8M | Massive multi-task instruction set |
| Open Assistant | ~160K | Community-generated conversations |
| Alpaca (Stanford) | 52K | GPT-4-generated, showed synthetic data works |
| Orca (Microsoft) | ~5M | Chain-of-thought explanation traces |
| UltraChat | ~1.5M | Multi-turn synthetic conversations |

> **Key insight:** As few as **1,000-10,000 high-quality examples** can dramatically change model behavior. Quality matters far more than quantity for SFT.

#### What SFT Changes

```
Before SFT (base model):
  User: "Translate 'hello' to French"
  Model: "Translate 'goodbye' to French
          Translate 'thank you' to French
          Translate 'please' to French..."
  
  (It continues the pattern, not answering the question)

After SFT:
  User: "Translate 'hello' to French"
  Model: "The French translation of 'hello' is 'bonjour'."
  
  (It follows the instruction and gives a direct answer)
```

### 3.2 Instruction Tuning

Instruction tuning is a specific form of SFT that focuses on making the model follow diverse instructions:

#### Multi-task Instruction Format

```
Task 1 - Summarization:
  Instruction: "Summarize the following article in 3 sentences: [article text]"
  Response: "[3-sentence summary]"

Task 2 - Translation:
  Instruction: "Translate to Spanish: 'The weather is nice today'"
  Response: "El clima está agradable hoy"

Task 3 - Code:
  Instruction: "Write a Python function to check if a number is prime"
  Response: "```python\ndef is_prime(n):\n    if n < 2:\n        return False..."

Task 4 - Reasoning:
  Instruction: "If John is taller than Mary, and Mary is taller than Bob, 
                who is the shortest?"
  Response: "Let me work through this step by step:\n1. John > Mary (height)\n
             2. Mary > Bob (height)\nTherefore, Bob is the shortest."

Task 5 - Creative:
  Instruction: "Write a haiku about machine learning"
  Response: "Weights slowly converge\nGradients flow like rivers\nLoss descends to peace"
```

#### FLAN: The Instruction Tuning Pioneer

Google's FLAN (Fine-tuned Language Net) showed that training on many tasks with instructions dramatically improves **zero-shot** performance on new tasks:

```
                  ┌───────────────────────────────────────┐
                  │    Zero-shot performance comparison     │
                  ├───────────────────────────────────────┤
                  │                                       │
                  │  Task never seen during fine-tuning:   │
                  │  "Classify this review as positive     │
                  │   or negative"                        │
                  │                                       │
                  │  Base model accuracy:     54%          │
                  │  FLAN-tuned accuracy:     82%  ↑↑↑    │
                  │                                       │
                  │  The model generalizes instruction-    │
                  │  following to NEW tasks!               │
                  └───────────────────────────────────────┘
```

### 3.3 Chat Templates and Special Tokens

Modern chat models use **structured templates** to distinguish between system prompts, user messages, and assistant responses.

#### ChatML Format (OpenAI-style)

```
<|im_start|>system
You are a helpful assistant that speaks like a pirate.<|im_end|>
<|im_start|>user
What's the weather like?<|im_end|>
<|im_start|>assistant
Arrr, matey! I be just a humble AI and can't check the weather 
for ye, but I'd suggest lookin' out the porthole or checkin' 
yer weather app! Fair winds to ye! ⛵<|im_end|>
```

#### Llama Chat Format

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'm doing well, thank you for asking! How can I help you today?<|eot_id|>
```

#### Why Templates Matter

```
Without template:                    With template:
─────────────────                    ──────────────
"User: Hi\nAssistant: Hello"        <|im_start|>user\nHi<|im_end|>
                                     <|im_start|>assistant\nHello<|im_end|>

Problem: Model might generate         Special tokens are NEVER
"User:" in its response,              generated naturally 
breaking the conversation             clean role separation!
```

#### Special Tokens Summary

| Token | Purpose | Example |
|-------|---------|---------|
| `<BOS>` / `<s>` | Beginning of sequence | Marks start of input |
| `<EOS>` / `</s>` | End of sequence | Model stops generating here |
| `<PAD>` | Padding | Fills shorter sequences in batches |
| `<|im_start|>` | Start of message | Begins a chat message |
| `<|im_end|>` | End of message | Ends a chat message |
| `<|system|>` | System role marker | Identifies system instructions |
| `<|tool_call|>` | Tool use marker | Model wants to call a function |
| `<|eot_id|>` | End of turn | Llama-style turn boundary |

---

## 4. Stage 3: Alignment

SFT makes the model follow instructions, but it doesn't make it *good* at following them. The model might:
- Give confidently wrong answers
- Help with harmful requests
- Be verbose when brevity is better
- Not match human preferences for style and tone

**Alignment** techniques teach the model what humans actually prefer.

### 4.1 RLHF: Reinforcement Learning from Human Feedback

RLHF was the breakthrough technique that made ChatGPT feel so much better than previous chatbots.

#### The Three Phases of RLHF

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF Pipeline                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: Collect Human Preferences                             │
│  ┌──────────────────────────────────────────┐                   │
│  │  Prompt: "Explain quantum computing"      │                   │
│  │                                           │                   │
│  │  Response A: "Quantum computing uses      │                   │
│  │  qubits that can be 0 and 1 at the        │                   │
│  │  same time..."                            │                   │
│  │                                           │  Human picks      │
│  │  Response B: "It's like magic computers   │  A > B            │
│  │  that are really fast..."                 │                   │
│  │                                           │                   │
│  │  [A is better] [B is better] [Tie]        │                   │
│  └──────────────────────────────────────────┘                   │
│                        │                                         │
│                        ▼                                         │
│  PHASE 2: Train Reward Model                                    │
│  ┌──────────────────────────────────────────┐                   │
│  │  Reward Model learns to predict           │                   │
│  │  which response humans prefer             │                   │
│  │                                           │                   │
│  │  Input: (prompt, response) → Score        │                   │
│  │                                           │                   │
│  │  "Explain quantum..."  + Response A → 0.8 │                   │
│  │  "Explain quantum..."  + Response B → 0.3 │                   │
│  └──────────────────────────────────────────┘                   │
│                        │                                         │
│                        ▼                                         │
│  PHASE 3: Optimize Policy with PPO                              │
│  ┌──────────────────────────────────────────┐                   │
│  │  LLM generates response                  │                   │
│  │       ↓                                   │                   │
│  │  Reward Model scores it                   │                   │
│  │       ↓                                   │                   │
│  │  PPO updates LLM weights to               │                   │
│  │  maximize reward score                    │                   │
│  │       ↓                                   │                   │
│  │  KL penalty prevents model from           │                   │
│  │  diverging too far from SFT model         │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 1: Collecting Human Preferences

Human annotators compare pairs of model outputs:

```
Prompt: "Write a poem about rain"

Response A:                          Response B:
"Rain falls from the sky,           "Drops cascade like silver threads
 Wet and cold it makes things,       Upon the window's canvas spread,
 Water everywhere."                  Each one a whispered lullaby
                                     That puts the waking world to bed."

Human judgment: B >> A
Reason: More creative, better imagery, proper poetic structure
```

Typical annotation guidelines ask raters to consider:
- **Helpfulness:** Does it answer the question well?
- **Honesty:** Is the information accurate?
- **Harmlessness:** Does it avoid harmful content?
- **Quality:** Is the writing clear and well-structured?

#### Phase 2: Training the Reward Model

The reward model is typically a transformer (often initialized from the SFT model) with a scalar output head:

```python
# Conceptual reward model training
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.transformer = base_model  # Same architecture as LLM
        self.reward_head = nn.Linear(hidden_dim, 1)  # Single score
    
    def forward(self, prompt, response):
        hidden = self.transformer(prompt + response)
        reward = self.reward_head(hidden[-1])  # Score from last token
        return reward

# Training loss: make preferred response score higher
# Bradley-Terry model:
# Loss = -log(sigmoid(r(preferred) - r(rejected)))
```

#### Phase 3: PPO Optimization

PPO (Proximal Policy Optimization) is the RL algorithm used to update the LLM:

```
Objective = E[Reward(response)] - β × KL(π_current ‖ π_sft)

Where:
  Reward(response) = Score from reward model
  KL(...) = How much the model has changed from the SFT version
  β = Controls the strength of the KL penalty

The KL penalty is crucial:
  - Without it, the model would "hack" the reward model
  - Example: Always saying "Great question!" gets high reward
    scores but isn't actually helpful
  - KL penalty keeps the model close to its SFT foundation
```

#### Why RLHF Works

```
Before RLHF:                         After RLHF:
─────────────                         ──────────
Q: "Is 2+2=5?"                       Q: "Is 2+2=5?"

"That's an interesting                "No, 2+2 equals 4, not 5.
 mathematical question.                This is a basic arithmetic
 The answer depends on                fact. 2+2=5 is incorrect."
 the axiomatic system..."
                                      (Direct, honest, clear)
(Hedging, not committing)
```

### 4.2 DPO: Direct Preference Optimization

DPO (2023) simplified RLHF by **eliminating the reward model entirely**:

```
┌─────────────────────────────────────────────────────────────┐
│                  RLHF vs DPO                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RLHF (3 steps):              DPO (1 step):                │
│  1. Train reward model        1. Directly optimize LLM     │
│  2. Generate responses           on preference pairs        │
│  3. PPO optimization                                        │
│                                                             │
│  Complex, unstable,           Simple, stable,               │
│  requires RL expertise        just supervised learning!     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### DPO Loss Function

```
L_DPO = -log σ(β × (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

Where:
  y_w = winning (preferred) response
  y_l = losing (rejected) response
  π = current model
  π_ref = reference (SFT) model
  β = temperature parameter
  σ = sigmoid function

Intuition: Increase probability of preferred responses
          and decrease probability of rejected ones,
          relative to the reference model.
```

#### DPO in Practice

```python
# Conceptual DPO training loop
for prompt, response_preferred, response_rejected in dataset:
    
    # Get log probabilities from current model
    log_p_preferred = model.log_prob(response_preferred | prompt)
    log_p_rejected  = model.log_prob(response_rejected  | prompt)
    
    # Get log probabilities from reference (frozen SFT) model
    log_ref_preferred = ref_model.log_prob(response_preferred | prompt)
    log_ref_rejected  = ref_model.log_prob(response_rejected  | prompt)
    
    # DPO loss
    log_ratio_preferred = log_p_preferred - log_ref_preferred
    log_ratio_rejected  = log_p_rejected  - log_ref_rejected
    
    loss = -log_sigmoid(beta * (log_ratio_preferred - log_ratio_rejected))
    loss.backward()
    optimizer.step()
```

#### RLHF vs DPO Comparison

| Aspect               | RLHF (PPO)                         | DPO                               |
| -------------------- | ---------------------------------- | --------------------------------- |
| Reward model needed? | Yes                                | No                                |
| RL expertise needed? | Yes                                | No (standard supervised learning) |
| Training stability   | Can be unstable                    | Generally stable                  |
| Computational cost   | High (multiple models in memory)   | Lower                             |
| Performance          | Slightly better on some benchmarks | Competitive, sometimes better     |
| Adoption (2025+)     | Still used at frontier             | Dominant for open models          |
| Variants             | REINFORCE, GRPO                    | IPO, KTO, ORPO, SimPO             |

### 4.3 Constitutional AI (Anthropic's Approach)

Constitutional AI (CAI) is Anthropic's approach to alignment, which reduces reliance on human annotators:

#### The Key Idea

Instead of humans rating every response, give the AI a set of **principles (a "constitution")** and let it critique and revise its own outputs:

```
┌──────────────────────────────────────────────────────────────┐
│                  Constitutional AI Pipeline                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Generate initial response                           │
│  ────────────────────────────────                            │
│  User: "How do I pick a lock?"                               │
│  Model: "Here's how to pick a lock: First, you need          │
│          a tension wrench and pick..."                        │
│                                                              │
│  Step 2: AI Self-Critique (using constitutional principles)  │
│  ──────────────────────────────────────────────────────       │
│  Principle: "Choose the response that is least likely         │
│  to facilitate illegal activity."                            │
│                                                              │
│  Critique: "My response provides detailed instructions       │
│  that could facilitate illegal breaking and entering.        │
│  I should instead provide a response that acknowledges       │
│  the question while being responsible."                      │
│                                                              │
│  Step 3: AI Self-Revision                                    │
│  ────────────────────────                                    │
│  Revised: "Lock picking is a skill used by licensed          │
│  locksmiths. If you're locked out, I'd recommend             │
│  calling a professional locksmith. If you're interested      │
│  in the hobby, look into lock sport communities that         │
│  practice on practice locks."                                │
│                                                              │
│  Step 4: Train on (original, revised) pairs using RLHF/DPO  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### Example Constitutional Principles

```
1. "Choose the response that is most helpful to the human."
2. "Choose the response that is least likely to cause harm."
3. "Choose the response that is most honest and truthful."
4. "Choose the response that most clearly acknowledges 
    uncertainty when appropriate."
5. "Choose the response that is least likely to be used 
    for illegal or harmful purposes."
6. "Choose the response that best respects the autonomy 
    and dignity of all people."
```

### 4.4 RLAIF: RL from AI Feedback

RLAIF extends Constitutional AI by using **AI-generated preferences** to train the reward model:

```
Traditional RLHF:
  Human annotators → Preference data → Reward model → RL training

RLAIF:
  AI model + principles → Preference data → Reward model → RL training

Hybrid (most common):
  Mix of human + AI preferences → Reward model → RL training
```

#### Why RLAIF?

| Factor | Human Feedback | AI Feedback |
|--------|---------------|-------------|
| **Cost** | $$$$ (annotator salaries) | $ (API costs) |
| **Speed** | Slow (weeks) | Fast (hours) |
| **Scale** | 10K-100K examples | Millions of examples |
| **Consistency** | Variable between annotators | More consistent |
| **Quality ceiling** | Human-level | May miss nuances |
| **Best for** | High-stakes alignment | Scaling alignment broadly |

#### The Virtuous Cycle

```
Better Model → Better AI Feedback → Better Training → Better Model → ...

This is why models keep improving:
- GPT-4 helps train GPT-4's successors
- Claude helps generate Constitutional AI data for next Claude
- Each generation bootstraps the next
```

---

## 5. Advanced Techniques

### 5.1 Synthetic Training Data

One of the most important trends in modern LLM training: **using AI to generate training data for AI**.

#### Types of Synthetic Data

```
┌─────────────────────────────────────────────────────────────┐
│              Synthetic Data Strategies                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DISTILLATION DATA                                       │
│     Strong model → generates responses → trains weak model  │
│     Example: GPT-4 responses → train 7B model               │
│                                                             │
│  2. SELF-PLAY / SELF-IMPROVEMENT                            │
│     Model generates + critiques its own outputs             │
│     Example: STaR (Self-Taught Reasoner)                    │
│                                                             │
│  3. TEXTBOOK-STYLE DATA                                     │
│     LLM writes synthetic "textbooks" on specific topics     │
│     Example: Microsoft's Phi models                         │
│                                                             │
│  4. INSTRUCTION AUGMENTATION                                │
│     LLM generates diverse instructions from seed examples   │
│     Example: Self-Instruct, Evol-Instruct (WizardLM)       │
│                                                             │
│  5. VERIFICATION-BASED                                      │
│     Generate many solutions, keep only verified correct ones │
│     Example: Math/code where answers can be checked          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Case Study: Microsoft's Phi Models

Microsoft's Phi series demonstrated that **"textbooks are all you need"**:

```
Phi-1 (1.3B params, 2023):
  - Trained on 6B tokens of "textbook quality" synthetic data
  - Competitive with models 10× larger on coding benchmarks
  
Phi-2 (2.7B params, 2023):
  - 1.4T tokens of synthetic + filtered web data
  - Matched GPT-3.5 on many benchmarks
  
Phi-3 mini (3.8B params, 2024):
  - 3.3T tokens of heavily filtered + synthetic data
  - Rivaled Llama 3 8B (2× the parameters)
  
Phi-4 (14B params, 2025):
  - Extensive use of synthetic training data
  - Strong reasoning capabilities for its size
```

**Key insight:** Carefully curated synthetic data can be 10-100× more token-efficient than raw web data.

### 5.2 Knowledge Distillation

Training a smaller model to mimic a larger one:

```
┌──────────────┐         ┌──────────────┐
│  Teacher     │         │   Student    │
│  (Large      │ ──────→ │   (Small     │
│   Model)     │ Teaches  │    Model)    │
│  405B params │         │   8B params  │
└──────────────┘         └──────────────┘

What the teacher provides:
1. Output probabilities (soft labels)  richer than hard labels
2. Response text  for training on completions
3. Reasoning traces  showing step-by-step thinking
```

#### Why Soft Labels Help

```
Hard label:  "Paris" = [0, 0, 0, 1, 0, 0, ...]
                                 ↑ Paris

Soft label from teacher:
  "Paris":     0.80
  "Lyon":      0.05    ← The teacher reveals that "Lyon" 
  "Marseille": 0.03       is more plausible than "Tokyo"
  "London":    0.02       This is useful information!
  "Tokyo":     0.001
  ...

The student learns from the teacher's uncertainty,
not just the correct answer.
```

#### Distillation Techniques

| Technique | Description | Example |
|-----------|-------------|---------|
| **Logit distillation** | Match teacher's output distribution | Classic KD |
| **Feature distillation** | Match intermediate layer representations | TinyBERT |
| **Response distillation** | Train on teacher's text outputs | Alpaca, Orca |
| **Chain-of-thought distillation** | Teacher shows reasoning steps | Orca 2, reasoning models |
| **On-policy distillation** | Student generates, teacher scores | GKD |

### 5.3 LoRA and Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates **all** model parameters. For a 70B model, this requires enormous GPU memory. **PEFT methods** update only a small fraction of parameters.

#### LoRA: Low-Rank Adaptation

The key idea: instead of updating a weight matrix W directly, add a low-rank decomposition:

```
Standard fine-tuning:
  W' = W + ΔW          (ΔW is full-rank: d × d matrix)
  
  For d=4096: ΔW has 4096 × 4096 = 16.7M parameters per layer

LoRA:
  W' = W + B × A       (A is d × r, B is r × d, where r << d)
  
  For d=4096, r=16: A has 4096×16 = 65K, B has 16×4096 = 65K
  Total: 131K parameters per layer (128× fewer!)
```

#### Visual Explanation

```
Standard Fine-Tuning:           LoRA:
┌──────────────┐                ┌──────────────┐
│              │                │              │
│   W + ΔW     │                │      W       │  (frozen!)
│              │                │      +       │
│  (all params │                │   B × A      │  (only A,B trained)
│   updated)   │                │              │
│              │                │  rank r=16   │
│  16.7M params│                │  131K params │
│  per layer   │                │  per layer   │
└──────────────┘                └──────────────┘

Memory comparison for 70B model:
  Full fine-tuning:  ~280 GB (4 bytes × 70B params)
  LoRA (r=16):       ~160 MB (0.1% of parameters)
```

#### QLoRA: Quantized LoRA

QLoRA (2023) combines LoRA with quantization for even more efficiency:

```
Full fine-tuning of 70B model:  8× A100 80GB GPUs (~$200K hardware)
LoRA fine-tuning of 70B model:  2× A100 80GB GPUs
QLoRA fine-tuning of 70B model: 1× A100 48GB GPU   ← Single GPU!

How QLoRA works:
1. Quantize base model to 4-bit (NF4 data type)
2. Add LoRA adapters in FP16/BF16
3. Train only the LoRA adapters
4. Gradients flow through quantized weights (via double quantization)
```

#### PEFT Methods Comparison

| Method | Trainable Params | Memory | Quality | Speed |
|--------|-----------------|--------|---------|-------|
| Full fine-tuning | 100% | Very High | Best | Slow |
| LoRA | 0.1-1% | Low | Near-full FT | Fast |
| QLoRA | 0.1-1% | Very Low | Slightly below LoRA | Fast |
| Prefix Tuning | <0.1% | Very Low | Good for specific tasks | Very Fast |
| Adapters | 1-5% | Low | Good | Fast |
| Prompt Tuning | <0.01% | Minimal | Moderate | Very Fast |

### 5.4 Model Merging

A surprisingly effective technique: **combining weights from multiple fine-tuned models** without additional training.

#### How It Works

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Model A     │  │  Model B     │  │  Model C     │
│  (fine-tuned │  │  (fine-tuned │  │  (fine-tuned │
│   for code)  │  │   for math)  │  │   for chat)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────────┬────┘────────────────┘
                    │
                    ▼
           ┌──────────────┐
           │ Merged Model │
           │ (good at all │
           │  three!)     │
           └──────────────┘
```

#### Merging Techniques

| Method | Formula | Description |
|--------|---------|-------------|
| **Linear (LERP)** | W = α×W_A + (1-α)×W_B | Simple weighted average |
| **SLERP** | Spherical interpolation | Better for high-dimensional spaces |
| **TIES** | Trim, Elect Sign, Merge | Resolves conflicting parameter updates |
| **DARE** | Random drop + rescale | Randomly drops delta params before merging |
| **Model Stock** | Geometric mean of fine-tuned weights | Treats weights as points in weight space |

#### Why Merging Works (Intuition)

```
Think of the loss landscape as a mountain range:

      /\    /\
     /  \  /  \    /\
    /    \/    \  /  \
   /      A     \/    \
  /               B    \
 /                      \

Model A and B are in different valleys (solutions).
If they started from the same base model and the valleys
are connected by a relatively flat path, then the average
of A and B lands in a good region too!

This is called "linear mode connectivity" and it works
because fine-tuning doesn't move weights very far from
the base model.
```

#### Merging in Practice (using mergekit)

```yaml
# mergekit configuration example
models:
  - model: codellama/CodeLlama-7b-Instruct-hf
    parameters:
      weight: 0.4
  - model: WizardLM/WizardMath-7B-V1.0
    parameters:
      weight: 0.3
  - model: meta-llama/Llama-2-7b-chat-hf
    parameters:
      weight: 0.3

merge_method: linear  # or slerp, ties, dare_ties
dtype: float16
```

This creates a model that's good at code, math, AND chat  without any additional training!

---

## 6. The Complete Pipeline Diagram

Here is the complete journey from raw internet text to a production chatbot:

```
═══════════════════════════════════════════════════════════════════════
                    THE COMPLETE LLM PIPELINE
═══════════════════════════════════════════════════════════════════════

STAGE 0: DATA PREPARATION
─────────────────────────
  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Common      │  │  GitHub  │  │  Books   │  │ Wikipedia│
  │ Crawl       │  │  Code    │  │          │  │          │
  │ (Web text)  │  │          │  │          │  │          │
  └──────┬──────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
         │              │             │              │
         └──────────────┴─────────────┴──────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Data Pipeline       │
                   │  • Deduplication     │
                   │  • Quality filtering │
                   │  • PII removal       │
                   │  • Tokenization      │
                   └──────────┬──────────┘
                              │
                   ~5-15 Trillion tokens
                              │
═══════════════════════════════════════════════════════════════════════

STAGE 1: PRETRAINING (~months, $10M-$100M+)
───────────────────
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Next-Token        │
                   │   Prediction        │
                   │                     │
                   │   "The cat sat" → ? │
                   │   Target: "on"      │
                   │                     │
                   │   1000s of GPUs     │
                   │   Weeks to months   │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   BASE MODEL        │
                   │   • Knows language   │
                   │   • Knows facts      │
                   │   • Completes text   │
                   │   • NOT a chatbot    │
                   └──────────┬──────────┘
                              │
═══════════════════════════════════════════════════════════════════════

STAGE 2: SUPERVISED FINE-TUNING (~days, $1K-$100K)
──────────────────────────────
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Instruction/       │
                   │   Response Pairs     │
                   │                     │
                   │   "Summarize X" →   │
                   │   "[summary]"        │
                   │                     │
                   │   10K-1M examples   │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   INSTRUCT MODEL    │
                   │   • Follows instrs  │
                   │   • Answers Qs      │
                   │   • Sometimes       │
                   │     robotic/unsafe  │
                   └──────────┬──────────┘
                              │
═══════════════════════════════════════════════════════════════════════

STAGE 3: ALIGNMENT (~days-weeks, $10K-$1M)
──────────────────
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
         ┌──────────────────┐  ┌──────────────────┐
         │   RLHF/PPO       │  │   DPO            │
         │   • Train reward  │  │   • Direct       │
         │     model         │  │     preference   │
         │   • RL training   │  │     optimization │
         │   • Complex       │  │   • Simpler      │
         └────────┬─────────┘  └────────┬─────────┘
                  │                     │
                  └──────────┬──────────┘
                             │
                  ┌──────────┴─────────────┐
                  │  + Constitutional AI    │
                  │  + Safety training      │
                  │  + Red-teaming          │
                  └──────────┬─────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   ALIGNED CHATBOT   │
                  │   • Helpful          │
                  │   • Harmless         │
                  │   • Honest           │
                  │   • Natural dialogue │
                  │   • Refuses harmful  │
                  │     requests         │
                  └──────────┬──────────┘
                             │
═══════════════════════════════════════════════════════════════════════

STAGE 4: DEPLOYMENT & OPTIMIZATION
───────────────────────────────────
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
         ┌──────────────────┐  ┌──────────────┐
         │  Quantization     │  │  Distillation│
         │  (FP16→INT4)      │  │  (70B→8B)    │
         └────────┬─────────┘  └────────┬─────┘
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │  PRODUCTION MODEL   │
                  │  • Optimized for    │
                  │    latency          │
                  │  • Served at scale  │
                  │  • Monitored        │
                  │  • Continuously     │
                  │    improved         │
                  └─────────────────────┘

═══════════════════════════════════════════════════════════════════════
```

### Cost Breakdown of the Pipeline

| Stage | Time | Compute Cost | Human Cost | Data Size |
|-------|------|-------------|------------|-----------|
| Data Prep | Weeks | $10K-100K | Engineers only | Petabytes → Terabytes |
| Pretraining | Months | $10M-100M+ | Engineers only | Trillions of tokens |
| SFT | Days | $1K-100K | Annotators for data | 10K-1M examples |
| Alignment | Days-Weeks | $10K-1M | Many annotators | 100K+ comparisons |
| Deployment | Ongoing | $1M+/month | Ops team | N/A |

### How Open-Source Models Make This Accessible

```
Frontier lab pipeline:           Open-source pipeline:
  Data: Proprietary crawl          Data: FineWeb, RedPajama
  Pretrain: From scratch           Pretrain: Use Llama/Mistral base
  SFT: Proprietary data            SFT: Open datasets + synthetic
  RLHF: Expert annotators          DPO: Open preference datasets
  Deploy: Custom infrastructure    Deploy: vLLM, TGI, Ollama
  
  Cost: $100M+                     Cost: $100-$10K (fine-tune only!)
```

---

## 7. Quiz

### Questions

**Q1:** What is the primary training objective during LLM pretraining?
- A) Classifying text into categories
- B) Predicting the next token in a sequence
- C) Translating between languages
- D) Answering questions correctly

**Q2:** Why do modern LLMs use subword tokenization (like BPE) instead of word-level tokenization?
- A) It's faster to compute
- B) It handles rare/unseen words by decomposing them into known subwords
- C) It produces shorter sequences
- D) It's more accurate for English

**Q3:** What is the purpose of the KL divergence penalty in RLHF?
- A) To make training faster
- B) To prevent the model from diverging too far from the SFT model
- C) To reduce the vocabulary size
- D) To improve tokenization

**Q4:** How does DPO differ from RLHF?
- A) DPO uses a larger model
- B) DPO requires more human annotations
- C) DPO eliminates the need for a separate reward model
- D) DPO only works for small models

**Q5:** In LoRA, what does "low-rank" refer to?
- A) The model has fewer layers
- B) The weight updates are decomposed into low-rank matrices, greatly reducing parameters
- C) The model uses lower precision numbers
- D) The training uses fewer GPUs

**Q6:** What is the Chinchilla scaling law's key insight?
- A) Bigger models are always better
- B) For optimal performance per compute dollar, scale model size and data roughly equally
- C) Training should always use the maximum learning rate
- D) More GPUs always means faster training

**Q7:** Which of the following best describes Constitutional AI?
- A) Training AI using government regulations
- B) Using a set of principles to have the AI self-critique and revise its outputs
- C) Training only on constitutionally protected speech
- D) A method for making AI follow the law

**Q8:** Why has the trend shifted toward "over-training" smaller models beyond the Chinchilla-optimal point?
- A) It produces better training data
- B) Smaller models are cheaper to deploy at inference time, which dominates total cost
- C) Larger models are no longer being developed
- D) Over-training prevents catastrophic forgetting

### Answers

**A1:** **B) Predicting the next token in a sequence.** This simple objective forces the model to learn grammar, facts, reasoning, and more  everything needed to predict what comes next in any text.

**A2:** **B) It handles rare/unseen words by decomposing them into known subwords.** BPE can tokenize any word by breaking it into smaller pieces, eliminating out-of-vocabulary problems while keeping common words as single tokens.

**A3:** **B) To prevent the model from diverging too far from the SFT model.** Without the KL penalty, the model could "hack" the reward model by generating degenerate but high-reward text (e.g., always saying "Great question!").

**A4:** **C) DPO eliminates the need for a separate reward model.** DPO directly optimizes the LLM on preference pairs using a clever mathematical reformulation, making alignment much simpler and more stable.

**A5:** **B) The weight updates are decomposed into low-rank matrices, greatly reducing parameters.** Instead of a full d×d update matrix, LoRA uses two smaller matrices (d×r and r×d where r << d), reducing trainable parameters by 100× or more.

**A6:** **B) For optimal performance per compute dollar, scale model size and data roughly equally.** The Chinchilla paper showed that many models were "undertrained"  using too few tokens for their parameter count  and that balance matters.

**A7:** **B) Using a set of principles to have the AI self-critique and revise its outputs.** Anthropic's approach uses a "constitution" of principles that the AI uses to evaluate and improve its own responses, reducing reliance on human annotators.

**A8:** **B) Smaller models are cheaper to deploy at inference time, which dominates total cost.** If you're serving billions of queries, the cost of running a smaller model vastly outweighs the one-time cost of extra training compute.

---

## Summary

In this part, we traced the complete journey from raw internet text to a production AI chatbot:

| Stage | Input | Output | Key Innovation |
|-------|-------|--------|---------------|
| Data Prep | Raw web crawl | Clean, tokenized corpus | Quality filtering, data mixing |
| Pretraining | Clean text | Base model | Next-token prediction at scale |
| SFT | Instruction pairs | Instruction-following model | Few examples, big behavior change |
| Alignment | Human preferences | Helpful, harmless chatbot | RLHF, DPO, Constitutional AI |
| Optimization | Large model | Efficient model | LoRA, quantization, distillation |

The remarkable insight is that each stage builds on the previous one, and the later stages require **dramatically less** data and compute than pretraining. A $100M pretraining run produces a base model that can be fine-tuned and aligned for a few thousand dollars.

In Part 6, we'll explore the diverse architectures that modern AI systems use  from Mixture of Experts to RAG systems to reasoning models  and when to use each one.
