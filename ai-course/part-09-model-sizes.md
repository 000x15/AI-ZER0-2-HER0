# Part 9: Understanding Model Sizes  From Pocket-Sized to Planet-Scale

## Introduction: Why Model Size Matters (And Why It's Not Everything)

Imagine you're hiring for a job. You could hire a fresh graduate, a mid-career professional, or a world-renowned expert. Each comes with different capabilities, costs, and space requirements (offices vary!). Similarly, AI models come in different sizes, each suited for different tasks, budgets, and hardware.

But here's the twist that makes 2025-2026 so exciting: **a well-trained junior can now outperform a poorly-trained senior**. Modern training techniques have made smaller models shockingly capable, upending the old assumption that bigger is always better.

```
The Model Size Spectrum (Parameter Count)

  1B          3B        7-8B       14B       32B        70B       400B+
  │           │          │          │         │          │          │
  ▼           ▼          ▼          ▼         ▼          ▼          ▼
┌─────┐   ┌──────┐  ┌───────┐  ┌──────┐  ┌──────┐  ┌───────┐  ┌────────┐
│Phone│   │Laptop│  │Desktop│  │ Work │  │Pro   │  │Server │  │Data    │
│Watch│   │Tablet│  │ GPU   │  │ GPU  │  │ GPU  │  │ Rack  │  │Center  │
└─────┘   └──────┘  └───────┘  └──────┘  └──────┘  └───────┘  └────────┘
  $0         $0      $200-500  $500-1K   $1K-2K   $2K-10K    $50K+
           (CPU)     (8GB GPU) (12-16GB) (24-48GB) (48-80GB)  (Multi-GPU)
```

### How to Think About Model Sizes

A "parameter" is a single learned number in the model  a weight or bias that was adjusted during training. When we say a model has "7 billion parameters," that means 7,000,000,000 individual numbers define its behavior.

**Quick mental model for memory:**

| Precision | Bytes per Parameter | 7B Model Size | 70B Model Size |
|-----------|-------------------|---------------|----------------|
| FP32 (full) | 4 bytes | ~28 GB | ~280 GB |
| FP16/BF16 (half) | 2 bytes | ~14 GB | ~140 GB |
| INT8 (8-bit quant) | 1 byte | ~7 GB | ~70 GB |
| INT4 (4-bit quant) | 0.5 bytes | ~3.5 GB | ~35 GB |
| GGUF Q4_K_M | ~0.55 bytes | ~4 GB | ~40 GB |

> **Rule of thumb**: To run a model, you need roughly `parameters × bytes_per_param + 1-2 GB overhead` of VRAM (GPU memory) or RAM (for CPU inference).

### What Determines a Model's Capability?

It's not just parameter count. Three factors matter:

1. **Parameters**  The model's capacity to store patterns
2. **Training data**  What and how much the model learned from
3. **Training compute**  How thoroughly the model was trained (measured in FLOPs)

The **Chinchilla scaling law** (2022) showed that many early LLMs were undertrained  they had lots of parameters but hadn't seen enough data. Modern models are often trained on 5-15× more tokens than Chinchilla-optimal, making them far more capable per parameter.

```
Capability ≈ f(Parameters, Training Data Quality, Training Compute, Architecture)

Old thinking:  "70B is always better than 14B"
New thinking:  "A well-trained 14B with great data can beat an older 70B"
```

---

## Tier 1: 1B Models  AI in Your Pocket

### The Landscape

| Model | Parameters | Context Window | Release | Key Strength |
|-------|-----------|---------------|---------|--------------|
| Phi-4-mini | 3.8B* | 128K | 2025 | Microsoft's efficient reasoning |
| Qwen3-0.6B | 0.6B | 32K | 2025 | Ultralight multilingual |
| Llama 3.2 1B | 1.2B | 128K | 2024 | Meta's edge-optimized |
| Gemma 3 1B | 1B | 32K | 2025 | Google's compact model |
| SmolLM2 1.7B | 1.7B | 8K | 2024 | HuggingFace's on-device |

*\*Phi-4-mini is technically 3.8B but is included here as Microsoft markets it in the "mini" edge category and it runs on mobile devices.*

### What They Can Do

These tiny models are surprisingly useful for focused tasks:

- ✅ **Text classification**  Sentiment analysis, topic categorization
- ✅ **Simple summarization**  Condensing short documents
- ✅ **Basic Q&A**  Answering factual questions from provided context
- ✅ **Code completion**  Simple autocomplete suggestions
- ✅ **Named entity recognition**  Extracting names, dates, locations
- ✅ **Translation**  Between common language pairs
- ✅ **Keyword extraction**  Pulling out important terms
- ✅ **Text formatting**  JSON extraction, reformatting

### What They Can't Do

- ❌ **Complex reasoning**  Multi-step logic problems
- ❌ **Long-form generation**  Quality degrades quickly past a few paragraphs
- ❌ **Nuanced understanding**  Sarcasm, irony, implicit context
- ❌ **Complex coding**  Can't write entire functions reliably
- ❌ **Mathematical reasoning**  Struggles with anything beyond basic arithmetic
- ❌ **Reliable instruction following**  Often ignores complex format requests

### Hardware Requirements

```
Minimum Hardware for 1B Models:
┌────────────────────────────────────────────────┐
│  Device          │ RAM/VRAM │ Speed (tok/s)    │
├──────────────────┼──────────┼──────────────────┤
│  Raspberry Pi 5  │  8 GB    │  ~5-10 tok/s     │
│  Smartphone      │  4+ GB   │  ~10-20 tok/s    │
│  Laptop (CPU)    │  4+ GB   │  ~15-30 tok/s    │
│  Laptop (iGPU)   │  4+ GB   │  ~20-40 tok/s    │
│  Desktop GPU     │  4+ GB   │  ~50-100 tok/s   │
└────────────────────────────────────────────────┘
Quantization: Q4_K_M recommended (~0.6-1 GB file size)
```

### Real-World Performance Example

**Task**: Classify customer reviews as positive, negative, or neutral.

```
Input:  "The product arrived on time but the packaging was damaged.
         The item itself works fine though."

1B Model Output:  "Neutral" ← Correct! Simple classification works great.

Input:  "Write a Python function that implements binary search,
         handles edge cases, includes docstrings, and returns
         the index or -1 if not found."

1B Model Output:  [Produces a basic function but misses edge cases,
                    has potential off-by-one errors, incomplete docstring]
```

### Best Use Cases

1. **On-device autocomplete**  Keyboard prediction, email suggestions
2. **Privacy-sensitive classification**  Medical triage, content filtering where data can't leave the device
3. **IoT and embedded**  Smart home devices, robotics with language understanding
4. **Preprocessing pipeline**  Fast classification before routing to larger models
5. **Offline applications**  Translation and summarization without internet

---

## Tier 2: 3B Models  The Laptop-Friendly Powerhouse

### The Landscape

| Model | Parameters | Context Window | Release | Key Strength |
|-------|-----------|---------------|---------|--------------|
| Llama 3.2 3B | 3.2B | 128K | 2024 | Excellent all-rounder |
| Phi-3-mini 3.8B | 3.8B | 128K | 2024 | Strong reasoning for size |
| Gemma 3 4B | 4B | 128K | 2025 | Google's efficient model |
| Qwen3-1.7B | 1.7B | 32K | 2025 | Compact multilingual |
| Qwen3-4B | 4B | 128K | 2025 | Strong thinking model |
| StableLM 3B | 3B | 4K | 2024 | Stability AI's compact LLM |

### Surprising Capabilities

The 3B tier is where models start becoming genuinely useful for real work:

- ✅ **Conversational AI**  Can hold coherent multi-turn conversations
- ✅ **Summarization**  Good quality summaries of articles and documents
- ✅ **Code generation**  Can write simple complete functions
- ✅ **Creative writing**  Short stories, marketing copy, emails
- ✅ **Instruction following**  Reasonably follows format requirements
- ✅ **RAG applications**  Answering questions from provided context
- ✅ **Basic reasoning**  Simple chain-of-thought works

### The "Surprisingly Good" Factor

What surprises most people about 3B models in 2026:

```
2023 Expectation:  "3B models are toys, barely useful"
2026 Reality:      "Gemma 3 4B matches GPT-3.5 on many benchmarks"

This happened because:
┌─────────────────────────────────────────────────────────┐
│  1. Training data improved 10× in quality               │
│  2. Architecture innovations (GQA, RoPE, RMSNorm)       │
│  3. Better tokenizers (less tokens = more efficient)     │
│  4. Distillation from larger models                      │
│  5. Curriculum learning and data mixing strategies       │
└─────────────────────────────────────────────────────────┘
```

### Hardware Requirements

```
Minimum Hardware for 3B Models:
┌────────────────────────────────────────────────┐
│  Device            │ RAM/VRAM │ Speed (tok/s)  │
├────────────────────┼──────────┼────────────────┤
│  Laptop (CPU, M1)  │  8 GB    │  ~15-25 tok/s  │
│  Laptop (CPU, x86) │  8 GB    │  ~10-18 tok/s  │
│  Intel iGPU/Arc    │  8+ GB   │  ~15-30 tok/s  │
│  GTX 1660 (6GB)    │  6 GB    │  ~30-50 tok/s  │
│  RTX 3060 (12GB)   │  12 GB   │  ~40-70 tok/s  │
│  Apple M2 (GPU)    │  8+ GB   │  ~30-50 tok/s  │
└────────────────────────────────────────────────┘
Quantization: Q4_K_M recommended (~2-2.5 GB file size)
```

### Real-World Performance Example

**Task**: Generate a professional email response.

```
Input:  "Write a polite reply declining a meeting invitation
         for next Tuesday, suggesting Thursday instead."

3B Model Output:
"Dear [Name],

Thank you for the meeting invitation for next Tuesday. Unfortunately,
I have a prior commitment at that time and won't be able to attend.

Would Thursday at the same time work for you instead? I'm flexible
on the time if that day works better.

Please let me know, and I apologize for any inconvenience.

Best regards,
[Your Name]"

← Clean, professional, appropriate. Perfectly usable!
```

### Best Use Cases

1. **Personal AI assistant**  Running on your laptop, always available
2. **Developer tools**  Code completion in IDEs, Git commit messages
3. **Content drafting**  First drafts of emails, reports, social media
4. **Chatbots**  Customer service bots for common queries
5. **Education**  Tutoring, explaining concepts, quiz generation
6. **Local RAG**  Question-answering over personal documents

---

## Tier 3: 7-8B Models  The Sweet Spot for Local AI

### The Landscape

| Model | Parameters | Context Window | Release | Key Strength |
|-------|-----------|---------------|---------|--------------|
| Llama 3.1 8B | 8B | 128K | 2024 | Meta's workhorse |
| Qwen 2.5 7B | 7.6B | 128K | 2025 | Excellent multilingual + code |
| Gemma 2 9B | 9.2B | 8K | 2024 | Strong benchmark performer |
| Mistral 7B v0.3 | 7.2B | 32K | 2024 | The model that started small LLM revolution |
| Llama 3.2 8B | 8B | 128K | 2024 | Multimodal capable |
| DeepSeek-R1-Distill-Qwen-7B | 7B | 64K | 2025 | Reasoning distilled |
| Qwen3-8B | 8B | 128K | 2025 | Latest generation with thinking |

### Why 7-8B Is the Sweet Spot

```
The Goldilocks Zone:
                                    ┌─────────────┐
  Too Small ◄───────────────────────┤  7-8B       ├──────────────────► Too Big
  (1-3B)                            │  "Just Right"│                   (70B+)
  - Limited capability              └──────┬──────┘                   - Needs expensive GPU
  - Can't handle complexity               │                           - Slow on consumer HW
  - Misses nuance                          │                           - High power draw
                                           │
                              ┌─────────────┴──────────────┐
                              │  ✓ Runs on consumer GPUs    │
                              │  ✓ Fast inference           │
                              │  ✓ Good quality output      │
                              │  ✓ Handles most daily tasks  │
                              │  ✓ Fine-tuning is feasible  │
                              │  ✓ Huge community support   │
                              └────────────────────────────┘
```

### Capabilities

The 7-8B tier is where models become genuinely competitive with cloud APIs for many tasks:

- ✅ **Strong coding**  Can write complete functions, debug, explain code
- ✅ **Analysis & reasoning**  Can break down problems step-by-step
- ✅ **Creative writing**  Stories, poems, marketing copy with personality
- ✅ **Detailed summarization**  Long documents with key point extraction
- ✅ **Translation**  High quality across many language pairs
- ✅ **Instruction following**  Reliably follows complex format instructions
- ✅ **Function calling**  Can use tools and APIs with structured output
- ✅ **RAG**  Excellent context comprehension for retrieval-augmented generation
- ✅ **Multi-turn conversation**  Maintains coherent long conversations

### Limitations

- ❌ **Expert-level reasoning**  Struggles with very complex logic chains
- ❌ **Advanced math**  Can't reliably solve competition-level problems
- ❌ **Very long generation**  Quality can drift in 2000+ word outputs
- ❌ **Obscure knowledge**  Less reliable on niche or recent topics
- ❌ **Complex multi-step planning**  Can lose track in agent scenarios

### Hardware Requirements

```
Hardware for 7-8B Models:
┌──────────────────────────────────────────────────────────┐
│  GPU/Device           │ VRAM   │ Quant │ Speed (tok/s)   │
├───────────────────────┼────────┼───────┼─────────────────┤
│  RTX 3060 12GB        │ 12 GB  │ Q4_K_M│  ~25-35 tok/s   │
│  RTX 3080 10GB        │ 10 GB  │ Q4_K_M│  ~35-50 tok/s   │
│  RTX 4060 8GB         │  8 GB  │ Q4_K_M│  ~30-45 tok/s   │
│  RTX 4070 12GB        │ 12 GB  │ Q5_K_M│  ~40-55 tok/s   │
│  RTX 4090 24GB        │ 24 GB  │ FP16  │  ~80-120 tok/s  │
│  Apple M2 Pro 16GB    │ 16 GB  │ Q4_K_M│  ~20-30 tok/s   │
│  Apple M3 Max 36GB    │ 36 GB  │ Q8_0  │  ~35-50 tok/s   │
│  CPU only (32GB RAM)  │  ---   │ Q4_K_M│  ~5-10 tok/s    │
└──────────────────────────────────────────────────────────┘
Quantized file size: Q4_K_M ≈ 4.5-5 GB
                     Q5_K_M ≈ 5.5-6 GB
                     FP16   ≈ 15-16 GB
```

### Real-World Performance Examples

**Task 1: Code generation**
```python
# Prompt: "Write a Python decorator that retries a function
#          up to N times with exponential backoff"

# Llama 3.1 8B output (typically correct):
import time
import functools

def retry(max_retries=3, base_delay=1.0, backoff_factor=2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (backoff_factor ** attempt)
                    time.sleep(delay)
        return wrapper
    return decorator
```

**Task 2: Analysis and reasoning**
```
Prompt: "Compare microservices vs monolithic architecture.
         Give me a table with trade-offs."

8B Model: Produces a well-organized comparison table with
accurate trade-offs for scalability, deployment, debugging,
team structure, and performance. Comparable to a mid-level
developer's analysis.
```

### Best Use Cases

1. **Local coding assistant**  Pair programmer on your desktop
2. **Content creation**  Blog posts, documentation, technical writing
3. **Business applications**  Report generation, data analysis summaries
4. **Fine-tuned specialists**  Domain-specific models (legal, medical, financial)
5. **Development & testing**  Prototype AI features before using cloud APIs
6. **Privacy-first applications**  Healthcare, finance where data stays local

---

## Tier 4: 14B Models  The Capability Jump

### The Landscape

| Model | Parameters | Context Window | Release | Key Strength |
|-------|-----------|---------------|---------|--------------|
| Qwen 2.5 14B | 14.2B | 128K | 2025 | Best-in-class for size |
| Phi-4 14B | 14B | 16K | 2025 | Microsoft's reasoning champion |
| Qwen3-14B | 14B | 128K | 2025 | Thinking + non-thinking modes |
| DeepSeek-R1-Distill-Qwen-14B | 14B | 64K | 2025 | Reasoning specialist |
| Gemma 3 12B | 12B | 128K | 2025 | Google's efficient mid-size |

### The 14B Inflection Point

Something remarkable happens at 14B parameters: models cross a **capability threshold** where they start handling tasks that were previously the domain of much larger models.

```
Capability vs. Model Size (conceptual):

Capability
    │                                          ╱──── 70B
    │                                    ╱────╱
    │                              ╱────╱
    │                        ╱────╱  ← Diminishing returns
    │                   ╱───╱
    │              ╱───╱   ← 14B: Major jump!
    │         ╱───╱
    │    ╱───╱
    │╱──╱
    └──────────────────────────────────────── Parameters
     1B   3B   7B  14B  32B  70B  405B

The jump from 7B to 14B is disproportionately large
compared to 14B to 32B for most practical tasks.
```

### Why Modern 14B Beats Older 70B

This is one of the most important lessons in modern AI:

| Factor | Old 70B (2023) | Modern 14B (2025-2026) |
|--------|---------------|----------------------|
| Training tokens | ~2T tokens | ~12-18T tokens |
| Data quality | Web crawls, some filtering | Curated, synthetic, high-quality |
| Architecture | Standard transformer | GQA, RoPE, RMSNorm, SwiGLU |
| Tokenizer | 32K vocab | 150K+ vocab (more efficient) |
| Training recipe | Single-phase | Multi-phase curriculum |
| Distillation | None | From frontier models |
| Alignment | Basic RLHF | DPO, GRPO, iterative refinement |

**Concrete example**: Qwen 2.5 14B (2025) outperforms Llama 2 70B (2023) on:
- MMLU (knowledge): 79.9 vs 68.9
- HumanEval (coding): 72.1 vs 29.9
- GSM8K (math): 83.5 vs 56.8
- MT-Bench (conversation): 8.2 vs 6.9

### Hardware Requirements

```
Hardware for 14B Models:
┌──────────────────────────────────────────────────────────┐
│  GPU/Device           │ VRAM   │ Quant │ Speed (tok/s)   │
├───────────────────────┼────────┼───────┼─────────────────┤
│  RTX 3060 12GB        │ 12 GB  │ Q4_K_M│  ~15-22 tok/s   │
│  RTX 4070 12GB        │ 12 GB  │ Q4_K_M│  ~20-30 tok/s   │
│  RTX 4070 Ti S 16GB   │ 16 GB  │ Q5_K_M│  ~25-35 tok/s   │
│  RTX 4090 24GB        │ 24 GB  │ Q8_0  │  ~50-70 tok/s   │
│  RTX 5090 32GB        │ 32 GB  │ FP16  │  ~60-80 tok/s   │
│  Apple M3 Max 36GB    │ 36 GB  │ Q8_0  │  ~20-30 tok/s   │
│  Apple M4 Pro 24GB    │ 24 GB  │ Q4_K_M│  ~18-28 tok/s   │
│  2× RTX 3090 48GB     │ 48 GB  │ FP16  │  ~40-60 tok/s   │
└──────────────────────────────────────────────────────────┘
Quantized file size: Q4_K_M ≈ 8-9 GB
                     Q5_K_M ≈ 10-11 GB
                     Q8_0   ≈ 15-16 GB
                     FP16   ≈ 28-30 GB
```

### What 14B Excels At

- ✅ **Professional coding**  Multi-file understanding, architecture suggestions
- ✅ **Complex reasoning**  Chain-of-thought, multi-step problem solving
- ✅ **Detailed analysis**  Nuanced comparison, pros/cons evaluation
- ✅ **Long-form writing**  Maintains quality over many paragraphs
- ✅ **Math**  Algebra, calculus problems, statistical analysis
- ✅ **Structured output**  JSON, XML, YAML generation with complex schemas
- ✅ **Tool use**  Reliable function calling and API interaction
- ✅ **Multilingual**  Strong performance across 20+ languages

### Best Use Cases

1. **Production coding assistant**  Code review, refactoring, documentation
2. **Enterprise chatbots**  Complex customer support with reasoning
3. **Data analysis**  SQL generation, data interpretation, reporting
4. **Legal/medical summarization**  Domain-specific fine-tuned applications
5. **The "one GPU" powerhouse**  Maximum capability on a single consumer GPU

---

## Tier 5: 32-34B Models  Professional-Grade Local AI

### The Landscape

| Model | Parameters | Context Window | Release | Key Strength |
|-------|-----------|---------------|---------|--------------|
| Qwen 2.5 32B | 32.5B | 128K | 2025 | Outstanding all-rounder |
| Qwen3-32B | 32B | 128K | 2025 | Hybrid thinking model |
| Command R 35B | 35B | 128K | 2024 | Cohere's RAG-optimized |
| DeepSeek-R1-Distill-Qwen-32B | 32B | 64K | 2025 | Best reasoning for size |
| Yi-1.5 34B | 34B | 200K | 2024 | Strong general purpose |
| Gemma 3 27B | 27B | 128K | 2025 | Google's prosumer model |

### The Professional Tier

32B models represent the upper boundary of what fits on a single high-end consumer GPU. They offer a compelling sweet spot: near-70B quality while being practical for local deployment.

```
The 32B Value Proposition:

  ┌─────────────────────────────────────────────┐
  │           Cost-Benefit Analysis              │
  │                                              │
  │  Quality:  ████████████████████░░  ~90% of   │
  │                                    70B       │
  │  Speed:    ████████████████░░░░░░  ~75% of   │
  │                                    14B       │
  │  Cost:     ████████░░░░░░░░░░░░░░  ~35% of   │
  │                                    70B       │
  │  Hardware: Single 24GB GPU (Q4) ✓            │
  │            Single 48GB GPU (Q8) ✓            │
  └─────────────────────────────────────────────┘
```

### Capabilities

At 32B, models approach the quality of frontier API services for most tasks:

- ✅ **Expert-level coding**  Complex algorithms, design patterns, debugging
- ✅ **Advanced reasoning**  Multi-step logic, planning, problem decomposition
- ✅ **Nuanced writing**  Matches human quality for most content types
- ✅ **Complex analysis**  Financial reports, research summaries, legal analysis
- ✅ **Reliable structured output**  Complex JSON schemas with nested objects
- ✅ **Strong agentic behavior**  Tool use, multi-step planning, self-correction
- ✅ **Multilingual mastery**  Professional quality across 25+ languages
- ✅ **Long-context comprehension**  Can process and reason over 100K+ tokens

### The DeepSeek-R1-Distill-Qwen-32B Phenomenon

This model deserves special mention. By distilling DeepSeek-R1's reasoning capabilities into a 32B model, it achieved something remarkable:

```
DeepSeek-R1-Distill-Qwen-32B vs GPT-4o (early 2025):

AIME 2024 (Math):           72.6  vs  79.2   (surprisingly close!)
MATH-500:                   94.3  vs  94.5   (basically tied!)
LiveCodeBench:              57.2  vs  60.3   (competitive)
GPQA-Diamond (Science):     62.1  vs  68.0   (respectable)

A 32B open model competing with GPT-4o on reasoning tasks 
this was unthinkable just a year earlier.
```

### Hardware Requirements

```
Hardware for 32B Models:
┌────────────────────────────────────────────────────────────┐
│  GPU/Device           │ VRAM    │ Quant  │ Speed (tok/s)   │
├───────────────────────┼─────────┼────────┼─────────────────┤
│  RTX 4090 24GB        │ 24 GB   │ Q4_K_M │  ~25-35 tok/s   │
│  RTX 5090 32GB        │ 32 GB   │ Q5_K_M │  ~30-45 tok/s   │
│  Apple M3 Ultra 96GB  │ 96 GB   │ FP16   │  ~15-25 tok/s   │
│  Apple M4 Max 64GB    │ 64 GB   │ Q8_0   │  ~18-28 tok/s   │
│  2× RTX 3090 48GB     │ 48 GB   │ Q8_0   │  ~20-35 tok/s   │
│  A6000 48GB           │ 48 GB   │ FP16   │  ~40-55 tok/s   │
│  CPU only (64GB RAM)  │  ---    │ Q4_K_M │  ~3-6 tok/s     │
└────────────────────────────────────────────────────────────┘
Quantized file size: Q4_K_M ≈ 18-20 GB
                     Q5_K_M ≈ 22-24 GB
                     Q8_0   ≈ 34-36 GB
                     FP16   ≈ 64-66 GB
```

### Best Use Cases

1. **Enterprise AI deployment**  Customer-facing AI with high quality requirements
2. **AI coding agents**  Complex multi-file refactoring and architecture tasks
3. **Research assistance**  Literature review, hypothesis generation
4. **Content creation**  Professional-grade articles, reports, documentation
5. **Competitive alternative to APIs**  When privacy or cost matters

---

## Tier 6: 70B Models  Near-Frontier Performance

### The Landscape

| Model | Parameters | Context Window | Release | Key Strength |
|-------|-----------|---------------|---------|--------------|
| Llama 3.1 70B | 70.6B | 128K | 2024 | Industry standard |
| Qwen 2.5 72B | 72.7B | 128K | 2025 | Top open model |
| DeepSeek-V2.5 | 236B MoE* | 128K | 2024 | MoE efficiency |
| Llama 4 Maverick | 400B MoE* | 1M | 2025 | Meta's MoE with 17B active |
| Mistral Large 2 | 123B | 128K | 2024 | Strong European model |
| Qwen3-235B-A22B | 235B MoE* | 128K | 2025 | Only 22B active params |

*\*MoE (Mixture of Experts) models have more total parameters but only activate a fraction per token, making them faster than their parameter count suggests. Active parameters shown.*

### Near-Frontier Performance

70B dense models (and large MoE models with similar active parameters) reach performance levels that were frontier-only in early 2024:

```
Where 70B Models Sit (2026):

  ┌──────────────────────────────────────────────────┐
  │  Frontier (GPT-4.5, Claude 4, Gemini 2.5 Pro)   │
  │  ════════════════════════════════════════════     │
  │  ▲ ~5-10% gap on average, larger on cutting-edge │
  │  │ tasks like novel reasoning and creativity      │
  │  │                                                │
  │  ├── 70B Open Models (Llama 3.1 70B, Qwen 72B)  │
  │  │   - Competitive on most benchmarks             │
  │  │   - Excellent for production workloads         │
  │  │   - Can be fine-tuned for specific domains     │
  │  │                                                │
  │  ├── The gap shrinks with fine-tuning             │
  │  │   and domain-specific optimization             │
  │  │                                                │
  │  ├── For many tasks, indistinguishable from       │
  │  │   frontier in blind evaluations                │
  └──────────────────────────────────────────────────┘
```

### Capabilities

At 70B, models can handle essentially any task a general user needs:

- ✅ **Expert coding**  Full application development, complex debugging
- ✅ **Advanced reasoning**  PhD-level problem solving on many topics
- ✅ **Research-quality writing**  Academic papers, technical documentation
- ✅ **Complex planning**  Multi-step project planning with contingencies
- ✅ **Sophisticated analysis**  Financial modeling, scientific reasoning
- ✅ **Reliable agentic behavior**  Self-correcting, tool-using agents
- ✅ **Creative excellence**  Near-human quality creative writing
- ✅ **Expert-level multilingual**  Professional quality in 50+ languages

### Hardware Requirements

This is where things get serious:

```
Hardware for 70B Models:
┌───────────────────────────────────────────────────────────────┐
│  Configuration          │ VRAM    │ Quant  │ Speed (tok/s)    │
├─────────────────────────┼─────────┼────────┼──────────────────┤
│  RTX 4090 24GB          │ 24 GB   │ Q3_K_M │  ~12-18 tok/s    │
│  RTX 5090 32GB          │ 32 GB   │ Q4_K_M │  ~15-22 tok/s    │
│  2× RTX 4090 48GB       │ 48 GB   │ Q5_K_M │  ~18-28 tok/s    │
│  Apple M3 Ultra 192GB   │ 192 GB  │ FP16   │  ~12-18 tok/s    │
│  Apple M4 Ultra 128GB   │ 128 GB  │ Q8_0   │  ~15-22 tok/s    │
│  A100 80GB              │ 80 GB   │ FP16   │  ~40-60 tok/s    │
│  H100 80GB              │ 80 GB   │ FP16   │  ~70-100 tok/s   │
│  2× A100 80GB           │ 160 GB  │ FP16   │  ~80-120 tok/s   │
│  CPU only (128GB RAM)   │  ---    │ Q4_K_M │  ~1-3 tok/s      │
└───────────────────────────────────────────────────────────────┘
Quantized file size: Q3_K_M ≈ 30-32 GB
                     Q4_K_M ≈ 40-42 GB
                     Q5_K_M ≈ 48-50 GB
                     Q8_0   ≈ 70-75 GB
                     FP16   ≈ 140-145 GB
```

### Cost Analysis: Local vs. Cloud

Running a 70B model has a meaningful cost equation:

```
Local Deployment Cost (one-time):
  ┌──────────────────────────────────────────┐
  │  Option A: Consumer GPU                   │
  │  2× RTX 4090 = ~$3,200                   │
  │  Power: ~600W = ~$50/month               │
  │  Break-even vs API: ~2-4 months heavy use│
  │                                           │
  │  Option B: Cloud GPU                      │
  │  1× A100 80GB = ~$1.50-2.50/hour         │
  │  H100 80GB = ~$2.50-4.00/hour            │
  │  Good for: Intermittent use              │
  │                                           │
  │  Option C: Apple Silicon                  │
  │  M3 Ultra 192GB = ~$8,000                │
  │  Power: ~60W = ~$5/month                 │
  │  Silent, desktop form factor              │
  └──────────────────────────────────────────┘

API Cost Comparison:
  ┌──────────────────────────────────────────┐
  │  Llama 3.1 70B (via API):                │
  │  ~$0.50-0.90 per million input tokens    │
  │  ~$0.50-0.90 per million output tokens   │
  │                                           │
  │  GPT-4o (for reference):                  │
  │  ~$2.50 per million input tokens          │
  │  ~$10.00 per million output tokens        │
  │                                           │
  │  If you process >100M tokens/month,       │
  │  self-hosting is likely cheaper.          │
  └──────────────────────────────────────────┘
```

### Best Use Cases

1. **Enterprise production**  Customer-facing AI products
2. **Research**  Academic and scientific applications
3. **Cloud AI services**  Backend for AI-powered SaaS
4. **Fine-tuned specialists**  Domain-expert models matching frontier quality
5. **AI agents**  Complex multi-step autonomous systems
6. **Alternative to frontier APIs**  Privacy, cost, or latency advantages

---

## Tier 7: 400B+ Models  Frontier Scale

### The Landscape

| Model | Parameters | Architecture | Release | Key Strength |
|-------|-----------|-------------|---------|--------------|
| Llama 3.1 405B | 405B | Dense | 2024 | Largest open dense model |
| DeepSeek-V3 | 671B (37B active) | MoE | 2024 | Cost-efficient training |
| DeepSeek-V3-0324 | 671B (37B active) | MoE | 2025 | Improved V3 |
| DeepSeek-R1 | 671B (37B active) | MoE | 2025 | Frontier reasoning |
| Qwen3-235B-A22B | 235B (22B active) | MoE | 2025 | Efficient frontier |
| Llama 4 Behemoth | 2T+ (288B active) | MoE | 2025 | Meta's largest |
| Mixtral 8x22B | 176B (39B active) | MoE | 2024 | Mistral's large MoE |

### Frontier-Scale Capabilities

These models represent the cutting edge of open AI:

```
What Frontier-Scale Models Can Do:

┌────────────────────────────────────────────────────────┐
│  CAPABILITIES AT THE FRONTIER                          │
│                                                        │
│  ✅ PhD-level reasoning across disciplines              │
│  ✅ Novel problem solving and creative thinking         │
│  ✅ Near-perfect instruction following                  │
│  ✅ State-of-the-art code generation                    │
│  ✅ Competition-level mathematics                       │
│  ✅ Expert-level multilingual (100+ languages)          │
│  ✅ Complex agentic workflows                           │
│  ✅ Long-context understanding (128K-1M+ tokens)        │
│  ✅ Advanced scientific reasoning                       │
│  ✅ Sophisticated multi-turn dialogue                   │
└────────────────────────────────────────────────────────┘
```

### The MoE Revolution

Most frontier-scale open models use **Mixture of Experts (MoE)** architecture. This is crucial to understand:

```
Dense Model (405B):                MoE Model (671B total, 37B active):
┌─────────────────────┐           ┌───────────────────────┐
│ ALL 405B parameters │           │ Router selects 8 of   │
│ activate for EVERY  │           │ 256 experts per token  │
│ single token        │           │                        │
│                     │           │  Expert 1 [■]          │
│ ████████████████    │           │  Expert 2 [ ]          │
│ ████████████████    │           │  Expert 3 [■]          │
│ ████████████████    │           │  Expert 4 [ ]          │
│ ████████████████    │           │  Expert 5 [■]          │
│                     │           │  ...                   │
│ VRAM: ~810 GB FP16  │           │  Expert 256 [ ]        │
│ Speed: Slow         │           │                        │
└─────────────────────┘           │  ■ = Active this token │
                                  │  [ ] = Inactive        │
                                  │                        │
                                  │  VRAM: ~400 GB* FP16   │
                                  │  Speed: Much faster!   │
                                  │  Quality: Similar!     │
                                  └───────────────────────┘
                                  *Need all params in memory,
                                   but compute is only for
                                   active experts.
```

### Hardware Requirements

Running these models requires serious infrastructure:

```
Hardware for 400B+ Models:
┌────────────────────────────────────────────────────────────────┐
│  Configuration              │ Memory   │ Speed     │ Cost      │
├─────────────────────────────┼──────────┼───────────┼───────────┤
│  DENSE 405B:                │          │           │           │
│  8× A100 80GB (FP16)       │ 640 GB   │ ~20 tk/s  │ ~$100K    │
│  4× H100 80GB (FP8)        │ 320 GB   │ ~50 tk/s  │ ~$120K    │
│  8× H100 80GB (FP16)       │ 640 GB   │ ~80 tk/s  │ ~$240K    │
│                             │          │           │           │
│  MoE 671B (DeepSeek-V3):   │          │           │           │
│  8× A100 80GB (FP16)       │ 640 GB   │ ~30 tk/s  │ ~$100K    │
│  4× H100 80GB (FP8)        │ 320 GB   │ ~60 tk/s  │ ~$120K    │
│  8× H200 141GB (FP8)       │ 1.1 TB   │ ~100 tk/s │ ~$280K    │
│                             │          │           │           │
│  Cloud hourly rates:        │          │           │           │
│  8× A100 instance          │          │           │ ~$15-25/hr│
│  8× H100 instance          │          │           │ ~$25-40/hr│
└────────────────────────────────────────────────────────────────┘
```

### Who Actually Runs These Models?

```
Target Audience for 400B+ Models:
┌────────────────────────────────────────────────────┐
│  1. AI companies building products                  │
│     → Serving millions of users                    │
│     → Need control over the model                  │
│                                                    │
│  2. Cloud providers (AWS, Azure, GCP)              │
│     → Offering model-as-a-service                  │
│                                                    │
│  3. Research labs                                   │
│     → Studying model behavior                      │
│     → Fine-tuning for specific domains             │
│                                                    │
│  4. Governments and military                       │
│     → Sovereign AI, air-gapped deployments         │
│                                                    │
│  5. Large enterprises                              │
│     → Financial institutions, pharma               │
│     → Data sovereignty requirements                │
└────────────────────────────────────────────────────┘

NOT for: Individual developers, small startups, hobbyists
(use APIs to access these models instead)
```

---

## Why Modern Small Models Beat Older Large Models

This is one of the most counterintuitive findings in AI  let's break down exactly why a 2025 14B model can outperform a 2023 70B model.

### 1. Better Training Data

```
The Data Quality Revolution:

2022-2023 Training Data:
┌─────────────────────────────────────┐
│  Common Crawl (raw web scraping)    │  ← Lots of noise
│  Wikipedia                          │  ← Good but limited
│  Books3, Pile                       │  ← Legal concerns
│  Some code (GitHub scrapes)         │  ← Unfiltered
│  Total: ~1-2T tokens               │
│  Quality: Mixed ██░░░░░░░░          │
└─────────────────────────────────────┘

2025-2026 Training Data:
┌─────────────────────────────────────┐
│  Curated web (quality-filtered)     │  ← Rigorous deduplication
│  Synthetic data from frontier LLMs  │  ← High-quality generated
│  Verified textbooks and papers      │  ← Accurate knowledge
│  Curated code (linted, tested)      │  ← Higher quality
│  Multilingual parallel corpora      │  ← Better translation
│  Math & reasoning chains            │  ← Step-by-step solutions
│  Instruction-response pairs         │  ← Fine-grained alignment
│  Total: ~12-18T tokens              │
│  Quality: High ████████████████░    │
└─────────────────────────────────────┘

Key Insight: A smaller model trained on 10× better data
learns more useful patterns than a larger model trained
on noisy data.
```

### 2. Improved Architectures

| Innovation | Old (2022-2023) | New (2025-2026) | Impact |
|-----------|----------------|-----------------|--------|
| Attention | Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) | 2-4× faster inference |
| Position Encoding | Absolute/Learned | RoPE (Rotary Position Embeddings) | Better long-context handling |
| Normalization | LayerNorm | RMSNorm | Faster, more stable training |
| Activation | GELU/ReLU | SwiGLU | Better gradient flow, +2-3% quality |
| Vocabulary | 32K tokens | 100K-150K+ tokens | More efficient encoding |
| KV Cache | Full precision | Sliding window + quantized KV | 50-75% memory savings |

### 3. Better Tokenizers

```
Tokenizer Efficiency Example:

Old tokenizer (32K vocab, 2023):
  "implementation" → ["implement", "ation"]           (2 tokens)
  "深度学习"       → ["深", "度", "学", "习"]          (4 tokens)
  "def fibonacci"  → ["def", " fib", "on", "acci"]   (4 tokens)

New tokenizer (150K vocab, 2025):
  "implementation" → ["implementation"]                (1 token)
  "深度学习"       → ["深度学习"]                       (1 token)
  "def fibonacci"  → ["def", " fibonacci"]             (2 tokens)

Fewer tokens means:
  • Same context window fits MORE text
  • Faster processing (fewer forward passes)
  • Better understanding (words aren't split awkwardly)
  • Especially impactful for non-English languages
```

### 4. Knowledge Distillation

```
How Distillation Works:

┌──────────────┐     Knowledge      ┌──────────────┐
│  Teacher     │    Transfer        │  Student     │
│  (70B-400B)  │ ───────────────►   │  (14B)       │
│              │                    │              │
│  "Soft"      │  Train student     │  Learns the  │
│  predictions │  to match          │  REASONING   │
│  with rich   │  teacher's full    │  patterns    │
│  probability │  output            │  not just    │
│  distributions│  distributions     │  answers     │
└──────────────┘                    └──────────────┘

Why it works:
  • Teacher's soft labels carry more information than hard labels
  • Student learns "dark knowledge"  e.g., "cat" is more similar
    to "dog" than to "airplane"
  • Teacher makes mistakes the student can learn to avoid
  • Most effective method: Chain-of-thought distillation
    (student learns the REASONING PROCESS, not just answers)
```

### 5. Training Compute Efficiency

```
Compute Efficiency Over Time:

Model         │ Year │ Params │ Training FLOPs │ Performance
──────────────┼──────┼────────┼────────────────┼───────────
GPT-3         │ 2020 │  175B  │  3.1×10²³      │  Baseline
Chinchilla    │ 2022 │   70B  │  5.8×10²³      │  +15%
Llama 2 70B   │ 2023 │   70B  │  ~1.0×10²⁴     │  +25%
Qwen 2.5 14B  │ 2025 │   14B  │  ~2.0×10²³     │  +35%!

The 14B model uses LESS compute than GPT-3 but
performs MUCH better. That's the power of
architectural + data improvements.
```

---

## Comprehensive Comparison Tables

### Size vs. Capability vs. Hardware

| Size | Daily Tasks | Coding | Reasoning | Writing | VRAM (Q4) | Speed* | Best For |
|------|------------|--------|-----------|---------|-----------|--------|----------|
| 1B | ★★☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | 1 GB | 50+ tk/s | Edge, classification |
| 3B | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | 2 GB | 40+ tk/s | Laptop assistant |
| 7-8B | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | 5 GB | 30+ tk/s | Local AI workhorse |
| 14B | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | 9 GB | 20+ tk/s | Pro single-GPU |
| 32B | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ | 20 GB | 15+ tk/s | Enterprise local |
| 70B | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | 40 GB | 10+ tk/s | Production server |
| 405B | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | 200 GB | 3+ tk/s | Cloud/datacenter |

*Speed on representative consumer/professional hardware with noted quantization.*

### Benchmark Comparison Across Sizes (2025-2026 Models)

| Benchmark | What It Tests | ~1B | ~3B | ~7B | ~14B | ~32B | ~70B | ~400B+ |
|-----------|--------------|-----|-----|-----|------|------|------|--------|
| MMLU | Knowledge | 35-42 | 50-58 | 62-70 | 72-80 | 78-83 | 82-86 | 85-88 |
| HumanEval | Python coding | 15-25 | 35-48 | 55-68 | 68-78 | 75-82 | 80-87 | 85-92 |
| GSM8K | Math (grade school) | 25-35 | 45-58 | 65-78 | 78-88 | 85-92 | 90-95 | 94-97 |
| MT-Bench | Conversation | 4.5-5.5 | 6.0-7.0 | 7.2-8.0 | 7.8-8.5 | 8.2-8.8 | 8.5-9.1 | 8.8-9.3 |
| MATH | Competition math | 8-15 | 18-30 | 30-48 | 45-62 | 58-72 | 65-78 | 75-88 |
| ARC-C | Science reasoning | 35-42 | 48-58 | 58-68 | 65-75 | 72-80 | 78-88 | 88-94 |
| IFEval | Instruction following | 30-40 | 45-55 | 55-70 | 68-78 | 75-85 | 82-90 | 88-93 |
| MBPP | More Python coding | 20-30 | 40-52 | 58-70 | 70-80 | 76-84 | 82-88 | 87-93 |

*Ranges represent variation across models within each tier, using best-available models in 2025-2026.*

### Cost to Run: Local vs. API

| Scenario | Local GPU | Local Cost/Month | API Provider | API Cost/Month |
|----------|----------|-----------------|-------------|---------------|
| Light use (1M tokens/day) | RTX 4060 + 7B | ~$15 (electricity) | GPT-4o-mini | ~$18 |
| Medium use (10M tokens/day) | RTX 4090 + 14B | ~$45 (electricity) | GPT-4o-mini | ~$180 |
| Heavy use (100M tokens/day) | 2× RTX 4090 + 70B | ~$90 (electricity) | GPT-4o | ~$37,500 |
| Enterprise (1B tokens/day) | 8× H100 + 70B | ~$3,000 (cloud) | GPT-4o | ~$375,000 |

*Local costs exclude hardware amortization. Including a 2-year amortization of hardware adds roughly 30-100% to local costs but still favors local for heavy workloads.*

### Decision Matrix: Which Size Should You Use?

```
START HERE
    │
    ▼
┌─────────────────────────┐     Yes    ┌──────────────────┐
│ Running on mobile/edge? ├───────────►│ 1B-3B models     │
└────────────┬────────────┘            │ Phi-4-mini,      │
             │ No                      │ Gemma 3 1B       │
             ▼                         └──────────────────┘
┌─────────────────────────┐     Yes    ┌──────────────────┐
│ Only have 8GB VRAM?     ├───────────►│ 3B-7B models     │
└────────────┬────────────┘            │ (Q4 quantized)   │
             │ No                      └──────────────────┘
             ▼
┌─────────────────────────┐     Yes    ┌──────────────────┐
│ Single consumer GPU?    ├───────────►│ 7B-14B models    │
│ (12-24GB VRAM)          │            │ Qwen 2.5 14B,    │
└────────────┬────────────┘            │ Llama 3.1 8B     │
             │ No (more VRAM)          └──────────────────┘
             ▼
┌─────────────────────────┐     Yes    ┌──────────────────┐
│ Need top local quality? ├───────────►│ 32B-70B models   │
│ (24-80GB VRAM)          │            │ Qwen3-32B,       │
└────────────┬────────────┘            │ Llama 3.1 70B    │
             │ No                      └──────────────────┘
             ▼
┌─────────────────────────┐     Yes    ┌──────────────────┐
│ Need absolute best?     ├───────────►│ 400B+ or API     │
│ Budget is not primary   │            │ DeepSeek-R1,     │
│ concern?                │            │ GPT-4.5, Claude  │
└─────────────────────────┘            └──────────────────┘
```

---

## How to Choose: A Practical Guide

### Step 1: Define Your Task Requirements

```
Task Complexity Levels:

Level 1 (Simple):     Classification, extraction, formatting
                      → 1B-3B models are sufficient

Level 2 (Moderate):   Summarization, Q&A, simple code
                      → 7B models work well

Level 3 (Complex):    Analysis, reasoning, full code generation
                      → 14B-32B models recommended

Level 4 (Expert):     Research, multi-step agents, creative work
                      → 70B+ models or frontier APIs
```

### Step 2: Consider Your Constraints

| Constraint | Recommendation |
|-----------|---------------|
| No GPU | 1B-3B (CPU inference), or use APIs |
| Privacy required | Largest model your hardware supports |
| Low latency needed | Smaller model, more quantization |
| Highest quality needed | Largest model you can afford |
| Budget-limited | Find the sweet spot (usually 7B-14B) |
| Batch processing | Can use larger models with slower speed |
| Real-time interaction | Need 15+ tokens/second minimum |

### Step 3: The Quantization Decision

```
Quantization Quality Impact:

                 ┌─ FP16 (baseline quality, 100%)
                 │
  Quality        ├─ Q8_0 (~99.5% of FP16 quality)
     │           │
     │           ├─ Q6_K (~99% of FP16 quality)
     │           │
     │           ├─ Q5_K_M (~98% of FP16 quality)  ← Sweet spot
     │           │
     │           ├─ Q4_K_M (~96% of FP16 quality)  ← Most popular
     │           │
     │           ├─ Q3_K_M (~93% of FP16 quality)
     │           │
     │           └─ Q2_K (~85% of FP16 quality)     ← Noticeable degradation
     │
     └────────────────────────────────────────► Memory Savings

Recommendation:
  • Q4_K_M: Best balance of quality and size for most users
  • Q5_K_M: When you want a bit more quality and have the VRAM
  • Q8_0: When quality matters most and you have ample VRAM
  • Q3_K_M: When you're squeezing a bigger model into limited VRAM
    (better to run 14B at Q3 than 7B at Q8 for most tasks)
```

### Pro Tip: Model Size vs. Quantization Trade-off

**Key insight**: It's often better to run a larger model at lower quantization than a smaller model at higher quantization:

```
14B Q4_K_M (~9 GB)  >  7B Q8_0 (~8 GB)  on most benchmarks
32B Q4_K_M (~20 GB) >  14B FP16 (~28 GB) on most benchmarks
70B Q3_K_M (~32 GB) ≈  32B Q8_0 (~34 GB) depends on task

Rule: Prioritize model size over quantization precision,
      down to about Q3_K_M. Below that, quality drops
      enough that a smaller model at Q4+ may be better.
```

---

## The Future: Where Model Sizes Are Heading

### Trends for 2026-2027

1. **Small models keep getting better**: Expect 7B models to match today's 32B
2. **MoE becomes dominant**: More parameters, fewer active, better efficiency
3. **Specialized small models**: Fine-tuned 3B models matching general 14B for specific tasks
4. **1-bit and 2-bit quantization**: New training methods may make ultra-low precision practical
5. **Hardware catching up**: RTX 5090 (32GB) makes 32B models accessible, consumer 48GB GPUs coming
6. **On-device AI explosion**: 3B models in every phone, laptop, and smart device

```
The Model Size Landscape is Shifting:

2023:  "You need 70B+ for real work"
2024:  "Actually, 14B is surprisingly good"
2025:  "7B with the right training is production-ready"
2026:  "3B is the new 7B, 14B is the new 70B"
2027:  "1B handles most routine tasks, 7B is professional-grade" (predicted)
```

---

## Quiz: Understanding Model Sizes

### Question 1
**A company wants to deploy an AI chatbot on their website that handles complex customer queries about technical products. They need fast responses (>20 tok/s) and have a single RTX 4090 (24GB VRAM). Which model size tier and quantization should they choose?**

<details>
<summary>Show Answer</summary>

**14B at Q4_K_M (~9 GB)** is the ideal choice. It provides:
- Complex reasoning and detailed technical answers (14B capability)
- Fast inference at 25-35 tok/s on RTX 4090
- Only uses ~9 GB of 24 GB VRAM, leaving room for KV cache and batching
- Could also consider 32B at Q4_K_M (~20 GB) if they want maximum quality and can accept slightly slower speeds (~25-35 tok/s), but would have less room for batching

A 7B model would also work but might struggle with complex technical queries. A 70B model wouldn't fit on a single RTX 4090 at reasonable quantization with good speed.
</details>

### Question 2
**Why can Qwen 2.5 14B (2025) outperform Llama 2 70B (2023) despite having 5× fewer parameters?**

<details>
<summary>Show Answer</summary>

Five main factors:
1. **Better training data**: Qwen 2.5 was trained on ~18T tokens of curated, high-quality data vs. ~2T tokens for Llama 2
2. **Improved architecture**: GQA, RoPE, SwiGLU, RMSNorm  each adding incremental quality improvements
3. **Better tokenizer**: 150K vocabulary vs 32K, meaning more efficient text encoding
4. **Knowledge distillation**: Smaller models trained to mimic the reasoning patterns of larger teacher models
5. **Training compute efficiency**: More FLOPs per parameter through longer training, better optimization, and curriculum learning

The key insight is that model capability depends on parameters × data quality × training compute × architecture, not just parameter count alone.
</details>

### Question 3
**You have 48 GB of total VRAM (2× RTX 3090). Which would give better results for a coding task: a 32B model at Q5_K_M or a 70B model at Q3_K_M?**

<details>
<summary>Show Answer</summary>

For **coding specifically**, the **70B at Q3_K_M (~32 GB)** would likely perform better, because:
- Coding benchmarks show significant jumps from 32B to 70B
- Q3_K_M retains ~93% of full quality, which is still quite good
- The extra reasoning capacity of 70B helps with complex coding patterns

However, it's close! The 32B at Q5_K_M (~24 GB) would be:
- Faster (more headroom for KV cache)
- Better for multi-turn conversations (more context in memory)
- Nearly as good on simpler coding tasks

**Practical advice**: Try both and benchmark on your specific use case. The answer depends on the complexity of the coding tasks.
</details>

### Question 4
**What is MoE (Mixture of Experts) and why does it matter for large models?**

<details>
<summary>Show Answer</summary>

MoE is an architecture where:
- The model has many "expert" sub-networks (e.g., 256 experts)
- For each input token, a router selects only a few experts (e.g., 8 of 256)
- Only the selected experts process that token
- Total parameters are large (e.g., 671B) but active parameters per token are small (e.g., 37B)

**Why it matters**:
- **Speed**: Inference is much faster since only ~5-15% of parameters are active per token
- **Quality**: Total parameter count gives the model capacity to store more knowledge
- **Efficiency**: You get 70B-like speed with much higher total knowledge capacity
- **Cost**: Training is more efficient per quality point

**Trade-off**: You still need enough memory to load ALL parameters, even though only a fraction are active per token. So a 671B MoE model still needs ~400GB+ of memory, even though it computes like a ~37B model.
</details>

### Question 5
**A startup wants to build a privacy-focused AI assistant that runs entirely on users' laptops (no internet needed). Most users have 16GB RAM and no dedicated GPU. What's the maximum useful model size they should target?**

<details>
<summary>Show Answer</summary>

**3B at Q4_K_M (~2 GB file size)** is the safest recommendation, with the possibility of stretching to **7B at Q4_K_M (~5 GB)** for users with better hardware.

Reasoning:
- With 16GB RAM and CPU-only inference, they need room for the OS, the application, and the model
- Realistically, 6-8 GB is available for the model after OS and application overhead
- A 3B Q4_K_M model uses ~2 GB and runs at ~10-18 tok/s on CPU  usable
- A 7B Q4_K_M model uses ~5 GB and runs at ~5-10 tok/s on CPU  borderline acceptable
- Going larger than 7B on CPU with 16GB RAM would be too slow (3 tok/s) or not fit

The startup could also consider:
- Supporting Apple Silicon GPU acceleration (unified memory makes larger models viable)
- Offering different model sizes based on detected hardware
- Using a 3B model with RAG to augment capabilities

Best models for this use case: Llama 3.2 3B, Gemma 3 4B, or Phi-3-mini 3.8B
</details>

---

## Summary

```
Key Takeaways:

1. Model size isn't everything  training data, architecture,
   and compute matter just as much

2. The sweet spot shifts over time: today's 7-14B range
   offers remarkable capability per dollar

3. Quantization lets you run larger models on smaller GPUs
   with minimal quality loss (Q4_K_M is usually fine)

4. MoE architectures are changing the game at the frontier 
   more knowledge, same speed

5. Always match model size to your actual task requirements 
   bigger isn't always better if a smaller model handles
   your use case

6. The gap between open and closed models continues to
   shrink, especially with fine-tuning
```

