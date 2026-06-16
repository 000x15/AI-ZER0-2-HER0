# Part 7: Famous AI Models of 2026  A Comprehensive Guide

> *"Knowing which model to use is as important as knowing how models work."*

---

## Introduction: The Model Landscape

Imagine walking into a car dealership  but instead of 20 cars, there are **hundreds** of AI models, each with different engines, fuel types, sizes, and price tags. Some are luxury sedans (GPT-4.1, Claude 4 Opus), some are sports cars (o3, DeepSeek-R1), some are reliable daily drivers (Llama 4, Gemma 3), and some are compact city cars that run on surprisingly little fuel (Phi-4-mini, Gemma 3 1B).

This chapter is your **buyer's guide** to the AI models that matter in 2026. For each model family, we'll cover who made it, what's under the hood, how big it is, what it's good at, what it struggles with, and how much it costs. By the end, you'll know exactly which model to reach for in any situation.

### How We Got Here: A Brief Timeline

```
2017 ─── Transformer invented (Google)
2018 ─── GPT-1 (117M params), BERT (340M params)
2019 ─── GPT-2 (1.5B params)
2020 ─── GPT-3 (175B params)  "scaling works!"
2022 ─── ChatGPT launches (GPT-3.5)  AI goes mainstream
2023 ─── GPT-4, Claude 2, Llama 2, Gemini 1.0
2024 ─── GPT-4o, Claude 3.5, Gemini 1.5, Llama 3, open-source explosion
2025 ─── o3, Claude 4, Gemini 2.5, Llama 4, DeepSeek-R1, Qwen 3
2026 ─── Mature ecosystem: specialized models, efficient inference, MoE dominance
```

### Key Concepts Before We Begin

| Term | Meaning |
|------|---------|
| **Parameters** | The learned weights in a model  more ≈ more capacity (but not always better) |
| **Context Window** | How much text the model can "see" at once (measured in tokens) |
| **MoE** | Mixture of Experts  only a subset of parameters activate per token |
| **Dense Model** | All parameters activate for every token |
| **Open Weight** | Model weights are publicly downloadable (may have usage restrictions) |
| **Open Source** | Weights + training code + data are all available |
| **Multimodal** | Can process multiple input types (text, images, audio, video) |
| **Reasoning Model** | Uses extended "thinking" before answering, excels at logic/math |

---

## 1. GPT Family (OpenAI)

### Overview

OpenAI is the company that started the modern AI revolution. Founded in 2015 as a non-profit (later restructured), OpenAI created the GPT (Generative Pre-trained Transformer) series that proved the scaling hypothesis  that making models bigger and training them on more data leads to emergent capabilities.

**Creator:** OpenAI (San Francisco, USA)
**Architecture:** Decoder-only Transformer (dense for base GPT models, details undisclosed for newer versions)
**Source:** Closed source / Proprietary (weights not publicly available)
**Access:** API, ChatGPT (web/mobile), Microsoft Azure OpenAI Service

### Model Lineup (as of mid-2026)

#### GPT-4o (Released May 2024)

GPT-4o ("o" for "omni") was OpenAI's breakthrough multimodal model  the first to natively handle text, images, and audio within a single unified architecture rather than stitching separate models together.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed (estimated ~200B) |
| **Context Window** | 128K tokens |
| **Modalities** | Text, image, audio input; text, audio output |
| **Training Cutoff** | October 2023 |
| **Key Strength** | Fast, multimodal, good all-rounder |
| **Pricing (API)** | $2.50 / 1M input tokens, $10.00 / 1M output tokens |

**Notable Versions:**
- `gpt-4o`  The flagship version
- `gpt-4o-mini`  Smaller, cheaper variant ($0.15 / 1M input, $0.60 / 1M output). Replaced GPT-3.5 Turbo as the budget option. Surprisingly capable for its cost.

#### GPT-4.1 (Released April 2025)

GPT-4.1 was positioned as a refinement of the GPT-4 lineage, focused specifically on coding, instruction following, and long-context performance. It was OpenAI's response to criticism that GPT-4o sometimes ignored complex instructions.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed |
| **Context Window** | 1,047,576 tokens (1M tokens) |
| **Modalities** | Text, image input; text output |
| **Key Strength** | Excellent coding, superior instruction following, massive context |
| **Pricing (API)** | $2.00 / 1M input tokens, $8.00 / 1M output tokens |

**Notable Versions:**
- `gpt-4.1`  Full model
- `gpt-4.1-mini`  Mid-range ($0.40 / 1M input, $1.60 / 1M output)
- `gpt-4.1-nano`  Smallest and cheapest ($0.10 / 1M input, $0.40 / 1M output)

#### o1 (Released December 2024)

The o1 model introduced OpenAI's "reasoning" paradigm  models that spend time "thinking" before answering. Instead of immediately generating a response, o1 produces an internal chain-of-thought, exploring multiple approaches before committing to an answer.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed |
| **Context Window** | 200K tokens |
| **Output Limit** | 100K tokens |
| **Key Strength** | Complex reasoning, math, science, coding competitions |
| **Weakness** | Slower, more expensive, can overthink simple questions |
| **Pricing (API)** | $15.00 / 1M input tokens, $60.00 / 1M output tokens |

**Notable Versions:**
- `o1`  Full reasoning model
- `o1-mini`  Faster, cheaper reasoning ($3.00 / 1M input, $12.00 / 1M output)
- `o1-pro`  Enhanced version available through ChatGPT Pro ($200/month subscription)

#### o3 (Released April 2025)

o3 represents the next generation of OpenAI's reasoning models, pushing performance on benchmarks like ARC-AGI (a test of novel reasoning), AIME (math competition), and GPQA (graduate-level science questions) to new heights.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed |
| **Context Window** | 200K tokens |
| **Output Limit** | 100K tokens |
| **Key Strength** | State-of-the-art reasoning, exceptional at math and science |
| **Weakness** | Very expensive, high latency for complex problems |
| **Pricing (API)** | $10.00 / 1M input tokens, $40.00 / 1M output tokens |

**Notable Versions:**
- `o3`  Full reasoning model
- `o3-mini`  Available in low/medium/high "reasoning effort" ($1.10 / 1M input, $4.40 / 1M output)

#### o4-mini (Released April 2025)

OpenAI's latest compact reasoning model, offering improved tool use and multimodal reasoning at an efficient price point.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed |
| **Context Window** | 200K tokens |
| **Output Limit** | 100K tokens |
| **Key Strength** | Efficient reasoning with tool use, good at agentic tasks |
| **Pricing (API)** | $1.10 / 1M input tokens, $4.40 / 1M output tokens |

### GPT Family: Strengths & Weaknesses

**Strengths:**
- Massive ecosystem and tooling support
- Best-in-class at following complex, nuanced instructions
- Strong coding abilities (especially GPT-4.1)
- Pioneered the reasoning model paradigm (o-series)
- Excellent multimodal capabilities (GPT-4o)
- Widest third-party integration support

**Weaknesses:**
- Fully closed source  no transparency into architecture or training
- Can be expensive at scale (especially o1/o3)
- Occasional "laziness" in long outputs (tends to summarize rather than complete)
- Privacy concerns (data may be used for training unless opted out via API)
- Rate limits can be restrictive for high-volume users

---

## 2. Claude Family (Anthropic)

### Overview

Anthropic was founded in 2021 by former OpenAI researchers (including Dario and Daniela Amodei) with a focus on AI safety. Their Claude models are known for being helpful, harmless, and honest  following a training methodology called Constitutional AI (CAI), where the model is trained against a set of principles rather than purely from human preferences.

**Creator:** Anthropic (San Francisco, USA)
**Architecture:** Decoder-only Transformer (dense, details partially disclosed)
**Source:** Closed source / Proprietary
**Access:** API, claude.ai (web/mobile), Amazon Bedrock, Google Cloud Vertex AI

### Model Lineup (as of mid-2026)

#### Claude 3.5 Sonnet (Released June 2024, Updated October 2024)

Claude 3.5 Sonnet was a landmark release  it matched or exceeded GPT-4o on most benchmarks while being faster and cheaper. The October 2024 update ("new Sonnet") further improved coding and instruction following, and introduced "computer use" capabilities.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed (estimated ~175B) |
| **Context Window** | 200K tokens |
| **Modalities** | Text, image input; text output |
| **Key Strength** | Exceptional coding, nuanced writing, strong safety |
| **Pricing (API)** | $3.00 / 1M input tokens, $15.00 / 1M output tokens |

#### Claude 3.5 Haiku (Released November 2024)

The fast, affordable member of the Claude 3.5 family. Haiku models are designed for high-throughput, low-latency applications where you need a capable model that won't break the bank.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed (estimated ~20-30B) |
| **Context Window** | 200K tokens |
| **Modalities** | Text, image input; text output |
| **Key Strength** | Speed, cost-efficiency, surprisingly capable for its size |
| **Pricing (API)** | $0.80 / 1M input tokens, $4.00 / 1M output tokens |

#### Claude 4 Sonnet (Released May 2025)

Claude 4 Sonnet represented a significant upgrade to the Sonnet tier  Anthropic's "sweet spot" model that balances capability and cost. It features improved coding, stronger reasoning, and better agentic capabilities for multi-step tool use.

| Attribute           | Details                                                          |
| ------------------- | ---------------------------------------------------------------- |
| **Parameter Count** | Undisclosed                                                      |
| **Context Window**  | 200K tokens                                                      |
| **Output Limit**    | Up to 64K tokens (with extended thinking)                        |
| **Modalities**      | Text, image input; text output                                   |
| **Key Strength**    | Top-tier coding, extended thinking mode, strong at agentic tasks |
| **Pricing (API)**   | $3.00 / 1M input tokens, $15.00 / 1M output tokens               |

**Key Features:**
- **Extended Thinking:** Like OpenAI's o-series, Claude 4 Sonnet can engage in step-by-step reasoning before answering, with a visible thinking trace
- **Tool Use:** Native support for function calling, computer use, and multi-step tool chains
- **Memory & Artifacts:** Integration with persistent memory and structured output generation

#### Claude 4 Opus (Released May 2025)

Anthropic's most powerful model  positioned as the "frontier" option for the hardest problems. Opus is the thinking person's model: slower and more expensive, but capable of deep analysis, complex reasoning, and nuanced understanding that lighter models miss.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed (estimated much larger than Sonnet) |
| **Context Window** | 200K tokens |
| **Output Limit** | Up to 64K tokens (with extended thinking) |
| **Modalities** | Text, image input; text output |
| **Key Strength** | Deepest reasoning, most nuanced understanding, best at ambiguous problems |
| **Pricing (API)** | $15.00 / 1M input tokens, $75.00 / 1M output tokens |

### Claude Family: Strengths & Weaknesses

**Strengths:**
- Exceptional at coding tasks (widely regarded as the best code assistant in mid-2025)
- Very strong at following nuanced instructions with long system prompts
- Best-in-class long-context understanding (actually uses the full 200K window effectively)
- Extended thinking enables transparent reasoning chains
- Constitutional AI approach leads to more predictable safety behavior
- Excellent at structured output and tool use
- Strong writing quality with natural, non-generic tone

**Weaknesses:**
- Closed source
- Can be overly cautious/refusing on edge cases (though improved in Claude 4)
- No native audio input/output (text and image only as of mid-2026)
- Smaller ecosystem than OpenAI (fewer third-party integrations)
- Opus tier is very expensive for high-volume use
- Limited fine-tuning options compared to OpenAI

---

## 3. Gemini Family (Google DeepMind)

### Overview

Google's Gemini models represent the merger of Google Brain and DeepMind's research capabilities. Gemini models are natively multimodal  trained from the ground up on text, images, audio, and video simultaneously (not retrofitted like some competitors). They also benefit from Google's massive infrastructure and integration with Google products.

**Creator:** Google DeepMind (Mountain View, USA / London, UK)
**Architecture:** Decoder-only Transformer (Mixture of Experts for larger variants)
**Source:** Closed source / Proprietary
**Access:** API (Google AI Studio, Vertex AI), Gemini app (web/mobile), integrated into Google products

### Model Lineup (as of mid-2026)

#### Gemini 2.0 Flash (Released February 2025)

Flash models are Google's speed-optimized offerings. Gemini 2.0 Flash pushed the boundaries of what a "fast" model could do  approaching Pro-level quality at a fraction of the cost and latency.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed |
| **Context Window** | 1,048,576 tokens (1M tokens) |
| **Modalities** | Text, image, audio, video input; text, image output |
| **Key Strength** | Incredible speed, massive context, natively multimodal |
| **Pricing (API)** | $0.10 / 1M input tokens, $0.40 / 1M output tokens (under 128K) |

**Key Feature:** Gemini 2.0 Flash was among the first models to offer native **image generation** alongside text generation  you could ask it to create and edit images as part of a conversation.

#### Gemini 2.5 Pro (Released March 2025)

Google's flagship thinking model. Gemini 2.5 Pro combines a massive 1M token context window with built-in reasoning ("thinking") capabilities. It quickly became a top contender on benchmarks, particularly for coding and math.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed (estimated MoE architecture) |
| **Context Window** | 1,048,576 tokens (1M tokens) |
| **Modalities** | Text, image, audio, video input; text output |
| **Thinking** | Built-in reasoning mode with adjustable thinking budget |
| **Key Strength** | Top-tier coding, excellent math/reasoning, massive context |
| **Pricing (API)** | $1.25 / 1M input tokens (≤200K), $2.50 (>200K); $10.00 / 1M output tokens |

**Benchmark Highlights:**
- #1 on multiple coding benchmarks (SWE-bench, LiveCodeBench) in early-mid 2025
- Competitive with o3 on math and reasoning benchmarks
- Best-in-class long-context performance (can actually use 1M tokens effectively)

#### Gemini 2.5 Flash (Released May 2025)

The latest Flash model combines the speed and cost efficiency of the Flash tier with thinking capabilities  a "reasoning model you can afford."

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed |
| **Context Window** | 1,048,576 tokens (1M tokens) |
| **Modalities** | Text, image, audio, video input; text output |
| **Thinking** | Built-in reasoning mode with adjustable budget |
| **Key Strength** | Best cost-to-performance ratio with thinking capabilities |
| **Pricing (API)** | $0.15 / 1M input tokens (≤200K), $1.00 / 1M output (non-thinking) |

### Gemini Family: Strengths & Weaknesses

**Strengths:**
- Best native multimodality (text, image, audio, video all in one model)
- Largest context window in the industry (1M tokens, genuinely functional)
- Extremely competitive pricing, especially Flash models
- Deep Google ecosystem integration (Search, Workspace, Android)
- Strong thinking/reasoning capabilities (2.5 Pro/Flash)
- Native image generation within conversation
- Excellent video understanding

**Weaknesses:**
- Historically inconsistent quality (Gemini 1.0 launch was rocky)
- Creative writing can feel "Google-ish"  safe, informational, sometimes bland
- System prompt adherence not always as strong as Claude/GPT
- Safety filters can be aggressive (especially for creative content)
- API can have high variability in response quality
- Closed source  limited transparency

---

## 4. Llama Family (Meta)

### Overview

Meta's Llama (Large Language Model Meta AI) family has been the **most impactful open-weight model series** in AI history. By releasing powerful models with relatively permissive licenses, Meta single-handedly created the open-source AI ecosystem that dozens of other projects build upon. Llama models serve as the foundation for countless fine-tuned variants, research experiments, and production systems.

**Creator:** Meta AI (Menlo Park, USA)
**Architecture:** Decoder-only Transformer (Dense for Llama 3.x, Mixture of Experts for Llama 4)
**Source:** Open weights (Llama Community License  free for most uses, restrictions for very large companies)
**Access:** Direct download, Hugging Face, Ollama, all major cloud providers

### Model Lineup (as of mid-2026)

#### Llama 3.1 (Released July 2024)

The model that proved open-weight models could compete with the best closed models. Llama 3.1 405B was the first open-weight model to rival GPT-4 on many benchmarks.

| Variant | Parameters | Context Window | Key Notes |
|---------|-----------|----------------|-----------|
| Llama 3.1 8B | 8 billion | 128K tokens | Great for local inference, fine-tuning |
| Llama 3.1 70B | 70 billion | 128K tokens | Production-grade, strong all-rounder |
| Llama 3.1 405B | 405 billion | 128K tokens | Frontier-class open model |

**Architecture Details:**
- Dense Transformer (all parameters active for every token)
- Grouped-Query Attention (GQA) for inference efficiency
- RoPE (Rotary Position Embedding) for position encoding
- SwiGLU activation function
- Trained on ~15 trillion tokens

#### Llama 3.3 (Released December 2024)

Llama 3.3 focused on the 70B size  essentially delivering 405B-level quality in a 70B package through improved training techniques. It made frontier-quality AI accessible on a single high-end GPU.

| Variant | Parameters | Context Window | Key Notes |
|---------|-----------|----------------|-----------|
| Llama 3.3 70B | 70 billion | 128K tokens | Near-405B quality, much more efficient |

#### Llama 4 Scout (Released April 2025)

Llama 4 marked Meta's transition to Mixture of Experts (MoE) architecture. Scout is the "efficient" variant  a large model that only activates a portion of its parameters for each token.

| Attribute | Details |
|-----------|---------|
| **Total Parameters** | 109 billion (17B active per token) |
| **Architecture** | MoE with 16 experts |
| **Context Window** | 10,000,000 tokens (10M!) |
| **Modalities** | Text, image input; text output |
| **Key Strength** | Enormous context window, efficient inference |

**The 10M Context Window:** Scout's 10 million token context window is remarkable  that's enough to process entire codebases, book series, or massive document collections in a single prompt. However, real-world performance degrades for very long contexts, and few inference engines can actually handle 10M tokens in practice.

#### Llama 4 Maverick (Released April 2025)

The larger, more capable Llama 4 variant  designed for quality-first use cases.

| Attribute | Details |
|-----------|---------|
| **Total Parameters** | 400 billion (17B active per token) |
| **Architecture** | MoE with 128 experts |
| **Context Window** | 1,048,576 tokens (1M tokens) |
| **Modalities** | Text, image input; text output |
| **Key Strength** | High quality with efficient per-token compute |

### Llama Family: Strengths & Weaknesses

**Strengths:**
- Open weights  download and run anywhere, modify freely
- Massive ecosystem of fine-tunes, tools, and infrastructure
- Available on every cloud platform, inference engine, and local setup
- Llama 4's MoE architecture is very efficient (17B active params)
- Excellent foundation for fine-tuning and customization
- Multiple sizes for different hardware budgets
- Permissive license for most commercial uses

**Weaknesses:**
- Llama 4 initial releases had mixed reception (quality concerns for the MoE variants)
- Base models need fine-tuning for many specific use cases
- 405B dense model requires massive hardware (800GB+ in FP16)
- Long-context performance (10M) is theoretical  practical limit much lower
- Not the best at creative writing compared to Claude/GPT
- Safety training can be inconsistent

---

## 5. Qwen Family (Alibaba Cloud)

### Overview

Qwen (通义千问, "Tongyi Qianwen") is Alibaba Cloud's model family and has emerged as one of the most competitive open-weight model series globally. The Qwen team has been remarkably prolific, releasing high-quality models across many sizes and modalities. Qwen models are particularly strong for multilingual tasks (especially Chinese) and have shown impressive benchmark performance.

**Creator:** Alibaba Cloud (Hangzhou, China)
**Architecture:** Decoder-only Transformer (dense and MoE variants)
**Source:** Open weights (Apache 2.0 license for most models  very permissive)
**Access:** Hugging Face, ModelScope, Ollama, major cloud platforms

### Model Lineup (as of mid-2026)

#### Qwen 2.5 (Released September 2024 and onward)

Qwen 2.5 was a comprehensive release across many sizes, establishing Qwen as a top-tier open model family.

| Variant | Parameters | Context Window | Key Notes |
|---------|-----------|----------------|-----------|
| Qwen 2.5 0.5B | 0.5 billion | 32K tokens | Tiny, edge-device capable |
| Qwen 2.5 1.5B | 1.5 billion | 32K tokens | Mobile and embedded |
| Qwen 2.5 3B | 3 billion | 32K tokens | Small but capable |
| Qwen 2.5 7B | 7 billion | 128K tokens | Best-in-class at 7B |
| Qwen 2.5 14B | 14 billion | 128K tokens | Strong mid-range |
| Qwen 2.5 32B | 32 billion | 128K tokens | Excellent balance |
| Qwen 2.5 72B | 72 billion | 128K tokens | Frontier-class open model |
| Qwen 2.5 Coder 32B | 32 billion | 128K tokens | Specialized for code |
| Qwen 2.5 Math 72B | 72 billion | 4K tokens | Specialized for math |

#### Qwen 3 (Released April 2025)

Qwen 3 introduced both dense and MoE variants with a novel "hybrid thinking" approach  models that can switch between fast responses and extended reasoning within a single model.

| Variant | Type | Total Params | Active Params | Context | Key Notes |
|---------|------|-------------|---------------|---------|-----------|
| Qwen 3 0.6B | Dense | 0.6B | 0.6B | 32K | Edge/mobile |
| Qwen 3 1.7B | Dense | 1.7B | 1.7B | 32K | Small efficient |
| Qwen 3 4B | Dense | 4B | 4B | 32K | Good small model |
| Qwen 3 8B | Dense | 8B | 8B | 128K | Strong general purpose |
| Qwen 3 14B | Dense | 14B | 14B | 128K | Mid-range powerhouse |
| Qwen 3 32B | Dense | 32B | 32B | 128K | High quality |
| Qwen 3 30B-A3B | MoE | 30B | 3B | 128K | Very efficient |
| Qwen 3 235B-A22B | MoE | 235B | 22B | 128K | Frontier MoE |

**Key Innovation  Hybrid Thinking:**
Qwen 3 models can operate in two modes:
- **Thinking Mode:** Extended chain-of-thought reasoning (like o3/R1)
- **Non-Thinking Mode:** Fast, direct responses (like standard models)

Users can switch between modes within a single model, or let the model decide when to "think" based on problem complexity. This is highly practical  you don't need separate models for simple vs. complex tasks.

#### QwQ (Released November 2024, updated March 2025)

QwQ (Qwen with Questions) is Alibaba's dedicated reasoning model  trained specifically for extended chain-of-thought reasoning.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 32 billion |
| **Context Window** | 128K tokens |
| **Key Strength** | Strong reasoning at 32B size, open weights |
| **License** | Apache 2.0 |

### Qwen Family: Strengths & Weaknesses

**Strengths:**
- Most comprehensive size range of any open model family (0.5B to 235B)
- Apache 2.0 license  maximally permissive for commercial use
- Excellent multilingual support, especially Chinese-English
- Hybrid thinking mode is uniquely practical
- Specialized variants (Coder, Math, VL) for specific tasks
- Very competitive benchmarks at every size
- Active development with frequent updates

**Weaknesses:**
- Less tested in production at scale compared to Llama
- Training data composition not fully disclosed
- Smaller Western community/ecosystem than Llama
- Some models show strength primarily on Chinese-language benchmarks
- Documentation sometimes lags behind releases

---

## 6. DeepSeek Family (DeepSeek AI)

### Overview

DeepSeek, founded in 2023 by the quantitative hedge fund High-Flyer, burst onto the scene in late 2024/early 2025 with models that stunned the industry. Their models achieved frontier-level performance while being trained at a fraction of the cost of Western labs  challenging the assumption that cutting-edge AI requires billions of dollars. DeepSeek-R1 in particular became a cultural moment, briefly causing NVIDIA stock to drop as markets questioned whether massive GPU investments were necessary.

**Creator:** DeepSeek AI (Hangzhou, China)
**Architecture:** Decoder-only Transformer, MoE with innovative attention mechanisms
**Source:** Open weights (MIT License  maximally permissive)
**Access:** Hugging Face, Ollama, DeepSeek API, major cloud platforms

### Model Lineup (as of mid-2026)

#### DeepSeek-V3 (Released December 2024)

DeepSeek-V3 demonstrated that a top-tier model could be trained for approximately $5.5 million in compute  a fraction of the estimated $100M+ for GPT-4. It introduced several architectural innovations.

| Attribute | Details |
|-----------|---------|
| **Total Parameters** | 671 billion |
| **Active Parameters** | 37 billion per token |
| **Architecture** | MoE with 256 routed experts + 1 shared expert |
| **Context Window** | 128K tokens |
| **Key Innovation** | Multi-head Latent Attention (MLA)  compresses KV cache dramatically |
| **Training Cost** | ~$5.5 million (2,788K H800 GPU hours) |
| **License** | MIT |

**Architectural Innovations:**
- **Multi-head Latent Attention (MLA):** Instead of caching full key-value pairs, MLA compresses them into a low-rank latent space, reducing memory usage by 5-13x during inference
- **DeepSeekMoE:** Load-balanced routing with auxiliary-loss-free balancing strategy
- **Multi-Token Prediction (MTP):** Predicts multiple future tokens simultaneously during training, improving sample efficiency

#### DeepSeek-R1 (Released January 2025)

The model that shook the world. DeepSeek-R1 is a reasoning model that matches o1 on many benchmarks while being open-weight and dramatically cheaper. Its release prompted a re-evaluation of the AI industry's economics.

| Attribute | Details |
|-----------|---------|
| **Total Parameters** | 671 billion |
| **Active Parameters** | 37 billion per token |
| **Architecture** | MoE (same as V3, fine-tuned for reasoning) |
| **Context Window** | 128K tokens |
| **Key Strength** | Frontier reasoning at open-weight, tiny cost |
| **API Pricing** | $0.55 / 1M input tokens, $2.19 / 1M output tokens |
| **License** | MIT |

**Training Approach:**
DeepSeek-R1 was trained using a novel approach:
1. Start with DeepSeek-V3 base
2. Apply reinforcement learning (RL) directly  minimal supervised fine-tuning
3. The model **discovered** chain-of-thought reasoning through RL reward signals
4. This was notable because it showed that reasoning capabilities can emerge from RL alone

**Distilled Variants:**
DeepSeek also released distilled versions of R1's reasoning capabilities into smaller models:

| Distilled Model | Base Model | Parameters | Key Notes |
|----------------|------------|-----------|-----------|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen 2.5 1.5B | 1.5B | Reasoning on edge devices |
| DeepSeek-R1-Distill-Qwen-7B | Qwen 2.5 7B | 7B | Strong small reasoning model |
| DeepSeek-R1-Distill-Qwen-14B | Qwen 2.5 14B | 14B | Excellent reasoning-to-cost ratio |
| DeepSeek-R1-Distill-Qwen-32B | Qwen 2.5 32B | 32B | Near-R1 reasoning in 32B |
| DeepSeek-R1-Distill-Llama-8B | Llama 3.1 8B | 8B | Llama-based reasoning |
| DeepSeek-R1-Distill-Llama-70B | Llama 3.3 70B | 70B | High-quality reasoning |

### DeepSeek Family: Strengths & Weaknesses

**Strengths:**
- Frontier performance at dramatically lower training cost
- MIT license  maximally open, no usage restrictions
- MLA architecture is genuinely innovative (industry-wide impact)
- R1 proves reasoning can emerge from pure RL
- Distilled variants bring reasoning to small models
- Incredibly cheap API pricing
- Architectural innovations adopted by other labs

**Weaknesses:**
- Chinese government censorship in API responses (sensitive topics)
- Self-hosted versions don't have censorship but require significant hardware (671B MoE)
- Company stability/longevity questions (startup backed by hedge fund)
- Limited multimodal capabilities compared to GPT-4o/Gemini
- R1 can produce very long, rambling reasoning chains
- Smaller community support and documentation than Llama

---

## 7. Mistral Family (Mistral AI)

### Overview

Mistral AI, founded in 2023 by former Meta and Google DeepMind researchers, is Europe's leading AI lab. Based in Paris, Mistral has positioned itself as the "European champion" of AI  producing both open and commercial models. They pioneered the use of MoE in open models (Mixtral) and have a strong focus on efficiency and multilingual capabilities.

**Creator:** Mistral AI (Paris, France)
**Architecture:** Decoder-only Transformer (dense and MoE variants)
**Source:** Mixed (some open-weight, some proprietary)
**Access:** La Plateforme (Mistral's API), Hugging Face, Ollama, major cloud providers

### Model Lineup (as of mid-2026)

#### Mistral Large (Latest: Mistral Large 2, Released July 2024)

Mistral's flagship commercial model  a powerful, closed-weight model designed to compete with GPT-4o and Claude 3.5 Sonnet.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 123 billion |
| **Context Window** | 128K tokens |
| **Key Strength** | Strong multilingual, good coding, competitive quality |
| **Pricing (API)** | $2.00 / 1M input tokens, $6.00 / 1M output tokens |

#### Mistral Medium (Mistral Small 3.1, Released March 2025)

Despite the confusing naming, Mistral Small 3.1 serves as Mistral's efficient mid-range offering. It's open-weight and punches well above its weight class.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 24 billion |
| **Context Window** | 128K tokens |
| **Modalities** | Text, image input; text output |
| **License** | Apache 2.0 |
| **Key Strength** | Vision capability, efficient, open-weight |
| **Pricing (API)** | $0.10 / 1M input tokens, $0.30 / 1M output tokens |

#### Codestral (Released May 2024)

Mistral's code-specialized model  trained specifically for code generation, code completion, and software engineering tasks.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 22 billion |
| **Context Window** | 32K tokens |
| **Key Strength** | Code generation, supports 80+ programming languages |
| **License** | Mistral AI Non-Production License (free for research/testing) |

#### Codestral Mamba (Released July 2024)

An interesting experiment  Codestral but built on the Mamba architecture (a state-space model rather than Transformer). This enables linear-time inference, meaning the model doesn't slow down with longer inputs.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 7.3 billion |
| **Architecture** | Mamba (state-space model, NOT a Transformer) |
| **Context Window** | Theoretically unlimited (linear complexity) |
| **Key Strength** | Constant inference speed regardless of context length |

#### Mistral Saba (Released February 2025)

A model specifically designed for Middle Eastern and South Asian languages  one of the few models explicitly targeting these language families.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 24 billion |
| **Context Window** | 32K tokens |
| **Key Strength** | Arabic, Hindi, and South Asian language support |

### Mistral Family: Strengths & Weaknesses

**Strengths:**
- Pioneer of open-weight MoE models (Mixtral series)
- Strong multilingual capabilities, especially European languages
- Codestral is excellent for code tasks
- Good efficiency  small models punch above their weight
- European AI sovereignty (data handling under EU regulations)
- Apache 2.0 licensing on key models

**Weaknesses:**
- Confusing naming/versioning (Mistral Small vs Medium vs Large changes)
- Smaller scale than US/Chinese labs
- Flagship (Large) model not as competitive with GPT-4.1/Claude 4
- Limited multimodal capabilities
- Smaller ecosystem and community
- Some models have restrictive licenses

---

## 8. Grok Family (xAI)

### Overview

xAI, founded by Elon Musk in 2023, created the Grok model family. Named after a term coined by Robert Heinlein meaning "to understand profoundly," Grok models are trained with access to real-time X (Twitter) data and positioned as uncensored, witty alternatives to more safety-focused models.

**Creator:** xAI (San Francisco / Memphis, USA)
**Architecture:** Decoder-only Transformer (dense and MoE)
**Source:** Mixed (Grok-1 was open-weight; Grok 3 is proprietary)
**Access:** X platform (via Grok chat), xAI API

### Model Lineup (as of mid-2026)

#### Grok 3 (Released February 2025)

Grok 3 was trained on xAI's "Colossus" supercomputer cluster (reportedly 100,000 NVIDIA H100 GPUs) and represented a massive jump in capability over Grok 2.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed (estimated very large) |
| **Context Window** | 128K tokens |
| **Key Strength** | Real-time information, less content filtering, strong reasoning |
| **Pricing (API)** | $3.00 / 1M input tokens, $15.00 / 1M output tokens |

#### Grok 3 Mini (Released February 2025)

A smaller, faster, cheaper variant of Grok 3 with built-in "thinking" capabilities.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | Undisclosed |
| **Context Window** | 128K tokens |
| **Key Strength** | Fast reasoning model, cost-effective |
| **Pricing (API)** | $0.30 / 1M input tokens, $0.50 / 1M output tokens |

### Grok Family: Strengths & Weaknesses

**Strengths:**
- Access to real-time X (Twitter) data for current events
- Fewer content restrictions  more willing to engage with edgy/controversial topics
- Strong reasoning capabilities (Grok 3 competitive on benchmarks)
- "DeepSearch" functionality for research tasks
- Grok 3 Mini offers good reasoning at low cost

**Weaknesses:**
- Tightly coupled to the X/Twitter ecosystem
- Limited third-party integrations
- Grok 3 weights not open (despite Musk's open-source advocacy)
- Smaller developer ecosystem
- Less refined safety training (feature for some, bug for others)
- API access less mature than OpenAI/Anthropic/Google

---

## 9. Phi Family (Microsoft)

### Overview

Microsoft's Phi models have been trailblazers in the "small language model" (SLM) space. Starting with Phi-1 (1.3B) in 2023, Microsoft Research demonstrated that carefully curated, high-quality training data ("textbook-quality") could produce surprisingly capable small models. The Phi family proves that size isn't everything.

**Creator:** Microsoft Research (Redmond, USA)
**Architecture:** Decoder-only Transformer (dense)
**Source:** Open weights (MIT License)
**Access:** Hugging Face, Ollama, Azure AI, ONNX Runtime

### Model Lineup (as of mid-2026)

#### Phi-3 (Released April 2024)

Phi-3 expanded the family to multiple sizes while maintaining the "quality over quantity" training philosophy.

| Variant | Parameters | Context Window | Key Notes |
|---------|-----------|----------------|-----------|
| Phi-3-mini | 3.8 billion | 4K / 128K | Runs on phones, surprisingly strong |
| Phi-3-small | 7 billion | 8K / 128K | Good balance |
| Phi-3-medium | 14 billion | 4K / 128K | Best Phi-3 quality |
| Phi-3-vision | 4.2 billion | 128K | Multimodal (text + image) |

#### Phi-4 (Released December 2024)

Phi-4 focused on reasoning quality, particularly mathematical and scientific reasoning.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 14 billion |
| **Context Window** | 16K tokens |
| **Key Strength** | Exceptional reasoning for its size, strong math |
| **License** | MIT |

**Training Innovation:** Phi-4 used synthetic data extensively  data generated by larger models, then carefully filtered and curated. This approach is controversial but proved effective: Phi-4 14B outperformed many 70B+ models on math benchmarks.

#### Phi-4-mini (Released February 2025)

A smaller variant designed for deployment on edge devices while maintaining strong reasoning.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 3.8 billion |
| **Context Window** | 128K tokens |
| **Key Strength** | Strong reasoning at 3.8B, long context, runs anywhere |
| **License** | MIT |

#### Phi-4-multimodal (Released February 2025)

Expands Phi-4 to handle text, images, audio, and video.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 5.6 billion |
| **Context Window** | 128K tokens |
| **Modalities** | Text, image, audio, video input; text output |
| **Key Strength** | Full multimodality in a 5.6B model |
| **License** | MIT |

### Phi Family: Strengths & Weaknesses

**Strengths:**
- Best-in-class performance at small sizes (3-14B parameter range)
- MIT License  fully open, no restrictions
- Excellent for edge/mobile deployment
- Pioneered the "quality data > more parameters" approach
- ONNX optimized for cross-platform deployment
- Strong mathematical reasoning relative to size
- Microsoft's research backing ensures continued innovation

**Weaknesses:**
- Limited to relatively small sizes (no 70B+ model)
- Creative writing and open-ended generation less strong than larger models
- Knowledge breadth limited compared to models trained on more diverse data
- Synthetic data training raises questions about capability ceilings
- Not as widely deployed in production as Llama or Qwen
- Smaller context windows on some variants (Phi-4 base is only 16K)

---

## 10. Gemma Family (Google)

### Overview

Gemma is Google's open-weight model family  essentially the "open-source little sibling" of Gemini. Built from the same research that powers Gemini, Gemma models are designed to be lightweight, efficient, and easy to deploy while still being highly capable. They're particularly popular for fine-tuning and research.

**Creator:** Google DeepMind (Mountain View, USA)
**Architecture:** Decoder-only Transformer (dense)
**Source:** Open weights (Gemma Terms of Use  permissive, free for most uses)
**Access:** Hugging Face, Kaggle, Ollama, Google Cloud Vertex AI

### Model Lineup (as of mid-2026)

#### Gemma 2 (Released June 2024)

Gemma 2 significantly improved over the original Gemma, with three size variants and several training innovations.

| Variant | Parameters | Context Window | Key Notes |
|---------|-----------|----------------|-----------|
| Gemma 2 2B | 2.6 billion | 8K tokens | Tiny but effective |
| Gemma 2 9B | 9 billion | 8K tokens | Best-in-class at 9B |
| Gemma 2 27B | 27 billion | 8K tokens | Near-70B quality |

**Training Innovations:**
- **Knowledge Distillation:** Gemma 2 models were trained using knowledge distillation from larger Gemini models, transferring capabilities more efficiently
- **Model Merging:** Used techniques from model merging literature during training

#### Gemma 3 (Released March 2025)

Gemma 3 brought multimodality, longer context, and better multilingual support.

| Variant | Parameters | Context Window | Key Notes |
|---------|-----------|----------------|-----------|
| Gemma 3 1B | 1 billion | 32K tokens | Edge devices, fastest |
| Gemma 3 4B | 4 billion | 128K tokens | Good for mobile/laptop |
| Gemma 3 12B | 12 billion | 128K tokens | Strong mid-range |
| Gemma 3 27B | 27 billion | 128K tokens | Best Gemma, near-frontier |

**Key Feature  Multimodality:** Gemma 3 (4B, 12B, 27B) supports image input alongside text  a first for the Gemma family. The vision capability uses a SigLIP-based vision encoder.

**Key Feature  ShieldGemma 2:** Released alongside Gemma 3, ShieldGemma 2 is a specialized safety classifier built on Gemma 3 4B, designed to filter harmful content in both text and images.

### Gemma Family: Strengths & Weaknesses

**Strengths:**
- Excellent quality-to-size ratio (distilled from Gemini)
- Easy to fine-tune and adapt
- Good for research and experimentation
- Long context (128K) in Gemma 3
- Multimodal capabilities (Gemma 3)
- Well-documented and easy to deploy
- Optimized for Google's TPU and Vertex AI ecosystem

**Weaknesses:**
- No very large variants (tops at 27B)
- Gemma 2 had limited 8K context
- License more restrictive than Apache 2.0 (Gemma Terms of Use)
- Fewer community fine-tunes than Llama
- Not as strong as Llama 3.1 70B for production deployments
- Limited audio/video capabilities

---

## 11. Command-R Family (Cohere)

### Overview

Cohere is an enterprise-focused AI company founded by former Google researchers (including Aidan Gomez, a co-author of the original Transformer paper). Their Command-R family is specifically designed for enterprise RAG (Retrieval-Augmented Generation)  models that excel at working with retrieved documents, following business instructions, and generating grounded responses.

**Creator:** Cohere (Toronto, Canada)
**Architecture:** Decoder-only Transformer (dense)
**Source:** Open weights (CC-BY-NC for research, commercial license available)
**Access:** Cohere API, Hugging Face, cloud providers

### Model Lineup (as of mid-2026)

#### Command R (Released March 2024)

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 35 billion |
| **Context Window** | 128K tokens |
| **Key Strength** | RAG, multilingual (10 languages), tool use |
| **License** | CC-BY-NC (non-commercial), commercial license available |

#### Command R+ (Released April 2024)

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 104 billion |
| **Context Window** | 128K tokens |
| **Key Strength** | Enhanced RAG, better reasoning, stronger multilingual |
| **License** | CC-BY-NC (non-commercial), commercial license available |

#### Command A (Released March 2025)

The latest evolution of Cohere's enterprise offering  designed specifically for agentic workflows and complex tool use.

| Attribute | Details |
|-----------|---------|
| **Parameter Count** | 111 billion |
| **Context Window** | 256K tokens |
| **Key Strength** | Agentic workflows, structured outputs, enterprise RAG |
| **Pricing (API)** | $2.50 / 1M input tokens, $10.00 / 1M output tokens |

### Command-R Family: Strengths & Weaknesses

**Strengths:**
- Best-in-class RAG capabilities (built-in citation generation)
- Excellent at grounded generation (answers sourced from provided documents)
- Strong multilingual support
- Enterprise-focused features (data privacy, deployment options)
- Good tool use and structured output generation
- Cohere Embed (embedding models) complement the generative models well

**Weaknesses:**
- Less capable for general-purpose creative tasks
- Smaller community than Llama/Qwen
- Non-commercial license for open-weight versions limits adoption
- Less competitive on pure benchmarks vs. Claude/GPT
- Limited multimodal capabilities
- Smaller model ecosystem (fewer fine-tunes and variants)

---

## 12. Other Notable Models

### Jamba (AI21 Labs)

**Creator:** AI21 Labs (Tel Aviv, Israel)
**Architecture:** Hybrid SSM-Transformer (combines Mamba SSM with Transformer layers in an MoE setup)
**Why It Matters:** Jamba was the first production model to combine State Space Models (SSMs) with Transformers  achieving linear-time inference for long contexts while maintaining Transformer-quality performance.

| Variant | Parameters | Context Window | Notes |
|---------|-----------|----------------|-------|
| Jamba 1.5 Mini | 52B total, 12B active | 256K tokens | Efficient, long context |
| Jamba 1.5 Large | 398B total, 94B active | 256K tokens | High quality MoE+SSM hybrid |

### DBRX (Databricks)

**Creator:** Databricks (San Francisco, USA)
**Architecture:** Fine-grained MoE (132B total, 36B active, 16 experts with 4 active)
**Why It Matters:** First major open MoE model from a data platform company, optimized for enterprise data tasks. Released March 2024 under Databricks Open Model License.

### Yi (01.AI)

**Creator:** 01.AI (Beijing, China)  founded by Kai-Fu Lee
**Architecture:** Dense Transformer

| Variant | Parameters | Context | Notes |
|---------|-----------|---------|-------|
| Yi-1.5 6B/9B/34B | 6B/9B/34B | 200K | Strong multilingual |
| Yi-Lightning | Undisclosed | Large | Speed-optimized inference |

**Why It Matters:** Demonstrated that smaller Chinese AI labs could produce competitive models, particularly strong in Chinese and English bilingual tasks.

### Falcon (Technology Innovation Institute)

**Creator:** Technology Innovation Institute (Abu Dhabi, UAE)
**Architecture:** Dense Transformer

| Variant | Parameters | Context | Notes |
|---------|-----------|---------|-------|
| Falcon 2 11B | 11 billion | 8K | Trained on 5.5T tokens |
| Falcon 3 (Family) | 1B/3B/7B/10B | 32K | Efficient, multilingual (incl. Arabic) |

**Why It Matters:** Major AI model from the Middle East; Falcon 3 uses depth up-scaling (training smaller models and expanding to larger ones), demonstrating efficiency in training.

### StableLM / Stable Diffusion (Stability AI)

**Creator:** Stability AI (London, UK)
**Why It Matters:** While Stability AI is best known for Stable Diffusion (image generation), they also produced StableLM language models. Stability AI pioneered the open-source image generation revolution:

| Model | Type | Notes |
|-------|------|-------|
| Stable Diffusion 3.5 | Image generation | MMDiT architecture |
| SDXL | Image generation | Still widely used |
| StableLM 2 1.6B | Language model | Small, efficient |
| Stable Audio | Audio generation | Music and sound |

### Colossal-LLaMA / Open-Source Fine-Tunes

The Llama ecosystem has spawned hundreds of notable fine-tuned variants:

| Model | Base | Purpose |
|-------|------|---------|
| Nous-Hermes | Llama/Qwen | General chat, instruction following |
| WizardLM | Llama | Complex instruction following |
| OpenHermes | Mistral/Llama | High-quality general purpose |
| Dolphin | Various | Uncensored general purpose |
| CodeLlama | Llama | Code-specialized |
| MedLlama | Llama | Medical domain |

---

## Comparison Tables

### Table 1: All Major Models  Size, Architecture, and Openness

| Model | Creator | Total Params | Active Params | Architecture | Open? | Context | Multimodal |
|-------|---------|-------------|---------------|-------------|-------|---------|------------|
| GPT-4o | OpenAI | ~200B (est.) | ~200B (est.) | Dense Transformer | ❌ Closed | 128K | ✅ Text, Image, Audio |
| GPT-4o-mini | OpenAI | Undisclosed | Undisclosed | Dense Transformer | ❌ Closed | 128K | ✅ Text, Image, Audio |
| GPT-4.1 | OpenAI | Undisclosed | Undisclosed | Transformer | ❌ Closed | 1M | ✅ Text, Image |
| GPT-4.1-mini | OpenAI | Undisclosed | Undisclosed | Transformer | ❌ Closed | 1M | ✅ Text, Image |
| GPT-4.1-nano | OpenAI | Undisclosed | Undisclosed | Transformer | ❌ Closed | 1M | ✅ Text, Image |
| o3 | OpenAI | Undisclosed | Undisclosed | Transformer | ❌ Closed | 200K | ✅ Text, Image |
| o3-mini | OpenAI | Undisclosed | Undisclosed | Transformer | ❌ Closed | 200K | ❌ Text |
| o4-mini | OpenAI | Undisclosed | Undisclosed | Transformer | ❌ Closed | 200K | ✅ Text, Image |
| Claude 3.5 Sonnet | Anthropic | ~175B (est.) | ~175B (est.) | Dense Transformer | ❌ Closed | 200K | ✅ Text, Image |
| Claude 3.5 Haiku | Anthropic | ~20-30B (est.) | ~20-30B (est.) | Dense Transformer | ❌ Closed | 200K | ✅ Text, Image |
| Claude 4 Sonnet | Anthropic | Undisclosed | Undisclosed | Dense Transformer | ❌ Closed | 200K | ✅ Text, Image |
| Claude 4 Opus | Anthropic | Undisclosed | Undisclosed | Dense Transformer | ❌ Closed | 200K | ✅ Text, Image |
| Gemini 2.0 Flash | Google | Undisclosed | Undisclosed | Transformer | ❌ Closed | 1M | ✅ Text, Img, Audio, Video |
| Gemini 2.5 Pro | Google | Undisclosed | Undisclosed | MoE Transformer | ❌ Closed | 1M | ✅ Text, Img, Audio, Video |
| Gemini 2.5 Flash | Google | Undisclosed | Undisclosed | Transformer | ❌ Closed | 1M | ✅ Text, Img, Audio, Video |
| Llama 3.1 8B | Meta | 8B | 8B | Dense Transformer | ✅ Open | 128K | ❌ Text |
| Llama 3.1 70B | Meta | 70B | 70B | Dense Transformer | ✅ Open | 128K | ❌ Text |
| Llama 3.1 405B | Meta | 405B | 405B | Dense Transformer | ✅ Open | 128K | ❌ Text |
| Llama 3.3 70B | Meta | 70B | 70B | Dense Transformer | ✅ Open | 128K | ❌ Text |
| Llama 4 Scout | Meta | 109B | 17B | MoE (16 experts) | ✅ Open | 10M | ✅ Text, Image |
| Llama 4 Maverick | Meta | 400B | 17B | MoE (128 experts) | ✅ Open | 1M | ✅ Text, Image |
| Qwen 2.5 7B | Alibaba | 7B | 7B | Dense Transformer | ✅ Open (Apache 2.0) | 128K | ❌ Text |
| Qwen 2.5 72B | Alibaba | 72B | 72B | Dense Transformer | ✅ Open (Apache 2.0) | 128K | ❌ Text |
| Qwen 3 8B | Alibaba | 8B | 8B | Dense Transformer | ✅ Open (Apache 2.0) | 128K | ❌ Text |
| Qwen 3 32B | Alibaba | 32B | 32B | Dense Transformer | ✅ Open (Apache 2.0) | 128K | ❌ Text |
| Qwen 3 235B-A22B | Alibaba | 235B | 22B | MoE Transformer | ✅ Open (Apache 2.0) | 128K | ❌ Text |
| DeepSeek-V3 | DeepSeek | 671B | 37B | MoE Transformer | ✅ Open (MIT) | 128K | ❌ Text |
| DeepSeek-R1 | DeepSeek | 671B | 37B | MoE Transformer | ✅ Open (MIT) | 128K | ❌ Text |
| Mistral Large 2 | Mistral | 123B | 123B | Dense Transformer | ❌ Closed | 128K | ❌ Text |
| Mistral Small 3.1 | Mistral | 24B | 24B | Dense Transformer | ✅ Open (Apache 2.0) | 128K | ✅ Text, Image |
| Codestral | Mistral | 22B | 22B | Dense Transformer | ⚠️ Non-prod | 32K | ❌ Text |
| Grok 3 | xAI | Undisclosed | Undisclosed | Transformer | ❌ Closed | 128K | ✅ Text, Image |
| Grok 3 Mini | xAI | Undisclosed | Undisclosed | Transformer | ❌ Closed | 128K | ❌ Text |
| Phi-4 | Microsoft | 14B | 14B | Dense Transformer | ✅ Open (MIT) | 16K | ❌ Text |
| Phi-4-mini | Microsoft | 3.8B | 3.8B | Dense Transformer | ✅ Open (MIT) | 128K | ❌ Text |
| Phi-4-multimodal | Microsoft | 5.6B | 5.6B | Dense Transformer | ✅ Open (MIT) | 128K | ✅ Text, Img, Audio, Video |
| Gemma 3 4B | Google | 4B | 4B | Dense Transformer | ✅ Open | 128K | ✅ Text, Image |
| Gemma 3 12B | Google | 12B | 12B | Dense Transformer | ✅ Open | 128K | ✅ Text, Image |
| Gemma 3 27B | Google | 27B | 27B | Dense Transformer | ✅ Open | 128K | ✅ Text, Image |
| Command A | Cohere | 111B | 111B | Dense Transformer | ✅ Open (CC-BY-NC) | 256K | ❌ Text |
| Jamba 1.5 Large | AI21 | 398B | 94B | SSM+Transformer MoE | ✅ Open | 256K | ❌ Text |

### Table 2: Best Models by Use Case (Mid-2026)

| Use Case | Best Closed | Best Open (Large) | Best Open (Small) | Notes |
|----------|------------|-------------------|-------------------|-------|
| **Coding** | Claude 4 Sonnet, Gemini 2.5 Pro | DeepSeek-V3, Qwen 3 235B | Qwen 2.5 Coder 32B, Qwen 3 32B | Claude & Gemini dominate coding benchmarks |
| **Math/Reasoning** | o3, Gemini 2.5 Pro | DeepSeek-R1 | QwQ 32B, DeepSeek-R1-Distill-32B | Reasoning models use chain-of-thought |
| **Creative Writing** | Claude 4 Opus | Llama 3.3 70B, Qwen 3 32B | Llama 3.1 8B (fine-tuned) | Claude has the most natural writing style |
| **Multilingual** | GPT-4.1, Gemini 2.5 Pro | Qwen 3 235B | Qwen 3 8B, Gemma 3 12B | Qwen excels in CJK languages |
| **Vision/Multimodal** | Gemini 2.5 Pro, GPT-4o | Llama 4 Maverick | Gemma 3 12B, Phi-4-multimodal | Gemini has broadest multimodal support |
| **RAG/Enterprise** | Claude 4 Sonnet, GPT-4.1 | Command A | Mistral Small 3.1 | Command-R family purpose-built for RAG |
| **Long Context** | Gemini 2.5 Pro (1M) | Llama 4 Scout (10M) | Gemma 3 4B (128K) | Gemini most reliable at long context |
| **Agentic/Tool Use** | Claude 4 Sonnet, o4-mini | Qwen 3 235B | Qwen 3 8B | Claude pioneered computer use |
| **Edge/Mobile** |  |  | Phi-4-mini (3.8B), Gemma 3 1B | Phi-4-mini best quality at tiny size |
| **Conversational AI** | GPT-4o, Claude 4 Sonnet | Llama 3.3 70B | Llama 3.1 8B | GPT-4o best voice interaction |

### Table 3: API Cost Comparison (Mid-2026, per 1M tokens)

| Model | Input Cost | Output Cost | Cost for 1K queries* | Tier |
|-------|-----------|-------------|----------------------|------|
| GPT-4.1-nano | $0.10 | $0.40 | $0.20 | 💚 Budget |
| Gemini 2.0 Flash | $0.10 | $0.40 | $0.20 | 💚 Budget |
| Gemini 2.5 Flash | $0.15 | $1.00 | $0.42 | 💚 Budget |
| Mistral Small 3.1 | $0.10 | $0.30 | $0.15 | 💚 Budget |
| GPT-4o-mini | $0.15 | $0.60 | $0.30 | 💚 Budget |
| Grok 3 Mini | $0.30 | $0.50 | $0.33 | 💚 Budget |
| GPT-4.1-mini | $0.40 | $1.60 | $0.80 | 💛 Mid-range |
| DeepSeek-R1 (API) | $0.55 | $2.19 | $1.04 | 💛 Mid-range |
| Claude 3.5 Haiku | $0.80 | $4.00 | $1.84 | 💛 Mid-range |
| o3-mini | $1.10 | $4.40 | $2.18 | 💛 Mid-range |
| o4-mini | $1.10 | $4.40 | $2.18 | 💛 Mid-range |
| Gemini 2.5 Pro | $1.25 | $10.00 | $4.13 | 💛 Mid-range |
| Mistral Large 2 | $2.00 | $6.00 | $3.20 | 💛 Mid-range |
| GPT-4.1 | $2.00 | $8.00 | $3.80 | 🟠 Premium |
| GPT-4o | $2.50 | $10.00 | $4.75 | 🟠 Premium |
| Command A | $2.50 | $10.00 | $4.75 | 🟠 Premium |
| Claude 4 Sonnet | $3.00 | $15.00 | $6.75 | 🟠 Premium |
| Grok 3 | $3.00 | $15.00 | $6.75 | 🟠 Premium |
| o3 | $10.00 | $40.00 | $19.00 | 🔴 Expensive |
| Claude 4 Opus | $15.00 | $75.00 | $33.75 | 🔴 Expensive |
| o1 | $15.00 | $60.00 | $28.50 | 🔴 Expensive |

*\*Estimated cost for 1,000 queries averaging 500 input tokens and 300 output tokens each*

### Table 4: Best Open-Weight Models by Size Category (Mid-2026)

| Size Category | Best Overall | Best for Code | Best for Reasoning | Best for Chat |
|--------------|-------------|---------------|-------------------|---------------|
| **≤ 1B** | Gemma 3 1B | Qwen 3 0.6B |  | Gemma 3 1B |
| **1-4B** | Phi-4-mini (3.8B) | Qwen 3 4B | DS-R1-Distill-1.5B | Gemma 3 4B |
| **4-10B** | Qwen 3 8B | Qwen 2.5 Coder 7B | DS-R1-Distill-Qwen-7B | Llama 3.1 8B |
| **10-20B** | Qwen 3 14B | Qwen 2.5 Coder 14B | DS-R1-Distill-Qwen-14B | Gemma 3 12B |
| **20-35B** | Qwen 3 32B | Qwen 2.5 Coder 32B | QwQ 32B / DS-R1-Distill-32B | Gemma 3 27B |
| **35-80B** | Llama 3.3 70B | DS-R1-Distill-Llama-70B | DS-R1-Distill-Llama-70B | Llama 3.3 70B |
| **80-200B** | Llama 4 Scout (109B) | Command A (111B) | Qwen 3 235B-A22B | Llama 4 Scout |
| **200B+** | DeepSeek-V3 (671B) | DeepSeek-V3 | DeepSeek-R1 (671B) | DeepSeek-V3 |

> **Note:** "Best" is inherently subjective and changes with benchmarks, tasks, and updates. This table represents broad community consensus as of mid-2026. Always test models on YOUR specific use case.

---

## Choosing the Right Model: A Decision Framework

```
                    What's your priority?
                          │
              ┌───────────┼───────────────┐
              │           │               │
          QUALITY      COST         PRIVACY/CONTROL
              │           │               │
     ┌────────┤      ┌────┤          ┌────┤
     │        │      │    │          │    │
  Reasoning General Budget Free   Local Cloud
     │        │      │    │      (Open  (Self-
     │        │      │    │     weights) host)
     ▼        ▼      ▼    ▼        ▼      ▼
   o3       Claude  Gem.  Self-  Llama  DeepSeek
   R1      4 Sonnet Flash host   Qwen    V3
   Gem     GPT-4.1  Mini  open   Gemma
   2.5P              Nano models  Phi
```

### Quick Decision Guide

**"I need the absolute best quality and don't care about cost"**
→ Claude 4 Opus (general), o3 (reasoning), Gemini 2.5 Pro (multimodal)

**"I need great quality at a reasonable price"**
→ Claude 4 Sonnet, GPT-4.1, Gemini 2.5 Pro

**"I need to process millions of tokens cheaply"**
→ Gemini 2.0/2.5 Flash, GPT-4.1-nano, DeepSeek API

**"I want to run AI locally on my own hardware"**
→ Qwen 3 8B/32B, Llama 3.3 70B, Gemma 3 12B/27B (depending on your GPU)

**"I'm building an enterprise RAG system"**
→ Command A, Claude 4 Sonnet, GPT-4.1

**"I need code assistance"**
→ Claude 4 Sonnet, Gemini 2.5 Pro, Qwen 2.5 Coder 32B

**"I need to run AI on a phone/edge device"**
→ Phi-4-mini (3.8B), Gemma 3 1B, Qwen 3 0.6B

**"I need the most permissive license possible"**
→ DeepSeek (MIT), Qwen 3 (Apache 2.0), Phi-4 (MIT)

---

## The Open vs. Closed Source Debate

### Arguments for Closed Models
1. **Higher ceiling:** The very best performance still comes from closed models (though the gap narrows)
2. **Easier to use:** API access, no infrastructure management
3. **Safety:** More controlled deployment, harder to misuse
4. **Features:** Often first to get new capabilities (vision, audio, tool use)

### Arguments for Open Models
1. **Privacy:** Your data never leaves your infrastructure
2. **Cost:** No per-token API charges (just compute costs)
3. **Control:** Fine-tune for your exact use case
4. **Transparency:** Understand what the model is doing
5. **Independence:** No vendor lock-in, no API changes
6. **Innovation:** Community builds on top of base models

### The Gap Is Closing

```
Model Quality Over Time (Illustrative)

Quality
  │
  │                                    ╭── Closed (Claude 4, GPT-4.1, Gemini 2.5)
  │                               ╭───╯
  │                          ╭────╯    ╭── Open (DeepSeek-R1, Qwen 3, Llama 4)
  │                     ╭────╯    ╭────╯
  │                ╭────╯    ╭────╯
  │           ╭────╯    ╭────╯
  │      ╭────╯   ╭─────╯
  │ ╭────╯   ╭────╯
  │─╯───╭────╯           Gap narrowing ←→
  └─────┴──────────────────────────────────── Time
  2022  2023    2024      2025      2026
```

---

## Emerging Trends in the Model Landscape

### 1. Mixture of Experts (MoE) Becomes Dominant
MoE models (DeepSeek-V3, Llama 4, Qwen 3, Mixtral) are becoming the standard for large models. By only activating a subset of parameters per token, MoE models achieve higher quality with lower per-token compute cost.

### 2. Reasoning/Thinking Models
The o1/o3 paradigm has been adopted across the industry. Gemini 2.5, Claude 4, DeepSeek-R1, QwQ, and Qwen 3 all support extended reasoning. This trend shows no signs of slowing  the ability to "think longer" consistently improves performance on hard problems.

### 3. Small Models Get Much Better
Phi-4-mini (3.8B) and Gemma 3 4B demonstrate that small models can now handle tasks that required 70B+ models two years ago. Better training data, distillation from larger models, and architectural improvements are driving this trend.

### 4. Multimodality Is Table Stakes
By mid-2026, virtually every major model family supports at least text + image. Audio and video support are following. The question is no longer "can it see?" but "how well does it see?"

### 5. Context Windows Keep Growing
From 4K (GPT-3.5) to 128K (GPT-4) to 1M (Gemini) to 10M (Llama 4 Scout) in just three years. Long context is becoming the norm, enabling entirely new use cases like whole-codebase analysis and book-length document processing.

### 6. Specialization vs. Generalization
While frontier models try to be good at everything, we're seeing increasing specialization: code models (Codestral, Qwen Coder), math models (Qwen Math), RAG models (Command-R), reasoning models (R1, QwQ), and more.

---

## Quiz: Famous Models of 2026

### Questions

**Q1.** What is Mixture of Experts (MoE), and which major model families use it?
Name at least three.

**Q2.** You need to process a 500,000-token document. Which models have context windows large enough, and which would you choose for cost-effectiveness?

**Q3.** What architectural innovation did DeepSeek introduce that dramatically reduces KV cache memory usage? How does it work at a high level?

**Q4.** Compare Claude 4 Sonnet and GPT-4.1: list two areas where each excels over the other.

**Q5.** You're building a mobile app that needs an on-device AI model (no internet required). The phone has 8GB RAM. Which models would you consider, and why?

**Q6.** What is "hybrid thinking" in Qwen 3 models? Why is this a practical advantage over having separate reasoning and non-reasoning models?

**Q7.** A startup wants to build a customer support chatbot. They process ~10 million tokens/day and need the cheapest possible solution while maintaining good quality. Recommend an approach (model + deployment strategy) and estimate daily costs.

**Q8.** Why was DeepSeek-R1's release considered a pivotal moment in AI? What assumptions about AI development did it challenge?

### Answers

**A1.** MoE is an architecture where the model contains many "expert" sub-networks but only activates a subset for each token. This allows models to have more total parameters (capacity) while keeping per-token compute constant. Major families using MoE:
- **DeepSeek** (V3, R1): 671B total, 37B active, 256 experts
- **Llama 4** (Scout, Maverick): 109-400B total, 17B active
- **Qwen 3** (30B-A3B, 235B-A22B): MoE variants alongside dense models
- **Mixtral** (Mistral): 8x7B (46.7B total, 12.9B active)
- **Jamba** (AI21): Hybrid SSM-Transformer MoE
- **DBRX** (Databricks): 132B total, 36B active

**A2.** Models with ≥500K context: Gemini 2.0 Flash (1M), Gemini 2.5 Pro (1M), Gemini 2.5 Flash (1M), GPT-4.1 (1M), Llama 4 Scout (10M), Llama 4 Maverick (1M). For cost-effectiveness: **Gemini 2.5 Flash** ($0.15/1M input for ≤200K, $0.375/1M for >200K) or **GPT-4.1-nano** ($0.10/1M input) would be the most economical choices. For self-hosted: Llama 4 Scout or Maverick (no per-token cost, just compute).

**A3.** **Multi-head Latent Attention (MLA)**. Instead of caching full key-value pairs for each attention head (which consumes enormous memory for long contexts), MLA compresses the KV representations into a low-rank latent space. The model learns to project keys and values into a much smaller dimensional space, stores those compressed representations, and then projects them back when needed. This reduces KV cache memory by 5-13x with minimal quality loss.

**A4.**
Claude 4 Sonnet excels over GPT-4.1 in:
1. **Coding tasks**  consistently ranked higher on coding benchmarks and real-world coding assistance
2. **Following nuanced, long system prompts**  better adherence to complex multi-part instructions with less drift

GPT-4.1 excels over Claude 4 Sonnet in:
1. **Long context**  1M tokens vs 200K tokens, with strong retrieval throughout
2. **Ecosystem/integrations**  wider third-party tool support, more mature API features

**A5.** With 8GB RAM on mobile, after accounting for OS and app overhead (~3-4GB available for the model):
- **Phi-4-mini (3.8B)** at Q4 quantization (~2.2GB)  Best quality-to-size ratio, MIT license
- **Gemma 3 1B** at Q8 quantization (~1.2GB)  Fast, multimodal (can process images)
- **Qwen 3 0.6B** at Q8 quantization (~0.7GB)  Extremely lightweight, hybrid thinking
- **Qwen 3 1.7B** at Q4 quantization (~1.0GB)  Good balance
These are all chosen because they fit in available memory with quantization while maintaining useful quality.

**A6.** Hybrid thinking in Qwen 3 means a single model can switch between:
- **Thinking mode**: Extended chain-of-thought reasoning (produces `<think>...</think>` blocks)
- **Non-thinking mode**: Fast, direct responses without reasoning overhead

The practical advantage is that you deploy ONE model that handles both simple queries (fast, cheap) and complex problems (slower, higher quality). Without hybrid thinking, you'd need to deploy two separate models (e.g., a base model + a reasoning model), doubling infrastructure costs and adding routing complexity.

**A7.** Recommendation: **Self-host Qwen 3 8B** or **Llama 3.1 8B** on a GPU server (e.g., RTX 4090 or cloud A10G).
- 10M tokens/day ÷ ~50 tokens/second ≈ ~55 hours of compute → need 3 GPUs for 24/7 service
- Cloud GPU cost: ~3 × A10G instances at ~$1/hr each = ~$72/day
- vs. API approach: Gemini 2.0 Flash at $0.10/1M input + $0.40/1M output ≈ ~$5/day
- **Best approach**: Use Gemini 2.0 Flash or GPT-4.1-nano via API at ~$2-5/day
- If privacy matters, self-host Qwen 3 8B at ~$72/day cloud or buy hardware (~$3K one-time for a single RTX 4090 setup, much less per day amortized)

**A8.** DeepSeek-R1 was pivotal because it:
1. **Matched o1-level reasoning** while being fully open-weight (MIT license)
2. **Was trained for ~$5.5M** vs. estimated $100M+ for comparable Western models
3. **Proved reasoning emerges from RL alone**  minimal supervised fine-tuning needed
4. **Challenged the "scaling requires massive capital" assumption**  suggesting clever architecture and training matter more than raw compute spending
5. **Shook financial markets**  NVIDIA stock dropped as investors questioned whether massive GPU purchases were necessary
6. **Democratized reasoning AI**  anyone could download and run frontier-level reasoning models

This challenged three key assumptions: (a) that frontier AI requires massive budgets, (b) that reasoning requires extensive human-annotated reasoning data, and (c) that closed-source labs have an insurmountable advantage.

---

## Summary

The AI model landscape in mid-2026 is characterized by **abundance, competition, and rapid democratization**. Key takeaways:

1. **No single model dominates everything.** Different models excel at different tasks.
2. **Open models are catching up fast.** DeepSeek-R1, Qwen 3, and Llama 4 rival closed models on many tasks.
3. **MoE architecture is winning.** It offers the best quality-per-compute tradeoff.
4. **Reasoning models are transformative.** The ability to "think" extends AI capabilities significantly.
5. **Small models are surprisingly good.** 3-14B parameter models handle tasks that needed 100B+ two years ago.
6. **Cost is dropping rapidly.** What cost $100/day in 2023 can be done for $1/day in 2026.
7. **Choose based on your needs,** not hype. The best model is the one that solves YOUR problem within YOUR budget.

In the next chapter, we'll explore how to run these models **locally** on your own hardware  turning your laptop or gaming PC into a personal AI server.
