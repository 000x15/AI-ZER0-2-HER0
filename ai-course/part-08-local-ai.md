# Part 8: Local AI  Running Models on Your Own Hardware

> *"The best AI is the one that works for you, on your terms, on your hardware."*

---

## Introduction: What Does "Local AI" Mean?

Imagine two ways to watch a movie. You can **stream it** from Netflix  convenient, works on any device, but you need internet, you pay monthly, and Netflix decides what's available. Or you can **download it** to your hard drive  you need storage space, but it's yours, it works offline, it's private, and nobody can take it away.

**Local AI** is like downloading the movie. Instead of sending your prompts to OpenAI's or Google's servers and waiting for a response, you run the AI model directly on your own computer, phone, or server. The model lives on your hardware, your data never leaves your machine, and you have complete control.

```
┌──────────────────────────────────────────────────────────────────┐
│                     CLOUD INFERENCE                               │
│                                                                   │
│  Your Computer ──── Internet ──── Cloud Server ──── GPU Cluster  │
│  "What is X?"        │           (OpenAI/Google)    (A100/H100)  │
│       │               │                │                          │
│       │          Your prompt         Model runs              │
│       │          travels to          on THEIR                │
│       │          their servers       hardware                │
│       ▼                                                          │
│  Response comes back (100-2000ms latency)                        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     LOCAL INFERENCE                                │
│                                                                   │
│  Your Computer ──── Your GPU/CPU ──── Model (on your disk)       │
│  "What is X?"           │                                         │
│       │            Model runs                                     │
│       │            on YOUR                                        │
│       │            hardware                                       │
│       ▼                                                           │
│  Response generated locally (no internet needed)                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Why Run AI Locally?

### 1. Privacy & Security

This is the #1 reason for most people and organizations.

When you use ChatGPT, Claude, or Gemini, your prompts are sent to remote servers. Even with privacy policies and data handling agreements, you're trusting a third party with your data. For many use cases, this is unacceptable:

| Scenario | Why Local Matters |
|----------|-------------------|
| **Medical data** | HIPAA requires strict data handling  sending patient info to cloud APIs may violate regulations |
| **Legal documents** | Attorney-client privilege could be compromised |
| **Source code** | Proprietary code sent to APIs could be exposed |
| **Financial data** | Trading strategies, financial models, personal finances |
| **Personal journals** | Private thoughts, therapy notes, personal writing |
| **Government/military** | Classified information cannot leave secure networks |
| **Corporate trade secrets** | R&D data, unreleased product info, internal communications |

With local AI, your data **never leaves your machine**. There's no API call, no network request, no third-party server. The computation happens entirely within your hardware.

### 2. Cost

API pricing adds up fast, especially at scale:

| Usage Level | Cloud Cost (GPT-4.1) | Cloud Cost (Budget API) | Local Cost* |
|------------|----------------------|------------------------|-------------|
| 10K queries/month | ~$38/month | ~$2/month | ~$0 (hardware amortized) |
| 100K queries/month | ~$380/month | ~$20/month | ~$15/month (electricity) |
| 1M queries/month | ~$3,800/month | ~$200/month | ~$50/month (electricity) |

*\*After initial hardware investment. Assumes home electricity costs.*

For high-volume users, the economics of local inference become compelling. A $1,500 GPU pays for itself within months if you're a heavy API user.

### 3. Offline Access

Local models work without internet. This matters for:
- Airplane or remote area usage
- Environments with unreliable internet
- Air-gapped networks (security-critical systems)
- Developing countries with limited connectivity
- Field research, military operations, disaster response

### 4. Customization & Control

With local models, you can:
- **Fine-tune** on your specific data and domain
- **Remove safety filters** if needed for your use case (research, creative writing)
- **Modify system prompts** without API restrictions
- **Control output format** precisely
- **Chain models** together without API latency
- **Run specialized models** fine-tuned for your exact task

### 5. Speed (Latency)

Cloud APIs have inherent latency:
- Network round-trip: 50-200ms
- Queue wait time: 0-5000ms (varies with load)
- Time-to-first-token: 200-2000ms

Local inference eliminates network latency entirely. For small models on good hardware, you can get time-to-first-token under 50ms  making the AI feel instantaneous.

### 6. Independence

- No API key management
- No rate limits
- No terms of service changes
- No model deprecation (you have the weights forever)
- No price increases
- No service outages
- No vendor lock-in

---

## Hardware for Local AI

### Understanding the Compute Pipeline

When a model generates text, here's what happens for each token:

```
┌─────────────────────────────────────────────────────────┐
│                TOKEN GENERATION PIPELINE                  │
│                                                           │
│  1. Load model weights from storage → RAM/VRAM            │
│     (One-time at startup)                                 │
│                                                           │
│  2. For each token:                                       │
│     a. Look up input token embedding                      │
│     b. Multiply through all layers:                       │
│        - Self-attention (matrix multiplications)          │
│        - Feed-forward network (matrix multiplications)    │
│        - Layer normalization                              │
│     c. Output probability distribution over vocabulary    │
│     d. Sample next token                                  │
│                                                           │
│  Key bottleneck: Matrix multiplications (step 2b)         │
│  These are MASSIVELY parallel operations                  │
│  → Perfect for GPUs (thousands of cores)                  │
│  → Slow on CPUs (few cores, but possible)                 │
└─────────────────────────────────────────────────────────┘
```

### CPU Inference

**Yes, you can run AI models on a CPU.** It's slower than GPU inference, but modern CPUs have special instructions that help.

#### How CPU Inference Works

CPUs process data sequentially (a few cores) but have specialized vector instructions that can process multiple numbers simultaneously:

**SIMD (Single Instruction, Multiple Data):**
Instead of adding numbers one at a time, SIMD instructions add 4, 8, 16, or even 32 numbers in a single clock cycle.

```
Traditional:
  Add(a1, b1) → c1
  Add(a2, b2) → c2
  Add(a3, b3) → c3
  Add(a4, b4) → c4
  = 4 operations

SIMD (AVX-256):
  Add([a1,a2,a3,a4], [b1,b2,b3,b4]) → [c1,c2,c3,c4]
  = 1 operation (4x faster!)
```

**Key CPU instruction sets for AI:**

| Instruction Set | Width | CPUs | Impact |
|----------------|-------|------|--------|
| **SSE4** | 128-bit (4 floats) | All modern CPUs | Baseline, 4x parallel |
| **AVX2** | 256-bit (8 floats) | Intel Haswell+ (2013), AMD Zen+ | 8x parallel |
| **AVX-512** | 512-bit (16 floats) | Intel Skylake-X+, AMD Zen 4+ | 16x parallel |
| **AVX-VNNI** | 256-bit | Intel Alder Lake+, AMD Zen 5 | Accelerated INT8 operations |
| **AMX** | Tiles (matrix) | Intel Sapphire Rapids+ | Native matrix multiplication |
| **ARM NEON** | 128-bit | All ARM chips (Apple M-series, Snapdragon) | 4x parallel on ARM |
| **ARM SME** | Scalable | ARM v9+ | Scalable matrix operations |

**Practical CPU Performance (tokens/second with Llama 3.1 8B Q4):**

| CPU | Cores/Threads | RAM Speed | ~Performance |
|-----|--------------|-----------|-------------|
| Intel i5-12400 | 6C/12T | DDR4-3200 | ~5-8 tok/s |
| Intel i7-13700K | 16C/24T | DDR5-5600 | ~10-15 tok/s |
| AMD Ryzen 7 7800X3D | 8C/16T | DDR5-5200 | ~10-14 tok/s |
| AMD Ryzen 9 7950X | 16C/32T | DDR5-5200 | ~15-22 tok/s |
| Apple M2 Pro | 12C | Unified 200GB/s | ~15-20 tok/s |
| Apple M3 Max | 16C | Unified 400GB/s | ~25-35 tok/s |
| Apple M4 Max | 16C | Unified 546GB/s | ~35-50 tok/s |

**Key Insight:** For CPU inference, **memory bandwidth** matters more than core count. The bottleneck is feeding data to the CPU fast enough, not the computation itself. This is why Apple Silicon excels  its unified memory has very high bandwidth.

### GPU Inference

GPUs are the **natural home** for AI inference. They were designed for exactly the type of computation neural networks need: thousands of simple operations running in parallel.

#### Why GPUs Are Perfect for AI

```
CPU vs GPU Architecture:

CPU (e.g., Ryzen 9 7950X):
┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐  ← 16 powerful cores
│  │ │  │ │  │ │  │ │  │ │  │ │  │ │  │     Each can do complex tasks
└──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘     independently
...  (up to 32 threads)

GPU (e.g., RTX 4090):
┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐ ← 16,384 CUDA cores
│││││││││││││││││││││││││││││││││││││││││││  Each does simple math
└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘   but ALL AT ONCE
... (thousands more)

AI inference = millions of multiply-add operations
= Perfect for GPU's massively parallel architecture
```

#### CUDA Cores vs Tensor Cores

NVIDIA GPUs have two types of cores relevant to AI:

**CUDA Cores:** General-purpose parallel processors. They handle floating-point arithmetic (multiplication, addition) one operation at a time per core, but there are thousands of them.

**Tensor Cores:** Specialized matrix multiplication hardware. Instead of multiplying individual numbers, tensor cores multiply entire small matrices (4×4) in a single operation. This is dramatically faster for the matrix multiplications that dominate neural network computation.

```
CUDA Core (one operation per cycle):
  a × b + c = result

Tensor Core (matrix operation per cycle):
  ┌         ┐   ┌         ┐   ┌         ┐   ┌         ┐
  │ a b c d │   │ e f g h │   │ i j k l │   │ Result  │
  │ . . . . │ × │ . . . . │ + │ . . . . │ = │ Matrix  │
  │ . . . . │   │ . . . . │   │ . . . . │   │         │
  │ . . . . │   │ . . . . │   │ . . . . │   │         │
  └         ┘   └         ┘   └         ┘   └         ┘
     4×4     ×      4×4     +      4×4     =    4×4
  = 128 multiply-add operations in ONE cycle!
```

**GPU Comparison for AI Inference:**

| GPU | VRAM | CUDA Cores | Tensor Cores | Memory BW | FP16 TFLOPS | ~Price (2026) |
|-----|------|-----------|--------------|-----------|-------------|---------------|
| RTX 3060 | 12 GB | 3,584 | 112 | 360 GB/s | 25.3 | ~$250 |
| RTX 4060 Ti 16GB | 16 GB | 4,352 | 136 | 288 GB/s | 22.1 | ~$400 |
| RTX 4070 | 12 GB | 5,888 | 184 | 504 GB/s | 29.2 | ~$500 |
| RTX 4070 Ti Super | 16 GB | 8,448 | 264 | 672 GB/s | 44.1 | ~$700 |
| RTX 4080 Super | 16 GB | 10,240 | 320 | 736 GB/s | 52.4 | ~$900 |
| RTX 4090 | 24 GB | 16,384 | 512 | 1,008 GB/s | 82.6 | ~$1,800 |
| RTX 5070 Ti | 16 GB | 8,960 | 280 | 896 GB/s | ~48 | ~$750 |
| RTX 5080 | 16 GB | 10,752 | 336 | 960 GB/s | ~57 | ~$1,000 |
| RTX 5090 | 32 GB | 21,760 | 680 | 1,792 GB/s | ~104 | ~$2,000 |
| A100 | 40/80 GB | 6,912 | 432 | 2,039 GB/s | 77.9 | ~$8,000+ |
| H100 | 80 GB | 14,592 | 456 | 3,350 GB/s | 267.6 | ~$25,000+ |

**Key Insight:** For inference (as opposed to training), **memory bandwidth** is usually the bottleneck, not compute. This is because the model weights need to be read from VRAM for every token generated. The RTX 4090 with 1,008 GB/s bandwidth often generates tokens faster than cards with more CUDA cores but less bandwidth.

### VRAM: The Critical Bottleneck

**VRAM (Video RAM)** is the most important specification for local AI. The model weights must fit in VRAM for GPU inference. If they don't fit, you have three options: use a smaller model, use more aggressive quantization, or offload to system RAM (much slower).

#### How to Calculate VRAM Needs

The formula is straightforward:

```
VRAM needed ≈ (Number of Parameters × Bytes per Parameter) + KV Cache + Overhead

Where bytes per parameter depends on precision:
  FP32  = 4 bytes per parameter
  FP16  = 2 bytes per parameter
  INT8  = 1 byte per parameter
  INT4  = 0.5 bytes per parameter
```

**Worked Example: Llama 3.1 8B**

```
FP32: 8B × 4 bytes = 32 GB  ← Won't fit on any consumer GPU!
FP16: 8B × 2 bytes = 16 GB  ← Fits on RTX 4080/4090/5090
INT8: 8B × 1 byte  =  8 GB  ← Fits on RTX 4070/4060 Ti 16GB
INT4: 8B × 0.5 bytes = 4 GB ← Fits on almost anything

Plus KV Cache (depends on context length):
  At 4K context: ~0.5 GB additional
  At 32K context: ~2-4 GB additional
  At 128K context: ~8-16 GB additional

Plus overhead (framework, CUDA kernels): ~0.5-1 GB
```

#### VRAM Requirements Table

| Model | FP16 | INT8 (Q8) | INT4 (Q4) | Min. GPU (Q4) |
|-------|------|-----------|-----------|---------------|
| Phi-4-mini (3.8B) | 7.6 GB | 3.8 GB | ~2.5 GB | Any 4GB+ GPU |
| Gemma 3 4B | 8 GB | 4 GB | ~2.8 GB | Any 4GB+ GPU |
| Llama 3.1 8B | 16 GB | 8 GB | ~5 GB | RTX 3060 12GB |
| Qwen 3 8B | 16 GB | 8 GB | ~5 GB | RTX 3060 12GB |
| Gemma 3 12B | 24 GB | 12 GB | ~7 GB | RTX 3060 12GB |
| Phi-4 (14B) | 28 GB | 14 GB | ~8 GB | RTX 4060 Ti 16GB |
| Qwen 3 14B | 28 GB | 14 GB | ~8 GB | RTX 4060 Ti 16GB |
| Gemma 3 27B | 54 GB | 27 GB | ~16 GB | RTX 4080 Super 16GB |
| Qwen 3 32B | 64 GB | 32 GB | ~19 GB | RTX 4090 24GB |
| QwQ 32B | 64 GB | 32 GB | ~19 GB | RTX 4090 24GB |
| Llama 3.3 70B | 140 GB | 70 GB | ~40 GB | 2× RTX 4090 or A100 80GB |
| Qwen 3 235B-A22B (MoE) | ~470 GB | ~235 GB | ~140 GB | Multi-GPU server |
| DeepSeek-R1 (671B MoE) | ~1.3 TB | ~671 GB | ~400 GB | 8× A100 80GB |
| Llama 3.1 405B | ~810 GB | ~405 GB | ~240 GB | Multi-GPU server |

> **MoE Note:** For MoE models, you need VRAM for ALL parameters (all experts), not just the active parameters, because any expert might be needed. However, only active parameters affect compute speed. So DeepSeek-R1 needs ~400 GB at Q4 but runs at the speed of a 37B model.

### Apple Silicon: The Local AI Sweet Spot

Apple's M-series chips have a unique advantage for local AI: **unified memory architecture (UMA)**. Unlike discrete GPUs that have separate VRAM, Apple Silicon shares a single pool of memory between CPU and GPU.

```
Traditional PC:
┌─────────────┐         ┌──────────────┐
│   CPU       │         │    GPU       │
│   System    │ ◄──────►│    VRAM      │
│   RAM       │  PCIe   │   (12-24GB)  │
│  (32-128GB) │  Bus    │   Separate   │
└─────────────┘         └──────────────┘
    ↑ Model overflow         ↑ Model here
      (SLOW access)            (FAST access)

Apple Silicon:
┌─────────────────────────────────────┐
│        Unified Memory (UMA)         │
│     (16GB to 192GB in one pool)     │
│                                     │
│   ┌──────┐    ┌──────┐              │
│   │ CPU  │    │ GPU  │              │
│   │ cores│    │ cores│              │
│   └──────┘    └──────┘              │
│                                     │
│   Both access the SAME memory       │
│   at FULL bandwidth                 │
│   (200-546 GB/s depending on chip)  │
└─────────────────────────────────────┘
```

**Why This Matters:**
- A MacBook Pro M4 Max with 128GB unified memory can run models that would require multiple expensive GPUs on a PC
- Memory bandwidth is high (up to 546 GB/s on M4 Max)  crucial for inference speed
- No "VRAM vs RAM" split  all memory is available to the GPU
- Power efficient  ideal for laptop use

**Apple Silicon for AI:**

| Chip     | Max Memory | Memory BW  | GPU Cores | Best Model (Q4)          |
| -------- | ---------- | ---------- | --------- | ------------------------ |
| M2       | 24 GB      | 100 GB/s   | 10        | Llama 3.1 8B, Qwen 3 14B |
| M2 Pro   | 32 GB      | 200 GB/s   | 19        | Qwen 3 32B (tight)       |
| M2 Max   | 96 GB      | 400 GB/s   | 38        | Llama 3.3 70B            |
| M2 Ultra | 192 GB     | 800 GB/s   | 76        | DeepSeek-V3 (Q2-Q3)      |
| M3       | 24 GB      | 100 GB/s   | 10        | Llama 3.1 8B, Qwen 3 14B |
| M3 Pro   | 36 GB      | 150 GB/s   | 18        | Qwen 3 32B               |
| M3 Max   | 128 GB     | 400 GB/s   | 40        | Llama 3.3 70B            |
| M3 Ultra | 192 GB     | 800 GB/s   | 80        | DeepSeek-R1 (Q3)         |
| M4       | 32 GB      | 120 GB/s   | 10        | Qwen 3 14B               |
| M4 Pro   | 48 GB      | 273 GB/s   | 20        | Llama 3.3 70B (Q3)       |
| M4 Max   | 128 GB     | 546 GB/s   | 40        | Llama 3.3 70B easily     |
| M4 Ultra | 192+ GB    | 1092+ GB/s | 80        | DeepSeek-R1 (Q4)         |

### NPUs and AI Accelerators

**NPU (Neural Processing Unit)** is a specialized chip designed exclusively for neural network operations. Unlike GPUs (which can do graphics, physics, video encoding, etc.), NPUs are laser-focused on the matrix multiplications and activations that neural networks need.

**Current NPU landscape:**

| NPU | Found In | TOPS* | Best For |
|-----|----------|-------|----------|
| Apple Neural Engine | M-series, A-series | 38 (M4) | On-device inference (iOS/macOS) |
| Intel NPU (Meteor Lake+) | Intel Core Ultra | 10-13 | Windows Copilot+ features |
| Qualcomm Hexagon | Snapdragon X Elite | 45 | Windows ARM laptops |
| AMD XDNA 2 | Ryzen AI 300+ | 50 | Windows Copilot+ |
| Google Tensor (TPU) | Pixel phones | ~10 | On-device Android AI |
| Samsung NPU | Exynos chips | ~10 | On-device Galaxy AI |

*\*TOPS = Trillions of Operations Per Second (INT8)*

**Current Reality (2026):** NPUs are still maturing for large language model inference. They're great for small, fixed-function tasks (image processing, voice transcription, background ML features) but most LLM inference still runs better on GPU or CPU. This is changing rapidly as software catches up to hardware.

---

## Model Formats and Optimization

### Quantization Explained

**Quantization** is the process of reducing the precision of a model's weights (numbers) to make it smaller and faster. Think of it like the difference between measuring something with a ruler marked in millimeters versus one marked in centimeters  less precision, but much easier to work with.

#### The Precision Spectrum

```
FP32 (Full Precision)  32 bits per number
████████████████████████████████
Very precise: 3.14159265358979...
Size: 4 bytes per parameter
A 7B model = 28 GB

FP16 (Half Precision)  16 bits per number
████████████████
Good precision: 3.14159...
Size: 2 bytes per parameter
A 7B model = 14 GB

BF16 (Brain Float 16)  16 bits per number
████████████████
Same range as FP32, less precision: 3.141...
Size: 2 bytes per parameter
A 7B model = 14 GB
(Preferred for training, same range as FP32)

INT8 (8-bit Integer)  8 bits per number
████████
OK precision: 3.14...
Size: 1 byte per parameter
A 7B model = 7 GB

INT4 (4-bit Integer)  4 bits per number
████
Low precision: 3.1 or 3.2
Size: 0.5 bytes per parameter
A 7B model = 3.5 GB
(+ quantization metadata overhead → ~4-4.5 GB actual)

INT2 (2-bit Integer)  2 bits per number
██
Very low: 3 or 4
Size: 0.25 bytes per parameter
A 7B model = 1.75 GB
(Significant quality loss  experimental)
```

#### Quality vs. Size Tradeoff

The key question: **how much quality do you lose with quantization?**

```
Quality Retention (approximate, varies by model):

FP16:  ████████████████████ 100% (baseline)
Q8:    ███████████████████░ ~99% (nearly lossless)
Q6_K:  ██████████████████░░ ~97%
Q5_K_M:████████████████░░░░ ~95%
Q4_K_M:██████████████░░░░░░ ~92% (sweet spot for most users)
Q4_0:  █████████████░░░░░░░ ~88%
Q3_K_M:███████████░░░░░░░░░ ~85%
Q2_K:  █████████░░░░░░░░░░░ ~75% (noticeable degradation)
IQ2_XXS:███████░░░░░░░░░░░░ ~65% (emergency mode)
```

**The Sweet Spot:** For most users, **Q4_K_M** (4-bit quantization with K-quant mixed precision) offers the best balance. You lose ~8% of quality but reduce model size by 75% compared to FP16. This often means the difference between "runs on your GPU" and "doesn't fit."

**Quantization Naming Conventions (GGUF):**

| Name | Bits | Method | Quality | Notes |
|------|------|--------|---------|-------|
| Q2_K | 2-bit | K-quant | Poor | Emergency/research only |
| IQ3_XXS | 3-bit | iMatrix | Fair | Very small, some models tolerate it |
| Q3_K_S | 3-bit | K-quant Small | Fair | Smallest usable for most models |
| Q3_K_M | 3-bit | K-quant Medium | OK | Better than Q3_K_S |
| Q4_0 | 4-bit | Basic | Good | Fast but older method |
| Q4_K_S | 4-bit | K-quant Small | Good | Efficient |
| Q4_K_M | 4-bit | K-quant Medium | Very Good | ⭐ Most popular choice |
| Q5_K_S | 5-bit | K-quant Small | Very Good | Quality-focused |
| Q5_K_M | 5-bit | K-quant Medium | Excellent | If you have the VRAM |
| Q6_K | 6-bit | K-quant | Near-FP16 | Best quality quantization |
| Q8_0 | 8-bit | Basic | ~Lossless | Large but nearly no quality loss |
| F16 | 16-bit | None | Baseline | Full precision, largest |

### GGUF Format

**GGUF (GPT-Generated Unified Format)** is the most popular format for local AI inference. Created by the llama.cpp project, it has become the de facto standard for running models locally.

#### What Is GGUF?

GGUF is a single-file model format that contains:
1. **Model weights** (quantized to your chosen precision)
2. **Model metadata** (architecture info, vocabulary, hyperparameters)
3. **Tokenizer data** (how text is converted to/from tokens)

Everything in one file  no separate config files, tokenizer files, or sharded weight files.

```
Traditional Format (Hugging Face):
  model/
  ├── config.json           ← Architecture config
  ├── tokenizer.json        ← Tokenizer
  ├── tokenizer_config.json ← Tokenizer config
  ├── special_tokens_map.json
  ├── model-00001-of-00004.safetensors  ← Weights (shard 1)
  ├── model-00002-of-00004.safetensors  ← Weights (shard 2)
  ├── model-00003-of-00004.safetensors  ← Weights (shard 3)
  └── model-00004-of-00004.safetensors  ← Weights (shard 4)

GGUF Format:
  model-Q4_K_M.gguf  ← EVERYTHING in one file
```

#### Why GGUF Is Popular

1. **One file = one model.** Easy to manage, share, and deploy.
2. **Multiple quantization levels.** Download exactly the quality/size you want.
3. **CPU + GPU hybrid inference.** Can split layers between GPU and CPU RAM.
4. **Wide compatibility.** Works with llama.cpp, Ollama, LM Studio, GPT4All, and more.
5. **mmap support.** Models can be memory-mapped, enabling fast loading and efficient memory use.
6. **Community standard.** Thousands of models available on Hugging Face in GGUF format.

**Where to find GGUF models:** Search for your desired model + "GGUF" on Hugging Face. Community quantizer accounts like **bartowski**, **mradermacher**, and **unsloth** regularly publish GGUF versions of new models within hours of release.

### ONNX (Open Neural Network Exchange)

**ONNX** is a cross-platform model format created by Microsoft and Facebook. While GGUF is the standard for llama.cpp-based tools, ONNX is the standard for broader ML deployment.

```
ONNX Ecosystem:

  PyTorch Model ──→ Export to ONNX ──→ Run on:
  TensorFlow Model ──→                  ├── CPU (any platform)
  JAX Model ──────────→                  ├── NVIDIA GPU (CUDA)
                                         ├── AMD GPU (ROCm)
                                         ├── Intel GPU (OpenVINO)
                                         ├── Mobile (Android/iOS)
                                         ├── Web (ONNX.js)
                                         └── Edge devices (NPUs)
```

**Key advantage:** Write once, run anywhere. ONNX models can run on any hardware that has an ONNX Runtime implementation  Windows, Linux, macOS, Android, iOS, web browsers.

**ONNX Runtime** (by Microsoft) is the execution engine that runs ONNX models. It automatically selects the best execution provider for your hardware (CUDA for NVIDIA, DirectML for Windows, CoreML for Apple, etc.).

### TensorRT (NVIDIA Optimization)

**TensorRT** is NVIDIA's proprietary inference optimization toolkit. It takes a trained model and optimizes it specifically for NVIDIA GPUs through:

1. **Layer fusion:** Combining multiple operations into single GPU kernels
2. **Precision calibration:** INT8/FP16 quantization optimized for NVIDIA tensor cores
3. **Kernel auto-tuning:** Testing different GPU kernel implementations and choosing the fastest
4. **Dynamic memory optimization:** Minimizing GPU memory allocation overhead

```
Model (FP32) ──→ TensorRT Optimizer ──→ Optimized Engine (.trt)
                  │
                  ├── Fuse layers
                  ├── Calibrate to INT8/FP16
                  ├── Auto-tune kernels
                  └── Optimize memory

Result: 2-5x faster inference on NVIDIA GPUs
```

**When to use TensorRT:**
- You're deploying on NVIDIA hardware in production
- You need maximum throughput (tokens/second)
- You're willing to spend time on optimization (it's not plug-and-play)
- You don't need to switch between GPU vendors

**TensorRT-LLM** is NVIDIA's specialized framework for LLM inference, built on TensorRT but with LLM-specific optimizations like KV cache management, in-flight batching, and tensor parallelism.

### ExLlamaV2

**ExLlamaV2** is a high-performance inference library specifically optimized for running quantized LLMs on NVIDIA GPUs. It achieves some of the fastest inference speeds for consumer GPUs.

**Key features:**
- **EXL2 quantization format:** Variable bits-per-weight within a single model  important layers get more precision, less important layers get fewer bits
- **Very fast:** Often the fastest option for single-GPU consumer cards
- **Flash Attention integration:** Memory-efficient attention computation
- **Speculative decoding support:** Draft model generates candidates, main model verifies (speeds up generation)

**EXL2 vs GGUF:**

| Feature | EXL2 | GGUF |
|---------|------|------|
| Hardware | NVIDIA GPU only | CPU + GPU (any) |
| Speed (NVIDIA) | Fastest | Fast |
| Quantization | Variable BPW per layer | Fixed scheme (Q4_K_M, etc.) |
| Flexibility | GPU only | CPU, GPU, hybrid |
| Community | Smaller | Largest |
| Typical use | Maximum speed on NVIDIA | Broad compatibility |

### AWQ (Activation-aware Weight Quantization)

AWQ is a quantization method that preserves model quality better than naive quantization by protecting the most "important" weights:

1. Analyze which weights have the most impact on model outputs (using calibration data)
2. Scale those important weights up before quantization (so they lose less precision)
3. Quantize all weights to INT4
4. During inference, reverse the scaling

Result: INT4 quantization with quality closer to FP16 than standard INT4.

### GPTQ (GPT Quantization)

GPTQ is an older but still popular quantization method:
- Uses a one-shot calibration process with sample data
- Layer-by-layer quantization with error compensation
- Good quality at INT4/INT3
- Widely supported but being superseded by AWQ and EXL2 for new models

**Format Comparison Summary:**

| Format | Best For | Quantization | Hardware | Speed |
|--------|----------|-------------|----------|-------|
| GGUF | General use, CPU+GPU | K-quants (Q4_K_M, etc.) | Any | Good |
| EXL2 | Max speed on NVIDIA | Variable BPW | NVIDIA GPU | Fastest |
| AWQ | Quality-sensitive INT4 | AWQ INT4 | NVIDIA GPU | Fast |
| GPTQ | Legacy compatibility | GPTQ INT4/INT3 | NVIDIA GPU | Good |
| ONNX | Cross-platform deploy | Various | Any | Good |
| TensorRT | NVIDIA production | INT8/FP16 optimized | NVIDIA GPU | Very Fast |

---

## Inference Engines

### llama.cpp

**llama.cpp** is the project that started the local AI revolution. Created by Georgi Gerganov in March 2023, it's a C/C++ implementation of LLM inference that runs on virtually any hardware.

**Key features:**
- Pure C/C++ with no dependencies (no Python, no PyTorch)
- Runs on CPU, NVIDIA GPU, AMD GPU, Apple Metal, Vulkan, SYCL
- Created the GGUF format
- Supports quantization from Q2 to Q8 and FP16
- GPU offloading  put some layers on GPU, rest on CPU
- Extremely active open-source community
- Supports most popular model architectures

**Usage Example:**
```bash
# Download a model
wget https://huggingface.co/bartowski/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf

# Run interactive chat
./llama-cli -m Qwen3-8B-Q4_K_M.gguf \
  -n 512 \           # max tokens to generate
  -ngl 99 \          # offload all layers to GPU
  -c 4096 \          # context size
  --interactive       # interactive chat mode

# Run as API server (OpenAI-compatible)
./llama-server -m Qwen3-8B-Q4_K_M.gguf \
  --port 8080 \
  -ngl 99 \
  -c 4096
```

### Ollama

**Ollama** is the easiest way to run AI models locally. Think of it as the "Docker for AI models"  it handles downloading, managing, and running models with simple commands.

**Installation:**
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

**Basic Usage:**
```bash
# Download and run a model (one command!)
ollama run qwen3:8b

# List available models
ollama list

# Pull a specific model without running it
ollama pull llama3.1:8b

# Run with specific parameters
ollama run qwen3:32b --verbose

# Serve as API (starts automatically, OpenAI-compatible)
# Default: http://localhost:11434
ollama serve
```

**Ollama Model Library (Popular Models):**

| Command                       | Model              | Size (Q4) | Min. RAM |
| ----------------------------- | ------------------ | --------- | -------- |
| `ollama run qwen3:0.6b`       | Qwen 3 0.6B        | ~0.5 GB   | 2 GB     |
| `ollama run gemma3:1b`        | Gemma 3 1B         | ~0.8 GB   | 2 GB     |
| `ollama run phi4-mini`        | Phi-4-mini 3.8B    | ~2.5 GB   | 4 GB     |
| `ollama run gemma3:4b`        | Gemma 3 4B         | ~2.8 GB   | 4 GB     |
| `ollama run llama3.1:8b`      | Llama 3.1 8B       | ~4.7 GB   | 8 GB     |
| `ollama run qwen3:8b`         | Qwen 3 8B          | ~5 GB     | 8 GB     |
| `ollama run gemma3:12b`       | Gemma 3 12B        | ~7.5 GB   | 12 GB    |
| `ollama run qwen3:14b`        | Qwen 3 14B         | ~8.5 GB   | 12 GB    |
| `ollama run qwen3:32b`        | Qwen 3 32B         | ~19 GB    | 24 GB    |
| `ollama run qwq`              | QwQ 32B            | ~19 GB    | 24 GB    |
| `ollama run deepseek-r1:32b`  | DS-R1-Distill 32B  | ~19 GB    | 24 GB    |
| `ollama run llama3.3:70b`     | Llama 3.3 70B      | ~40 GB    | 48 GB    |
| `ollama run deepseek-r1:671b` | DeepSeek R1 (full) | ~400 GB   | 512 GB   |

**Ollama Modelfile (Customization):**
```dockerfile
# Create a custom model configuration
FROM qwen3:8b

SYSTEM """You are a helpful coding assistant specialized in Python.
Always provide working code examples with comments."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
```

```bash
# Create and run custom model
ollama create my-code-helper -f Modelfile
ollama run my-code-helper
```

**Using Ollama's API:**
```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'qwen3:8b',
    'prompt': 'Explain quantum computing in simple terms',
    'stream': False
})

print(response.json()['response'])
```

Ollama also exposes an **OpenAI-compatible API** at `http://localhost:11434/v1/`, meaning any tool built for OpenAI's API can work with local models by simply changing the base URL.

### LM Studio

**LM Studio** is a desktop application (GUI) for running local AI models. It's the most user-friendly option  no command line needed.

**Key Features:**
- Beautiful GUI for chatting with models
- Built-in model browser and downloader (searches Hugging Face)
- Drag-and-drop model loading
- OpenAI-compatible API server
- Side-by-side model comparison
- System resource monitoring (VRAM, RAM, CPU usage)
- Supports GGUF, and some other formats
- Available for Windows, macOS, Linux

```
┌─────────────────────────────────────────────────────────┐
│  LM Studio                                     [─][□][×]│
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│  Model List  │  Chat Interface                         │
│              │                                          │
│  ◉ Qwen3-8B │  You: What is machine learning?         │
│  ○ Llama-8B  │                                          │
│  ○ Gemma-12B │  AI: Machine learning is a subset of    │
│  ○ Phi4-mini │  artificial intelligence that enables    │
│              │  systems to learn from data...           │
│              │                                          │
│  ┌────────┐  │  [────────────────────────] [Send]      │
│  │Download │  │                                          │
│  │ More    │  │  VRAM: ████░░░░ 5.2/12GB               │
│  └────────┘  │  Speed: 32 tok/s                        │
└──────────────┴──────────────────────────────────────────┘
```

**When to use LM Studio:**
- You're new to local AI and want a simple start
- You want to quickly try different models
- You need a local API server with a nice interface
- You prefer GUI over command line

### vLLM

**vLLM** (Virtual LLM) is a high-throughput inference engine designed for **serving** LLMs  when you need to handle many concurrent requests efficiently.

**Key Innovation  PagedAttention:**

The KV cache (which stores attention computation results for all previous tokens) is the biggest memory consumer during inference. Traditional implementations allocate a fixed block of memory for each request, wasting space when requests are shorter than the maximum.

PagedAttention (inspired by OS virtual memory) solves this:

```
Traditional KV Cache:
Request 1: [████████░░░░░░░░]  ← Allocated for max length, mostly wasted
Request 2: [██████░░░░░░░░░░]  ← Same problem
Request 3: [████████████░░░░]  ← Only this one is nearly full
Total waste: ~50% of VRAM

PagedAttention:
Request 1: [████][████]          ← Pages allocated as needed
Request 2: [████][██]            ← No waste
Request 3: [████][████][████]    ← Grows dynamically
Free pages: [    ][    ][    ]   ← Available for new requests
Total waste: <5% of VRAM
```

**Result:** vLLM can serve 2-4x more concurrent requests than naive implementations with the same hardware.

**Usage:**
```bash
# Install
pip install vllm

# Start server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9

# It automatically creates an OpenAI-compatible API at port 8000
```

**When to use vLLM:**
- You're serving a model to multiple users
- Throughput (total tokens/second across all users) matters more than single-user latency
- You're deploying in production
- You need OpenAI-compatible API for your stack

### LocalAI

**LocalAI** is a self-hosted, OpenAI-compatible API server that supports multiple model types  not just LLMs but also image generation, audio transcription, text-to-speech, and embeddings.

**Key Features:**
- Drop-in replacement for OpenAI's API (same endpoints, same format)
- Supports GGUF, Diffusers (image), Whisper (audio), and more
- Docker-based deployment
- GPU and CPU inference
- Function calling support

**Usage:**
```bash
# Run with Docker
docker run -p 8080:8080 \
  -v /path/to/models:/models \
  localai/localai:latest

# Use exactly like OpenAI's API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**When to use LocalAI:**
- You want a unified local API for text, images, and audio
- You're migrating from OpenAI's API to self-hosted
- You need Docker-based deployment
- You want multi-model serving from one endpoint

### Inference Engine Comparison

| Engine | Best For | Ease of Use | Speed | Multi-user | GPU Required |
|--------|----------|------------|-------|------------|-------------|
| **Ollama** | Personal use, getting started | ⭐⭐⭐⭐⭐ | Good | Basic | No (CPU ok) |
| **LM Studio** | GUI users, experimentation | ⭐⭐⭐⭐⭐ | Good | Via API | No (CPU ok) |
| **llama.cpp** | Custom setups, max control | ⭐⭐⭐ | Very Good | Basic | No (CPU ok) |
| **vLLM** | Production serving, high throughput | ⭐⭐⭐ | Excellent | ⭐⭐⭐⭐⭐ | Yes |
| **ExLlamaV2** | Maximum single-GPU speed | ⭐⭐ | Fastest | No | Yes (NVIDIA) |
| **LocalAI** | OpenAI API replacement | ⭐⭐⭐⭐ | Good | Good | No (CPU ok) |
| **TensorRT-LLM** | NVIDIA production deployment | ⭐⭐ | Fastest (NVIDIA) | ⭐⭐⭐⭐⭐ | Yes (NVIDIA) |

---

## Hardware Configurations: What Can You Run?

### Configuration 1: Laptop (16GB RAM, Integrated GPU)

**Hardware:** Intel/AMD laptop with 16GB DDR5, Intel Iris Xe or AMD RDNA3 integrated graphics
**Inference method:** Primarily CPU (integrated GPUs have minimal VRAM sharing system RAM)

**What you can run:**

| Model | Quantization | Size | Speed | Usable? |
|-------|-------------|------|-------|---------|
| Qwen 3 0.6B | Q8_0 | ~0.7 GB | ~30-40 tok/s | ✅ Fast, but limited |
| Gemma 3 1B | Q8_0 | ~1.2 GB | ~20-30 tok/s | ✅ Good for simple tasks |
| Phi-4-mini (3.8B) | Q4_K_M | ~2.5 GB | ~8-15 tok/s | ✅ Best quality at this tier |
| Gemma 3 4B | Q4_K_M | ~2.8 GB | ~7-12 tok/s | ✅ Decent general use |
| Llama 3.1 8B | Q4_K_M | ~4.7 GB | ~4-8 tok/s | ⚠️ Usable but slow |
| Qwen 3 8B | Q4_K_M | ~5 GB | ~4-7 tok/s | ⚠️ Usable but slow |
| Qwen 3 14B | Q3_K_M | ~6.5 GB | ~2-4 tok/s | ⚠️ Very slow |
| Qwen 3 14B | Q4_K_M | ~8.5 GB | ~1-3 tok/s | ❌ Too slow, tight on RAM |

**Recommendations:**
- **Daily driver:** Phi-4-mini Q4_K_M  best quality you'll get at comfortable speeds
- **Quick tasks:** Gemma 3 1B Q8  fast for simple questions, summarization
- **Stretch goal:** Qwen 3 8B Q4_K_M  slower but notably smarter
- **Tool:** Ollama or LM Studio

**Tips for laptop inference:**
- Close other applications to free RAM
- Use shorter context lengths (2K-4K) to save memory
- Consider battery impact  CPU inference is power-hungry
- Plug in the laptop (CPU inference drains battery fast)

### Configuration 2: Gaming PC (RTX 4070, 12GB VRAM, 32GB RAM)

**Hardware:** RTX 4070 (12GB VRAM, 504 GB/s bandwidth), 32GB DDR5 system RAM
**Inference method:** GPU primary with CPU offloading for larger models

**What you can run:**

| Model | Quantization | Size | Fits in VRAM? | Speed | Usable? |
|-------|-------------|------|--------------|-------|---------|
| Phi-4-mini (3.8B) | Q4_K_M | ~2.5 GB | ✅ Fully | ~80-100 tok/s | ✅ Lightning fast |
| Gemma 3 4B | Q4_K_M | ~2.8 GB | ✅ Fully | ~70-90 tok/s | ✅ Excellent |
| Llama 3.1 8B | Q4_K_M | ~4.7 GB | ✅ Fully | ~50-65 tok/s | ✅ Great |
| Qwen 3 8B | Q4_K_M | ~5 GB | ✅ Fully | ~45-60 tok/s | ✅ Great |
| Gemma 3 12B | Q4_K_M | ~7.5 GB | ✅ Fully | ~30-40 tok/s | ✅ Very good |
| Qwen 3 14B | Q4_K_M | ~8.5 GB | ✅ Tight | ~25-35 tok/s | ✅ Good |
| Qwen 3 14B | Q5_K_M | ~10.5 GB | ⚠️ Tight | ~20-30 tok/s | ✅ Higher quality |
| Gemma 3 27B | Q4_K_M | ~16 GB | ❌ Partial offload | ~8-15 tok/s | ⚠️ Slow |
| Qwen 3 32B | Q3_K_M | ~14 GB | ❌ Partial offload | ~6-12 tok/s | ⚠️ Slow |
| Qwen 3 32B | Q4_K_M | ~19 GB | ❌ Heavy offload | ~3-7 tok/s | ⚠️ Very slow |

**Recommendations:**
- **Sweet spot:** Qwen 3 14B Q4_K_M  best quality that fits fully in VRAM
- **Fastest good model:** Qwen 3 8B or Llama 3.1 8B at Q4_K_M or Q5_K_M
- **Reasoning:** DeepSeek-R1-Distill-Qwen-14B Q4_K_M  great reasoning at this tier
- **Reach model:** Gemma 3 27B or Qwen 3 32B with partial CPU offload (slower but higher quality)
- **Tool:** Ollama for ease, llama.cpp for max control

**GPU Offloading Explained:**
```
Full GPU: All layers on GPU ─── Fastest
  Model: [GPU GPU GPU GPU GPU GPU GPU]

Partial Offload: Some layers on GPU, rest on CPU
  Model: [GPU GPU GPU GPU CPU CPU CPU]
  ↑ GPU layers run fast  ↑ CPU layers run slow
  Overall speed determined by slowest component

Full CPU: All layers on CPU ─── Slowest
  Model: [CPU CPU CPU CPU CPU CPU CPU]
```

With llama.cpp/Ollama, you control this with `--gpu-layers` (or `-ngl`):
```bash
# All layers on GPU (fastest, needs enough VRAM)
ollama run qwen3:14b  # Ollama auto-detects

# Or with llama.cpp, explicitly:
./llama-cli -m model.gguf -ngl 99    # All on GPU
./llama-cli -m model.gguf -ngl 20    # 20 layers on GPU, rest on CPU
./llama-cli -m model.gguf -ngl 0     # All on CPU
```

### Configuration 3: Workstation (RTX 4090, 24GB VRAM, 64GB RAM)

**Hardware:** RTX 4090 (24GB VRAM, 1,008 GB/s bandwidth), 64GB DDR5 system RAM
**Inference method:** GPU primary  this is the local AI sweet spot for enthusiasts

**What you can run:**

| Model | Quantization | Size | Fits in VRAM? | Speed | Usable? |
|-------|-------------|------|--------------|-------|---------|
| Qwen 3 8B | Q8_0 | ~8.5 GB | ✅ Fully | ~70-90 tok/s | ✅ Near-lossless quality |
| Qwen 3 14B | Q5_K_M | ~10.5 GB | ✅ Fully | ~40-55 tok/s | ✅ Excellent quality |
| Qwen 3 14B | Q8_0 | ~15 GB | ✅ Fully | ~30-40 tok/s | ✅ Maximum quality |
| Gemma 3 27B | Q4_K_M | ~16 GB | ✅ Fully | ~25-35 tok/s | ✅ Excellent |
| Gemma 3 27B | Q5_K_M | ~19.5 GB | ✅ Fully | ~20-30 tok/s | ✅ High quality |
| QwQ 32B | Q4_K_M | ~19 GB | ✅ Tight | ~20-28 tok/s | ✅ Strong reasoning |
| Qwen 3 32B | Q4_K_M | ~19 GB | ✅ Tight | ~20-28 tok/s | ✅ Great all-rounder |
| Llama 3.3 70B | Q3_K_M | ~33 GB | ❌ Partial offload | ~8-12 tok/s | ⚠️ Usable |
| Llama 3.3 70B | Q4_K_M | ~40 GB | ❌ Partial offload | ~5-10 tok/s | ⚠️ Slow but good |
| DS-R1-Distill-70B | Q3_K_M | ~33 GB | ❌ Partial offload | ~6-10 tok/s | ⚠️ Slow |

**Recommendations:**
- **Best overall:** Qwen 3 32B Q4_K_M  frontier-class quality, fits in VRAM
- **Best reasoning:** QwQ 32B Q4_K_M  dedicated reasoning model
- **Best quality-per-token:** Gemma 3 27B Q5_K_M  higher quant for better quality
- **Reach model:** Llama 3.3 70B Q3_K_M with CPU offload  premium quality, slower
- **Tool:** Ollama for daily use, ExLlamaV2 for maximum speed

**RTX 5090 (32GB) extends the sweet spot further:**
The RTX 5090's 32GB VRAM opens up Qwen 3 32B at Q5_K_M (~22GB) fully on GPU, and makes Llama 3.3 70B at Q3_K_M (~33GB) nearly fit (minimal CPU offload).

### Configuration 4: Server (2× A100 80GB)

**Hardware:** 2× NVIDIA A100 80GB (160GB total VRAM, 4,078 GB/s combined bandwidth), 512GB system RAM
**Inference method:** Full GPU with tensor parallelism across both GPUs

**What you can run:**

| Model | Quantization | Size | Speed | Notes |
|-------|-------------|------|-------|-------|
| Qwen 3 32B | FP16 | ~64 GB | ~80-100 tok/s | Full precision, maximum quality |
| Llama 3.3 70B | FP16 | ~140 GB | ~40-60 tok/s | Full precision across both GPUs |
| Llama 3.3 70B | Q8_0 | ~70 GB | ~50-70 tok/s | Near-lossless, single GPU |
| QwQ 32B | FP16 | ~64 GB | ~80-100 tok/s | Full precision reasoning |
| Qwen 3 235B-A22B | Q4_K_M | ~140 GB | ~30-50 tok/s | Frontier MoE model |
| DeepSeek-V3 (671B) | Q3_K_M | ~320 GB | N/A | Needs 4× A100 minimum |
| DeepSeek-R1 (671B) | Q3_K_M | ~320 GB | N/A | Needs 4× A100 minimum |
| Llama 3.1 405B | Q4_K_M | ~240 GB | N/A | Needs 4× A100 minimum |

**Multi-GPU Setups:**

| Configuration | Total VRAM | Can Run (Q4) | Use Case |
|--------------|-----------|-------------|----------|
| 2× RTX 4090 | 48 GB | 70B models comfortably | Enthusiast/prosumer |
| 2× RTX 5090 | 64 GB | 70B at Q5+, some MoE | High-end prosumer |
| 2× A100 80GB | 160 GB | 235B MoE, 70B at FP16 | Professional/research |
| 4× A100 80GB | 320 GB | DeepSeek-R1 at Q3-Q4 | Serious deployment |
| 8× A100 80GB | 640 GB | DeepSeek-R1 at Q8, 405B at Q4 | Enterprise |
| 8× H100 80GB | 640 GB | Anything, at high speed | Data center |

**Tensor Parallelism vs Pipeline Parallelism:**

When spreading a model across multiple GPUs, there are two main strategies:

```
Tensor Parallelism (split each layer across GPUs):
  GPU 1: [Half of Layer 1][Half of Layer 2][Half of Layer 3]...
  GPU 2: [Half of Layer 1][Half of Layer 2][Half of Layer 3]...
  → Both GPUs work on every layer simultaneously
  → Requires fast inter-GPU communication (NVLink)
  → Lower latency per token

Pipeline Parallelism (put different layers on different GPUs):
  GPU 1: [Layer 1][Layer 2][Layer 3]...[Layer 16]
  GPU 2: [Layer 17][Layer 18][Layer 19]...[Layer 32]
  → Each GPU handles different layers sequentially
  → Works with regular PCIe connections
  → Simpler to implement, higher latency
```

For consumer multi-GPU setups (PCIe-connected), pipeline parallelism is typical. For data center GPUs with NVLink, tensor parallelism gives better performance.

### Configuration 5: Apple Silicon MacBook

**Hardware:** MacBook Pro M4 Max (128GB unified memory, 546 GB/s bandwidth)
**Inference method:** Metal GPU acceleration, unified memory advantage

**What you can run:**

| Model | Quantization | Size | Speed | Notes |
|-------|-------------|------|-------|-------|
| Qwen 3 8B | Q4_K_M | ~5 GB | ~35-50 tok/s | Easy, fast |
| Qwen 3 14B | Q4_K_M | ~8.5 GB | ~25-35 tok/s | Comfortable |
| Qwen 3 32B | Q4_K_M | ~19 GB | ~18-25 tok/s | Great balance |
| Qwen 3 32B | Q8_0 | ~34 GB | ~12-18 tok/s | Higher quality |
| Llama 3.3 70B | Q4_K_M | ~40 GB | ~12-18 tok/s | Fits easily! |
| Llama 3.3 70B | Q5_K_M | ~50 GB | ~10-15 tok/s | High quality |
| Llama 3.3 70B | Q8_0 | ~70 GB | ~8-12 tok/s | Near-lossless |
| Qwen 3 235B-A22B | Q4_K_M | ~140 GB | ~5-8 tok/s | Frontier MoE! |

**The Apple Silicon Advantage:**
A MacBook Pro with 128GB unified memory can run 70B models at quality levels that would require a $1,800 GPU (RTX 4090) or multi-GPU setup on PC. The tradeoff is speed  Apple's GPU cores are slower per-core than NVIDIA's  but the massive memory pool means you can run models that simply won't fit on any single consumer GPU.

**This makes Apple Silicon the best option for:**
- Running large models (70B+) without a server
- Portable high-quality inference (on a laptop!)
- Models that exceed 24GB VRAM (PC GPU limit until RTX 5090)

---

## VRAM Requirements Reference Table

This comprehensive table shows what you need for popular models at different quantization levels:

| Model | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | FP16 |
|-------|--------|--------|--------|------|------|------|
| Qwen 3 0.6B | 0.4 GB | 0.5 GB | 0.5 GB | 0.6 GB | 0.7 GB | 1.2 GB |
| Gemma 3 1B | 0.6 GB | 0.8 GB | 0.9 GB | 1.0 GB | 1.2 GB | 2.0 GB |
| Phi-4-mini 3.8B | 1.9 GB | 2.5 GB | 2.8 GB | 3.2 GB | 3.8 GB | 7.6 GB |
| Gemma 3 4B | 2.0 GB | 2.8 GB | 3.2 GB | 3.6 GB | 4.3 GB | 8.0 GB |
| Llama 3.1 8B | 3.9 GB | 4.7 GB | 5.5 GB | 6.3 GB | 8.0 GB | 16 GB |
| Qwen 3 8B | 4.1 GB | 5.0 GB | 5.8 GB | 6.6 GB | 8.5 GB | 16 GB |
| Gemma 3 12B | 5.8 GB | 7.5 GB | 8.6 GB | 9.8 GB | 12.5 GB | 24 GB |
| Qwen 3 14B | 6.5 GB | 8.5 GB | 10.0 GB | 11.2 GB | 14.5 GB | 28 GB |
| Mistral Small 3.1 24B | 11 GB | 14 GB | 16 GB | 19 GB | 24 GB | 48 GB |
| Gemma 3 27B | 12.5 GB | 16 GB | 18.5 GB | 21 GB | 27 GB | 54 GB |
| Qwen 3 32B | 14.5 GB | 19 GB | 22 GB | 25 GB | 32 GB | 64 GB |
| QwQ 32B | 14.5 GB | 19 GB | 22 GB | 25 GB | 32 GB | 64 GB |
| Llama 3.3 70B | 33 GB | 40 GB | 48 GB | 55 GB | 70 GB | 140 GB |
| Llama 4 Scout 109B | 50 GB | 65 GB | 75 GB | 85 GB | 109 GB | 218 GB |
| Qwen 3 235B-A22B | 110 GB | 140 GB | 165 GB | 190 GB | 235 GB | 470 GB |
| Llama 4 Maverick 400B | 190 GB | 240 GB | 280 GB | 320 GB | 400 GB | 800 GB |
| DeepSeek-R1 671B | 320 GB | 400 GB | 470 GB | 540 GB | 671 GB | 1,342 GB |

> **Note:** These are approximate sizes for the model weights only. Add 0.5-2 GB for runtime overhead and KV cache (more for longer context lengths). Actual sizes vary slightly by quantization implementation and metadata.

---

## Practical Guide: Getting Started with Local AI

### Step-by-Step: Your First Local Model

**Option A: Ollama (Recommended for Beginners)**

```bash
# 1. Install Ollama
#    Windows: Download from https://ollama.com/download
#    macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh

# 2. Run your first model
ollama run phi4-mini
# Wait for download (~2.5 GB), then start chatting!

# 3. Try a larger model
ollama run qwen3:8b
# Better quality, ~5 GB download

# 4. Try a reasoning model
ollama run qwq
# Extended thinking, ~19 GB download (needs 24GB+ RAM/VRAM)
```

**Option B: LM Studio (Recommended for GUI Users)**

1. Download LM Studio from https://lmstudio.ai
2. Open the app → go to "Discover" tab
3. Search for "Phi-4-mini GGUF" or "Qwen 3 8B GGUF"
4. Download a Q4_K_M quantization
5. Go to "Chat" tab → select the downloaded model
6. Start chatting!

### Choosing the Right Model for Your Hardware

```
How much VRAM/Unified Memory do you have?

  4 GB or less:
    → Phi-4-mini Q4 or Gemma 3 1B Q8 (CPU inference also works)
    
  6-8 GB:
    → Qwen 3 8B Q4 or Llama 3.1 8B Q4
    
  10-12 GB:
    → Qwen 3 14B Q4 or Gemma 3 12B Q4
    
  16 GB:
    → Gemma 3 27B Q4 or Qwen 3 14B Q5/Q6
    
  24 GB:
    → Qwen 3 32B Q4 or QwQ 32B Q4
    
  32 GB:
    → Qwen 3 32B Q6/Q8 or Llama 3.3 70B Q3 (tight)
    
  48+ GB:
    → Llama 3.3 70B Q4-Q5
    
  80+ GB:
    → Llama 3.3 70B Q8/FP16 or Qwen 3 235B MoE Q3
    
  160+ GB:
    → DeepSeek-R1 distilled at high quant or Qwen 3 235B at Q4+
```

### Performance Optimization Tips

1. **Use the right quantization:** Q4_K_M is the sweet spot for most users. Don't go lower unless you have to.

2. **Keep the model fully in VRAM:** Partial CPU offload is 5-10x slower per offloaded layer. Better to use a smaller model that fits entirely.

3. **Reduce context length:** If you don't need 128K tokens, set context to 4K-8K. This saves significant memory for the KV cache.

4. **Use Flash Attention:** Most modern inference engines enable this by default. It reduces memory usage for attention computation.

5. **Close other GPU applications:** Games, video editing, multiple browser tabs with hardware acceleration  these compete for VRAM.

6. **Monitor your resources:** Use `nvidia-smi` (NVIDIA), Task Manager (Windows), or Activity Monitor (macOS) to track VRAM and RAM usage.

7. **Consider batch size:** If using vLLM for serving, tune batch size to balance throughput and latency.

8. **Update your inference engine:** llama.cpp, Ollama, and others are updated frequently with performance improvements. Keep them current.

---

## The Future of Local AI

### Trends to Watch

1. **Models keep getting more efficient:** Every generation of models (Phi, Gemma, Qwen) produces better quality at smaller sizes. Today's 8B model matches 2024's 70B in many tasks.

2. **Hardware is catching up:** The RTX 5090 (32GB), Mac M4 Ultra (192+ GB), and upcoming AI-specific consumer chips are making local inference more accessible.

3. **NPUs will mature:** As software support improves, NPUs in laptops and phones will handle small model inference natively, making AI a built-in OS feature.

4. **Hybrid local-cloud inference:** Run small models locally for fast, private responses; fall back to cloud APIs for tasks that need frontier-class models.

5. **On-device fine-tuning:** We're approaching the ability to fine-tune models on consumer hardware, creating truly personalized AI assistants.

6. **Speculative decoding:** Using small draft models to speed up large model inference by 2-3x  making large local models much more practical.

7. **1-bit and sub-2-bit quantization:** Research into extreme quantization (BitNet, etc.) could enable running today's large models on future smartphones.

---

## Quiz: Local AI

### Questions

**Q1.** Explain why VRAM is the primary bottleneck for local AI inference, not compute power. What determines how much VRAM you need?

**Q2.** You have an RTX 4070 with 12GB VRAM and 32GB system RAM. You want to run Qwen 3 32B. Calculate the VRAM needed at Q4_K_M quantization. Will it fit? What are your options?

**Q3.** What is the difference between CUDA cores and Tensor cores? Why do Tensor cores matter for AI inference?

**Q4.** Explain quantization: what is Q4_K_M, and why is it considered the "sweet spot" for most users? What is the approximate quality loss compared to FP16?

**Q5.** Compare GGUF and EXL2 formats. When would you choose each one?

**Q6.** Why is Apple Silicon particularly good for local AI compared to a similarly-priced Windows laptop? What architectural feature gives it an advantage?

**Q7.** What is PagedAttention (used in vLLM), and what problem does it solve? Why does it matter for serving models to multiple users?

**Q8.** A company wants to self-host DeepSeek-R1 (671B parameters, MoE) for their engineering team of 50 people. What hardware would they need? Estimate the cost.

### Answers

**A1.** VRAM is the bottleneck because model weights must be loaded into VRAM (GPU memory) before inference can begin. Each token generation requires reading all model weights from VRAM  this means the model must fit in VRAM for GPU inference. The compute (matrix multiplication) finishes quickly, but reading the weights from memory is what takes time. This is called being "memory-bandwidth bound."

VRAM needed = (Parameters × Bytes per parameter at chosen precision) + KV cache + overhead. For example, a 14B model at Q4 (0.5 bytes/param) needs ~8.5 GB, plus 0.5-2 GB for KV cache and overhead, totaling ~9-10.5 GB.

**A2.** Qwen 3 32B at Q4_K_M ≈ 32B × 0.5 bytes + quantization overhead ≈ **~19 GB**. This exceeds the 12GB VRAM of the RTX 4070. Options:
1. **Partial CPU offload:** Put ~60% of layers on GPU, rest on CPU. Speed: ~3-7 tok/s (slow but functional).
2. **Use Q3_K_M:** ~14.5 GB  still doesn't fit fully. Some offloading still needed.
3. **Use a smaller model:** Qwen 3 14B at Q4_K_M (~8.5 GB) fits fully and runs at ~25-35 tok/s.
4. **Use Qwen 3 30B-A3B (MoE):** Only 3B active params but all 30B in VRAM  still too large.
Best recommendation: Run Qwen 3 14B Q4_K_M for full-speed GPU inference, or accept slower speeds with 32B partially offloaded.

**A3.** **CUDA cores** are general-purpose parallel processors. Each performs one floating-point operation (multiply or add) per cycle. A GPU has thousands, enabling parallel computation. **Tensor cores** are specialized matrix multiplication units. Each Tensor core performs an entire 4×4 matrix multiply-accumulate in a single cycle  that's 128 multiply-add operations at once, versus one per CUDA core.

For AI inference, Tensor cores matter because neural networks are fundamentally sequences of matrix multiplications. Tensor cores execute these 10-20x faster than using CUDA cores alone, especially for FP16 and INT8 precisions. However, for very small batch sizes (typical of single-user local inference), the memory bandwidth bottleneck may matter more than raw compute.

**A4.** Q4_K_M stands for: **Q4** = 4-bit quantization, **K** = K-quant method (uses different bit widths for different parts of the weight tensors based on importance), **M** = Medium quality profile (balances important and less-important weights).

It's the "sweet spot" because:
- Reduces model size by ~75% compared to FP16
- Retains ~92% of FP16 quality (only ~8% degradation)
- The K-quant method preserves quality better than naive 4-bit by giving more bits to important weights
- Usually means the difference between "fits on your GPU" and "doesn't fit"
- The quality loss is imperceptible for most practical tasks

**A5.**
- **GGUF:** Universal format. Works on CPU, NVIDIA GPU, AMD GPU, Apple Silicon. Uses K-quant quantization. Best for cross-platform compatibility, CPU+GPU hybrid inference, and Apple Silicon. Most widely supported by tools (Ollama, LM Studio, llama.cpp).
- **EXL2:** NVIDIA-only format. Uses variable bits-per-weight (important layers get more precision). Fastest possible inference on NVIDIA GPUs. Smaller community, fewer tools.
Choose GGUF when: you use non-NVIDIA hardware, want maximum compatibility, or use Ollama/LM Studio.
Choose EXL2 when: you have an NVIDIA GPU and want absolute maximum speed.

**A6.** Apple Silicon uses **Unified Memory Architecture (UMA)**  CPU and GPU share the same pool of memory with no separate "VRAM." This means:
1. A Mac with 128GB memory gives ALL 128GB to the GPU for model loading
2. A PC with 128GB RAM but an RTX 4090 only gives 24GB to the GPU
3. No slow PCIe transfer between CPU RAM and GPU VRAM  unified memory is equally fast for both
4. Memory bandwidth is high (up to 546 GB/s on M4 Max)

A $3,500 MacBook Pro M4 Max (128GB) can run 70B models that would need $3,600+ in GPUs (2× RTX 4090) on a PC. The tradeoff: Apple's GPU is slower per-core, so tokens/second is lower, but the ability to run larger models is the key advantage.

**A7.** PagedAttention manages the KV cache (key-value cache used in attention) using a paging system inspired by operating system virtual memory. Traditional implementations allocate a fixed, contiguous block of memory for each request's KV cache (sized for maximum context length). This wastes memory when requests are shorter.

PagedAttention divides the KV cache into small, non-contiguous "pages" that are allocated dynamically as needed. This eliminates wasted memory from over-allocation. For multi-user serving, this means 2-4x more concurrent requests can be handled with the same VRAM, because memory is used proportionally to actual context length rather than maximum possible length.

**A8.** DeepSeek-R1 has 671B total parameters. At Q4_K_M, it needs ~400 GB of VRAM.

Hardware options:
- **8× A100 80GB** (640 GB total)  comfortable fit at Q4, with room for KV cache. Cost: ~$100,000-160,000 for the server hardware (GPUs + server chassis).
- **5× A100 80GB** (400 GB total)  tight fit at Q4, minimal KV cache room. Risky for 50 concurrent users.
- **4× H100 80GB** (320 GB total)  Q3 quantization needed. H100's higher bandwidth means faster inference. Cost: ~$120,000-200,000+.
- **Cloud rental:** 8× A100 on cloud: ~$20-30/hour → ~$15,000-22,000/month.

For 50 concurrent users, you'd want 2 instances for redundancy and to handle load. **Realistic budget: $30,000-50,000/month on cloud, or $200,000-350,000 one-time for owned hardware** (plus electricity, cooling, maintenance).

Alternative: Use DeepSeek-R1 API ($0.55/1M input, $2.19/1M output)  at 50 users × 50 queries/day × 1K tokens each, that's ~2.5M tokens/day ≈ ~$5-8/day. **Self-hosting only makes economic sense at much higher volumes or when privacy requirements mandate it.**

---

## Summary

Local AI has transformed from a niche hobby to a practical reality in 2026:

1. **It's accessible:** Ollama makes running a model as easy as `ollama run qwen3:8b`
2. **It's capable:** Qwen 3 32B running locally rivals cloud APIs from 2024
3. **The hardware exists:** Consumer GPUs (RTX 4090/5090), Apple Silicon (M4 Max), and even CPUs can run useful models
4. **Quantization is the enabler:** Q4_K_M reduces models to 1/4 their size with minimal quality loss
5. **Privacy is the killer feature:** Your data never leaves your machine
6. **The ecosystem is mature:** GGUF format, Ollama, LM Studio, llama.cpp  tools are polished and reliable

The key decision: **choose the largest model that fits comfortably in your VRAM at Q4_K_M quantization.** That's usually your best quality option for local inference.
