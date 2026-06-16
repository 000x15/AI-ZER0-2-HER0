# Part 3: Deep Learning  Architectures That Changed Everything

> *"Deep learning is not just a bigger hammer  it's a fundamentally different way of building tools."*

---

## Table of Contents

1. [Introduction: Why Architectures Matter](#1-introduction-why-architectures-matter)
2. [CNNs  Convolutional Neural Networks](#2-cnns--convolutional-neural-networks)
3. [RNNs  Recurrent Neural Networks](#3-rnns--recurrent-neural-networks)
4. [LSTMs  Long Short-Term Memory](#4-lstms--long-short-term-memory)
5. [GRUs  Gated Recurrent Units](#5-grus--gated-recurrent-units)
6. [Autoencoders](#6-autoencoders)
7. [VAEs  Variational Autoencoders](#7-vaes--variational-autoencoders)
8. [GANs  Generative Adversarial Networks](#8-gans--generative-adversarial-networks)
9. [Diffusion Models](#9-diffusion-models)
10. [Architecture Comparison Table](#10-architecture-comparison-table)
11. [Quiz](#11-quiz)

---

## 1. Introduction: Why Architectures Matter

In Part 2, we learned that neural networks can approximate any function. So why do we need different architectures? Why not just use a giant fully-connected network for everything?

**Analogy:** Think of tools in a workshop. A hammer, a screwdriver, and a saw can all damage wood  but each is *designed* for a specific job. A screwdriver is shaped to transfer rotational force into a screw. A saw's teeth are arranged to cut along a line. Similarly, neural network architectures are *shaped* to exploit the structure in different types of data.

| Data Type | Structure to Exploit | Best Architecture |
|-----------|---------------------|-------------------|
| Images | Spatial locality, translation invariance | CNNs |
| Sequences (text, audio) | Temporal ordering, dependencies | RNNs, LSTMs, Transformers |
| Tabular data | Feature interactions | MLPs, Gradient Boosting |
| Graphs | Node relationships | GNNs |
| Generating new data | Latent distributions | VAEs, GANs, Diffusion |

The key insight of deep learning isn't just "more layers"  it's **inductive bias**: building assumptions about data structure directly into the network architecture. This section covers the architectures that defined the deep learning revolution, why each was invented, and where they stand in 2026.

---

## 2. CNNs  Convolutional Neural Networks

### 2.1 Why Were CNNs Created?

**The Problem:** In the late 1980s, researchers wanted neural networks to recognize handwritten digits and objects in images. But images are *big*. A modest 256×256 grayscale image has 65,536 pixels. If you connected every pixel to every neuron in the first hidden layer (say, 1000 neurons), you'd need **65.5 million** parameters  just in layer one. This was computationally impossible in 1989 and wildly inefficient even today.

**The Insight:** Yann LeCun (inspired by Hubel and Wiesel's discoveries about the visual cortex) realized two critical things about images:

1. **Local patterns matter:** An edge, a corner, or a texture is defined by *nearby* pixels, not by pixels on opposite sides of the image.
2. **Patterns can appear anywhere:** A cat's ear looks the same whether it's in the top-left or bottom-right of an image.

These two observations  **locality** and **translation invariance**  led to the convolutional neural network.

### 2.2 How Convolution Works

**Analogy:** Imagine you're a detective with a magnifying glass examining a giant painting for a specific pattern  say, a particular shade of blue next to gold. You don't look at the whole painting at once. You slide your magnifying glass across the painting, checking each small region for the pattern. That's exactly what a convolutional filter (kernel) does.

#### The Convolution Operation

A **kernel** (or filter) is a small matrix of learnable weights (typically 3×3, 5×5, or 7×7). It slides across the input image, computing a dot product at each position:

```
INPUT IMAGE (5×5)                  KERNEL (3×3)
┌───┬───┬───┬───┬───┐             ┌────┬────┬────┐
│ 1 │ 0 │ 1 │ 0 │ 1 │             │  1 │  0 │ -1 │
├───┼───┼───┼───┼───┤             ├────┼────┼────┤
│ 0 │ 1 │ 0 │ 1 │ 0 │             │  1 │  0 │ -1 │
├───┼───┼───┼───┼───┤             ├────┼────┼────┤
│ 1 │ 0 │ 1 │ 0 │ 1 │             │  1 │  0 │ -1 │
├───┼───┼───┼───┼───┤             └────┴────┴────┘
│ 0 │ 1 │ 0 │ 1 │ 0 │        (This kernel detects
├───┼───┼───┼───┼───┤         vertical edges!)
│ 1 │ 0 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┴───┘
```

**Step-by-step for position (0,0):**

```
Overlay kernel on top-left 3×3 region:

Image patch:        Kernel:         Element-wise multiply:
┌───┬───┬───┐      ┌────┬────┬────┐    ┌──────┬──────┬──────┐
│ 1 │ 0 │ 1 │  ×   │  1 │  0 │ -1 │ =  │  1×1 │  0×0 │ 1×-1│
├───┼───┼───┤      ├────┼────┼────┤    ├──────┼──────┼──────┤
│ 0 │ 1 │ 0 │  ×   │  1 │  0 │ -1 │ =  │  0×1 │  1×0 │ 0×-1│
├───┼───┼───┤      ├────┼────┼────┤    ├──────┼──────┼──────┤
│ 1 │ 0 │ 1 │  ×   │  1 │  0 │ -1 │ =  │  1×1 │  0×0 │ 1×-1│
└───┴───┴───┘      └────┴────┴────┘    └──────┴──────┴──────┘

Sum = (1 + 0 + (-1)) + (0 + 0 + 0) + (1 + 0 + (-1)) = 0
```

**Step-by-step for position (0,1):**

```
Slide kernel one step right:

Image patch:        Kernel:         Element-wise multiply & sum:
┌───┬───┬───┐      ┌────┬────┬────┐
│ 0 │ 1 │ 0 │  ×   │  1 │  0 │ -1 │
├───┼───┼───┤      ├────┼────┼────┤
│ 1 │ 0 │ 1 │  ×   │  1 │  0 │ -1 │
├───┼───┼───┤      ├────┼────┼────┤
│ 0 │ 1 │ 0 │  ×   │  1 │  0 │ -1 │
└───┴───┴───┘      └────┴────┴────┘

Sum = (0 + 0 + 0) + (1 + 0 + (-1)) + (0 + 0 + 0) = 0
```

The complete output (called a **feature map** or **activation map**) for a 5×5 input with a 3×3 kernel (no padding, stride 1) is 3×3:

```
OUTPUT FEATURE MAP (3×3)
┌───┬───┬───┐
│ 0 │ 0 │ 0 │
├───┼───┼───┤
│ 0 │ 0 │ 0 │
├───┼───┼───┤
│ 0 │ 0 │ 0 │
└───┴───┴───┘

(All zeros because this checkerboard pattern
 has no vertical edges!)
```

**Output size formula:**
```
Output_size = (Input_size - Kernel_size + 2 × Padding) / Stride + 1
```

For our example: (5 - 3 + 0) / 1 + 1 = 3 ✓

### 2.3 Key CNN Concepts

#### Multiple Filters = Multiple Feature Maps

A single kernel detects one type of feature. Real CNN layers use **many** kernels simultaneously:

```
                    ┌──────────────────┐
                    │ Feature Map 1    │  ← Vertical edges
                    │ (from Kernel 1)  │
                    ├──────────────────┤
   INPUT IMAGE ────►│ Feature Map 2    │  ← Horizontal edges
                    │ (from Kernel 2)  │
                    ├──────────────────┤
                    │ Feature Map 3    │  ← Diagonal edges
                    │ (from Kernel 3)  │
                    ├──────────────────┤
                    │      ...         │
                    ├──────────────────┤
                    │ Feature Map 64   │  ← Some complex pattern
                    │ (from Kernel 64) │
                    └──────────────────┘
```

A typical first convolutional layer might have 64 kernels, producing 64 feature maps. The network **learns** what features to detect through backpropagation.

#### Padding

To prevent the output from shrinking, we can add zeros around the input:

```
ZERO PADDING (pad=1) around a 3×3 input:

  ┌───┬───┬───┬───┬───┐
  │ 0 │ 0 │ 0 │ 0 │ 0 │  ← added zeros
  ├───┼───┼───┼───┼───┤
  │ 0 │ 2 │ 1 │ 3 │ 0 │
  ├───┼───┼───┼───┼───┤
  │ 0 │ 4 │ 5 │ 2 │ 0 │  ← original data in center
  ├───┼───┼───┼───┼───┤
  │ 0 │ 1 │ 3 │ 4 │ 0 │
  ├───┼───┼───┼───┼───┤
  │ 0 │ 0 │ 0 │ 0 │ 0 │  ← added zeros
  └───┴───┴───┴───┴───┘

With a 3×3 kernel: output = (5 - 3 + 0)/1 + 1 = 3×3 (same as input!)
```

This is called **"same" padding**  the output has the same spatial dimensions as the input.

#### Stride

Stride controls how far the kernel moves at each step:

```
STRIDE = 1 (default):          STRIDE = 2:
┌───┬───┬───┬───┐              ┌───┬───┬───┬───┐
│ X │ X │ . │ . │  Step 1      │ X │ X │ . │ . │  Step 1
│ X │ X │ . │ . │              │ X │ X │ . │ . │
│ . │ . │ . │ . │              │ . │ . │ . │ . │
│ . │ . │ . │ . │              │ . │ . │ . │ . │
└───┴───┴───┴───┘              └───┴───┴───┴───┘

┌───┬───┬───┬───┐              ┌───┬───┬───┬───┐
│ . │ X │ X │ . │  Step 2      │ . │ . │ X │ X │  Step 2
│ . │ X │ X │ . │              │ . │ . │ X │ X │
│ . │ . │ . │ . │              │ . │ . │ . │ . │
│ . │ . │ . │ . │              │ . │ . │ . │ . │
└───┴───┴───┴───┘              └───┴───┴───┴───┘

→ 3 output positions             → Jumps to position (0,2)!
                                    Only 2 output positions per row
```

Larger strides reduce the output size, acting as a form of downsampling.

### 2.4 Pooling Layers

After convolution, **pooling** reduces the spatial dimensions while keeping the most important information.

#### Max Pooling (most common)

```
INPUT FEATURE MAP (4×4)         MAX POOL (2×2, stride 2)
┌────┬────┬────┬────┐          ┌────┬────┐
│  1 │  3 │  2 │  1 │          │    │    │
├────┼────┼────┼────┤    →     │  4 │  6 │  max of each 2×2 block
│  4 │  2 │  6 │  4 │          │    │    │
├────┼────┼────┼────┤          ├────┼────┤
│  5 │  1 │  3 │  2 │          │    │    │
├────┼────┼────┼────┤    →     │  5 │  4 │
│  2 │  3 │  4 │  1 │          │    │    │
└────┴────┴────┴────┘          └────┴────┘

Block 1: max(1,3,4,2) = 4      Block 2: max(2,1,6,4) = 6
Block 3: max(5,1,2,3) = 5      Block 4: max(3,2,4,1) = 4
```

**Why pool?**
- Reduces computation in subsequent layers
- Provides a degree of translation invariance (the exact position of a feature doesn't matter just that it's roughly "there")
- Reduces overfitting by reducing the number of parameters

#### Average Pooling

Instead of taking the maximum, takes the average of each block. Used less often, but useful in some architectures (e.g., as the final global average pooling layer).

### 2.5 The Complete CNN Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────────┐
│             │    │              │    │           │    │              │
│ INPUT IMAGE │───►│ CONV + ReLU  │───►│ MAX POOL  │───►│ CONV + ReLU  │──┐
│ 224×224×3   │    │ 224×224×64   │    │ 112×112×64│    │ 112×112×128  │  │
│             │    │              │    │           │    │              │  │
└─────────────┘    └──────────────┘    └───────────┘    └──────────────┘  │
                                                                          │
┌─────────────────────────────────────────────────────────────────────────┘
│
│   ┌───────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│   │           │    │              │    │              │    │          │
└──►│ MAX POOL  │───►│ CONV + ReLU  │───►│ FLATTEN      │───►│ DENSE    │
    │ 56×56×128 │    │ 56×56×256    │    │ 802,816      │    │ → 10     │
    │           │    │              │    │ (vector)     │    │ classes  │
    └───────────┘    └──────────────┘    └──────────────┘    └──────────┘

    ◄────── Feature Extraction ──────►   ◄─── Classification ──►
```

**Key insight:** As we go deeper:
- Spatial dimensions **decrease** (224 → 112 → 56 → ...)
- Number of channels **increases** (3 → 64 → 128 → 256 → ...)
- Features become more **abstract** (edges → textures → parts → objects)

### 2.6 What CNNs Learn: The Feature Hierarchy

```
Layer 1 (early):     Layer 3 (middle):     Layer 5 (deep):
┌──────────┐         ┌──────────┐          ┌──────────┐
│ ─  │  \  │         │ ◠◡ │ ⊞  │          │  🐱     │
│────│─────│         │────│─────│          │─────────│
│ |  │  /  │         │ ○  │ ≋  │          │  🐕     │
│────│─────│         │────│─────│          │─────────│
│ ·  │  ×  │         │ ⌐¬ │ ≡  │          │  🚗     │
└──────────┘         └──────────┘          └──────────┘
 Edges, colors        Textures, parts       Whole objects
```

This hierarchical feature learning is one of the most powerful aspects of CNNs. The network automatically discovers the right features  no hand-engineering needed.

### 2.7 Famous CNN Architectures (Historical Progression)

| Architecture     | Year | Key Innovation                        | Depth    | Top-5 Error (ImageNet) |
| ---------------- | ---- | ------------------------------------- | -------- | ---------------------- |
| **LeNet-5**      | 1998 | First practical CNN (handwriting)     | 5        | N/A                    |
| **AlexNet**      | 2012 | GPU training, ReLU, dropout           | 8        | 16.4%                  |
| **VGGNet**       | 2014 | Small 3×3 filters, very deep          | 19       | 7.3%                   |
| **GoogLeNet**    | 2014 | Inception modules (parallel paths)    | 22       | 6.7%                   |
| **ResNet**       | 2015 | Residual (skip) connections           | 152      | 3.6%                   |
| **EfficientNet** | 2019 | Compound scaling                      | Variable | 2.9%                   |
| **ConvNeXt**     | 2022 | Modernized CNN with Transformer ideas | Variable | Competitive            |

### 2.8 Residual Connections (The ResNet Revolution)

Before ResNet, making networks deeper caused them to perform *worse*  not because of overfitting, but because gradients couldn't flow through so many layers. ResNet solved this with **skip connections**:

```
                   ┌───────────────────┐
                   │                   │
                   │  SKIP CONNECTION  │
                   │  (identity)       │
       ┌───────────┤                   ├──────────┐
       │           └───────────────────┘          │
       │                                          │
       ▼                                          │
  ┌─────────┐     ┌─────────┐     ┌─────────┐    │   ┌────────┐
  │  Input  │────►│ Conv    │────►│ Conv    │────(+)─►│ Output │
  │    x    │     │ + ReLU  │     │         │    ▲    │ F(x)+x │
  └─────────┘     └─────────┘     └─────────┘    │   └────────┘
                                                  │
                                     F(x)         │
                                                  │
                   Instead of learning H(x) directly,
                   learn F(x) = H(x) - x (the residual)
                   Output = F(x) + x
```

**Why this works:** If the optimal transformation is close to the identity (do nothing), it's much easier for the network to learn F(x) ≈ 0 than to learn H(x) ≈ x. The skip connection gives gradients a "highway" to flow through during backpropagation.

### 2.9 Strengths, Weaknesses, and Modern Relevance (2026)

**Strengths:**
- Excellent at spatial pattern recognition
- Parameter-efficient through weight sharing (same kernel applied everywhere)
- Translation invariant by design
- Well-understood, mature technology

**Weaknesses:**
- Struggle with global context (fixed receptive field)
- Not naturally suited for sequential data
- Pooling discards precise spatial information

**Modern Relevance (2026):**
- Still dominant in **real-time** and **edge** applications (phones, embedded systems) where efficiency matters
- **ConvNeXt** and similar architectures remain competitive with Vision Transformers while being simpler
- Used as **feature extractors** inside larger systems (e.g., the image encoder in multimodal models)
- **Hybrid architectures** combining CNNs with Transformers are common (e.g., CNN for local features + Transformer for global attention)
- Essential in medical imaging, autonomous driving perception, and manufacturing inspection

---

## 3. RNNs  Recurrent Neural Networks

### 3.1 Why Were RNNs Created?

**The Problem:** Regular neural networks (feedforward/CNNs) process each input independently. But language, music, stock prices, and many real-world signals are **sequential**  the meaning of each element depends on what came before.

**Analogy:** Imagine reading a book where someone erased your memory between each word. "The cat sat on the ___"  you could guess "mat" only because you remember all the previous words. A feedforward network reads each word with amnesia.

RNNs were designed to give neural networks **memory**  the ability to maintain a hidden state that accumulates information from previous time steps.

### 3.2 How RNNs Work

The key idea: the network has a **loop**. The output at each time step is fed back as input to the next time step.

#### Compact vs. Unrolled View

```
COMPACT VIEW:                    UNROLLED VIEW:

    ┌──────┐                     ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
    │      │                     │      │    │      │    │      │    │      │
───►│ RNN  │───►  output         │ RNN  │───►│ RNN  │───►│ RNN  │───►│ RNN  │
    │      │                     │      │    │      │    │      │    │      │
    └──┬───┘                     └──────┘    └──────┘    └──────┘    └──────┘
       │   ▲                       ▲           ▲           ▲           ▲
       │   │                       │           │           │           │
       └───┘                      x₁          x₂          x₃          x₄
    (self-loop                   "The"       "cat"       "sat"        "on"
     = memory)
                                  h₁          h₂          h₃          h₄
                                  ▼           ▼           ▼           ▼
                                  y₁          y₂          y₃          y₄
```

**The same weights are shared across all time steps!** This is what makes it "recurrent."

#### The Math (Simple RNN / Elman Network)

At each time step t:

```
h_t = tanh(W_hh · h_{t-1}  +  W_xh · x_t  +  b_h)
y_t = W_hy · h_t + b_y

Where:
  h_t     = hidden state at time t (the "memory")
  x_t     = input at time t
  y_t     = output at time t
  W_hh    = hidden-to-hidden weights (memory transformation)
  W_xh    = input-to-hidden weights
  W_hy    = hidden-to-output weights
  b_h,b_y = biases
```

#### Worked Example

Let's trace through a tiny RNN with hidden size 2 processing the sequence [1, 0, 1]:

```
Parameters (randomly initialized):
  W_xh = [0.5, -0.3]ᵀ       (2×1 matrix, since input dim = 1)
  W_hh = [[0.2, 0.1],        (2×2 matrix)
           [0.3, -0.1]]
  b_h = [0, 0]ᵀ

Initial hidden state: h₀ = [0, 0]ᵀ

TIME STEP 1: x₁ = 1
  W_xh · x₁ = [0.5, -0.3]ᵀ
  W_hh · h₀ = [0, 0]ᵀ
  h₁ = tanh([0.5, -0.3]ᵀ) = [0.462, -0.291]ᵀ

TIME STEP 2: x₂ = 0
  W_xh · x₂ = [0, 0]ᵀ
  W_hh · h₁ = [0.2×0.462 + 0.1×(-0.291), 0.3×0.462 + (-0.1)×(-0.291)]ᵀ
             = [0.063, 0.168]ᵀ
  h₂ = tanh([0.063, 0.168]ᵀ) = [0.063, 0.166]ᵀ

TIME STEP 3: x₃ = 1
  W_xh · x₃ = [0.5, -0.3]ᵀ
  W_hh · h₂ = [0.2×0.063 + 0.1×0.166, 0.3×0.063 + (-0.1)×0.166]ᵀ
             = [0.029, 0.002]ᵀ
  h₃ = tanh([0.529, -0.298]ᵀ) = [0.485, -0.289]ᵀ
```

Notice how the hidden state **accumulates** information from all previous inputs. By time step 3, h₃ contains traces of all three inputs.

### 3.3 The Vanishing Gradient Problem

**This is the Achilles' heel of simple RNNs.**

During backpropagation through time (BPTT), gradients must flow backwards through each time step. At each step, the gradient is multiplied by W_hh and the derivative of tanh:

```
GRADIENT FLOW (backwards through time):

  ∂Loss    ∂h₅   ∂h₄   ∂h₃   ∂h₂
  ───── = ───── × ───── × ───── × ─────
  ∂h₁     ∂h₄   ∂h₃   ∂h₂   ∂h₁

Each factor involves W_hh and tanh'(·)

Since tanh'(x) ∈ (0, 1] and is often < 1:
```

```
VANISHING GRADIENT VISUALIZATION:

Gradient magnitude at each time step (going backwards):

Step 10:  ████████████████████  1.000  (loss computed here)
Step  9:  ██████████████████    0.850
Step  8:  ███████████████       0.700
Step  7:  ████████████          0.550
Step  6:  █████████             0.400
Step  5:  ██████                0.250
Step  4:  ████                  0.130
Step  3:  ██                    0.060
Step  2:  █                     0.020
Step  1:  ▏                     0.003  ← almost zero!

The network CANNOT learn from early inputs!
Long-range dependencies are lost.
```

**Practical consequence:** A simple RNN might process a 100-word sentence but effectively "forget" everything before word 80. It can't connect "The movie, which was directed by Christopher Nolan and featured..." (word 1) to "...was excellent" (word 15).

**The exploding gradient problem** is the opposite: when eigenvalues of W_hh > 1, gradients grow exponentially. This is usually solved with **gradient clipping** (capping the gradient norm).

### 3.4 Strengths, Weaknesses, and Modern Relevance (2026)

**Strengths:**
- Conceptually simple and elegant
- Can process variable-length sequences
- Shared weights across time steps (parameter efficient)

**Weaknesses:**
- Vanishing/exploding gradients limit practical sequence length
- Sequential processing  cannot parallelize across time steps
- Largely superseded by LSTMs, GRUs, and especially Transformers

**Modern Relevance (2026):**
- Rarely used directly in production
- Important **conceptual foundation** for understanding LSTMs, GRUs
- Some niche uses in tiny embedded systems where model size is critical
- Historical significance: the concept of "recurrence" influenced many later designs

---

## 4. LSTMs  Long Short-Term Memory

### 4.1 Why Were LSTMs Created?

In 1997, Sepp Hochreiter and Jürgen Schmidhuber tackled the vanishing gradient problem head-on. Their insight: the RNN needs a way to **selectively remember and forget** information over long time spans. Simple multiplication through tanh inevitably kills gradients. The solution? Create a separate **memory highway** with controlled access.

**Analogy:** Think of an LSTM cell as a **conveyor belt** in a factory. Items (information) travel along the belt undisturbed. At certain stations, workers can:
1. **Remove items** from the belt (forget gate)
2. **Add new items** to the belt (input gate)
3. **Read items** from the belt to do work (output gate)

The belt itself provides an uninterrupted path for information to flow over long distances.

### 4.2 The LSTM Architecture

```
LSTM CELL (Detailed View):

                    ┌─────────────────────────────────────────────┐
                    │                                             │
    c_{t-1} ───────►│─────────────(×)──────────(+)───────────────►│──── c_t
    (cell state)    │              ▲              ▲               │  (cell state)
                    │              │              │               │
                    │         ┌────┴────┐   ┌────┴────────┐      │
                    │         │ FORGET   │   │   (×)       │      │
                    │         │  GATE    │   │    ▲   ▲    │      │
                    │         │  f_t     │   │    │   │    │      │
                    │         └────┬────┘   │ ┌──┴┐ ┌┴──┐ │      │
                    │              │        │ │ i │ │ c̃ │ │      │
                    │              │        │ │ _t│ │ _t│ │      │
                    │              │        │ │IN │ │CAN│ │      │
                    │              │        │ │PUT│ │DID│ │      │
                    │              │        │ │   │ │ATE│ │      │
                    │              │        │ └─┬─┘ └─┬─┘ │      │
                    │              │        └───┼─────┼───┘      │
                    │              │            │     │           │
    h_{t-1} ──────►│──────────────┼────────────┼─────┼──────┐    │
    (hidden)       │              │            │     │      │    │
                    │         ┌────┴────────────┴─────┴──┐   │    │
                    │         │      [h_{t-1}, x_t]      │   │    │
                    │         │     (concatenated)        │   │    │
                    │         └────┬─────────────────┬───┘   │    │
                    │              │                 │       │    │
                    │              │            ┌────┴────┐  │    │
                    │              │            │ OUTPUT  │  │    │
                    │              │            │  GATE   │  │    │
                    │              │            │  o_t    │  │    │
                    │              │            └────┬────┘  │    │
                    │              │                 │       │    │
                    │              │            tanh(c_t)────(×)──┼───► h_t
                    │              │                         │    │   (hidden)
       x_t ───────►│──────────────┘                         │    │
      (input)      │                                        │    │
                    └────────────────────────────────────────┘    │
                                                                  │
```

### 4.3 The Four Components

The LSTM has **three gates** and one **candidate memory**:

```
Given: concatenated input z = [h_{t-1}, x_t]

1. FORGET GATE:     f_t = σ(W_f · z + b_f)        → What to remove from memory
2. INPUT GATE:      i_t = σ(W_i · z + b_i)        → What new info to store
3. CANDIDATE:       c̃_t = tanh(W_c · z + b_c)     → New candidate values
4. OUTPUT GATE:     o_t = σ(W_o · z + b_o)        → What to output

Cell state update:  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
Hidden state:       h_t = o_t ⊙ tanh(c_t)

Where σ = sigmoid (outputs 0-1), ⊙ = element-wise multiplication
```

### 4.4 Gate Intuition with Example

Let's say we're processing the sentence: **"The cat, which was very fluffy, sat on the mat."**

```
Processing "The cat":
  ┌─────────────────────────────────────────────────┐
  │ Cell state stores: SUBJECT = "cat" (singular)    │
  │ Forget gate: ~1.0 (keep everything, fresh start) │
  │ Input gate:  ~1.0 (store "cat" info)             │
  └─────────────────────────────────────────────────┘

Processing ", which was very fluffy,":
  ┌─────────────────────────────────────────────────┐
  │ Cell state still holds: SUBJECT = "cat"          │
  │ Forget gate: ~1.0 (DON'T forget the subject!)   │
  │ Input gate:  ~0.3 (store some adjective info)    │
  │ Output gate: ~0.2 (not outputting yet)           │
  └─────────────────────────────────────────────────┘

Processing "sat":
  ┌─────────────────────────────────────────────────┐
  │ Cell state: SUBJECT = "cat" (still remembered!) │
  │ Output gate: ~0.9 (now we need the subject to   │
  │   verify subject-verb agreement: "cat SAT" ✓)   │
  └─────────────────────────────────────────────────┘
```

**Why this solves vanishing gradients:** The cell state c_t has an **additive** update path (c_t = f_t ⊙ c_{t-1} + ...). When f_t ≈ 1, the gradient flows through multiplication by ~1, preserving it over long sequences. Compare this to simple RNNs where the gradient must pass through tanh repeatedly.

### 4.5 Strengths, Weaknesses, and Modern Relevance (2026)

**Strengths:**
- Effectively captures long-range dependencies (100-1000+ time steps)
- Solves the vanishing gradient problem via the cell state highway
- Well-proven in practice across many domains

**Weaknesses:**
- Complex: 4× the parameters of a simple RNN (four weight matrices instead of one)
- Still sequential  cannot parallelize across time steps
- Slower to train than Transformers on modern hardware
- Overkill for short sequences where simpler models suffice

**Modern Relevance (2026):**
- Still used in **time-series forecasting** and **signal processing** where data is truly sequential
- Found in some **speech recognition** pipelines
- Being replaced by Transformers and **State Space Models (SSMs)** like Mamba in most NLP tasks
- Important for understanding the evolution toward attention mechanisms
- Used in edge devices where Transformer inference is too expensive

---

## 5. GRUs  Gated Recurrent Units

### 5.1 Why Were GRUs Created?

In 2014, Kyunghyun Cho et al. asked: "Do we really need all four components of an LSTM?" They created the **Gated Recurrent Unit**  a simplified version with only **two gates** that performs comparably to LSTMs in many tasks.

**Analogy:** If an LSTM is a luxury car with every feature, a GRU is a sports car  fewer controls, but gets you there just as fast (and sometimes faster because it's lighter).

### 5.2 GRU Architecture

```
GRU CELL:

                    ┌──────────────────────────────────────┐
                    │                                      │
    h_{t-1} ──────►│──┬──────────────────────────────┐    │
                    │  │                              │    │
                    │  │    ┌──────────┐              │    │
                    │  ├───►│  RESET   │──┐           │    │
                    │  │    │  GATE r  │  │           │    │
                    │  │    └──────────┘  │           │    │
                    │  │                  ▼           │    │
                    │  │          (r ⊙ h_{t-1})      │    │
                    │  │                  │           │    │
                    │  │            ┌─────▼─────┐    │    │
                    │  │            │ CANDIDATE  │    │    │
                    │  │            │    h̃_t    │    │    │
                    │  │            └─────┬─────┘    │    │
                    │  │                  │          │    │
                    │  │    ┌──────────┐  │          │    │
                    │  ├───►│  UPDATE  │  │          │    │
                    │  │    │  GATE z  │  │          │    │
                    │  │    └────┬─────┘  │          │    │
                    │  │         │        │          │    │
                    │  │    ┌────▼────────▼────┐     │    │
                    │  │    │                  │     │    │
                    │  └───►│ (1-z)⊙h_{t-1}   │     │    │
                    │       │    + z⊙h̃_t      │─────┼───►│──── h_t
                    │       │                  │     │    │
                    │       └──────────────────┘     │    │
                    │                                │    │
       x_t ───────►│────────────────────────────────┘    │
                    └──────────────────────────────────────┘
```

### 5.3 The Math

```
Given: z = [h_{t-1}, x_t]  (concatenated)

1. UPDATE GATE:    z_t = σ(W_z · z + b_z)         → How much to update
2. RESET GATE:     r_t = σ(W_r · z + b_r)         → How much past to forget
3. CANDIDATE:      h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t] + b)
4. FINAL STATE:    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Key difference from LSTM:** The update gate z_t acts as BOTH the forget and input gates. When z_t ≈ 0, the old state is kept (forget nothing, input nothing new). When z_t ≈ 1, the old state is replaced with the candidate.

### 5.4 LSTM vs. GRU Comparison

```
LSTM:                              GRU:
┌──────────────────────┐           ┌──────────────────────┐
│ Gates: 3             │           │ Gates: 2             │
│  • Forget gate       │           │  • Reset gate        │
│  • Input gate        │           │  • Update gate       │
│  • Output gate       │           │                      │
│                      │           │                      │
│ States: 2            │           │ States: 1            │
│  • Cell state (c_t)  │           │  • Hidden state (h_t)│
│  • Hidden state (h_t)│           │                      │
│                      │           │                      │
│ Parameters per unit: │           │ Parameters per unit: │
│  4 × (n² + n·m + n) │           │  3 × (n² + n·m + n) │
│  (about 33% more)    │           │  (more efficient)    │
│                      │           │                      │
│ Better for:          │           │ Better for:          │
│  • Very long deps    │           │  • Smaller datasets  │
│  • Complex patterns  │           │  • Faster training   │
│  • When in doubt     │           │  • Simpler tasks     │
└──────────────────────┘           └──────────────────────┘
```

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | More (~33%) | Fewer |
| Training speed | Slower | Faster |
| Long sequences | Slightly better | Comparable |
| Small datasets | May overfit | Better generalization |
| Separate memory | Yes (cell state) | No (combined) |
| Modern usage | Time-series, speech | Similar applications |

### 5.5 Modern Relevance (2026)

- GRUs and LSTMs are roughly **interchangeable** in most applications
- Both are being **replaced by Transformers** for most NLP tasks
- Both remain relevant for **streaming/online** applications where you process data one step at a time
- **State Space Models** (Mamba, S4) are emerging as potential successors that combine the efficiency of RNN-style sequential processing with the performance of Transformers

---

## 6. Autoencoders

### 6.1 Why Were Autoencoders Created?

**The Problem:** How can a neural network learn useful representations of data **without labels**? In supervised learning, we need labeled data (expensive!). Can we learn structure from raw, unlabeled data?

**The Insight (1986, Rumelhart et al.):** Force a network to compress data through a bottleneck, then reconstruct it. If the reconstruction is good, the bottleneck must contain a **meaningful compressed representation**.

**Analogy:** Imagine sending a postcard to describe a photograph. You can only write 50 words. You'd focus on the essential elements: "Beach at sunset, golden sand, two palm trees on the left, small sailboat on the horizon." Someone reading your description could imagine a reasonable approximation of the photo. Your 50-word description is the **latent representation**  compressed but meaningful.

### 6.2 Architecture

```
AUTOENCODER ARCHITECTURE:

INPUT                ENCODER              LATENT SPACE           DECODER              OUTPUT
(original)           (compress)           (bottleneck)           (decompress)         (reconstruction)

┌─────────┐     ┌─────────────┐       ┌───────────┐       ┌─────────────┐     ┌─────────┐
│ ■ □ ■ □ │     │             │       │           │       │             │     │ ■ □ ■ □ │
│ □ ■ □ ■ │     │  784 → 256  │       │           │       │  32 → 256   │     │ □ ■ □ ■ │
│ ■ □ ■ □ │ ───►│  256 → 128  │ ────► │  z (32)   │ ────► │  256 → 784  │ ───►│ ■ □ ■ □ │
│ □ ■ □ ■ │     │  128 → 32   │       │           │       │             │     │ □ ■ □ ■ │
│ ■ □ ■ □ │     │             │       │           │       │             │     │ ■ □ ■ □ │
└─────────┘     └─────────────┘       └───────────┘       └─────────────┘     └─────────┘
  28×28=784         ↓↓↓                   32 dims             ↑↑↑               28×28=784
  dimensions     (compression)         (compressed!)       (expansion)          dimensions

                 Loss = ||input - output||²  (reconstruction error)
```

**The bottleneck forces compression.** 784 dimensions → 32 dimensions means the network must learn to represent the essential structure of the data in just 32 numbers.

### 6.3 Types of Autoencoders

#### Undercomplete Autoencoder
The standard type: latent dimension < input dimension. Forces meaningful compression.

#### Sparse Autoencoder
Latent dimension can be ≥ input dimension, but a sparsity penalty ensures most latent units are inactive. Only a few units "fire" for any given input.

```
Dense latent (standard):     Sparse latent:
[0.5, 0.8, 0.3, 0.7,        [0.0, 0.0, 0.9, 0.0,
 0.2, 0.6, 0.4, 0.9]         0.0, 0.8, 0.0, 0.0]
 (all active)                 (mostly zero  sparse!)
```

#### Denoising Autoencoder
Input is **corrupted** (noise added, pixels masked), and the network must reconstruct the **clean** original. This forces it to learn robust features, not just copy.

```
DENOISING AUTOENCODER:

Clean input:     Corrupted input:     Reconstructed:
┌─────────┐      ┌─────────┐         ┌─────────┐
│ ■ ■ ■   │      │ ■ x ■   │         │ ■ ■ ■   │
│ ■   ■   │  ──► │ x   ■   │  ────►  │ ■   ■   │
│ ■ ■ ■   │      │ ■ ■ x   │         │ ■ ■ ■   │  (digit "8")
└─────────┘      └─────────┘         └─────────┘
                  x = noise            Recovered!
```

### 6.4 What the Latent Space Looks Like

```
LATENT SPACE VISUALIZATION (2D for illustration):

    ▲ z₂
    │
    │    ○○○         ××
    │   ○○○○        ×××
    │    ○○○       ××××
    │              ×××
    │
    │        △△△
    │       △△△△        □□
    │        △△△       □□□□
    │                   □□□
    │
    ──────────────────────────► z₁

    ○ = digit "0"    × = digit "1"
    △ = digit "7"    □ = digit "9"

Similar inputs cluster together in latent space!
```

### 6.5 Use Cases

| Application | How It Works |
|-------------|-------------|
| **Dimensionality reduction** | Like PCA but nonlinear  captures complex structure |
| **Anomaly detection** | Normal data reconstructs well; anomalies don't |
| **Denoising** | Trained on noisy→clean pairs; removes noise from new data |
| **Feature learning** | Encoder's latent representation used as features for downstream tasks |
| **Data compression** | Learned compression tailored to specific data types |
| **Pretraining** | Train encoder unsupervised, then fine-tune with labels (historically important) |

### 6.6 Modern Relevance (2026)

- **Sparse autoencoders** are crucial for **mechanistic interpretability**  used to decompose the activations of large language models (like Claude, GPT) into interpretable features
- **Denoising autoencoders** are the conceptual precursor to **diffusion models** (Section 9)
- **Masked autoencoders (MAE)** are state-of-the-art for self-supervised vision pretraining
- Used extensively in **anomaly detection** in cybersecurity and manufacturing
- The encoder-decoder paradigm influences virtually all modern generative models

---

## 7. VAEs  Variational Autoencoders

### 7.1 Why Were VAEs Created?

**The Problem with Regular Autoencoders:** You can encode images to latent vectors and decode them back. But what if you want to **generate new images**? You'd need to sample a random point in latent space and decode it. Problem: the latent space of a regular autoencoder has "holes"  regions that don't correspond to any valid data.

**Analogy:** Imagine a filing cabinet where some drawers contain documents and others are empty. If you randomly open a drawer, you might get nothing. A VAE reorganizes the cabinet so that **every** drawer contains something meaningful, and nearby drawers contain similar documents.

### 7.2 Key Insight: Regularized Latent Space

VAEs (Kingma & Welling, 2013) don't encode each input to a single point  they encode it to a **probability distribution** (a mean and variance):

```
AUTOENCODER:                          VAE:

Input → Encoder → z (point) → Decoder    Input → Encoder → μ, σ → Sample z → Decoder

Latent space:                         Latent space:
    │                                     │
    │  •    •                             │  (···)  (···)
    │     •                               │ (·····)(·····)
    │        •                            │  (···)  (···)
    │  •                                  │
    │        •                            │ (overlapping distributions
    │                                     │  create smooth, filled space)
    └──────────►                          └──────────────────►

  Scattered points                      Smooth, continuous
  Gaps between clusters                 Can sample anywhere!
```

### 7.3 The VAE Architecture

```
VAE ARCHITECTURE:

                 ENCODER                                    DECODER
            ┌─────────────────┐                      ┌─────────────────┐
            │                 │     ┌─────┐          │                 │
            │                 │────►│  μ  │──┐       │                 │
 Input x ──►│  Neural Net     │     └─────┘  │       │  Neural Net     │──► x̂ (reconstruction)
            │  (recognition   │              ▼       │  (generative    │
            │   model)        │     ┌──────────┐     │   model)        │
            │                 │     │ z = μ +  │────►│                 │
            │                 │     │ σ⊙ε     │     │                 │
            │                 │────►│          │     │                 │
            │                 │     └──────────┘     │                 │
            └─────────────────┘     ┌─────┐  ▲       └─────────────────┘
                                    │ σ²  │──┘
                                    └─────┘
                                         ▲
                                    ε ~ N(0,1)
                                  (random noise)

                        "Reparameterization Trick"
```

### 7.4 The Reparameterization Trick

**Problem:** We need to sample z from N(μ, σ²), but sampling is not differentiable  we can't backpropagate through random sampling!

**Solution:** Instead of sampling z directly, compute z = μ + σ ⊙ ε, where ε ~ N(0,1) is fixed random noise. Now gradients can flow through μ and σ, and the randomness comes from ε (which doesn't need gradients).

```
NON-DIFFERENTIABLE:               REPARAMETERIZED (differentiable):

  μ, σ → [SAMPLE] → z             μ ──────────────(+)──► z
           ▲   ✗                   σ ──────(×)──────┘
           │   (can't                       ▲
           │    backprop)                    │
                                       ε ~ N(0,1)
                                    (fixed random noise,
                                     no gradient needed)
```

### 7.5 The VAE Loss Function

```
Loss = Reconstruction Loss  +  KL Divergence

     = ||x - x̂||²           +  D_KL(q(z|x) || p(z))

     ↑                         ↑
     "Output should match      "Latent distribution should
      the input"                be close to standard normal N(0,1)"
```

The **KL divergence** term is the magic sauce. It pushes the encoder's output distributions toward a standard normal, ensuring:
- The latent space is **smooth** (no gaps)
- The latent space is **centered** (around origin)
- Different inputs' distributions **overlap** slightly (enabling interpolation)

### 7.6 Generating New Data

Once trained, you can **discard the encoder** and generate new data by:

```
1. Sample z ~ N(0, 1)         (random point from standard normal)
2. Pass z through decoder      (decode to data space)
3. Output is a new, plausible data point!

LATENT SPACE INTERPOLATION:

 z_A ─────•─────•─────•─────•─────• z_B
          │     │     │     │     │
          ▼     ▼     ▼     ▼     ▼
         [😊]  [😐]  [🤨]  [😠]  [😤]

Smoothly interpolating between two faces in latent space
produces smooth transitions in image space!
```

### 7.7 Modern Relevance (2026)

- **Latent Diffusion Models** (Stable Diffusion) use a VAE encoder to compress images into latent space, run diffusion there, then use the VAE decoder to reconstruct images  this is the standard architecture in 2026
- VAE theory underpins much of modern **generative modeling**
- **VQ-VAE** (Vector Quantized VAE) with discrete latent codes is used in audio (music generation, speech synthesis)
- Less commonly used standalone for image generation (outperformed by diffusion), but critical as **components** in larger systems
- Important in **drug discovery** and **molecular design** for exploring molecular space

---

## 8. GANs  Generative Adversarial Networks

### 8.1 Why Were GANs Created?

In 2014, Ian Goodfellow had an insight at a bar (legend has it): what if two neural networks competed against each other? One generates fake data, the other tries to detect fakes. Through competition, the generator becomes incredibly good at producing realistic data.

**Analogy:** Think of an **art forger** (generator) and an **art detective** (discriminator).
- The forger creates fake paintings and tries to pass them off as real
- The detective examines paintings and judges whether they're real or fake
- As the detective gets better, the forger must improve to fool them
- As the forger improves, the detective must get sharper
- After many rounds, the forger produces paintings indistinguishable from real art

### 8.2 GAN Architecture

```
GAN ARCHITECTURE:

                    GENERATOR (G)                     DISCRIMINATOR (D)
              ┌──────────────────────┐          ┌──────────────────────┐
              │                      │          │                      │
 Random       │  Neural Network      │  Fake    │  Neural Network      │
 Noise z ────►│  (learns to create   │─────────►│  (learns to tell     │──► Real or Fake?
 ~ N(0,1)     │   realistic data)    │    ┌────►│   real from fake)    │    (probability)
              │                      │    │     │                      │
              └──────────────────────┘    │     └──────────────────────┘
                                          │
 Real Data ───────────────────────────────┘


TRAINING DYNAMICS:

 Round 1:  Generator: [scribbles]    Discriminator: "Obviously fake" ✓
 Round 10: Generator: [crude face]   Discriminator: "Still fake" ✓
 Round 100: Generator: [decent face] Discriminator: "Hmm... fake?" ~50/50
 Round 1000: Generator: [photorealistic face] Discriminator: "I can't tell!" 
```

### 8.3 The Math: A Minimax Game

```
min_G max_D  V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

In plain English:
  - D wants to MAXIMIZE: correctly classify real as real, fake as fake
  - G wants to MINIMIZE: make D unable to tell fake from real

Breaking it down:
  E[log D(x)]        → D's score on real data (D wants this HIGH, near log(1) = 0)
  E[log(1 - D(G(z)))] → D's score on fake data (D wants D(G(z))→0, so this→log(1)=0)
                                                  (G wants D(G(z))→1, so this→log(0)=-∞)
```

### 8.4 Training Process (Alternating Updates)

```
STEP 1: Train Discriminator (freeze Generator)
  ┌────────────────────────────────────────────┐
  │ • Feed real images → D should output ~1.0  │
  │ • Feed G(z) fake images → D should output  │
  │   ~0.0                                     │
  │ • Update D's weights to improve accuracy   │
  └────────────────────────────────────────────┘
                         │
                         ▼
STEP 2: Train Generator (freeze Discriminator)
  ┌────────────────────────────────────────────┐
  │ • Generate fake images G(z)                │
  │ • Feed through D → get score               │
  │ • Update G's weights to make D's score     │
  │   closer to 1.0 (fool the discriminator)   │
  └────────────────────────────────────────────┘
                         │
                         ▼
              Repeat thousands of times
```

### 8.5 GAN Training Challenges

GANs are notoriously difficult to train. Three major issues:

#### Mode Collapse
```
REAL DATA DISTRIBUTION:         MODE COLLAPSE:

    ██  ██  ██  ██              ████████████████
    ██  ██  ██  ██              ████████████████
    ██  ██  ██  ██              (all in one mode!)
    ██  ██  ██  ██
 (4 modes: 4 faces)            Generator only makes
                                ONE type of face

The generator finds one output that fools the discriminator
and keeps producing variations of it, ignoring other modes.
```

#### Training Instability
```
Loss during GAN training (actual):

D_loss │ ╱╲   ╱╲    ╱╲   ╱╲
       │╱  ╲ ╱  ╲  ╱  ╲ ╱  ╲     ← Oscillating, never converges!
       │    ╲╱    ╲╱    ╲╱    ╲
───────┼──────────────────────────► epochs

vs. standard supervised learning:

Loss   │╲
       │ ╲
       │  ╲__________              ← Smooth convergence
───────┼──────────────────────────► epochs
```

#### Vanishing Gradients for Generator
If D becomes too good (outputs 0 for all fakes), G gets zero gradient and can't improve. It's a delicate balance.

### 8.6 Important GAN Variants

| Variant | Year | Innovation |
|---------|------|-----------|
| **DCGAN** | 2015 | Convolutional architecture, stable training guidelines |
| **WGAN** | 2017 | Wasserstein distance instead of JS divergence  more stable training |
| **Progressive GAN** | 2017 | Grow resolution gradually (4×4 → 8×8 → ... → 1024×1024) |
| **StyleGAN** | 2018 | Style-based generator, incredible face generation |
| **StyleGAN2/3** | 2020-21 | Improved quality and reduced artifacts |
| **CycleGAN** | 2017 | Unpaired image-to-image translation (horse↔zebra) |
| **Pix2Pix** | 2016 | Paired image-to-image translation (sketch→photo) |
| **BigGAN** | 2018 | Large-scale class-conditional generation |

### 8.7 Conditional GANs

Standard GANs generate random outputs. **Conditional GANs** let you control what's generated:

```
CONDITIONAL GAN:

 Random noise z ──────┐
                      ├──► Generator ──► Fake image of a cat
 Label "cat" ─────────┘

 Image ───────────────┐
                      ├──► Discriminator ──► Real/Fake + correct class?
 Label "cat" ─────────┘
```

### 8.8 Modern Relevance (2026)

- **Largely superseded by diffusion models** for image generation (better quality, more stable training)
- StyleGAN3 still used for some **face generation** and **face editing** tasks
- **Pix2Pix / CycleGAN** concepts live on in paired/unpaired style transfer
- GAN discriminator concept used as a component in some diffusion model training (adversarial diffusion)
- Still relevant in **video game** asset generation and **data augmentation**
- GAN-based **super-resolution** (Real-ESRGAN) still widely used in 2026
- **Historical importance:** GANs proved that neural networks could generate photorealistic images, paving the way for the generative AI revolution

---

## 9. Diffusion Models

### 9.1 Why Were Diffusion Models Created?

**The Problem:** GANs produce stunning images but are hard to train (mode collapse, instability). VAEs are stable but produce blurry images. Can we get the best of both worlds  stable training AND high-quality generation?

**The Insight (Sohl-Dickstein et al., 2015; Ho et al., 2020):** Instead of learning to generate images in one shot, learn to **gradually remove noise** from a completely noisy image. This is inspired by thermodynamic diffusion  the process of particles spreading out until they reach equilibrium.

**Analogy:** Imagine you have a beautiful sand sculpture on a beach. Every hour, the wind scatters a tiny bit of sand randomly. After 1000 hours, it's just a flat pile of random sand. Now, if you could **record** exactly how the wind scattered the sand at each step, you could **reverse the process**  starting from random sand and gradually rebuilding the sculpture. A diffusion model learns to reverse the noise process.

### 9.2 The Two Processes

```
FORWARD PROCESS (adding noise  no learning needed):

Original Image          Slightly noisy       More noisy         Pure noise
     x₀           →        x₁          →       x₂      → ... →    x_T

┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
│  🏔️     │        │  🏔️+ε₁   │       │  🏔️+ε₂  │       │ ░░░░░░░░ │
│ 🌲🌲🌊  │  ───► │ 🌲+ε 🌊  │  ───► │ ░🌲░░░   │  ───► │ ░░░░░░░░ │
│  🏠🏠   │       │ 🏠+ε🏠   │       │ ░░░░░░   │       │ ░░░░░░░░ │
└──────────┘       └──────────┘       └──────────┘       └──────────┘

q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)

Each step adds a small amount of Gaussian noise (controlled by schedule β_t)
After T steps (typically T=1000), the image is indistinguishable from pure Gaussian noise.


REVERSE PROCESS (removing noise  THIS is what the neural network learns):

Pure noise           Less noisy          Even less noisy     Clean image!
    x_T         →       x_{T-1}     →      x_{T-2}    → ... →    x₀

┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
│ ░░░░░░░░ │       │ ░░░?░░░  │       │ ░🏔️░░   │       │  🏔️     │
│ ░░░░░░░░ │  ───► │ ░░░░░░░  │  ───► │ 🌲░🌊░   │  ───► │ 🌲🌲🌊  │
│ ░░░░░░░░ │       │ ░░░░░░░  │       │ ░🏠░░░   │       │  🏠🏠   │
└──────────┘       └──────────┘       └──────────┘       └──────────┘

p_θ(x_{t-1} | x_t)  Neural network predicts the noise to remove
```

### 9.3 What Does the Network Actually Learn?

The neural network (typically a **U-Net**) doesn't predict the clean image directly. It predicts **the noise** that was added at each step:

```
TRAINING:

1. Take clean image x₀
2. Sample random time step t ~ Uniform(1, T)
3. Sample noise ε ~ N(0, I)
4. Create noisy image: x_t = √(ᾱ_t) · x₀ + √(1-ᾱ_t) · ε
5. Train network to predict: ε_θ(x_t, t) ≈ ε

Loss = ||ε - ε_θ(x_t, t)||²    (simple MSE!)

┌────────────┐
│ Noisy      │──┐
│ image x_t  │  │    ┌──────────────┐     ┌─────────────┐
└────────────┘  ├───►│              │     │             │
                │    │   U-Net      │────►│ Predicted   │
┌────────────┐  │    │   ε_θ(x_t,t) │     │ noise ε̂    │
│ Time step  │──┘    │              │     │             │
│     t      │       └──────────────┘     └──────┬──────┘
└────────────┘                                   │
                                                 │ Compare with
                                                 ▼
                                          ┌─────────────┐
                                          │ Actual noise │
                                          │     ε        │
                                          └─────────────┘
```

### 9.4 The U-Net Architecture

The workhorse of diffusion models is the **U-Net**  named for its U-shaped architecture:

```
U-NET ARCHITECTURE:

Input                                                          Output
(noisy image)                                                  (predicted noise)
    │                                                              ▲
    ▼                                                              │
┌────────┐                                                    ┌────────┐
│64×64×64│─────────────────── Skip Connection ───────────────►│64×64×64│
└───┬────┘                                                    └────▲───┘
    │ ↓downsample                                        upsample↑ │
┌───▼────┐                                                    ┌────┴───┐
│32×32   │─────────────────── Skip Connection ───────────────►│32×32   │
│  ×128  │                                                    │  ×128  │
└───┬────┘                                                    └────▲───┘
    │ ↓                                                          ↑ │
┌───▼────┐                                                    ┌────┴───┐
│16×16   │─────────────────── Skip Connection ───────────────►│16×16   │
│  ×256  │                                                    │  ×256  │
└───┬────┘                                                    └────▲───┘
    │ ↓                                                          ↑ │
    │              ┌──────────────────┐                            │
    └─────────────►│    BOTTLENECK    │────────────────────────────┘
                   │   8×8 × 512     │
                   │ + time embedding│
                   │ + text embedding│
                   └──────────────────┘
```

**Skip connections** concatenate encoder features with decoder features, preserving fine-grained details. The **time step t** is embedded and injected into each layer so the network knows how much noise is present.

### 9.5 How Stable Diffusion / DALL-E Work

Modern text-to-image systems don't run diffusion directly on pixel space (too expensive for high-resolution images). Instead, they use **Latent Diffusion Models (LDMs)**:

```
LATENT DIFFUSION (Stable Diffusion Architecture):

"A cat wearing                     ┌──────────────────┐
 a top hat"                        │   TEXT ENCODER    │
      │                            │   (CLIP / T5)    │
      └───────────────────────────►│                  │
                                   └────────┬─────────┘
                                            │ text embeddings
                                            ▼
                              ┌──────────────────────────┐
  Random noise ──────────────►│                          │
  in LATENT space             │    U-NET (Denoiser)      │
  (64×64×4, not 512×512×3!)   │    Runs T denoising     │
                              │    steps in latent       │◄── Noise schedule
                              │    space                 │
                              └────────────┬─────────────┘
                                           │ clean latent
                                           ▼
                              ┌──────────────────────────┐
                              │    VAE DECODER           │
                              │    (latent → pixels)     │
                              │    64×64×4 → 512×512×3   │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │              │
                                    │  Final Image │
                                    │  512×512     │
                                    │              │
                                    └──────────────┘
```

**Key components:**

1. **Text Encoder (CLIP or T5):** Converts text prompt to numerical embeddings
2. **VAE Encoder/Decoder:** Compresses images to/from a smaller latent space (64×64×4 instead of 512×512×3 = **48× fewer values**)
3. **U-Net:** Performs the iterative denoising in latent space, conditioned on text embeddings
4. **Noise Scheduler:** Controls how much noise to remove at each step

### 9.6 Classifier-Free Guidance

How does the model ensure the output matches the text prompt? Through **classifier-free guidance**:

```
During training:
  - Randomly drop the text condition (replace with ∅ "null" prompt) 10-20% of the time
  - This teaches the model to generate both conditional and unconditional outputs

During generation:
  noise_pred = unconditional_pred + guidance_scale × (conditional_pred - unconditional_pred)

  guidance_scale = 1.0  → Ignores the prompt
  guidance_scale = 7.5  → Good balance (typical default)
  guidance_scale = 20.0 → Follows prompt very strictly (but may be oversaturated)
```

### 9.7 DALL-E 3 and Modern Systems (2026)

By 2026, diffusion-based systems have evolved significantly:

| System | Key Features |
|--------|-------------|
| **Stable Diffusion 3 / SDXL** | Open-source, latent diffusion, community ecosystem |
| **DALL-E 3** | Deep integration with ChatGPT, prompt rewriting for better results |
| **Midjourney v6+** | Aesthetic quality focus, proprietary |
| **Imagen 3** (Google) | High text rendering quality, T5-XXL text encoder |
| **Flux** | Uses flow matching (a diffusion variant), transformer-based denoiser (DiT replacing U-Net) |
| **Video models** | Sora, Veo, Kling  extend diffusion to video generation |

**Recent trends (2025-2026):**
- **Diffusion Transformers (DiT):** Replacing U-Net with Transformer architecture
- **Flow Matching:** A cleaner mathematical framework for the diffusion process
- **Consistency Models:** Generate in fewer steps (1-4 instead of 20-50)
- **Rectified Flow:** Straighter sampling paths for faster generation
- **ControlNet:** Add spatial control (pose, depth, edges) to the generation process

### 9.8 Strengths, Weaknesses, and Modern Relevance (2026)

**Strengths:**
- **Best image quality** of any generative approach
- **Stable training**  simple MSE loss, no adversarial dynamics
- **Mode coverage**  doesn't suffer from mode collapse like GANs
- **Flexible conditioning**  can condition on text, images, poses, depth maps, etc.
- **Mathematically principled**  grounded in score-based generative modeling

**Weaknesses:**
- **Slow generation**  requires many iterative denoising steps (though improving rapidly)
- **Computationally expensive**  both training and inference
- **Can hallucinate details**  may generate plausible-looking but incorrect content
- **Text rendering**  historically weak at generating legible text in images (improving)

**Modern Relevance (2026):**
- **Dominant paradigm** for image, video, and audio generation
- Powers most commercial image generation tools
- Extended to **3D generation**, **video**, and **audio/music**
- Central to the creative AI revolution
- Active research in making generation faster and more controllable

---

## 10. Architecture Comparison Table

| Architecture | Year | Data Type | Generative? | Parallel? | Key Strength | Key Weakness | 2026 Status |
|---|---|---|---|---|---|---|---|
| **CNN** | 1989/2012 | Spatial (images) | No | Yes | Spatial pattern detection | Limited global context | Active (edge, hybrid) |
| **RNN** | 1986 | Sequential | Can be | No | Simple, sequential | Vanishing gradients | Mostly obsolete |
| **LSTM** | 1997 | Sequential | Can be | No | Long-range memory | Slow, complex | Niche (time-series) |
| **GRU** | 2014 | Sequential | Can be | No | Simpler than LSTM | Same limitations | Niche (time-series) |
| **Autoencoder** | 1986 | Any | Limited | Yes | Unsupervised features | Can only reconstruct | Active (interpretability) |
| **VAE** | 2013 | Any | Yes | Yes | Smooth latent space | Blurry outputs | Active (as component) |
| **GAN** | 2014 | Any | Yes | Yes | Sharp outputs | Training instability | Declining |
| **Diffusion** | 2020 | Any | Yes | Partially | Best quality, stable | Slow generation | Dominant |

---

## 11. Quiz

### Questions

**Q1:** In a CNN, what is the output size when applying a 3×3 kernel (stride=1, no padding) to a 7×7 input?

**Q2:** Why do LSTMs solve the vanishing gradient problem that plagues simple RNNs? Explain in terms of the cell state.

**Q3:** What is mode collapse in GANs, and why doesn't it occur in diffusion models?

**Q4:** In a VAE, why do we encode inputs as distributions (μ, σ) rather than single points? What would happen if we used a regular autoencoder for generation?

**Q5:** Modern text-to-image diffusion models (like Stable Diffusion) don't run the diffusion process directly on pixel space. What do they do instead, and why?

---

### Answers

**A1:** Using the formula: Output = (Input - Kernel + 2×Padding) / Stride + 1 = (7 - 3 + 0) / 1 + 1 = **5×5**. The kernel can be placed in 5 positions horizontally and 5 positions vertically.

**A2:** LSTMs solve vanishing gradients through the **cell state** (c_t), which has an **additive** update path: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t. When the forget gate f_t ≈ 1, the cell state is preserved almost unchanged, and gradients flow through multiplication by ~1 (instead of being repeatedly squashed by tanh as in simple RNNs). This creates a "gradient highway" that allows information and gradients to persist over hundreds of time steps.

**A3:** **Mode collapse** occurs when a GAN generator discovers a single output (or small set of outputs) that consistently fools the discriminator, and it stops exploring other modes of the data distribution. It produces limited variety. **Diffusion models** don't suffer from this because they're trained with a simple reconstruction objective (predicting noise) rather than an adversarial game. They learn to model the entire data distribution through the denoising process, covering all modes naturally. There's no discriminator to "trick" with a single solution.

**A4:** Encoding as distributions creates a **smooth, continuous latent space** with no gaps. The KL divergence loss pushes all distributions toward a standard normal, ensuring that every point in latent space decodes to something meaningful. With a regular autoencoder, the latent space has "holes"  regions between encoded points that don't correspond to any valid data. If you randomly sample from such a space, you'd likely get garbage outputs. VAEs guarantee that you can sample anywhere near the origin and get a valid output.

**A5:** They use **Latent Diffusion Models (LDMs)**. First, a VAE encoder compresses the image from pixel space (e.g., 512×512×3 = 786,432 values) to a much smaller latent space (e.g., 64×64×4 = 16,384 values  a 48× reduction). The diffusion process (iterative denoising) runs entirely in this latent space, then the VAE decoder converts the clean latent back to pixels. This is done because running diffusion in full pixel space would be **prohibitively expensive**  the U-Net would need to process ~50× more data at each of the many denoising steps. Working in latent space makes training and inference dramatically faster while maintaining image quality.


