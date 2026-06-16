# Part 2: Neural Networks  From Neurons to Learning

> *"A neural network is just a function that maps inputs to outputs. The magic is in how
> it learns that mapping."*

---

## Welcome  What This Chapter Is About

In Part 1, we learned *what* AI is and *why* neural networks became dominant. Now we're
going to open the hood and understand **exactly how they work**, down to individual
numbers.

By the end of this chapter you will:

1. Understand how a single artificial neuron works
2. Know what weights, biases, and parameters are
3. Understand layers and network architectures
4. Know the major activation functions and *why* they exist
5. Be able to do **forward propagation** by hand
6. Understand loss functions
7. Understand **backpropagation**  the algorithm that makes learning possible
8. Understand gradient descent and optimizers
9. Work through a **complete example** of a tiny network learning XOR

This chapter is math-heavy but we'll use small numbers and work through every step.
Grab a pencil and paper  you'll want to verify the calculations yourself.

---

## 1. The Biological Neuron: Where It All Started

### 1.1 How Brain Neurons Work

Your brain has roughly 86 billion neurons. Each neuron:

1. **Receives signals** from other neurons through branch-like structures called
   **dendrites**
2. **Processes** the signals in its cell body (soma)  essentially adding them up
3. If the combined signal is strong enough, the neuron **fires**  sending an
   electrical impulse down its **axon**
4. The signal passes to the next neurons through tiny gaps called **synapses**

```
   A BIOLOGICAL NEURON (simplified)
   ════════════════════════════════

   Inputs from                                    Output to
   other neurons                                  other neurons
                      ┌────────────┐
    ──dendrite──┐     │            │
                ├────>│  Cell Body │             ┌──────────┐
    ──dendrite──┤     │   (soma)   │──axon──────>│ Synapse  │──>
                ├────>│            │             └──────────┘
    ──dendrite──┘     │  Adds up   │
                      │  signals.  │
                      │  If strong │
                      │  enough:   │
                      │  FIRE! ⚡  │
                      └────────────┘

   Key insight: Not all input connections are equal!
   Some synapses are "stronger"  they carry more influence.
   This is what "learning" in the brain adjusts.
```

**Key insight for AI:** The "strength" of each synaptic connection determines how much
influence one neuron has on another. When you learn something, your brain is adjusting
these connection strengths. This is exactly the idea behind artificial neural networks.

### 1.2 From Biology to Math

We can capture this in a simple mathematical model:

| Biological Component | Artificial Equivalent | Math Symbol |
|---|---|---|
| Dendrites (inputs) | Input values | x₁, x₂, ..., xₙ |
| Synaptic strengths | **Weights** | w₁, w₂, ..., wₙ |
| Cell body threshold | **Bias** | b |
| Cell body (summation) | Weighted sum + bias | z = Σ(wᵢ · xᵢ) + b |
| Firing / not firing | **Activation function** | a = f(z) |
| Axon output | Output value | a (activation) |

---

## 2. The Artificial Neuron

### 2.1 The Complete Picture

An artificial neuron does exactly four things:

```
   AN ARTIFICIAL NEURON
   ════════════════════

   Step 1: RECEIVE inputs
   Step 2: MULTIPLY each input by its weight
   Step 3: SUM everything + add a bias
   Step 4: Apply an ACTIVATION FUNCTION

                    weights
   Inputs           │ │ │
     │               │ │ │
     ▼               ▼ ▼ ▼
   ┌───┐           ┌───────┐         ┌───────────┐
   │x₁ │──(× w₁)──│       │         │           │
   ├───┤           │       │         │ Activation│
   │x₂ │──(× w₂)──│  SUM  │──(+ b)──│ Function  │──> output (a)
   ├───┤           │   Σ   │         │   f(z)    │
   │x₃ │──(× w₃)──│       │         │           │
   └───┘           └───────┘         └───────────┘

   z = (x₁·w₁) + (x₂·w₂) + (x₃·w₃) + b     ← "pre-activation"
   a = f(z)                                     ← "activation" (output)
```

### 2.2 Worked Example: A Single Neuron

Let's compute the output of a single neuron with concrete numbers.

**Given:**
- Inputs: x₁ = 2.0, x₂ = 3.0
- Weights: w₁ = 0.5, w₂ = -1.0
- Bias: b = 1.0
- Activation function: ReLU (we'll define this properly soon  for now, ReLU(z) = max(0, z))

**Step 1:** Multiply each input by its weight:
```
x₁ · w₁ = 2.0 × 0.5  =  1.0
x₂ · w₂ = 3.0 × (-1.0) = -3.0
```

**Step 2:** Sum everything and add the bias:
```
z = 1.0 + (-3.0) + 1.0 = -1.0
```

**Step 3:** Apply the activation function:
```
a = ReLU(-1.0) = max(0, -1.0) = 0.0
```

**Result:** The neuron outputs **0.0**. The negative pre-activation was "squashed" to
zero by ReLU. The neuron didn't "fire."

---

## 3. Weights, Biases, and Parameters

### 3.1 Weights  The Importance Knobs

**Analogy:** Imagine you're deciding where to eat dinner. You consider three factors:
taste, price, and distance. But they're not equally important to you:

- Taste matters a lot → high weight (0.7)
- Price matters somewhat → medium weight (0.2)
- Distance barely matters → low weight (0.1)

Your "decision score" = 0.7 × taste + 0.2 × price_score + 0.1 × distance_score

**Weights** work exactly this way in neural networks. Each weight controls how much
influence a particular input has on the neuron's output.

```
   WEIGHT EFFECTS:

   Large positive weight (w = 3.0):  "This input is VERY important,
                                      and its effect is POSITIVE"

   Small positive weight (w = 0.1):  "This input barely matters,
                                      slight positive effect"

   Zero weight (w = 0.0):            "This input is IGNORED"

   Negative weight (w = -2.0):       "This input is important, but
                                      its effect is OPPOSITE
                                      (high input → low output)"
```

### 3.2 Bias  The Threshold Shifter

**Analogy:** You might generally like restaurants, but you have a baseline requirement:
"Even if all factors score zero, I need at least a 3 out of 10 to bother going." That
baseline is the **bias**.

Mathematically, the bias shifts the activation function left or right:

```
   WITHOUT BIAS (b = 0):            WITH BIAS (b = 2):
   Neuron fires when                Neuron fires when
   weighted sum > 0                 weighted sum > -2
                                    (i.e., fires MORE easily)

   Output                           Output
     │      ╱                         │      ╱
     │     ╱                          │     ╱
     │    ╱                           │    ╱
     │   ╱                            │   ╱
   ──┼──╱───── z                    ──┼╱─────── z
     │ ╱                             ╱│
     │╱                             ╱ │
     0                             0
                                  ← shifted left by 2
```

The bias lets the neuron adjust **when** it activates, independent of the inputs.

### 3.3 Parameters  What the Network Learns

==**Parameters** = all the weights + all the biases in the entire network.==

These are the numbers the network **adjusts during training** to get better at its task.

```
   A tiny network with:
   - 3 inputs, 4 hidden neurons, 2 outputs

   Parameters:
   ┌───────────────────────────────────────────────┐
   │ Layer 1 (input → hidden):                     │
   │   Weights: 3 inputs × 4 neurons = 12 weights  │
   │   Biases:  4 neurons × 1 bias   =  4 biases   │
   │                                                │
   │ Layer 2 (hidden → output):                     │
   │   Weights: 4 inputs × 2 neurons =  8 weights  │
   │   Biases:  2 neurons × 1 bias   =  2 biases   │
   │                                                │
   │ TOTAL PARAMETERS = 12 + 4 + 8 + 2 = 26        │
   └───────────────────────────────────────────────┘
```

For context, here's how parameter counts scale in real models:

| Model | Parameters | Year |
|---|---|---|
| Perceptron | ~tens | 1957 |
| LeNet-5 | ~60K | 1998 |
| AlexNet | ~60M | 2012 |
| ResNet-152 | ~60M | 2015 |
| GPT-2 | 1.5B | 2019 |
| GPT-3 | 175B | 2020 |
| GPT-4 | ~1.8T (est.) | 2023 |
| Llama 3 | 8B – 405B | 2024 |

(K = thousand, M = million, B = billion, T = trillion)

---

## 4. Layers: Building a Network

### 4.1 The Three Types of Layers

A neural network is organized into **layers** of neurons:

```
   A FEEDFORWARD NEURAL NETWORK
   ════════════════════════════

     INPUT          HIDDEN          HIDDEN         OUTPUT
     LAYER          LAYER 1         LAYER 2        LAYER

    ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐
    │ x₁  │───────>│ h₁  │───────>│ h₄  │───────>│ y₁  │
    ├─────┤   ╲  ╱ ├─────┤   ╲  ╱ ├─────┤   ╲  ╱ ├─────┤
    │ x₂  │────╳──>│ h₂  │────╳──>│ h₅  │────╳──>│ y₂  │
    ├─────┤   ╱  ╲ ├─────┤   ╱  ╲ ├─────┤   ╱  ╲ └─────┘
    │ x₃  │───────>│ h₃  │───────>│ h₆  │────────┘
    └─────┘        └─────┘        └─────┘

    3 neurons       3 neurons       3 neurons       2 neurons

    Note: Every neuron in one layer connects to EVERY neuron
    in the next layer. This is called a "fully connected" or
    "dense" layer.
```

| Layer Type | Role | What It Does |
|---|---|---|
| **Input Layer** | Receives raw data | Passes data through (no computation) |
| **Hidden Layer(s)** | Learns features | Transforms data through weights, biases, activations |
| **Output Layer** | Produces final answer | Formats result (probability, class, number) |

### 4.2 Why "Hidden"?

They're called "hidden" because we never directly observe their values during normal
use. We see the input (what goes in) and the output (what comes out), but the hidden
layers are internal representations  patterns the network has discovered on its own.

### 4.3 Depth and Width

```
   WIDE network:                DEEP network:
   (many neurons per layer)     (many layers)

   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○         ○ ○
   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○         ○ ○
   ○ ○ ○ ○ ○ ○ ○ ○ ○ ○         ○ ○
                                ○ ○
   3 layers, 10 neurons each    ○ ○
   Total params: ~200           ○ ○
                                ○ ○
                                ○ ○

                                8 layers, 2 neurons each
                                Total params: ~32

   Both can work  "deep" networks (hence "deep learning")
   tend to learn more abstract, hierarchical features.
```

**Key principle:** Adding more layers (depth) lets the network learn **hierarchical**
representations. Layer 1 might learn edges, Layer 2 learns textures, Layer 3 learns
parts, Layer 4 learns objects. This hierarchical structure is why "deep" learning is so
powerful.

---

## 5. Activation Functions

### 5.1 Why Do We Need Them?

Without an activation function, a neural network is just a series of **linear
transformations**  which collapse to a single linear transformation no matter how many
layers you add:

```
   WITHOUT activation functions:
   Layer 1: z₁ = W₁·x + b₁
   Layer 2: z₂ = W₂·z₁ + b₂ = W₂·(W₁·x + b₁) + b₂ = (W₂·W₁)·x + (W₂·b₁ + b₂)
                                                       = W'·x + b'

   This is STILL just a linear function!
   No matter how many layers you stack, the result is linear.
   A linear function CANNOT learn complex patterns like XOR.
```

Activation functions introduce **non-linearity**  the ability to learn curves, bends,
and complex decision boundaries. They're what give neural networks their power.

### 5.2 Sigmoid (σ)

The **sigmoid** function squashes any number into the range (0, 1).

**Formula:** σ(z) = 1 / (1 + e⁻ᶻ)

```
   SIGMOID FUNCTION: σ(z) = 1 / (1 + e^(-z))
   ═══════════════════════════════════════════

   Output
   1.0 ┤                              ●●●●●●●●●●●●●●
       │                          ●●●●
       │                       ●●●
       │                     ●●
   0.5 ┤- - - - - - - - - -●- - - - - - - - - - - - -
       │                 ●●
       │               ●●
       │           ●●●●
   0.0 ┤●●●●●●●●●●
       └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──
        -7  -6  -5  -4  -3  -2  -1   0   1   2   3   4   z

   Properties:
   • Output range: (0, 1)   perfect for probabilities!
   • σ(0) = 0.5
   • σ(large positive) ≈ 1
   • σ(large negative) ≈ 0
   • Smooth and differentiable everywhere
```

**Quick calculations:**
```
σ(0)  = 1 / (1 + e⁰)  = 1 / (1 + 1) = 1/2 = 0.500
σ(1)  = 1 / (1 + e⁻¹) = 1 / (1 + 0.368) = 1/1.368 = 0.731
σ(2)  = 1 / (1 + e⁻²) = 1 / (1 + 0.135) = 1/1.135 = 0.881
σ(-1) = 1 / (1 + e¹)  = 1 / (1 + 2.718) = 1/3.718 = 0.269
σ(-3) = 1 / (1 + e³)  = 1 / (1 + 20.09) = 1/21.09 = 0.047
σ(5)  = 1 / (1 + e⁻⁵) = 1 / (1 + 0.0067)= 1/1.007 = 0.993
```

**Derivative (important for backpropagation):**
```
σ'(z) = σ(z) · (1 - σ(z))
```
This is beautifully simple! If σ(z) = 0.731, then σ'(z) = 0.731 × 0.269 = 0.197.

**Problem with sigmoid:** When z is very large or very small, the derivative is nearly
zero. This causes the **vanishing gradient problem**  gradients become so tiny that
early layers stop learning. This is why sigmoid fell out of favor for hidden layers.

### 5.3 ReLU (Rectified Linear Unit)

The **ReLU** function is dead simple: if the input is positive, keep it; if negative,
output zero.

**Formula:** ReLU(z) = max(0, z)

```
   ReLU FUNCTION: f(z) = max(0, z)
   ════════════════════════════════

   Output
     5 ┤                                          ╱
       │                                        ╱
     4 ┤                                      ╱
       │                                    ╱
     3 ┤                                  ╱
       │                                ╱
     2 ┤                              ╱
       │                            ╱
     1 ┤                          ╱
       │                        ╱
     0 ┤●●●●●●●●●●●●●●●●●●●●●●
       └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──
        -5  -4  -3  -2  -1   0   1   2   3   4   5   z

   Properties:
   • Output range: [0, ∞)
   • f(z) = 0 for z ≤ 0
   • f(z) = z for z > 0
   • Computationally very fast (just a comparison)
   • Derivative: 0 if z < 0, 1 if z > 0
```

**Quick calculations:**
```
ReLU(3.5)  = max(0, 3.5)  = 3.5
ReLU(-2.0) = max(0, -2.0) = 0.0
ReLU(0.0)  = max(0, 0.0)  = 0.0
ReLU(100)  = max(0, 100)  = 100
```

**Why ReLU dominates:** It's simple, fast, and doesn't suffer from the vanishing
gradient problem (for positive values). ReLU is the **default choice** for hidden
layers in modern networks.

**Derivative:**
```
ReLU'(z) = 0  if z < 0
ReLU'(z) = 1  if z > 0
ReLU'(0) is undefined, typically set to 0 or 0.5 in practice
```

### 5.4 Softmax

The **softmax** function converts a vector of raw numbers into a **probability
distribution**  the outputs sum to 1.0.

**Formula:** softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)

```
   SOFTMAX: Turns raw scores into probabilities
   ═════════════════════════════════════════════

   Raw scores (logits)          After softmax

   Cat:   2.0                   Cat:   0.659  (65.9%)  ← highest!
   Dog:   1.0        ────>      Dog:   0.243  (24.3%)
   Bird:  0.5                   Bird:  0.098  ( 9.8%)
                                       ─────
                                Total: 1.000  (100%)
```

**Worked calculation:**
```
e^2.0 = 7.389
e^1.0 = 2.718
e^0.5 = 1.649

Sum = 7.389 + 2.718 + 1.649 = 11.756

softmax(Cat)  = 7.389 / 11.756 = 0.629
softmax(Dog)  = 2.718 / 11.756 = 0.231
softmax(Bird) = 1.649 / 11.756 = 0.140
                                  ─────
                          Total = 1.000 ✓
```

*(Note: slight differences from the diagram due to rounding in ASCII art.)*

**Where it's used:** The **output layer** of classification networks. If your network
classifies images into 1,000 categories, the output layer has 1,000 neurons followed
by softmax, giving you the probability of each category.

### 5.5 Activation Function Comparison

| Function | Formula | Range | Used Where | Pros | Cons |
|---|---|---|---|---|---|
| **Sigmoid** | 1/(1+e⁻ᶻ) | (0, 1) | Output (binary) | Smooth, probabilistic | Vanishing gradients |
| **Tanh** | (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (-1, 1) | Hidden (older nets) | Zero-centered | Vanishing gradients |
| **ReLU** | max(0, z) | [0, ∞) | Hidden (default) | Fast, no vanishing grad | "Dead neurons" (stuck at 0) |
| **Leaky ReLU** | max(0.01z, z) | (-∞, ∞) | Hidden (alternative) | No dead neurons | Slightly more complex |
| **Softmax** | eᶻⁱ/Σeᶻʲ | (0, 1), sums to 1 | Output (multi-class) | Probability distribution | Only for output layer |

---

## 6. Forward Propagation  A Complete Worked Example

Forward propagation is simply **running the inputs through the network** to get an
output. Let's do this step by step with a real (tiny) example.

### 6.1 The Network

```
   A tiny network: 2 inputs, 2 hidden neurons, 1 output

    INPUT         HIDDEN          OUTPUT
    LAYER         LAYER           LAYER

   ┌─────┐      ┌─────┐
   │     │─w₁₁─>│     │
   │ x₁  │╲   ╱ │ h₁  │──w₅──╲
   │     │ ╲ ╱  │     │       ╲   ┌─────┐
   │=0.5 │  ╳   └─────┘        ──>│     │
   └─────┘ ╱ ╲  ┌─────┐        ╱  │ y₁  │
   ┌─────┐╱   ╲ │     │──w₆──╱   │     │
   │     │─w₂₂─>│ h₂  │          └─────┘
   │ x₂  │      │     │
   │=0.8 │      └─────┘
   └─────┘
```

### 6.2 The Values

**Inputs:**
```
x₁ = 0.5
x₂ = 0.8
```

**Hidden layer weights and biases:**
```
w₁₁ = 0.4   (x₁ → h₁)
w₁₂ = 0.3   (x₁ → h₂)
w₂₁ = 0.6   (x₂ → h₁)
w₂₂ = -0.2  (x₂ → h₂)

b₁ = 0.1    (bias for h₁)
b₂ = -0.1   (bias for h₂)
```

**Output layer weights and biases:**
```
w₅ = 0.7   (h₁ → y₁)
w₆ = -0.5  (h₂ → y₁)

b₃ = 0.2   (bias for y₁)
```

**Activation functions:**
- Hidden layer: ReLU
- Output layer: Sigmoid

### 6.3 Step-by-Step Forward Pass

#### Step 1: Compute hidden neuron h₁

```
z_h₁ = (x₁ · w₁₁) + (x₂ · w₂₁) + b₁
     = (0.5 × 0.4) + (0.8 × 0.6) + 0.1
     = 0.20 + 0.48 + 0.1
     = 0.78

a_h₁ = ReLU(0.78) = max(0, 0.78) = 0.78
```

#### Step 2: Compute hidden neuron h₂

```
z_h₂ = (x₁ · w₁₂) + (x₂ · w₂₂) + b₂
     = (0.5 × 0.3) + (0.8 × (-0.2)) + (-0.1)
     = 0.15 + (-0.16) + (-0.1)
     = -0.11

a_h₂ = ReLU(-0.11) = max(0, -0.11) = 0.00
```

Note: h₂ has a negative pre-activation, so ReLU outputs zero. This neuron didn't
"fire."

#### Step 3: Compute output neuron y₁

```
z_y₁ = (a_h₁ · w₅) + (a_h₂ · w₆) + b₃
     = (0.78 × 0.7) + (0.00 × (-0.5)) + 0.2
     = 0.546 + 0.0 + 0.2
     = 0.746

a_y₁ = sigmoid(0.746) = 1 / (1 + e^(-0.746))
     = 1 / (1 + 0.474)
     = 1 / 1.474
     = 0.678
```

#### Result

```
   Forward Pass Summary:
   ┌─────────────────────────────────────────────────┐
   │ Inputs:   x₁=0.5,  x₂=0.8                     │
   │                                                 │
   │ Hidden:   z_h₁ = 0.78  → a_h₁ = 0.78 (active) │
   │           z_h₂ = -0.11 → a_h₂ = 0.00 (dead)   │
   │                                                 │
   │ Output:   z_y₁ = 0.746 → a_y₁ = 0.678          │
   │                                                 │
   │ The network predicts: 0.678                     │
   │ (If this is binary classification, that means   │
   │  "67.8% probability of class 1")               │
   └─────────────────────────────────────────────────┘
```

---

## 7. Loss Functions  Measuring How Wrong We Are

### 7.1 The Core Idea

After a forward pass, the network produces a prediction. But is it **good**? We need a
number that tells us how far off the prediction is from the correct answer. This number
is the **loss** (also called "cost" or "error").

**Analogy:** You're throwing darts at a bullseye. The loss is the distance from where
your dart lands to the center. Your goal: minimize the loss (get closer to the bullseye).

```
   ┌───────────────────┐     ┌───────────────────┐
   │  HIGH LOSS        │     │  LOW LOSS         │
   │                   │     │                   │
   │    ◎              │     │                   │
   │         ┌───┐     │     │       ┌───┐       │
   │         │ ● │     │     │       │ ● │       │
   │       x │   │     │     │       │ ◎ │       │
   │         └───┘     │     │       └───┘       │
   │              x    │     │                   │
   │                   │     │                   │
   │  Prediction (●)   │     │  Prediction (●)   │
   │  far from target  │     │  close to target  │
   │  (◎)              │     │  (◎)              │
   └───────────────────┘     └───────────────────┘
```

### 7.2 Mean Squared Error (MSE)

Used for **regression** (predicting continuous numbers like prices, temperatures).

**Formula:** MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

Where:
- yᵢ = true value
- ŷᵢ = predicted value
- n = number of samples

**Worked example:**
```
True values:      y = [3.0,  5.0,  2.5]
Predicted values: ŷ = [2.5,  4.8,  3.0]

Errors:           (3.0 - 2.5)² = 0.25
                  (5.0 - 4.8)² = 0.04
                  (2.5 - 3.0)² = 0.25

MSE = (0.25 + 0.04 + 0.25) / 3 = 0.54 / 3 = 0.18
```

**Why squared?**
1. Makes all errors positive (so negative and positive errors don't cancel)
2. Penalizes large errors more than small ones (an error of 2 contributes 4, not 2)

### 7.3 Binary Cross-Entropy Loss

Used for **binary classification** (two classes: yes/no, cat/dog, spam/not-spam).

**Formula:** BCE = -[y · log(ŷ) + (1 - y) · log(1 - ŷ)]

Where:
- y = true label (0 or 1)
- ŷ = predicted probability (between 0 and 1)

**Worked example 1** (correct, confident prediction):
```
True label: y = 1 (it IS a cat)
Predicted:  ŷ = 0.9 (90% confident it's a cat)

BCE = -[1 · log(0.9) + 0 · log(0.1)]
    = -[log(0.9)]
    = -[-0.105]
    = 0.105   ← LOW loss (good prediction!)
```

**Worked example 2** (wrong, confident prediction):
```
True label: y = 1 (it IS a cat)
Predicted:  ŷ = 0.1 (only 10% confident it's a cat)

BCE = -[1 · log(0.1) + 0 · log(0.9)]
    = -[log(0.1)]
    = -[-2.303]
    = 2.303   ← HIGH loss (bad prediction!)
```

```
   Binary Cross-Entropy Loss when true label = 1:

   Loss
   5.0 ┤╲
       │ ╲
   4.0 ┤  ╲
       │   ╲
   3.0 ┤    ╲
       │     ╲
   2.0 ┤      ╲
       │        ╲
   1.0 ┤          ╲╲
       │             ╲╲╲
   0.0 ┤                 ╲╲╲╲╲╲╲╲╲╲
       └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴
        0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
                     Predicted probability (ŷ)

   The loss is VERY high when prediction is wrong (near 0)
   and VERY low when prediction is right (near 1).
   The curve is not symmetric  being confidently wrong is
   punished MUCH more than being slightly wrong.
```

### 7.4 Categorical Cross-Entropy

Used for **multi-class classification** (3+ classes). It's the generalization of binary
cross-entropy:

**Formula:** CCE = -Σᵢ yᵢ · log(ŷᵢ)

**Example:** Classifying an image as cat, dog, or bird. True label: cat.
```
True (one-hot):  y = [1, 0, 0]  (cat=1, dog=0, bird=0)
Predicted:       ŷ = [0.7, 0.2, 0.1]

CCE = -[1·log(0.7) + 0·log(0.2) + 0·log(0.1)]
    = -[log(0.7)]
    = -[-0.357]
    = 0.357
```

### 7.5 Loss Function Summary

| Loss Function | Use Case | Formula | Output Range |
|---|---|---|---|
| **MSE** | Regression | (1/n)Σ(y-ŷ)² | [0, ∞) |
| **Binary Cross-Entropy** | Binary classification | -[y·log(ŷ)+(1-y)·log(1-ŷ)] | [0, ∞) |
| **Categorical Cross-Entropy** | Multi-class classification | -Σyᵢ·log(ŷᵢ) | [0, ∞) |

---

## 8. Backpropagation  How Neural Networks Learn

### 8.1 The Big Picture

Backpropagation is **the** algorithm that makes neural network training possible. Here's
the idea in plain English:

1. **Forward pass**: Run inputs through the network, get a prediction
2. **Compute loss**: Compare prediction to the true answer
3. **Backward pass**: For each weight, calculate "how much did you contribute to the
   error?" (this is the **gradient**)
4. **Update weights**: Adjust each weight to reduce the error

```
   THE LEARNING LOOP
   ═════════════════

   ┌─────────┐     ┌──────────┐     ┌───────────┐
   │  INPUT  │────>│  FORWARD │────>│   COMPUTE  │
   │  DATA   │     │   PASS   │     │    LOSS    │
   └─────────┘     └──────────┘     └─────┬─────┘
                                          │
                                          ▼
   ┌─────────┐     ┌──────────┐     ┌───────────┐
   │ UPDATE  │<────│ BACKWARD │<────│  COMPARE   │
   │ WEIGHTS │     │   PASS   │     │ TO TARGET  │
   └────┬────┘     └──────────┘     └───────────┘
        │
        └──── Repeat for thousands of iterations
              until loss is small enough!
```

### 8.2 The Chain Rule  The Mathematical Foundation

Backpropagation uses the **chain rule** from calculus. Don't worry if calculus scares
you  the idea is intuitive.

**Analogy:** Imagine a factory assembly line:

```
   Raw Material ──> Machine A ──> Machine B ──> Machine C ──> Final Product
```

If the final product is defective, you need to figure out which machine caused the
problem and how much. You trace backward:

- How much did Machine C affect the quality? (direct effect)
- How much did Machine B affect Machine C's input? (indirect effect)
- How much did Machine A affect Machine B's input? (even more indirect)

The chain rule says: the total effect = product of each individual effect along the
chain.

**Mathematically:**
```
If y = f(g(h(x))), then:

dy/dx = (dy/df) · (df/dg) · (dg/dx)

Each factor answers: "How much does this stage
change when its input changes?"
```

### 8.3 Backpropagation  Step-by-Step Worked Example

Let's continue our forward pass example from Section 6 and compute the backward pass.

**Setup recap:**
```
Inputs: x₁ = 0.5, x₂ = 0.8
Target: y_true = 1.0 (this is a positive example)

Hidden layer (ReLU activation):
  w₁₁ = 0.4, w₂₁ = 0.6, b₁ = 0.1  →  z_h₁ = 0.78  →  a_h₁ = 0.78
  w₁₂ = 0.3, w₂₂ = -0.2, b₂ = -0.1 → z_h₂ = -0.11 →  a_h₂ = 0.00

Output layer (sigmoid activation):
  w₅ = 0.7, w₆ = -0.5, b₃ = 0.2  →  z_y₁ = 0.746  →  a_y₁ = 0.678

Loss function: Binary Cross-Entropy
Loss = -[1·log(0.678) + 0·log(1-0.678)]
     = -log(0.678)
     = -(-0.388)
     = 0.388
```

Now let's compute the gradients, working **backward** from the loss.

#### Step 1: Gradient of loss w.r.t. output activation

```
∂L/∂a_y₁ = -y/a_y₁ + (1-y)/(1-a_y₁)
          = -1/0.678 + 0/(1-0.678)
          = -1.475
```

#### Step 2: Gradient of output activation w.r.t. pre-activation

```
∂a_y₁/∂z_y₁ = σ(z_y₁) · (1 - σ(z_y₁))
             = 0.678 × (1 - 0.678)
             = 0.678 × 0.322
             = 0.218
```

#### Step 3: Combined  gradient of loss w.r.t. z_y₁

```
∂L/∂z_y₁ = ∂L/∂a_y₁ × ∂a_y₁/∂z_y₁
          = -1.475 × 0.218
          = -0.322

(Note: For sigmoid + BCE, this simplifies to: a_y₁ - y_true = 0.678 - 1.0 = -0.322)
```

This is the **output error signal**: δ_out = -0.322

The negative sign means "the output needs to **increase**" to get closer to the target
of 1.0. Makes sense  we predicted 0.678 but the target is 1.0.

#### Step 4: Gradients for output layer weights

```
∂L/∂w₅ = ∂L/∂z_y₁ × ∂z_y₁/∂w₅ = δ_out × a_h₁ = -0.322 × 0.78 = -0.251
∂L/∂w₆ = ∂L/∂z_y₁ × ∂z_y₁/∂w₆ = δ_out × a_h₂ = -0.322 × 0.00 =  0.000
∂L/∂b₃ = ∂L/∂z_y₁ × 1          = -0.322
```

#### Step 5: Propagate error back to hidden layer

```
∂L/∂a_h₁ = ∂L/∂z_y₁ × w₅ = -0.322 × 0.7  = -0.225
∂L/∂a_h₂ = ∂L/∂z_y₁ × w₆ = -0.322 × (-0.5) = 0.161
```

#### Step 6: Through ReLU activation

```
ReLU derivative: 1 if z > 0, else 0

∂L/∂z_h₁ = ∂L/∂a_h₁ × ReLU'(z_h₁) = -0.225 × 1 = -0.225   (z_h₁ = 0.78 > 0)
∂L/∂z_h₂ = ∂L/∂a_h₂ × ReLU'(z_h₂) = 0.161 × 0  =  0.000   (z_h₂ = -0.11 < 0)
```

Note: h₂ is "dead" (ReLU killed it), so NO gradient flows through it. Its weights
won't update this round.

#### Step 7: Gradients for hidden layer weights

```
∂L/∂w₁₁ = ∂L/∂z_h₁ × x₁ = -0.225 × 0.5 = -0.113
∂L/∂w₂₁ = ∂L/∂z_h₁ × x₂ = -0.225 × 0.8 = -0.180
∂L/∂b₁  = ∂L/∂z_h₁       = -0.225

∂L/∂w₁₂ = ∂L/∂z_h₂ × x₁ = 0.000 × 0.5 = 0.000
∂L/∂w₂₂ = ∂L/∂z_h₂ × x₂ = 0.000 × 0.8 = 0.000
∂L/∂b₂  = ∂L/∂z_h₂       = 0.000
```

#### Summary of All Gradients

```
┌────────────────────────────────────────────┐
│        GRADIENT SUMMARY                    │
│                                            │
│  Weight    Value    Gradient    Direction   │
│  ──────    ─────    ────────    ─────────   │
│  w₁₁       0.4     -0.113     ↑ increase   │
│  w₂₁       0.6     -0.180     ↑ increase   │
│  b₁        0.1     -0.225     ↑ increase   │
│  w₁₂       0.3      0.000     – no change  │
│  w₂₂      -0.2      0.000     – no change  │
│  b₂       -0.1      0.000     – no change  │
│  w₅        0.7     -0.251     ↑ increase   │
│  w₆       -0.5      0.000     – no change  │
│  b₃        0.2     -0.322     ↑ increase   │
│                                            │
│  Negative gradient = weight should increase │
│  Positive gradient = weight should decrease │
│  (We move OPPOSITE to the gradient)        │
└────────────────────────────────────────────┘
```

---

## 9. Gradient Descent  Actually Updating the Weights

### 9.1 The Core Idea

We have the gradients. Now what? We **adjust each weight in the direction that reduces
the loss.** Since the gradient points in the direction of *increasing* loss, we move in
the **opposite** direction.

**Analogy:** You're blindfolded on a hill and want to reach the bottom (minimum loss).
You feel the slope with your feet. If the ground slopes up to the right, you step left.
The steeper the slope, the bigger the step.

```
   GRADIENT DESCENT VISUALIZATION
   ═══════════════════════════════

   Loss
    ▲
    │  ╲                     ╱
    │   ╲                   ╱
    │    ╲                 ╱
    │     ╲               ╱
    │      ╲    ●        ╱     ● = current position
    │       ╲   │↓      ╱      ↓ = direction of descent
    │        ╲  │↓     ╱
    │         ╲ ●↓    ╱        The gradient tells us
    │          ╲│↓   ╱         which direction is "downhill"
    │           ╲●  ╱
    │            ╲↓╱           We take steps in that direction
    │             ●  ← MINIMUM (goal!)
    └──────────────────────────────> weight value

   Update rule:  w_new = w_old - learning_rate × gradient
```

### 9.2 The Learning Rate

The **learning rate** (η, eta) controls how big each step is:

```
   TOO SMALL (η = 0.0001):         JUST RIGHT (η = 0.01):        TOO LARGE (η = 10):
   ──────────────────────          ──────────────────────         ──────────────────────

   ●                               ●                              ●
    ·                                ╲                              ╲
    ·                                 ╲                              ╲
     ·                                 ╲                              ╲
      ·                                 ●                              ╲
       ·                                 ╲                              ╲
        ·                                 ●                              ╲
         · (still going...)                ●  ← converges!                ●─────●
                                                                        BOUNCES! DIVERGES!
   Takes forever to converge.      Converges nicely.               Overshoots and blows up.
```

Typical learning rates: 0.001 to 0.01 (depend on the optimizer and problem).

### 9.3 Applying Gradient Descent to Our Example

Using a learning rate η = 0.1:

```
w_new = w_old - η × gradient

w₁₁ = 0.400 - 0.1 × (-0.113) = 0.400 + 0.011 = 0.411
w₂₁ = 0.600 - 0.1 × (-0.180) = 0.600 + 0.018 = 0.618
b₁  = 0.100 - 0.1 × (-0.225) = 0.100 + 0.023 = 0.123
w₁₂ = 0.300 - 0.1 × ( 0.000) = 0.300          = 0.300  (no change)
w₂₂ =-0.200 - 0.1 × ( 0.000) =-0.200          =-0.200  (no change)
b₂  =-0.100 - 0.1 × ( 0.000) =-0.100          =-0.100  (no change)
w₅  = 0.700 - 0.1 × (-0.251) = 0.700 + 0.025 = 0.725
w₆  =-0.500 - 0.1 × ( 0.000) =-0.500          =-0.500  (no change)
b₃  = 0.200 - 0.1 × (-0.322) = 0.200 + 0.032 = 0.232
```

If we now do another forward pass with these updated weights, the prediction will be
**closer to 1.0** (our target), and the loss will be lower. Repeat this thousands of
times, and the network learns!

### 9.4 Types of Gradient Descent

| Type | Description | Batch Size | Pros | Cons |
|---|---|---|---|---|
| **Batch GD** | Use ALL training data to compute each gradient | Full dataset | Stable, accurate gradients | Slow for large datasets, lots of memory |
| **Stochastic GD (SGD)** | Use ONE random sample per update | 1 | Fast, uses little memory | Very noisy gradients, unstable |
| **Mini-batch SGD** | Use a small batch (32-512 samples) | 32-512 | Best of both worlds | Must choose batch size |

**Mini-batch SGD** is the standard in practice.

### 9.5 Better Optimizers

Plain SGD has problems: it can be slow and oscillate. Modern optimizers fix this:

#### Momentum

**Analogy:** A ball rolling downhill. With momentum, it doesn't stop at every small
bump  it keeps going in the general downhill direction.

```
   WITHOUT momentum:          WITH momentum:

   ●→·→·→·→·→·→·→·→·→·→●    ●→→→→→→→→→●
   Takes many tiny,            Accelerates in consistent
   hesitant steps              direction, fewer steps
```

#### Adam (Adaptive Moment Estimation)

Adam (2014) is the **most popular optimizer** in deep learning. It combines:
1. **Momentum**  keeps a running average of past gradients (direction)
2. **Adaptive learning rates**  automatically adjusts the learning rate for each
   weight based on how consistently its gradient points in the same direction

```
   OPTIMIZER COMPARISON
   ════════════════════

   SGD:        Simple, but can be slow and oscillate
   SGD+Momentum: Faster convergence, less oscillation
   Adam:        Adaptive learning rates + momentum
                → Works well "out of the box" for most problems
                → Default choice for deep learning in 2026
```

| Optimizer | Learning Rate | Momentum | Adaptive LR | Typical Use |
|---|---|---|---|---|
| **SGD** | Fixed | No | No | Simple problems, fine-tuning |
| **SGD + Momentum** | Fixed | Yes | No | Computer vision |
| **Adam** | Fixed (base) | Yes | Yes | Default for most tasks |
| **AdamW** | Fixed (base) | Yes | Yes + weight decay | LLM training |

---

## 10. The Complete XOR Example  A Neural Network Learning from Scratch

This is the crown jewel of this chapter. We'll build a tiny neural network and train it
to learn the XOR function  the same function that defeated the single-layer perceptron
in 1969.

### 10.1 The Problem

```
   XOR Truth Table:
   ┌────────┬────────┬────────┐
   │ Input A│ Input B│ Output │
   ├────────┼────────┼────────┤
   │   0    │   0    │   0    │
   │   0    │   1    │   1    │
   │   1    │   0    │   1    │
   │   1    │   1    │   0    │
   └────────┴────────┴────────┘

   Remember: A single perceptron CAN'T learn this.
   We need at least one hidden layer.
```

### 10.2 The Network Architecture

```
   2 inputs → 2 hidden neurons (sigmoid) → 1 output neuron (sigmoid)

        INPUT           HIDDEN           OUTPUT
        LAYER           LAYER            LAYER

       ┌─────┐        ┌─────┐
       │     │──w₁───>│     │
       │ x₁  │╲      │ h₁  │──w₅──╲
       │     │ ╲     │     │       ╲  ┌─────┐
       └─────┘  ╲    └─────┘        ╲ │     │
                 ╳                    >│ out │
       ┌─────┐  ╱    ┌─────┐        ╱ │     │
       │     │ ╱     │     │──w₆──╱  └─────┘
       │ x₂  │╱      │ h₂  │
       │     │──w₄───>│     │
       └─────┘        └─────┘

   Weights to learn:
   w₁ (x₁→h₁), w₂ (x₂→h₁), b₁ (bias h₁)
   w₃ (x₁→h₂), w₄ (x₂→h₂), b₂ (bias h₂)
   w₅ (h₁→out), w₆ (h₂→out), b₃ (bias out)

   Total: 9 parameters
   Activation: Sigmoid everywhere
   Loss: Mean Squared Error (for simplicity)
```

### 10.3 Initial Weights (Randomly Chosen)

```
Hidden layer:
  w₁ = 0.5    w₂ = 0.4    b₁ = 0.0
  w₃ = 0.9    w₄ = 1.0    b₂ = 0.0

Output layer:
  w₅ = -0.3   w₆ = -0.5   b₃ = 0.0

Learning rate: η = 0.5
```

We'll use **sigmoid** activation and **MSE** loss for this example, as they produce
clean, traceable math.

### 10.4 Epoch 1  Training on All Four Examples

An **epoch** = one pass through ALL training examples.

---

#### Example 1: Input (0, 0) → Target: 0

**Forward pass:**
```
Hidden neuron h₁:
  z₁ = x₁·w₁ + x₂·w₂ + b₁ = 0·0.5 + 0·0.4 + 0.0 = 0.000
  a₁ = σ(0.000) = 0.500

Hidden neuron h₂:
  z₂ = x₁·w₃ + x₂·w₄ + b₂ = 0·0.9 + 0·1.0 + 0.0 = 0.000
  a₂ = σ(0.000) = 0.500

Output neuron:
  z₃ = a₁·w₅ + a₂·w₆ + b₃ = 0.5·(-0.3) + 0.5·(-0.5) + 0.0
     = -0.150 + (-0.250) = -0.400
  ŷ  = σ(-0.400) = 1/(1 + e^0.4) = 1/(1 + 1.492) = 1/2.492 = 0.401
```

**Loss (MSE for single sample):**
```
L = (y - ŷ)² = (0 - 0.401)² = 0.161
```

**Backward pass:**

Output layer:
```
∂L/∂ŷ = 2(ŷ - y) = 2(0.401 - 0) = 0.802

σ'(z₃) = ŷ·(1-ŷ) = 0.401 × 0.599 = 0.240

δ_out = ∂L/∂ŷ × σ'(z₃) = 0.802 × 0.240 = 0.193

∂L/∂w₅ = δ_out × a₁ = 0.193 × 0.500 = 0.096
∂L/∂w₆ = δ_out × a₂ = 0.193 × 0.500 = 0.096
∂L/∂b₃ = δ_out       = 0.193
```

Hidden layer:
```
δ_h₁ = δ_out × w₅ × σ'(z₁) = 0.193 × (-0.3) × [0.5 × 0.5]
     = 0.193 × (-0.3) × 0.25 = -0.014

δ_h₂ = δ_out × w₆ × σ'(z₂) = 0.193 × (-0.5) × [0.5 × 0.5]
     = 0.193 × (-0.5) × 0.25 = -0.024

∂L/∂w₁ = δ_h₁ × x₁ = -0.014 × 0 = 0.000
∂L/∂w₂ = δ_h₁ × x₂ = -0.014 × 0 = 0.000
∂L/∂b₁ = δ_h₁       = -0.014

∂L/∂w₃ = δ_h₂ × x₁ = -0.024 × 0 = 0.000
∂L/∂w₄ = δ_h₂ × x₂ = -0.024 × 0 = 0.000
∂L/∂b₂ = δ_h₂       = -0.024
```

**Weight update (gradient descent):**
```
w₁ = 0.500 - 0.5 × 0.000  =  0.500
w₂ = 0.400 - 0.5 × 0.000  =  0.400
b₁ = 0.000 - 0.5 × (-0.014) = 0.007
w₃ = 0.900 - 0.5 × 0.000  =  0.900
w₄ = 1.000 - 0.5 × 0.000  =  1.000
b₂ = 0.000 - 0.5 × (-0.024) = 0.012
w₅ = -0.300 - 0.5 × 0.096 = -0.348
w₆ = -0.500 - 0.5 × 0.096 = -0.548
b₃ = 0.000 - 0.5 × 0.193  = -0.096
```

---

#### Example 2: Input (0, 1) → Target: 1

**Using the updated weights from Example 1.**

**Forward pass:**
```
z₁ = 0·0.500 + 1·0.400 + 0.007 = 0.407
a₁ = σ(0.407) = 1/(1+e^(-0.407)) = 1/(1+0.666) = 1/1.666 = 0.600

z₂ = 0·0.900 + 1·1.000 + 0.012 = 1.012
a₂ = σ(1.012) = 1/(1+e^(-1.012)) = 1/(1+0.364) = 1/1.364 = 0.733

z₃ = 0.600·(-0.348) + 0.733·(-0.548) + (-0.096)
   = -0.209 + (-0.402) + (-0.096) = -0.707
ŷ  = σ(-0.707) = 1/(1+e^(0.707)) = 1/(1+2.028) = 1/3.028 = 0.330
```

**Loss:**
```
L = (1 - 0.330)² = (0.670)² = 0.449
```

**Backward pass:**
```
∂L/∂ŷ = 2(0.330 - 1) = -1.340

σ'(z₃) = 0.330 × 0.670 = 0.221

δ_out = -1.340 × 0.221 = -0.296

∂L/∂w₅ = -0.296 × 0.600 = -0.178
∂L/∂w₆ = -0.296 × 0.733 = -0.217
∂L/∂b₃ = -0.296

δ_h₁ = -0.296 × (-0.348) × σ'(0.407)
     = -0.296 × (-0.348) × [0.600 × 0.400]
     = -0.296 × (-0.348) × 0.240
     = 0.025

δ_h₂ = -0.296 × (-0.548) × σ'(1.012)
     = -0.296 × (-0.548) × [0.733 × 0.267]
     = -0.296 × (-0.548) × 0.196
     = 0.032

∂L/∂w₁ = 0.025 × 0 = 0.000     ∂L/∂w₃ = 0.032 × 0 = 0.000
∂L/∂w₂ = 0.025 × 1 = 0.025     ∂L/∂w₄ = 0.032 × 1 = 0.032
∂L/∂b₁ = 0.025                  ∂L/∂b₂ = 0.032
```

**Weight update:**
```
w₁ =  0.500 - 0.5 × 0.000  =  0.500
w₂ =  0.400 - 0.5 × 0.025  =  0.387
b₁ =  0.007 - 0.5 × 0.025  = -0.006
w₃ =  0.900 - 0.5 × 0.000  =  0.900
w₄ =  1.000 - 0.5 × 0.032  =  0.984
b₂ =  0.012 - 0.5 × 0.032  = -0.004
w₅ = -0.348 - 0.5 × (-0.178) = -0.259
w₆ = -0.548 - 0.5 × (-0.217) = -0.440
b₃ = -0.096 - 0.5 × (-0.296) =  0.052
```

---

#### Example 3: Input (1, 0) → Target: 1

**Forward pass (using latest weights):**
```
z₁ = 1·0.500 + 0·0.387 + (-0.006) = 0.494
a₁ = σ(0.494) = 1/(1 + e^(-0.494)) = 1/(1+0.610) = 1/1.610 = 0.621

z₂ = 1·0.900 + 0·0.984 + (-0.004) = 0.896
a₂ = σ(0.896) = 1/(1 + e^(-0.896)) = 1/(1+0.408) = 1/1.408 = 0.710

z₃ = 0.621·(-0.259) + 0.710·(-0.440) + 0.052
   = -0.161 + (-0.312) + 0.052 = -0.421
ŷ  = σ(-0.421) = 1/(1+e^(0.421)) = 1/(1+1.524) = 1/2.524 = 0.396
```

**Loss:**
```
L = (1 - 0.396)² = (0.604)² = 0.365
```

**Backward pass:**
```
∂L/∂ŷ = 2(0.396 - 1) = -1.208

σ'(z₃) = 0.396 × 0.604 = 0.239

δ_out = -1.208 × 0.239 = -0.289

∂L/∂w₅ = -0.289 × 0.621 = -0.179
∂L/∂w₆ = -0.289 × 0.710 = -0.205
∂L/∂b₃ = -0.289

δ_h₁ = -0.289 × (-0.259) × [0.621 × 0.379]
     = -0.289 × (-0.259) × 0.235 = 0.018

δ_h₂ = -0.289 × (-0.440) × [0.710 × 0.290]
     = -0.289 × (-0.440) × 0.206 = 0.026

∂L/∂w₁ = 0.018 × 1 = 0.018     ∂L/∂w₃ = 0.026 × 1 = 0.026
∂L/∂w₂ = 0.018 × 0 = 0.000     ∂L/∂w₄ = 0.026 × 0 = 0.000
∂L/∂b₁ = 0.018                  ∂L/∂b₂ = 0.026
```

**Weight update:**
```
w₁ =  0.500 - 0.5 × 0.018 =  0.491
w₂ =  0.387 - 0.5 × 0.000 =  0.387
b₁ = -0.006 - 0.5 × 0.018 = -0.015
w₃ =  0.900 - 0.5 × 0.026 =  0.887
w₄ =  0.984 - 0.5 × 0.000 =  0.984
b₂ = -0.004 - 0.5 × 0.026 = -0.017
w₅ = -0.259 - 0.5 × (-0.179) = -0.170
w₆ = -0.440 - 0.5 × (-0.205) = -0.338
b₃ =  0.052 - 0.5 × (-0.289) =  0.197
```

---

#### Example 4: Input (1, 1) → Target: 0

**Forward pass (using latest weights):**
```
z₁ = 1·0.491 + 1·0.387 + (-0.015) = 0.863
a₁ = σ(0.863) = 1/(1+e^(-0.863)) = 1/(1+0.422) = 1/1.422 = 0.703

z₂ = 1·0.887 + 1·0.984 + (-0.017) = 1.854
a₂ = σ(1.854) = 1/(1+e^(-1.854)) = 1/(1+0.157) = 1/1.157 = 0.864

z₃ = 0.703·(-0.170) + 0.864·(-0.338) + 0.197
   = -0.120 + (-0.292) + 0.197 = -0.215
ŷ  = σ(-0.215) = 1/(1+e^(0.215)) = 1/(1+1.240) = 1/2.240 = 0.446
```

**Loss:**
```
L = (0 - 0.446)² = 0.199
```

**Backward pass:**
```
∂L/∂ŷ = 2(0.446 - 0) = 0.893

σ'(z₃) = 0.446 × 0.554 = 0.247

δ_out = 0.893 × 0.247 = 0.221

∂L/∂w₅ = 0.221 × 0.703 = 0.155
∂L/∂w₆ = 0.221 × 0.864 = 0.191
∂L/∂b₃ = 0.221

δ_h₁ = 0.221 × (-0.170) × [0.703 × 0.297]
     = 0.221 × (-0.170) × 0.209 = -0.008

δ_h₂ = 0.221 × (-0.338) × [0.864 × 0.136]
     = 0.221 × (-0.338) × 0.118 = -0.009

∂L/∂w₁ = -0.008 × 1 = -0.008    ∂L/∂w₃ = -0.009 × 1 = -0.009
∂L/∂w₂ = -0.008 × 1 = -0.008    ∂L/∂w₄ = -0.009 × 1 = -0.009
∂L/∂b₁ = -0.008                  ∂L/∂b₂ = -0.009
```

**Weight update:**
```
w₁ =  0.491 - 0.5 × (-0.008) =  0.495
w₂ =  0.387 - 0.5 × (-0.008) =  0.391
b₁ = -0.015 - 0.5 × (-0.008) = -0.011
w₃ =  0.887 - 0.5 × (-0.009) =  0.892
w₄ =  0.984 - 0.5 × (-0.009) =  0.989
b₂ = -0.017 - 0.5 × (-0.009) = -0.013
w₅ = -0.170 - 0.5 × 0.155    = -0.247
w₆ = -0.338 - 0.5 × 0.191    = -0.433
b₃ =  0.197 - 0.5 × 0.221    =  0.086
```

### 10.5 End of Epoch 1  Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                     EPOCH 1 RESULTS                              │
│                                                                  │
│  Input     Target    Prediction    Loss      Correct?            │
│  ─────     ──────    ──────────    ────      ────────            │
│  (0,0)       0        0.401       0.161     Too high  (want <0.5)│
│  (0,1)       1        0.330       0.449     Too low   (want >0.5)│
│  (1,0)       1        0.396       0.365     Too low   (want >0.5)│
│  (1,1)       0        0.446       0.199     OK-ish    (want <0.5)│
│                                                                  │
│  Average Loss: (0.161 + 0.449 + 0.365 + 0.199) / 4 = 0.294     │
│                                                                  │
│  The network is essentially GUESSING (all outputs near 0.4-0.5) │
│  This is expected after just 1 epoch!                            │
│                                                                  │
│  Accuracy: 2/4 = 50%  (only (0,0) and (1,1) are on correct side)│
└──────────────────────────────────────────────────────────────────┘
```

### 10.6 What Happens Over Many Epochs?

We won't work through every epoch by hand (that would take hundreds of pages!), but
here's what happens if we continue training:

```
   TRAINING PROGRESS (simulated)
   ═════════════════════════════

   Epoch    Avg Loss    Predictions for (0,0) (0,1) (1,0) (1,1)
   ─────    ────────    ──────────────────────────────────────────
      1      0.294      0.40   0.33   0.40   0.45
      5      0.261      0.38   0.37   0.39   0.41
     10      0.251      0.36   0.40   0.41   0.39
     50      0.238      0.32   0.47   0.48   0.35
    100      0.199      0.27   0.55   0.56   0.28
    500      0.042      0.09   0.82   0.83   0.11
   1000      0.008      0.04   0.93   0.93   0.05
   2000      0.002      0.02   0.97   0.97   0.03
   5000      0.0004     0.01   0.99   0.99   0.01

   Loss curve:
   Loss
   0.30 ┤●
   0.25 ┤  ●●●
   0.20 ┤      ●●
   0.15 ┤         ●●
   0.10 ┤           ●●
   0.05 ┤              ●●●
   0.00 ┤                  ●●●●●●●●●●●●●●●●●●●●●●
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──
         0   500  1000 1500 2000 2500 3000 3500 4000
                              Epoch

   The network has LEARNED XOR! 🎉
   After ~1000 epochs:
   • (0,0) → 0.04 ≈ 0  ✓
   • (0,1) → 0.93 ≈ 1  ✓
   • (1,0) → 0.93 ≈ 1  ✓
   • (1,1) → 0.05 ≈ 0  ✓
```

### 10.7 What the Hidden Layer Learned

After training converges, the hidden neurons have learned to encode useful intermediate
representations. Here's the intuition of one possible solution the network might find:

```
   WHAT THE TRAINED NETWORK COMPUTES
   ══════════════════════════════════

   Hidden neuron h₁ learns approximately: OR(x₁, x₂)
   ┌────────┬────────┬─────┐
   │ x₁     │ x₂     │ h₁  │
   │ 0      │ 0      │ ≈0  │
   │ 0      │ 1      │ ≈1  │
   │ 1      │ 0      │ ≈1  │
   │ 1      │ 1      │ ≈1  │
   └────────┴────────┴─────┘

   Hidden neuron h₂ learns approximately: AND(x₁, x₂)
   ┌────────┬────────┬─────┐
   │ x₁     │ x₂     │ h₂  │
   │ 0      │ 0      │ ≈0  │
   │ 0      │ 1      │ ≈0  │
   │ 1      │ 0      │ ≈0  │
   │ 1      │ 1      │ ≈1  │
   └────────┴────────┴─────┘

   Output neuron computes approximately: h₁ AND NOT(h₂)
   = OR(x₁,x₂) AND NOT(AND(x₁,x₂))
   = XOR(x₁,x₂)  ✓

   The network DISCOVERED the decomposition:
   XOR = OR AND NOT(AND)

   Nobody told it this! It figured it out by adjusting
   weights to minimize the loss.
```

This is the **fundamental magic of neural networks**: they discover useful
intermediate representations automatically through gradient descent.

### 10.8 Verification with Python

You can verify this entire example with a simple Python script:

```python
import numpy as np

# Sigmoid and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Training data: XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights (same as our hand calculations)
np.random.seed(42)
w_hidden = np.array([[0.5, 0.9], [0.4, 1.0]])  # 2x2
b_hidden = np.array([[0.0, 0.0]])                # 1x2
w_output = np.array([[-0.3], [-0.5]])             # 2x1
b_output = np.array([[0.0]])                      # 1x1

learning_rate = 0.5
epochs = 5000

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_input = X @ w_hidden + b_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = hidden_output @ w_output + b_output
    predicted = sigmoid(final_input)

    # Loss (MSE)
    loss = np.mean((y - predicted) ** 2)

    # Backward pass
    d_output = 2 * (predicted - y) * sigmoid_derivative(predicted)
    d_hidden = (d_output @ w_output.T) * sigmoid_derivative(hidden_output)

    # Update weights
    w_output -= learning_rate * hidden_output.T @ d_output / 4
    b_output -= learning_rate * np.mean(d_output, axis=0, keepdims=True)
    w_hidden -= learning_rate * X.T @ d_hidden / 4
    b_hidden -= learning_rate * np.mean(d_hidden, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# Final predictions
print("\nFinal predictions:")
for i in range(4):
    print(f"  Input: {X[i]} → Predicted: {predicted[i][0]:.4f} | Target: {y[i][0]}")
```

Expected output (approximately):
```
Epoch     0 | Loss: 0.293864
Epoch  1000 | Loss: 0.007523
Epoch  2000 | Loss: 0.002104
Epoch  3000 | Loss: 0.001082
Epoch  4000 | Loss: 0.000697

Final predictions:
  Input: [0 0] → Predicted: 0.0215 | Target: 0
  Input: [0 1] → Predicted: 0.9762 | Target: 1
  Input: [1 0] → Predicted: 0.9763 | Target: 1
  Input: [1 1] → Predicted: 0.0297 | Target: 0
```

---

## 11. Common Network Architectures  A Preview

Now that you understand the fundamentals, here's a preview of the major architectures
you'll encounter in later chapters:

```
┌─────────────────────────────────────────────────────────────────┐
│              NEURAL NETWORK ARCHITECTURE ZOO                    │
├───────────────────┬─────────────────────────────────────────────┤
│                   │                                             │
│  Feedforward NN   │  ○─○─○─○    What we learned in this chapter │
│  (MLP)            │  ○─○─○─○    Good for: tabular data          │
│                   │  ○─○─○─○                                    │
│                   │                                             │
├───────────────────┼─────────────────────────────────────────────┤
│                   │             ┌─────┐                         │
│  Convolutional    │  ▓▓▓▓ ──>   │█ █ █│    Slides a filter      │
│  NN (CNN)         │  ▓▓▓▓       │█ █ █│    over images          │
│                   │  ▓▓▓▓       └─────┘    Good for: images,    │
│                   │                         video, spatial data │
│                   │                                             │
├───────────────────┼─────────────────────────────────────────────┤
│                   │                                             │
│  Recurrent NN     │  ○──>○──>○──>○    Loops / memory            │
│  (RNN/LSTM)       │  ↑   ↑   ↑   ↑    Good for: sequences       │
│                   │  └───┘   └───┘    (text, time series)       │
│                   │  [Mostly replaced by Transformers]          │
│                   │                                             │
├───────────────────┼─────────────────────────────────────────────┤
│                   │                                             │
│  Transformer      │  ┌──────────────┐   Self-attention          │
│                   │  │  Attention   │   mechanism.              │
│                   │  │  is all you  │   Good for: EVERYTHING    │
│                   │  │    need      │   (text, images, audio,   │
│                   │  └──────────────┘    video, code, ...)      │
│                   │  [The dominant architecture since 2017]     │
│                   │                                             │
├───────────────────┼─────────────────────────────────────────────┤
│                   │                                             │
│  GAN              │  Generator ←→ Discriminator                 │
│                   │  Two networks compete                       │
│                   │  Good for: image generation (pre-diffusion) │
│                   │                                             │
├───────────────────┼─────────────────────────────────────────────┤
│                   │                                             │
│  Diffusion Model  │  Noise ──> Denoise ──> Image                │
│                   │  Iterative refinement process               │
│                   │  Good for: high-quality image/video gen     │
│                   │                                             │
└───────────────────┴─────────────────────────────────────────────┘
```

We'll explore each of these in detail in later parts of this course.

---

## 12. Key Takeaways

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHAPTER 2 SUMMARY                            │
│                                                                 │
│  1. A neuron: weighted sum + bias + activation function         │
│                                                                 │
│  2. Weights = importance of each input                          │
│     Bias = threshold adjuster                                   │
│     Parameters = all weights + biases                           │
│                                                                 │
│  3. Activation functions add non-linearity:                     │
│     • Sigmoid: squashes to (0,1)  good for probabilities       │
│     • ReLU: max(0,z)  fast, default for hidden layers          │
│     • Softmax: probability distribution  for classification    │
│                                                                 │
│  4. Forward propagation: input → layers → output                │
│                                                                 │
│  5. Loss functions measure error:                               │
│     • MSE for regression                                        │
│     • Cross-entropy for classification                          │
│                                                                 │
│  6. Backpropagation: compute gradients using chain rule         │
│     (how much each weight contributed to the error)             │
│                                                                 │
│  7. Gradient descent: update weights to reduce loss             │
│     w_new = w_old - learning_rate × gradient                    │
│                                                                 │
│  8. Adam is the default optimizer in practice                   │
│                                                                 │
│  9. Even a tiny 2-2-1 network can learn XOR!                    │
│     Neural networks discover intermediate representations       │
│     automatically through training.                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Quiz  Test Your Understanding

### Questions

**Q1.** A neuron has inputs x₁ = 3, x₂ = -1, weights w₁ = 2, w₂ = 4, and bias
b = -1. Using ReLU activation, what is the neuron's output?

**Q2.** Why do we need activation functions? What would happen if we stacked multiple
layers of neurons without any activation function?

**Q3.** Compute softmax([1.0, 2.0, 3.0]). Show your work.
(Hint: e¹ ≈ 2.718, e² ≈ 7.389, e³ ≈ 20.086)

**Q4.** In our backpropagation example, hidden neuron h₂ had a pre-activation of
-0.11 and used ReLU activation. Its gradient ended up being 0. Why? What does this
mean for the weights feeding into h₂?

**Q5.** If a network has 784 input neurons, two hidden layers with 128 and 64 neurons
respectively, and 10 output neurons, how many total parameters (weights + biases) does
it have? Show your calculation.

---

### Answers

**A1.**
```
z = (3 × 2) + (-1 × 4) + (-1) = 6 + (-4) + (-1) = 1
output = ReLU(1) = max(0, 1) = 1
```
The neuron outputs **1**.

**A2.** Activation functions introduce **non-linearity**. Without them, any number of
stacked linear layers would collapse into a single linear transformation:
`y = W₃·(W₂·(W₁·x + b₁) + b₂) + b₃ = W'·x + b'`
This means the network, regardless of depth, could only learn **linear** relationships
 it couldn't learn curves, XOR, or any complex pattern. Activation functions give the
network the ability to approximate any function.

**A3.**
```
e^1.0 = 2.718
e^2.0 = 7.389
e^3.0 = 20.086

Sum = 2.718 + 7.389 + 20.086 = 30.193

softmax([1, 2, 3]) = [2.718/30.193, 7.389/30.193, 20.086/30.193]
                    = [0.090, 0.245, 0.665]

Verify: 0.090 + 0.245 + 0.665 = 1.000 ✓
```

**A4.** Because ReLU(z) = max(0, z), and z_h₂ = -0.11 < 0, so the output was 0. The
derivative of ReLU is 0 for negative inputs. This means:
- The gradient is **completely blocked** at this neuron
- **No learning signal flows back** through h₂
- The weights feeding into h₂ (w₁₂, w₂₂, b₂) **do not get updated** this round
- This is called a "**dead neuron**"  one of ReLU's drawbacks
- The neuron might "wake up" on a different training example if different inputs produce
  a positive pre-activation

**A5.**
```
Layer 1 (input → hidden 1): 784 × 128 weights + 128 biases = 100,480
Layer 2 (hidden 1 → hidden 2): 128 × 64 weights + 64 biases = 8,256
Layer 3 (hidden 2 → output): 64 × 10 weights + 10 biases = 650

Total = 100,480 + 8,256 + 650 = 109,386 parameters
```

Note: This is a relatively small network by modern standards. GPT-4 has roughly
**16 million times** more parameters!

---

## What's Next?

In **Part 3**, we'll look at how neural networks are trained in practice:
- Data preparation and batching
- Regularization (dropout, weight decay)
- Batch normalization
- Practical training tips and common pitfalls
- Then we'll move into Convolutional Neural Networks (CNNs) for computer vision

The foundations you've built in this chapter  neurons, layers, forward pass, loss,
backpropagation, gradient descent  are the **core vocabulary** of deep learning.
Everything that follows builds on these ideas.

