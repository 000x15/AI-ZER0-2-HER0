# Appendix A: Comprehensive AI Glossary (2026 Edition)

Welcome to the comprehensive glossary. This reference guide contains over 100 terms fundamental to artificial intelligence, machine learning, deep learning, systems engineering, deployment, alignment, and security. It is organized alphabetically for easy navigation.

---

## A

### Activation Function
A mathematical function applied to the output of a neural network node to introduce non-linearity. Without activation functions, a neural network, no matter how many layers it has, behaves exactly like a single-layer linear model. Common activation functions include ReLU, Sigmoid, Tanh, and Softmax.

### Active Parameters
In a Mixture of Experts (MoE) architecture, the subset of total parameters that are actively computed for a specific token. For example, a model might have 671 billion total parameters, but only 37 billion active parameters are used to process any single token, which dramatically reduces the computational cost of inference.

### Adam (Adaptive Moment Estimation)
An optimization algorithm that extends stochastic gradient descent by calculating adaptive learning rates for each parameter. It keeps track of both the first moment (the mean) and the second moment (the uncentered variance) of the gradients, making it the default optimizer for training deep learning and transformer models.

### Adversarial Attack
A technique used to deceive a machine learning model by introducing subtle, often imperceptible perturbations to the input data. In computer vision, an adversarial attack might cause a model to classify a stop sign as a speed limit sign; in LLMs, it can bypass alignment safety guardrails.

### Agentic AI
AI systems designed to operate autonomously by planning tasks, choosing appropriate tools, evaluating their progress, and correcting their actions dynamically to achieve a user-defined goal. Unlike simple conversational chatbots, agents execute continuous observe-think-act loops.

### Alignment
The process of training and adjusting an AI model's behaviors, outputs, and internal objectives to ensure they match human intentions, safety protocols, ethical standards, and operational guidelines.

### API (Application Programming Interface)
In the context of AI, a cloud-hosted interface that allows developer applications to send inputs (prompts) to a model hosted on remote servers and receive the generated outputs (responses) without needing local hardware to run the model.

### Artificial General Intelligence (AGI)
A hypothetical state of AI development where a software system can understand, learn, and apply knowledge to solve any cognitive task that a human can perform, matching or exceeding human performance across all domains.

### Attention Mechanism
A mathematical component in neural network architectures that allows the model to dynamic calculate the relevance of different parts of the input sequence to each other. It enables models to capture long-range dependencies and relational context within data.

### Autoencoder
An unsupervised neural network architecture trained to compress input data into a lower-dimensional representation (the bottleneck or latent space) and then reconstruct the original input from this representation. Used for dimensionality reduction, denoising, and representation learning.

### Autoregressive Model
A model that predicts future values based on past values in a sequence. In NLP, autoregressive language models (like GPT) predict the next token in a sentence given all the preceding tokens, generating text one token at a time.

### AWQ (Activation-aware Weight Quantization)
An optimization technique that quantizes LLM weights while protecting the small percentage of weights that are most critical to model performance (salient weights). It maintains higher accuracy at low bitwidths (e.g., 4-bit) compared to standard quantization methods.

---

## B

### Backpropagation
The core mathematical algorithm used to train neural networks. It calculates the gradient of the loss function with respect to the network's weights and biases by applying the chain rule of calculus from the output layer backward to the input layer.

### Base Model
A large AI model trained on vast amounts of raw, uncurated data (usually text, images, or code) via unsupervised learning. It excels at predicting patterns but lacks conversational alignment, instruction-following capabilities, or safety guardrails.

### Batch Size
The number of training examples processed together in a single forward and backward pass during model training. Larger batch sizes stabilize gradient estimates but require significantly more GPU memory.

### Bias (Statistical)
A systematic error introduced into a machine learning model because of assumptions made during the learning process or skewed representation in the training dataset, leading to unfair or inaccurate predictions.

### Bias (Neural Network Parameter)
An extra parameter added to each neuron in a neural network, alongside weights. It shifts the activation function curve left or right, allowing the neuron to activate even when all input values are zero.

### BPE (Byte-Pair Encoding)
A subword tokenization algorithm that builds a vocabulary by iteratively merging the most frequent pairs of characters or bytes in a corpus. It allows models to handle rare words and prevent out-of-vocabulary errors.

---

## C

### Chain-of-Thought (CoT) Prompting
A technique where an LLM is prompted or trained to generate intermediate reasoning steps before producing a final answer. This dramatically improves performance on complex reasoning, logic, and math tasks.

### Chunking
The process of breaking a large document into smaller, manageable text segments (chunks) prior to generating embeddings for retrieval. Proper chunking is vital for Retrieval-Augmented Generation (RAG) to ensure relevance and fit within model context windows.

### Claude
A family of frontier LLMs developed by Anthropic, known for advanced reasoning, coding capabilities, safety-focused alignment (Constitutional AI), and large context windows.

### Closed-Source Model
An AI model whose weights, training data, and detailed architectures are kept secret by the developer. It is accessed exclusively through managed APIs (e.g., OpenAI's GPT-4o or Anthropic's Claude 4).

### CNN (Convolutional Neural Network)
A deep learning architecture that uses convolution operations (sliding filters) to scan input data. It is highly effective at capturing spatial hierarchies and patterns, making it the foundation of modern computer vision.

### Compute
The processing power (CPU cycles, GPU execution, TPU operations) required to train, fine-tune, or perform inference with machine learning models.

### Constitutional AI
An alignment methodology developed by Anthropic that trains models using a set of written principles (a "constitution"). Instead of relying solely on human feedback, the model critiques and refines its own responses to align with these guidelines.

### Context Window
The maximum number of tokens a model can process, read, and write in a single interaction. In 2026, context windows range from 128,000 tokens to over 2 million tokens.

### Cosine Similarity
A mathematical metric used to measure the similarity of two vectors by calculating the cosine of the angle between them. Commonly used in vector databases to find text chunks related to a user query.

### Cross-Attention
An attention mechanism in encoder-decoder architectures where the decoder queries the encoder's outputs, allowing the generating sequence to look back at the original source sequence (e.g., in machine translation).

### Cross-Entropy Loss
A loss function commonly used in classification tasks and language modeling. It measures the difference between two probability distributions (the model's predicted probabilities and the true target labels).

---

## D

### Data Poisoning
An attack on machine learning models where a malicious actor manipulates the training data (injecting biased, corrupted, or backdoored examples) to compromise the model's behavior after training.

### Decoder-Only Architecture
A transformer architecture that generates text by processing preceding tokens and predicting subsequent ones. This is the standard architecture for modern LLMs (e.g., GPT, Llama, Qwen).

### Deep Learning
A subfield of machine learning focused on training artificial neural networks with multiple hidden layers (hence "deep") to extract high-level representations and features from raw data.

### DeepSeek
A prominent Chinese AI company known for its open-weight Mixture of Experts models (DeepSeek-V3) and reasoning models (DeepSeek-R1) that rival Western frontier models at a fraction of training costs.

### Dense Model
A model in which all parameters are computed for every single input token during training and inference, contrasting with Sparse/Mixture of Experts models.

### Diffusion Model
A class of generative models that works by gradually adding noise to an image (forward process) and learning to reverse this process (denoising/reverse process) to synthesize realistic images or video from random noise.

### Direct Preference Optimization (DPO)
An alignment algorithm that optimizes a language model directly on human preference data without training a separate reward model, offering a simpler, more stable alternative to RLHF.

### Distillation (Knowledge Distillation)
A compression technique where a smaller "student" model is trained to reproduce the behavior and output distribution of a larger, highly capable "teacher" model.

---

## E

### Embedding
A continuous vector representation of data (like words, sentences, or images) in a high-dimensional space. Embeddings capture semantic meaning; words with similar meanings are represented by vectors that are close to each other.

### Encoder-Decoder Architecture
 A transformer architecture that uses an encoder to process an input sequence into a vector representation and a decoder to generate an output sequence (e.g., T5, BART).

### Epoch
A complete pass of the entire training dataset through a machine learning model during the training phase.

### Expert System
An early form of AI (dominant in the 1970s and 1980s) that relies on a database of hardcoded rules (if-then statements) written by human domain experts to solve problems, rather than learning from data.

### ExLlamaV2
An highly optimized inference engine designed to run quantized LLMs (EXL2 format) at high speeds on local consumer GPUs.

---

## F

### Feature Map
The output of a convolution layer in a CNN, representing the spatial distribution of specific detected features (edges, textures, shapes) across the input image.

### Fine-Tuning
The process of taking a pre-trained model and training it further on a smaller, task-specific dataset to adapt it to specific behaviors, domains, or formats.

### FLOPs (Floating-Point Operations)
A measure of computational work. Used to quantify the scale of AI training runs (e.g., training a frontier model in 2026 requires upwards of 10²⁶ FLOPs).

### Forward Propagation
The process where input data travels through the layers of a neural network, undergoing mathematical transformations (multiplication by weights, adding biases, applying activations) to produce a prediction.

### Function Calling
The ability of an LLM to detect when an external tool is required, format the query in JSON or code, and output this structured text so that an application can execute the command and return the result to the model.

---

## G

### GAN (Generative Adversarial Network)
A generative framework consisting of two neural networks: a Generator (which creates synthetic data) and a Discriminator (which evaluates if the data is real or fake), competing against each other in a zero-sum game.

### Gemini
Google's family of natively multimodal frontier models, characterized by massive context windows (up to 2 million tokens) and integration across the Google ecosystem.

### Gemma
Google's family of open-weight models, built on the same research and technology stack as the proprietary Gemini models.

### Generative AI
A branch of artificial intelligence focused on creating new content (text, code, images, audio, video, 3D assets) by learning patterns from massive training datasets.

### GGUF (GPT-Generated Unified Format)
A file format optimized for running LLMs on consumer hardware (especially CPUs and Apple Silicon). It packs the model weights, vocabulary, and metadata into a single file and is designed for fast loading and CPU/GPU split-inference.

### GPU (Graphics Processing Unit)
A specialized electronic circuit designed to rapidly manipulate and alter memory. Its massively parallel architecture makes it far more efficient than a CPU for the matrix multiplications required by neural network training and inference.

### Gradient Descent
The primary optimization algorithm used to minimize the loss function in machine learning. It adjusts model weights step-by-step in the direction opposite to the gradient (the steepest path downhill).

### GRU (Gated Recurrent Unit)
A variant of Recurrent Neural Networks (RNNs) that uses gating mechanisms (reset and update gates) to track long-term dependencies while utilizing a simpler architecture than LSTMs.

### Guardrails
Software mechanisms (such as input filtering, output sanitization, or safety classifiers) wrapped around an AI model to prevent it from processing toxic prompts or generating dangerous, illegal, or biased content.

---

## H

### Hallucination
An occurrence where an LLM generates text that is factually incorrect, nonsensical, or ungrounded in its training data or input context, presenting the information with high confidence.

### Hidden Layer
Any layer in a neural network located between the input layer and the output layer. It performs intermediate mathematical operations to extract abstract representations of the input.

### Hugging Face
An open-source platform and community repository that hosts models, datasets, and libraries (like `transformers` and `diffusers`) for machine learning developers.

---

## I

### Inference
The process of using a trained machine learning model to make predictions or generate outputs for new, unseen input data.

### Instruction Tuning
A phase of fine-tuning where a base model is trained on pairs of natural language instructions and their correct responses, transforming it from a text completion tool into a helpful assistant.

---

## J

### Jailbreaking
The act of crafting specific, adversarial prompts designed to bypass a model's alignment safety guardrails, tricking the model into generating prohibited content.

### JEPA (Joint Embedding Predictive Architecture)
An AI architecture proposed by Yann LeCun that learns representations of data by predicting the representation of parts of the input from other parts, focusing on abstract semantic content rather than pixel-level reconstruction.

---

## K

### KV Cache (Key-Value Cache)
An inference optimization technique that stores the Key and Value matrices of previously processed tokens in GPU memory. This prevents the model from recalculating attention for older tokens at every step of generating new words.

---

## L

### Latent Space
A low-dimensional vector space where compressed representation of data is stored. Points close to each other in latent space share similar underlying characteristics or meanings.

### Layer Normalization
A regularization technique that normalizes the activations of a layer across its features for each training example, stabilizing training and speeding up convergence in transformers.

### Learning Rate
A hyperparameter in gradient descent that determines the step size taken towards the minimum of the loss function at each iteration. Too large a step can cause divergence; too small can make training slow or stuck.

### Llama
Meta's open-weight model family, which catalyzed the open-source AI developer ecosystem since its initial release in 2023.

### LM Studio
A popular desktop application that provides a user-friendly graphical interface for downloading, running, and testing GGUF models locally on consumer computers.

### Local AI
Running machine learning models directly on user devices (laptops, desktops, local servers) without sending data or queries to external cloud servers.

### Long-Context Model
Models designed to ingest and reason over large spans of text (typically 100K+ tokens up to millions), allowing users to upload entire textbooks, code repositories, or hours of audio.

### LoRA (Low-Rank Adaptation)
A parameter-efficient fine-tuning (PEFT) technique that freezes a model's original weights and injects small, trainable low-rank decomposition matrices into the transformer layers, reducing training memory requirements by over 80%.

### Loss Function
A mathematical formula that measures the difference between a model's predicted output and the actual target output, serving as the feedback signal to guide training.

### LSTM (Long Short-Term Memory)
A specialized RNN architecture designed to overcome the vanishing gradient problem by using gate mechanisms (input, forget, output gates) to selectively maintain or discard information over long sequences.

---

## M

### Machine Learning
A branch of AI focused on building systems that learn patterns from data and improve their performance over time without being explicitly programmed.

### MCP (Model Context Protocol)
A standardized, open protocol created by Anthropic that acts as a uniform connection layer between AI models and external data sources or tools (databases, files, APIs).

### Mixture of Experts (MoE)
An architecture design that splits model layers into multiple "experts." A routing network sends each input token to the most qualified experts, allowing models to scale parameter size while keeping active computation cost low.

### Model Poisoning
An adversarial attack where a model's training process or weight matrix is directly manipulated, embedding permanent vulnerabilities or "sleeper agents" that trigger on specific inputs.

### Multi-Head Attention
An attention structure that splits Query, Key, and Value matrices into multiple subspaces (heads), allowing the model to attend to information from different representation spaces and positions simultaneously.

### Multimodal
The capability of an AI model to process, understand, and generate multiple types of data inputs and outputs (e.g., text, images, video, audio).

---

## N

### Next-Token Prediction
The fundamental training objective of autoregressive language models, where the network is trained to predict the next word or subword token in a sequence given the history of preceding tokens.

### NLI (Natural Language Interface)
A user interface that allows users to command systems using everyday natural language (speech or text) instead of rigid menus or CLI syntaxes.

### NPU (Neural Processing Unit)
A specialized silicon accelerator engineered specifically to optimize the mathematical operations (such as low-precision matrix multiplication) used in neural network execution, commonly found in modern laptops and mobile devices.

---

## O

### Ollama
An open-source terminal-based tool that simplifies running, managing, and interacting with local models (in GGUF format) on Windows, macOS, and Linux.

### ONNX (Open Neural Network Exchange)
An open format built to represent machine learning models, allowing developers to transfer models between different frameworks (e.g., PyTorch to TensorFlow) and optimize them for specific hardware backends.

### Open-Weight Model
A model whose trained parameters (weights) are published and free to download. Users can inspect, run, modify, or fine-tune the model locally, though the training data and recipe may remain proprietary.

### Optimization
The process of adjusting a model's parameters (weights and biases) during training to minimize the value calculated by the loss function.

---

## P

### PagedAttention
An memory-management algorithm for LLM inference (originating in vLLM) that borrows concepts from virtual memory in operating systems. It stores the KV cache in non-contiguous physical memory blocks, preventing fragmentation and enabling higher batch sizes.

### Parameter
Any of the internal variables (weights and biases) that a machine learning model learns and updates during its training process.

### Parameter Count
The total number of weights and biases in a model, usually measured in billions (B). It correlates roughly with a model's capability, training cost, and hardware requirements.

### PEFT (Parameter-Efficient Fine-Tuning)
Methods (like LoRA, Prefix Tuning, or QLoRA) designed to fine-tune models by training only a tiny fraction of parameters, keeping the rest frozen to reduce memory and compute costs.

### Positional Encoding
Vectors added to input token embeddings in a transformer model to inject information about the order of words in a sequence, compensating for the transformer's parallel, order-agnostic processing.

### Pretraining
The initial phase of training a neural network on a massive, unstructured dataset to learn foundational representations before applying fine-tuning.

### Prompt Injection
A security vulnerability where a malicious input tricks an LLM into ignoring its developer instructions (system prompt) and executing unintended actions.

### Qwen
Alibaba's family of highly capable, open-weight LLMs, renowned for strong performance in multilingual benchmarks, coding, and mathematical reasoning.

---

## R

### RAG (Retrieval-Augmented Generation)
An architecture pattern that extends an LLM's knowledge by querying an external database for relevant documents matching a user's prompt, appending those documents to the prompt context before sending it to the model.

### ReAct (Reasoning and Acting)
An agent prompting pattern where models generate reasoning traces and task-specific actions in an interleaved manner ("Thought, Action, Observation"), allowing the agent to plan and interact with external APIs.

### Recurrent Neural Network (RNN)
An older neural network architecture designed for sequential data. It processes tokens one by one, keeping a hidden state vector (memory) that is passed forward to the next step.

### Red Teaming (AI)
The practice of systematically testing AI models and applications for safety flaws, security vulnerabilities, toxic outputs, and jailbreaks by simulating real-world attacker behaviors.

### Reinforcement Learning from Human Feedback (RLHF)
A method of aligning LLMs where humans rate model responses. This feedback trains a reward model, which then guides the LLM's training using reinforcement learning algorithms (like PPO) to favor highly rated behaviors.

### ReLU (Rectified Linear Unit)
A simple, computationally efficient activation function defined as $f(x) = \max(0, x)$. It outputs the input if positive, and zero if negative, helping alleviate vanishing gradients in deep networks.

### RLAIF (Reinforcement Learning from AI Feedback)
An alignment method similar to RLHF, but using a high-quality model (AI judge) instead of human evaluators to score outputs and guide the target model's reinforcement learning process.

### RoPE (Rotary Position Embedding)
A modern positional embedding method that encodes relative position information by rotating Query and Key representations in the complex plane, allowing models to scale to longer context lengths more reliably.

---

## T

### Tensor
A multi-dimensional mathematical array of numbers (generalizing scalars, vectors, and matrices) that serves as the fundamental data structure in deep learning.

### TensorRT
An SDK developed by NVIDIA for high-performance deep learning inference, optimizing models to run at maximum speed on NVIDIA GPUs.

### Token
The basic unit of data processed by an LLM. A token can represent a single character, a subword, a word, or punctuation. On average, 100 English words correspond to roughly 130–140 tokens.

### Tokenizer
A preprocessing utility that converts raw text into numerical token IDs that can be ingested by a neural network, and converts token IDs back into text.

### Tool Use (Function Calling)
The process where an AI model identifies that it needs external capabilities, formats a call to an API or tool, and incorporates the tool's output back into its context to complete a task.

### Transformer
The neural network architecture introduced in 2017 ("Attention Is All You Need") based entirely on attention mechanisms, which replaced RNNs and became the foundation of modern Generative AI.

---

## V

### VAE (Variational Autoencoder)
A generative autoencoder variant that maps inputs to probability distributions in latent space rather than single fixed points, allowing it to generate new data by sampling from this distribution.

### Vanishing Gradient Problem
A training challenge in deep networks and RNNs where gradients shrink exponentially as they propagate backward, causing the weights of early layers to stop updating and preventing the network from learning long-term patterns.

### Vector Database
A database optimized for storing, indexing, and querying vector embeddings. It allows applications to perform fast similarity searches across millions of high-dimensional data points.

### vLLM
A fast, easy-to-use library for LLM serving and offline inference, famous for introducing PagedAttention to dramatically increase GPU serving throughput.

### VRAM (Video RAM)
Dedicated high-speed memory on a graphics card (GPU). It holds the model weights, activation values, KV cache, and input tensors during inference and training. In local AI, VRAM is the primary bottleneck.

---

## W

### Weight
A learnable parameter in a neural network layer. It represents the strength of the connection between two neurons, determining how much influence an input signal has on the output.

### World Model
An AI system trained to simulate and predict the physical dynamics, geometry, and temporal progression of a system (or the physical world) to help agents plan actions.

### Word2Vec
An early (2013) algorithm that represents words as dense vectors by training a shallow network to predict words based on their context, laying the groundwork for modern embeddings.

---

## X

### XOR Problem
A classic problem in computer science demonstrating that single-layer perceptrons cannot learn non-linear functions (like the XOR logical operation), which historically led to the development of multi-layer networks with non-linear activation functions.
