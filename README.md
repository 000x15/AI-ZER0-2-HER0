# AI ZER0 2 HERO

<img src="image.jpg" width="200" align="center">




Welcome to the definitive self-study textbook: **"AI ZER0 2 HER0 (2026 Edition)"**. 

This course is designed for technically minded students, software engineers, systems architects, and curious professionals with no prior formal background in artificial intelligence. It starts from first principles and progresses systematically to advanced system designs, deployment architectures, local AI orchestration, and security vectors current to mid-2026.



## 📚 Curriculum Structure

The course is divided into 13 comprehensive parts and 2 appendices. Each chapter contains detailed analogies, mathematical intuition, structural diagrams (ASCII art), tables, and self-assessment quizzes.

### 🌟 Part 1: Foundations
*   **Path**: [part-01-foundations.md](./ai-course/part-01-foundations.md)
*   **Concepts**: AI vs. ML vs. DL vs. GenAI, history of AI, expert systems, statistical learning, neural networks, why connectionism became dominant.

### 🧠 Part 2: Neural Networks
*   **Path**: [part-02-neural-networks.md](./ai-course/part-02-neural-networks.md)
*   **Concepts**: Biological vs. artificial neurons, weights, biases, activations (Sigmoid, ReLU, Softmax), forward propagation, loss functions (MSE, Cross-Entropy), backpropagation, gradient descent, fully worked training example (XOR).

### 📐 Part 3: Deep Learning
*   **Path**: [part-03-deep-learning.md](./ai-course/part-03-deep-learning.md)
*   **Concepts**: CNNs (convolutions, pooling), RNNs, LSTMs (gates), GRUs, Autoencoders, Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Diffusion Models (forward & reverse processes).

### ⚡ Part 4: Transformers
*   **Path**: [part-04-transformers.md](./ai-course/part-04-transformers.md)
*   **Concepts**: Why Transformers replaced RNNs, Embeddings, Positional Encoding (sinusoidal & RoPE), Scaled Dot-Product Attention, Self-Attention, Multi-Head Attention, Encoder vs. Decoder vs. Encoder-Decoder architectures, KV caching.

### 💬 Part 5: Large Language Models (LLMs)
*   **Path**: [part-05-large-language-models.md](./ai-course/part-05-large-language-models.md)
*   **Concepts**: Pretraining (tokenization, scraping, clusters), scaling laws, Supervised Fine-Tuning (SFT), Instruction Tuning, alignment (RLHF, DPO, Constitutional AI), synthetic training data, knowledge distillation.

### 🏛️ Part 6: Modern AI Architectures (2026)
*   **Path**: [part-06-modern-architectures.md](./ai-course/part-06-modern-architectures.md)
*   **Concepts**: Dense vs. Sparse (MoE) models, Retrieval-Augmented Generation (RAG) structures, function calling & tool-use agents (MCP), reasoning models (chain-of-thought, test-time scaling, o1/R1), long-context scaling, multimodal models.

### 🏆 Part 7: Famous Models of 2026
*   **Path**: [part-07-famous-models-2026.md](./ai-course/part-07-famous-models-2026.md)
*   **Concepts**: Comprehensive comparison of GPT, Claude, Gemini, Llama, Qwen, DeepSeek, Mistral, Grok, Phi, Gemma, and Command-R model families. Includes detailed capability tables and cost indices.

### 💻 Part 8: Local AI
*   **Path**: [part-08-local-ai.md](./ai-course/part-08-local-ai.md)
*   **Concepts**: Running models on your device, CPU vs. GPU, VRAM limits, Quantization mechanics (AWQ, GPTQ, GGUF), local execution engines (Ollama, LM Studio, vLLM, llama.cpp), hardware setup profiles (laptop to server).

### 📊 Part 9: Understanding Model Sizes
*   **Path**: [part-09-model-sizes.md](./ai-course/part-09-model-sizes.md)
*   **Concepts**: Tier breakdown (1B, 3B, 7B, 14B, 32B, 70B, 400B+), capabilities vs. hardware profiles, why modern small models outperform legacy giants.

### 🥞 Part 10: Building an AI Stack
*   **Path**: [part-10-building-ai-stack.md](./ai-course/part-10-building-ai-stack.md)
*   **Concepts**: Full stack blueprint (User → Application → Agent → RAG → Vector DB → Inference Engine → Hardware), evaluation, observability, prompt engineering patterns, and safety guardrails.

### 🛡️ Part 11: AI Security
*   **Path**: [part-11-ai-security.md](./ai-course/part-11-ai-security.md)
*   **Concepts**: Prompt injection (direct/indirect), jailbreaking, training data poisoning, model poisoning (backdoors), model extraction, privacy threats, AI red-teaming methodologies, supply-chain exploits.

### 🧪 Part 12: Practical Labs
*   **Path**: [part-12-practical-labs.md](./ai-course/part-12-practical-labs.md)
*   **Concepts**: Hands-on laboratories with fully runnable Python and Shell code:
    1.  Setting up local Ollama & LM Studio.
    2.  Querying local models via APIs.
    3.  Text categorization with Hugging Face Transformers.
    4.  Coding an interactive chatbot.
    5.  Building an end-to-end RAG system with ChromaDB.
    6.  Quantizing models manually with llama.cpp.

### 🚀 Part 13: Future Directions
*   **Path**: [part-13-future-directions.md](./ai-course/part-13-future-directions.md)
*   **Concepts**: Agentic AI evolution, AI OS interfaces (GUI to NLI), world models (JEPA), robotics foundation models, native multimodal reasoning, EU AI Act & global regulations, and speculative horizons (Quantum ML, AGI).

---

## 🗂️ Appendices

### 📖 Appendix A: Glossary
*   **Path**: [appendix-a-glossary.md](./ai-course/appendix-a-glossary.md)
*   **Concepts**: 100+ AI terms compiled and defined alphabetically, spanning mathematics, training, deployment, modern agent protocols, and security vectors.


---

## 🎓 Tips for Success

1.  **Read Sequentially**: The course is built hierarchically. Skip chapters only if you are already comfortable with the prerequisite concepts (e.g., don't read *Part 4: Transformers* without understanding *Part 2: Neural Networks*).
2.  **Follow the Worked Examples**: Grab a piece of paper or a calculator and work through the forward and backpropagation examples in Part 2. Getting your hands dirty with the math demystifies the "magic."
3.  **Run the Labs**: Do not just read the code in Part 12. Install Python, pull Ollama, and run the chatbots and RAG systems locally.
4.  **Take the Quizzes**: Test your knowledge at the end of each section. The questions are designed to check conceptual grasp rather than rote memorization.
