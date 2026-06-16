# Part 12: Practical Labs  Hands-On AI Projects

> *"The best way to learn AI is to build with it."*

---

## Table of Contents

- [Lab 1: Setting Up Your AI Environment](#lab-1-setting-up-your-ai-environment)
- [Lab 2: Running Local Models with Ollama](#lab-2-running-local-models-with-ollama)
- [Lab 3: Using Hugging Face Transformers](#lab-3-using-hugging-face-transformers)
- [Lab 4: Building a Chatbot](#lab-4-building-a-chatbot)
- [Lab 5: Building a RAG System](#lab-5-building-a-rag-system)
- [Lab 6: Fine-Tuning a Small Model](#lab-6-fine-tuning-a-small-model)
- [Lab 7: Quantizing Models](#lab-7-quantizing-models)

---

## Prerequisites

Before starting these labs, you should have:
- A computer with at least **8 GB RAM** (16 GB recommended)
- **Python 3.10+** installed
- Basic familiarity with the command line
- A text editor or IDE (VS Code recommended)
- Labs 2-5 require a GPU with **6+ GB VRAM** for best experience (CPU works but is slower)

---

## Lab 1: Setting Up Your AI Environment

**Goal**: Install all the tools you'll need for the remaining labs.

**Time**: ~30 minutes

### Step 1: Install Python

If you don't already have Python installed:

**Windows**:
```powershell
# Option A: Download from python.org
# Visit https://www.python.org/downloads/ and download Python 3.11+
# IMPORTANT: Check "Add Python to PATH" during installation

# Option B: Install via winget
winget install Python.Python.3.11

# Verify installation
python --version
pip --version
```

**macOS**:
```bash
# Install via Homebrew (install Homebrew first if needed: https://brew.sh)
brew install python@3.11

# Verify installation
python3 --version
pip3 --version
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Verify installation
python3 --version
pip3 --version
```

### Step 2: Create a Virtual Environment

Always use virtual environments to keep your projects isolated:

```bash
# Create a project directory
mkdir ai-labs
cd ai-labs

# Create a virtual environment
python -m venv venv

# Activate it
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.\venv\Scripts\activate.bat

# macOS/Linux:
source venv/bin/activate

# You should see (venv) at the beginning of your prompt
# All pip installs will now go into this virtual environment

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Core Python Libraries

```bash
# Core AI/ML libraries
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install sentence-transformers

# Utilities
pip install requests
pip install python-dotenv
pip install tqdm
pip install rich  # Beautiful terminal output

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

> **Note on PyTorch**: If you have an NVIDIA GPU, install the CUDA version for best performance:
> ```bash
> # For CUDA 12.x (check your CUDA version with: nvidia-smi)
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 4: Install Ollama

Ollama lets you run LLMs locally with a single command.

**Windows**:
```powershell
# Download from https://ollama.com/download/windows
# Run the installer

# Or via winget:
winget install Ollama.Ollama

# Verify installation
ollama --version
```

**macOS**:
```bash
# Download from https://ollama.com/download/mac
# Or via Homebrew:
brew install ollama

# Start the Ollama service
ollama serve
```

**Linux**:
```bash
# One-line install script
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service
sudo systemctl start ollama
# Or manually:
ollama serve
```

### Step 5: Pull Your First Model

```bash
# Pull a small model to test (3.8B parameters, ~2.3GB download)
ollama pull phi3:mini

# Test it
ollama run phi3:mini "What is machine learning in one sentence?"

# You should see a response within a few seconds!
# Type /bye to exit the interactive mode
```

### Step 6: Install LM Studio (Optional GUI)

LM Studio provides a graphical interface for running local models.

1. Download from **https://lmstudio.ai**
2. Install and launch
3. Go to the **Search** tab
4. Search for "phi-3-mini" and download a GGUF version
5. Go to the **Chat** tab and start chatting!

LM Studio also provides a local API server compatible with the OpenAI API format.

### Step 7: Verify Everything Works

Create a file called `verify_setup.py`:

```python
#!/usr/bin/env python3
"""Verify that all lab dependencies are installed correctly."""

import sys
from rich.console import Console
from rich.table import Table

console = Console()

def check_import(module_name: str, display_name: str = None) -> bool:
    """Try to import a module and report success/failure."""
    if display_name is None:
        display_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "NOT INSTALLED"

def main():
    console.print("\n[bold blue]🔍 AI Labs Environment Check[/bold blue]\n")
    
    table = Table(title="Dependency Status")
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version", style="yellow")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("sentence_transformers", "Sentence Transformers"),
        ("requests", "Requests"),
        ("tqdm", "TQDM"),
        ("rich", "Rich"),
    ]
    
    all_ok = True
    for module, name in dependencies:
        success, version = check_import(module, name)
        status = "✅ OK" if success else "❌ Missing"
        if not success:
            all_ok = False
        table.add_row(name, status, version)
    
    console.print(table)
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    console.print(f"\nPython version: {py_version}")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            console.print(f"[green]CUDA available: {torch.cuda.get_device_name(0)}[/green]")
            console.print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            console.print("[yellow]CUDA not available (CPU only - labs will be slower)[/yellow]")
    except Exception:
        pass
    
    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            console.print(f"[green]Ollama running with {len(models)} model(s)[/green]")
            for model in models:
                console.print(f"  - {model['name']}")
        else:
            console.print("[yellow]Ollama running but returned unexpected status[/yellow]")
    except Exception:
        console.print("[yellow]Ollama not running (start with: ollama serve)[/yellow]")
    
    if all_ok:
        console.print("\n[bold green]✅ All core dependencies installed! Ready for labs.[/bold green]\n")
    else:
        console.print("\n[bold red]❌ Some dependencies missing. Install them with pip.[/bold red]\n")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python verify_setup.py
```

> **✅ Checkpoint**: You should see all green checkmarks. If Ollama shows a warning, make sure `ollama serve` is running in another terminal.

---

## Lab 2: Running Local Models with Ollama

**Goal**: Pull models of different sizes, compare them, and use the Ollama API programmatically.

**Time**: ~45 minutes

### Step 1: Explore Available Models

```bash
# List popular models on Ollama
# Visit https://ollama.com/library for the full catalog

# Pull models of different sizes for comparison
ollama pull qwen2.5:0.5b      # 0.5B params - ~400MB - very fast
ollama pull qwen2.5:1.5b      # 1.5B params - ~1GB
ollama pull phi3:mini          # 3.8B params - ~2.3GB
ollama pull llama3.2:3b        # 3B params   - ~2GB
ollama pull llama3.1:8b        # 8B params   - ~4.7GB (needs 8GB+ RAM)

# List your downloaded models
ollama list
```

### Step 2: Interactive Chat

```bash
# Start a chat session with a model
ollama run phi3:mini

# Try these prompts:
# >>> What is the capital of France?
# >>> Write a Python function to calculate Fibonacci numbers
# >>> Explain quantum computing to a 10-year-old
# >>> /bye   (to exit)
```

### Step 3: Compare Model Sizes Programmatically

Create `lab2_compare_models.py`:

```python
#!/usr/bin/env python3
"""
Lab 2: Compare different model sizes on the same prompts.
Measures response quality, speed (tokens/sec), and response length.
"""

import requests
import time
import json
from rich.console import Console
from rich.table import Table

console = Console()

OLLAMA_URL = "http://localhost:11434"

def generate(model: str, prompt: str, temperature: float = 0.7) -> dict:
    """
    Send a generation request to Ollama and measure performance.
    
    Returns a dict with: response text, total time, token count, tokens/sec.
    """
    start_time = time.time()
    
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,          # Get the full response at once
            "options": {
                "temperature": temperature,
                "num_predict": 200,    # Limit response to 200 tokens
            },
        },
        timeout=120,
    )
    
    elapsed = time.time() - start_time
    data = response.json()
    
    # Ollama provides detailed timing in the response
    eval_count = data.get("eval_count", 0)           # Tokens generated
    eval_duration = data.get("eval_duration", 0)      # Nanoseconds spent generating
    prompt_eval_count = data.get("prompt_eval_count", 0)  # Prompt tokens
    
    # Calculate tokens per second
    if eval_duration > 0:
        tokens_per_sec = eval_count / (eval_duration / 1e9)
    else:
        tokens_per_sec = 0
    
    return {
        "response": data.get("response", ""),
        "total_time": elapsed,
        "tokens_generated": eval_count,
        "prompt_tokens": prompt_eval_count,
        "tokens_per_sec": tokens_per_sec,
    }


def check_available_models() -> list:
    """Get list of locally available models."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except Exception as e:
        console.print(f"[red]Error connecting to Ollama: {e}[/red]")
        console.print("Make sure Ollama is running: ollama serve")
        return []


def main():
    console.print("\n[bold blue]🏋️ Lab 2: Model Comparison[/bold blue]\n")
    
    # Check which models are available
    available = check_available_models()
    if not available:
        return
    
    console.print(f"Available models: {', '.join(available)}\n")
    
    # Models to compare (use what's available)
    test_models = []
    preferred_models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "phi3:mini", 
                        "llama3.2:3b", "llama3.1:8b"]
    for model in preferred_models:
        if model in available:
            test_models.append(model)
    
    if len(test_models) < 2:
        console.print("[yellow]Need at least 2 models for comparison.[/yellow]")
        console.print("Pull more models: ollama pull qwen2.5:0.5b")
        # Use whatever is available
        test_models = available[:3]
    
    # Test prompts of varying difficulty
    test_prompts = [
        {
            "name": "Simple Fact",
            "prompt": "What is the capital of Japan? Answer in one sentence.",
        },
        {
            "name": "Reasoning",
            "prompt": "If all cats are animals, and some animals are pets, can we conclude all cats are pets? Explain your reasoning briefly.",
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function called 'is_palindrome' that checks if a string is a palindrome. Include a docstring.",
        },
        {
            "name": "Creative Writing",
            "prompt": "Write a haiku about artificial intelligence.",
        },
    ]
    
    # Run comparisons
    results = {}
    for prompt_info in test_prompts:
        console.print(f"\n[bold cyan]Test: {prompt_info['name']}[/bold cyan]")
        console.print(f"Prompt: {prompt_info['prompt']}\n")
        
        table = Table()
        table.add_column("Model", style="cyan", width=20)
        table.add_column("Tokens/sec", style="green", width=12)
        table.add_column("Time (s)", style="yellow", width=10)
        table.add_column("Tokens", style="magenta", width=8)
        table.add_column("Response", style="white", width=60)
        
        for model in test_models:
            console.print(f"  Testing {model}...", end=" ")
            try:
                result = generate(model, prompt_info["prompt"])
                # Truncate response for display
                display_response = result["response"].strip()[:150]
                if len(result["response"].strip()) > 150:
                    display_response += "..."
                
                table.add_row(
                    model,
                    f"{result['tokens_per_sec']:.1f}",
                    f"{result['total_time']:.2f}",
                    str(result['tokens_generated']),
                    display_response,
                )
                console.print("[green]done[/green]")
            except Exception as e:
                table.add_row(model, "ERROR", "-", "-", str(e)[:60])
                console.print(f"[red]error: {e}[/red]")
        
        console.print(table)
    
    # Summary
    console.print("\n[bold green]📊 Key Takeaways:[/bold green]")
    console.print("• Smaller models are MUCH faster (higher tokens/sec)")
    console.print("• Larger models generally give better quality answers")
    console.print("• For simple tasks, small models may be sufficient")
    console.print("• The sweet spot depends on your use case and hardware")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python lab2_compare_models.py
```

### Step 4: Using the Ollama API with Streaming

Create `lab2_streaming.py`:

```python
#!/usr/bin/env python3
"""
Lab 2 Extra: Demonstrate streaming responses from Ollama.
Watch tokens appear in real-time, just like ChatGPT.
"""

import requests
import json
import sys

OLLAMA_URL = "http://localhost:11434"

def stream_generate(model: str, prompt: str):
    """Stream a response from Ollama, printing tokens as they arrive."""
    
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,  # Enable streaming
        },
        stream=True,         # Tell requests to stream too
        timeout=120,
    )
    
    print(f"\n[{model}]: ", end="", flush=True)
    
    total_tokens = 0
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            print(token, end="", flush=True)  # Print each token as it arrives
            total_tokens += 1
            
            if data.get("done", False):
                # Final message contains stats
                eval_duration = data.get("eval_duration", 0)
                if eval_duration > 0:
                    tokens_per_sec = total_tokens / (eval_duration / 1e9)
                    print(f"\n\n[Stats: {total_tokens} tokens, "
                          f"{tokens_per_sec:.1f} tokens/sec]")
                break
    
    print()  # Final newline


def main():
    model = "phi3:mini"  # Change to any model you have
    
    print("=" * 60)
    print("Ollama Streaming Demo")
    print("=" * 60)
    print(f"Model: {model}")
    print("Type a prompt and watch the response stream in real-time.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            stream_generate(model, prompt)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
```

Run it:
```bash
python lab2_streaming.py
```

> **✅ Checkpoint**: You can now interact with local LLMs programmatically and understand the performance tradeoffs between model sizes.

---

## Lab 3: Using Hugging Face Transformers

**Goal**: Load models from Hugging Face and perform text generation, sentiment analysis, and named entity recognition.

**Time**: ~45 minutes

### Step 1: Install Dependencies

```bash
pip install transformers accelerate sentencepiece protobuf
```

### Step 2: Text Generation with a Small Model

Create `lab3_text_generation.py`:

```python
#!/usr/bin/env python3
"""
Lab 3: Text generation using Hugging Face Transformers.
Uses a small GPT-2 model that runs on any hardware.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def basic_text_generation():
    """Use the pipeline API for quick text generation."""
    print("=" * 60)
    print("Part 1: Basic Text Generation with Pipeline API")
    print("=" * 60)
    
    # The pipeline API is the easiest way to use models
    # It handles tokenization, model loading, and decoding automatically
    generator = pipeline(
        "text-generation",
        model="gpt2",               # 124M parameter model (~500MB)
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time, in a land of machines,",
        "The most important invention of the 21st century is",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        result = generator(
            prompt,
            max_new_tokens=80,       # Generate up to 80 new tokens
            num_return_sequences=1,  # Number of different completions
            temperature=0.8,         # Creativity (higher = more random)
            top_p=0.9,               # Nucleus sampling threshold
            do_sample=True,          # Enable sampling (vs greedy)
            pad_token_id=50256,      # GPT-2's EOS token as pad token
        )
        generated = result[0]["generated_text"]
        # Show only the new text (after the prompt)
        new_text = generated[len(prompt):]
        print(f"Generated: {new_text.strip()}")
        print("-" * 40)


def advanced_text_generation():
    """Load model manually for more control."""
    print("\n" + "=" * 60)
    print("Part 2: Advanced Generation with Manual Model Loading")
    print("=" * 60)
    
    model_name = "gpt2"
    
    # Load tokenizer and model separately
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    prompt = "Artificial intelligence will transform"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: '{prompt}'")
    print(f"Token IDs: {inputs['input_ids'].tolist()}")
    print(f"Number of input tokens: {inputs['input_ids'].shape[1]}")
    
    # Generate with different strategies
    strategies = {
        "Greedy (temperature=0)": {
            "do_sample": False,
            "max_new_tokens": 50,
        },
        "Sampling (temp=0.7)": {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 50,
        },
        "High creativity (temp=1.5)": {
            "do_sample": True,
            "temperature": 1.5,
            "max_new_tokens": 50,
        },
        "Top-k (k=50)": {
            "do_sample": True,
            "top_k": 50,
            "max_new_tokens": 50,
        },
        "Beam search (beams=5)": {
            "do_sample": False,
            "num_beams": 5,
            "max_new_tokens": 50,
        },
    }
    
    for name, params in strategies.items():
        with torch.no_grad():
            output = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                **params,
            )
        
        # Decode only the new tokens
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"\n[{name}]")
        print(f"  {generated_text.strip()}")
    
    print()


if __name__ == "__main__":
    basic_text_generation()
    advanced_text_generation()
```

Run it:
```bash
python lab3_text_generation.py
```

### Step 3: Sentiment Analysis

Create `lab3_sentiment.py`:

```python
#!/usr/bin/env python3
"""
Lab 3: Sentiment analysis using Hugging Face Transformers.
Demonstrates classification with pre-trained models.
"""

from transformers import pipeline
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    console.print("\n[bold blue]🎭 Sentiment Analysis Lab[/bold blue]\n")
    
    # Load a sentiment analysis pipeline
    # Uses 'distilbert-base-uncased-finetuned-sst-2-english' by default
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    
    # Test sentences with varying sentiment
    test_sentences = [
        "I absolutely love this product! Best purchase ever!",
        "This is the worst experience I've ever had.",
        "The weather is okay today, nothing special.",
        "I'm cautiously optimistic about the future of AI.",
        "The food was terrible but the service was excellent.",
        "Not bad, not great. Just mediocre.",
        "This movie made me cry tears of joy. Masterpiece!",
        "I regret buying this. Complete waste of money.",
        "The presentation was interesting and well-organized.",
        "I can't believe how awful the customer support was.",
    ]
    
    # Classify each sentence
    table = Table(title="Sentiment Analysis Results")
    table.add_column("Sentence", style="white", width=55)
    table.add_column("Sentiment", style="cyan", width=12)
    table.add_column("Confidence", style="green", width=12)
    
    results = classifier(test_sentences)
    
    for sentence, result in zip(test_sentences, results):
        label = result["label"]
        score = result["score"]
        
        # Color code by sentiment
        if label == "POSITIVE":
            label_display = f"[green]{label}[/green]"
        else:
            label_display = f"[red]{label}[/red]"
        
        # Truncate long sentences
        display_sentence = sentence[:52] + "..." if len(sentence) > 55 else sentence
        
        table.add_row(display_sentence, label_display, f"{score:.4f}")
    
    console.print(table)
    
    # Interactive mode
    console.print("\n[bold]Try your own sentences![/bold]")
    console.print("Type a sentence and press Enter. Type 'quit' to exit.\n")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            
            result = classifier(text)[0]
            color = "green" if result["label"] == "POSITIVE" else "red"
            console.print(
                f"  [{color}]{result['label']}[/{color}] "
                f"(confidence: {result['score']:.4f})\n"
            )
        except KeyboardInterrupt:
            break
    
    console.print("\n[bold green]Lab complete![/bold green]")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python lab3_sentiment.py
```

### Step 4: Named Entity Recognition (NER)

Create `lab3_ner.py`:

```python
#!/usr/bin/env python3
"""
Lab 3: Named Entity Recognition (NER) using Hugging Face Transformers.
Identifies people, organizations, locations, and other entities in text.
"""

from transformers import pipeline
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    console.print("\n[bold blue]🏷️ Named Entity Recognition Lab[/bold blue]\n")
    
    # Load NER pipeline
    ner = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",  # Group sub-word tokens into entities
    )
    
    # Test texts
    test_texts = [
        "Apple CEO Tim Cook announced new AI features at the WWDC conference in San Francisco.",
        "Elon Musk's company SpaceX launched a Falcon 9 rocket from Cape Canaveral, Florida.",
        "The European Union passed the AI Act in Brussels, affecting companies like Google and Microsoft.",
        "Dr. Sarah Chen published her research on neural networks at MIT in Cambridge, Massachusetts.",
        "Toyota and BMW announced a joint venture to develop electric vehicles in Tokyo.",
    ]
    
    for text in test_texts:
        console.print(f"\n[bold]Text:[/bold] {text}")
        
        entities = ner(text)
        
        if not entities:
            console.print("  [yellow]No entities found.[/yellow]")
            continue
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Entity", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Confidence", style="yellow")
        
        for entity in entities:
            # Map entity types to readable names
            entity_types = {
                "PER": "👤 Person",
                "ORG": "🏢 Organization",
                "LOC": "📍 Location",
                "MISC": "📋 Miscellaneous",
            }
            entity_type = entity_types.get(entity["entity_group"], entity["entity_group"])
            
            table.add_row(
                entity["word"],
                entity_type,
                f"{entity['score']:.4f}",
            )
        
        console.print(table)
    
    # Interactive mode
    console.print("\n[bold]Try your own text![/bold]")
    console.print("Type a sentence and press Enter. Type 'quit' to exit.\n")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            
            entities = ner(text)
            if entities:
                for e in entities:
                    console.print(
                        f"  [cyan]{e['word']}[/cyan] → "
                        f"{e['entity_group']} ({e['score']:.3f})"
                    )
            else:
                console.print("  [yellow]No entities found.[/yellow]")
            print()
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
```

Run it:
```bash
python lab3_ner.py
```

> **✅ Checkpoint**: You can now use Hugging Face models for text generation, sentiment analysis, and NER. Notice how the pipeline API makes this incredibly simple.

---

## Lab 4: Building a Chatbot

**Goal**: Build a fully functional chatbot with conversation history, system prompts, and streaming responses using Ollama.

**Time**: ~30 minutes

### Step 1: Simple Chatbot

Create `lab4_chatbot.py`:

```python
#!/usr/bin/env python3
"""
Lab 4: Build a chatbot using Ollama's chat API.
Features:
- Conversation history management
- System prompts for personality customization
- Streaming responses for real-time output
- Conversation saving/loading
"""

import requests
import json
import sys
import os
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

OLLAMA_URL = "http://localhost:11434"


class Chatbot:
    """A chatbot that uses Ollama's chat API with conversation history."""
    
    def __init__(self, model: str = "phi3:mini", system_prompt: str = None):
        """
        Initialize the chatbot.
        
        Args:
            model: The Ollama model to use
            system_prompt: Optional system prompt to set personality/behavior
        """
        self.model = model
        self.conversation_history = []
        
        # Set default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful, friendly AI assistant. "
                "Give clear, concise answers. Use markdown formatting "
                "when appropriate. If you don't know something, say so."
            )
        
        # Add system prompt as the first message
        self.conversation_history.append({
            "role": "system",
            "content": system_prompt,
        })
        
        self.system_prompt = system_prompt
    
    def chat(self, user_message: str, stream: bool = True) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: The user's message
            stream: Whether to stream the response token-by-token
            
        Returns:
            The assistant's response text
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        if stream:
            response_text = self._stream_chat()
        else:
            response_text = self._batch_chat()
        
        # Add assistant's response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })
        
        return response_text
    
    def _stream_chat(self) -> str:
        """Stream the response, printing tokens as they arrive."""
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": self.model,
                "messages": self.conversation_history,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        
        full_response = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                print(token, end="", flush=True)
                full_response.append(token)
                
                if data.get("done", False):
                    break
        
        print()  # Newline after response
        return "".join(full_response)
    
    def _batch_chat(self) -> str:
        """Get the full response at once (no streaming)."""
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": self.model,
                "messages": self.conversation_history,
                "stream": False,
            },
            timeout=120,
        )
        
        data = response.json()
        return data.get("message", {}).get("content", "")
    
    def get_history_length(self) -> int:
        """Return the number of messages in conversation history."""
        return len(self.conversation_history) - 1  # Exclude system prompt
    
    def clear_history(self):
        """Clear conversation history but keep the system prompt."""
        self.conversation_history = [self.conversation_history[0]]
    
    def save_conversation(self, filepath: str):
        """Save the conversation to a JSON file."""
        data = {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "timestamp": datetime.now().isoformat(),
            "messages": self.conversation_history,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_conversation(self, filepath: str):
        """Load a conversation from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.model = data.get("model", self.model)
        self.system_prompt = data.get("system_prompt", "")
        self.conversation_history = data.get("messages", [])


# ─── Pre-built personality templates ───

PERSONALITIES = {
    "default": (
        "You are a helpful, friendly AI assistant. Give clear, "
        "concise answers."
    ),
    "tutor": (
        "You are a patient, encouraging AI tutor. Explain concepts "
        "step by step. Use analogies and examples. Ask follow-up "
        "questions to check understanding. If the student makes a "
        "mistake, gently guide them to the correct answer."
    ),
    "pirate": (
        "You are a pirate AI assistant. Speak like a pirate in all "
        "responses (use 'arr', 'matey', 'ye', etc.) but still be "
        "helpful and accurate. End each response with a pirate "
        "saying or joke."
    ),
    "coder": (
        "You are a senior software engineer AI assistant. Always "
        "provide code examples. Explain code clearly with comments. "
        "Consider edge cases, performance, and best practices. "
        "Use Python unless the user specifies another language."
    ),
    "socratic": (
        "You are a Socratic teacher. Never give direct answers. "
        "Instead, ask thoughtful questions that guide the student "
        "to discover the answer themselves. Be encouraging but "
        "always respond with questions."
    ),
}


def print_help():
    """Print available commands."""
    console.print(Panel("""
[bold cyan]Available Commands:[/bold cyan]
  /help       - Show this help message
  /clear      - Clear conversation history
  /history    - Show conversation history
  /save FILE  - Save conversation to file
  /load FILE  - Load conversation from file
  /persona X  - Switch personality (default, tutor, pirate, coder, socratic)
  /model X    - Switch model (e.g., /model llama3.1:8b)
  /quit       - Exit the chatbot
    """, title="Help"))


def main():
    console.print(Panel(
        "[bold blue]🤖 AI Chatbot Lab[/bold blue]\n\n"
        "Chat with a local AI model powered by Ollama.\n"
        "Type [cyan]/help[/cyan] for available commands.",
        title="Lab 4",
    ))
    
    # Initialize chatbot
    model = "phi3:mini"  # Default model, change as needed
    bot = Chatbot(model=model, system_prompt=PERSONALITIES["default"])
    
    console.print(f"[dim]Model: {model} | Personality: default[/dim]")
    console.print(f"[dim]Type your message and press Enter.[/dim]\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if command == "/quit" or command == "/exit":
                    console.print("[dim]Goodbye![/dim]")
                    break
                
                elif command == "/help":
                    print_help()
                
                elif command == "/clear":
                    bot.clear_history()
                    console.print("[green]Conversation history cleared.[/green]\n")
                
                elif command == "/history":
                    for msg in bot.conversation_history:
                        role = msg["role"].upper()
                        content = msg["content"][:100]
                        if len(msg["content"]) > 100:
                            content += "..."
                        console.print(f"  [{role}] {content}")
                    console.print(
                        f"\n  Total messages: {bot.get_history_length()}\n"
                    )
                
                elif command == "/save":
                    filename = arg or "conversation.json"
                    bot.save_conversation(filename)
                    console.print(f"[green]Saved to {filename}[/green]\n")
                
                elif command == "/load":
                    filename = arg or "conversation.json"
                    if os.path.exists(filename):
                        bot.load_conversation(filename)
                        console.print(f"[green]Loaded from {filename}[/green]\n")
                    else:
                        console.print(f"[red]File not found: {filename}[/red]\n")
                
                elif command == "/persona":
                    if arg in PERSONALITIES:
                        bot = Chatbot(
                            model=bot.model,
                            system_prompt=PERSONALITIES[arg],
                        )
                        console.print(
                            f"[green]Switched to '{arg}' personality.[/green]\n"
                        )
                    else:
                        available = ", ".join(PERSONALITIES.keys())
                        console.print(
                            f"[red]Unknown personality. Available: {available}[/red]\n"
                        )
                
                elif command == "/model":
                    if arg:
                        bot.model = arg
                        bot.clear_history()
                        console.print(
                            f"[green]Switched to model: {arg}[/green]\n"
                        )
                    else:
                        console.print(
                            f"[yellow]Current model: {bot.model}[/yellow]\n"
                        )
                
                else:
                    console.print(
                        f"[red]Unknown command: {command}. "
                        f"Type /help for options.[/red]\n"
                    )
                
                continue
            
            # Regular chat message
            console.print(f"\n[bold green]{model}[/bold green]: ", end="")
            bot.chat(user_input, stream=True)
            print()  # Extra newline for readability
            
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except requests.exceptions.ConnectionError:
            console.print(
                "[red]Cannot connect to Ollama. "
                "Make sure it's running: ollama serve[/red]\n"
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python lab4_chatbot.py
```

### Step 2: Try Different Personalities

Once the chatbot is running:

```
You: /persona pirate
You: What is machine learning?

You: /persona socratic
You: How does gravity work?

You: /persona coder
You: Write a binary search function
```

### Step 3: Experiment with Context Management

Notice how conversation gets slower as history grows. This is because the entire conversation history is sent with each request. In production, you'd implement context window management:

```python
def trim_history(self, max_messages: int = 20):
    """Keep only the most recent messages to fit context window.
    
    This is a simple strategy. More sophisticated approaches:
    - Summarize old messages instead of dropping them
    - Keep the system prompt + most recent N messages
    - Use embeddings to retrieve relevant past messages
    """
    if len(self.conversation_history) > max_messages + 1:  # +1 for system
        # Keep system prompt and last N messages
        self.conversation_history = (
            [self.conversation_history[0]] +  # System prompt
            self.conversation_history[-(max_messages):]  # Recent messages
        )
```

> **✅ Checkpoint**: You've built a full-featured chatbot with personality switching, conversation persistence, and streaming output.

---

## Lab 5: Building a RAG System

**Goal**: Build a complete Retrieval-Augmented Generation (RAG) system from scratch.

**Time**: ~60 minutes

### Step 1: Install RAG Dependencies

```bash
pip install chromadb sentence-transformers langchain langchain-community
pip install pypdf docx2txt  # For reading PDFs and Word docs
```

### Step 2: The Complete RAG Pipeline

Create `lab5_rag.py`:

```python
#!/usr/bin/env python3
"""
Lab 5: Complete RAG (Retrieval-Augmented Generation) System.

This lab builds a full RAG pipeline:
1. Load documents (text files, or built-in sample data)
2. Split documents into chunks
3. Generate embeddings using sentence-transformers
4. Store embeddings in ChromaDB (vector database)
5. Retrieve relevant chunks for a query
6. Generate answers using Ollama with retrieved context

Usage:
    python lab5_rag.py                    # Use built-in sample data
    python lab5_rag.py --docs ./my_docs   # Use your own documents
"""

import os
import sys
import argparse
import hashlib
import requests
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ─── Step 1: Document Loading ───

SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to Neural Networks",
        "content": """
Neural networks are computing systems inspired by biological neural networks 
in the human brain. They consist of layers of interconnected nodes (neurons) 
that process information. Each connection has a weight that adjusts during 
training. The three main types of layers are: input layers (receive raw data), 
hidden layers (process information), and output layers (produce results).

Training a neural network involves forward propagation (computing predictions) 
and backpropagation (adjusting weights based on errors). The loss function 
measures how wrong the predictions are, and the optimizer (like SGD or Adam) 
updates weights to minimize this loss.

Common architectures include feedforward networks (MLPs), convolutional 
neural networks (CNNs) for images, and recurrent neural networks (RNNs) 
for sequences. Modern deep learning has largely shifted to transformer 
architectures, which use attention mechanisms instead of recurrence.
""",
    },
    {
        "title": "Transformer Architecture",
        "content": """
The transformer architecture was introduced in the 2017 paper "Attention Is 
All You Need" by Vaswani et al. at Google. It revolutionized NLP and later 
computer vision by replacing recurrent layers with self-attention mechanisms.

Key components of a transformer include:
- Multi-head self-attention: Allows the model to attend to different positions 
  in the input simultaneously. Each "head" can focus on different aspects 
  (syntax, semantics, etc.)
- Positional encoding: Since transformers process all tokens in parallel 
  (unlike RNNs), positional encodings add information about token position.
- Feed-forward layers: Dense layers applied to each position independently.
- Layer normalization and residual connections: Help with training stability.

The original transformer has an encoder-decoder structure. BERT uses only 
the encoder (good for understanding), GPT uses only the decoder (good for 
generation), and T5 uses both (good for sequence-to-sequence tasks).

Transformers scale remarkably well. Modern LLMs like GPT-4, Claude, and 
Llama contain billions of parameters and are trained on trillions of tokens.
""",
    },
    {
        "title": "Large Language Models (LLMs)",
        "content": """
Large Language Models (LLMs) are transformer-based models trained on vast 
amounts of text data. They learn to predict the next token in a sequence, 
which gives them the ability to generate coherent text, answer questions, 
write code, and perform many other language tasks.

Key milestones in LLM development:
- GPT-1 (2018): 117M parameters, showed language model pre-training works
- GPT-2 (2019): 1.5B parameters, surprisingly coherent text generation
- GPT-3 (2020): 175B parameters, demonstrated few-shot learning
- ChatGPT (2022): Applied RLHF to make GPT-3.5 conversational
- GPT-4 (2023): Multimodal, significant reasoning improvements
- Open source: Llama, Mistral, Qwen families (2023-2026)

LLMs are trained in stages: pre-training on large text corpora (next-token 
prediction), supervised fine-tuning (SFT) on instruction-response pairs, 
and alignment training (RLHF or DPO) to follow human preferences.

Challenges include hallucination (generating false information), bias 
(reflecting biases in training data), cost (training requires millions 
of dollars of compute), and safety (preventing misuse).
""",
    },
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": """
Retrieval-Augmented Generation (RAG) is a technique that enhances LLM 
responses by retrieving relevant information from external knowledge bases 
before generating an answer. This combines the generative power of LLMs 
with the accuracy of information retrieval.

The RAG pipeline typically works as follows:
1. Documents are split into chunks and embedded into vectors
2. Vectors are stored in a vector database (ChromaDB, Pinecone, etc.)
3. When a user asks a question, the question is also embedded
4. Similar document chunks are retrieved via vector similarity search
5. Retrieved chunks are added to the LLM prompt as context
6. The LLM generates an answer based on the retrieved context

Benefits of RAG over plain LLMs:
- Reduces hallucination by grounding answers in actual documents
- Enables access to private/proprietary data without retraining
- Keeps information up-to-date (update documents, not the model)
- Provides source attribution (can cite which documents were used)

RAG is widely used in enterprise applications, customer support bots, 
internal knowledge bases, and research assistants.
""",
    },
    {
        "title": "Model Quantization",
        "content": """
Quantization reduces the precision of model weights from high-precision 
formats (like 32-bit floating point) to lower-precision formats (like 
8-bit or 4-bit integers). This dramatically reduces model size and memory 
usage while maintaining most of the model's quality.

Common quantization formats:
- FP32 (32-bit float): Full precision, baseline quality
- FP16 (16-bit float): Half precision, minimal quality loss
- INT8 (8-bit integer): ~4x smaller than FP32, slight quality loss  
- INT4 (4-bit integer): ~8x smaller than FP32, noticeable quality loss
- GGUF: Format used by llama.cpp, supports various quantization levels
  (Q4_0, Q4_K_M, Q5_K_M, Q8_0, etc.)

Popular quantization methods include GPTQ (GPU-optimized, post-training), 
AWQ (activation-aware, often better quality than GPTQ at same bit width), 
and GGML/GGUF (CPU-friendly, used by llama.cpp and Ollama).

A 70B parameter model in FP16 requires ~140GB of memory. With Q4 
quantization, it fits in ~35GB  making it runnable on consumer hardware.
""",
    },
    {
        "title": "Fine-Tuning and LoRA",
        "content": """
Fine-tuning adapts a pre-trained model to a specific task or domain by 
continuing training on task-specific data. Full fine-tuning updates all 
model parameters, which is expensive for large models.

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method 
that freezes the original model weights and injects small trainable 
matrices into each layer. Instead of updating a full weight matrix W 
(dimensions d x d), LoRA decomposes the update into two smaller matrices 
A (d x r) and B (r x d), where r << d (typically r = 8 to 64).

This means LoRA typically trains only 0.1-1% of the total parameters, 
making it much faster and cheaper than full fine-tuning. LoRA adapters 
are small files (usually 10-100MB) that can be swapped in and out of 
the base model.

QLoRA combines quantization with LoRA: the base model is loaded in 4-bit 
precision, and LoRA adapters are trained in higher precision. This allows 
fine-tuning a 65B model on a single 48GB GPU.

Common fine-tuning frameworks include Hugging Face PEFT, Unsloth 
(optimized for speed), and Axolotl (configuration-driven).
""",
    },
]


def load_documents_from_directory(directory: str) -> list:
    """Load text documents from a directory."""
    documents = []
    supported_extensions = {".txt", ".md", ".py", ".json", ".csv"}
    
    dir_path = Path(directory)
    if not dir_path.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        return documents
    
    for file_path in dir_path.rglob("*"):
        if file_path.suffix.lower() in supported_extensions:
            try:
                content = file_path.read_text(encoding="utf-8")
                documents.append({
                    "title": file_path.name,
                    "content": content,
                    "source": str(file_path),
                })
            except Exception as e:
                console.print(f"[yellow]Skipping {file_path}: {e}[/yellow]")
    
    return documents


# ─── Step 2: Text Chunking ───

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to split
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    
    Why chunk?
    - Embedding models have input length limits
    - Smaller chunks improve retrieval precision
    - Overlap ensures we don't lose context at boundaries
    """
    chunks = []
    start = 0
    text = text.strip()
    
    while start < len(text):
        # Find the end of this chunk
        end = start + chunk_size
        
        if end < len(text):
            # Try to break at a sentence boundary
            # Look for the last period, question mark, or newline
            for sep in ["\n\n", "\n", ". ", "? ", "! "]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:  # Don't add empty chunks
            chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - chunk_overlap
        if start <= chunks[-1] if chunks else 0:
            start = end  # Prevent infinite loops
    
    return chunks


# ─── Step 3 & 4: Embedding + Vector Storage ───

class SimpleVectorStore:
    """
    A simple vector store using ChromaDB.
    Handles embedding generation and similarity search.
    """
    
    def __init__(self, collection_name: str = "rag_lab"):
        """Initialize ChromaDB and the embedding model."""
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        console.print("[dim]Loading embedding model...[/dim]")
        # all-MiniLM-L6-v2 is small (~80MB), fast, and good quality
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        console.print("[dim]Initializing vector database...[/dim]")
        # Use persistent storage (saves to disk)
        self.client = chromadb.Client()  # In-memory for simplicity
        
        # Delete collection if it exists (fresh start)
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
    
    def add_documents(self, chunks: list, metadata_list: list = None):
        """
        Embed and store document chunks.
        
        Args:
            chunks: List of text chunks to embed
            metadata_list: Optional list of metadata dicts for each chunk
        """
        if not chunks:
            return
        
        console.print(f"[dim]Embedding {len(chunks)} chunks...[/dim]")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        
        # Generate unique IDs for each chunk
        ids = [
            hashlib.md5(chunk.encode()).hexdigest()[:12] + f"_{i}"
            for i, chunk in enumerate(chunks)
        ]
        
        # Prepare metadata
        if metadata_list is None:
            metadata_list = [{"source": "unknown"} for _ in chunks]
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadata_list,
        )
        
        console.print(
            f"[green]✅ Stored {len(chunks)} chunks in vector database[/green]"
        )
    
    def search(self, query: str, top_k: int = 3) -> list:
        """
        Search for chunks similar to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'text', 'metadata', and 'distance'
        """
        # Embed the query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        
        return formatted


# ─── Step 5: RAG Query Engine ───

class RAGEngine:
    """
    The complete RAG engine.
    Retrieves relevant context and generates answers with Ollama.
    """
    
    def __init__(self, vector_store: SimpleVectorStore, model: str = "phi3:mini"):
        self.vector_store = vector_store
        self.model = model
    
    def query(self, question: str, top_k: int = 3, show_sources: bool = True) -> str:
        """
        Answer a question using RAG.
        
        1. Search for relevant document chunks
        2. Build a prompt with the retrieved context
        3. Generate an answer with Ollama
        """
        # Step 1: Retrieve relevant chunks
        console.print(f"\n[dim]🔍 Searching for relevant context...[/dim]")
        results = self.vector_store.search(question, top_k=top_k)
        
        if show_sources:
            console.print(f"[dim]Found {len(results)} relevant chunks:[/dim]")
            for i, result in enumerate(results):
                preview = result["text"][:80].replace("\n", " ")
                source = result["metadata"].get("source", "unknown")
                distance = result["distance"]
                console.print(
                    f"[dim]  {i+1}. [{source}] (similarity: {1-distance:.3f}) "
                    f"'{preview}...'[/dim]"
                )
        
        # Step 2: Build the RAG prompt
        context = "\n\n---\n\n".join([r["text"] for r in results])
        
        rag_prompt = f"""Use the following context to answer the question. 
If the answer is not found in the context, say "I don't have enough information 
to answer that question based on the available documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        # Step 3: Generate answer with Ollama
        console.print(f"[dim]💬 Generating answer...[/dim]\n")
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": self.model,
                "prompt": rag_prompt,
                "stream": True,
                "options": {
                    "temperature": 0.3,  # Low temperature for factual answers
                    "num_predict": 500,
                },
            },
            stream=True,
            timeout=120,
        )
        
        full_response = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                print(token, end="", flush=True)
                full_response.append(token)
                if data.get("done", False):
                    break
        
        print("\n")
        answer = "".join(full_response)
        
        # Show sources
        if show_sources:
            console.print("[dim]📚 Sources used:[/dim]")
            for i, result in enumerate(results):
                source = result["metadata"].get("source", "unknown")
                console.print(f"[dim]  {i+1}. {source}[/dim]")
        
        return answer


# ─── Main Application ───

OLLAMA_URL = "http://localhost:11434"

def main():
    parser = argparse.ArgumentParser(description="RAG System Lab")
    parser.add_argument(
        "--docs", type=str, default=None,
        help="Path to a directory of documents to index",
    )
    parser.add_argument(
        "--model", type=str, default="phi3:mini",
        help="Ollama model to use for generation",
    )
    args = parser.parse_args()
    
    console.print(Panel(
        "[bold blue]📚 RAG System Lab[/bold blue]\n\n"
        "Build a Retrieval-Augmented Generation system from scratch.\n"
        "Ask questions and get answers grounded in real documents.",
        title="Lab 5",
    ))
    
    # Step 1: Load documents
    if args.docs:
        console.print(f"\n[bold]Loading documents from: {args.docs}[/bold]")
        documents = load_documents_from_directory(args.docs)
    else:
        console.print("\n[bold]Using built-in sample documents (AI knowledge base)[/bold]")
        documents = SAMPLE_DOCUMENTS
    
    if not documents:
        console.print("[red]No documents found![/red]")
        return
    
    console.print(f"Loaded {len(documents)} documents.\n")
    
    # Step 2: Chunk documents
    console.print("[bold]Chunking documents...[/bold]")
    all_chunks = []
    all_metadata = []
    
    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"source": doc["title"]})
    
    console.print(f"Created {len(all_chunks)} chunks from {len(documents)} documents.\n")
    
    # Step 3 & 4: Embed and store
    console.print("[bold]Building vector database...[/bold]")
    vector_store = SimpleVectorStore()
    vector_store.add_documents(all_chunks, all_metadata)
    
    # Step 5: Create RAG engine
    rag = RAGEngine(vector_store, model=args.model)
    
    # Interactive query loop
    console.print(
        f"\n[bold green]✅ RAG system ready![/bold green]"
        f"\n[dim]Model: {args.model} | Documents: {len(documents)} | "
        f"Chunks: {len(all_chunks)}[/dim]"
    )
    
    # Show example queries
    console.print("\n[bold]Example queries to try:[/bold]")
    example_queries = [
        "What is RAG and how does it work?",
        "Explain the transformer architecture",
        "What is LoRA and why is it useful?",
        "How does quantization reduce model size?",
        "What are the key milestones in LLM development?",
    ]
    for q in example_queries:
        console.print(f"  • {q}")
    
    console.print("\nType your question and press Enter. Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            
            rag.query(question)
            print()
        except KeyboardInterrupt:
            break
        except requests.exceptions.ConnectionError:
            console.print(
                "[red]Cannot connect to Ollama. Is it running?[/red]\n"
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
    
    console.print("\n[bold green]Lab complete![/bold green]")


if __name__ == "__main__":
    main()
```

Run it:
```bash
# With built-in sample data
python lab5_rag.py

# With your own documents
python lab5_rag.py --docs ./my_documents

# With a specific model
python lab5_rag.py --model llama3.1:8b
```

### Key Concepts to Observe

1. **Chunking matters**: Try different chunk sizes (200, 500, 1000) and see how retrieval quality changes
2. **Embedding quality**: The embedding model determines how well semantic similarity works
3. **Top-k retrieval**: More chunks provide more context but may include noise
4. **Temperature**: Low temperature (0.1-0.3) gives more factual answers from context

> **✅ Checkpoint**: You've built a complete RAG system! This is the same fundamental architecture used by enterprise AI applications, ChatGPT with web browsing, and document Q&A systems.

---

## Lab 6: Fine-Tuning a Small Model

**Goal**: Fine-tune a small language model using LoRA for a specific task.

**Time**: ~60 minutes (plus training time)

> **⚠️ Requirements**: This lab requires a GPU with at least **12 GB VRAM** (e.g., RTX 3060 12GB or better). If you don't have a GPU, you can use Google Colab (free tier provides a T4 GPU).

### Step 1: Install Dependencies

```bash
pip install peft bitsandbytes datasets trl
pip install unsloth  # Faster alternative (optional, recommended)
```

### Step 2: Fine-Tune with Hugging Face PEFT

Create `lab6_finetune.py`:

```python
#!/usr/bin/env python3
"""
Lab 6: Fine-tune a small language model using LoRA.

This lab demonstrates:
1. Loading a base model with 4-bit quantization (QLoRA)
2. Preparing a custom dataset
3. Applying LoRA adapters
4. Training with the SFTTrainer
5. Saving and loading the fine-tuned adapter
6. Running inference with the fine-tuned model

We'll fine-tune a small model to be a helpful assistant that responds
in a specific style (concise, technical, with emoji).
"""

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# ─── Configuration ───

MODEL_NAME = "microsoft/phi-2"  # 2.7B params, manageable on consumer GPUs
OUTPUT_DIR = "./lora_adapter"
MAX_SEQ_LENGTH = 512


# ─── Step 1: Prepare Training Data ───

def create_training_dataset():
    """
    Create a small fine-tuning dataset.
    
    In practice, you'd load from a file or Hugging Face datasets.
    For this lab, we'll create a small example dataset.
    
    The format follows the instruction-tuning convention:
    Each example has an instruction and a response.
    """
    
    examples = [
        {
            "text": """<|user|>
What is a neural network?
<|assistant|>
🧠 A neural network is a computational model inspired by the brain's structure. It consists of layers of interconnected nodes (neurons) that transform inputs through weighted connections. Key points:
• Input layer receives raw data
• Hidden layers extract features
• Output layer produces predictions
• Learning = adjusting weights via backpropagation
Think of it as a function approximator that learns from examples.""",
        },
        {
            "text": """<|user|>
Explain gradient descent in simple terms.
<|assistant|>
⛰️ Gradient descent is an optimization algorithm  imagine you're blindfolded on a mountain trying to reach the lowest valley:
1. Feel the slope beneath your feet (compute gradient)
2. Take a step downhill (update parameters)
3. Repeat until you reach the bottom (loss is minimized)
The "learning rate" controls step size: too big = overshoot, too small = slow convergence. Variants: SGD, Adam, AdaGrad.""",
        },
        {
            "text": """<|user|>
What is the difference between CNN and RNN?
<|assistant|>
📊 CNNs vs RNNs  two architectures for different data types:

CNN (Convolutional Neural Network):
• Best for: spatial data (images, grids)
• Key op: convolution filters slide over input
• Strength: translation invariance, local patterns
• Use: image classification, object detection

RNN (Recurrent Neural Network):
• Best for: sequential data (text, time series)
• Key op: hidden state passed between timesteps
• Strength: captures temporal dependencies
• Weakness: vanishing gradients → use LSTM/GRU

💡 In 2026, transformers have largely replaced both for many tasks.""",
        },
        {
            "text": """<|user|>
How does attention work in transformers?
<|assistant|>
🎯 Attention lets a model focus on relevant parts of the input:

1. Each token creates three vectors: Query (Q), Key (K), Value (V)
2. Attention score = softmax(Q · K^T / √d_k)
3. Output = weighted sum of Values using attention scores

Self-attention: every token attends to every other token
Multi-head: run attention in parallel with different learned projections

Result: token "bank" can attend to "river" or "money" to disambiguate meaning. This is what makes transformers so powerful! 🚀""",
        },
        {
            "text": """<|user|>
What is transfer learning?
<|assistant|>
🔄 Transfer learning = reusing knowledge from one task for another.

Analogy: Learning Spanish is easier if you already know Italian.

How it works:
1. Pre-train a model on a large general dataset (expensive, done once)
2. Fine-tune on your specific task (cheap, done many times)

Examples:
• ImageNet → medical image classification
• GPT base model → customer service chatbot
• BERT → sentiment analysis

Benefits: less data needed, faster training, better performance.
This is THE fundamental paradigm of modern AI. 🎯""",
        },
        # Add more examples as needed...
        {
            "text": """<|user|>
Explain overfitting vs underfitting.
<|assistant|>
📈 Two sides of the same coin:

Overfitting (too complex):
• Model memorizes training data noise
• Great training accuracy, poor test accuracy
• Fix: regularization, dropout, more data, early stopping

Underfitting (too simple):
• Model can't capture the underlying pattern
• Poor accuracy on both training and test data
• Fix: more parameters, longer training, better features

The sweet spot is the bias-variance tradeoff:
• Bias → underfitting (model is too simple)
• Variance → overfitting (model is too sensitive)

🎯 Goal: low bias AND low variance.""",
        },
        {
            "text": """<|user|>
What is LoRA?
<|assistant|>
🔧 LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method:

Instead of updating ALL model weights (expensive):
• Freeze the original weight matrix W
• Add two small matrices: A (d×r) and B (r×d)
• Update = A × B, where r << d (rank, typically 8-64)

Result:
• Train only 0.1-1% of parameters
• Adapter files are tiny (10-100MB vs multi-GB model)
• Can swap adapters in/out of base model
• QLoRA = LoRA + 4-bit quantized base model

It's like adding a small plugin to a big model instead of rewriting the whole thing. 🔌""",
        },
    ]
    
    return Dataset.from_list(examples)


# ─── Step 2: Load Model and Apply LoRA ───

def setup_model_and_lora():
    """Load the base model with quantization and apply LoRA."""
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model with 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 quantization
        bnb_4bit_compute_dtype=torch.float16, # Compute in FP16
        bnb_4bit_use_double_quant=True,      # Double quantization for extra savings
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Configure LoRA
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16,                     # Rank of the LoRA matrices
        lora_alpha=32,            # Scaling factor
        target_modules=[          # Which layers to apply LoRA to
            "q_proj", "k_proj", "v_proj", "dense",
            "fc1", "fc2",
        ],
        lora_dropout=0.05,        # Dropout for regularization
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    # Expected output: something like:
    # trainable params: 6,553,600 || all params: 2,784,808,960 || trainable%: 0.235%
    
    return model, tokenizer


# ─── Step 3: Train ───

def train(model, tokenizer, dataset):
    """Fine-tune the model using SFTTrainer."""
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,              # Number of training epochs
        per_device_train_batch_size=2,   # Batch size (adjust for your GPU)
        gradient_accumulation_steps=4,   # Effective batch size = 2 * 4 = 8
        learning_rate=2e-4,              # Learning rate
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=5,                 # Log every 5 steps
        save_strategy="epoch",
        fp16=True,                       # Mixed precision training
        optim="paged_adamw_8bit",        # Memory-efficient optimizer
        report_to="none",               # Disable wandb/tensorboard
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Save the LoRA adapter
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Training complete! Adapter saved to {OUTPUT_DIR}")
    
    return model, tokenizer


# ─── Step 4: Inference ───

def test_model(model, tokenizer):
    """Test the fine-tuned model."""
    
    print("\n" + "=" * 60)
    print("Testing fine-tuned model")
    print("=" * 60 + "\n")
    
    test_prompts = [
        "What is backpropagation?",
        "Explain the difference between supervised and unsupervised learning.",
        "What is a loss function?",
    ]
    
    for prompt in test_prompts:
        full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        print(f"Q: {prompt}")
        print(f"A: {response.strip()}")
        print("-" * 40)


# ─── Main ───

def main():
    print("=" * 60)
    print("Lab 6: Fine-Tuning with LoRA")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠️  No GPU detected. Training will be very slow.")
        print("Consider using Google Colab for this lab.")
    
    # Create dataset
    print("\nPreparing dataset...")
    dataset = create_training_dataset()
    print(f"Dataset size: {len(dataset)} examples")
    
    # Load model and apply LoRA
    model, tokenizer = setup_model_and_lora()
    
    # Train
    model, tokenizer = train(model, tokenizer, dataset)
    
    # Test
    test_model(model, tokenizer)
    
    print("\n✅ Lab 6 complete!")
    print(f"Your LoRA adapter is saved at: {OUTPUT_DIR}")
    print("It's only a few MB  compare that to the multi-GB base model!")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python lab6_finetune.py
```

### What to Observe

1. **Trainable parameters**: LoRA trains only ~0.2% of the model's parameters
2. **Training speed**: Even with a small dataset, training completes in minutes
3. **Adapter size**: The saved adapter is only 10-50 MB
4. **Style transfer**: The model should learn the emoji-heavy, structured response style from the training data

> **✅ Checkpoint**: You've fine-tuned a language model! The adapter file in `./lora_adapter` is all you need to customize the model's behavior.

---

## Lab 7: Quantizing Models

**Goal**: Understand quantization by converting models to different formats and comparing quality.

**Time**: ~30 minutes

### Step 1: Install llama.cpp

```bash
# Option A: Use pre-built binaries
# Download from: https://github.com/ggerganov/llama.cpp/releases

# Option B: Build from source (requires CMake and a C++ compiler)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release

# The quantize binary will be at: build/bin/llama-quantize
```

### Step 2: Understand Quantization Levels

Create `lab7_quantization.py`:

```python
#!/usr/bin/env python3
"""
Lab 7: Understanding Model Quantization.

This lab demonstrates:
1. How quantization reduces model size
2. The tradeoff between size and quality
3. Comparing different quantization levels
4. Using Ollama to compare quantized models

Since full quantization with llama.cpp requires downloading large models,
this lab focuses on understanding the concepts and comparing via Ollama.
"""

import requests
import json
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

OLLAMA_URL = "http://localhost:11434"


# ─── Part 1: Understanding Quantization (Theory) ───

def explain_quantization():
    """Visual explanation of quantization."""
    
    console.print(Panel("""
[bold cyan]What is Quantization?[/bold cyan]

Quantization reduces the precision of numbers used to store model weights.

[bold]FP32 (32-bit float)  Full precision:[/bold]
  Each weight: ████████████████████████████████  (32 bits)
  Example:     3.14159265358979...
  
[bold]FP16 (16-bit float)  Half precision:[/bold]  
  Each weight: ████████████████                  (16 bits, 2× smaller)
  Example:     3.14159
  
[bold]INT8 (8-bit integer)  Quarter precision:[/bold]
  Each weight: ████████                          (8 bits, 4× smaller)
  Example:     3.14
  
[bold]INT4 (4-bit integer)  Eighth precision:[/bold]
  Each weight: ████                              (4 bits, 8× smaller)
  Example:     3.1

[bold yellow]The tradeoff:[/bold yellow]
  More compression → Smaller model + Faster inference
  More compression → More quality loss (especially at Q2-Q3)
    """, title="Quantization Explained"))


def show_quantization_table():
    """Show the common GGUF quantization levels."""
    
    table = Table(title="GGUF Quantization Levels (for a 7B parameter model)")
    table.add_column("Quant Level", style="cyan")
    table.add_column("Bits/Weight", style="green")
    table.add_column("Approx Size", style="yellow")
    table.add_column("Quality", style="magenta")
    table.add_column("Speed", style="blue")
    table.add_column("Use Case", style="white")
    
    quantization_levels = [
        ("F32", "32", "~28 GB", "★★★★★", "★☆☆☆☆", "Reference only"),
        ("F16", "16", "~14 GB", "★★★★★", "★★☆☆☆", "GPU with lots of VRAM"),
        ("Q8_0", "8", "~7.5 GB", "★★★★☆", "★★★☆☆", "Best quality quantized"),
        ("Q6_K", "6.5", "~5.5 GB", "★★★★☆", "★★★☆☆", "Near-lossless"),
        ("Q5_K_M", "5.5", "~5.0 GB", "★★★★☆", "★★★★☆", "Great balance"),
        ("Q5_0", "5", "~4.7 GB", "★★★☆☆", "★★★★☆", "Good balance"),
        ("Q4_K_M", "4.8", "~4.4 GB", "★★★☆☆", "★★★★☆", "Most popular"),
        ("Q4_0", "4", "~3.8 GB", "★★★☆☆", "★★★★★", "Fastest 4-bit"),
        ("Q3_K_M", "3.9", "~3.3 GB", "★★☆☆☆", "★★★★★", "Aggressive compression"),
        ("Q2_K", "2.6", "~2.7 GB", "★☆☆☆☆", "★★★★★", "Extreme, quality loss"),
        ("IQ2_XXS", "2.1", "~2.2 GB", "★☆☆☆☆", "★★★★★", "Experimental extreme"),
    ]
    
    for row in quantization_levels:
        table.add_row(*row)
    
    console.print(table)
    
    console.print("""
[bold]Key insights:[/bold]
• [cyan]Q4_K_M[/cyan] is the most popular level  good balance of size and quality
• [cyan]Q5_K_M[/cyan] and above are nearly indistinguishable from full precision
• [cyan]Q3_K_M[/cyan] and below show noticeable quality degradation
• The "K" variants use k-quant (importance-based quantization) = better quality
• The "M" suffix means "medium" (between "S"mall and "L"arge configs)
    """)


# ─── Part 2: Compare Quantized Models via Ollama ───

def compare_quantized_models():
    """
    Compare different quantization levels using Ollama.
    
    Ollama makes it easy to pull the same model in different quantizations.
    """
    console.print("\n[bold]Comparing quantized models via Ollama[/bold]\n")
    
    # These are example model tags  available quantizations vary by model
    # For Qwen2.5, typical options include:
    models_to_compare = [
        "qwen2.5:0.5b",   # Small model, default quantization
        "qwen2.5:1.5b",   # Slightly larger
    ]
    
    # Check which are available
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        available = [m["name"] for m in response.json().get("models", [])]
    except Exception:
        console.print("[red]Cannot connect to Ollama. Is it running?[/red]")
        return
    
    # Filter to available models
    test_models = [m for m in models_to_compare if m in available]
    
    if len(test_models) < 1:
        console.print("[yellow]Pull at least one model to compare:[/yellow]")
        console.print("  ollama pull qwen2.5:0.5b")
        console.print("  ollama pull qwen2.5:1.5b")
        return
    
    # Test prompts that reveal quality differences
    test_prompts = [
        {
            "name": "Factual Knowledge",
            "prompt": "List the first 5 prime numbers and explain why 1 is not prime.",
        },
        {
            "name": "Logical Reasoning",
            "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Explain step by step.",
        },
        {
            "name": "Code Quality",
            "prompt": "Write a Python function to check if a number is prime. Include edge cases.",
        },
    ]
    
    for prompt_info in test_prompts:
        console.print(f"\n[bold cyan]═══ {prompt_info['name']} ═══[/bold cyan]")
        console.print(f"[dim]Prompt: {prompt_info['prompt']}[/dim]\n")
        
        for model in test_models:
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt_info["prompt"],
                        "stream": False,
                        "options": {
                            "temperature": 0,      # Deterministic for comparison
                            "num_predict": 200,
                        },
                    },
                    timeout=120,
                )
                
                data = response.json()
                elapsed = time.time() - start_time
                text = data.get("response", "").strip()
                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 1)
                tps = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
                
                console.print(f"[bold green]{model}[/bold green] "
                            f"({elapsed:.1f}s, {tps:.1f} tok/s):")
                console.print(f"  {text[:300]}")
                if len(text) > 300:
                    console.print(f"  [dim]...({len(text) - 300} more chars)[/dim]")
                console.print()
                
            except Exception as e:
                console.print(f"[red]{model}: Error - {e}[/red]\n")


# ─── Part 3: Manual Quantization Steps ───

def show_quantization_workflow():
    """Show the workflow for quantizing a model with llama.cpp."""
    
    console.print(Panel("""
[bold cyan]How to Quantize a Model with llama.cpp[/bold cyan]

[bold]Step 1: Get the model in GGUF format[/bold]
  If starting from a Hugging Face model (safetensors/PyTorch):
  
  [green]# Convert HF model to GGUF (from llama.cpp directory)
  python convert_hf_to_gguf.py /path/to/hf_model \\
      --outfile model-f16.gguf \\
      --outtype f16[/green]

[bold]Step 2: Quantize to desired level[/bold]
  [green]# Quantize from F16 to Q4_K_M
  ./build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
  
  # Quantize to Q5_K_M (higher quality)
  ./build/bin/llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M
  
  # Quantize to Q8_0 (near-lossless)
  ./build/bin/llama-quantize model-f16.gguf model-q8_0.gguf Q8_0[/green]

[bold]Step 3: Test the quantized model[/bold]
  [green]# Run with llama.cpp
  ./build/bin/llama-cli -m model-q4_k_m.gguf \\
      -p "Hello, how are you?" \\
      -n 100
  
  # Or import into Ollama
  # Create a Modelfile:
  echo 'FROM ./model-q4_k_m.gguf' > Modelfile
  ollama create my-model -f Modelfile
  ollama run my-model[/green]

[bold]Step 4: Compare sizes[/bold]
  [green]# Check file sizes
  ls -lh model-*.gguf[/green]
  
  Expected sizes for a 7B model:
    model-f16.gguf      ~14 GB
    model-q8_0.gguf     ~7.5 GB
    model-q5_k_m.gguf   ~5.0 GB
    model-q4_k_m.gguf   ~4.4 GB
    model-q3_k_m.gguf   ~3.3 GB
    """, title="Quantization Workflow"))


# ─── Part 4: Quantization Quality Test ───

def quantization_quality_test():
    """
    Demonstrate quantization effects on simple number representations.
    Shows how precision loss works at a fundamental level.
    """
    import struct
    
    console.print("\n[bold cyan]Quantization Quality Demo[/bold cyan]\n")
    console.print("How precision loss works with actual numbers:\n")
    
    table = Table(title="Precision Loss by Data Type")
    table.add_column("Original Value", style="cyan")
    table.add_column("FP32", style="green")
    table.add_column("FP16", style="yellow")
    table.add_column("INT8 (scaled)", style="magenta")
    table.add_column("INT4 (scaled)", style="red")
    
    import numpy as np
    
    test_values = [3.14159265, 0.001234, -2.71828, 0.99999, 42.0, 0.0001]
    
    for val in test_values:
        # FP32 - full precision
        fp32 = float(np.float32(val))
        
        # FP16 - half precision 
        fp16 = float(np.float16(val))
        
        # INT8 - simulate with scaling
        # Scale to [-127, 127] range
        scale_8 = 127.0 / max(abs(val), 1e-10)
        int8_val = round(val * scale_8) / scale_8
        
        # INT4 - simulate with scaling
        # Scale to [-7, 7] range
        scale_4 = 7.0 / max(abs(val), 1e-10)
        int4_val = round(val * scale_4) / scale_4
        
        table.add_row(
            f"{val:.8f}",
            f"{fp32:.8f}",
            f"{fp16:.6f}",
            f"{int8_val:.6f}",
            f"{int4_val:.4f}",
        )
    
    console.print(table)
    
    console.print("""
[bold]Observations:[/bold]
• FP32 → FP16: Almost no loss for typical weight values
• FP16 → INT8: Small rounding errors accumulate over billions of weights
• INT8 → INT4: Significant loss, but k-quants mitigate by varying precision
  across layers (more important layers get more bits)
    """)


# ─── Main ───

def main():
    console.print(Panel(
        "[bold blue]📐 Quantization Lab[/bold blue]\n\n"
        "Understand how model quantization works and\n"
        "compare quality across quantization levels.",
        title="Lab 7",
    ))
    
    # Part 1: Theory
    explain_quantization()
    input("\nPress Enter to continue to quantization levels table...")
    
    show_quantization_table()
    input("\nPress Enter to continue to quality demo...")
    
    # Part 2: Quality demo
    quantization_quality_test()
    input("\nPress Enter to continue to model comparison...")
    
    # Part 3: Compare via Ollama
    compare_quantized_models()
    
    # Part 4: Show workflow
    input("\nPress Enter to see the quantization workflow...")
    show_quantization_workflow()
    
    console.print("\n[bold green]✅ Lab 7 complete![/bold green]")
    console.print("""
[bold]Key Takeaways:[/bold]
1. Quantization trades precision for speed and size
2. Q4_K_M is the sweet spot for most use cases
3. Q5_K_M and above are nearly indistinguishable from full precision
4. Below Q3, quality degrades noticeably
5. K-quant variants are better than basic quants (Q4_K_M > Q4_0)
6. Quantization makes it possible to run 70B models on consumer hardware
    """)


if __name__ == "__main__":
    main()
```

Run it:
```bash
python lab7_quantization.py
```

> **✅ Checkpoint**: You now understand quantization  one of the key techniques that makes running large models on consumer hardware possible.

---

## Summary: What You've Built

| Lab | What You Built | Key Skills |
|-----|---------------|------------|
| **Lab 1** | AI development environment | Python, venv, Ollama, libraries |
| **Lab 2** | Model comparison tool | API usage, performance measurement |
| **Lab 3** | NLP pipeline tools | Hugging Face, text gen, NER, sentiment |
| **Lab 4** | Interactive chatbot | Chat APIs, conversation management, streaming |
| **Lab 5** | RAG system | Embeddings, vector DB, retrieval, generation |
| **Lab 6** | Fine-tuned model | LoRA, QLoRA, dataset prep, training |
| **Lab 7** | Quantization analysis | GGUF, precision tradeoffs, llama.cpp |

### Next Steps

After completing these labs, you can:

1. **Extend the chatbot** (Lab 4) with tool use and web search
2. **Build a RAG system** (Lab 5) with your own documents
3. **Fine-tune a model** (Lab 6) on your company's domain data
4. **Deploy a model** using Ollama's API in a web application
5. **Contribute to open-source** AI projects on GitHub

