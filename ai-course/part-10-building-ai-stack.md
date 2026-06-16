# Part 10: Building an AI Stack  From Hardware to User Interface

## Introduction: The Full Picture

Building an AI-powered application in 2026 isn't just about picking a model. It's about assembling a **stack**  layers of technology that work together, each solving a specific problem. Think of it like building a house: you need a foundation (hardware), plumbing (inference), structure (frameworks), and a roof (user interface).

This chapter walks through every layer of the modern AI application stack, from the silicon running the computations to the interface your users interact with.

```
THE COMPLETE AI APPLICATION STACK

┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                          │
│              Web Apps · Chat UIs · IDE Plugins                   │
├─────────────────────────────────────────────────────────────────┤
│                    APPLICATION LAYER                             │
│         LangChain · LlamaIndex · Semantic Kernel                │
│             Vercel AI SDK · Spring AI · Haystack                │
├─────────────────────────────────────────────────────────────────┤
│                    AGENT LAYER                                   │
│        Planning · Tool Use · Memory · Orchestration             │
│        CrewAI · AutoGen · LangGraph · OpenAI Agents SDK         │
├─────────────────────────────────────────────────────────────────┤
│                    RAG LAYER                                     │
│       Document Processing · Chunking · Embedding · Retrieval    │
├─────────────────────────────────────────────────────────────────┤
│                    VECTOR DATABASE LAYER                         │
│        Pinecone · Weaviate · Qdrant · ChromaDB · pgvector       │
├─────────────────────────────────────────────────────────────────┤
│                    LLM LAYER                                     │
│         Model Selection · API Providers · Self-Hosted           │
│         OpenAI · Anthropic · Google · Open Weight Models        │
├─────────────────────────────────────────────────────────────────┤
│                    INFERENCE LAYER                               │
│          vLLM · TGI · Triton · Ollama · llama.cpp              │
├─────────────────────────────────────────────────────────────────┤
│                    GPU / HARDWARE LAYER                          │
│       NVIDIA GPUs · Cloud Instances · Apple Silicon · Edge      │
└─────────────────────────────────────────────────────────────────┘

Data flows UP from hardware to user.
Requests flow DOWN from user to hardware.
```

---

## Layer 1: User Interface Layer

### What This Layer Does

The user interface (UI) is how humans interact with AI. It's the **window into intelligence**. A brilliant model behind a bad UI will fail; a mediocre model with an excellent UI can delight users.

### Types of AI Interfaces

```
AI Interface Taxonomy:

┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. CHAT INTERFACES                                     │
│     ┌──────────────────────────────────────────┐        │
│     │  User: How do I fix this error?          │        │
│     │  AI:   The error occurs because...       │        │
│     │  User: Can you show me an example?       │        │
│     │  AI:   Here's a code example:            │        │
│     └──────────────────────────────────────────┘        │
│     Examples: ChatGPT, Claude, Gemini                   │
│                                                         │
│  2. INLINE / EMBEDDED AI                                │
│     ┌──────────────────────────────────────────┐        │
│     │  def calculate_tax(income):              │        │
│     │      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░       │        │
│     │      # AI suggestion appears inline      │        │
│     │      tax_rate = 0.3 if income > 50000    │        │
│     └──────────────────────────────────────────┘        │
│     Examples: GitHub Copilot, Cursor, Windsurf          │
│                                                         │
│  3. SEARCH + ANSWER                                     │
│     ┌──────────────────────────────────────────┐        │
│     │  Q: What caused the 2008 financial crisis?│       │
│     │                                          │        │
│     │  AI-Generated Answer with Citations:     │        │
│     │  The crisis was triggered by... [1][2]   │        │
│     └──────────────────────────────────────────┘        │
│     Examples: Perplexity, Google AI Overviews           │
│                                                         │
│  4. VOICE / MULTIMODAL                                  │
│     ┌──────────────────────────────────────────┐        │
│     │  🎤 "Hey assistant, what's in this photo?"│       │
│     │  📷  [Image of a plant]                   │       │
│     │  🔊 "That's a Monstera deliciosa..."     │        │
│     └──────────────────────────────────────────┘        │
│     Examples: GPT-4o voice, Gemini Live                 │
│                                                         │
│  5. AUTONOMOUS AGENTS                                   │
│     ┌──────────────────────────────────────────┐        │
│     │  Task: "Book me a flight to NYC under    │        │
│     │         $500 for next Friday"            │        │
│     │                                          │        │
│     │  Agent: Searching flights...             │        │
│     │  Agent: Found 3 options...               │        │
│     │  Agent: Booked Delta DL102 for $423      │        │
│     └──────────────────────────────────────────┘        │
│     Examples: Claude Computer Use, Operator             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Technology Choices for UI

| Approach | Technologies | Best For | Complexity |
|----------|-------------|----------|------------|
| Web Chat | React, Next.js, Svelte | General audience | Medium |
| Streaming Chat | Vercel AI SDK, SSE | Real-time responses | Medium |
| IDE Extension | VS Code Extension API, LSP | Developer tools | High |
| CLI Tool | Python Click/Typer, Node.js | Developer utilities | Low |
| Mobile App | React Native, Swift, Kotlin | Mobile-first products | High |
| Slack/Discord Bot | Bolt SDK, Discord.js | Team collaboration | Low-Medium |
| Browser Extension | Chrome Extension API | Web augmentation | Medium |
| API-only | REST/GraphQL | B2B, embedded AI | Low |

### Building a Chat UI: Key Patterns

```
Essential Chat UI Features:

┌─────────────────────────────────────────────────────────┐
│  CONVERSATION THREAD                                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │ User      Today at 2:30 PM                      │    │
│  │ How do I implement authentication in Next.js?   │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ Assistant  Today at 2:30 PM                     │    │
│  │ There are several approaches to authentication  │    │
│  │ in Next.js. Here are the most common:           │    │
│  │                                                  │    │
│  │ 1. **NextAuth.js** - The most popular...        │    │
│  │ 2. **Clerk** - Managed authentication...        │    │
│  │                                                  │    │
│  │ ```typescript                                    │    │
│  │ // Example with NextAuth.js                     │    │
│  │ import NextAuth from "next-auth"                │    │
│  │ ```                                             │    │
│  │                                                  │    │
│  │ [Copy] [Regenerate] [👍] [👎]                    │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ [📎 Attach] Type your message...        [Send ➤]│    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  Model: GPT-4o ▼  │  Temperature: 0.7  │  System ⚙️    │
└─────────────────────────────────────────────────────────┘

Key UX Patterns:
  • Token-by-token streaming (never show a loading spinner)
  • Markdown rendering with syntax highlighting
  • Code copy buttons
  • Regenerate response
  • Feedback buttons (thumbs up/down)
  • File/image upload
  • Conversation history sidebar
  • Model/parameter selection
```

### Example: Minimal Streaming Chat (Next.js + Vercel AI SDK)

```typescript
// app/api/chat/route.ts  Server-side streaming endpoint
import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openai('gpt-4o'),
    system: 'You are a helpful coding assistant.',
    messages,
  });

  return result.toDataStreamResponse();
}
```

```tsx
// app/page.tsx  Client-side chat component
'use client';
import { useChat } from '@ai-sdk/react';

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat();

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      <div className="flex-1 overflow-y-auto space-y-4">
        {messages.map((msg) => (
          <div key={msg.id} className={msg.role === 'user' ? 'text-right' : 'text-left'}>
            <span className="font-bold">{msg.role === 'user' ? 'You' : 'AI'}</span>
            <p>{msg.content}</p>
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2 mt-4">
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="Ask me anything..."
          className="flex-1 border rounded p-2"
        />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
    </div>
  );
}
```

---

## Layer 2: Application Layer

### What This Layer Does

The application layer provides **frameworks and abstractions** that simplify building AI-powered applications. Instead of manually managing API calls, context windows, prompt templates, and output parsing, these frameworks handle the plumbing.

### The Major Frameworks (2026)

```
AI Application Framework Landscape:

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  PYTHON ECOSYSTEM                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │  LangChain  │  │  LlamaIndex │  │  Haystack            │ │
│  │             │  │             │  │  (by deepset)        │ │
│  │  General-   │  │  Data/RAG   │  │  Production-focused  │ │
│  │  purpose    │  │  focused    │  │  NLP pipelines       │ │
│  │  AI apps    │  │  framework  │  │                      │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
│                                                              │
│  .NET / C# ECOSYSTEM                                         │
│  ┌──────────────────────────┐                                │
│  │  Semantic Kernel         │                                │
│  │  (by Microsoft)          │                                │
│  │  Enterprise AI, plugins, │                                │
│  │  planners, connectors    │                                │
│  └──────────────────────────┘                                │
│                                                              │
│  JAVASCRIPT / TYPESCRIPT ECOSYSTEM                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │  Vercel     │  │  LangChain  │  │  Mastra              │ │
│  │  AI SDK     │  │  .js        │  │  (TypeScript agent   │ │
│  │             │  │             │  │   framework)         │ │
│  │  Streaming, │  │  JS port of │  │                      │ │
│  │  React/Next │  │  LangChain  │  │  Workflows, tools,   │ │
│  │  focused    │  │  Python     │  │  integrations        │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
│                                                              │
│  JAVA / JVM ECOSYSTEM                                        │
│  ┌─────────────┐  ┌─────────────┐                            │
│  │  Spring AI  │  │  LangChain4j│                            │
│  │             │  │             │                            │
│  │  Spring     │  │  Java port  │                            │
│  │  ecosystem  │  │  of         │                            │
│  │  integration│  │  LangChain  │                            │
│  └─────────────┘  └─────────────┘                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Framework Comparison

| Framework | Language | Best For | RAG | Agents | Maturity | Community |
|-----------|---------|----------|-----|--------|----------|-----------|
| LangChain | Python | General AI apps | ★★★★★ | ★★★★★ | ★★★★☆ | Huge |
| LlamaIndex | Python | Data/RAG apps | ★★★★★ | ★★★☆☆ | ★★★★☆ | Large |
| Semantic Kernel | C#/Python | Enterprise/.NET | ★★★★☆ | ★★★★★ | ★★★★☆ | Growing |
| Vercel AI SDK | TypeScript | Web apps | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | Large |
| Haystack | Python | Production NLP | ★★★★★ | ★★★☆☆ | ★★★★★ | Medium |
| Spring AI | Java | Enterprise Java | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | Growing |
| LangChain.js | TypeScript | JS AI apps | ★★★★☆ | ★★★★☆ | ★★★★☆ | Large |
| Mastra | TypeScript | TS-native agents | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ | Growing |

### What These Frameworks Provide

```
Without a Framework (Raw API calls):
─────────────────────────────────────
  1. Manually construct prompts
  2. Handle API rate limits and retries
  3. Parse responses (hope the JSON is valid)
  4. Manage conversation history yourself
  5. Implement RAG from scratch
  6. Build tool-calling logic
  7. Handle streaming manually
  8. Manage multiple model providers
  Total: ~2000 lines of boilerplate

With a Framework:
──────────────────
  1. Define your chain/pipeline
  2. Configure components
  3. Run it
  Total: ~50-200 lines of application logic
```

### Example: LangChain RAG Pipeline

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Load and chunk documents
loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(documents)

# 2. Create vector store with embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 3. Create retrieval-augmented QA chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
)

# 4. Ask questions
answer = qa_chain.invoke("What is the company's vacation policy?")
print(answer["result"])
```

### Example: LlamaIndex Document Query

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# 1. Load documents (handles PDF, DOCX, TXT, etc.)
documents = SimpleDirectoryReader("./company_docs/").load_data()

# 2. Build index (automatically chunks, embeds, and stores)
index = VectorStoreIndex.from_documents(documents)

# 3. Create query engine
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-4o"),
    similarity_top_k=5,
)

# 4. Query
response = query_engine.query("What are the key metrics for Q3?")
print(response)
```

---

## Layer 3: Agent Layer

### What This Layer Does

Agents are AI systems that can **plan**, **use tools**, **remember**, and **act autonomously**. While a basic chatbot just responds to messages, an agent can break down complex tasks, call APIs, read files, search the web, and iterate until a goal is achieved.

```
Chatbot vs. Agent:

CHATBOT (Reactive):
  User asks question → LLM generates answer → Done

AGENT (Proactive):
  User gives goal → Agent plans steps → Agent executes step 1
  → Observes result → Adjusts plan → Executes step 2
  → ... → Reports result to user

  ┌──────────────────────────────────────────────────────┐
  │                   AGENT LOOP                          │
  │                                                      │
  │    ┌─────────┐    ┌──────────┐    ┌──────────┐      │
  │    │  THINK  │───►│   ACT    │───►│ OBSERVE  │      │
  │    │ (Plan)  │    │ (Tools)  │    │ (Results)│      │
  │    └─────────┘    └──────────┘    └──────┬───┘      │
  │         ▲                                │          │
  │         │         ┌──────────┐           │          │
  │         └─────────┤  DONE?   │◄──────────┘          │
  │                   │ (Check)  │                      │
  │                   └──────────┘                      │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

### Agent Frameworks (2026)

| Framework | Language | Architecture | Best For | Key Feature |
|-----------|---------|-------------|----------|-------------|
| LangGraph | Python | Graph-based workflows | Complex multi-step agents | State machines, cycles |
| CrewAI | Python | Role-based multi-agent | Team simulation | Agent roles & delegation |
| AutoGen (AG2) | Python | Conversational agents | Research, multi-agent chat | Agent conversations |
| OpenAI Agents SDK | Python | OpenAI-native | OpenAI ecosystem | Built-in tool use |
| Anthropic Claude MCP | Multi | Tool integration protocol | Claude ecosystem | Model Context Protocol |
| Semantic Kernel | C#/Python | Plugin-based | Enterprise | Microsoft ecosystem |
| smolagents | Python | Lightweight agents | Simple tool use | HuggingFace ecosystem |
| Google ADK | Python | Google-native | Gemini ecosystem | Multi-agent pipelines |

### Anatomy of an Agent

```
┌──────────────────────────────────────────────────────────────┐
│                        AI AGENT                               │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │   MEMORY     │  │   PLANNING    │  │   TOOLS          │  │
│  │              │  │               │  │                  │  │
│  │ Short-term:  │  │ Goal decomp.  │  │ 🔍 Web Search    │  │
│  │ Conversation │  │ Step planning │  │ 📁 File I/O      │  │
│  │ context      │  │ Prioritization│  │ 🧮 Calculator    │  │
│  │              │  │ Self-critique │  │ 💻 Code Exec     │  │
│  │ Long-term:   │  │ Re-planning   │  │ 🌐 API Calls     │  │
│  │ Vector DB    │  │               │  │ 📊 Data Query    │  │
│  │ summaries    │  │               │  │ 📧 Email Send    │  │
│  └──────────────┘  └───────────────┘  └──────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                    LLM BACKBONE                       │    │
│  │              (GPT-4o, Claude 4, Gemini 2.5 Pro)      │    │
│  │           Generates plans, selects tools, reasons     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                   GUARDRAILS                          │    │
│  │        Safety checks, cost limits, human approval     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Tool Use: How Agents Interact with the World

Modern LLMs can use tools through **function calling**  the model outputs structured requests to invoke specific functions:

```python
# Define tools the agent can use
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# The agent loop
import openai
client = openai.OpenAI()

messages = [
    {"role": "system", "content": "You are a research assistant. Use tools to find and analyze information."},
    {"role": "user", "content": "What is the current GDP of Japan and how does it compare to 5 years ago?"}
]

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    message = response.choices[0].message

    if message.tool_calls:
        # Agent wants to use a tool
        for tool_call in message.tool_calls:
            result = execute_tool(tool_call.function.name,
                                  tool_call.function.arguments)
            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    else:
        # Agent is done, has a final answer
        print(message.content)
        break
```

### Multi-Agent Systems

```
MULTI-AGENT ARCHITECTURE (CrewAI-style):

┌──────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                             │
│              (Manages workflow, delegates tasks)               │
│                                                              │
│    ┌──────────┐    ┌──────────┐    ┌──────────────────┐      │
│    │ RESEARCH │    │ ANALYST  │    │    WRITER        │      │
│    │  AGENT   │───►│  AGENT   │───►│    AGENT         │      │
│    │          │    │          │    │                  │      │
│    │ Tools:   │    │ Tools:   │    │ Tools:           │      │
│    │ - Search │    │ - Python │    │ - File I/O       │      │
│    │ - Browse │    │ - SQL    │    │ - Template       │      │
│    │ - Scrape │    │ - Charts │    │ - Publish        │      │
│    └──────────┘    └──────────┘    └──────────────────┘      │
│                                                              │
│    Input: "Write a market analysis report on EV industry"    │
│                                                              │
│    1. Research Agent gathers data from web and databases      │
│    2. Analyst Agent processes data, creates charts            │
│    3. Writer Agent compiles everything into a report         │
│                                                              │
│    Output: Complete market analysis report with data & charts │
└──────────────────────────────────────────────────────────────┘
```

### Example: LangGraph Agent with State

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

def research_node(state: AgentState) -> AgentState:
    """Research agent gathers information"""
    # Call LLM with research tools
    result = research_llm.invoke(state["messages"])
    return {"messages": [result], "next_step": "analyze"}

def analyze_node(state: AgentState) -> AgentState:
    """Analysis agent processes findings"""
    result = analyst_llm.invoke(state["messages"])
    return {"messages": [result], "next_step": "write"}

def write_node(state: AgentState) -> AgentState:
    """Writer agent produces final output"""
    result = writer_llm.invoke(state["messages"])
    return {"messages": [result], "next_step": "end"}

def router(state: AgentState) -> str:
    """Route to next step based on state"""
    return state.get("next_step", "research")

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("write", write_node)

workflow.set_entry_point("research")
workflow.add_conditional_edges("research", router, {
    "analyze": "analyze",
    "research": "research",  # Can loop back
})
workflow.add_conditional_edges("analyze", router, {
    "write": "write",
    "analyze": "analyze",
})
workflow.add_edge("write", END)

app = workflow.compile()
```

---

## Layer 4: RAG Layer (Retrieval-Augmented Generation)

### What This Layer Does

RAG is the technique of **retrieving relevant information from external sources** and including it in the LLM's context before generating a response. It's how you give an LLM access to your private data, recent information, or specialized knowledge.

```
WHY RAG?

Without RAG:
  User: "What's our Q3 revenue?"
  LLM:  "I don't have access to your company's financial data." ❌

With RAG:
  User: "What's our Q3 revenue?"
  System: [Retrieves relevant docs from vector DB]
  System: [Adds retrieved context to prompt]
  LLM:  "Based on the Q3 financial report, revenue was $4.2M,
         up 15% from Q2..." ✅
```

### The RAG Pipeline

```
THE COMPLETE RAG PIPELINE

INDEXING PHASE (done once, or periodically):
═══════════════════════════════════════════

  Documents          Chunking           Embedding         Vector Store
  ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
  │ PDF     │       │ Chunk 1 │       │ [0.12,  │       │ ■■■■■■  │
  │ DOCX    │──────►│ Chunk 2 │──────►│  0.85,  │──────►│ ■■■■■■  │
  │ HTML    │       │ Chunk 3 │       │  0.03,  │       │ ■■■■■■  │
  │ Code    │       │ ...     │       │  ...]   │       │ ■■■■■■  │
  │ CSV     │       │ Chunk N │       │         │       │ ■■■■■■  │
  └─────────┘       └─────────┘       └─────────┘       └─────────┘
       │                 │                 │                  │
  Load & parse    Split into         Convert text      Store vectors
  documents       overlapping        to numerical       for fast
                  chunks             vectors            similarity
                                                        search

QUERY PHASE (per user query):
═══════════════════════════════

  User Query        Embed Query        Search           Augment + Generate
  ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
  │"What is │       │ [0.15,  │       │ Top-K   │       │ Context │
  │ our PTO │──────►│  0.82,  │──────►│ similar │──────►│ + Query │──► LLM
  │ policy?"│       │  0.01,  │       │ chunks  │       │ = Answer│
  └─────────┘       │  ...]   │       └─────────┘       └─────────┘
                    └─────────┘
```

### Document Processing

The first step is getting documents into a usable format:

| Source | Loader | Challenge | Solution |
|--------|--------|-----------|----------|
| PDF | PyPDF2, pdfplumber, Unstructured | Tables, images, headers | Use Unstructured for complex PDFs |
| DOCX | python-docx | Formatting, embedded objects | Extract text + metadata |
| HTML | BeautifulSoup, Trafilatura | Navigation, ads, boilerplate | Use Trafilatura for article extraction |
| Code | Tree-sitter, custom parsers | Preserve structure | Parse by function/class boundaries |
| CSV/Excel | pandas | Column relationships | Convert to natural language descriptions |
| Markdown | Built-in parsers | Header hierarchy | Split by headers, preserve hierarchy |
| Images/PDFs | Vision LLMs, OCR | Scanned documents | Use GPT-4o or Gemini vision for OCR |

### Chunking Strategies

Chunking is how you split documents into pieces that fit in the LLM's context:

```
CHUNKING STRATEGIES COMPARED:

1. FIXED-SIZE CHUNKING (simple but naive)
   ┌───────────────────────────────────────────┐
   │ chunk 1 (500 tokens) │ chunk 2 (500 to│   │
   │                      │kens)           │   │
   │ "The company was     │ "revenue grew  │   │
   │  founded in 1999..." │  by 15%..."    │   │
   └──────────────────────┴────────────────┘
   Problem: May split sentences or ideas mid-stream

2. RECURSIVE CHARACTER SPLITTING (most common)
   Split by: \n\n → \n → sentence → word
   ┌────────────────────┐ ┌──────────────────┐
   │ Paragraph 1        │ │ Paragraph 2      │
   │ (complete thought) │ │ (complete thought)│
   └────────────────────┘ └──────────────────┘
   With overlap (e.g., 200 chars shared between chunks)

3. SEMANTIC CHUNKING (smarter, slower)
   Uses embeddings to find natural topic boundaries
   ┌─────────────┐ ┌───────────┐ ┌───────────────┐
   │ Topic A     │ │ Topic B   │ │ Topic C       │
   │ (related    │ │ (related  │ │ (related      │
   │  sentences) │ │  sentences)│ │  sentences)   │
   └─────────────┘ └───────────┘ └───────────────┘

4. DOCUMENT-STRUCTURE-AWARE CHUNKING
   Respects headers, sections, code blocks
   ┌──────────────────────┐
   │ ## Section 1          │
   │ Content about topic A │
   ├──────────────────────┤
   │ ## Section 2          │
   │ Content about topic B │
   │ ### Subsection 2.1    │
   │ Details...            │
   └──────────────────────┘

5. AGENTIC / CONTEXTUAL CHUNKING (2025-2026 best practice)
   Uses an LLM to add context to each chunk
   ┌──────────────────────────────────────────┐
   │ Context: "This chunk is from the Q3 2025│
   │ financial report, Section: Revenue"      │
   │                                          │
   │ Original chunk text: "Revenue grew by    │
   │ 15% compared to Q2, reaching $4.2M..."  │
   └──────────────────────────────────────────┘
```

### Embedding Models

Embedding models convert text to vectors (lists of numbers) that capture semantic meaning:

| Model | Dimensions | Max Tokens | Quality | Speed | Cost |
|-------|-----------|-----------|---------|-------|------|
| text-embedding-3-small (OpenAI) | 1536 | 8191 | ★★★★☆ | Fast | $0.02/1M tokens |
| text-embedding-3-large (OpenAI) | 3072 | 8191 | ★★★★★ | Medium | $0.13/1M tokens |
| Voyage-3 (Voyage AI) | 1024 | 32000 | ★★★★★ | Fast | $0.06/1M tokens |
| Gemini Embedding (Google) | 768 | 2048 | ★★★★☆ | Fast | Free tier available |
| BGE-M3 (BAAI) | 1024 | 8192 | ★★★★★ | Medium | Free (open source) |
| Nomic-embed-v2 | 768 | 8192 | ★★★★☆ | Fast | Free (open source) |
| all-MiniLM-L6-v2 | 384 | 512 | ★★★☆☆ | Very Fast | Free (open source) |
| Cohere embed-v4 | 1024 | 128000 | ★★★★★ | Medium | $0.10/1M tokens |

### Advanced RAG Techniques

```
BASIC RAG vs. ADVANCED RAG:

BASIC (Naive RAG):
  Query → Embed → Top-K retrieval → Stuff into prompt → Generate
  Problems: Irrelevant chunks, missed context, no re-ranking

ADVANCED RAG TECHNIQUES:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. QUERY TRANSFORMATION                                    │
│     ┌──────┐    ┌────────────────┐    ┌──────────┐         │
│     │Query │───►│ Rewrite/Expand │───►│ Multiple │         │
│     └──────┘    │ "Also search   │    │ Queries  │         │
│                 │  for synonyms" │    └──────────┘         │
│                 └────────────────┘                          │
│                                                             │
│  2. HYBRID SEARCH                                           │
│     ┌────────────┐  ┌─────────────┐                        │
│     │ Semantic    │  │ Keyword     │──► Combine scores      │
│     │ (vectors)   │  │ (BM25/TF-IDF)│   (Reciprocal Rank  │
│     └────────────┘  └─────────────┘     Fusion)            │
│                                                             │
│  3. RE-RANKING                                              │
│     Top-K results → Cross-encoder re-ranker → Top-N results│
│     (Cohere Rerank, bge-reranker, ColBERT)                  │
│                                                             │
│  4. CONTEXTUAL COMPRESSION                                  │
│     Retrieved chunks → LLM extracts only relevant parts     │
│     → Compressed context sent to final LLM                  │
│                                                             │
│  5. SELF-RAG / CORRECTIVE RAG                               │
│     Generate → Check if retrieval was needed →              │
│     Check if answer is supported → Regenerate if not        │
│                                                             │
│  6. GRAPH RAG                                               │
│     Build a knowledge graph from documents →                │
│     Traverse relationships for richer context                │
│     (Microsoft's GraphRAG approach)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 5: Vector Database Layer

### What This Layer Does

Vector databases store and search high-dimensional vectors (embeddings) efficiently. They're the **memory system** that makes RAG fast  enabling similarity search over millions of documents in milliseconds.

```
Traditional DB vs. Vector DB:

Traditional Database (SQL):
  SELECT * FROM products WHERE category = 'shoes' AND price < 100
  → Exact match on structured fields

Vector Database:
  "Find me comfortable running shoes for wide feet under $100"
  → Semantic similarity search on unstructured text
  → Returns results ranked by meaning, not just keywords
```

### Vector Database Options (2026)

```
VECTOR DATABASE LANDSCAPE:

  MANAGED / CLOUD-NATIVE          SELF-HOSTED / OPEN SOURCE
  ┌─────────────────────┐        ┌──────────────────────────┐
  │                     │        │                          │
  │  ┌──────────────┐   │        │  ┌──────────────────┐    │
  │  │  Pinecone    │   │        │  │  Qdrant          │    │
  │  │  (Serverless)│   │        │  │  (Rust, fast)    │    │
  │  └──────────────┘   │        │  └──────────────────┘    │
  │                     │        │                          │
  │  ┌──────────────┐   │        │  ┌──────────────────┐    │
  │  │  Weaviate    │   │        │  │  ChromaDB        │    │
  │  │  Cloud       │   │        │  │  (Python-native) │    │
  │  └──────────────┘   │        │  └──────────────────┘    │
  │                     │        │                          │
  │  ┌──────────────┐   │        │  ┌──────────────────┐    │
  │  │  Zilliz/     │   │        │  │  Milvus          │    │
  │  │  Milvus Cloud│   │        │  │  (Scalable)      │    │
  │  └──────────────┘   │        │  └──────────────────┘    │
  │                     │        │                          │
  └─────────────────────┘        │  ┌──────────────────┐    │
                                 │  │  pgvector         │    │
  EMBEDDED / LIGHTWEIGHT         │  │  (PostgreSQL ext) │    │
  ┌─────────────────────┐        │  └──────────────────┘    │
  │  ┌──────────────┐   │        │                          │
  │  │  ChromaDB    │   │        │  ┌──────────────────┐    │
  │  │  (in-process)│   │        │  │  FAISS           │    │
  │  └──────────────┘   │        │  │  (Meta, library) │    │
  │  ┌──────────────┐   │        │  └──────────────────┘    │
  │  │  LanceDB     │   │        │                          │
  │  │  (serverless)│   │        │  ┌──────────────────┐    │
  │  └──────────────┘   │        │  │  Weaviate        │    │
  │                     │        │  │  (Docker)        │    │
  └─────────────────────┘        │  └──────────────────┘    │
                                 └──────────────────────────┘
```

### Comparison Table

| Database | Type | Scalability | Ease of Use | Hybrid Search | Pricing Model |
|----------|------|------------|-------------|--------------|---------------|
| Pinecone | Managed | Excellent | Very Easy | Yes | Per-vector storage |
| Weaviate | Both | Excellent | Easy | Yes (BM25) | Open source + cloud |
| Qdrant | Both | Excellent | Easy | Yes | Open source + cloud |
| ChromaDB | Embedded/Server | Good | Very Easy | Basic | Open source |
| pgvector | Extension | Good | Easy (if using PG) | Via PostgreSQL | Free (open source) |
| Milvus | Self-hosted/Cloud | Excellent | Medium | Yes | Open source + cloud |
| FAISS | Library | Good | Medium | No | Free (library) |
| LanceDB | Embedded | Good | Easy | Yes | Open source + cloud |

### When to Use What

```
DECISION TREE: Choosing a Vector Database

START
  │
  ├─ Prototype / small project (< 100K vectors)?
  │   └─► ChromaDB or LanceDB (embedded, zero config)
  │
  ├─ Already using PostgreSQL?
  │   └─► pgvector (add vector search to existing DB)
  │
  ├─ Need managed, zero-ops?
  │   └─► Pinecone (serverless, easiest to operate)
  │
  ├─ Need maximum performance + self-hosted?
  │   └─► Qdrant (Rust, very fast, great API)
  │
  ├─ Need advanced features (multi-modal, BM25)?
  │   └─► Weaviate (feature-rich, good ecosystem)
  │
  └─ Need massive scale (billions of vectors)?
      └─► Milvus / Zilliz (battle-tested at scale)
```

### Example: ChromaDB (Simplest Start)

```python
import chromadb

# Create a client (persisted to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}  # distance metric
)

# Add documents (ChromaDB handles embedding automatically)
collection.add(
    documents=[
        "Python is a high-level programming language.",
        "JavaScript is the language of the web.",
        "Rust is known for memory safety and performance.",
    ],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[
        {"topic": "python"},
        {"topic": "javascript"},
        {"topic": "rust"},
    ]
)

# Query
results = collection.query(
    query_texts=["Which language is best for web development?"],
    n_results=2,
)
print(results["documents"])
# [['JavaScript is the language of the web.',
#   'Python is a high-level programming language.']]
```

---

## Layer 6: LLM Layer

### What This Layer Does

The LLM layer is where you choose **which model** powers your application and **how you access it**  via API or self-hosted.

### Model Selection Strategy

```
MODEL SELECTION DECISION MATRIX:

┌─────────────────┬────────────────────────────────────────────┐
│  REQUIREMENT     │  RECOMMENDED MODELS                       │
├─────────────────┼────────────────────────────────────────────┤
│  Best overall    │  Claude 4 Sonnet, GPT-4o, Gemini 2.5 Pro │
│  quality         │                                           │
├─────────────────┼────────────────────────────────────────────┤
│  Best for coding │  Claude 4 Sonnet, GPT-4.1, Gemini 2.5    │
│                  │  Pro, DeepSeek-V3                         │
├─────────────────┼────────────────────────────────────────────┤
│  Cheapest API    │  GPT-4o-mini, Claude 4 Haiku,             │
│                  │  Gemini 2.0 Flash, DeepSeek-V3            │
├─────────────────┼────────────────────────────────────────────┤
│  Best reasoning  │  Claude 4 Opus, o3/o4-mini, Gemini 2.5   │
│  (hard problems) │  Pro (thinking), DeepSeek-R1              │
├─────────────────┼────────────────────────────────────────────┤
│  Self-hosted     │  Llama 3.1 70B, Qwen 2.5 72B,            │
│  (open weight)   │  DeepSeek-V3, Qwen3-32B                  │
├─────────────────┼────────────────────────────────────────────┤
│  On-device       │  Phi-4-mini, Gemma 3 4B, Llama 3.2 3B    │
│  (edge/mobile)   │                                           │
├─────────────────┼────────────────────────────────────────────┤
│  Privacy/data    │  Any open model, self-hosted              │
│  sovereignty     │                                           │
├─────────────────┼────────────────────────────────────────────┤
│  Long context    │  Gemini 2.5 Pro (1M), Claude 4 (200K),   │
│                  │  GPT-4.1 (1M)                             │
├─────────────────┼────────────────────────────────────────────┤
│  Multimodal      │  GPT-4o, Gemini 2.5, Claude 4 Sonnet     │
│  (image+text)    │                                           │
└─────────────────┴────────────────────────────────────────────┘
```

### API Providers Comparison

| Provider | Models | Pricing (Input/Output per 1M tokens) | Strengths |
|----------|--------|--------------------------------------|-----------|
| OpenAI | GPT-4o, GPT-4o-mini, o3, o4-mini | $2.50/$10 (4o), $0.15/$0.60 (mini) | Largest ecosystem, function calling |
| Anthropic | Claude 4 Opus/Sonnet/Haiku | $15/$75 (Opus), $3/$15 (Sonnet) | Best instruction following, safety |
| Google | Gemini 2.5 Pro/Flash | $1.25/$5 (Pro), $0.15/$0.60 (Flash) | Long context, multimodal, generous free tier |
| DeepSeek | DeepSeek-V3, R1 | $0.27/$1.10 (V3) | Extremely cheap, competitive quality |
| Groq | Llama, Mixtral | $0.05-0.80 (varies) | Ultra-fast inference (LPU hardware) |
| Together AI | Many open models | $0.10-2.00 (varies) | Wide model selection |
| Fireworks AI | Many open models | $0.10-1.60 (varies) | Fast inference, fine-tuning |

### Self-Hosted vs. API: The Trade-off

```
SELF-HOSTED                              API
┌────────────────────────┐    ┌─────────────────────────┐
│ ✅ Full data privacy    │    │ ✅ Zero infrastructure   │
│ ✅ No per-token costs   │    │ ✅ Latest models always  │
│ ✅ Customizable         │    │ ✅ Auto-scaling          │
│ ✅ No rate limits       │    │ ✅ No GPU management     │
│ ✅ Works offline        │    │ ✅ Lower upfront cost    │
│                        │    │                         │
│ ❌ Hardware costs       │    │ ❌ Data leaves your infra│
│ ❌ Ops complexity       │    │ ❌ Per-token pricing     │
│ ❌ Model updates manual │    │ ❌ Rate limits           │
│ ❌ Scaling is hard      │    │ ❌ Vendor lock-in risk   │
│ ❌ GPU procurement      │    │ ❌ Latency variability   │
└────────────────────────┘    └─────────────────────────┘

HYBRID APPROACH (most common in production):
  • Use APIs for frontier quality (complex tasks)
  • Self-host for high-volume, simpler tasks
  • Route based on task complexity and data sensitivity
```

---

## Layer 7: Inference Layer

### What This Layer Does

The inference layer is the **engine** that actually runs the model. It handles loading model weights into GPU memory, processing input tokens, and generating output tokens efficiently. Different inference engines offer different trade-offs in speed, features, and ease of use.

### Inference Engine Landscape

```
INFERENCE ENGINES BY USE CASE:

  LOCAL / DESKTOP                PRODUCTION / SERVER
  ┌──────────────────┐          ┌──────────────────────────┐
  │  ┌────────────┐  │          │  ┌────────────────────┐  │
  │  │  Ollama    │  │          │  │  vLLM              │  │
  │  │  (easiest) │  │          │  │  (industry std)    │  │
  │  └────────────┘  │          │  └────────────────────┘  │
  │  ┌────────────┐  │          │  ┌────────────────────┐  │
  │  │  llama.cpp │  │          │  │  TGI (HuggingFace) │  │
  │  │  (fastest  │  │          │  │  (easy deployment) │  │
  │  │   on CPU)  │  │          │  └────────────────────┘  │
  │  └────────────┘  │          │  ┌────────────────────┐  │
  │  ┌────────────┐  │          │  │  SGLang            │  │
  │  │  LM Studio │  │          │  │  (highest perf)    │  │
  │  │  (GUI app) │  │          │  └────────────────────┘  │
  │  └────────────┘  │          │  ┌────────────────────┐  │
  │  ┌────────────┐  │          │  │  NVIDIA Triton     │  │
  │  │  GPT4All   │  │          │  │  (enterprise)      │  │
  │  │  (simple)  │  │          │  └────────────────────┘  │
  │  └────────────┘  │          │  ┌────────────────────┐  │
  │                  │          │  │  TensorRT-LLM      │  │
  │                  │          │  │  (NVIDIA optimized) │  │
  │                  │          │  └────────────────────┘  │
  └──────────────────┘          └──────────────────────────┘
```

### Comparison

| Engine | Target | GPU Required | Quantization | Batching | API Compatible | Ease of Use |
|--------|--------|-------------|-------------|----------|---------------|-------------|
| Ollama | Desktop | Optional | GGUF (Q4-Q8) | No | OpenAI-compat | ★★★★★ |
| llama.cpp | Desktop/Server | Optional | GGUF (Q2-FP16) | Basic | OpenAI-compat | ★★★☆☆ |
| LM Studio | Desktop (GUI) | Optional | GGUF | No | OpenAI-compat | ★★★★★ |
| vLLM | Server | Yes | AWQ, GPTQ, FP16 | Continuous | OpenAI-compat | ★★★★☆ |
| TGI | Server | Yes | GPTQ, AWQ, EETQ | Continuous | Custom + OpenAI | ★★★★☆ |
| SGLang | Server | Yes | AWQ, GPTQ, FP16 | Continuous | OpenAI-compat | ★★★☆☆ |
| Triton | Enterprise | Yes | All | Continuous | Custom | ★★☆☆☆ |
| TensorRT-LLM | Enterprise | Yes (NVIDIA) | All | Continuous | Custom | ★★☆☆☆ |

### Key Inference Concepts

```
INFERENCE OPTIMIZATION TECHNIQUES:

1. CONTINUOUS BATCHING
   Instead of processing one request at a time, the engine
   dynamically batches multiple requests together.

   Without batching:          With continuous batching:
   [Request 1 ████████]      [Req 1 ██████            ]
   [Request 2     ████████]  [Req 2    ██████████      ]
   [Request 3         ████]  [Req 3      ████          ]
   Sequential, slow           Parallel, GPU stays busy!

2. KV-CACHE MANAGEMENT
   Stores computed attention keys/values to avoid recomputation.
   PagedAttention (vLLM) manages this like virtual memory.

   Request 1:  [System prompt KV cache]──shared──►
   Request 2:  [System prompt KV cache]──shared──►
   Memory savings: ~30-60% when requests share common prefixes

3. SPECULATIVE DECODING
   Uses a small "draft" model to predict several tokens,
   then the large model verifies them in one forward pass.

   Without:  Large Model → 1 token → Large Model → 1 token → ...
   With:     Draft(7B) → 5 tokens → Large(70B) verifies all 5 → accept 3-4
   Speedup: 1.5-2.5× for well-matched draft models

4. QUANTIZATION AT INFERENCE
   Reduce precision to fit in GPU and run faster.
   FP16 → INT8 → INT4 (each step: 2× less memory, ~0-5% quality loss)
```

### Example: Running Models Locally

**Ollama (simplest possible start):**
```bash
# Install Ollama, then:
ollama run llama3.1:8b

# That's it! You now have a local AI chat.
# It also provides an OpenAI-compatible API on port 11434.

# Use from Python:
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Not used but required
)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

**vLLM (production server):**
```bash
# Install vLLM
pip install vllm

# Start a production-grade server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# Provides OpenAI-compatible API with:
# - Continuous batching
# - PagedAttention
# - Streaming
# - Multiple concurrent requests
```

---

## Layer 8: GPU / Hardware Layer

### What This Layer Does

This is the physical foundation. AI models are mathematically intensive  they perform trillions of floating-point operations. GPUs (Graphics Processing Units) are the workhorses because they can perform thousands of operations in parallel.

### Hardware Landscape (2026)

```
GPU TIERS FOR AI:

CONSUMER (Gaming GPUs, also great for AI)
┌──────────────────────────────────────────────────────┐
│  RTX 4060    │  8 GB  │ ~$300  │ 7B models (Q4)     │
│  RTX 4070    │ 12 GB  │ ~$550  │ 14B models (Q4)    │
│  RTX 4070 TiS│ 16 GB  │ ~$800  │ 14B models (Q5-Q8) │
│  RTX 4090    │ 24 GB  │ ~$1600 │ 32B models (Q4)    │
│  RTX 5070 Ti │ 16 GB  │ ~$750  │ 14B models (Q5-Q8) │
│  RTX 5080    │ 16 GB  │ ~$1000 │ 14B models (Q5-Q8) │
│  RTX 5090    │ 32 GB  │ ~$2000 │ 32B models (Q5-Q8) │
└──────────────────────────────────────────────────────┘

PROFESSIONAL (Workstation GPUs)
┌──────────────────────────────────────────────────────┐
│  RTX A6000   │ 48 GB  │ ~$4500 │ 70B models (Q4)    │
│  RTX 6000 Ada│ 48 GB  │ ~$6800 │ 70B models (Q5)    │
└──────────────────────────────────────────────────────┘

DATA CENTER (Server GPUs)
┌──────────────────────────────────────────────────────┐
│  A100 80GB   │ 80 GB  │ ~$15K  │ 70B models (FP16)  │
│  H100 80GB   │ 80 GB  │ ~$30K  │ 70B models + fast  │
│  H200 141GB  │ 141 GB │ ~$35K  │ 405B models (multi)│
│  B100        │ 192 GB │ ~$40K+ │ Frontier models     │
│  B200        │ 192 GB │ ~$50K+ │ Frontier + fast     │
│  GB200 NVL72 │ ~14 TB │ ~$3M+  │ Rack-scale AI      │
│   (72 GPUs)  │ total  │        │                    │
└──────────────────────────────────────────────────────┘

APPLE SILICON (Unified memory, energy efficient)
┌──────────────────────────────────────────────────────┐
│  M3 Pro      │ 18/36 GB│ ~$2000│ 7B-14B models      │
│  M3 Max      │ 36/96 GB│ ~$3200│ 14B-70B models     │
│  M3 Ultra    │ 96/192GB│ ~$7000│ 70B-405B models    │
│  M4 Pro      │ 24/48 GB│ ~$1800│ 14B-32B models     │
│  M4 Max      │ 36/64GB │ ~$2500│ 32B-70B models     │
│  M4 Ultra    │ up to   │ ~$6000│ 70B-405B models    │
│              │ 256 GB  │       │                    │
└──────────────────────────────────────────────────────┘

CLOUD GPU INSTANCES (Pay per hour)
┌──────────────────────────────────────────────────────────┐
│  Provider │ GPU      │ VRAM   │ $/hour │ Best For        │
├───────────┼──────────┼────────┼────────┼─────────────────┤
│  AWS      │ A10G     │ 24 GB  │ ~$1.00 │ 14B inference   │
│  AWS      │ A100     │ 80 GB  │ ~$3.50 │ 70B inference   │
│  AWS      │ H100     │ 80 GB  │ ~$5.00 │ High-perf inf.  │
│  GCP      │ L4       │ 24 GB  │ ~$0.70 │ Budget inference│
│  GCP      │ A100     │ 80 GB  │ ~$3.00 │ 70B inference   │
│  GCP      │ H100     │ 80 GB  │ ~$4.50 │ High-perf inf.  │
│  Azure    │ A100     │ 80 GB  │ ~$3.50 │ 70B inference   │
│  Lambda   │ A100     │ 80 GB  │ ~$1.50 │ Budget GPU      │
│  Lambda   │ H100     │ 80 GB  │ ~$2.50 │ Budget H100     │
│  RunPod   │ A100     │ 80 GB  │ ~$1.60 │ Flexible spot   │
│  Vast.ai  │ Varies   │ Varies │ ~$0.30+│ Cheapest option │
└──────────────────────────────────────────────────────────┘
```

### Key Hardware Concepts

```
WHY GPUs BEAT CPUs FOR AI:

CPU (good at sequential tasks):
  Core 1: [████████████████████████████████]
  Core 2: [████████████████████████████████]
  ...
  Core 16:[████████████████████████████████]
  = 16 very powerful cores

GPU (good at parallel tasks):
  [██][██][██][██][██][██][██][██][██][██][██][██]
  [██][██][██][██][██][██][██][██][██][██][██][██]
  [██][██][██][██][██][██][██][██][██][██][██][██]
  ... (thousands more)
  = 10,000+ smaller cores working together

Matrix multiplication (the core of AI):
  CPU:  Process one row at a time     → 16 operations/cycle
  GPU:  Process entire matrix at once → 10,000+ operations/cycle

Result: GPU is 100-1000× faster for AI workloads!

MEMORY BANDWIDTH (equally important):
  DDR5 RAM:  ~100 GB/s    (CPU memory)
  HBM3e:    ~4,000 GB/s   (H100 GPU memory)

  For LLM inference, memory bandwidth often matters more
  than raw compute because you're constantly reading model
  weights from memory. This is why Apple Silicon (with
  unified memory at ~400 GB/s) is surprisingly good for
  inference despite slower compute.
```

---

## Complete Architecture Diagrams

Now let's put it all together with real-world architectures.

### Architecture 1: Simple Chatbot

```
SIMPLE CHATBOT ARCHITECTURE

┌─────────────────────────────────────────────────────────────┐
│                      USER                                    │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │           WEB FRONTEND (React/Next.js)            │       │
│  │  • Chat interface with streaming                  │       │
│  │  • Markdown rendering                             │       │
│  │  • Conversation history (localStorage)            │       │
│  └────────────────────┬─────────────────────────────┘       │
│                       │ HTTP POST /api/chat                  │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │           API SERVER (Next.js API routes)         │       │
│  │  • Rate limiting                                  │       │
│  │  • Authentication                                 │       │
│  │  • Conversation management                        │       │
│  │  • System prompt injection                        │       │
│  └────────────────────┬─────────────────────────────┘       │
│                       │ API call (streaming)                 │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │              LLM PROVIDER                         │       │
│  │  OpenAI API / Anthropic API / Self-hosted vLLM    │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  Tech Stack:                                                 │
│  • Frontend: Next.js + Vercel AI SDK                        │
│  • Backend: Next.js API routes                               │
│  • LLM: GPT-4o-mini (cheap) or Claude 4 Haiku               │
│  • Hosting: Vercel (frontend + API) + OpenAI (LLM)          │
│  • Cost: ~$20/month for moderate usage                       │
└─────────────────────────────────────────────────────────────┘
```

### Architecture 2: RAG-Powered Knowledge Base

```
RAG KNOWLEDGE BASE ARCHITECTURE

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   INGESTION PIPELINE (batch, runs periodically)                 │
│   ═══════════════════════════════════════════                    │
│                                                                 │
│   ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌────────┐  │
│   │Documents │    │ Processor │    │ Embedding│    │ Vector │  │
│   │          │───►│           │───►│  Model   │───►│   DB   │  │
│   │ PDFs     │    │ Parse     │    │          │    │        │  │
│   │ Docs     │    │ Chunk     │    │ text-    │    │ Qdrant │  │
│   │ Web pages│    │ Clean     │    │ embed-3  │    │ or     │  │
│   │ Confluence│   │ Add metadata│  │ -small   │    │ Pinecone│ │
│   └──────────┘    └───────────┘    └──────────┘    └────────┘  │
│                                                                 │
│   QUERY PIPELINE (real-time, per user query)                    │
│   ═══════════════════════════════════════════                    │
│                                                                 │
│   ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌────────┐  │
│   │  User    │    │  Embed    │    │  Vector  │    │Re-rank │  │
│   │  Query   │───►│  Query    │───►│  Search  │───►│  Top   │  │
│   │          │    │           │    │  (top 20)│    │  5     │  │
│   └──────────┘    └───────────┘    └──────────┘    └────┬───┘  │
│                                                         │      │
│   ┌──────────────────────────────────────────────────────┘      │
│   │                                                             │
│   ▼                                                             │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  AUGMENTED PROMPT                                        │  │
│   │                                                          │  │
│   │  System: "Answer based on the provided context.          │  │
│   │           Cite sources. Say 'I don't know' if            │  │
│   │           the context doesn't contain the answer."       │  │
│   │                                                          │  │
│   │  Context: [Retrieved chunk 1] [Retrieved chunk 2] ...    │  │
│   │                                                          │  │
│   │  User: "What is our vacation policy?"                    │  │
│   └────────────────────────┬─────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              LLM (GPT-4o / Claude Sonnet)                │  │
│   │  Generates answer with citations from context            │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Tech Stack:                                                   │
│   • Framework: LlamaIndex or LangChain                          │
│   • Vector DB: Qdrant (self-hosted) or Pinecone (managed)       │
│   • Embeddings: OpenAI text-embedding-3-small                   │
│   • Re-ranker: Cohere rerank-v3 or bge-reranker                │
│   • LLM: GPT-4o for quality, GPT-4o-mini for cost              │
│   • Document processing: Unstructured                           │
│   • Frontend: Next.js + chat UI                                 │
│   • Cost: ~$100-500/month depending on volume                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture 3: AI Coding Assistant

```
AI CODING ASSISTANT ARCHITECTURE

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  IDE (VS Code / JetBrains)               │   │
│   │                                                         │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │  EXTENSION / PLUGIN                              │    │   │
│   │  │                                                  │    │   │
│   │  │  • Inline completions (ghost text)               │    │   │
│   │  │  • Chat sidebar                                  │    │   │
│   │  │  • Command palette actions                       │    │   │
│   │  │  • Code actions (refactor, explain, test)        │    │   │
│   │  │  • Terminal integration                          │    │   │
│   │  │  • Diff review                                   │    │   │
│   │  └────────────────────┬────────────────────────────┘    │   │
│   └───────────────────────┼──────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                 CONTEXT ENGINE                           │   │
│   │                                                         │   │
│   │  Gathers relevant context from:                         │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │   │
│   │  │ Current  │ │ Open     │ │ Project  │ │ Git       │  │   │
│   │  │ File     │ │ Files    │ │ Structure│ │ History   │  │   │
│   │  │          │ │ (tabs)   │ │ (tree)   │ │ (diffs)   │  │   │
│   │  └──────────┘ └──────────┘ └──────────┘ └───────────┘  │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │   │
│   │  │ LSP      │ │ Symbols/ │ │ Docs &   │ │ Error     │  │   │
│   │  │ Info     │ │ Types    │ │ Comments │ │ Messages  │  │   │
│   │  └──────────┘ └──────────┘ └──────────┘ └───────────┘  │   │
│   └────────────────────┬────────────────────────────────────┘   │
│                        │                                         │
│                        ▼                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                 ROUTING LAYER                            │   │
│   │                                                         │   │
│   │  Inline completion → Fast model (Codestral, GPT-4o-mini)│   │
│   │  Chat / explain   → Quality model (Claude Sonnet, GPT-4o)│  │
│   │  Refactor / debug → Best model (Claude Opus, o3)        │   │
│   │  Simple complete  → Local model (Qwen3-8B via Ollama)   │   │
│   └────────────────────┬────────────────────────────────────┘   │
│                        │                                         │
│              ┌─────────┼──────────┐                              │
│              ▼         ▼          ▼                               │
│   ┌────────────┐ ┌──────────┐ ┌───────────┐                    │
│   │  API LLMs  │ │  Local   │ │ Specialized│                    │
│   │  (Cloud)   │ │  LLMs    │ │ Code Models│                    │
│   │  GPT-4o    │ │  Ollama  │ │ Codestral  │                    │
│   │  Claude    │ │  llama.cpp│ │ StarCoder  │                    │
│   └────────────┘ └──────────┘ └───────────┘                    │
│                                                                 │
│   Tech Stack:                                                   │
│   • Extension: VS Code Extension API + TypeScript                │
│   • Context: Tree-sitter (parsing), LSP (types/symbols)         │
│   • Fast completions: Codestral 22B or local Qwen3-8B            │
│   • Quality chat: Claude 4 Sonnet or GPT-4o                     │
│   • Reasoning: Claude 4 Opus or o3 for hard debugging           │
│   • Hosting: Local + cloud hybrid                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture 4: Multi-Agent System

```
MULTI-AGENT SYSTEM ARCHITECTURE (e.g., Automated Research Pipeline)

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   USER: "Research the impact of AI on healthcare costs          │
│          in the US over the past 5 years and write a report"    │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              ORCHESTRATOR AGENT                          │   │
│   │  (LangGraph / CrewAI / AutoGen)                         │   │
│   │                                                         │   │
│   │  • Decomposes task into subtasks                        │   │
│   │  • Assigns agents                                       │   │
│   │  • Manages dependencies                                │   │
│   │  • Handles failures and retries                         │   │
│   │  • Aggregates results                                   │   │
│   │  Model: GPT-4o or Claude Sonnet (best planning)         │   │
│   └────────┬────────────┬──────────────┬────────────────────┘   │
│            │            │              │                         │
│            ▼            ▼              ▼                         │
│   ┌────────────┐ ┌────────────┐ ┌──────────────┐               │
│   │  RESEARCH  │ │  DATA      │ │  WRITING     │               │
│   │  AGENT     │ │  ANALYST   │ │  AGENT       │               │
│   │            │ │  AGENT     │ │              │               │
│   │  Tools:    │ │  Tools:    │ │  Tools:      │               │
│   │  • Web     │ │  • Python  │ │  • File I/O  │               │
│   │    search  │ │    exec    │ │  • Templates │               │
│   │  • Paper   │ │  • SQL     │ │  • Markdown  │               │
│   │    search  │ │    query   │ │  • Charts    │               │
│   │  • Web     │ │  • Stats   │ │  • PDF       │               │
│   │    scraping│ │    library │ │    export    │               │
│   │            │ │  • Chart   │ │              │               │
│   │  Model:    │ │    gen     │ │  Model:      │               │
│   │  GPT-4o   │ │            │ │  Claude      │               │
│   │            │ │  Model:    │ │  Sonnet      │               │
│   │            │ │  GPT-4o   │ │              │               │
│   └─────┬──────┘ └─────┬──────┘ └──────┬───────┘               │
│         │              │               │                        │
│         ▼              ▼               ▼                        │
│   ┌─────────┐    ┌─────────┐    ┌──────────────┐              │
│   │Raw data │    │Analysis │    │Draft report  │              │
│   │& sources│───►│& charts │───►│with citations│              │
│   └─────────┘    └─────────┘    └──────┬───────┘              │
│                                        │                       │
│                                        ▼                       │
│                              ┌──────────────────┐              │
│                              │  REVIEW AGENT    │              │
│                              │                  │              │
│                              │  • Fact-checks   │              │
│                              │  • Style review  │              │
│                              │  • Completeness  │              │
│                              │  • Citations     │              │
│                              │                  │              │
│                              │  Model: Claude   │              │
│                              │  Opus (critical  │              │
│                              │  thinking)       │              │
│                              └────────┬─────────┘              │
│                                       │                        │
│                                       ▼                        │
│                              ┌──────────────────┐              │
│                              │  FINAL REPORT    │              │
│                              │  (delivered to   │              │
│                              │   user)          │              │
│                              └──────────────────┘              │
│                                                                 │
│   Shared Resources:                                             │
│   • Vector DB (Qdrant)  stores research findings               │
│   • Message Queue  agent-to-agent communication                │
│   • Shared file system  intermediate artifacts                 │
│   • Observability  LangSmith traces all agent steps            │
│                                                                 │
│   Tech Stack:                                                   │
│   • Orchestration: LangGraph (state machines with cycles)       │
│   • Agents: LangChain agents with tool bindings                 │
│   • Models: GPT-4o (research/analysis), Claude (writing/review) │
│   • Tools: Tavily (search), E2B (code exec), Playwright (scrape)│
│   • Observability: LangSmith                                    │
│   • Cost: ~$5-50 per full research report (depends on depth)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Observability, Evaluation, and Safety

### Observability: Seeing What Your AI Is Doing

You can't improve what you can't measure. AI observability tools let you trace every step of your AI pipeline  from user query to final response.

```
OBSERVABILITY STACK:

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  TRACING (What happened?)                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  User Query                                            │  │
│  │   └─► Embedding (23ms, $0.0001)                        │  │
│  │       └─► Vector Search (45ms, 5 results)              │  │
│  │           └─► Re-ranking (120ms, $0.001)               │  │
│  │               └─► LLM Call (1.2s, $0.003)              │  │
│  │                   ├─ Input: 2,847 tokens               │  │
│  │                   ├─ Output: 312 tokens                │  │
│  │                   ├─ Model: gpt-4o                     │  │
│  │                   └─ Total cost: $0.0041               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  TOOLS:                                                      │
│  • LangSmith  Best for LangChain/LangGraph apps            │
│  • Weights & Biases (W&B) Prompts  ML-focused              │
│  • Arize Phoenix  Open-source, embeddings analysis          │
│  • Helicone  API proxy for logging & analytics              │
│  • Braintrust  Evaluation + logging                        │
│  • OpenLLMetry  OpenTelemetry for LLMs                      │
│                                                              │
│  KEY METRICS TO TRACK:                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  • Latency (time to first token, total time)           │  │
│  │  • Token usage (input/output per request)              │  │
│  │  • Cost per request                                    │  │
│  │  • Error rate                                          │  │
│  │  • Retrieval quality (relevance of RAG results)        │  │
│  │  • User satisfaction (thumbs up/down, regenerations)   │  │
│  │  • Hallucination rate                                  │  │
│  │  • Guardrail trigger rate                              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Evaluation: Is Your AI Actually Good?

```
EVALUATION METHODS:

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  1. AUTOMATED BENCHMARKS                                     │
│     • Run standard benchmarks (MMLU, HumanEval, etc.)       │
│     • Good for: Model selection, regression testing          │
│     • Bad for: Measuring real-world quality                  │
│                                                              │
│  2. LLM-AS-JUDGE                                             │
│     • Use a strong LLM (GPT-4o, Claude) to evaluate outputs │
│     • Score on: Accuracy, Helpfulness, Harmlessness          │
│     • Good for: Scalable, consistent evaluation              │
│     • Bad for: May miss subtle errors, judge bias            │
│                                                              │
│     Example prompt for LLM judge:                            │
│     "Rate this response on a scale of 1-5 for:              │
│      - Accuracy: Is the information correct?                 │
│      - Helpfulness: Does it answer the question?             │
│      - Safety: Is it free of harmful content?                │
│      Provide your reasoning before scoring."                 │
│                                                              │
│  3. HUMAN EVALUATION                                         │
│     • Real humans rate outputs (A/B testing, Likert scales)  │
│     • Gold standard for quality                              │
│     • Good for: Ground truth, catching LLM judge blind spots │
│     • Bad for: Expensive, slow, subjective                   │
│                                                              │
│  4. DOMAIN-SPECIFIC EVALUATION                               │
│     • Custom test suites for your specific use case          │
│     • Example: For a medical chatbot, test against           │
│       known correct diagnoses                                │
│     • Good for: Measuring actual business value              │
│                                                              │
│  5. REGRESSION TESTING                                       │
│     • "Golden dataset" of input/expected-output pairs        │
│     • Run before every deployment                            │
│     • Catch regressions from model updates or prompt changes │
│                                                              │
│  TOOLS:                                                      │
│  • Braintrust  Evaluation framework with LLM judges        │
│  • RAGAS  RAG-specific evaluation metrics                   │
│  • DeepEval  Unit testing for LLMs                          │
│  • Promptfoo  Prompt evaluation and comparison              │
│  • Inspect (UK AISI)  Model evaluation framework            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Prompt Engineering Patterns

Effective prompting is as important as model selection. Here are the key patterns:

```
PROMPT ENGINEERING PATTERNS:

1. SYSTEM PROMPT (set behavior and constraints)
   ┌──────────────────────────────────────────────────────────┐
   │  "You are a senior Python developer at a fintech company.│
   │   Always write type-hinted code. Follow PEP 8.          │
   │   If you're unsure about something, say so.             │
   │   Never make up API endpoints or library functions."     │
   └──────────────────────────────────────────────────────────┘

2. FEW-SHOT EXAMPLES (show, don't just tell)
   ┌──────────────────────────────────────────────────────────┐
   │  "Classify the sentiment of these reviews:              │
   │                                                         │
   │   Review: 'Great product, fast shipping!'               │
   │   Sentiment: Positive                                   │
   │                                                         │
   │   Review: 'Terrible quality, broke after one day.'      │
   │   Sentiment: Negative                                   │
   │                                                         │
   │   Review: 'It works as described, nothing special.'     │
   │   Sentiment: Neutral                                    │
   │                                                         │
   │   Review: '{user_input}'                                │
   │   Sentiment:"                                           │
   └──────────────────────────────────────────────────────────┘

3. CHAIN-OF-THOUGHT (force step-by-step reasoning)
   ┌──────────────────────────────────────────────────────────┐
   │  "Solve this problem step by step:                      │
   │   1. First, identify the key information               │
   │   2. Then, determine the approach                      │
   │   3. Show your work                                    │
   │   4. State the final answer"                            │
   └──────────────────────────────────────────────────────────┘

4. OUTPUT FORMAT SPECIFICATION (structured output)
   ┌──────────────────────────────────────────────────────────┐
   │  "Respond in valid JSON with exactly this schema:       │
   │   {                                                     │
   │     'summary': string (max 100 words),                 │
   │     'key_points': string[] (3-5 items),                │
   │     'sentiment': 'positive' | 'negative' | 'neutral',  │
   │     'confidence': number (0-1)                          │
   │   }"                                                    │
   └──────────────────────────────────────────────────────────┘

5. PERSONA + CONSTRAINTS (role-based prompting)
   ┌──────────────────────────────────────────────────────────┐
   │  "You are a medical triage assistant. You MUST:         │
   │   - Never diagnose conditions                           │
   │   - Always recommend seeing a doctor for serious symptoms│
   │   - Use simple language (8th grade reading level)       │
   │   - Ask clarifying questions before giving advice       │
   │   You MUST NOT:                                         │
   │   - Prescribe medications                               │
   │   - Provide emergency medical advice                    │
   │   - Replace professional medical consultation"           │
   └──────────────────────────────────────────────────────────┘

6. META-PROMPTING (prompt that generates prompts)
   ┌──────────────────────────────────────────────────────────┐
   │  "Given this task description, write an optimal system  │
   │   prompt for an LLM that will perform this task. Include│
   │   role definition, constraints, output format, and      │
   │   examples."                                            │
   └──────────────────────────────────────────────────────────┘
```

### Guardrails and Safety

```
GUARDRAILS ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  User Input                                                     │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  INPUT GUARDRAILS                                        │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ Prompt       │  │ PII          │  │ Topic        │   │   │
│  │  │ Injection    │  │ Detection    │  │ Classifier   │   │   │
│  │  │ Detection    │  │ & Redaction  │  │ (off-topic?) │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐                     │   │
│  │  │ Rate         │  │ Content      │                     │   │
│  │  │ Limiting     │  │ Moderation   │                     │   │
│  │  └──────────────┘  └──────────────┘                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────┐                                               │
│  │     LLM      │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  OUTPUT GUARDRAILS                                       │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ Hallucination│  │ Toxicity     │  │ PII in       │   │   │
│  │  │ Detection    │  │ Filter       │  │ Output Check │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐                     │   │
│  │  │ Schema       │  │ Factual      │                     │   │
│  │  │ Validation   │  │ Grounding    │                     │   │
│  │  │ (JSON valid?)│  │ (cites RAG?) │                     │   │
│  │  └──────────────┘  └──────────────┘                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Safe Response → User                                           │
│                                                                 │
│  TOOLS:                                                         │
│  • Guardrails AI  Python framework for LLM guardrails          │
│  • NeMo Guardrails (NVIDIA)  Programmable guardrails           │
│  • Lakera Guard  API-based prompt injection protection         │
│  • Rebuff  Prompt injection detection                          │
│  • Llama Guard 3  Open safety classification model             │
│  • OpenAI Moderation API  Free content moderation              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Guardrail Implementation Example

```python
from guardrails import Guard
from guardrails.hub import DetectPII, ToxicLanguage

# Define guardrails
guard = Guard().use_many(
    DetectPII(
        pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN"],
        on_fail="fix"  # Automatically redact PII
    ),
    ToxicLanguage(
        threshold=0.8,
        on_fail="refrain"  # Block toxic outputs
    ),
)

# Use guard with LLM call
result = guard(
    llm_api=openai.chat.completions.create,
    model="gpt-4o",
    messages=[
        {"role": "user", "content": user_input}
    ],
)

if result.validation_passed:
    print(result.validated_output)
else:
    print("Response blocked by guardrails")
    print(result.error_message)
```

---

## Putting It All Together: A Production Checklist

```
AI APPLICATION PRODUCTION CHECKLIST:

ARCHITECTURE
  □ Choose model(s) based on task requirements
  □ Design for model swappability (don't hard-code one provider)
  □ Plan for fallback models (if primary API is down)
  □ Design caching strategy (cache common queries)
  □ Plan for rate limiting and queuing

DATA PIPELINE
  □ Document ingestion pipeline (for RAG)
  □ Chunking strategy tested and optimized
  □ Embedding model selected and benchmarked
  □ Vector database provisioned and indexed
  □ Incremental update strategy (add/remove/update docs)

QUALITY
  □ Evaluation suite with golden test cases
  □ LLM-as-judge automated evaluation
  □ Human evaluation process defined
  □ A/B testing infrastructure
  □ Prompt version control

SAFETY
  □ Input guardrails (injection, PII, moderation)
  □ Output guardrails (toxicity, hallucination, PII)
  □ Rate limiting per user
  □ Cost limits and alerts
  □ Human escalation path for edge cases

OBSERVABILITY
  □ Request tracing (LangSmith, Helicone, etc.)
  □ Cost tracking per user/feature
  □ Latency monitoring and alerting
  □ Error rate dashboard
  □ User feedback collection

OPERATIONS
  □ Model version pinning (don't auto-upgrade)
  □ Graceful degradation (fallback to simpler model)
  □ Load testing with realistic traffic patterns
  □ Disaster recovery plan
  □ Data retention and privacy compliance (GDPR, etc.)

SECURITY
  □ API key rotation
  □ Network isolation for self-hosted models
  □ Audit logging for all AI interactions
  □ Prompt injection testing
  □ Red team testing for adversarial use cases
```

---

## Technology Selection Cheat Sheet

### By Project Size

| Project Stage | Frontend | Framework | Vector DB | LLM | Inference | Observability |
|--------------|----------|-----------|-----------|-----|-----------|---------------|
| Prototype | Streamlit | LangChain | ChromaDB | GPT-4o-mini | API | Console logs |
| MVP | Next.js | LangChain/LlamaIndex | Qdrant/Pinecone | GPT-4o | API | LangSmith free |
| Production | Next.js/Custom | LangChain + LangGraph | Qdrant/Pinecone | Multi-model | vLLM + API | LangSmith + custom |
| Enterprise | Custom | Semantic Kernel | Weaviate/pgvector | Self-hosted + API | vLLM/TensorRT | Full stack (W&B, Datadog) |

### By Use Case

| Use Case | Stack Recommendation |
|----------|---------------------|
| Internal chatbot | Streamlit + LangChain + ChromaDB + GPT-4o-mini |
| Customer support | Next.js + LangChain + Pinecone + Claude Sonnet + guardrails |
| Code assistant | VS Code extension + context engine + Claude Sonnet + Codestral |
| Document Q&A | LlamaIndex + Qdrant + GPT-4o + Cohere reranker |
| AI writing tool | Next.js + Vercel AI SDK + Claude Sonnet (streaming) |
| Data analysis | Streamlit + LangChain + GPT-4o + Python code execution |
| Multi-agent research | LangGraph + GPT-4o + Tavily + E2B + LangSmith |
| On-device AI | React Native + llama.cpp + Phi-4-mini or Gemma 3 4B |

---

## Common Architecture Anti-Patterns

```
ANTI-PATTERNS TO AVOID:

1. ❌ "STUFF EVERYTHING IN THE PROMPT"
   Putting entire documents in the system prompt instead of using RAG.
   → Will hit context limits, increase costs, reduce quality.
   ✅ Use RAG to retrieve only relevant chunks.

2. ❌ "ONE MODEL FITS ALL"
   Using GPT-4o for everything, even simple classification.
   → Wasteful, slow, expensive.
   ✅ Route tasks to appropriate model sizes.

3. ❌ "NO EVALUATION"
   Shipping without systematic quality testing.
   → You'll learn about problems from angry users.
   ✅ Build evaluation into your CI/CD pipeline.

4. ❌ "TRUST THE MODEL COMPLETELY"
   No guardrails, no output validation.
   → Hallucinations, PII leaks, harmful content.
   ✅ Always validate outputs, especially for high-stakes uses.

5. ❌ "BUILD EVERYTHING FROM SCRATCH"
   Writing your own vector DB, embedding pipeline, etc.
   → Months of engineering for worse results.
   ✅ Use existing tools and frameworks, customize as needed.

6. ❌ "IGNORE COSTS UNTIL PRODUCTION"
   Not tracking token usage during development.
   → Surprise $10,000 cloud bill.
   ✅ Monitor costs from day one, set spending alerts.

7. ❌ "NO CACHING"
   Making API calls for identical or similar queries every time.
   → 10× higher costs, slower responses.
   ✅ Cache at multiple levels (exact match, semantic similarity).

8. ❌ "GIANT MONOLITHIC AGENT"
   One agent that does research, analysis, writing, and review.
   → Unreliable, hard to debug, poor quality.
   ✅ Decompose into specialized agents with clear responsibilities.
```

---

## Quiz: Building an AI Stack

### Question 1
**You're building a customer support chatbot for an e-commerce company. Customers need answers about their orders, return policies, and product specifications. The company has 500+ pages of documentation. Which layers of the AI stack do you need, and what technologies would you choose for each?**

<details>
<summary>Show Answer</summary>

You need the full stack:

1. **UI Layer**: Next.js web chat widget embedded on the support page (Vercel AI SDK for streaming)
2. **Application Layer**: LangChain for orchestrating the RAG pipeline
3. **Agent Layer**: Not needed for V1  a straightforward RAG chain suffices. Add agent capabilities later for order status lookups (tool use)
4. **RAG Layer**: 
   - Document processing: Unstructured to parse PDFs, HTML, DOCX
   - Chunking: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
   - Embedding: text-embedding-3-small (cheap, good quality)
   - Re-ranking: Cohere rerank-v3 for better relevance
5. **Vector DB**: Pinecone (managed, zero ops) or Qdrant (self-hosted, cost savings)
6. **LLM Layer**: GPT-4o-mini for most queries (cheap), GPT-4o for complex cases (route by query complexity)
7. **Inference Layer**: API-based (OpenAI handles it)
8. **Hardware Layer**: Not applicable (using APIs)

Plus:
- **Guardrails**: Input moderation (OpenAI Moderation API), output grounding check (is answer supported by retrieved docs?)
- **Observability**: LangSmith for tracing, Helicone for cost tracking
- **Evaluation**: Golden test set of 100 Q&A pairs, LLM-as-judge weekly evaluation

Estimated cost: $200-500/month for moderate traffic (10K queries/month)
</details>

### Question 2
**What's the difference between an inference engine like vLLM and a model serving platform like Ollama? When would you use each?**

<details>
<summary>Show Answer</summary>

**Ollama**:
- Designed for **local/desktop** use
- Extremely easy setup (`ollama run llama3.1:8b`  that's it)
- Single-user focused (no concurrent request batching)
- Manages model downloads and GGUF quantizations
- OpenAI-compatible API for local development
- **Use when**: Developing locally, personal AI assistant, prototyping, running models on your laptop

**vLLM**:
- Designed for **production server** deployment
- Continuous batching  efficiently handles hundreds of concurrent requests
- PagedAttention for memory-efficient KV cache management
- Tensor parallelism for multi-GPU deployments
- Advanced features: speculative decoding, prefix caching
- OpenAI-compatible API
- **Use when**: Serving models in production, handling concurrent users, needing maximum throughput, deploying on GPU servers

**Key analogy**: Ollama is like a personal car (easy, convenient, single passenger), while vLLM is like a bus system (complex to set up, but efficiently serves many passengers simultaneously).
</details>

### Question 3
**A startup has a RAG application with 10,000 documents. Users report that answers are often irrelevant. What Advanced RAG techniques should they implement, in order of priority?**

<details>
<summary>Show Answer</summary>

Priority order for improving RAG quality:

1. **Re-ranking** (highest impact, easiest to add):
   - Add a cross-encoder re-ranker (Cohere rerank-v3 or bge-reranker-v2)
   - Retrieve top-20 results, re-rank to top-5
   - Typical improvement: 15-25% better relevance

2. **Chunking strategy review**:
   - Current chunks may be too large (diluting information) or too small (missing context)
   - Switch to semantic chunking or document-structure-aware chunking
   - Add contextual descriptions to each chunk using an LLM
   - Typical improvement: 10-20%

3. **Hybrid search** (semantic + keyword):
   - Combine vector search with BM25 keyword search
   - Use Reciprocal Rank Fusion (RRF) to merge results
   - Catches queries where exact keywords matter (product names, IDs, codes)
   - Typical improvement: 10-15%

4. **Query transformation**:
   - Use an LLM to rewrite ambiguous queries
   - Generate multiple search queries from one user question (multi-query RAG)
   - Expand acronyms and domain-specific terms
   - Typical improvement: 5-15%

5. **Better embeddings**:
   - Upgrade from all-MiniLM to text-embedding-3-large or Voyage-3
   - May need to re-embed all documents
   - Typical improvement: 5-10%

Implementing just #1 and #2 often solves most relevance problems.
</details>

### Question 4
**Explain the concept of "model routing"  using different models for different tasks within the same application. Give a concrete example.**

<details>
<summary>Show Answer</summary>

**Model routing** is the practice of directing different types of requests to different AI models based on the complexity, cost sensitivity, or domain of the task.

**Why**: No single model is optimal for all tasks. Using GPT-4o for everything is wasteful; using GPT-4o-mini for everything sacrifices quality on hard problems.

**Concrete example  AI coding assistant:**

```
User action → Router → Model
─────────────────────────────
Autocomplete → Local Qwen3-8B (fast, free, good enough)
"Explain this code" → GPT-4o-mini (simple task, cheap)
"Refactor this module" → Claude 4 Sonnet (complex, needs quality)
"Debug this race condition" → o3 (hard reasoning, worth the cost)
"Generate unit tests" → GPT-4o-mini (template-like, cheap)
```

**How to implement routing:**
1. **Rule-based**: Route based on the type of action (simplest)
2. **Classifier-based**: Use a small model to classify query complexity, then route
3. **Cascading**: Try a cheap model first; if confidence is low, escalate to a more expensive model
4. **Cost-aware**: Set a cost budget and automatically select the best model within budget

Routing can reduce costs by 60-80% while maintaining quality on complex tasks.
</details>

### Question 5
**You're tasked with building an AI system that must NEVER hallucinate medical information. What guardrails and safety measures would you implement?**

<details>
<summary>Show Answer</summary>

For a medical AI system with zero-tolerance for hallucination:

**Input Guardrails:**
- Topic classifier to reject non-medical queries
- PII detection and redaction (HIPAA compliance)
- Prompt injection detection (Lakera Guard or Llama Guard 3)

**Architecture Guardrails:**
- **RAG-only responses**: The model should ONLY answer from retrieved medical documents, never from its parametric knowledge
- **Constrained system prompt**: "Only answer based on the provided medical references. If the references don't contain the answer, say 'I don't have information about this. Please consult a healthcare professional.'"
- **Source citation required**: Every claim must cite a specific document

**Output Guardrails:**
- **Factual grounding check**: A second LLM verifies that every claim in the output is supported by the retrieved documents
- **Medical entity validation**: Check that drug names, dosages, and conditions mentioned actually exist in the knowledge base
- **Confidence scoring**: If the model's response has low confidence, route to a human expert
- **Disclaimer injection**: Always append "This is not medical advice. Please consult a healthcare professional."

**Evaluation:**
- Golden test set of 500+ medical Q&A pairs reviewed by doctors
- Weekly automated evaluation with LLM-as-judge + human review
- Red team testing for adversarial prompts that try to extract harmful advice
- Ongoing monitoring of user feedback for incorrect answers

**Operational:**
- Human-in-the-loop review for any novel questions not in the training set
- Audit logging of all interactions
- Regular knowledge base updates when medical guidelines change
- Incident response plan for incorrect medical information

The key principle: **make it easier for the system to say "I don't know" than to hallucinate.**
</details>

---

## Summary

```
KEY TAKEAWAYS:

1. The AI stack has 8 distinct layers  each with technology choices
   that depend on your requirements.

2. Start simple (chatbot with API), add complexity only when needed
   (RAG, agents, self-hosting).

3. Framework choice matters less than architecture design 
   LangChain, LlamaIndex, and Semantic Kernel all work well.

4. RAG is the most common pattern for production AI  master
   chunking, embedding, retrieval, and re-ranking.

5. Agents are powerful but complex  start with simple tool use,
   add multi-agent orchestration only for genuinely complex tasks.

6. Observability isn't optional  you MUST trace, measure, and
   evaluate your AI system in production.

7. Safety and guardrails are table stakes  prompt injection,
   PII protection, and output validation are required for
   any production system.

8. Model routing (using different models for different tasks)
   can reduce costs by 60-80% while maintaining quality.

9. The best architecture is one you can iterate on quickly 
   start with APIs, move to self-hosting when the economics
   make sense.

10. Anti-patterns like "stuff everything in the prompt" or
    "no evaluation" will catch up with you  invest in
    proper architecture from the start.
```

```
THE MODERN AI STACK IN ONE DIAGRAM:

  ┌─────────── User ───────────┐
  │                             │
  │  ┌──── UI (Next.js) ────┐  │
  │  │  ┌── App (LangChain) │  │
  │  │  │  ┌── Agents ────┐ │  │
  │  │  │  │  ┌── RAG ──┐ │ │  │
  │  │  │  │  │ VectorDB │ │ │  │
  │  │  │  │  │ Qdrant   │ │ │  │
  │  │  │  │  └──────────┘ │ │  │
  │  │  │  └───────────────┘ │  │
  │  │  └────────────────────┘  │
  │  └──────────────────────────┘
  │  ┌──── LLM (GPT-4o) ────┐  │
  │  │  ┌── Inference ─────┐ │  │
  │  │  │  ┌── GPU/HW ──┐  │ │  │
  │  │  │  │  H100/RTX   │  │ │  │
  │  │  │  └─────────────┘  │ │  │
  │  │  └───────────────────┘ │  │
  │  └────────────────────────┘  │
  └──────────────────────────────┘
```

