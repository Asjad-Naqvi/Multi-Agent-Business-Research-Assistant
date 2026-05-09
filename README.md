# Multi-Agent Business Research Assistant

A production-ready multi-agent system built with **LangGraph** that collects and synthesises business intelligence. Four specialised agents collaborate via a directed graph, with human-in-the-loop support and persistent multi-turn memory.

---

## Architecture

```
START
  │
  ▼
┌──────────────────┐   "clear"      ┌───────────────┐
│  Clarity Agent   │ ──────────────▶│ Research Agent │
│                  │                │  (+ Tavily)    │
│ clarity_status:  │                └───────┬────────┘
│  clear /         │                        │
│  needs_clarif.   │      confidence ≥ 6    │  confidence < 6
└────────┬─────────┘                        │
         │ "needs_clarification"    ┌────────▼────────┐
         ▼                         │ Validator Agent  │
┌─────────────────┐                │                  │
│  HITL Interrupt │                │ validation:      │
│  (ask user)     │                │  sufficient /    │
│                 │                │  insufficient    │
│  ← user input   │                └────────┬─────────┘
└────────┬────────┘                         │
         │ loops back to                    │ sufficient  (or attempts ≥ 3)
         │ Clarity Agent                    │
         │                        ┌─────────▼─────────┐
         │                        │  Synthesis Agent   │
         │                        │                    │
         │                        │  final_answer      │
         │                        └─────────┬──────────┘
         │                                  │
         └──────────────────────────────────▼
                                          END
```

### Agent Responsibilities

| Agent | Input | Key Output | Routes to |
|-------|-------|------------|-----------|
| **Clarity Agent** | query + history | `clarity_status` | Research Agent OR Interrupt |
| **Research Agent** | query + history | `research_findings`, `confidence_score` | Validator OR Synthesis |
| **Validator Agent** | findings + score | `validation_result` | Research (retry) OR Synthesis |
| **Synthesis Agent** | findings + history | `final_answer` | END |

---

## Features

- **4 specialised agents** with clear separation of concerns  
- **Human-in-the-loop** via LangGraph `interrupt()` — pauses when a query is ambiguous and waits for user clarification  
- **Retry loop** — Validator ↔ Research loop (max 3 attempts) for low-confidence results  
- **Multi-turn memory** via `MemorySaver` — follow-up questions like *"What about their CEO?"* carry full context  
- **Tavily search integration** — live web search; graceful mock fallback if key absent  
- **Structured state schema** — typed `TypedDict` tracks all agent outputs  

---

## Setup

### 1. Install dependencies

```bash
pip install langgraph langchain langchain-anthropic langchain-community \
            tavily-python python-dotenv
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Required:
- `ANTHROPIC_API_KEY` — from [console.anthropic.com](https://console.anthropic.com)

Optional (recommended):
- `TAVILY_API_KEY` — from [tavily.com](https://tavily.com) for live search

### 3. Run

**Interactive CLI:**
```bash
python cli.py
```

**Run tests:**
```bash
python tests.py
```

**Programmatic usage:**
```python
from agents import run_query

# Simple query
result = run_query("What are Apple's latest earnings?", thread_id="my-session")
print(result["answer"])

# Multi-turn follow-up (same thread_id preserves context)
result2 = run_query("Who are their main competitors?", thread_id="my-session")
print(result2["answer"])

# Handling clarification interrupts
result = run_query("Tell me about that company", thread_id="t1")
if result["status"] == "needs_clarification":
    print(result["question"])          # "Which company did you mean?"
    user_says = input("You: ")
    result = run_query("", thread_id="t1", resume_value=user_says)
    print(result["answer"])
```

---

## State Schema

```python
class ResearchState(TypedDict):
    messages: list               # Full conversation history
    current_query: str           # Active query
    clarity_status: str | None   # 'clear' | 'needs_clarification'
    clarification_question: str  # Question to ask user (if unclear)
    research_findings: str       # Raw research from search
    confidence_score: float      # 0–10 research quality score
    validation_result: str       # 'sufficient' | 'insufficient'
    research_attempts: int       # Loop counter (max 3)
    final_answer: str            # Synthesised output
```

---

## Files

```
research_assistant/
├── agents.py          # Core — all 4 agents + graph definition
├── cli.py             # Interactive command-line interface
├── tests.py           # Automated test suite (4 scenarios)
├── .env.example       # Environment variable template
└── README.md          # This file
```
