"""
Multi-Agent Business Research Assistant
Built with LangGraph — 4 specialized agents with HITL and multi-turn conversation
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import TypedDict

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ─────────────────────────────────────────────
# Search tool (Tavily if key present, else mock)
# ─────────────────────────────────────────────
def _make_search_tool():
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            return TavilySearchResults(max_results=5, tavily_api_key=tavily_key)
        except Exception:
            pass

    # Graceful fallback — returns a clearly-labelled mock result so the
    # pipeline still runs end-to-end without a Tavily key.
    @tool
    def mock_search(query: str) -> str:
        """Search for business information (mock — set TAVILY_API_KEY for live data)."""
        return (
            f"[MOCK RESULT for '{query}'] "
            "No live search key configured. "
            "Please set TAVILY_API_KEY in your .env file to retrieve real data. "
            "Sample data: The company reported strong Q1 2025 earnings, "
            "with revenue up 12 % YoY. The CEO recently announced an expansion "
            "into Southeast Asian markets. Competitors include several mid-size "
            "players in the same vertical."
        )

    return mock_search


search_tool = _make_search_tool()


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
class ResearchState(TypedDict):
    # Full conversation history (human + AI messages)
    messages: Annotated[list, add_messages]

    # Current query being processed
    current_query: str

    # Clarity Agent output
    clarity_status: Literal["clear", "needs_clarification"] | None
    clarification_question: str | None

    # Research Agent output
    research_findings: str | None
    confidence_score: float | None

    # Validator Agent output
    validation_result: Literal["sufficient", "insufficient"] | None
    research_attempts: int

    # Final answer
    final_answer: str | None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _format_history(state: ResearchState, max_pairs: int = 6) -> str:
    """Return the last N human/AI message pairs as a readable string."""
    msgs = state["messages"]
    lines: list[str] = []
    for m in msgs[-(max_pairs * 2):]:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines) if lines else "(no prior conversation)"


def _call_llm_json(system: str, user: str) -> dict:
    """Call the LLM, expecting a JSON object back."""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = response.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: extract first JSON object
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"LLM did not return valid JSON:\n{raw}")


# ─────────────────────────────────────────────
# Agent 1 — Clarity Agent
# ─────────────────────────────────────────────
def clarity_agent(state: ResearchState) -> dict:
    """
    Evaluates whether the query is specific enough to research.
    Sets clarity_status → 'clear' | 'needs_clarification'
    """
    history = _format_history(state)
    query = state["current_query"]

    result = _call_llm_json(
        system=(
            "You are a Clarity Agent. Your job is to decide whether a business research "
            "query is precise enough to act on.\n\n"
            "Rules:\n"
            "- 'clear' if a specific company name is present OR context from conversation "
            "  history makes the target company unambiguous.\n"
            "- 'needs_clarification' if the query is vague, names multiple companies without "
            "  focus, or is missing a company name entirely.\n\n"
            "Respond ONLY with a JSON object:\n"
            '{"clarity_status": "clear"|"needs_clarification", '
            '"clarification_question": "<question to ask user, or null if clear>", '
            '"reasoning": "<brief reasoning>"}'
        ),
        user=(
            f"Conversation history:\n{history}\n\n"
            f"Current query: {query}"
        ),
    )

    return {
        "clarity_status": result.get("clarity_status", "needs_clarification"),
        "clarification_question": result.get("clarification_question"),
    }


# ─────────────────────────────────────────────
# Agent 2 — Research Agent
# ─────────────────────────────────────────────
def research_agent(state: ResearchState) -> dict:
    """
    Searches for company data and returns findings + a confidence score.
    """
    history = _format_history(state)
    query = state["current_query"]

    # Build a targeted search query
    search_query_result = _call_llm_json(
        system=(
            "You are a Research Planning Agent. Given a user query and conversation "
            "history, produce 1–3 focused search queries to retrieve business data "
            "(financials, news, leadership, competitors, etc.).\n"
            'Respond ONLY with JSON: {"queries": ["query1", "query2", ...]}'
        ),
        user=f"Conversation history:\n{history}\n\nUser query: {query}",
    )
    queries: list[str] = search_query_result.get("queries", [query])[:3]

    # Execute searches
    raw_results: list[str] = []
    for q in queries:
        try:
            result = search_tool.invoke(q)
            if isinstance(result, list):
                for r in result:
                    raw_results.append(r.get("content", str(r)) if isinstance(r, dict) else str(r))
            else:
                raw_results.append(str(result))
        except Exception as e:
            raw_results.append(f"Search error for '{q}': {e}")

    combined = "\n\n---\n\n".join(raw_results) if raw_results else "No results found."

    # Ask LLM to synthesise findings and assign confidence
    synthesis = _call_llm_json(
        system=(
            "You are a Research Agent. Analyse the raw search results and extract "
            "structured business intelligence relevant to the user's query.\n\n"
            "Assign a confidence_score (0–10) reflecting:\n"
            "  10 = comprehensive, authoritative, up-to-date\n"
            "   6 = adequate for a solid summary\n"
            "   < 6 = incomplete, outdated, or contradictory\n\n"
            "Respond ONLY with JSON:\n"
            '{"findings": "<detailed findings>", "confidence_score": <0-10>, '
            '"gaps": "<what is missing or uncertain>"}'
        ),
        user=(
            f"User query: {query}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Raw search results:\n{combined}"
        ),
    )

    return {
        "research_findings": synthesis.get("findings", combined),
        "confidence_score": float(synthesis.get("confidence_score", 5)),
        "research_attempts": state.get("research_attempts", 0) + 1,
    }


# ─────────────────────────────────────────────
# Agent 3 — Validator Agent
# ─────────────────────────────────────────────
def validator_agent(state: ResearchState) -> dict:
    """
    Assesses research quality and decides if it is sufficient.
    """
    history = _format_history(state)
    result = _call_llm_json(
        system=(
            "You are a Validator Agent. Assess whether the research findings "
            "adequately answer the user's query.\n\n"
            "Mark 'sufficient' if the findings contain specific, relevant, actionable "
            "information. Mark 'insufficient' if data is vague, generic, or clearly "
            "incomplete.\n\n"
            "Respond ONLY with JSON:\n"
            '{"validation_result": "sufficient"|"insufficient", '
            '"reasoning": "<brief explanation>"}'
        ),
        user=(
            f"User query: {state['current_query']}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Research findings:\n{state.get('research_findings', '')}\n\n"
            f"Confidence score: {state.get('confidence_score', 0)}"
        ),
    )

    return {"validation_result": result.get("validation_result", "insufficient")}


# ─────────────────────────────────────────────
# Agent 4 — Synthesis Agent
# ─────────────────────────────────────────────
def synthesis_agent(state: ResearchState) -> dict:
    """
    Produces the final, user-facing answer in a clean, structured format.
    """
    history = _format_history(state)
    attempts = state.get("research_attempts", 1)
    confidence = state.get("confidence_score", 0)
    validation = state.get("validation_result", "unknown")

    response = llm.invoke([
        SystemMessage(content=(
            "You are a Synthesis Agent — a professional business analyst. "
            "Write a clear, well-structured response to the user's question "
            "based on the research findings.\n\n"
            "Format guidelines:\n"
            "• Start with a one-sentence executive summary.\n"
            "• Use short sections with bold headings where appropriate.\n"
            "• Be concise but thorough — aim for quality over length.\n"
            "• If data confidence is low, flag this transparently.\n"
            "• Preserve context: reference earlier parts of the conversation "
            "  if the query is a follow-up.\n"
            "• End with a 'Sources & Confidence' note if relevant.\n"
        )),
        HumanMessage(content=(
            f"User query: {state['current_query']}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Research findings:\n{state.get('research_findings', 'No findings available.')}\n\n"
            f"Confidence score: {confidence}/10  |  "
            f"Validation: {validation}  |  "
            f"Research attempts: {attempts}"
        )),
    ])

    answer = response.content
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ─────────────────────────────────────────────
# Human-in-the-Loop node
# ─────────────────────────────────────────────
def clarification_interrupt(state: ResearchState) -> dict:
    """
    Pauses execution and asks the user for clarification.
    When resumed, the user's response is stored and the query is updated.
    """
    question = state.get("clarification_question") or (
        "Could you please clarify which company you're asking about, "
        "and what specific information you need?"
    )

    # LangGraph interrupt — execution pauses here until .invoke() is called again
    user_response: str = interrupt(question)

    # Merge clarification into the query and reset clarity
    updated_query = f"{state['current_query']} — Clarification: {user_response}"
    return {
        "current_query": updated_query,
        "clarity_status": None,
        "messages": [
            AIMessage(content=f"📋 {question}"),
            HumanMessage(content=user_response),
        ],
    }


# ─────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────
def route_clarity(state: ResearchState) -> str:
    if state.get("clarity_status") == "clear":
        return "research_agent"
    return "clarification_interrupt"


def route_research(state: ResearchState) -> str:
    score = state.get("confidence_score", 0)
    if score >= 6:
        return "synthesis_agent"
    return "validator_agent"


def route_validator(state: ResearchState) -> str:
    validation = state.get("validation_result")
    attempts = state.get("research_attempts", 0)

    if validation == "sufficient" or attempts >= 3:
        return "synthesis_agent"
    return "research_agent"


# ─────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────
def build_graph():
    builder = StateGraph(ResearchState)

    # Nodes
    builder.add_node("clarity_agent", clarity_agent)
    builder.add_node("clarification_interrupt", clarification_interrupt)
    builder.add_node("research_agent", research_agent)
    builder.add_node("validator_agent", validator_agent)
    builder.add_node("synthesis_agent", synthesis_agent)

    # Edges
    builder.add_edge(START, "clarity_agent")

    builder.add_conditional_edges(
        "clarity_agent",
        route_clarity,
        {"research_agent": "research_agent", "clarification_interrupt": "clarification_interrupt"},
    )

    # After clarification, re-check clarity
    builder.add_edge("clarification_interrupt", "clarity_agent")

    builder.add_conditional_edges(
        "research_agent",
        route_research,
        {"synthesis_agent": "synthesis_agent", "validator_agent": "validator_agent"},
    )

    builder.add_conditional_edges(
        "validator_agent",
        route_validator,
        {"research_agent": "research_agent", "synthesis_agent": "synthesis_agent"},
    )

    builder.add_edge("synthesis_agent", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["clarification_interrupt"])


# ─────────────────────────────────────────────
# Public API helpers
# ─────────────────────────────────────────────
graph = build_graph()


def run_query(
    query: str,
    thread_id: str = "default",
    resume_value: str | None = None,
) -> dict[str, Any]:
    """
    Execute (or resume) a research query.

    Parameters
    ----------
    query        : The user's question (ignored when resuming an interrupt).
    thread_id    : Conversation thread identifier (enables multi-turn memory).
    resume_value : If the graph was interrupted for clarification, pass the
                   user's clarification string here to resume.

    Returns
    -------
    dict with keys:
        status        : 'complete' | 'needs_clarification'
        answer        : Final answer string (if complete)
        question      : Clarification question (if needs_clarification)
        state         : Full graph state dict
    """
    config = {"configurable": {"thread_id": thread_id}}

    if resume_value is not None:
        # Resume after human clarification — LangGraph expects Command(resume=...)
        from langgraph.types import Command
        final_state = graph.invoke(Command(resume=resume_value), config=config)
    else:
        # Fresh query — seed state
        initial: ResearchState = {
            "messages": [HumanMessage(content=query)],
            "current_query": query,
            "clarity_status": None,
            "clarification_question": None,
            "research_findings": None,
            "confidence_score": None,
            "validation_result": None,
            "research_attempts": 0,
            "final_answer": None,
        }
        final_state = graph.invoke(initial, config=config)

    # Check if we hit an interrupt
    snapshot = graph.get_state(config)
    if snapshot.next and "clarification_interrupt" in snapshot.next:
        question = "Could you please clarify your question?"
        # Extract the actual interrupt value from pending tasks
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                question = task.interrupts[0].value
                break
        return {"status": "needs_clarification", "question": question, "state": snapshot.values}

    return {
        "status": "complete",
        "answer": final_state.get("final_answer", "No answer generated."),
        "state": final_state,
    }