"""
FastAPI Backend — Multi-Agent Business Research Assistant
Run: uvicorn server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import run_query
import uuid

app = FastAPI(title="Business Research Assistant API")

# Allow any frontend (React, mobile, etc.) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────
class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None   # omit to auto-generate a new thread


class ClarifyRequest(BaseModel):
    clarification: str
    thread_id: str             # must match the interrupted thread


class QueryResponse(BaseModel):
    status: str                # "complete" | "needs_clarification"
    thread_id: str
    answer: str | None = None
    question: str | None = None   # clarification question if status == needs_clarification
    confidence_score: float | None = None
    research_attempts: int | None = None


# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Business Research Assistant API is running."}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Submit a new research query.
    Returns either a final answer or a clarification question.
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        result = run_query(req.query, thread_id=thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    state = result.get("state", {})
    return QueryResponse(
        status=result["status"],
        thread_id=thread_id,
        answer=result.get("answer"),
        question=result.get("question"),
        confidence_score=state.get("confidence_score"),
        research_attempts=state.get("research_attempts"),
    )


@app.post("/clarify", response_model=QueryResponse)
def clarify(req: ClarifyRequest):
    """
    Resume a paused (needs_clarification) query with the user's clarification.
    Use the same thread_id returned from /query.
    """
    try:
        result = run_query("", thread_id=req.thread_id, resume_value=req.clarification)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    state = result.get("state", {})
    return QueryResponse(
        status=result["status"],
        thread_id=req.thread_id,
        answer=result.get("answer"),
        question=result.get("question"),
        confidence_score=state.get("confidence_score"),
        research_attempts=state.get("research_attempts"),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
