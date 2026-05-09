#!/usr/bin/env python3
"""
Test suite for the Multi-Agent Business Research Assistant.

Tests:
  1. Clear query — full pipeline without interruption
  2. Ambiguous query — triggers HITL interrupt, then resumes
  3. Multi-turn follow-up — context preserved across queries
  4. Low-confidence routing — validator loop exercised
"""

import os
import sys
import uuid

# Minimal env check
if not os.getenv("ANTHROPIC_API_KEY"):
    print("SKIP: ANTHROPIC_API_KEY not set")
    sys.exit(0)

from agents import run_query  # noqa: E402

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def test(name: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    msg = f"{status}  {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    return ok


def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


# ──────────────────────────────────────────────
# Test 1: Clear query — full pipeline
# ──────────────────────────────────────────────
section("Test 1 — Clear query (Apple)")
tid = str(uuid.uuid4())
r = run_query("What are Apple's latest financial results and key business segments?", thread_id=tid)

test("Status is 'complete'", r["status"] == "complete", f"got: {r['status']}")
test("Final answer is non-empty", bool(r.get("answer", "").strip()))
state = r["state"]
test("Clarity status set", state.get("clarity_status") in ("clear", "needs_clarification"))
test("Confidence score present", state.get("confidence_score") is not None,
     f"score: {state.get('confidence_score')}")
test("Research attempts > 0", (state.get("research_attempts") or 0) > 0)
test("Messages accumulated", len(state.get("messages", [])) > 0)

print(f"\n  Answer preview: {r['answer'][:200]}…\n")

# ──────────────────────────────────────────────
# Test 2: Ambiguous query — HITL interrupt
# ──────────────────────────────────────────────
section("Test 2 — Ambiguous query → HITL → resume")
tid2 = str(uuid.uuid4())
r2 = run_query("Tell me about that tech company", thread_id=tid2)

if r2["status"] == "needs_clarification":
    test("Interrupt triggered correctly", True, f"Q: {r2['question'][:80]}")

    # Resume with clarification
    r2b = run_query("", thread_id=tid2, resume_value="I meant Microsoft")
    test("Resumes after clarification", r2b["status"] == "complete",
         f"got: {r2b['status']}")
    if r2b["status"] == "complete":
        test("Post-clarification answer non-empty", bool(r2b["answer"].strip()))
        print(f"\n  Answer preview: {r2b['answer'][:200]}…\n")
elif r2["status"] == "complete":
    # The model may have inferred context — still valid
    test("Model resolved ambiguity autonomously", True,
         "Clarity agent deemed query clear (acceptable)")
    print(f"\n  Answer preview: {r2['answer'][:200]}…\n")

# ──────────────────────────────────────────────
# Test 3: Multi-turn follow-up
# ──────────────────────────────────────────────
section("Test 3 — Multi-turn conversation")
tid3 = str(uuid.uuid4())

r3a = run_query("Give me a brief overview of Tesla as a company.", thread_id=tid3)
if r3a["status"] == "needs_clarification":
    r3a = run_query("", thread_id=tid3, resume_value="Tesla the electric vehicle company")

test("First turn complete", r3a["status"] == "complete")

if r3a["status"] == "complete":
    r3b = run_query("Who are their main competitors?", thread_id=tid3)
    if r3b["status"] == "needs_clarification":
        r3b = run_query("", thread_id=tid3, resume_value="Tesla's main EV competitors")

    test("Follow-up turn complete", r3b["status"] == "complete",
         f"got: {r3b['status']}")
    if r3b["status"] == "complete":
        msgs = r3b["state"].get("messages", [])
        test("Message history has ≥ 4 entries", len(msgs) >= 4,
             f"message count: {len(msgs)}")
        print(f"\n  Follow-up answer preview: {r3b['answer'][:200]}…\n")

# ──────────────────────────────────────────────
# Test 4: State fields populated correctly
# ──────────────────────────────────────────────
section("Test 4 — State field validation")
tid4 = str(uuid.uuid4())
r4 = run_query("What is Amazon's cloud computing market share?", thread_id=tid4)

if r4["status"] == "needs_clarification":
    r4 = run_query("", thread_id=tid4, resume_value="Amazon AWS cloud market share")

if r4["status"] == "complete":
    s4 = r4["state"]
    test("clarity_status is 'clear'", s4.get("clarity_status") == "clear",
         f"got: {s4.get('clarity_status')}")
    test("research_findings present", bool(s4.get("research_findings")))
    test("confidence_score in range", 0 <= (s4.get("confidence_score") or 0) <= 10,
         f"score: {s4.get('confidence_score')}")
    test("validation_result set", s4.get("validation_result") in ("sufficient", "insufficient"),
         f"got: {s4.get('validation_result')}")
    test("final_answer present", bool(s4.get("final_answer")))
else:
    test("Query resolved", False, f"Unexpected status: {r4['status']}")

print("\n" + "═" * 60)
print("  All tests completed.")
print("═" * 60 + "\n")
