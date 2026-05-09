#!/usr/bin/env python3
"""
Interactive CLI for the Multi-Agent Business Research Assistant.
Run: python cli.py
"""

import os
import sys
import uuid
from dotenv import load_dotenv

load_dotenv()

# Validate API key
if not os.getenv("GROQ_API_KEY"):
    print("❌  GROQ_API_KEY not set. Add it to .env or your environment.")
    sys.exit(1)

from agents import run_query  # noqa: E402 — import after env check

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        🔍  Multi-Agent Business Research Assistant           ║
║        Powered by LangGraph + Claude + Tavily               ║
╚══════════════════════════════════════════════════════════════╝

Agents in the pipeline:
  1. Clarity Agent     — checks if your query is specific enough
  2. Research Agent    — searches for live business data
  3. Validator Agent   — verifies research quality
  4. Synthesis Agent   — composes your final answer

Commands:
  new   — start a fresh conversation thread
  quit  — exit
  
Ask about any company (e.g. "Tell me about Apple's latest earnings")
"""


def _print_section(title: str, content: str, emoji: str = "📄"):
    width = 64
    print(f"\n{'─' * width}")
    print(f"{emoji}  {title}")
    print('─' * width)
    print(content)
    print('─' * width)


def main():
    print(BANNER)
    thread_id = str(uuid.uuid4())
    print(f"🧵  Thread ID: {thread_id[:8]}…  (multi-turn memory is active)\n")

    pending_clarification = False

    while True:
        try:
            if pending_clarification:
                prompt = "🗣️  Your clarification: "
            else:
                prompt = "💬  You: "

            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("👋  Goodbye!")
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            pending_clarification = False
            print(f"\n🆕  New thread started: {thread_id[:8]}…\n")
            continue

        print("\n⏳  Processing", end="", flush=True)

        try:
            if pending_clarification:
                result = run_query("", thread_id=thread_id, resume_value=user_input)
                pending_clarification = False
            else:
                result = run_query(user_input, thread_id=thread_id)

            print(" done.\n")

            if result["status"] == "needs_clarification":
                pending_clarification = True
                _print_section(
                    "Clarity Agent — Clarification Needed",
                    result["question"],
                    "🤔",
                )

            elif result["status"] == "complete":
                state = result["state"]

                # Show agent pipeline summary
                summary_lines = []
                if state.get("clarity_status"):
                    summary_lines.append(f"  Clarity:     {state['clarity_status']}")
                if state.get("confidence_score") is not None:
                    summary_lines.append(f"  Confidence:  {state['confidence_score']:.1f}/10")
                if state.get("validation_result"):
                    summary_lines.append(f"  Validation:  {state['validation_result']}")
                if state.get("research_attempts"):
                    summary_lines.append(f"  Attempts:    {state['research_attempts']}")

                if summary_lines:
                    _print_section(
                        "Agent Pipeline Summary",
                        "\n".join(summary_lines),
                        "📊",
                    )

                _print_section("Research Answer", result["answer"], "✅")

        except Exception as exc:
            print(f"\n\n❌  Error: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()