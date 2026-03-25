"""
main.py — Execution Script for ReAct Agent Assignment

Runs all three benchmark tasks using a SINGLE ReActAgent instance:
    Task 1: Planning & Quantitative Reasoning (population fraction)
    Task 2: Technical Specificity (display spec comparison)
    Task 3: Resilience & Reflection (startup CEO lookup)

Usage:
    python main.py
    python main.py --provider tavily       # use Tavily instead of DuckDuckGo
    python main.py --model gpt-4o-mini     # specify model
    python main.py --max-iter 7            # override max iterations
"""

import argparse
import sys
from dotenv import load_dotenv

from agent import ReActAgent
from tools import create_search_tool
from prompts import SYSTEM_PROMPT


# ------------------------------------------------------------------ #
#  Benchmark Tasks                                                     #
# ------------------------------------------------------------------ #

TASKS = [
    {
        "id": "Task 1",
        "title": "Planning & Quantitative Reasoning",
        "question": (
            "What fraction of Japan's population is Taiwan's population "
            "as of 2025?"
        ),
    },
    {
        "id": "Task 2",
        "title": "Technical Specificity",
        "question": (
            "Compare the main display specs of iPhone 15 and Samsung S24."
        ),
    },
    {
        "id": "Task 3",
        "title": "Resilience & Reflection Test",
        "question": (
            "Who is the CEO of the startup 'Morphic' AI search?"
        ),
    },
]


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    load_dotenv()

    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="ReAct Agent — Assignment Runner")
    parser.add_argument(
        "--provider",
        type=str,
        default="duckduckgo",
        help="Search provider: duckduckgo (default) or tavily",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Max ReAct loop iterations (default: 5)",
    )
    args = parser.parse_args()

    # --- Initialize search tool ---
    print(f"[SETUP] Search provider : {args.provider}")
    print(f"[SETUP] LLM model       : {args.model}")
    print(f"[SETUP] Max iterations   : {args.max_iter}")
    print()

    try:
        search = create_search_tool(args.provider)
    except (ImportError, ValueError) as e:
        print(f"[ERROR] Failed to initialize search tool: {e}")
        sys.exit(1)

    # --- Initialize ONE agent for all tasks ---
    try:
        agent = ReActAgent(
            system_prompt=SYSTEM_PROMPT,
            model=args.model,
            max_iterations=args.max_iter,
        )
    except ValueError as e:
        print(f"[ERROR] Failed to initialize agent: {e}")
        sys.exit(1)

    agent.register_tool("Search", search)

    # --- Execute all tasks sequentially ---
    results = {}

    for task in TASKS:
        print(f"\n{'#'*60}")
        print(f"# {task['id']}: {task['title']}")
        print(f"{'#'*60}")

        answer = agent.execute(task["question"], verbose=True)
        results[task["id"]] = answer

        # Print the full trace for report purposes
        agent.print_trace()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*60}")

    for task in TASKS:
        tid = task["id"]
        print(f"\n[{tid}] {task['title']}")
        print(f"  Q: {task['question']}")
        print(f"  A: {results[tid]}")

    print(f"\n{'='*60}")
    print("All tasks completed.")


if __name__ == "__main__":
    main()
