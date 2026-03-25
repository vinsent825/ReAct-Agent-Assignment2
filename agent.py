"""
agent.py — General-Purpose ReAct Agent (Reasoning + Acting)

Architecture:
    - ReActAgent class with a generic Thought → Action → Observation loop
    - Tool registry: register any callable as a named tool
    - Stop sequence prevents LLM from hallucinating Observations
    - Regex-based action parsing supports extensible action formats
    - Full conversation history maintained for multi-step reasoning
    - Configurable max iterations as a safety guardrail

Design Decisions:
    - System prompt is injected once; few-shot examples live there
    - Each iteration appends the LLM's Thought+Action AND the real Observation
    - The LLM sees the full trace, enabling reflection and self-correction
    - No hardcoded retry logic — reflection emerges from the prompt design

Usage:
    from agent import ReActAgent
    from tools import create_search_tool

    agent = ReActAgent(system_prompt=SYSTEM_PROMPT, model="gpt-4o-mini")
    agent.register_tool("Search", create_search_tool())
    answer = agent.execute("What fraction of Japan's population is Taiwan's?")
"""

import re
import os
from typing import Callable, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class ReActAgent:
    """
    A general-purpose ReAct (Reasoning + Acting) agent.

    The agent iterates through Thought → Action → Observation cycles,
    using an LLM for reasoning and registered tools for acting.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            system_prompt: The full system prompt including few-shot examples.
            model: OpenAI model identifier.
            max_iterations: Hard limit on reasoning loop iterations.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        """
        self.system_prompt = system_prompt
        self.model = model
        self.max_iterations = max_iterations

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Pass it directly or set OPENAI_API_KEY in your .env file."
            )
        self.client = OpenAI(api_key=resolved_key)

        # Tool registry: name → callable(query_str) → result_str
        self._tools: dict[str, Callable[[str], str]] = {}

        # Conversation history for the current execution
        self._messages: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Tool Management                                                     #
    # ------------------------------------------------------------------ #

    def register_tool(self, name: str, func: Callable[[str], str]) -> None:
        """
        Register a tool that the agent can invoke by name.

        Args:
            name: The action name the LLM will emit (e.g., "Search").
            func: A callable that takes a string argument and returns a string.
        """
        self._tools[name] = func

    @property
    def available_tools(self) -> list[str]:
        """Return a list of registered tool names."""
        return list(self._tools.keys())

    # ------------------------------------------------------------------ #
    #  Action Parsing                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_action(text: str) -> Optional[tuple[str, str]]:
        """
        Parse an action line from the LLM output.

        Expected formats:
            Action: ToolName["argument"]
            Action: ToolName["argument with spaces"]

        Returns:
            A tuple (tool_name, argument) if found, else None.
        """
        # Match pattern: Action: Name["..."] or Action: Name['...']
        pattern = r'Action:\s*(\w+)\s*\[\s*["\'](.+?)["\']\s*\]'
        match = re.search(pattern, text)
        if match:
            return match.group(1), match.group(2)
        return None

    # ------------------------------------------------------------------ #
    #  LLM Interaction                                                     #
    # ------------------------------------------------------------------ #

    def _call_llm(self) -> str:
        """
        Send the current message history to the LLM and return its response.

        Uses 'Observation:' as a stop sequence so the model halts
        after emitting an Action, preventing hallucinated observations.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages,
                temperature=0,
                stop=["Observation:"],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM Error: {type(e).__name__} — {e}"

    # ------------------------------------------------------------------ #
    #  Core Execution Loop                                                 #
    # ------------------------------------------------------------------ #

    def execute(self, query: str, verbose: bool = True) -> str:
        """
        Run the ReAct loop to answer a user query.

        Args:
            query: The user's question.
            verbose: If True, print the full reasoning trace to console.

        Returns:
            The final answer string.
        """
        # Reset history for this execution
        self._messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"[USER QUERY] {query}")
            print(f"{'='*60}")

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            # Step 1: Call LLM — it will produce Thought + Action (or Finish)
            llm_output = self._call_llm()

            if verbose:
                print(f"[LLM OUTPUT]\n{llm_output}")

            # Step 2: Check for Finish action
            action = self.parse_action(llm_output)

            if action and action[0] == "Finish":
                final_answer = action[1]
                if verbose:
                    print(f"\n[FINISH] {final_answer}")
                return final_answer

            # Step 3: Execute tool action
            if action:
                tool_name, tool_arg = action

                if tool_name in self._tools:
                    if verbose:
                        print(f"[ACTION] {tool_name}[\"{tool_arg}\"]")

                    observation = self._tools[tool_name](tool_arg)
                else:
                    observation = (
                        f"Error: Unknown tool '{tool_name}'. "
                        f"Available tools: {self.available_tools}"
                    )

                if verbose:
                    # Truncate long observations for readability
                    display = observation[:500] + "..." if len(observation) > 500 else observation
                    print(f"[OBSERVATION] {display}")

            else:
                # No parseable action found — nudge the model
                observation = (
                    "Error: Could not parse a valid Action from your response. "
                    "Please respond with the format: Action: ToolName[\"argument\"] "
                    "or Action: Finish[\"your final answer\"]"
                )
                if verbose:
                    print(f"[PARSE ERROR] {observation}")

            # Step 4: Append the LLM output + Observation to history
            # The LLM output (Thought + Action) goes as assistant message
            # The Observation goes as user message (simulating environment feedback)
            self._messages.append({
                "role": "assistant",
                "content": llm_output,
            })
            self._messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })

        # Loop exhausted without a Finish action
        fallback = (
            "I was unable to reach a final answer within the maximum number of "
            f"iterations ({self.max_iterations}). Here is my reasoning so far:\n\n"
        )
        # Extract the last LLM output as partial reasoning
        for msg in reversed(self._messages):
            if msg["role"] == "assistant":
                fallback += msg["content"]
                break

        if verbose:
            print(f"\n[MAX ITERATIONS REACHED]\n{fallback}")

        return fallback

    # ------------------------------------------------------------------ #
    #  Trace Export (for report)                                           #
    # ------------------------------------------------------------------ #

    def get_trace(self) -> list[dict]:
        """
        Return the full message history of the last execution.
        Useful for generating the report's console trace.
        """
        return list(self._messages)

    def print_trace(self) -> None:
        """Pretty-print the full trace of the last execution."""
        print(f"\n{'='*60}")
        print("FULL EXECUTION TRACE")
        print(f"{'='*60}")
        for i, msg in enumerate(self._messages):
            role = msg["role"].upper()
            content = msg["content"]
            print(f"\n[{i}] {role}:")
            print(content)
        print(f"\n{'='*60}")
