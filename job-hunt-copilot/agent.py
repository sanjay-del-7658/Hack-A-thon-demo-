"""
agent.py — the agent loop. This is the whole idea of an "agent":

    while not done:
        1. send conversation + tool list to the LLM
        2. if the LLM asks for tools -> run them, add results to conversation
        3. if the LLM gives a final answer -> stop

Run:  python agent.py "Find React / full-stack fresher jobs in Hyderabad"
"""

import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from tools import TOOL_SCHEMAS, run_tool

load_dotenv()

MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
MAX_STEPS = 30  # safety brake (raised: tailoring needs more tool calls) so a confused agent can't loop forever

SYSTEM_PROMPT = """You are a job-search agent helping a fresher software developer.

Your process:
1. Read the resume first.
2. Search for jobs matching the user's request.
3. For the 3-5 most promising jobs, fetch the full description.
4. Score each job 0-100 for fit. Be honest: only count skills actually in the resume.
   List skills the job wants that the resume lacks.
5. Save the scored shortlist with save_shortlist.
6. For the top 2 jobs, tailor the resume:
   a. List the job's must-have requirements.
   b. For each requirement, call search_profile with a short focused query.
   c. Write 3-5 bullets (action verb + what you built + tech + result) using ONLY
      the retrieved evidence. Cite each bullet's source_ids.
   d. Save with save_tailored_resume. If bullets are rejected, fix or drop them.
7. Reply with a short ranked summary and the tailored files saved.

Rules:
- Never invent skills, numbers or experience. If search_profile finds nothing
  relevant for a requirement, that requirement is a skill gap — say so.
- If a tool returns an error, adapt (e.g. use the snippet) instead of giving up.
"""


def run_agent(task: str) -> str:
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    messages = [{"role": "user", "content": task}]

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Keep the LLM's reply in the conversation (its "memory" for this task).
        messages.append({"role": "assistant", "content": response.content})

        # Print what the agent is thinking/doing — poor man's observability.
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[step {step}] 💭 {block.text.strip()[:300]}")
            elif block.type == "tool_use":
                print(f"[step {step}] 🔧 {block.name}({json.dumps(block.input)[:150]})")

        # No tool requested -> the agent is done.
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        # Run every tool the LLM asked for and send results back.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                print(f"           ↳ {result[:150]}...")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # links result to the request
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: hit MAX_STEPS without finishing."


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "Find full-stack / React fresher jobs in Hyderabad and shortlist the best fits."
    print(f"🎯 Task: {task}")
    answer = run_agent(task)
    print("\n" + "=" * 60 + "\n✅ FINAL ANSWER\n" + "=" * 60)
    print(answer)
