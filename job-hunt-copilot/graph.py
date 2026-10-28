"""
graph.py — Phase 3: multi-agent Job Hunt Copilot in LangGraph.

                 ┌────────────┐
      START ───> │ supervisor │ <──────────────┐
                 └─────┬──────┘                │
          picks next   │                       │ every worker reports back
       ┌───────────────┼───────────────┐       │
       v               v               v       │
   ┌───────┐      ┌─────────┐     ┌────────┐   │
   │ scout │      │ matcher │     │ writer │ ──┘
   └───────┘      └─────────┘     └────────┘
   finds jobs     scores fit      tailors resume
                       │
          supervisor says FINISH ──> END

Run:  python graph.py "Find React / full-stack fresher jobs in Hyderabad"
"""

import json
import operator
import os
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import tools as T

load_dotenv()

MODEL = os.getenv("MODEL", "claude-sonnet-5-5")              # workers: need quality
SMALL_MODEL = os.getenv("SMALL_MODEL", "claude-haiku-5-5")   # supervisor: simple routing, cheap
MAX_SUPERVISOR_STEPS = 8   # safety brake for the whole graph
TOP_N_TO_TAILOR = 2


# ═════════════════════════════════════════════════════════════ 1. STATE
# The shared notebook every node reads from and writes to.
# Each node returns ONLY the keys it changed; LangGraph merges them in.

class State(TypedDict, total=False):
    task: str
    jobs: list[dict]          # written by scout
    scored: list[dict]        # written by matcher
    tailored: list[dict]      # written by writer
    next: str                 # written by supervisor
    steps: int
    # Annotated[..., operator.add] = APPEND instead of overwrite (a "reducer")
    log: Annotated[list[str], operator.add]


# ═════════════════════════════════════════════════════════════ 2. TOOLS
# Same functions as Phase 1-2, wrapped with @tool so LangChain builds the
# JSON schema from the type hints + docstring automatically.

@tool
def search_jobs(role: str, location: str = "Hyderabad", limit: int = 8) -> str:
    """Search current job listings in India for a role and city."""
    return T.run_tool("search_jobs", {"role": role, "location": location, "limit": limit})


@tool
def fetch_job_description(url: str) -> str:
    """Fetch the full text of a job posting from its url."""
    return T.run_tool("fetch_job_description", {"url": url})


@tool
def read_resume() -> str:
    """Read the candidate's resume summary."""
    return T.run_tool("read_resume", {})


@tool
def search_profile(query: str, k: int = 4) -> str:
    """Search the candidate's detailed project notes by meaning. Use one short query per
    requirement, e.g. 'REST API design'. Results carry a source_id to cite."""
    return T.run_tool("search_profile", {"query": query, "k": k})


class Bullet(BaseModel):
    text: str = Field(description="Action verb + what was built + tech + result")
    source_ids: list[str] = Field(description="source_ids from search_profile backing this bullet")


@tool
def save_tailored_resume(company: str, job_title: str, bullets: list[Bullet]) -> str:
    """Save 3-5 tailored resume bullets for one job. Bullets without real source_ids are rejected."""
    return T.run_tool("save_tailored_resume", {
        "company": company, "job_title": job_title,
        "bullets": [b.model_dump() if isinstance(b, BaseModel) else b for b in bullets],
    })


# ═════════════════════════════════════════ 3. STRUCTURED OUTPUTS
# Each worker must hand back data in a fixed shape, not free text.
# That's what lets the next agent use it reliably.

class Job(BaseModel):
    title: str
    company: str
    location: str = ""
    url: str
    description: str = Field(description="Full description if fetched, else the snippet")


class ScoutResult(BaseModel):
    jobs: list[Job]


class ScoredJob(BaseModel):
    title: str
    company: str
    url: str
    score: int = Field(ge=0, le=100)
    reason: str
    missing_skills: list[str] = []


class MatchResult(BaseModel):
    scored: list[ScoredJob]


class TailoredJob(BaseModel):
    company: str
    job_title: str
    file: str = Field(description="Path returned by save_tailored_resume, or '' if not saved")
    skill_gaps: list[str] = []


class WriterResult(BaseModel):
    tailored: list[TailoredJob]


class Route(BaseModel):
    next: Literal["scout", "matcher", "writer", "FINISH"]
    reason: str = Field(description="One short sentence")


# ═════════════════════════════════════════════════════════ 4. AGENTS

def make_model(name: str):
    return ChatAnthropic(model=name, max_tokens=4000)


_agents: dict = {}


def get_agent(name: str):
    """Build each worker once. A worker = LLM + its own small toolset + prompt."""
    if name in _agents:
        return _agents[name]
    model = make_model(MODEL)
    if name == "scout":
        agent = create_agent(
            model, [search_jobs, fetch_job_description], response_format=ScoutResult,
            system_prompt=(
                "You are the Scout. Find 5-8 jobs matching the task. Fetch full descriptions "
                "for the most promising ones (if fetching fails, keep the snippet). "
                "Return them; do not score them."),
        )
    elif name == "matcher":
        agent = create_agent(
            model, [read_resume, search_profile], response_format=MatchResult,
            system_prompt=(
                "You are the Matcher. Read the resume, then score every job 0-100 for fit. "
                "Use search_profile to check specific requirements. Only count skills with evidence. "
                "List missing skills honestly. Sort best first."),
        )
    elif name == "writer":
        agent = create_agent(
            model, [search_profile, save_tailored_resume], response_format=WriterResult,
            system_prompt=(
                "You are the Writer. For each job given: list its must-have requirements, call "
                "search_profile once per requirement, then write 3-5 bullets using ONLY retrieved "
                "evidence and cite source_ids. Save with save_tailored_resume. If bullets are "
                "rejected, fix or drop them. Requirements with no evidence are skill_gaps."),
        )
    else:
        raise ValueError(name)
    _agents[name] = agent
    return agent


def run_worker(name: str, message: str):
    """Run one worker agent to completion and return its structured result."""
    result = get_agent(name).invoke({"messages": [{"role": "user", "content": message}]})
    return result["structured_response"]


# ══════════════════════════════════════════════════════════ 5. NODES
# A node = a plain function: State in -> partial State update out.

def supervisor(state: State) -> dict:
    steps = state.get("steps", 0) + 1
    summary = (
        f"Task: {state['task']}\n"
        f"Jobs found by scout: {len(state.get('jobs', []))}\n"
        f"Jobs scored by matcher: {len(state.get('scored', []))}\n"
        f"Jobs tailored by writer: {len(state.get('tailored', []))}\n"
        f"Recent log:\n" + "\n".join(state.get("log", [])[-5:])
    )
    router = make_model(SMALL_MODEL).with_structured_output(Route)
    route = router.invoke([
        ("system",
         "You are the supervisor of a job-search team. Workers: scout (finds jobs), "
         "matcher (scores fit), writer (tailors resume bullets for top jobs). "
         "Pick the next worker, or FINISH when the task is done. Usual order: "
         "scout -> matcher -> writer -> FINISH. Skip steps the task doesn't need."),
        ("user", summary),
    ])

    # LLM decides, CODE enforces. Never trust the router blindly.
    nxt = route.next
    if steps > MAX_SUPERVISOR_STEPS:
        nxt = "FINISH"
    elif nxt == "matcher" and not state.get("jobs"):
        nxt = "scout"
    elif nxt == "writer" and not state.get("scored"):
        nxt = "matcher" if state.get("jobs") else "scout"

    return {"next": nxt, "steps": steps, "log": [f"supervisor -> {nxt} ({route.reason})"]}


def scout(state: State) -> dict:
    res = run_worker("scout", state["task"])
    jobs = [j.model_dump() for j in res.jobs]
    return {"jobs": jobs, "log": [f"scout: found {len(jobs)} jobs"]}


def matcher(state: State) -> dict:
    res = run_worker("matcher", "Score these jobs:\n" + json.dumps(state["jobs"], indent=1))
    scored = sorted((s.model_dump() for s in res.scored), key=lambda s: -s["score"])
    T.save_shortlist(scored)  # a guaranteed side effect: done by code, not left to the LLM
    top = scored[0] if scored else None
    note = f"top = {top['company']} ({top['score']})" if top else "no jobs scored"
    return {"scored": scored, "log": [f"matcher: scored {len(scored)}, {note}"]}


def writer(state: State) -> dict:
    by_url = {j["url"]: j for j in state.get("jobs", [])}
    top = [
        {**s, "description": by_url.get(s["url"], {}).get("description", "")}
        for s in state["scored"][:TOP_N_TO_TAILOR]
    ]
    res = run_worker("writer", "Tailor the resume for these jobs:\n" + json.dumps(top, indent=1))
    tailored = [t.model_dump() for t in res.tailored]
    saved = sum(1 for t in tailored if t["file"])
    return {"tailored": tailored, "log": [f"writer: saved {saved}/{len(tailored)} tailored resumes"]}


def route_from_supervisor(state: State) -> str:
    return END if state["next"] == "FINISH" else state["next"]


# ══════════════════════════════════════════════════════════ 6. GRAPH

def build_graph():
    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    g.add_node("scout", scout)
    g.add_node("matcher", matcher)
    g.add_node("writer", writer)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor,
                            ["scout", "matcher", "writer", END])
    for worker in ("scout", "matcher", "writer"):
        g.add_edge(worker, "supervisor")  # always report back
    return g.compile()


def main():
    task = " ".join(sys.argv[1:]) or "Find full-stack / React fresher jobs in Hyderabad and tailor my resume for the best fits."
    graph = build_graph()
    print(f"🎯 Task: {task}\n")

    final = {}
    # stream_mode="updates": print what each node changed, as it happens
    for update in graph.stream({"task": task, "log": []}, stream_mode="updates"):
        for node, changes in update.items():
            for line in changes.get("log", []):
                print(f"[{node:>10}] {line}")
            final.update(changes)

    print("\n" + "=" * 60 + "\n✅ RESULT\n" + "=" * 60)
    for s in final.get("scored", [])[:5]:
        print(f"{s['score']:>3}  {s['title']} @ {s['company']}  missing: {', '.join(s['missing_skills']) or '-'}")
    for t in final.get("tailored", []):
        print(f"📝 {t['file'] or '(not saved)'}  gaps: {', '.join(t['skill_gaps']) or '-'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--draw":
        print(build_graph().get_graph().draw_mermaid())  # paste into mermaid.live
    else:
        main()
