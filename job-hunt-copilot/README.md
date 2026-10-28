# Job Hunt Copilot — Phase 1: Single Agent

A hand-built agent loop (no framework) with 4 tools. Learn how agents work before LangGraph hides it.

## Setup (5 min)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env
```

1. Put your Anthropic API key in `.env` (console.anthropic.com → API Keys; add ~$5 credit).
2. Paste your resume text into `resume.txt`.
3. Optional: free Adzuna keys (developer.adzuna.com) for real jobs. Without them it uses `sample_jobs.json`.

## Run

```bash
python agent.py "Find React or full-stack fresher jobs in Hyderabad"
```

Watch the terminal: 🔧 = tool call, ↳ = tool result, 💭 = agent thinking. Results land in `shortlist.csv`.

## Files

| File | What it teaches |
|---|---|
| `agent.py` | **The agent loop**: LLM → tool request → run tool → result back → repeat |
| `tools.py` | **Tool calling**: Python function + JSON schema the LLM reads |
| `SYSTEM_PROMPT` in agent.py | **Planning**: the step-by-step process the agent follows |
| `messages` list | **Short-term memory**: the whole conversation, resent every step |
| `MAX_STEPS`, try/except in `run_tool` | **Guardrails**: agents must fail safely |

## Exercises (do these before Phase 2)

1. **Add a tool**: `get_salary_estimate(role)` — return a hard-coded dict. Add the schema, the function, register it, and ask the agent about salaries.
2. **Break it**: make `fetch_job_description` always return an error. Does the agent adapt and use the snippet?
3. **Change the plan**: edit `SYSTEM_PROMPT` so it only shortlists jobs scoring ≥ 70. Observe the difference.
4. **Count cost**: print `response.usage.input_tokens` each step. Notice it grows — the whole history is resent every time.

---

# Phase 2: RAG + Tailored Resumes

The agent now searches your **knowledge base** (detailed project notes) by meaning, and writes resume bullets tailored to each job — every bullet must cite its source.

## Setup

```bash
pip install -r requirements.txt      # adds chromadb
```

1. Write your projects in `knowledge/projects/*.md` (see `knowledge/README.md`). Delete the example file.
2. Build the index: `python rag.py index` (first run downloads an ~80 MB embedding model once).
3. Test retrieval: `python rag.py search "REST API design"`
4. Run the agent as before. Tailored bullets land in `output/tailored/`.

## How it works

```
knowledge/*.md ─chunk by ## heading─> chunks ─embed─> vectors ─> ChromaDB (.chroma/)
job requirement ─embed─> vector ─nearest neighbours─> top chunks + source_ids ─> LLM writes bullets
```

| File | What it teaches |
|---|---|
| `rag.py` `chunk_file` | **Chunking**: split text into focused, searchable pieces |
| `rag.py` `build_index` | **Embeddings + vector DB**: text → numbers → stored |
| `rag.py` `search` | **Retrieval**: find chunks closest in meaning, not exact words |
| `tools.py` `save_tailored_resume` | **Grounding guardrail**: bullets with no real source are rejected |
| `knowledge/` | **Long-term memory**: facts about you that persist across runs |

## Phase 2 exercises

1. **Semantic search**: search `"database indexing"` when your notes say `"added indexes on barcode"`. It still matches — that's embeddings, not keyword search.
2. **Chunk size**: put a whole project under one `##` heading. Re-index and compare search quality. Why is it worse?
3. **Relevance threshold**: in `search_profile`, drop results with `distance` above some value so weak matches don't sneak in. Find a good cut-off by testing.
4. **Break the guardrail**: ask the agent to "make the resume sound more senior". Watch the rejections in the terminal.

---

# Phase 3: Multi-Agent with LangGraph

Same job as Phase 1-2, now split across a **supervisor** and **3 specialist agents**. `agent.py` stays for comparison.

```bash
pip install -r requirements.txt
python rag.py index
python graph.py "Find React fresher jobs in Hyderabad and tailor my resume"
python graph.py --draw        # prints the graph — paste into https://mermaid.live
```

```
START -> supervisor -> scout   -> supervisor
                    -> matcher -> supervisor
                    -> writer  -> supervisor -> FINISH -> END
```

## Concepts in graph.py (read top to bottom — numbered sections)

| Section | Concept |
|---|---|
| 1. STATE | **Shared state**: the notebook all agents read/write. `log` uses a **reducer** (`operator.add`) so entries append instead of overwrite |
| 2. TOOLS | `@tool` turns a typed function + docstring into a tool schema automatically |
| 3. STRUCTURED OUTPUTS | Workers return **Pydantic objects**, not free text, so the next agent can rely on the shape |
| 4. AGENTS | `create_agent` = LLM + toolset + prompt. Each specialist only gets the tools it needs |
| 5. NODES | A node is a plain function: state in → partial update out. **"LLM decides, code enforces"** guards in `supervisor` |
| 6. GRAPH | Nodes + edges + **conditional edges** (the supervisor's choice picks the next node) |

Cost note: the supervisor uses a small, cheap model (`SMALL_MODEL`), since it only routes. Workers use the stronger `MODEL`.

## Phase 3 exercises (one per teammate)

1. **New worker**: add an `outreach` agent that drafts a referral LinkedIn message for the #1 job. Add it to state, node, graph edges, and the supervisor's prompt.
2. **Skip logic**: run `python graph.py "score jobs only, no resume tailoring"`. Does the supervisor skip the writer? If not, improve its prompt.
3. **Parallel**: scout searches one role. Make it search 2 roles at once with two scout nodes running in parallel (hint: two edges out of one node).
4. **Compare**: run `agent.py` and `graph.py` on the same task. Compare speed, cost (tokens) and quality. When is multi-agent NOT worth it?

## Next: Phase 4

Human-in-the-loop (pause for your approval before sending anything), checkpoint memory across runs, and Gmail via MCP.
