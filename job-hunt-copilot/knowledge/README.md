# knowledge/ — the agent's long-term knowledge about you

Write one markdown file per project or experience. The more specific you are,
the better the tailored bullets. The agent can ONLY claim what's written here.

Rules for good RAG:
- Start the file with `# Project Name`.
- Split it into `## Sections` (each section becomes one searchable chunk).
- Include tech names, what YOU did, and real numbers (users, speed-ups, rows).
- One topic per section. Short, focused sections retrieve better than long ones.

After editing, rebuild the index:  `python rag.py index`
Check retrieval:                    `python rag.py search "REST APIs"`
