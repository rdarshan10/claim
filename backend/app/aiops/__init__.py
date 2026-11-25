"""AIOps: an agent that runs the operation, not just the conversation.

signals.py    what the system is doing right now (read-only SQL over its own tables)
detectors.py  which of those are abnormal — fixed thresholds plus a baseline z-score
playbooks.py  the only actions the agent may take, each with a risk tier
graph.py      the LangGraph agent: collect → detect → diagnose → plan →
              auto-fix (low risk) → manager approval (high risk) → verify → record
"""
