"""
Dash Evaluations
================

Eval framework for testing Dash's capabilities.

Usage:
    python -m evals
    python -m evals --category security
    python -m evals --verbose
"""

from agno.models.openai import OpenAIResponses

JUDGE_MODEL = OpenAIResponses(id="gpt-5.6-sol")


CATEGORIES: dict[str, dict] = {
    "security": {"type": "judge_binary", "module": "evals.cases.security"},
    "governance": {"type": "judge_binary", "module": "evals.cases.governance"},
    "boundaries": {"type": "judge_binary", "module": "evals.cases.boundaries"},
    "routing": {"type": "reliability", "module": "evals.cases.routing"},
    "accuracy": {"type": "accuracy", "module": "evals.cases.accuracy"},
}
