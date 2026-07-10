"""
Dash Self-Improvement Loop
==========================

Runs smoke tests, analyzes failures with GPT-5.6-sol, applies targeted
improvements to instructions and knowledge, then re-runs to verify.

Usage:
    python -m evals improve
    python -m evals improve --rounds 5
    python -m evals improve --dry-run --verbose
"""

from __future__ import annotations

import importlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from dash.paths import BUSINESS_DIR, DASH_DIR, QUERIES_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INSTRUCTIONS_PATH = DASH_DIR / "instructions.py"
METRICS_PATH = BUSINESS_DIR / "metrics.json"
QUERIES_PATH = QUERIES_DIR / "common_queries.sql"

# Files the loop is allowed to modify
ALLOWED_FILES: dict[str, Path] = {
    "instructions.py": INSTRUCTIONS_PATH,
    "metrics.json": METRICS_PATH,
    "common_queries.sql": QUERIES_PATH,
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Change:
    file: str  # key in ALLOWED_FILES
    old_text: str
    new_text: str
    rationale: str


@dataclass
class ImprovementPlan:
    analysis: str
    changes: list[Change] = field(default_factory=list)


@dataclass
class RoundReport:
    round_number: int
    before_pass: int
    before_fail: int
    after_pass: int
    after_fail: int
    analysis: str
    changes_applied: list[str]
    regressions: list[str]
    duration: float


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a prompt engineer improving a multi-agent data analytics system called Dash.

Dash has three agents:
- **Leader**: Routes requests and synthesizes responses. Responds directly to greetings.
- **Analyst**: Runs read-only SQL queries, provides data insights.
- **Engineer**: Creates views and tables in the `dash` schema only.

Your job: analyze smoke test results, diagnose failures, and suggest targeted
changes to the agent instructions or knowledge files to fix them.

Rules:
- Only suggest changes to files you're shown (instructions.py, metrics.json, common_queries.sql).
- Each change must use exact `old_text` strings that appear in the file (for find/replace).
- Keep changes minimal and targeted. Don't rewrite entire sections.
- Don't break the Python syntax of instructions.py.
- Don't remove function definitions (build_leader_instructions, etc.).
- Focus on the highest-impact failures first.
- If a test is passing, don't change what's working.

Respond with JSON in this exact format:
{
    "analysis": "What's working, what's failing, and why (2-3 paragraphs)",
    "changes": [
        {
            "file": "instructions.py",
            "old_text": "exact text to find in the file",
            "new_text": "replacement text",
            "rationale": "why this helps"
        }
    ]
}

If all tests pass or no improvements are needed, return an empty changes array.\
"""


def _build_analysis_prompt(
    results: list,
    instructions_content: str,
    metrics_content: str,
    queries_content: str,
) -> str:
    """Build the user prompt for the improvement LLM."""
    # Format test results
    test_lines: list[str] = []
    for r in results:
        entry = {
            "id": r.test.id,
            "name": r.test.name,
            "group": r.test.group,
            "prompt": r.test.prompt,
            "status": r.status,
            "failures": r.failures,
            "response_preview": r.response[:500] if r.response else "",
        }
        test_lines.append(json.dumps(entry))

    return f"""## Smoke Test Results

{chr(10).join(test_lines)}

## Current Instructions (dash/instructions.py)

```python
{instructions_content}
```

## Business Rules (knowledge/business/metrics.json)

```json
{metrics_content}
```

## Validated Queries (knowledge/queries/common_queries.sql)

```sql
{queries_content[:3000]}
```

Analyze the failing tests and suggest targeted changes to fix them."""


def get_improvement_plan(
    results: list,
    instructions_content: str,
    metrics_content: str,
    queries_content: str,
) -> ImprovementPlan:
    """Call GPT-5.6-sol to analyze failures and suggest improvements."""
    from openai import OpenAI

    client = OpenAI()

    prompt = _build_analysis_prompt(results, instructions_content, metrics_content, queries_content)

    response = client.chat.completions.create(
        model="gpt-5.6-sol",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = json.loads(response.choices[0].message.content or "{}")

    changes = []
    for c in raw.get("changes", []):
        if c.get("file") not in ALLOWED_FILES:
            continue
        changes.append(
            Change(
                file=c["file"],
                old_text=c["old_text"],
                new_text=c["new_text"],
                rationale=c.get("rationale", ""),
            )
        )

    return ImprovementPlan(
        analysis=raw.get("analysis", ""),
        changes=changes,
    )


# ---------------------------------------------------------------------------
# Change application
# ---------------------------------------------------------------------------


def _backup(path: Path, round_num: int) -> Path:
    """Create a backup of a file before modifying it."""
    backup_path = path.parent / f"{path.name}.bak.round-{round_num}"
    shutil.copy2(path, backup_path)
    return backup_path


def _restore(path: Path, round_num: int) -> bool:
    """Restore a file from its backup."""
    backup_path = path.parent / f"{path.name}.bak.round-{round_num}"
    if backup_path.exists():
        shutil.copy2(backup_path, path)
        return True
    return False


def apply_changes(changes: list[Change], round_num: int) -> list[str]:
    """Apply changes to files. Returns list of descriptions of applied changes."""
    applied: list[str] = []
    backed_up: set[str] = set()

    for change in changes:
        path = ALLOWED_FILES[change.file]
        if not path.exists():
            print(f"    WARNING: {change.file} not found, skipping")
            continue

        # Backup on first touch
        if change.file not in backed_up:
            _backup(path, round_num)
            backed_up.add(change.file)

        content = path.read_text()

        if change.old_text not in content:
            print(f"    WARNING: Could not find old_text in {change.file}: {change.old_text[:80]}...")
            continue

        content = content.replace(change.old_text, change.new_text, 1)
        path.write_text(content)
        applied.append(f"{change.file}: {change.rationale}")

    # Validate instructions.py if it was modified
    if "instructions.py" in backed_up:
        try:
            source = INSTRUCTIONS_PATH.read_text()
            compile(source, "instructions.py", "exec")
            # Check that builder functions still exist
            for fn in ["build_leader_instructions", "build_analyst_instructions", "build_engineer_instructions"]:
                if fn not in source:
                    raise ValueError(f"Missing function: {fn}")
        except Exception as e:
            print(f"    ERROR: instructions.py validation failed: {e}")
            print("    Rolling back instructions.py")
            _restore(INSTRUCTIONS_PATH, round_num)
            applied = [a for a in applied if not a.startswith("instructions.py:")]

    return applied


# ---------------------------------------------------------------------------
# Team reload
# ---------------------------------------------------------------------------


def reload_team():
    """Reload all Dash modules so instruction changes take effect."""
    import dash.agents.analyst
    import dash.agents.engineer
    import dash.context.business_rules
    import dash.context.semantic_model
    import dash.instructions
    import dash.team

    importlib.reload(dash.instructions)
    importlib.reload(dash.context.semantic_model)
    importlib.reload(dash.context.business_rules)
    importlib.reload(dash.agents.analyst)
    importlib.reload(dash.agents.engineer)
    importlib.reload(dash.team)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _test_status_map(results: list) -> dict[str, str]:
    """Build {test_id: status} map from results."""
    return {r.test.id: r.status for r in results}


def run_improvement_loop(
    rounds: int = 3,
    verbose: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run the self-improvement loop.

    Returns True if all smoke tests pass after improvement.
    """
    from evals.smoke import run_smoke_tests

    print(f"\nDash Self-Improvement Loop — {rounds} rounds {'(dry run)' if dry_run else ''}")
    print("=" * 60)

    for round_num in range(1, rounds + 1):
        round_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_num}/{rounds}")
        print(f"{'=' * 60}")

        # 1. Baseline
        print("\n  Running smoke tests (baseline)...")
        before_results = run_smoke_tests(verbose=verbose)
        before_map = _test_status_map(before_results)

        before_pass = sum(1 for r in before_results if r.status == "PASS")
        before_fail = len(before_results) - before_pass

        # 2. Check if already passing
        if before_fail == 0:
            print(f"\n  All {before_pass} tests passing. No improvements needed.")
            return True

        print(f"\n  Baseline: {before_pass} passed, {before_fail} failed")

        # 3. Read current state
        print("  Reading current instructions and knowledge...")
        instructions_content = INSTRUCTIONS_PATH.read_text()
        metrics_content = METRICS_PATH.read_text() if METRICS_PATH.exists() else "{}"
        queries_content = QUERIES_PATH.read_text() if QUERIES_PATH.exists() else ""

        # 4. Get improvement plan
        print("  Analyzing failures with GPT-5.6-sol...")
        plan = get_improvement_plan(before_results, instructions_content, metrics_content, queries_content)

        print("\n  Analysis:\n")
        for line in plan.analysis.split("\n"):
            print(f"    {line}")

        if not plan.changes:
            print("\n  No changes suggested. Stopping.")
            return before_fail == 0

        print(f"\n  Proposed changes ({len(plan.changes)}):\n")
        for i, change in enumerate(plan.changes, 1):
            print(f"    {i}. [{change.file}] {change.rationale}")
            if verbose:
                print(f"       old: {change.old_text[:100]}...")
                print(f"       new: {change.new_text[:100]}...")

        # 5. Apply (or skip in dry run)
        if dry_run:
            print("\n  Dry run — skipping application.")
            continue

        print("\n  Applying changes...")
        applied = apply_changes(plan.changes, round_num)

        if not applied:
            print("  No changes could be applied. Stopping.")
            return before_fail == 0

        for desc in applied:
            print(f"    Applied: {desc}")

        # 6. Reload team
        print("  Reloading team...")
        try:
            reload_team()
        except Exception as e:
            print(f"  ERROR reloading team: {e}")
            print("  Rolling back all changes...")
            for file_key in ALLOWED_FILES:
                _restore(ALLOWED_FILES[file_key], round_num)
            reload_team()
            return False

        # 7. Verify
        print("\n  Running smoke tests (verification)...")
        after_results = run_smoke_tests(verbose=verbose)
        after_map = _test_status_map(after_results)

        after_pass = sum(1 for r in after_results if r.status == "PASS")
        after_fail = len(after_results) - after_pass

        # 8. Check for regressions
        regressions: list[str] = []
        for test_id, before_status in before_map.items():
            after_status = after_map.get(test_id, "ERROR")
            if before_status == "PASS" and after_status != "PASS":
                regressions.append(test_id)

        round_duration = round(time.time() - round_start, 1)

        # Print round report
        report = RoundReport(
            round_number=round_num,
            before_pass=before_pass,
            before_fail=before_fail,
            after_pass=after_pass,
            after_fail=after_fail,
            analysis=plan.analysis,
            changes_applied=applied,
            regressions=regressions,
            duration=round_duration,
        )
        _print_round_report(report)

        # Rollback on regression
        if regressions:
            print(f"\n  REGRESSION DETECTED in tests: {', '.join(regressions)}")
            print("  Rolling back all changes...")
            for file_key in ALLOWED_FILES:
                _restore(ALLOWED_FILES[file_key], round_num)
            reload_team()
            print("  Rolled back. Stopping improvement loop.")
            return False

        # All passing? Stop early.
        if after_fail == 0:
            print(f"\n  All tests passing after round {round_num}!")
            return True

    # Final status
    print(f"\n{'=' * 60}")
    print(f"Improvement loop complete ({rounds} rounds)")
    print(f"{'=' * 60}\n")
    return False


def _print_round_report(report: RoundReport) -> None:
    """Print a summary for one round."""
    delta = report.after_pass - report.before_pass
    direction = f"+{delta}" if delta > 0 else str(delta)

    print(f"\n  Round {report.round_number} Summary ({report.duration}s):")
    print(f"    Before: {report.before_pass} passed, {report.before_fail} failed")
    print(f"    After:  {report.after_pass} passed, {report.after_fail} failed ({direction})")
    print(f"    Changes applied: {len(report.changes_applied)}")
    if report.regressions:
        print(f"    REGRESSIONS: {', '.join(report.regressions)}")
