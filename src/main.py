import sys
from datetime import datetime, timezone
from pathlib import Path

from src.core.graph import build_graph
from src.schemas.state import LeetCodeSolverState

SAMPLE_PROBLEM_TITLE = "Two Sum"

SAMPLE_PROBLEM_DESCRIPTION = """\
Given an array of integers `nums` and an integer `target`, return indices of the two numbers
such that they add up to `target`. You may assume that each input would have exactly one solution,
and you may not use the same element twice. You can return the answer in any order.\
"""

SAMPLE_PROBLEM_CONSTRAINTS = """\
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
- Only one valid answer exists.\
"""


def build_initial_state(
    title: str,
    description: str,
    constraints: str,
) -> LeetCodeSolverState:
    return LeetCodeSolverState(
        problem_title=title,
        problem_description=description,
        problem_constraints=constraints,
        naive_solution=None,
        optimal_solution=None,
        naive_complexity=None,
        optimal_complexity=None,
        generated_test_cases=[],
        test_results=[],
        all_tests_passed=False,
        markdown_report=None,
        current_node="",
        supervisor_routing="",
        error_logs=[],
        agent_logs=[],
        execution_start_time=datetime.now(timezone.utc).isoformat(),
        execution_end_time=None,
        total_agents_run=0,
        graph_status="running",
    )


def run(title: str, description: str, constraints: str, output_path: str | None = None) -> str:
    graph = build_graph()
    initial_state = build_initial_state(title, description, constraints)
    final_state: LeetCodeSolverState = graph.invoke(initial_state)

    report = final_state.get("markdown_report") or ""
    status = final_state.get("graph_status", "unknown")
    agents_run = final_state.get("total_agents_run", 0)
    errors = final_state.get("error_logs", [])

    print(report)
    print(f"\n---\nStatus: {status} | Agents run: {agents_run}", file=sys.stderr)
    if errors:
        print("Errors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
        print(f"Report saved to {output_path}", file=sys.stderr)

    return report


if __name__ == "__main__":
    run(
        title=SAMPLE_PROBLEM_TITLE,
        description=SAMPLE_PROBLEM_DESCRIPTION,
        constraints=SAMPLE_PROBLEM_CONSTRAINTS,
        output_path="report.md",
    )
