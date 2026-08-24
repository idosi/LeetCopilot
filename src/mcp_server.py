import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from src.core.graph import build_graph  # noqa: E402

mcp = FastMCP("LeetCode Solver")

_graph = build_graph()


@mcp.tool()
def solve_algorithm_problem(problem_description: str, language: str = "Python") -> str:
    initial_state = {
        "problem_description": problem_description,
        "problem_title": "",
        "problem_constraints": "",
        "language": language,
        "naive_solution": None,
        "optimal_solution": None,
        "naive_complexity": None,
        "optimal_complexity": None,
        "generated_test_cases": [],
        "test_results": [],
        "all_tests_passed": False,
        "markdown_report": None,
        "current_node": "",
        "supervisor_routing": "",
        "error_logs": [],
        "agent_logs": [],
        "execution_start_time": datetime.now().isoformat(),
        "execution_end_time": None,
        "total_agents_run": 0,
        "graph_status": "running",
    }

    try:
        result = _graph.invoke(initial_state)
        report = result.get("markdown_report")
        if report:
            return report
        errors = result.get("error_logs", [])
        return "\n".join(errors) if errors else "Graph completed but produced no report."
    except Exception as exc:
        return str(exc)


if __name__ == "__main__":
    mcp.run()
