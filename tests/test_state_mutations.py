# tests/test_state_mutations.py
from src.schemas.state import LeetCodeSolverState, TestResult


def test_user_code_test_results_field_exists():
    """LeetCodeSolverState must accept user_code_test_results as an optional list."""
    state: LeetCodeSolverState = {
        "problem_description": "",
        "problem_title": "",
        "problem_constraints": "",
        "language": "Python",
        "mode": "review",
        "study_output": None,
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
        "execution_start_time": "",
        "execution_end_time": None,
        "total_agents_run": 0,
        "graph_status": "",
        "user_code": None,
        "user_code_review": None,
        "user_code_test_results": None,   # new field
    }
    assert state["user_code_test_results"] is None


def test_user_code_test_results_accepts_list():
    result = TestResult(
        test_case_id="r0",
        passed=True,
        actual_output=42,
        execution_time_ms=10.0,
        error_message=None,
    )
    state: LeetCodeSolverState = {
        "problem_description": "",
        "problem_title": "",
        "problem_constraints": "",
        "language": "Python",
        "mode": "review",
        "study_output": None,
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
        "execution_start_time": "",
        "execution_end_time": None,
        "total_agents_run": 0,
        "graph_status": "",
        "user_code": None,
        "user_code_review": None,
        "user_code_test_results": [result],
    }
    assert len(state["user_code_test_results"]) == 1
    assert state["user_code_test_results"][0]["passed"] is True
