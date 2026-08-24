from unittest.mock import MagicMock, patch

from src.agents.tester import tester_node
from src.schemas.state import TestCase, TestResult


def _base_review_state(**overrides):
    state = {
        "mode": "review",
        "user_code": "class Solution:\n    def twoSum(self, nums, target):\n        for i in range(len(nums)):\n            for j in range(i+1, len(nums)):\n                if nums[i] + nums[j] == target:\n                    return [i, j]",
        "language": "Python",
        "problem_title": "Two Sum",
        "problem_description": "Given nums and target, return indices of two numbers that add up to target.",
        "problem_constraints": "2 <= nums.length <= 10^4",
        "naive_solution": None,
        "optimal_solution": None,
        "total_agents_run": 0,
        "agent_logs": [],
        "error_logs": [],
    }
    state.update(overrides)
    return state


@patch("src.agents.tester.generate_test_cases")
@patch("src.agents.tester.execute_solution")
def test_review_mode_populates_user_code_test_results(mock_exec, mock_gen):
    mock_gen.return_value = [
        TestCase(input_data={"nums": [2, 7, 11, 15], "target": 9},
                 expected_output=[0, 1], case_type="base", description="base")
    ]
    mock_exec.return_value = ([0, 1], 8.0, None)

    result = tester_node(_base_review_state())

    assert "user_code_test_results" in result
    assert len(result["user_code_test_results"]) == 1
    assert result["user_code_test_results"][0]["passed"] is True
    assert result["all_tests_passed"] is True


@patch("src.agents.tester.generate_test_cases")
@patch("src.agents.tester.execute_solution")
def test_review_mode_does_not_write_test_results(mock_exec, mock_gen):
    """test_results (solver-mode field) must remain empty in review mode."""
    mock_gen.return_value = [
        TestCase(input_data={"nums": [3, 2, 4], "target": 6},
                 expected_output=[1, 2], case_type="base", description="base")
    ]
    mock_exec.return_value = ([1, 2], 5.0, None)

    result = tester_node(_base_review_state())

    assert result["test_results"] == []


@patch("src.agents.tester.generate_test_cases")
@patch("src.agents.tester.execute_solution")
def test_review_mode_captures_timeout(mock_exec, mock_gen):
    mock_gen.return_value = [
        TestCase(input_data={"nums": list(range(10000)), "target": 19997},
                 expected_output=[9998, 9999], case_type="large", description="stress")
    ]
    mock_exec.return_value = (None, 5000.0, "Execution timed out after 5s")

    result = tester_node(_base_review_state())

    r = result["user_code_test_results"][0]
    assert r["passed"] is False
    assert "timed out" in r["error_message"].lower()


@patch("src.agents.tester.generate_test_cases")
def test_review_mode_with_no_user_code_sets_empty_results(mock_gen):
    mock_gen.return_value = []
    result = tester_node(_base_review_state(user_code=""))
    assert result["user_code_test_results"] == []


@patch("src.agents.tester.generate_test_cases")
@patch("src.agents.tester.execute_solution")
def test_review_mode_all_tests_passed_false_when_fail(mock_exec, mock_gen):
    mock_gen.return_value = [
        TestCase(input_data={"nums": [2, 7], "target": 9},
                 expected_output=[0, 1], case_type="base", description="base")
    ]
    mock_exec.return_value = ([0, 2], 5.0, None)  # wrong answer → FAIL

    result = tester_node(_base_review_state())

    assert result["user_code_test_results"][0]["passed"] is False
    assert result["all_tests_passed"] is False


@patch("src.agents.tester.generate_test_cases")
@patch("src.agents.tester.execute_solution")
def test_full_mode_unaffected(mock_exec, mock_gen):
    """In full mode, user_code_test_results should be empty and test_results populated normally."""
    mock_gen.return_value = [
        TestCase(input_data={"nums": [2, 7], "target": 9},
                 expected_output=[0, 1], case_type="base", description="base")
    ]
    mock_exec.return_value = ([0, 1], 6.0, None)

    state = {
        "mode": "full",
        "user_code": None,
        "language": "Python",
        "problem_title": "Two Sum",
        "problem_description": "desc",
        "problem_constraints": "2 <= n <= 10^4",
        "naive_solution": {"code": "class Solution:\n    def twoSum(self, nums, target): return [0,1]", "description": "", "language": "Python", "approach": "naive"},
        "optimal_solution": None,
        "total_agents_run": 0,
        "agent_logs": [],
        "error_logs": [],
    }
    result = tester_node(state)

    assert result["user_code_test_results"] == []
    assert len(result["test_results"]) == 1
