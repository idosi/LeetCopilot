# Review Mode TLE Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Code Review mode run user code through the Tester before review, and force the Review Agent to detect and label TLE / suboptimal complexity.

**Architecture:** Add `user_code_test_results` to state; extend tester to test `user_code` when `mode == "review"`; reroute the graph so review mode flows `tester → code_review`; strengthen the code_review system prompt and inject runtime test summaries into the user prompt.

**Tech Stack:** Python, LangGraph (`StateGraph`), Pydantic, pytest

---

## File Map

| File | Change |
|------|--------|
| `src/schemas/state.py` | Add `user_code_test_results: Optional[List[TestResult]]` to `LeetCodeSolverState` |
| `src/core/graph.py` | Add `route_after_tester`; update `route_by_mode` and `build_graph` for review flow |
| `src/agents/tester.py` | Extend `tester_node` to test `user_code` in review mode |
| `src/agents/code_review.py` | Add TLE instruction to system prompt; add `_format_test_summary`; inject into user prompt |
| `tests/test_state_mutations.py` | Tests for new state field |
| `tests/test_graph_flow.py` | Tests for routing functions |
| `tests/test_tester_node.py` | Tests for tester in review mode (new file) |
| `tests/test_code_review.py` | Tests for `_format_test_summary` (new file) |

---

## Task 1: Add `user_code_test_results` to State Schema

**Files:**
- Modify: `src/schemas/state.py`
- Test: `tests/test_state_mutations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state_mutations.py
from src.schemas.state import LeetCodeSolverState


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
    from src.schemas.state import TestResult
    result = TestResult(
        test_case_id="r0",
        passed=True,
        actual_output=42,
        execution_time_ms=10.0,
        error_message=None,
    )
    state: LeetCodeSolverState = {"user_code_test_results": [result]}  # type: ignore[typeddict-item]
    assert len(state["user_code_test_results"]) == 1
    assert state["user_code_test_results"][0]["passed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_state_mutations.py -v
```

Expected: `FAILED` — `TypedDict` missing key `user_code_test_results` or `KeyError`.

- [ ] **Step 3: Add the field to `LeetCodeSolverState`**

Open `src/schemas/state.py` and add one line inside `LeetCodeSolverState` after `user_code_review`:

```python
    user_code_test_results: Optional[List[TestResult]]
```

The full class tail should look like:

```python
    user_code: Optional[str]
    user_code_review: Optional[UserCodeReview]
    user_code_test_results: Optional[List[TestResult]]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_state_mutations.py -v
```

Expected: `PASSED` both tests.

- [ ] **Step 5: Commit**

```bash
git add src/schemas/state.py tests/test_state_mutations.py
git commit -m "feat: add user_code_test_results field to LeetCodeSolverState"
```

---

## Task 2: Update Graph Routing for Review Mode

**Files:**
- Modify: `src/core/graph.py`
- Test: `tests/test_graph_flow.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_flow.py
from src.core.graph import route_by_mode, route_after_tester


def test_route_by_mode_review_goes_to_tester():
    state = {"mode": "review"}
    assert route_by_mode(state) == "tester"


def test_route_by_mode_full_goes_to_solver():
    state = {"mode": "full"}
    assert route_by_mode(state) == "solver"


def test_route_by_mode_study_goes_to_documenter():
    state = {"mode": "study"}
    assert route_by_mode(state) == "documenter"


def test_route_by_mode_default_goes_to_solver():
    state = {}
    assert route_by_mode(state) == "solver"


def test_route_after_tester_review_goes_to_code_review():
    state = {"mode": "review"}
    assert route_after_tester(state) == "code_review"


def test_route_after_tester_full_goes_to_documenter():
    state = {"mode": "full"}
    assert route_after_tester(state) == "documenter"


def test_route_after_tester_default_goes_to_documenter():
    state = {}
    assert route_after_tester(state) == "documenter"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_graph_flow.py -v
```

Expected: `ImportError` (`route_after_tester` not found) and `FAILED` for review routing.

- [ ] **Step 3: Update `src/core/graph.py`**

Replace the entire file with:

```python
from langgraph.graph import END, StateGraph

from src.agents.code_review import code_review_node
from src.agents.documenter import documenter_node
from src.agents.solver import solver_node
from src.agents.tester import tester_node
from src.schemas.state import LeetCodeSolverState


def _make_stub(node_name: str):
    def _stub(state: LeetCodeSolverState) -> LeetCodeSolverState:
        new_state = dict(state)
        new_state["current_node"] = node_name
        new_state["total_agents_run"] = state.get("total_agents_run", 0) + 1
        return new_state
    _stub.__name__ = f"{node_name}_node"
    return _stub


error_handler_node = _make_stub("error_handler")


# --- Routing functions ---

def route_by_mode(state: LeetCodeSolverState) -> str:
    mode = state.get("mode", "full")
    if mode == "study":
        return "documenter"
    if mode == "review":
        return "tester"
    return "solver"


def route_after_tester(state: LeetCodeSolverState) -> str:
    if state.get("mode") == "review":
        return "code_review"
    return "documenter"


# --- Graph construction ---

def build_graph():
    graph = StateGraph(LeetCodeSolverState)

    graph.add_node("solver", solver_node)
    graph.add_node("tester", tester_node)
    graph.add_node("documenter", documenter_node)
    graph.add_node("code_review", code_review_node)
    graph.add_node("error_handler", error_handler_node)

    graph.set_conditional_entry_point(
        route_by_mode,
        {"solver": "solver", "documenter": "documenter", "tester": "tester"},
    )

    # full: solver -> tester -> documenter -> END
    graph.add_edge("solver", "tester")
    graph.add_conditional_edges(
        "tester",
        route_after_tester,
        {"code_review": "code_review", "documenter": "documenter"},
    )
    graph.add_edge("documenter", END)

    # review: tester -> code_review -> END
    graph.add_edge("code_review", END)

    graph.add_edge("error_handler", END)

    return graph.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_graph_flow.py -v
```

Expected: all 7 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/core/graph.py tests/test_graph_flow.py
git commit -m "feat: route review mode through tester before code_review"
```

---

## Task 3: Extend Tester Node to Test `user_code` in Review Mode

**Files:**
- Modify: `src/agents/tester.py`
- Test: `tests/test_tester_node.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tester_node.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tester_node.py -v
```

Expected: multiple failures — `user_code_test_results` key missing, `test_results` non-empty in review mode, etc.

- [ ] **Step 3: Update `tester_node` in `src/agents/tester.py`**

Replace `tester_node` with the following (the helpers `_run_solution_subprocess`, `_run_solution_semantic`, etc. are unchanged):

```python
def tester_node(state: LeetCodeSolverState) -> LeetCodeSolverState:
    new_state = dict(state)
    new_state["current_node"] = "tester"

    error_msg: Optional[str] = None
    test_results: List[TestResult] = []
    user_code_test_results: List[TestResult] = []
    test_cases: List[TestCase] = []
    status = "failed"
    mode = state.get("mode", "full")

    naive_solution = state.get("naive_solution")
    optimal_solution = state.get("optimal_solution")
    language = state.get("language", "Python")
    use_semantic = language.lower() in _SEMANTIC_LANGUAGES
    problem_title = state.get("problem_title", "")

    try:
        test_cases = generate_test_cases(
            problem_title=problem_title,
            problem_description=state.get("problem_description", ""),
            problem_constraints=state.get("problem_constraints", ""),
            solution_code=optimal_solution["code"] if optimal_solution else "",
        )

        def _run(code: str, approach: str) -> List[TestResult]:
            if use_semantic:
                return _run_solution_semantic(code, approach, language, problem_title)
            return _run_solution_subprocess(code, approach, test_cases)

        if naive_solution and naive_solution.get("code"):
            test_results.extend(_run(naive_solution["code"], "naive"))

        if optimal_solution and optimal_solution.get("code"):
            test_results.extend(_run(optimal_solution["code"], "optimal"))

        if mode == "review":
            user_code = state.get("user_code", "")
            if user_code:
                user_code_test_results = _run(user_code, "user_code")
            status = "success" if user_code_test_results else "partial_success"
        elif not test_results:
            test_results = [TestResult(
                test_case_id="no_results_0",
                passed=False,
                actual_output=None,
                execution_time_ms=0.0,
                error_message="No test cases were generated or no solution code was available.",
            )]
            status = "partial_success"
        else:
            status = "success"

    except Exception as exc:
        error_msg = str(exc)
        status = "failed"
        error_result = TestResult(
            test_case_id="error_0",
            passed=False,
            actual_output=None,
            execution_time_ms=0.0,
            error_message=f"Test execution failed: {error_msg}",
        )
        if mode == "review":
            user_code_test_results = [error_result]
        else:
            test_results = [error_result]

    all_passed = bool(test_results) and all(r["passed"] for r in test_results)
    failed_count = sum(1 for r in test_results if not r["passed"])
    user_code_passed = sum(1 for r in user_code_test_results if r["passed"])

    log_entry: AgentLog = {
        "agent_name": "tester",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "execute_test_cases",
        "status": "completed" if status in ("success", "partial_success") else "failed",
        "metadata": {
            "tester_status": status,
            "total_tests": len(test_results),
            "passed": len(test_results) - failed_count,
            "failed": failed_count,
            "test_cases_generated": len(test_cases),
            "error": error_msg,
            "user_code_total": len(user_code_test_results),
            "user_code_passed": user_code_passed,
            "user_code_failed": len(user_code_test_results) - user_code_passed,
        },
    }

    new_state["generated_test_cases"] = test_cases
    new_state["test_results"] = test_results
    new_state["all_tests_passed"] = all_passed
    new_state["user_code_test_results"] = user_code_test_results
    new_state["total_agents_run"] = state.get("total_agents_run", 0) + 1
    new_state["agent_logs"] = list(state.get("agent_logs", [])) + [log_entry]

    if error_msg:
        new_state["error_logs"] = list(state.get("error_logs", [])) + [
            f"Tester: {error_msg}"
        ]

    return new_state
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tester_node.py -v
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tester.py tests/test_tester_node.py
git commit -m "feat: test user_code in review mode, store in user_code_test_results"
```

---

## Task 4: Strengthen Code Review — TLE Prompt + Runtime Summary Injection

**Files:**
- Modify: `src/agents/code_review.py`
- Test: `tests/test_code_review.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_code_review.py
from src.agents.code_review import _format_test_summary


def test_format_empty_results_returns_empty_string():
    assert _format_test_summary([]) == ""


def test_format_all_pass():
    results = [
        {"passed": True, "execution_time_ms": 10.0, "error_message": None, "actual_output": 1},
        {"passed": True, "execution_time_ms": 8.0,  "error_message": None, "actual_output": 2},
    ]
    summary = _format_test_summary(results)
    assert "PASS" in summary
    assert "Total: 2/2 passed" in summary
    assert "FAIL" not in summary
    assert "TIMEOUT" not in summary


def test_format_timeout_labeled_correctly():
    results = [
        {"passed": False, "execution_time_ms": 5000.0,
         "error_message": "Execution timed out after 5s", "actual_output": None},
    ]
    summary = _format_test_summary(results)
    assert "TIMEOUT" in summary
    assert "5000ms" in summary
    assert "Total: 0/1 passed" in summary


def test_format_fail_labeled_correctly():
    results = [
        {"passed": False, "execution_time_ms": 12.0,
         "error_message": None, "actual_output": 5},
    ]
    summary = _format_test_summary(results)
    assert "FAIL" in summary
    assert "Total: 0/1 passed" in summary


def test_format_error_labeled_correctly():
    results = [
        {"passed": False, "execution_time_ms": 3.0,
         "error_message": "IndexError: list index out of range", "actual_output": None},
    ]
    summary = _format_test_summary(results)
    assert "ERROR" in summary
    assert "IndexError" in summary


def test_format_mixed_results():
    results = [
        {"passed": True,  "execution_time_ms": 5.0,    "error_message": None,                          "actual_output": 1},
        {"passed": False, "execution_time_ms": 5000.0, "error_message": "Execution timed out after 5s","actual_output": None},
        {"passed": False, "execution_time_ms": 7.0,    "error_message": None,                          "actual_output": 99},
    ]
    summary = _format_test_summary(results)
    assert "PASS" in summary
    assert "TIMEOUT" in summary
    assert "FAIL" in summary
    assert "Total: 1/3 passed" in summary
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_code_review.py -v
```

Expected: `ImportError` — `_format_test_summary` not defined yet.

- [ ] **Step 3: Add `_format_test_summary` and update `_SYSTEM_PROMPT`, `_USER_TEMPLATE`, and `code_review_node` in `src/agents/code_review.py`**

Replace the entire file content with:

```python
from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel, Field, field_validator

from src.core.llm import get_llm
from src.schemas.state import AgentLog, LeetCodeSolverState, UserCodeReview


class _UserCodeReviewModel(BaseModel):
    time_complexity: str
    time_explanation: str
    space_complexity: str
    space_explanation: str
    is_optimal: str  # שינוי מ-bool ל-str כדי למנוע מ-Groq להכריז על כשל
    optimality_gap: str
    optimization_roadmap: list[str]
    optimal_solution: str

    @property
    def is_optimal_bool(self) -> bool:
        return str(self.is_optimal).lower() in ("true", "1", "yes")


_llm_review = get_llm().with_structured_output(_UserCodeReviewModel)

_SYSTEM_PROMPT = """\
You are an expert algorithm analyst and strict QA engineer performing a code review. Given a user's {language} solution to a LeetCode problem, evaluate it with absolute precision.

CRITICAL INSTRUCTIONS:
1. DO NOT ASSUME CORRECTNESS: Even if the code uses the correct data structure (e.g., HashSet for Longest Consecutive Sequence), you MUST dry-run the code line-by-line.
2. EDGE-CASE DRY-RUN: Explicitly trace the code execution with these test cases in your head:
   - Empty input (e.g., null or empty array `[]`)
   - Single-element input (e.g., `[1]`)
   - Inputs with duplicate values (e.g., `[1, 2, 0, 1]`)
   - Inputs with negative numbers or non-sequential elements
3. BUG DETECTION: Check if the user's code produces an WRONG ANSWER, NullPointerException, or IndexOutOfBoundsException on any of these cases.
4. TLE / COMPLEXITY RISK:
   - First, identify the known theoretical optimal time complexity for this problem class (e.g., Two Sum → O(N), comparison sort → O(N log N), graph BFS/DFS → O(V+E)).
   - Compare the user's ACTUAL time complexity against that optimal.
   - If the user's complexity is WORSE than optimal (e.g., O(N²) vs O(N), O(2^N) vs O(N log N)), you MUST:
     * Set `is_optimal` to false.
     * Begin `optimality_gap` with the EXACT phrase "TLE RISK: Time Limit Exceeded" followed by a concrete estimate using the problem's maximum constraints (e.g., "TLE RISK: Time Limit Exceeded — O(N²) with N=10⁵ yields ≈10¹⁰ operations, exceeding the ~10⁸ op/s typical limit").
   - If Runtime Test Results are provided below and any test shows TIMEOUT, you MUST set `is_optimal` to false and include "TLE RISK: Time Limit Exceeded" in `optimality_gap`.
   - This rule applies even when the code produces correct answers on small inputs.

If the user's code has a logical bug, edge-case failure, or incorrect return value:
- You MUST set `is_optimal` to false.
- You MUST explicitly document the bug and the failing test case inside `optimality_gap` (e.g., "The code fails on an empty array [] because it throws IndexOutOfBoundsException, and incorrectly returns 1 for duplicate elements...").

Fields to return:
- time_complexity: Big O expression for time (e.g., O(n), O(n^2)).
- time_explanation: Justify by citing specific loops, recursion depth, or operations in the code.
- space_complexity: Big O expression for auxiliary space (excluding input).
- space_explanation: Justify by citing specific allocations (arrays, hashmaps, call stack, etc.) in the code.
- is_optimal: true ONLY if the code is 100% bug-free, handles all edge cases correctly, AND matches the theoretical optimal time/space complexity.
- optimality_gap: If there are bugs, edge-case failures, TLE risks, or suboptimal complexity, describe them clearly here with example inputs. If the code is completely correct and optimal, write "None."
- optimization_roadmap: An ordered list of concrete steps to fix bugs or optimize the current approach. If already optimal and bug-free, return a list with one item: "None needed."
- optimal_solution: The exact, clean, bug-free {language} implementation of the optimal solution for this problem. Always provide this regardless of the user's solution status.
"""

_USER_TEMPLATE = """\
Problem Description:
{description}

User's Solution:
```{language_lower}
{user_code}
```
{test_summary}"""


def _format_test_summary(results: list) -> str:
    """Format user_code_test_results into a human-readable string for the LLM prompt."""
    if not results:
        return ""
    lines = ["", "Runtime Test Results (from sandbox execution):"]
    for i, r in enumerate(results, 1):
        err = r.get("error_message") or ""
        ms = r.get("execution_time_ms", 0.0)
        if "timed out" in err.lower():
            lines.append(f"- Test {i}: TIMEOUT after {ms:.0f}ms")
        elif err:
            lines.append(f"- Test {i}: ERROR — {err[:120]} ({ms:.0f}ms)")
        elif r.get("passed"):
            lines.append(f"- Test {i}: PASS ({ms:.0f}ms)")
        else:
            lines.append(f"- Test {i}: FAIL — got {r.get('actual_output')!r} ({ms:.0f}ms)")
    passed = sum(1 for r in results if r.get("passed"))
    lines.append(f"Total: {passed}/{len(results)} passed")
    return "\n".join(lines)


def code_review_node(state: LeetCodeSolverState) -> LeetCodeSolverState:
    new_state = dict(state)
    new_state["current_node"] = "code_review"

    user_code = state.get("user_code", "")
    language = state.get("language", "Python")
    user_code_review = None
    error_msg = None

    test_summary = _format_test_summary(state.get("user_code_test_results") or [])

    try:
        result: _UserCodeReviewModel = _llm_review.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT.format(language=language)},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    description=state.get("problem_description", ""),
                    language_lower=language.lower(),
                    user_code=user_code,
                    test_summary=test_summary,
                ),
            },
        ])
        user_code_review = UserCodeReview(
            time_complexity=result.time_complexity,
            time_explanation=result.time_explanation,
            space_complexity=result.space_complexity,
            space_explanation=result.space_explanation,
            is_optimal=result.is_optimal,
            optimality_gap=result.optimality_gap,
            optimization_roadmap=result.optimization_roadmap,
            optimal_solution=result.optimal_solution,
        )
    except Exception as exc:
        error_msg = str(exc)

    log_entry: AgentLog = {
        "agent_name": "code_review",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "review_user_code",
        "status": "completed" if user_code_review is not None else "failed",
        "metadata": {"error": error_msg},
    }

    new_state["user_code_review"] = user_code_review
    new_state["total_agents_run"] = state.get("total_agents_run", 0) + 1
    new_state["agent_logs"] = list(state.get("agent_logs", [])) + [log_entry]
    new_state["graph_status"] = "completed" if user_code_review else "error"

    if error_msg:
        new_state["error_logs"] = list(state.get("error_logs", [])) + [
            f"code_review: {error_msg}"
        ]

    return new_state
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_code_review.py -v
```

Expected: all 6 tests `PASSED`.

- [ ] **Step 5: Run the full test suite to verify no regressions**

```bash
pytest tests/ -v
```

Expected: all tests pass. No regressions in `test_state_mutations.py`, `test_graph_flow.py`, `test_tester_node.py`, `test_code_review.py`.

- [ ] **Step 6: Commit**

```bash
git add src/agents/code_review.py tests/test_code_review.py
git commit -m "feat: add TLE detection to code review prompt and inject runtime test summary"
```
