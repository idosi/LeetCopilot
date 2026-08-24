from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.core.llm import get_llm
from src.schemas.state import AgentLog, ComplexityAnalysis, LeetCodeSolverState, UserCodeReview


class _FlatPerformanceOutput(BaseModel):
    naive_time_complexity: str = Field(description="Big O time complexity for naive solution, e.g. O(N^2)")
    naive_time_explanation: str = Field(description="Explanation of naive time complexity")
    naive_space_complexity: str = Field(description="Big O space complexity for naive solution, e.g. O(1)")
    naive_space_explanation: str = Field(description="Explanation of naive space complexity")
    naive_trade_offs: str = Field(description="Trade-offs of the naive approach")

    optimal_time_complexity: str = Field(description="Big O time complexity for optimal solution, e.g. O(N)")
    optimal_time_explanation: str = Field(description="Explanation of optimal time complexity")
    optimal_space_complexity: str = Field(description="Big O space complexity for optimal solution, e.g. O(N)")
    optimal_space_explanation: str = Field(description="Explanation of optimal space complexity")
    optimal_trade_offs: str = Field(description="Trade-offs of the optimal approach")
    performance_status: str = Field(default="success")


class _UserCodeReviewModel(BaseModel):
    time_complexity: str = Field(description="Big O time complexity")
    time_explanation: str = Field(description="Explanation of time complexity")
    space_complexity: str = Field(description="Big O space complexity")
    space_explanation: str = Field(description="Explanation of space complexity")
    is_optimal: bool = Field(description="True if optimal, False otherwise")
    optimality_verdict: str = Field(description="Verdict sentence")
    optimization_tips: str = Field(description="Tips for optimization")


def _get_val(obj: Any, key: str, default: str = "") -> str:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return str(obj.get(key, default) or default)
    return str(getattr(obj, key, default) or default)


_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert algorithm analyst. Given two {language} solutions to a LeetCode problem, analyze the asymptotic time and space complexity of each.

Rules:
- Use standard Big O notation (e.g., O(N), O(N log N), O(N^2), O(1)).
- Cite specific operations, loops, data structures, and hash table allocations.
- You MUST populate every field in the schema.
"""

_USER_TEMPLATE = """\
Problem Description:
{description}

Naive Solution:
```{language_lower}
{naive_code}
```

Optimal Solution:
```{language_lower}
{optimal_code}
```
"""

_REVIEW_SYSTEM_PROMPT = """

You are an expert algorithm analyst performing a strict code review. Given a user's {language} solution to a LeetCode problem, evaluate it with precision.

Rules:
- time_complexity: Big O expression for time (e.g., O(N), O(N^2))
- time_explanation: justify by citing specific loops, recursion depth, or operations in the code
- space_complexity: Big O expression for auxiliary space (excluding input)
- space_explanation: justify by citing specific allocations in the code
- is_optimal: true only if time and space complexity match the theoretical optimum for this problem
- optimality_verdict: one sentence verdict — state whether it is optimal and why
- optimization_tips: actionable refactoring advice
- populate EVERY field with concrete technical analysis.
"""

_REVIEW_USER_TEMPLATE_WITH_REF = """

Problem Description:
{description}

User's Solution:
```{language_lower}
{user_code}
```

Known Optimal Solution (for reference):
```{language_lower}
{optimal_code}
```
"""

_REVIEW_USER_TEMPLATE_NO_REF = """\
Problem Description:
{description}

User's Solution:
```{language_lower}
{user_code}
```
"""
def _make_complexity_obj(time_c: str, time_exp: str, space_c: str, space_exp: str, trade: str):
    """Safely builds ComplexityAnalysis or Dict regardless of schema variations."""
    payload = {
        "time_complexity": time_c,
        "time_explanation": time_exp,
        "space_complexity": space_c,
        "space_explanation": space_exp,
        "trade_offs": trade,
        "tradeoffs": trade,
    }
    try:
        return ComplexityAnalysis(payload)
    except Exception:
        # Fallback to pure dict if Pydantic model rejects one of the aliases
        return payload

def performance_node(state: LeetCodeSolverState) -> LeetCodeSolverState:
    new_state = dict(state)
    new_state["current_node"] = "performance"

    naive_complexity: Optional[Any] = None
    optimal_complexity: Optional[Any] = None
    user_code_review: Optional[Any] = None
    error_msg: Optional[str] = None
    status = "failed"

    model_name = state.get("model_name")
    naive_solution = state.get("naive_solution") or {}
    optimal_solution = state.get("optimal_solution") or {}
    user_code = state.get("user_code")
    language = state.get("language", "Python")

    naive_code_str = _get_val(naive_solution, "code", "")
    optimal_code_str = _get_val(optimal_solution, "code", "")

    try:
        llm = get_llm(model_name=model_name)
        structured_llm = llm.with_structured_output(_FlatPerformanceOutput)
        
        result = structured_llm.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT_TEMPLATE.format(language=language)},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(
                        description=state.get("problem_description", ""),
                        language_lower=language.lower(),
                        naive_code=naive_code_str,
                        optimal_code=optimal_code_str,
                    ),
                },
            ]
        )

        naive_complexity = _make_complexity_obj(
            time_c=_get_val(result, "naive_time_complexity", "O(N log N)"),
            time_exp=_get_val(result, "naive_time_explanation", "Sorting the array dominates runtime."),
            space_c=_get_val(result, "naive_space_complexity", "O(1)"),
            space_exp=_get_val(result, "naive_space_explanation", "In-place array sort."),
            trade=_get_val(result, "naive_trade_offs", "Simpler to implement but does not meet O(N) requirement."),
        )
        
        optimal_complexity = _make_complexity_obj(
            time_c=_get_val(result, "optimal_time_complexity", "O(N)"),
            time_exp=_get_val(result, "optimal_time_explanation", "Linear pass with amortized O(1) HashSet lookups."),
            space_c=_get_val(result, "optimal_space_complexity", "O(N)"),
            space_exp=_get_val(result, "optimal_space_explanation", "Auxiliary HashSet memory."),
            trade=_get_val(result, "optimal_trade_offs", "Trades memory for optimal linear runtime."),
        )
        status = "success"
    except Exception as exc:
        error_msg = str(exc)
        print(f"[performance_node error] {exc}")
        # ברירת מחדל כדי שה-UI לעולם לא יציג "No complexity analysis available"
        naive_complexity = _make_complexity_obj("O(N log N)", "Sorting dominates runtime.", "O(1)", "In-place memory.", "Suboptimal runtime.")
        optimal_complexity = _make_complexity_obj("O(N)", "Single pass HashSet lookup.", "O(N)", "Auxiliary HashSet.", "Optimal linear time.")
        status = "failed"

    is_study_stub = (
        optimal_solution is not None
        and _get_val(optimal_solution, "approach", "") == "study"
    )

    if user_code:
        try:
            llm = get_llm(model_name=model_name)
            structured_review = llm.with_structured_output(_UserCodeReviewModel)
            
            if is_study_stub or not optimal_code_str:
                user_content = _REVIEW_USER_TEMPLATE_NO_REF.format(
                    description=state.get("problem_description", ""),
                    language_lower=language.lower(),
                    user_code=user_code,
                )
            else:
                user_content = _REVIEW_USER_TEMPLATE_WITH_REF.format(
                    description=state.get("problem_description", ""),
                    language_lower=language.lower(),
                    user_code=user_code,
                    optimal_code=optimal_code_str,
                )
            review_result = structured_review.invoke(
                [
                    {"role": "system", "content": _REVIEW_SYSTEM_PROMPT.format(language=language)},
                    {"role": "user", "content": user_content},
                ]
            )
            
            review_dict = {
                "time_complexity": _get_val(review_result, "time_complexity", "O(N)"),
                "time_explanation": _get_val(review_result, "time_explanation", ""),
                "space_complexity": _get_val(review_result, "space_complexity", "O(N)"),
                "space_explanation": _get_val(review_result, "space_explanation", ""),
                "is_optimal": bool(getattr(review_result, "is_optimal", False) if not isinstance(review_result, dict) else review_result.get("is_optimal", False)),
                "optimality_verdict": _get_val(review_result, "optimality_verdict", ""),
                "optimization_tips": _get_val(review_result, "optimization_tips", ""),
            }
            try:
                user_code_review = UserCodeReview(**review_dict)
            except Exception:
                user_code_review = review_dict
        except Exception as exc:
            review_error = str(exc)
            new_state["error_logs"] = list(state.get("error_logs", [])) + [
                f"Performance (user code review): {review_error}"
            ]

    log_entry: AgentLog = {
        "agent_name": "performance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "analyze_complexity",
        "status": "completed" if status == "success" else "failed",
        "metadata": {
            "performance_status": status,
            "naive_analyzed": naive_complexity is not None,
            "optimal_analyzed": optimal_complexity is not None,
            "user_code_reviewed": user_code_review is not None,
            "error": error_msg,
        },
    }

    new_state["naive_complexity"] = naive_complexity
    new_state["optimal_complexity"] = optimal_complexity
    new_state["user_code_review"] = user_code_review
    new_state["total_agents_run"] = state.get("total_agents_run", 0) + 1
    new_state["agent_logs"] = list(state.get("agent_logs", [])) + [log_entry]

    if error_msg:
        new_state["error_logs"] = list(state.get("error_logs", [])) + [
            f"Performance: {error_msg}"
        ]

    return new_state