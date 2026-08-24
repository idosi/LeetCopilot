from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field

from src.core.llm import get_llm
from src.schemas.state import AgentLog, LeetCodeSolverState

class CodeImprovement(BaseModel):
    category: str = Field(
        description="One of: 'Naming & Readability', 'Generics & Types', 'Idiomatic Syntax', or 'Defensive Coding'"
    )
    issue: str = Field(description="The specific defect, raw type, or poor practice identified")
    suggestion: str = Field(description="Actionable fix with exact code syntax to use")

class CleanCodeTip(BaseModel):
    category: str = Field(description="Area: e.g. Generics, Naming, Idiomatic Java/Python, Readability")
    finding: str = Field(description="The specific issue or anti-pattern identified in the code")
    recommendation: str = Field(description="Actionable fix or refactoring advice with code snippet")


class PerformanceTip(BaseModel):
    area: str = Field(description="Area: e.g. Pre-sizing Capacity, Early Guards, Memory Footprint, Autoboxing")
    observation: str = Field(description="What causes unnecessary overhead or memory allocation")
    optimization: str = Field(description="Exact technical micro-optimization to eliminate overhead")

class _UserCodeReviewModel(BaseModel):
    time_complexity: str = Field(description="Big O time complexity, e.g., O(N) or O(N log N)")
    time_explanation: str = Field(description="Detailed explanation of time complexity")
    space_complexity: str = Field(description="Big O space complexity, e.g., O(1) or O(N)")
    space_explanation: str = Field(description="Detailed explanation of auxiliary memory usage")
    is_optimal: bool = Field(description="True if asymptotically optimal, False otherwise")
    optimality_gap: str = Field(description="Assessment comparing submitted code to theoretical optimum")
    code_quality_improvements: List[CleanCodeTip] = Field(
        description="2-4 clean-code and idiomatic suggestions. Must not be empty."
    )
    constant_factor_tips: List[PerformanceTip] = Field(
        description="2-4 runtime/memory micro-optimizations. Must not be empty."
    )
    optimization_roadmap: List[str] = Field(
        description="3-5 step-by-step refactoring actions"
    )
    optimal_solution: str = Field(
        description="Complete production-grade optimal code implementation"
    )


_SYSTEM_PROMPT = """\
You are a Principal Software Engineer conducting a thorough code review on submitted code.
Even if the algorithmic Big-O is optimal, modern production engineering requires:
1. Identifying generic safety issues (raw types), variable naming, readability.
2. Identifying micro-optimizations: capacity pre-sizing (e.g. `new HashSet<>(nums.length * 2)` to avoid rehashing), null/empty defensive guard clauses, avoiding unnecessary boxing or iterator allocations.
3. Providing clear step-by-step refactoring actions in `optimization_roadmap`.
4. Providing full optimal reference code.

You MUST populate `code_quality_improvements`, `constant_factor_tips`, and `optimization_roadmap` with concrete items.
"""

_USER_TEMPLATE = """\
Problem Title: {title}
Description:
{description}

Constraints:
{constraints}

User Submission ({language}):
```{language_lower}
{user_code}
```
Perform the comprehensive code review now."""


def _format_test_summary(results: list) -> str:
    """Format user_code_test_results into a human-readable string for the LLM prompt."""
    if not results:
        return ""
    lines = ["", "Runtime Test Results:"]
    for i, r in enumerate(results, 1):
        err = r.get("error_message") or ""
        ms_raw = r.get("execution_time_ms")
        ms_str = f"{ms_raw:.0f}ms" if ms_raw is not None else "?ms"
        if "timed out" in err.lower():
            lines.append(f"- Test {i}: TIMEOUT after {ms_str}")
        elif err:
            lines.append(f"- Test {i}: ERROR — {err[:120]} ({ms_str})")
        elif r.get("passed"):
            lines.append(f"- Test {i}: PASS ({ms_str})")
        else:
            lines.append(f"- Test {i}: FAIL — got {r.get('actual_output')!r} ({ms_str})")
        passed = sum(1 for r in results if r.get("passed"))
        lines.append(f"Total: {passed}/{len(results)} passed")
    return "\n".join(lines)


def code_review_node(state: LeetCodeSolverState) -> LeetCodeSolverState:
    new_state = dict(state)
    new_state["current_node"] = "code_review"

    user_code = state.get("user_code", "")
    language = state.get("language", "Java")
    model_name = state.get("model_name")
    user_code_review = None
    error_msg = None

    try:
        llm = get_llm(model_name=model_name)
        structured_llm = llm.with_structured_output(_UserCodeReviewModel)
        result: _UserCodeReviewModel = structured_llm.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    title=state.get("problem_title", "LeetCode Problem"),
                    description=state.get("problem_description", ""),
                    constraints=state.get("problem_constraints") or "Standard bounds.",
                    language=language,
                    language_lower=language.lower(),
                    user_code=user_code,
                ),
            },
        ])

        # המרה למבנה נתונים פשוט ויציב
        quality_list = [
            f"{item.finding}: {item.recommendation}" if isinstance(item, CleanCodeTip) else str(item)
            for item in result.code_quality_improvements
        ]
        constant_list = [
            f"{item.observation}: {item.optimization}" if isinstance(item, PerformanceTip) else str(item)
            for item in result.constant_factor_tips
        ]

        user_code_review = {
            "time_complexity": result.time_complexity,
            "time_explanation": result.time_explanation,
            "space_complexity": result.space_complexity,
            "space_explanation": result.space_explanation,
            "is_optimal": bool(result.is_optimal),
            "optimality_gap": result.optimality_gap,
            "optimization_roadmap": [str(x) for x in result.optimization_roadmap],
            "code_quality_improvements": quality_list,
            "constant_factor_tips": constant_list,
            "optimal_solution": result.optimal_solution or "",
        }
    except Exception as exc:
        error_msg = str(exc)
        print(f"[code_review error] {exc}")

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