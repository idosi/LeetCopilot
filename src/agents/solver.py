from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from src.core.llm import get_llm
from src.schemas.state import AgentLog, LeetCodeSolverState, Solution


class _SolutionModel(BaseModel):
    code: str
    description: str
    language: str
    approach: str


class _SolverOutput(BaseModel):
    naive_solution: _SolutionModel
    optimal_solution: _SolutionModel
    solver_status: str
    solver_error: Optional[str] = None


   
class _HintModel(BaseModel):
    hint_number: int
    text: str
    guiding_question: str


class _StudySolverOutput(BaseModel):
    pattern_name: str
    core_intuition: str
    hint_1: _HintModel
    hint_2: _HintModel
    hint_3: _HintModel
    why_not_brute_force: str
    key_data_structure: str
    solver_status: str
    solver_error: Optional[str] = None


_llm = get_llm().with_structured_output(_SolverOutput)
_study_llm = get_llm().with_structured_output(_StudySolverOutput)

_STUDY_SYSTEM_PROMPT = """\
You are a Socratic programming tutor. Your goal is to guide the student to the solution — NOT to give it.

STRICT RULES (non-negotiable):
- DO NOT write any functional code whatsoever. Not even a single line.
- DO NOT reveal the full algorithm or complete step-by-step solution.
- DO NOT use code blocks of any kind.

Your task:
1. Identify the optimal algorithmic PATTERN for this problem (e.g. "Sliding Window", "Two Pointers", "Prefix Sum + HashMap").
2. Write the CORE INTUITION: a 2-3 sentence explanation of WHY this pattern applies — the insight that bridges the problem structure to the technique.
3. Generate exactly 3 INCREMENTAL HINTS, each building on the previous:
   - Hint 1: Structural clue (what property of the input suggests the pattern?)
   - Hint 2: Invariant or key observation (what must remain true at each step?)
   - Hint 3: Concrete nudge toward the data structure or loop structure (without code)
   Each hint must include a guiding_question that prompts the student to think, e.g. "What happens to the window as we move right?"
4. Explain WHY brute force is insufficient (complexity argument).
5. Name the KEY DATA STRUCTURE and why it achieves the optimal complexity.
"""

_SYSTEM_PROMPT_TEMPLATE = """\
You are a world-class competitive programmer specializing in algorithmically extreme solutions. \
Given a LeetCode problem, generate exactly two {language} solutions.

NAIVE SOLUTION rules:
- approach field = "naive"
- Brute-force / straightforward implementation — clarity over efficiency

OPTIMAL SOLUTION rules (non-negotiable — read every point):
- approach field = "optimal"
- You MUST reach the absolute theoretical lower bound for both time AND space complexity for this \
problem class. "Good enough" is not acceptable.
- Before choosing an algorithm, ask: can this be solved in fewer passes? Can auxiliary space be \
reduced to O(1) or O(k) for a fixed alphabet/range k?
- Prefer fixed-size arrays or buckets over hash maps whenever the input domain is bounded \
(e.g., ASCII characters → int[128], digits → int[10], lowercase letters → int[26]). \
Hash maps carry constant-factor overhead and poor cache locality — eliminate them when a \
bounded array suffices.
- Prefer in-place mutation, two-pointer, sliding window, bit manipulation, or monotonic \
structures over allocating auxiliary data structures whenever the problem permits.
- If the theoretical time lower bound is Ω(n), do not produce an O(n log n) solution. \
If O(1) extra space is achievable, do not produce an O(n) space solution.
- Never use a standard library sort when a counting/radix/bucket sort would be asymptotically \
or practically superior given the stated constraints.
- The solution must still be correct and handle all edge cases stated in the problem.

BOTH solutions must:
- Be syntactically valid, self-contained {language} code
- Use a standard LeetCode-style class/function structure appropriate for {language}
- Include a brief single-line comment describing the approach
- Set language = "{language_lower}"

OUTPUT fields:
- solver_status: "success" if both generated, "partial_success" if only one, "failed" if neither
- solver_error: null on success, error message otherwise
"""

_USER_TEMPLATE = """\
Problem Title: {title}

Problem Description:
{description}

Constraints:
{constraints}
"""


def _run_full_mode(state: LeetCodeSolverState):
    language = state.get("language", "Python")
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        language=language,
        language_lower=language.lower(),
    )
    result: _SolverOutput = _llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _USER_TEMPLATE.format(
            title=state.get("problem_title", ""),
            description=state.get("problem_description", ""),
            constraints=state.get("problem_constraints", ""),
        )},
    ])
    naive = Solution(
        code=result.naive_solution.code,
        description=result.naive_solution.description,
        language=result.naive_solution.language,
        approach=result.naive_solution.approach,
    )
    optimal = Solution(
        code=result.optimal_solution.code,
        description=result.optimal_solution.description,
        language=result.optimal_solution.language,
        approach=result.optimal_solution.approach,
    )
    return naive, optimal, None, result.solver_status or "success", None


def _run_study_mode(state: LeetCodeSolverState):
    user_msg = _USER_TEMPLATE.format(
        title=state.get("problem_title", ""),
        description=state.get("problem_description", ""),
        constraints=state.get("problem_constraints", ""),
    )
    result: _StudySolverOutput = _study_llm.invoke([
        {"role": "system", "content": _STUDY_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    study_output = {
        "pattern_name": result.pattern_name,
        "core_intuition": result.core_intuition,
        "why_not_brute_force": result.why_not_brute_force,
        "key_data_structure": result.key_data_structure,
        "hints": [
            {"hint_number": result.hint_1.hint_number, "text": result.hint_1.text, "guiding_question": result.hint_1.guiding_question},
            {"hint_number": result.hint_2.hint_number, "text": result.hint_2.text, "guiding_question": result.hint_2.guiding_question},
            {"hint_number": result.hint_3.hint_number, "text": result.hint_3.text, "guiding_question": result.hint_3.guiding_question},
        ],
    }
    # Stub solutions so route_solver routes to performance without error
    language = state.get("language", "Python")
    stub = Solution(code="# Study Mode — no code generated", description="Study mode active.", language=language, approach="study")
    return stub, stub, study_output, result.solver_status or "success", result.solver_error


def solver_node(state: LeetCodeSolverState) -> LeetCodeSolverState:
    new_state = dict(state)
    new_state["current_node"] = "solver"

    error_msg: Optional[str] = None
    naive: Optional[Solution] = None
    optimal: Optional[Solution] = None
    study_output = None
    status = "failed"

    try:
        is_study = state.get("mode", "") == "study"
        if is_study:
            naive, optimal, study_output, status, error_msg = _run_study_mode(state)
        else:
            naive, optimal, _, status, error_msg = _run_full_mode(state)
    except Exception as exc:
        error_msg = str(exc)
        status = "failed"

    log_entry: AgentLog = {
        "agent_name": "solver",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "generate_hints" if state.get("mode", "") == "study" else "generate_solutions",
        "status": "completed" if status == "success" else "failed",
        "metadata": {
            "solver_status": status,
            "mode": state.get("mode", "full"),
            "naive_generated": naive is not None,
            "optimal_generated": optimal is not None,
            "error": error_msg,
        },
    }

    new_state["naive_solution"] = naive
    new_state["optimal_solution"] = optimal
    new_state["study_output"] = study_output
    new_state["total_agents_run"] = state.get("total_agents_run", 0) + 1
    new_state["agent_logs"] = list(state.get("agent_logs", [])) + [log_entry]

    if error_msg:
        new_state["error_logs"] = list(state.get("error_logs", [])) + [f"Solver: {error_msg}"]
    if status == "failed":
        new_state["graph_status"] = "failed"

    return new_state
