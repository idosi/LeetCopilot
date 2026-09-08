from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from src.core.llm import get_llm
from src.schemas.state import AgentLog, LeetCodeSolverState, Solution


class _SolutionModel(BaseModel):
    code: str = Field(description="Complete, well-formatted, multi-line code with proper newlines and indentation. Never minify.")
    description: str = Field(description="Detailed step-by-step technical walkthrough of the solution (3-5 sentences).")
    language: str
    approach: str
    why_it_works: str = Field(description="Clear explanation of why this approach works, the mathematical/algorithmic intuition, and edge cases handled.")


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
You are a world-class competitive programmer and algorithm tutor specializing in clean, optimal solutions.
Given a LeetCode problem, generate exactly two {language} solutions: NAIVE and OPTIMAL.

CODE FORMATTING RULES:
- The `code` field MUST be valid, beautifully formatted {language} code with explicit newlines (\\n) and standard indentation.
- NEVER minify the code or output it on a single line.
- Use standard LeetCode class/method structure.
- Set language = "{language_lower}".

EXPLANATION & ANALYSIS RULES:
- `description`: Provide a detailed technical walkthrough (3-5 sentences) explaining the intuition, data structures, pointer manipulations, and why the algorithm handles edge cases correctly. DO NOT output a short one-line summary.

NAIVE SOLUTION rules:
- approach field = "naive"
- Brute-force or straightforward implementation.

OPTIMAL SOLUTION rules:
- approach field = "optimal"
- Must reach the absolute theoretical lower bound for time and space complexity.
- Clearly explain the algorithmic leap that beats the naive approach.

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
    
    # אתחול ה-LLM רק בזמן ריצה
    model_name = state.get("model_name")
    llm = get_llm(model_name=model_name).with_structured_output(_SolverOutput)

    result: _SolverOutput = llm.invoke([
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
    
    # אתחול ה-LLM רק בזמן ריצה
    model_name = state.get("model_name")
    study_llm = get_llm(model_name=model_name).with_structured_output(_StudySolverOutput)

    result: _StudySolverOutput = study_llm.invoke([
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