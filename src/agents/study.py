from typing import List
from pydantic import BaseModel, Field
from src.core.llm import get_llm
from src.schemas.state import LeetCodeSolverState

class IncrementalHint(BaseModel):
    hint_number: int = Field(default=1, description="1, 2, or 3")
    guiding_question: str = Field(default="", description="Question or title for the hint")
    text: str = Field(default="", description="The hint explanation")

class StudyModeOutput(BaseModel):
    pattern_name: str = Field(
        default="Pattern Recognition",
        description="e.g. Hash Set Sequence Recognition"
    )
    why_not_brute_force: str = Field(
        default="",
        description="Explanation of why brute force is inefficient / TLE"
    )
    key_data_structure: str = Field(
        default="",
        description="Optimal data structure to use"
    )
    core_intuition: str = Field(
        default="",
        description="High level insight without writing code"
    )
    hints: List[IncrementalHint] = Field(
        default_factory=list,
        description="Exactly 3 progressive hints"
    )
    cognitive_steps: List[str] = Field(
        default_factory=lambda: [
            "Identify what causes brute force to be inefficient",
            "Determine the invariant that signals the start of a valid sequence",
            "Select an O(1) lookup data structure to track elements",
            "Trace logic on a small example array",
            "Check boundary conditions and edge cases"
        ],
        description="4 to 5 step-by-step thinking milestones specific to this problem"
    )
    target_time_complexity: str = Field(
        default="O(n)",
        description="Optimal time complexity bound, e.g. O(n)"
    )
    target_space_complexity: str = Field(
        default="O(n)",
        description="Optimal space complexity bound, e.g. O(n)"
    )

def study_node(state: LeetCodeSolverState) -> dict:
    llm = get_llm()
    structured_llm = llm.with_structured_output(StudyModeOutput)

    prompt = f"""You are a Principal Software Engineer and Socratic LeetCode Mentor.
Analyze this problem and provide structured study guidance.

Problem Title: {state.get('problem_title')}
Description:
{state.get('problem_description')}

Constraints:
{state.get('problem_constraints')}

CRITICAL: You MUST populate ALL fields in the output schema:
- `pattern_name`, `why_not_brute_force`, `key_data_structure`, `core_intuition`
- `hints` (exactly 3 progressive hints)
- `cognitive_steps` (4-5 problem-specific reasoning steps)
- `target_time_complexity` (e.g. 'O(n)')
- `target_space_complexity` (e.g. 'O(n)')
"""
    result: StudyModeOutput = structured_llm.invoke(prompt)

    return {
        "study_output": result.model_dump(),
        "current_node": "study",
        "agent_logs": state.get("agent_logs", []) + [{
            "agent_name": "Study Mentor",
            "action": "Generated structured Socratic hints and steps",
            "status": "completed"
        }]
    }