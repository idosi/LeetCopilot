from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

import config.settings as settings
from src.schemas.state import TestCase


class _TestCaseModel(BaseModel):
    input_data: Dict[str, Any]
    expected_output: Any
    case_type: str
    description: str


class _GeneratorOutput(BaseModel):
    test_cases: List[_TestCaseModel]


_llm = ChatAnthropic(model=settings.LLM_MODEL, timeout=settings.LLM_TIMEOUT_SECONDS, temperature=0).with_structured_output(
    _GeneratorOutput
)

_SYSTEM_PROMPT = """\
You are a test case generator for LeetCode problems. Generate exactly 8 test cases:
- 3 base cases: typical examples directly from the problem description
- 3 edge cases: boundary conditions (empty input, single element, duplicates, negatives, max/min values, overflow)
- 2 large/stress cases: larger inputs to verify performance under load

Rules:
- input_data: dict whose keys are the exact parameter names from the function signature
- expected_output: the correct output for the given inputs (must be accurate per problem semantics)
- case_type: exactly "base", "edge", or "large"
- description: brief human-readable label for what is being tested
- All values must be JSON-serializable (use lists not tuples)
"""

_USER_TEMPLATE = """\
Problem Title: {title}

Problem Description:
{description}

Constraints:
{constraints}

Solution signature reference:
```python
{solution_code}
```
"""


def generate_test_cases(
    problem_title: str,
    problem_description: str,
    problem_constraints: str,
    solution_code: str,
) -> List[TestCase]:
    try:
        result: _GeneratorOutput = _llm.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(
                        title=problem_title,
                        description=problem_description,
                        constraints=problem_constraints,
                        solution_code=solution_code,
                    ),
                },
            ]
        )
        return [
            TestCase(
                input_data=tc.input_data,
                expected_output=tc.expected_output,
                case_type=tc.case_type,
                description=tc.description,
            )
            for tc in result.test_cases
        ]
    except Exception:
        return []
