from typing import TypedDict, Optional, Dict, Any


class TestCase(TypedDict):
    input_data: Dict[str, Any]
    expected_output: Any
    case_type: str
    description: str


class TestResult(TypedDict):
    test_case_id: str
    passed: bool
    actual_output: Optional[Any]
    execution_time_ms: float
    error_message: Optional[str]
