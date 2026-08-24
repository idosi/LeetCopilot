from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime


class Solution(TypedDict):
    """Represents a single solution variant."""
    code: str
    description: str
    language: str
    approach: str


class ComplexityAnalysis(TypedDict):
    """Time and space complexity metrics."""
    time_complexity: str
    time_explanation: str
    space_complexity: str
    space_explanation: str
    trade_offs: str


class TestCase(TypedDict):
    """Represents a single test case."""
    input_data: Dict[str, Any]
    expected_output: Any
    case_type: str
    description: str


class TestResult(TypedDict):
    """Result of executing a test case."""
    test_case_id: str
    passed: bool
    actual_output: Optional[Any]
    execution_time_ms: float
    error_message: Optional[str]


class AgentLog(TypedDict):
    """Log entry for agent processing."""
    agent_name: str
    timestamp: str
    action: str
    status: str
    metadata: Dict[str, Any]


class UserCodeReview(TypedDict):
    """Code review result for user-submitted code."""
    time_complexity: str
    time_explanation: str
    space_complexity: str
    space_explanation: str
    is_optimal: bool
    optimality_gap: str
    code_quality_improvements: List[str]
    constant_factor_tips: List[str]
    optimization_roadmap: List[str]
    optimal_solution: str


class LeetCodeSolverState(TypedDict):
    """Complete application state for LangGraph."""
    problem_description: str
    problem_title: str
    problem_constraints: str
    language: str
    mode: str  # "full" | "study" | "review"
    study_output: Optional[Dict[str, Any]]  # populated only in study mode

    # LLM Engine & Usage Tracking
    model_name: Optional[str]
    total_tokens: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    estimated_cost_usd: Optional[float]

    naive_solution: Optional[Solution]
    optimal_solution: Optional[Solution]

    naive_complexity: Optional[ComplexityAnalysis]
    optimal_complexity: Optional[ComplexityAnalysis]

    generated_test_cases: List[TestCase]
    test_results: List[TestResult]
    all_tests_passed: bool

    markdown_report: Optional[str]

    current_node: str
    supervisor_routing: str
    error_logs: List[str]
    agent_logs: List[AgentLog]
    execution_start_time: str
    execution_end_time: Optional[str]

    total_agents_run: int
    graph_status: str
    user_code: Optional[str]
    user_code_review: Optional[UserCodeReview]
    user_code_test_results: Optional[List[TestResult]]