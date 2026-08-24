from typing import TypedDict, Optional, List

from src.schemas.solution import Solution
from src.schemas.complexity import ComplexityAnalysis
from src.schemas.test_case import TestCase, TestResult


class SupervisorInput(TypedDict):
    problem_description: str
    problem_title: str
    problem_constraints: str


class SupervisorOutput(TypedDict):
    routing_decision: str
    next_agent: str
    reasoning: str
    supervisor_routing: str


class SolverInput(TypedDict):
    problem_description: str
    problem_title: str
    problem_constraints: str


class SolverOutput(TypedDict):
    naive_solution: Solution
    optimal_solution: Solution
    solver_status: str
    solver_error: Optional[str]


class PerformanceInput(TypedDict):
    naive_solution_code: str
    optimal_solution_code: str
    problem_description: str


class PerformanceOutput(TypedDict):
    naive_complexity: ComplexityAnalysis
    optimal_complexity: ComplexityAnalysis
    performance_comparison: str
    performance_status: str


class TesterInput(TypedDict):
    naive_solution_code: str
    optimal_solution_code: str
    generated_test_cases: List[TestCase]


class TesterOutput(TypedDict):
    test_results: List[TestResult]
    all_tests_passed: bool
    failure_summary: Optional[str]
    tester_status: str


class DocumenterInput(TypedDict):
    problem_title: str
    problem_description: str
    problem_constraints: str
    naive_solution: Solution
    optimal_solution: Solution
    naive_complexity: ComplexityAnalysis
    optimal_complexity: ComplexityAnalysis
    test_results: List[TestResult]
    all_tests_passed: bool


class DocumenterOutput(TypedDict):
    markdown_report: str
    documenter_status: str
