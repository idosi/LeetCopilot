# Multi-Agent LeetCode Solver - Architecture Document

## 1. Project Structure

```
leetcode-solver/
├── src/
│   ├── __init__.py
│   ├── main.py                          # Entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── graph.py                     # LangGraph workflow definition
│   │   ├── state_manager.py             # State initialization and mutations
│   │   └── constants.py                 # Global constants and timeouts
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py                # Supervisor (Manager) agent
│   │   ├── solver.py                    # Solver agent (naive + optimal)
│   │   ├── performance.py               # Performance analysis agent
│   │   ├── tester.py                    # Tester agent with sandbox
│   │   └── documenter.py                # Documenter agent
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── state.py                     # State TypedDict definitions
│   │   ├── messages.py                  # Agent input/output schemas
│   │   ├── solution.py                  # Solution representation schemas
│   │   └── test_case.py                 # Test case and result schemas
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── executor.py                  # Code execution with timeout/limits
│   │   ├── test_generator.py            # Edge case generation logic
│   │   └── result_parser.py             # Parse execution results
│   └── utils/
│       ├── __init__.py
│       ├── llm_client.py                # LLM abstraction layer
│       └── validators.py                # Input validation helpers
├── tests/
│   ├── __init__.py
│   ├── test_graph_flow.py               # Graph execution tests
│   ├── test_sandbox.py                  # Sandbox isolation tests
│   └── test_state_mutations.py          # State mutation tests
├── config/
│   ├── __init__.py
│   └── settings.py                      # Configuration (LLM keys, timeouts)
├── requirements.txt                      # Python dependencies
├── pyproject.toml                        # Project metadata
└── README.md                             # Project documentation
```

---

## 2. State Definition

### 2.1 Core State Schema

```python
# schemas/state.py

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime

class Solution(TypedDict):
    """Represents a single solution variant."""
    code: str
    description: str
    language: str  # "python", "java", etc.
    approach: str  # "naive", "optimal"

class ComplexityAnalysis(TypedDict):
    """Time and space complexity metrics."""
    time_complexity: str  # e.g., "O(n log n)"
    time_explanation: str
    space_complexity: str  # e.g., "O(n)"
    space_explanation: str
    trade_offs: str

class TestCase(TypedDict):
    """Represents a single test case."""
    input_data: Dict[str, Any]
    expected_output: Any
    case_type: str  # "base", "edge", "large"
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
    status: str  # "started", "completed", "failed"
    metadata: Dict[str, Any]

class LeetCodeSolverState(TypedDict):
    """Complete application state for LangGraph."""
    # === Input ===
    problem_description: str
    problem_title: str
    problem_constraints: str
    
    # === Solutions (Solver Agent Output) ===
    naive_solution: Optional[Solution]
    optimal_solution: Optional[Solution]
    
    # === Performance Analysis (Performance Agent Output) ===
    naive_complexity: Optional[ComplexityAnalysis]
    optimal_complexity: Optional[ComplexityAnalysis]
    
    # === Test Cases (Tester Agent Input/Output) ===
    generated_test_cases: List[TestCase]  # Input to tester
    test_results: List[TestResult]        # Output from tester
    all_tests_passed: bool
    
    # === Documentation (Documenter Agent Output) ===
    markdown_report: Optional[str]
    
    # === Control Flow ===
    current_node: str
    supervisor_routing: str  # Next agent to route to
    error_logs: List[str]
    agent_logs: List[AgentLog]
    execution_start_time: str
    execution_end_time: Optional[str]
    
    # === Metadata ===
    total_agents_run: int
    graph_status: str  # "running", "completed", "failed"
```

### 2.2 State Mutation Constraints

- **Immutability**: All state updates are deterministic and occur only at node boundaries.
- **No Partial Updates**: Each agent mutates its designated output fields atomically.
- **Versioning**: `agent_logs` tracks every state transition for auditability.
- **Defaults**: Uninitialized optional fields MUST be `None` until explicitly populated.
- **Determinism**: State mutations are idempotent for retry scenarios.

---

## 3. Agent Interfaces

### 3.1 Supervisor (Manager) Agent

**Purpose**: Route tasks to appropriate downstream agents; coordinate workflow execution.

**Input Schema**:
```python
# schemas/messages.py

class SupervisorInput(TypedDict):
    """Input to Supervisor node."""
    problem_description: str
    problem_title: str
    problem_constraints: str
```

**Output Schema**:
```python
class SupervisorOutput(TypedDict):
    """Output from Supervisor node."""
    routing_decision: str  # "solver", "complete", "skip"
    next_agent: str       # Target agent name
    reasoning: str
    supervisor_routing: str  # Stored in state
```

**Responsibilities**:
1. Parse and validate problem description.
2. Decide execution order: typically Solver → Performance → Tester → Documenter.
3. Short-circuit execution if problem is unsolvable or invalid.
4. Update state with routing decision.

**State Mutations**:
- `supervisor_routing`: Set to next agent name.
- `current_node`: Set to `"supervisor"`.
- `agent_logs`: Append entry with routing decision.

---

### 3.2 Solver Agent

**Purpose**: Generate naive and optimal solutions to the problem.

**Input Schema**:
```python
class SolverInput(TypedDict):
    """Input to Solver node."""
    problem_description: str
    problem_title: str
    problem_constraints: str
```

**Output Schema**:
```python
class SolverOutput(TypedDict):
    """Output from Solver node."""
    naive_solution: Solution
    optimal_solution: Solution
    solver_status: str  # "success", "partial_success", "failed"
    solver_error: Optional[str]
```

**Responsibilities**:
1. Call LLM to generate a naive (brute-force) solution.
2. Call LLM to generate an optimal (refined) solution.
3. Validate solution syntax (Python code parsing).
4. Ensure both solutions have:
   - Valid Python code (parseable).
   - Clear docstrings.
   - Function signature matching problem requirements.
5. Fallback: If optimal solution fails, attempt alternative approaches.

**State Mutations**:
- `naive_solution`: Populated with validated solution.
- `optimal_solution`: Populated with validated solution.
- `current_node`: Set to `"solver"`.
- `agent_logs`: Append entry with generation status.

**Constraints**:
- Both solutions MUST be syntactically valid Python.
- Code MUST NOT import external libraries (unless pre-approved).
- Function signature MUST match problem specification.

---

### 3.3 Performance Agent

**Purpose**: Analyze time and space complexity of generated solutions.

**Input Schema**:
```python
class PerformanceInput(TypedDict):
    """Input to Performance node."""
    naive_solution_code: str
    optimal_solution_code: str
    problem_description: str
```

**Output Schema**:
```python
class PerformanceOutput(TypedDict):
    """Output from Performance node."""
    naive_complexity: ComplexityAnalysis
    optimal_complexity: ComplexityAnalysis
    performance_comparison: str  # Summary of differences
    performance_status: str  # "success", "failed"
```

**Responsibilities**:
1. Analyze naive solution:
   - Time complexity (Big O).
   - Space complexity (Big O).
   - Reasoning for each.
2. Analyze optimal solution:
   - Time complexity.
   - Space complexity.
   - Trade-offs vs. naive approach.
3. Validate complexity calculations (heuristic checks).
4. Generate comparison summary.

**State Mutations**:
- `naive_complexity`: Populated with analysis.
- `optimal_complexity`: Populated with analysis.
- `current_node`: Set to `"performance"`.
- `agent_logs`: Append entry with analysis results.

**Constraints**:
- Complexity expressions MUST follow standard Big O notation.
- Explanations MUST be clear and justified.
- Trade-offs MUST be explicitly documented.

---

### 3.4 Tester Agent

**Purpose**: Execute solutions against test cases in an isolated sandbox; validate correctness.

**Input Schema**:
```python
class TesterInput(TypedDict):
    """Input to Tester node."""
    naive_solution_code: str
    optimal_solution_code: str
    generated_test_cases: List[TestCase]
```

**Output Schema**:
```python
class TesterOutput(TypedDict):
    """Output from Tester node."""
    test_results: List[TestResult]
    all_tests_passed: bool
    failure_summary: Optional[str]  # If any test failed
    tester_status: str  # "success", "partial_success", "failed"
```

**Responsibilities**:
1. Receive pre-generated test cases.
2. Execute naive solution against all test cases (timeout: 5s per case).
3. Execute optimal solution against all test cases (timeout: 5s per case).
4. Capture:
   - Execution time (milliseconds).
   - Actual output.
   - Pass/fail status.
   - Error messages (if any).
5. Aggregate results and determine overall pass/fail.
6. Sanitize error messages (no sensitive output).

**State Mutations**:
- `test_results`: Populated with all test results.
- `all_tests_passed`: Set to `True` if all tests pass, else `False`.
- `current_node`: Set to `"tester"`.
- `agent_logs`: Append entry with test summary.

**Constraints**:
- **Sandbox Isolation**: Code execution MUST occur in isolated subprocess.
- **Timeout**: Each test case MUST have max 5-second execution limit.
- **Memory Limit**: Limit subprocess memory to 512 MB.
- **No Side Effects**: Code MUST NOT write to disk or network.
- **Error Capture**: All exceptions MUST be caught and logged.

---

### 3.5 Documenter Agent

**Purpose**: Compile findings into a comprehensive Markdown report.

**Input Schema**:
```python
class DocumenterInput(TypedDict):
    """Input to Documenter node."""
    problem_title: str
    problem_description: str
    problem_constraints: str
    naive_solution: Solution
    optimal_solution: Solution
    naive_complexity: ComplexityAnalysis
    optimal_complexity: ComplexityAnalysis
    test_results: List[TestResult]
    all_tests_passed: bool
```

**Output Schema**:
```python
class DocumenterOutput(TypedDict):
    """Output from Documenter node."""
    markdown_report: str
    documenter_status: str  # "success", "failed"
```

**Responsibilities**:
1. Structure Markdown report with sections:
   - Problem Statement.
   - Naive Solution (code + explanation).
   - Optimal Solution (code + explanation).
   - Complexity Analysis (table comparison).
   - Test Results (pass/fail summary).
   - Edge Cases Covered.
   - Key Insights.
2. Format code blocks with syntax highlighting.
3. Include execution times and performance metrics.
4. Generate test matrix (input → expected → actual → status).

**State Mutations**:
- `markdown_report`: Populated with full report.
- `current_node`: Set to `"documenter"`.
- `execution_end_time`: Set to current timestamp.
- `graph_status`: Set to `"completed"`.
- `agent_logs`: Append entry with documentation completion.

**Constraints**:
- Report MUST be valid Markdown (no embedded HTML).
- Code blocks MUST use triple backticks with language specification.
- All numerical values (complexity, execution time) MUST be included.

---

## 4. Graph Flow

### 4.1 Workflow Topology

```
START
  │
  ├─→ [SUPERVISOR]
      │
      ├─ Validates problem
      ├─ Routes to next agent
      └─ Logs routing decision
          │
          └─→ [SOLVER]
              │
              ├─ Generate naive solution
              ├─ Generate optimal solution
              ├─ Validate syntax
              └─ Return solutions OR error
                  │
                  ├─ Success? YES
                  │   └─→ [PERFORMANCE]
                  │       │
                  │       ├─ Analyze naive complexity
                  │       ├─ Analyze optimal complexity
                  │       ├─ Compare trade-offs
                  │       └─ Return analysis
                  │           │
                  │           └─→ [TEST_CASE_GENERATION]
                  │               │ (Internal to Tester)
                  │               │
                  │               └─→ [TESTER]
                  │                   │
                  │                   ├─ Execute naive against tests
                  │                   ├─ Execute optimal against tests
                  │                   ├─ Capture results
                  │                   └─ Return test matrix
                  │                       │
                  │                       └─→ [DOCUMENTER]
                  │                           │
                  │                           ├─ Compile report
                  │                           ├─ Format Markdown
                  │                           └─ Return report
                  │                               │
                  │                               └─→ END (Success)
                  │
                  └─ Success? NO
                      └─→ [ERROR_HANDLER]
                          │
                          ├─ Log error
                          ├─ Attempt retry (optional)
                          └─→ END (Failed)
```

### 4.2 Node Definitions and Transitions

#### 4.2.1 START Node
- **Type**: Implicit entry point.
- **Action**: Initialize state with user input.
- **Output**: LeetCodeSolverState with `problem_description`, `problem_title`, `problem_constraints`.
- **Next**: `supervisor`

#### 4.2.2 SUPERVISOR Node
- **Type**: Routing/Conditional.
- **Input**: LeetCodeSolverState
- **Processing**:
  1. Validate problem description (non-empty, meaningful).
  2. Determine execution path.
  3. Update `supervisor_routing` with next agent name.
- **Output**: LeetCodeSolverState with `supervisor_routing` set.
- **Edges**:
  - **Valid problem**: Route to `solver`.
  - **Invalid problem**: Route to error handler, set `graph_status = "failed"`.

#### 4.2.3 SOLVER Node
- **Type**: Computational/Agent.
- **Input**: LeetCodeSolverState (problem fields).
- **Processing**:
  1. Generate naive solution (LLM call).
  2. Validate syntax.
  3. Generate optimal solution (LLM call).
  4. Validate syntax.
  5. Update state.
- **Output**: LeetCodeSolverState with `naive_solution`, `optimal_solution`.
- **Edges**:
  - **Both solutions generated**: Route to `performance`.
  - **Generation failed**: Route to error handler.

#### 4.2.4 PERFORMANCE Node
- **Type**: Computational/Agent.
- **Input**: LeetCodeSolverState (solution codes).
- **Processing**:
  1. Analyze naive solution complexity.
  2. Analyze optimal solution complexity.
  3. Generate comparison.
  4. Update state.
- **Output**: LeetCodeSolverState with `naive_complexity`, `optimal_complexity`.
- **Edges**:
  - **Always route to**: `tester` (even if analysis is partial).

#### 4.2.5 TEST_CASE_GENERATION (Sub-process within Tester)
- **Type**: Computational.
- **Input**: Problem description, constraints.
- **Processing**:
  1. Generate base test cases (common examples).
  2. Generate edge case test cases (boundaries, special values).
  3. Generate stress test cases (large inputs).
- **Output**: List of TestCase objects.
- **Next**: Execute test cases via sandbox.

#### 4.2.6 TESTER Node
- **Type**: Computational/Agent + Sandbox.
- **Input**: LeetCodeSolverState (solutions + test cases).
- **Processing**:
  1. Generate test cases (internal sub-process).
  2. Execute naive solution in sandbox.
  3. Execute optimal solution in sandbox.
  4. Aggregate results.
  5. Update state.
- **Output**: LeetCodeSolverState with `test_results`, `all_tests_passed`.
- **Edges**:
  - **Always route to**: `documenter` (even if tests fail).

#### 4.2.7 DOCUMENTER Node
- **Type**: Computational/Agent.
- **Input**: LeetCodeSolverState (all fields).
- **Processing**:
  1. Compile Markdown report.
  2. Format solutions, complexity, test results.
  3. Update state.
- **Output**: LeetCodeSolverState with `markdown_report`.
- **Edges**:
  - **Always route to**: END.

#### 4.2.8 END Node
- **Type**: Implicit exit point.
- **Action**: Return final LeetCodeSolverState.
- **Output**: Markdown report + execution logs.

### 4.3 Conditional Routing Logic

```python
# Pseudo-code for routing decisions

def route_supervisor(state: LeetCodeSolverState) -> str:
    if not is_valid_problem(state.problem_description):
        return "error_handler"
    else:
        return "solver"

def route_solver(state: LeetCodeSolverState) -> str:
    if state.naive_solution and state.optimal_solution:
        return "performance"
    else:
        return "error_handler"

def route_performance(state: LeetCodeSolverState) -> str:
    return "tester"  # Unconditional

def route_tester(state: LeetCodeSolverState) -> str:
    return "documenter"  # Unconditional

def route_documenter(state: LeetCodeSolverState) -> str:
    return "end"  # Unconditional
```

---

## 5. Execution Guarantees

### 5.1 Determinism
- Same input MUST produce same output (given same LLM model/temperature).
- State transitions are fully logged in `agent_logs`.

### 5.2 Fault Tolerance
- **Timeout Handling**: All LLM calls have 30-second timeout.
- **Sandbox Timeout**: All code execution has 5-second timeout per test case.
- **Graceful Degradation**: Failures in optional agents (e.g., Performance) do not block Documenter.

### 5.3 Audit Trail
- Every state mutation logged to `agent_logs` with timestamp.
- Execution timeline captured: `execution_start_time`, `execution_end_time`.
- Error logs preserved in `error_logs` for post-mortem analysis.

---

## 6. Schema Definitions - Complete Reference

### 6.1 Solution Schema
```python
class Solution(TypedDict):
    code: str                    # Full, executable code
    description: str             # High-level explanation
    language: str                # "python" (extensible)
    approach: str                # "naive", "optimal"
```

### 6.2 ComplexityAnalysis Schema
```python
class ComplexityAnalysis(TypedDict):
    time_complexity: str         # e.g., "O(n^2)"
    time_explanation: str        # Justification
    space_complexity: str        # e.g., "O(n)"
    space_explanation: str       # Justification
    trade_offs: str              # vs. other approaches
```

### 6.3 TestCase Schema
```python
class TestCase(TypedDict):
    input_data: Dict[str, Any]   # Serializable input dict
    expected_output: Any         # Serializable expected result
    case_type: str               # "base", "edge", "large"
    description: str             # Human-readable description
```

### 6.4 TestResult Schema
```python
class TestResult(TypedDict):
    test_case_id: str            # Unique identifier
    passed: bool                 # Pass/fail verdict
    actual_output: Optional[Any] # Actual execution result
    execution_time_ms: float     # Wall-clock time
    error_message: Optional[str] # Exception message (if failed)
```

### 6.5 AgentLog Schema
```python
class AgentLog(TypedDict):
    agent_name: str              # "supervisor", "solver", etc.
    timestamp: str               # ISO 8601 format
    action: str                  # What agent was doing
    status: str                  # "started", "completed", "failed"
    metadata: Dict[str, Any]     # Flexible data
```

---

## 7. Implementation Constraints

### 7.1 Separation of Concerns
- **Agents**: Each agent file (`agents/`) contains only agent logic and LLM calls.
- **Schemas**: All TypedDicts defined in `schemas/`.
- **Execution**: Sandbox isolated in `sandbox/executor.py`.
- **Routing**: All conditional logic in `core/graph.py`.

### 7.2 State Immutability
- State modifications ONLY at agent boundaries.
- Use `state.copy()` before mutations (Python TypedDict pattern).
- No in-place modifications of nested structures.

### 7.3 Error Handling
- All exceptions caught at node level.
- Error details logged to `error_logs` and `agent_logs`.
- Execution continues where possible (non-fatal errors).

### 7.4 Code Quality
- All agent functions have clear input/output contracts.
- Function signatures match schema types exactly.
- No implicit type conversions.

---

## 8. LangGraph Integration Points

### 8.1 Graph Definition
```python
# core/graph.py - Pseudo-structure

from langgraph.graph import StateGraph
from schemas.state import LeetCodeSolverState

def build_graph():
    """Construct the LeetCode Solver graph."""
    graph = StateGraph(LeetCodeSolverState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("solver", solver_node)
    graph.add_node("performance", performance_node)
    graph.add_node("tester", tester_node)
    graph.add_node("documenter", documenter_node)
    
    # Add edges
    graph.add_edge("supervisor", "solver")
    graph.add_edge("solver", "performance")
    graph.add_edge("performance", "tester")
    graph.add_edge("tester", "documenter")
    graph.add_edge("documenter", END)
    
    # Conditional edges
    graph.add_conditional_edges("supervisor", route_supervisor)
    graph.add_conditional_edges("solver", route_solver)
    
    # Set entry point
    graph.set_entry_point("supervisor")
    
    return graph.compile()
```

### 8.2 Node Function Signature
```python
# Each node function follows this contract:

def agent_node(state: LeetCodeSolverState) -> LeetCodeSolverState:
    """
    Process input state and return mutated state.
    
    Args:
        state: Current application state
        
    Returns:
        Updated state with agent outputs populated
    """
    # Mutation logic
    return updated_state
```

---

## 9. Testing Strategy

### 9.1 Unit Tests
- **test_graph_flow.py**: Verify node transitions and routing logic.
- **test_state_mutations.py**: Validate state schema compliance.
- **test_sandbox.py**: Verify code execution isolation and timeouts.

### 9.2 Integration Tests
- End-to-end workflow execution with mock LLM.
- State consistency across all nodes.
- Error handling and fallback paths.

### 9.3 Sandbox Validation
- Execute unsafe code (infinite loops, memory bombs).
- Verify timeout enforcement.
- Verify no side-effects (disk, network).

---

## 10. Configuration & Secrets

### 10.1 Environment Variables
```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4
LLM_TIMEOUT_SECONDS=30
SANDBOX_TIMEOUT_SECONDS=5
SANDBOX_MEMORY_LIMIT_MB=512
```

### 10.2 Runtime Settings
- Defined in `config/settings.py`.
- Loaded via environment or config file.
- Validated on startup.

---

## 11. Output Specification

### 11.1 Final Output
The system returns:
1. **Markdown Report** (via `markdown_report` field).
2. **Execution Logs** (via `agent_logs` and `error_logs`).
3. **State Snapshot** (complete final `LeetCodeSolverState`).

### 11.2 Markdown Report Structure
```
# Problem: [Title]

## Problem Statement
[Description + Constraints]

## Naive Solution
### Code
\`\`\`python
[Code]
\`\`\`
### Explanation
[Description]

## Optimal Solution
### Code
\`\`\`python
[Code]
\`\`\`
### Explanation
[Description]

## Complexity Analysis

| Metric | Naive | Optimal |
|--------|-------|---------|
| Time | [T1] | [T2] |
| Space | [S1] | [S2] |

### Analysis
[naive_complexity details]
[optimal_complexity details]

## Test Results
[Test matrix: input → expected → actual → status]
[Pass rate: X/Y]

## Edge Cases Covered
[List of edge cases tested]

## Key Insights
[Key takeaways]
```

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **Naive Solution** | Brute-force, straightforward approach without optimization. |
| **Optimal Solution** | Refined approach with better time/space complexity. |
| **Big O Analysis** | Asymptotic complexity notation (time/space). |
| **Sandbox** | Isolated subprocess for executing untrusted code. |
| **State Mutation** | Modification of `LeetCodeSolverState` at node boundaries. |
| **Routing** | LangGraph conditional logic determining next node. |
| **Determinism** | Guarantee that same input always produces same output. |

---

## 13. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-05-25 | Architecture | Initial comprehensive specification |