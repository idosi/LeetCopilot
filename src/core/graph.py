from datetime import datetime, timezone
from langgraph.graph import END, StateGraph

from src.agents.study import study_node
from src.agents.code_review import code_review_node
from src.agents.documenter import documenter_node
from src.agents.solver import solver_node
from src.agents.performance import performance_node
from src.agents.tester import tester_node
from src.schemas.state import AgentLog, LeetCodeSolverState


def supervisor_node(state: LeetCodeSolverState) -> LeetCodeSolverState:
    new_state = dict(state)
    mode = state.get("mode", "full")
    new_state["current_node"] = "supervisor"
    new_state["supervisor_routing"] = mode
    new_state["total_agents_run"] = state.get("total_agents_run", 0) + 1

    log_entry: AgentLog = {
        "agent_name": "supervisor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": f"validate_and_route ({mode} mode)",
        "status": "completed",
        "metadata": {"mode": mode},
    }
    new_state["agent_logs"] = list(state.get("agent_logs", [])) + [log_entry]
    return new_state


def _make_stub(node_name: str):
    def _stub(state: LeetCodeSolverState) -> LeetCodeSolverState:
        new_state = dict(state)
        new_state["current_node"] = node_name
        new_state["total_agents_run"] = state.get("total_agents_run", 0) + 1
        return new_state
    _stub.__name__ = f"{node_name}_node"
    return _stub


error_handler_node = _make_stub("error_handler")


# --- Routing functions ---

def route_from_supervisor(state: LeetCodeSolverState) -> str:
    mode = state.get("mode", "full")
    if mode == "study":
        return "study"
    if mode == "review":
        return "code_review"
    return "solver"


def route_after_tester(state: LeetCodeSolverState) -> str:
    if state.get("error_logs"):
        return "error_handler"
    return "documenter"


# --- Graph construction ---

def build_graph():
    graph = StateGraph(LeetCodeSolverState)

    # 1. רישום כל הצמתים (כולל supervisor ו-performance)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("study", study_node)
    graph.add_node("solver", solver_node)
    graph.add_node("performance", performance_node)
    graph.add_node("tester", tester_node)
    graph.add_node("code_review", code_review_node)
    graph.add_node("documenter", documenter_node)
    graph.add_node("error_handler", error_handler_node)

    # 2. נקודת הכניסה היא ה-Supervisor Node
    graph.set_entry_point("supervisor")

    # 3. ניתוב מתוך ה-Supervisor לפי המוד
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "study": "study",
            "code_review": "code_review",
            "solver": "solver",
        },
    )

    # 4. מסלול Full Mode המלא: solver -> performance -> tester -> documenter
    graph.add_edge("solver", "performance")
    graph.add_edge("performance", "tester")
    graph.add_conditional_edges(
        "tester",
        route_after_tester,
        {
            "documenter": "documenter",
            "error_handler": "error_handler",
        },
    )

    # 5. מסלולי Study ו-Review עוברים ישר ל-documenter
    graph.add_edge("study", "documenter")
    graph.add_edge("code_review", "documenter")

    # 6. סגירת הגרף
    graph.add_edge("documenter", END)
    graph.add_edge("error_handler", END)

    return graph.compile()