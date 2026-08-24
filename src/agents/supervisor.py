from datetime import datetime, timezone

from src.schemas.state import AgentLog, LeetCodeSolverState


def _is_valid_problem(description: str) -> bool:
    stripped = description.strip() if description else ""
    return len(stripped) >= 20


def supervisor_node(state: LeetCodeSolverState) -> LeetCodeSolverState:
    new_state = dict(state)

    valid = _is_valid_problem(state.get("problem_description", ""))
    routing = "solver" if valid else "error_handler"

    log_entry: AgentLog = {
        "agent_name": "supervisor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": f"routing_decision:{routing}",
        "status": "completed" if valid else "failed",
        "metadata": {
            "routing_decision": routing,
            "problem_title": state.get("problem_title", ""),
            "valid_problem": valid,
        },
    }

    new_state["current_node"] = "supervisor"
    new_state["supervisor_routing"] = routing
    new_state["total_agents_run"] = state.get("total_agents_run", 0) + 1
    new_state["agent_logs"] = list(state.get("agent_logs", [])) + [log_entry]

    if not valid:
        new_state["graph_status"] = "failed"
        new_state["error_logs"] = list(state.get("error_logs", [])) + [
            f"Supervisor: invalid or missing problem description "
            f"(title='{state.get('problem_title', '')}')."
        ]

    return new_state
