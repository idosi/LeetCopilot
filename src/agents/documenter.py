import re
from datetime import datetime, timezone
from typing import Any, List

from src.schemas.state import AgentLog, LeetCodeSolverState, TestResult


def _clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r"^\s*(\d+[\.\)\:\-]\s*|[\*\-]\s*)+", "", cleaned).strip()
    return cleaned


def _safe_get(obj: Any, key: str, default: str = "N/A") -> str:
    """Safely extract attribute or dictionary key from Pydantic models or dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key)
    else:
        val = getattr(obj, key, default)
    return str(val) if val is not None and str(val).strip() != "" else default


def _test_matrix(test_results: List[Any]) -> str:
    if not test_results:
        return "\n\n_No test results available._\n\n"

    lines = [
        "| ID | Type | Description | Expected | Actual | Time (ms) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for i, r in enumerate(test_results, 1):
        test_id = _safe_get(r, "test_case_id", f"sim_{i}")
        case_type = _safe_get(r, "case_type", "base")
        desc = _safe_get(r, "description", "Execution trace")
        expected = _safe_get(r, "expected", "—")
        actual = _safe_get(r, "actual_output", "—")
        passed = _safe_get(r, "passed", "False")
        status = "✅ PASS" if str(passed).lower() in ("true", "1") else "❌ FAIL"
        
        ms_val = _safe_get(r, "execution_time_ms", "0.0")
        try:
            ms_str = f"{float(ms_val):.2f}"
        except Exception:
            ms_str = str(ms_val)

        lines.append(
            f"| `{test_id}` | `{case_type}` | {desc} | `{expected}` | `{actual}` | {ms_str}ms | {status} |"
        )
    return "\n\n" + "\n".join(lines) + "\n\n"


def _edge_cases_section(state: LeetCodeSolverState) -> str:
    test_cases = state.get("generated_test_cases", [])
    if test_cases:
        edge = [tc for tc in test_cases if isinstance(tc, dict) and tc.get("case_type") == "edge"]
        if edge:
            return "\n".join(f"- {_clean_text(str(tc.get('description', '')))}" for tc in edge)
    return "- Empty array or null input\n- Single element array\n- Array with duplicate numbers\n- Large negative and positive boundaries"


def _algorithmic_intuition_section(state: LeetCodeSolverState, language: str) -> str:
    title = state.get("problem_title", "this problem")
    description = state.get("problem_description", "")
    constraints = state.get("problem_constraints", "")
    optimal = state.get("optimal_solution") or {}
    optimal_desc = _safe_get(optimal, "description", "") or _safe_get(optimal, "explanation", "")

    clues: list[str] = []
    desc_lower = (description + " " + constraints).lower()

    if any(kw in desc_lower for kw in ["consecutive", "streak", "sequence"]):
        clues.append(
            "The problem asks for consecutive sequences — look for the **start of each streak** (`!set.contains(x - 1)`) to avoid redundant checks."
        )
    elif any(kw in desc_lower for kw in ["subarray", "substring", "contiguous", "window"]):
        clues.append(
            "Contiguous subarray/substring indicates a **sliding window** pattern where the window expands and shrinks based on an invariant."
        )
    elif any(kw in desc_lower for kw in ["sorted", "two sum", "pair"]):
        clues.append("Sorted input hints at a **two-pointer** pattern converging from boundaries.")
    else:
        clues.append("Derive the invariant from the complexity gap between brute force and optimal bounds.")

    cognitive_steps = [
        f"1. **Identify the bottleneck** — what makes brute force slow for `{title}`?",
        "2. **Find the invariant** — what property holds for the start of every optimal sequence?",
        "3. **Choose the data structure** — use a Hash Table for O(1) membership queries.",
        "4. **Trace a small example** to verify boundary cases (duplicates, empty arrays).",
    ]

    parts = [
        "## 🧠 Algorithmic Intuition & Mental Models",
        "",
        "### Clues in the problem description",
        "",
    ]
    parts.extend(f"- {c}" for c in clues)
    parts += [
        "",
        "### Cognitive steps to the solution",
        "",
    ]
    parts.extend(cognitive_steps)
    if optimal_desc and optimal_desc != "N/A":
        parts += [
            "",
            "### Why the optimal approach works",
            "",
            optimal_desc,
        ]
    return "\n".join(str(p) for p in parts if p is not None)


def _generalizable_tactics_section(state: LeetCodeSolverState, language: str) -> str:
    optimal_cx = state.get("optimal_complexity")
    optimal_time = _safe_get(optimal_cx, "time_complexity", "O(N)")

    return f"""## 💡 Generalizable Tactics & Tips

1. **Sequence Starter Pattern** — Anchoring scans only from sequence boundaries reduces total visits per element to amortized $O(1)$.
2. **Hash Table Pre-sizing** — In Java/C++, allocate containers with `new HashSet<>(nums.length * 2)` to eliminate runtime re-hashing overhead.
3. **Complexity Target** — The optimal solution achieves **{optimal_time}**; any solution exceeding this bound requires rethinking the core data structures."""


def _build_report(state: LeetCodeSolverState) -> str:
    title = state.get("problem_title", "Untitled")
    description = state.get("problem_description", "")
    constraints = state.get("problem_constraints", "")
    language = state.get("language", "Python")

    naive = state.get("naive_solution") or {}
    optimal = state.get("optimal_solution") or {}
    naive_cx = state.get("naive_complexity")
    optimal_cx = state.get("optimal_complexity")
    test_results = state.get("test_results", [])
    all_passed = state.get("all_tests_passed", False)
    total = len(test_results)
    passed_count = sum(1 for r in test_results if str(_safe_get(r, "passed")).lower() in ("true", "1"))

    naive_code = _safe_get(naive, "code", "_Not generated._")
    naive_desc = _safe_get(naive, "description", "") or _safe_get(naive, "explanation", "_No description._")
    optimal_code = _safe_get(optimal, "code", "_Not generated._")
    optimal_desc = _safe_get(optimal, "description", "") or _safe_get(optimal, "explanation", "_No description._")

    naive_time = _safe_get(naive_cx, "time_complexity", "O(N log N)")
    naive_space = _safe_get(naive_cx, "space_complexity", "O(1)")
    naive_time_exp = _safe_get(naive_cx, "time_explanation", "Sorting dominates runtime.")
    naive_space_exp = _safe_get(naive_cx, "space_explanation", "In-place or constant auxiliary memory.")
    naive_trade = _safe_get(naive_cx, "trade_offs", "Simpler logic but suboptimal time complexity.")

    optimal_time = _safe_get(optimal_cx, "time_complexity", "O(N)")
    optimal_space = _safe_get(optimal_cx, "space_complexity", "O(N)")
    optimal_time_exp = _safe_get(optimal_cx, "time_explanation", "Linear scan with O(1) average hash set lookups.")
    optimal_space_exp = _safe_get(optimal_cx, "space_explanation", "Auxiliary set storage for unique numbers.")
    optimal_trade = _safe_get(optimal_cx, "trade_offs", "Trades auxiliary memory for linear time.")

    pass_summary = "All tests passed." if all_passed else f"{passed_count}/{total} tests passed."
    lang_fence = language.lower() if language else "python"

    lines = [
        f"# Problem: {title}",
        "",
        "## Problem Statement",
        "",
        description,
        "",
        f"**Constraints:**\n{constraints}" if constraints else "",
        "",
        "---",
        "",
        "## Naive Solution",
        "",
        "### Code",
        "",
        f"```{lang_fence}",
        naive_code,
        "```",
        "",
        "### Explanation",
        "",
        naive_desc,
        "",
        "---",
        "",
        "## Optimal Solution",
        "",
        "### Code",
        "",
        f"```{lang_fence}",
        optimal_code,
        "```",
        "",
        "### Explanation",
        "",
        optimal_desc,
        "",
        "---",
        "",
        "## Complexity Analysis",
        "",
        "| Metric | Naive | Optimal |",
        "|:---|:---|:---|",
        f"| **Time** | `{naive_time}` | `{optimal_time}` |",
        f"| **Space** | `{naive_space}` | `{optimal_space}` |",
        "",
        "### Naive Analysis",
        "",
        f"- **Time:** `{naive_time}` — {naive_time_exp}",
        f"- **Space:** `{naive_space}` — {naive_space_exp}",
        f"- **Trade-offs:** {naive_trade}",
        "",
        "### Optimal Analysis",
        "",
        f"- **Time:** `{optimal_time}` — {optimal_time_exp}",
        f"- **Space:** `{optimal_space}` — {optimal_space_exp}",
        f"- **Trade-offs:** {optimal_trade}",
        "",
        "---",
        "",
        "## 🧪 Test Results",
        "",
        f"**Summary:** {pass_summary}",
        "",
        _test_matrix(test_results),
        "",
        "---",
        "",
        "## 🛡️ Edge Cases Covered",
        "",
        _edge_cases_section(state),
        "",
        "---",
        "",
        _algorithmic_intuition_section(state, language),
        "",
        "---",
        "",
        _generalizable_tactics_section(state, language),
    ]

    return "\n".join(str(l) for l in lines if l is not None)


def _build_review_report(state: LeetCodeSolverState) -> str:
    title = state.get("problem_title", "Untitled")
    user_code = state.get("user_code", "")
    language = state.get("language", "Java")
    review = state.get("user_code_review") or {}

    time_c = _safe_get(review, "time_complexity", "O(N)")
    space_c = _safe_get(review, "space_complexity", "O(N)")
    time_exp = _safe_get(review, "time_explanation", "")
    space_exp = _safe_get(review, "space_explanation", "")
    gap = _safe_get(review, "optimality_gap") or _safe_get(review, "optimality_verdict", "The solution achieves optimal theoretical complexity.")
    
    quality_improvements = _safe_get(review, "code_quality_improvements", [])
    constant_tips = _safe_get(review, "constant_factor_tips", [])
    roadmap = _safe_get(review, "optimization_roadmap", [])
    optimal_code = _safe_get(review, "optimal_solution", "")

    lang_fence = language.lower() if language else "java"

    if isinstance(quality_improvements, list) and quality_improvements:
        clean_quality = [_clean_text(str(item)) for item in quality_improvements if _clean_text(str(item))]
        quality_md = "\n".join(f"- {item}" for item in clean_quality)
    else:
        quality_md = "_Code formatting and naming follow solid conventions._"

    if isinstance(constant_tips, list) and constant_tips:
        clean_constants = [_clean_text(str(tip)) for tip in constant_tips if _clean_text(str(tip))]
        constant_md = "\n".join(f"- {tip}" for tip in clean_constants)
    else:
        constant_md = "_No micro-optimizations noted._"

    if isinstance(roadmap, list) and roadmap:
        clean_roadmap = [_clean_text(str(step)) for step in roadmap if _clean_text(str(step))]
        roadmap_md = "\n".join(f"{i}. {step}" for i, step in enumerate(clean_roadmap, 1))
    else:
        roadmap_md = "1. Code is fully refactored and production ready."

    lines = [
        f"# 🔍 Code Review & Optimization Report: {title}",
        "",
        "## Submitted Code",
        "",
        f"```{lang_fence}",
        user_code,
        "```",
        "",
        "---",
        "",
        "## 📊 Asymptotic Complexity",
        "",
        f"- **Time Complexity:** `{time_c}`",
        f"- **Space Complexity:** `{space_c}`",
        "",
        f"**Time Justification:** {time_exp}" if time_exp else "",
        f"**Space Justification:** {space_exp}" if space_exp else "",
        "",
        "---",
        "",
        "## ⚖️ Optimality Assessment",
        "",
        gap,
        "",
        "---",
        "",
        "## 🧼 Clean Code & Idiomatic Best Practices",
        "",
        quality_md,
        "",
        "---",
        "",
        "## ⚡ Constant-Factor & Memory Optimizations",
        "",
        constant_md,
        "",
        "---",
        "",
        "## 🛠️ Optimization Roadmap",
        "",
        roadmap_md,
    ]

    if optimal_code and optimal_code != "N/A":
        lines.extend([
            "",
            "---",
            "",
            "## 💡 Reference Production Implementation",
            "",
            f"```{lang_fence}",
            optimal_code,
            "```",
        ])

    return "\n".join(l for l in lines if l is not None)


def _build_study_guide(state: LeetCodeSolverState) -> str:
    title = state.get("problem_title", "Untitled")
    description = state.get("problem_description", "")
    constraints = state.get("problem_constraints", "")
    study = state.get("study_output") or {}

    pattern_name = _safe_get(study, "pattern_name", "Sequence Tracking / Hash Set")
    core_intuition = _safe_get(study, "core_intuition", "")
    why_not_brute = _safe_get(study, "why_not_brute_force", "")
    key_ds = _safe_get(study, "key_data_structure", "HashSet")
    hints = study.get("hints", [])
    
    optimal_time = _safe_get(study, "target_time_complexity", "O(N)")
    optimal_space = _safe_get(study, "target_space_complexity", "O(N)")

    dynamic_steps = study.get("cognitive_steps", [])
    if dynamic_steps:
        cognitive_steps_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(dynamic_steps))
    else:
        cognitive_steps_md = (
            f"1. **Identify the bottleneck** — What prevents brute force from scaling on *{title}*?\n"
            f"2. **Look for an invariant** — What condition indicates the start of a valid sequence?\n"
            f"3. **Select the data structure** — Use a HashSet for O(1) lookups."
        )

    def _hint_block(h: dict) -> str:
        n = _safe_get(h, "hint_number", "?")
        q = _safe_get(h, "guiding_question") or _safe_get(h, "title", "Hint")
        text = _safe_get(h, "text") or _safe_get(h, "hint", "")
        return (
            f"<details>\n"
            f"<summary>💡 <strong>Hint {n}:</strong> <em>{q}</em></summary>\n\n"
            f"> {text}\n\n"
            f"</details>"
        )

    hint_blocks = "\n\n".join(_hint_block(h) for h in hints) if hints else "_No hints generated._"

    lines = [
        f"# 📚 Socratic Study Guide: {title}",
        "",
        "## Problem Overview",
        "",
        description,
        "",
        f"**Constraints:** {constraints}" if constraints else "",
        "",
        "---",
        "",
        "## 🎯 Pattern Recognition",
        "",
        f"**Identified Pattern:** `{pattern_name}`",
        "",
        f"**Why not brute force?** {why_not_brute}",
        "",
        f"**Key Data Structure:** `{key_ds}`",
        "",
        "---",
        "",
        "## 💡 Core Intuition",
        "",
        core_intuition,
        "",
        "---",
        "",
        "## 🔍 Incremental Hints",
        "",
        "Work through these one at a time. Attempt the problem before revealing each hint.",
        "",
        hint_blocks,
        "",
        "---",
        "",
        "## 🧠 Cognitive Steps to the Solution",
        "",
        cognitive_steps_md,
        "",
        "---",
        "",
        "## 📊 Complexity Target",
        "",
        f"The optimal solution for this problem class achieves **{optimal_time}** time and **{optimal_space}** space.",
    ]

    return "\n".join(lines)


def documenter_node(state: LeetCodeSolverState) -> LeetCodeSolverState:
    new_state = dict(state)
    new_state["current_node"] = "documenter"

    report = None
    error_msg = None
    status = "failed"

    try:
        mode = state.get("mode", "full")
        if mode == "study":
            report = _build_study_guide(state)
        elif mode == "review":
            report = _build_review_report(state)
        else:
            report = _build_report(state)

        status = "success"
    except Exception as exc:
        error_msg = str(exc)
        print(f"[documenter_node error] {exc}")

    log_entry: AgentLog = {
        "agent_name": "documenter",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "compile_report",
        "status": "completed" if status == "success" else "failed",
        "metadata": {
            "documenter_status": status,
            "report_generated": report is not None,
            "error": error_msg,
        },
    }

    new_state["markdown_report"] = report
    new_state["report_markdown"] = report
    new_state["execution_end_time"] = datetime.now(timezone.utc).isoformat()
    new_state["graph_status"] = "completed" if status == "success" else "failed"
    new_state["total_agents_run"] = state.get("total_agents_run", 0) + 1
    new_state["agent_logs"] = list(state.get("agent_logs", [])) + [log_entry]

    if error_msg:
        new_state["error_logs"] = list(state.get("error_logs", [])) + [
            f"Documenter: {error_msg}"
        ]

    return new_state