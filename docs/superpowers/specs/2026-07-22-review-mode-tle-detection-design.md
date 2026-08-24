# Review Mode TLE Detection & Tester-First Pipeline

**Date:** 2026-07-22
**Status:** Approved

## Goal

Make Code Review mode reliably detect Time Limit Exceeded (TLE) risks and high time complexity by:
1. Running user code through the Tester node before Code Review so concrete runtime failures (including timeouts) are captured.
2. Strengthening the Review Agent system prompt to force explicit complexity comparison and TLE labeling.

---

## Architecture

### Current Review Mode Flow

```
entry → code_review → END
```

### New Review Mode Flow

```
entry → tester → code_review → END
```

Full mode is unchanged: `entry → solver → tester → documenter → END`

---

## Section 1: Graph Routing (`src/core/graph.py`)

**Changes:**

1. `route_by_mode`: map `"review"` → `"tester"` (was `"code_review"`).
2. Add `route_after_tester(state)` conditional routing function:
   - `mode == "review"` → `"code_review"`
   - otherwise → `"documenter"`
3. Replace unconditional `graph.add_edge("tester", "documenter")` with `graph.add_conditional_edges("tester", route_after_tester, {"code_review": "code_review", "documenter": "documenter"})`.
4. Update `set_conditional_entry_point` mapping: `"review"` → `"tester"`.

---

## Section 2: Tester Node (`src/agents/tester.py`)

**Changes:**

After the existing naive/optimal test block, add a review-mode branch:
- If `state.get("mode") == "review"` and `user_code` is present:
  - Generate test cases using `generate_test_cases(..., solution_code="")` — empty string is already accepted; cases are derived from problem description and constraints.
  - Run user_code through the same `_run()` helper (subprocess for Python, semantic for others using `language`).
  - Store results in `new_state["user_code_test_results"]`.
  - Log pass/fail counts in the `log_entry` metadata under a `"user_code"` key.
- `test_results` (the solver-mode field) is NOT written in review mode — the two modes stay cleanly separated.

---

## Section 3: State Schema (`src/schemas/state.py`)

**Change:** Add one optional field to `LeetCodeSolverState`:

```python
user_code_test_results: Optional[List[TestResult]]
```

No other schema changes.

---

## Section 4: Code Review Node (`src/agents/code_review.py`)

### 4a: System Prompt — TLE / Complexity Risk Block

Add a fourth critical instruction after the existing three:

> **4. TLE / COMPLEXITY RISK:**
> - First, identify the known theoretical optimal time complexity for this problem class (e.g., Two Sum → O(N), any sort → O(N log N)).
> - Compare the user's actual time complexity against that optimal.
> - If the user's complexity is **worse** than optimal (e.g., O(N²) vs O(N)), you MUST:
>   - Set `is_optimal` to `false`.
>   - Include the phrase **"TLE RISK: Time Limit Exceeded"** at the start of `optimality_gap`, followed by a concrete explanation using the problem constraints (e.g., "TLE RISK: Time Limit Exceeded — O(N²) with N=10⁵ yields ~10¹⁰ operations, well above the ~10⁸ op/sec limit").
> - This rule applies even if the code produces correct answers on small inputs.

### 4b: User Prompt — Runtime Test Summary Injection

Before invoking the LLM, read `user_code_test_results` from state. If present and non-empty, append a formatted section to the user prompt:

```
Runtime Test Results (from sandbox execution):
- Test 1: PASS (12ms)
- Test 2: TIMEOUT after 5000ms
- Test 3: FAIL — expected 3, got 5 (8ms)
Total: 1/3 passed
```

This gives the LLM concrete evidence of actual failures rather than requiring it to speculate.

---

## Error Handling

- If `generate_test_cases` fails in review mode, `user_code_test_results` is set to `[]` (empty list) and an error is appended to `error_logs`. Code_review proceeds without runtime context.
- If `user_code` is absent in review mode, tester skips the user_code block and sets `user_code_test_results = []`.

---

## What Does NOT Change

- Full mode (`solver → tester → documenter`) is unaffected.
- Study mode is unaffected.
- The `test_results` field is only written by the solver-mode tester path.
- The executor timeout (5s) is unchanged.
