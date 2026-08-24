from src.agents.code_review import _format_test_summary


def test_format_empty_results_returns_empty_string():
    assert _format_test_summary([]) == ""


def test_format_all_pass():
    results = [
        {"passed": True, "execution_time_ms": 10.0, "error_message": None, "actual_output": 1},
        {"passed": True, "execution_time_ms": 8.0,  "error_message": None, "actual_output": 2},
    ]
    summary = _format_test_summary(results)
    assert "PASS" in summary
    assert "Total: 2/2 passed" in summary
    assert "FAIL" not in summary
    assert "TIMEOUT" not in summary


def test_format_timeout_labeled_correctly():
    results = [
        {"passed": False, "execution_time_ms": 5000.0,
         "error_message": "Execution timed out after 5s", "actual_output": None},
    ]
    summary = _format_test_summary(results)
    assert "TIMEOUT" in summary
    assert "5000ms" in summary
    assert "Total: 0/1 passed" in summary


def test_format_fail_labeled_correctly():
    results = [
        {"passed": False, "execution_time_ms": 12.0,
         "error_message": None, "actual_output": 5},
    ]
    summary = _format_test_summary(results)
    assert "FAIL" in summary
    assert "Total: 0/1 passed" in summary


def test_format_error_labeled_correctly():
    results = [
        {"passed": False, "execution_time_ms": 3.0,
         "error_message": "IndexError: list index out of range", "actual_output": None},
    ]
    summary = _format_test_summary(results)
    assert "ERROR" in summary
    assert "IndexError" in summary


def test_format_mixed_results():
    results = [
        {"passed": True,  "execution_time_ms": 5.0,    "error_message": None,                          "actual_output": 1},
        {"passed": False, "execution_time_ms": 5000.0, "error_message": "Execution timed out after 5s","actual_output": None},
        {"passed": False, "execution_time_ms": 7.0,    "error_message": None,                          "actual_output": 99},
    ]
    summary = _format_test_summary(results)
    assert "PASS" in summary
    assert "TIMEOUT" in summary
    assert "FAIL" in summary
    assert "Total: 1/3 passed" in summary
