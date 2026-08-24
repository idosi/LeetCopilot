from src.core.graph import route_by_mode, route_after_tester


def test_route_by_mode_review_goes_to_tester():
    state = {"mode": "review"}
    assert route_by_mode(state) == "tester"


def test_route_by_mode_full_goes_to_solver():
    state = {"mode": "full"}
    assert route_by_mode(state) == "solver"


def test_route_by_mode_study_goes_to_documenter():
    state = {"mode": "study"}
    assert route_by_mode(state) == "documenter"


def test_route_by_mode_default_goes_to_solver():
    state = {}
    assert route_by_mode(state) == "solver"


def test_route_after_tester_review_goes_to_code_review():
    state = {"mode": "review"}
    assert route_after_tester(state) == "code_review"


def test_route_after_tester_full_goes_to_documenter():
    state = {"mode": "full"}
    assert route_after_tester(state) == "documenter"


def test_route_after_tester_default_goes_to_documenter():
    state = {}
    assert route_after_tester(state) == "documenter"


def test_route_after_tester_study_goes_to_documenter():
    state = {"mode": "study"}
    assert route_after_tester(state) == "documenter"
