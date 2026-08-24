import json
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

TIMEOUT_SECONDS = 5

_RUNNER_TEMPLATE = """\
import json, sys
from typing import *
from collections import *
import heapq, bisect, math, itertools, functools

{code}

_input_data = json.loads(sys.stdin.read())
_sol = Solution()
_methods = [m for m in dir(_sol) if not m.startswith('_')]
_result = getattr(_sol, _methods[0])(**_input_data)
print(json.dumps(_result, default=str))
"""


def execute_solution(
    code: str,
    input_data: Dict[str, Any],
    timeout: int = TIMEOUT_SECONDS,
    language: str = "python",
) -> Tuple[Optional[Any], float, Optional[str]]:
    """Execute solution code with input_data via subprocess. Returns (output, time_ms, error)."""
    if language.lower() != "python":
        return None, 0.0, f"Subprocess executor only supports Python; got '{language}'"
    script = _RUNNER_TEMPLATE.format(code=code)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        if proc.returncode != 0:
            return None, elapsed_ms, (proc.stderr.strip() or "Non-zero exit")[:500]
        stdout = proc.stdout.strip()
        if not stdout:
            return None, elapsed_ms, "No output produced"
        return json.loads(stdout), elapsed_ms, None
    except subprocess.TimeoutExpired:
        elapsed_ms = timeout * 1000.0
        return None, elapsed_ms, f"Execution timed out after {timeout}s"
    except json.JSONDecodeError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return None, elapsed_ms, f"Output parse error: {exc}"
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return None, elapsed_ms, str(exc)
