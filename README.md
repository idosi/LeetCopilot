# Multi-Agent LeetCode Solver

A Streamlit application that uses a LangGraph multi-agent pipeline powered by Claude to solve LeetCode problems. Given a problem statement, it generates naive and optimal Python solutions, analyzes time/space complexity, runs sandboxed tests, and compiles a Markdown report.

## Architecture

```
Supervisor → Solver → Performance → Tester → Documenter
```

| Agent | Responsibility |
|---|---|
| Supervisor | Validates the problem and routes the workflow |
| Solver | Generates naive + optimal Python solutions |
| Performance | Analyzes Big O time & space complexity |
| Tester | Runs solutions in a sandboxed subprocess |
| Documenter | Compiles the final Markdown report |

## Requirements

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)

## Installation

```bash
git clone <repo-url>
cd leetCode-Solver
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):

| Package | Version |
|---|---|
| streamlit | 1.50.0 |
| langchain-anthropic | 0.3.22 |
| langgraph | 0.6.11 |
| python-dotenv | 1.2.1 |
| pydantic | 2.13.4 |

## Configuration

Copy `.env.example` to `.env` and set your values:

```bash
cp .env.example .env
```

`.env` variables:

```dotenv
ANTHROPIC_API_KEY=your_api_key_here
LLM_MODEL=claude-sonnet-4-6
LLM_TIMEOUT_SECONDS=30
SANDBOX_TIMEOUT_SECONDS=5
SANDBOX_MEMORY_LIMIT_MB=512
```

`LLM_MODEL` must be set to `claude-sonnet-4-6`. The other variables have the defaults shown above.

## Running

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

1. Enter the problem title (e.g. `Two Sum`)
2. Paste the full problem description including examples
3. Optionally add constraints
4. Click **Solve Problem** and wait ~30–90 seconds

Results are presented across four tabs: Problem & Report, Code Solutions, Performance, and Test Results.


## 🚀 Integration with Claude Desktop

To use this LeetCode Solver directly from your Claude Desktop application, follow these steps:

1. Open your Claude Desktop configuration file:
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the following configuration under the `mcpServers` key (make sure to replace `/path/to/your/project` with your actual absolute project path, and update your API key):

```json
"leetcode_solver": {
  "command": "bash",
  "args": [
    "-c",
    "PYTHONPATH=/path/to/leetcode-Solver /path/to/leetcode-Solver/.venv/bin/python /path/to/leetcode-Solver/src/mcp_server.py"
  ],
  "env": {
    "ANTHROPIC_API_KEY": "your_sk_ant_key"
  }
}