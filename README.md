# 🧠 LeetCopilot — Multi-Agent Algorithm Mentor

An AI-powered LeetCode mentor built with **Streamlit** and **LangGraph**. It orchestrates specialized agents to solve problems, teach algorithmic intuition, audit code complexity, and execute sandboxed test simulations across multiple programming languages.

---

## 🌟 Key Features

- **🚀 Full Solution Mode**: Generates Naive & Optimal solutions with deep Big-O complexity breakdowns and edge-case testing.
- **🧠 Study Mode**: Socratic hint-based workflow focusing on algorithmic intuition without spoiling the full solution.
- **🔍 Review Mode**: Analyzes your own code, audits constants/bottlenecks, and gives an actionable optimization roadmap.
- **🌍 Polyglot Support**: Python, Java, JavaScript, and C++.
- **⚡ Multi-Model Engine**: Seamless support for Anthropic Claude, OpenAI, and Google Gemini models.

---

## 🏗️ Multi-Agent Architecture
```text
Supervisor ──┬──▶ Study Agent ──────────────────────────────▶ Documenter ──▶ Output
             ├──▶ Code Reviewer ────────────────────────────▶ Documenter ──▶ Output
             └──▶ Solver ──▶ Performance ──▶ Tester ────────▶ Documenter ──▶ Output
```

| Agent | Responsibility |
|---|---|
| **Supervisor** | Validates input format, determines strategy, and routes execution |
| **Study Agent** | Provides Socratic guidance, mental models, and progressive hints |
| **Solver** | Implements both Naive and Optimal algorithm implementations |
| **Performance** | Formal Big-O analysis ($T(N)$, $S(N)$) and space-time trade-offs |
| **Tester** | Simulates and verifies test cases (Base, Edge, Boundary) |
| **Code Reviewer** | Audits user code for algorithmic bottlenecks and clean code principles |
| **Documenter** | Compiles unified, structured Markdown reports |

---

## 📦 Requirements & Installation

- Python 3.10+
- API Key for **Anthropic**, **OpenAI**, or **Google GenAI**

```bash
# Clone the repository
git clone [https://github.com/idosi/LeetCopilot.git](https://github.com/idosi/LeetCopilot.git)
cd LeetCopilot
```

# Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

# Install dependencies
```bash
pip install -r requirements.txt
```

⚙️ Configuration
Create a .env file in the root directory:

**Dependencies** (`requirements.txt`):

| Package | Version |
|---|---|
| streamlit | 1.50.0 |
| langchain-anthropic | 0.3.22 |
| langgraph | 0.6.11 |
| python-dotenv | 1.2.1 |
| langchain-google-genai | 2.0.0 |
| pydantic | 2.13.4 |


# At least one LLM key is required:
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_gemini_api_key

# Optional settings
LLM_TIMEOUT_SECONDS=45
SANDBOX_TIMEOUT_SECONDS=5

🚀 Running the Web App
```bash
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

🔌 Integration with Claude Desktop (MCP)
To connect LeetCopilot directly as a tool inside Claude Desktop:

1. Open your Claude Desktop configuration file:
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the following configuration under the `mcpServers` key (make sure to replace `/path/to/your/project` with your actual absolute project path, and update your API key):

```json
{
  "mcpServers": {
    "leetcode_copilot": {
      "command": "bash",
      "args": [
        "-c",
        "PYTHONPATH=/path/to/LeetCopilot /path/to/LeetCopilot/.venv/bin/python /path/to/LeetCopilot/src/mcp_server.py"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "your_sk_ant_key"
      }
    }
  }
}
