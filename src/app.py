import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from langchain_community.callbacks import get_openai_callback

from src.core.graph import build_graph
from src.core.llm import calculate_cost, SUPPORTED_MODELS
from src.schemas.state import LeetCodeSolverState

st.set_page_config(
    page_title="LeetCopilot",
    page_icon="🧠",
    layout="wide",
)

if "is_running" not in st.session_state:
    st.session_state.is_running = False

_ICON_MAP = {"started": "⏳", "completed": "✅", "failed": "❌"}

_SUPPORTED_LANGUAGES = ["Python", "Java", "JavaScript", "C++"]


def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

/* ── Tokens ─────────────────────────────────── */
:root {
    --bg:            #f1f5f9;
    --bg-alt:        #f8fafc;
    --card:          #ffffff;
    --border:        #e2e8f0;
    --border-focus:  #4f46e5;
    --accent:        #4f46e5;
    --accent-hover:  #4338ca;
    --accent-soft:   rgba(79, 70, 229, 0.08);
    --accent-blue:   #3b82f6;
    --accent-blue-s: rgba(59, 130, 246, 0.10);
    --success:       #10b981;
    --success-soft:  rgba(16, 185, 129, 0.10);
    --warning:       #f59e0b;
    --warning-soft:  rgba(245, 158, 11, 0.10);
    --danger:        #ef4444;
    --danger-soft:   rgba(239, 68, 68, 0.10);
    --fg:            #0f172a;
    --fg-muted:      #64748b;
    --fg-subtle:     #94a3b8;
    --shadow-sm:     0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:     0 4px 16px rgba(0,0,0,0.07), 0 2px 6px rgba(0,0,0,0.04);
    --shadow-lg:     0 10px 30px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05);
    --radius-sm:     10px;
    --radius-md:     16px;
    --radius-lg:     20px;
}

/* ── Global ──────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--fg) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    color: var(--fg) !important;
    z-index: 100 !important;
    height: 0px !important;
}

.stApp {
    margin-top: 0 !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
}

[data-testid="stAppViewContainer"] > section:nth-child(2) {
    padding-top: 0rem !important;
}

[data-testid="stMain"],
.main .block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    margin-top: 0 !important;
}

/* ── Sidebar ─────────────────────────────────── */
[data-testid="stSidebar"] {
    top: 0 !important;
    height: 100vh !important;
    background: var(--bg) !important;
    border-right: 1px solid #cbd5e1 !important;
}

[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarUserContent"] {
    padding-top: 1.25rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

[data-testid="stSidebarCollapseButton"] {
    position: absolute !important;
    top: 0.5rem !important;
    right: 0.5rem !important;
    z-index: 1000 !important;
}

[data-testid="stSidebarCollapseButton"] button,
button[kind="header"] {
    color: #4f46e5 !important;
}

[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
#MainMenu,
footer {
    display: none !important;
}

/* ── Typography ──────────────────────────────── */
h1, h2, h3, h4 {
    color: var(--fg) !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.025em !important;
}

/* ── Tabs ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 4px !important;
    gap: 2px !important;
    width: fit-content !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 7px !important;
    color: var(--fg-muted) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: all 0.18s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--fg) !important;
    background: rgba(0,0,0,0.04) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent) !important;
    box-shadow: var(--shadow-sm) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 0 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; height: 0 !important; }
.stTabs > div > div:first-child        { border-bottom: none !important; }

/* ── Inputs ──────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}
                
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
}
                
.stTextArea > div > div > textarea {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
}
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: transparent !important;
    border: none !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label {
    color: var(--fg) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}

/* ── Buttons ─────────────────────────────────── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--accent) 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #ffffff !important;
    cursor: pointer !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.75rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.30) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, var(--accent-hover) 0%, #4f46e5 100%) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.40) !important;
    transform: translateY(-1px) !important;
}

/* ── Code blocks ─────────────────────────────── */
div[data-testid="stCodeBlock"] {
    background-color: #1e1e2e !important;
    border: 1px solid #2d2d3f !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm) !important;
}

div[data-testid="stCodeBlock"]:hover,
div[data-testid="stCodeBlock"] pre,
div[data-testid="stCodeBlock"] code {
    background-color: #1e1e2e !important;
}

.stMarkdown code {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    border-radius: 5px !important;
    padding: 2px 7px !important;
    font-size: 0.82em !important;
}

/* ── Utility classes ─────────────────────────── */
.lc-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 1.25rem;
}
.lc-panel-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.lc-panel-header-icon {
    width: 38px; height: 38px; flex-shrink: 0;
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem;
    box-shadow: 0 4px 12px rgba(79,70,229,0.30);
}
.lc-panel-header-text-title {
    font-size: 0.975rem; font-weight: 700;
    color: #0f172a; letter-spacing: -0.01em;
}
.lc-panel-header-text-sub {
    font-size: 0.75rem; color: #6366f1; font-weight: 500; margin-top: 1px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-indigo {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid rgba(79,70,229,0.2);
}
.badge-blue {
    background: var(--accent-blue-s);
    color: #2563eb;
    border: 1px solid rgba(59,130,246,0.2);
}
.badge-green {
    background: var(--success-soft);
    color: #059669;
    border: 1px solid rgba(16,185,129,0.2);
}
.badge-amber {
    background: var(--warning-soft);
    color: #b45309;
    border: 1px solid rgba(245,158,11,0.2);
}
.lc-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--fg-subtle);
    margin-bottom: 0.35rem;
    display: block;
}
.callout {
    border-radius: var(--radius-sm);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}
.callout-success {
    background: var(--success-soft);
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 3px solid var(--success);
    color: #064e3b;
}
.callout-danger {
    background: var(--danger-soft);
    border: 1px solid rgba(239,68,68,0.2);
    border-left: 3px solid var(--danger);
    color: #7f1d1d;
}
.agent-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
}
.agent-row:last-child { border-bottom: none; }
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-blue) 100%);
    border: none;
    border-radius: 4px;
    margin: 1.5rem 0;
    opacity: 0.25;
}
</style>
""", unsafe_allow_html=True)


def _build_initial_state(
    title: str,
    description: str,
    constraints: str,
    language: str,
    mode: str = "full",
    user_code: str = "",
    model_name: str = "claude-3-5-sonnet-20241022",
) -> LeetCodeSolverState:
    return LeetCodeSolverState(
        problem_title=title,
        problem_description=description,
        problem_constraints=constraints,
        language=language,
        mode=mode,
        model_name=model_name,
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        estimated_cost_usd=0.0,
        study_output=None,
        naive_solution=None,
        optimal_solution=None,
        naive_complexity=None,
        optimal_complexity=None,
        generated_test_cases=[],
        test_results=[],
        all_tests_passed=False,
        markdown_report=None,
        current_node="",
        supervisor_routing="",
        error_logs=[],
        agent_logs=[],
        execution_start_time=datetime.now(timezone.utc).isoformat(),
        execution_end_time=None,
        total_agents_run=0,
        graph_status="running",
        user_code=user_code or None,
        user_code_review=None,
        user_code_test_results=[],
    )


def _render_sidebar_guide():
    st.sidebar.markdown("""
    <div style="background:var(--bg,#f1f5f9);border-bottom:1px solid #e2e8f0;padding:0.2rem 1rem 0.85rem;margin-top:-0.25rem;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:38px;height:38px;background:linear-gradient(135deg,#4f46e5,#6366f1);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.15rem;box-shadow:0 4px 10px rgba(79,70,229,0.3);">🧠</div>
            <div>
                <div style="font-size:1.05rem;font-weight:800;background:linear-gradient(135deg,#4f46e5,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.02em;line-height:1.2;">LeetCopilot</div>
                <div style="font-size:0.68rem;color:#64748b;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;margin-top:1px;">AI-Powered Mentor</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div style="padding:0.75rem 1rem 0;">', unsafe_allow_html=True)
        st.markdown('<span class="lc-label">MODEL CONFIGURATION</span>', unsafe_allow_html=True)
        st.selectbox(
            "Select LLM Engine",
            SUPPORTED_MODELS,
            index=0,
            key="global_selected_model",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown("""
<div style="padding:1rem;">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:0.75rem;">How It Works</div>
    <div style="display:flex;flex-direction:column;gap:10px;">
        <div style="display:flex;gap:10px;align-items:flex-start;">
            <div style="width:28px;height:28px;background:#eef2ff;border:1px solid rgba(79,70,229,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;flex-shrink:0;">🔀</div>
            <div><div style="color:#0f172a;font-weight:600;font-size:0.82rem;">Supervisor</div><div style="color:#64748b;font-size:0.76rem;margin-top:1px;">Validates &amp; routes the workflow</div></div>
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start;">
            <div style="width:28px;height:28px;background:#eef2ff;border:1px solid rgba(79,70,229,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;flex-shrink:0;">📚</div>
            <div><div style="color:#0f172a;font-weight:600;font-size:0.82rem;">Study Agent</div><div style="color:#64748b;font-size:0.76rem;margin-top:1px;">Provides Socratic intuition &amp; hints</div></div>
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start;">
            <div style="width:28px;height:28px;background:#eef2ff;border:1px solid rgba(79,70,229,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;flex-shrink:0;">⚡</div>
            <div><div style="color:#0f172a;font-weight:600;font-size:0.82rem;">Solver</div><div style="color:#64748b;font-size:0.76rem;margin-top:1px;">Generates naive + optimal solutions</div></div>
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start;">
            <div style="width:28px;height:28px;background:#eef2ff;border:1px solid rgba(79,70,229,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;flex-shrink:0;">📊</div>
            <div><div style="color:#0f172a;font-weight:600;font-size:0.82rem;">Performance</div><div style="color:#64748b;font-size:0.76rem;margin-top:1px;">Analyzes Big O complexity</div></div>
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start;">
            <div style="width:28px;height:28px;background:#eef2ff;border:1px solid rgba(79,70,229,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;flex-shrink:0;">🧪</div>
            <div><div style="color:#0f172a;font-weight:600;font-size:0.82rem;">Tester</div><div style="color:#64748b;font-size:0.76rem;margin-top:1px;">Runs solutions in a sandbox</div></div>
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start;">
            <div style="width:28px;height:28px;background:#eef2ff;border:1px solid rgba(79,70,229,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;flex-shrink:0;">🔍</div>
            <div><div style="color:#0f172a;font-weight:600;font-size:0.82rem;">Reviewer</div><div style="color:#64748b;font-size:0.76rem;margin-top:1px;">Audits clean code &amp; constant factors</div></div>
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start;">
            <div style="width:28px;height:28px;background:#eef2ff;border:1px solid rgba(79,70,229,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;flex-shrink:0;">📝</div>
            <div><div style="color:#0f172a;font-weight:600;font-size:0.82rem;">Documenter</div><div style="color:#64748b;font-size:0.76rem;margin-top:1px;">Compiles the final report</div></div>
        </div>
    </div>
    <div style="height:1px;background:#e2e8f0;margin:1rem 0;"></div>
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:0.5rem;">Tips</div>
    <div style="color:#64748b;font-size:0.8rem;line-height:1.7;">
        • Paste the <span style="color:#4f46e5;font-weight:600;">full problem statement</span> with examples<br/>
        • Constraints are optional but improve quality<br/>
        • Runs typically take <span style="color:#10b981;font-weight:600;">30–90 seconds</span>
    </div>
</div>
""", unsafe_allow_html=True)


def _render_sidebar_status(final_state: LeetCodeSolverState):
    graph_status = final_state.get("graph_status", "unknown")
    agents_run = final_state.get("total_agents_run", 0)
    agent_logs = final_state.get("agent_logs", [])
    error_logs = final_state.get("error_logs", [])

    model_used = final_state.get("model_name", "gpt-4o-mini")
    total_tokens = final_state.get("total_tokens", 0) or 0
    cost_usd = final_state.get("estimated_cost_usd", 0.0) or 0.0

    is_ok = graph_status == "completed"
    status_icon = "✓" if is_ok else "✗"
    status_text = "COMPLETED" if is_ok else graph_status.upper()

    st.sidebar.markdown("""
<div style="background:var(--bg,#f1f5f9);border-bottom:1px solid #e2e8f0;padding-bottom:0.75rem;margin-bottom:0.5rem;">
    <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:38px;height:38px;background:linear-gradient(135deg,#4f46e5,#6366f1);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.15rem;box-shadow:0 4px 10px rgba(79,70,229,0.3);">🧠</div>
        <div>
            <div style="font-size:1.05rem;font-weight:800;background:linear-gradient(135deg,#4f46e5,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.02em;line-height:1.2;">LeetCopilot</div>
            <div style="font-size:0.68rem;color:#64748b;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;margin-top:1px;">AI-Powered Mentor</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div style="padding:0.5rem 0 0;"><span class="lc-label">MODEL CONFIGURATION</span></div>', unsafe_allow_html=True)
        st.selectbox(
            "Select LLM Engine",
            SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(model_used) if model_used in SUPPORTED_MODELS else 0,
            key="global_selected_model",
            label_visibility="collapsed"
        )

    summary_html = f"""<div style="padding:0.75rem 0 0.5rem;">
<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:0.85rem;">Run Summary</div>

<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:0.65rem;">
    <span style="color:#64748b;font-size:0.82rem;font-weight:500;white-space:nowrap;">Status</span>
    <span style="background:rgba(79,70,229,0.08);color:#4f46e5;font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:6px;letter-spacing:0.03em;">{status_icon} {status_text}</span>
</div>

<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:0.65rem;">
    <span style="color:#64748b;font-size:0.82rem;font-weight:500;white-space:nowrap;">Model</span>
    <span title="{model_used}" style="background:rgba(79,70,229,0.08);color:#4f46e5;font-weight:600;padding:3px 10px;border-radius:6px;font-size:0.72rem;font-family:'Fira Code',monospace;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:right;">{model_used}</span>
</div>

<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:0.65rem;">
    <span style="color:#64748b;font-size:0.82rem;font-weight:500;white-space:nowrap;">Total Tokens</span>
    <span style="background:rgba(79,70,229,0.08);color:#4f46e5;font-weight:700;padding:3px 10px;border-radius:6px;font-size:0.75rem;font-family:'Fira Code',monospace;">{total_tokens:,}</span>
</div>

<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:0.65rem;">
    <span style="color:#64748b;font-size:0.82rem;font-weight:500;white-space:nowrap;">Run Cost</span>
    <span style="background:rgba(79,70,229,0.08);color:#4f46e5;font-weight:700;padding:3px 10px;border-radius:6px;font-size:0.75rem;font-family:'Fira Code',monospace;">${cost_usd:.5f}</span>
</div>

<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
    <span style="color:#64748b;font-size:0.82rem;font-weight:500;white-space:nowrap;">Agents Run</span>
    <span style="background:rgba(79,70,229,0.08);color:#4f46e5;font-weight:700;padding:3px 10px;border-radius:6px;font-size:0.75rem;font-family:'Fira Code',monospace;">{agents_run}</span>
</div>
</div>"""
    st.sidebar.markdown(summary_html, unsafe_allow_html=True)

    if agent_logs:
        rows_html = "".join([
            f'<div class="agent-row"><span style="font-size:0.85rem;">{_ICON_MAP.get(log.get("status", ""), "•")}</span><div><span style="color:#0f172a;font-weight:600;font-size:0.8rem;">{log.get("agent_name", "unknown")}</span><span style="color:#64748b;font-size:0.75rem;"> — {log.get("action", "")}</span></div></div>'
            for log in agent_logs
        ])
        timeline_html = f"""<div style="padding:0.5rem 0 1rem;">
<div style="height:1px;background:#e2e8f0;margin-bottom:0.75rem;"></div>
<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:0.5rem;">Agent Timeline</div>
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:0.5rem 0.75rem;">
{rows_html}
</div>
</div>"""
        st.sidebar.markdown(timeline_html, unsafe_allow_html=True)

    if error_logs:
        st.sidebar.markdown('<div style="padding:0.5rem 0 0;font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#dc2626;margin-bottom:0.4rem;">Errors</div>', unsafe_allow_html=True)
        for err in error_logs:
            st.sidebar.error(err)


def _render_problem_inputs(prefix: str = "default"):
    problem_title = st.text_input(
        "Problem Title",
        placeholder="e.g. Two Sum",
        key=f"{prefix}_problem_title"
    )
    problem_description = st.text_area(
        "Problem Description",
        height=200,
        placeholder="Paste the full LeetCode problem description here...",
        key=f"{prefix}_problem_description"
    )
    problem_constraints = st.text_area(
        "Constraints (optional)",
        height=100,
        placeholder="e.g.\n- 2 <= nums.length <= 10^4\n- Only one valid answer exists.",
        key=f"{prefix}_problem_constraints"
    )
    selected_language = st.selectbox(
        "Language",
        _SUPPORTED_LANGUAGES,
        index=0,
        key=f"{prefix}_selected_language"
    )
    return problem_title, problem_description, problem_constraints, selected_language


def _render_problem_tab(final_state: LeetCodeSolverState, is_study: bool = False):
    markdown_report = final_state.get("markdown_report") or final_state.get("report_markdown")

    if markdown_report:
        st.markdown(markdown_report)
    else:
        title = final_state.get("problem_title", "Untitled")
        description = final_state.get("problem_description", "")
        constraints = final_state.get("problem_constraints", "")

        lines = [
            f"# Problem: {title}",
            "",
            "## Problem Statement",
            "",
            description,
        ]
        if constraints:
            lines.extend(["", f"**Constraints:**\n{constraints}"])
        st.markdown("\n".join(lines))


def _render_solutions_tab(final_state: LeetCodeSolverState):
    naive = final_state.get("naive_solution")
    optimal = final_state.get("optimal_solution")

    if not naive and not optimal:
        st.info("No solutions available.")
        return

    col_naive, col_optimal = st.columns(2)

    with col_naive:
        st.markdown('<span class="badge badge-amber">Naive Approach</span>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        if naive:
            if naive.get("approach"):
                st.markdown(f'<span class="lc-label">Approach</span><div style="color:#334155;font-size:0.9rem;font-weight:600;margin-bottom:0.6rem;">{naive["approach"]}</div>', unsafe_allow_html=True)
            if naive.get("description"):
                st.markdown(f'<div style="color:#64748b;font-size:0.875rem;line-height:1.65;margin-bottom:1rem;">{naive["description"]}</div>', unsafe_allow_html=True)
            st.code(naive.get("code", ""), language=naive.get("language", "python"))
        else:
            st.info("No naive solution generated.")

    with col_optimal:
        st.markdown('<span class="badge badge-green">Optimal Approach</span>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        if optimal:
            if optimal.get("approach"):
                st.markdown(f'<span class="lc-label">Approach</span><div style="color:#334155;font-size:0.9rem;font-weight:600;margin-bottom:0.6rem;">{optimal["approach"]}</div>', unsafe_allow_html=True)
            if optimal.get("description"):
                st.markdown(f'<div style="color:#64748b;font-size:0.875rem;line-height:1.65;margin-bottom:1rem;">{optimal["description"]}</div>', unsafe_allow_html=True)
            st.code(optimal.get("code", ""), language=optimal.get("language", "python"))
        else:
            st.info("No optimal solution generated.")


def _render_tests_tab(final_state: LeetCodeSolverState):
    test_results = final_state.get("test_results", [])
    user_test_results = final_state.get("user_code_test_results", [])
    generated_cases = final_state.get("generated_test_cases", [])
    
    active_results = user_test_results if user_test_results else test_results
    all_passed = final_state.get("all_tests_passed", False)

    if not active_results and not generated_cases:
        st.info("No test results available.")
        return

    def _val(obj, key: str, default: str = "—") -> str:
        if not obj:
            return default
        if isinstance(obj, dict):
            val = obj.get(key)
        else:
            val = getattr(obj, key, default)
        return str(val) if val is not None and str(val).strip() != "" else default

    total = len(active_results)
    passed = sum(1 for r in active_results if str(_val(r, "passed")).lower() in ("true", "1"))
    failed = total - passed

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tests", total)
    col2.metric("Passed", passed)
    col3.metric("Failed", failed)

    st.markdown("<br/>", unsafe_allow_html=True)

    if all_passed or (total > 0 and passed == total):
        st.markdown("""
<div class="callout callout-success">
    <strong style="font-size:0.9rem;">✓ All tests passed — solution is correct</strong>
</div>""", unsafe_allow_html=True)
    elif total > 0:
        st.markdown(f"""
<div class="callout callout-danger">
    <strong style="font-size:0.9rem;">✗ {failed} test(s) failed — review the cases below</strong>
</div>""", unsafe_allow_html=True)

    if active_results:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<span class="lc-label" style="margin-bottom:0.75rem;display:block;">Test Matrix</span>', unsafe_allow_html=True)
        rows = []
        for i, result in enumerate(active_results):
            case = generated_cases[i] if i < len(generated_cases) else {}
            
            desc = _val(result, "description")
            if desc == "—":
                desc = _val(case, "description", "—")

            case_type = _val(result, "case_type")
            if case_type == "—":
                case_type = _val(case, "case_type", "base")

            expected = _val(result, "expected")
            if expected == "—":
                expected = _val(case, "expected_output", "—")

            ms_val = _val(result, "execution_time_ms", "0.0")
            try:
                ms_str = f"{float(ms_val):.1f}"
            except Exception:
                ms_str = str(ms_val)

            is_pass = str(_val(result, "passed")).lower() in ("true", "1")

            rows.append({
                "ID": _val(result, "test_case_id", str(i)),
                "Type": case_type,
                "Description": desc,
                "Expected": expected,
                "Actual": _val(result, "actual_output", "—"),
                "Time (ms)": ms_str,
                "Status": "✅ PASS" if is_pass else "❌ FAIL",
                "Error": _val(result, "error_message", ""),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ── Layout ───────────────────────────────────────────────────────────────────

_inject_css()

if "final_state" not in st.session_state:
    _render_sidebar_guide()
else:
    _render_sidebar_status(st.session_state.final_state)

# Hero header
st.markdown("""
<div style="padding:0.2rem 0 1rem;">
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:0.75rem;">
        <div style="width:48px;height:48px;background:linear-gradient(135deg,#4f46e5,#6366f1);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;box-shadow:0 8px 20px rgba(79,70,229,0.3);">🧠</div>
        <div>
            <h1 style="margin:0;font-size:clamp(1.6rem,3.5vw,2.25rem);font-weight:800;color:#0f172a;letter-spacing:-0.03em;line-height:1.1;">LeetCopilot</h1>
        </div>
    </div>
    <p style="color:#64748b;font-size:1rem;margin:0;line-height:1.5;max-width:520px;">
        Your AI-powered LeetCode mentor — solve, study, and review algorithm problems with a coordinated agent pipeline.
    </p>
</div>
""", unsafe_allow_html=True)

tab_full, tab_study, tab_review = st.tabs(["🚀 Full Solution", "🧠 Study Mode", "🔍 Review Mode"])

submitted = False
mode = None
is_btn_disabled = st.session_state.is_running

with tab_full:
    with st.form("form_full"):
        st.markdown("""
        <div class="lc-panel-header">
          <div class="lc-panel-header-icon">🚀</div>
          <div>
            <div class="lc-panel-header-text-title">Full Solution</div>
            <div class="lc-panel-header-text-sub">Generate naive + optimal solutions with automated tests</div>
          </div>
          <span class="badge badge-indigo" style="margin-left:auto;">Full Mode</span>
        </div>
        """, unsafe_allow_html=True)
        _render_problem_inputs("full")
        if st.form_submit_button("⚡ Solve Problem", type="primary", use_container_width=True, disabled=is_btn_disabled):
            submitted = True
            mode = "full"

with tab_study:
    with st.form("form_study"):
        st.markdown("""
        <div class="lc-panel-header">
          <div class="lc-panel-header-icon">🧠</div>
          <div>
            <div class="lc-panel-header-text-title">Study Mode</div>
            <div class="lc-panel-header-text-sub">Get hints and a guided approach without a full solution</div>
          </div>
          <span class="badge badge-blue" style="margin-left:auto;">Hints Only</span>
        </div>
        """, unsafe_allow_html=True)
        _render_problem_inputs("study")
        if st.form_submit_button("🧠 Get Hints", type="primary", use_container_width=True, disabled=is_btn_disabled):
            submitted = True
            mode = "study"

with tab_review:
    with st.form("form_review"):
        st.markdown("""
        <div class="lc-panel-header">
          <div class="lc-panel-header-icon">🔍</div>
          <div>
            <div class="lc-panel-header-text-title">Review Mode</div>
            <div class="lc-panel-header-text-sub">Analyze your code's complexity and get an optimization roadmap</div>
          </div>
          <span class="badge badge-green" style="margin-left:auto;">Code Review</span>
        </div>
        """, unsafe_allow_html=True)
        _render_problem_inputs("review")
        st.text_area(
            "Paste Your Code Here",
            height=200,
            placeholder="Paste your solution code here for review...",
            key="review_user_code",
        )
        if st.form_submit_button("🔍 Review My Code", type="primary", use_container_width=True, disabled=is_btn_disabled):
            submitted = True
            mode = "review"

if submitted and mode and not st.session_state.is_running:
    raw_desc = st.session_state.get(f"{mode}_problem_description", "")
    raw_title = st.session_state.get(f"{mode}_problem_title", "")
    raw_constraints = st.session_state.get(f"{mode}_problem_constraints", "")
    raw_lang = st.session_state.get(f"{mode}_selected_language", "Python")
    raw_user_code = st.session_state.get("review_user_code", "")
    selected_model = st.session_state.get("global_selected_model", SUPPORTED_MODELS[0])

    if not raw_desc or not raw_desc.strip():
        st.error("Problem description cannot be empty.")
        st.stop()

    if mode == "review" and not raw_user_code.strip():
        st.error("Please paste your code to review.")
        st.stop()

    st.session_state.is_running = True
    spinner_msg = f"Agents running on {selected_model} — this may take a minute..."

    try:
        with st.spinner(spinner_msg):
            graph = build_graph()
            initial_state = _build_initial_state(
                title=raw_title.strip() or "Untitled Problem",
                description=raw_desc.strip(),
                constraints=raw_constraints.strip(),
                language=raw_lang,
                mode=mode,
                user_code=raw_user_code.strip() if mode == "review" else "",
                model_name=selected_model,
            )

            final_output = graph.invoke(initial_state)

            total_prompt_chars = len(raw_desc) + len(raw_user_code) + 1500
            total_completion_chars = len(str(final_output.get("user_code_review") or "")) + len(str(final_output.get("markdown_report") or ""))
            
            est_input_tokens = max(500, total_prompt_chars // 4)
            est_output_tokens = max(300, total_completion_chars // 4)
            
            run_cost = calculate_cost(selected_model, est_input_tokens, est_output_tokens)
            
            final_output["total_tokens"] = est_input_tokens + est_output_tokens
            final_output["prompt_tokens"] = est_input_tokens
            final_output["completion_tokens"] = est_output_tokens
            final_output["estimated_cost_usd"] = run_cost

            st.session_state.final_state = final_output
            st.session_state.run_mode = mode
    finally:
        st.session_state.is_running = False

    st.rerun()

if "final_state" in st.session_state:
    final_state = st.session_state.final_state
    run_mode = st.session_state.get("run_mode", "full")

    st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

    if run_mode == "review":
        tab_report, tab_tests = st.tabs([
            "📋 Review & Optimization Report",
            "🧪 Test Results",
        ])
        with tab_report:
            _render_problem_tab(final_state, is_study=False)
        with tab_tests:
            _render_tests_tab(final_state)

    elif run_mode == "study":
        tab_study_guide, tab_tests = st.tabs([
            "📚 Socratic Study Guide",
            "🧪 Test Cases",
        ])
        with tab_study_guide:
            _render_problem_tab(final_state, is_study=True)
        with tab_tests:
            _render_tests_tab(final_state)

    else:
        tab_report, tab_code, tab_tests = st.tabs([
            "📝 Problem & Full Report",
            "💻 Code Solutions",
            "🧪 Test Results",
        ])
        with tab_report:
            _render_problem_tab(final_state, is_study=False)
        with tab_code:
            _render_solutions_tab(final_state)
        with tab_tests:
            _render_tests_tab(final_state)