import os
from dotenv import load_dotenv

# טעינת המשתנים מקובץ ה-.env לתוך ה-Environment
load_dotenv()

OPENAI_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
LLM_MODEL: str = os.environ["LLM_MODEL"]
LLM_TIMEOUT_SECONDS: int = int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
SANDBOX_TIMEOUT_SECONDS: int = int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", "5"))
SANDBOX_MEMORY_LIMIT_MB: int = int(os.environ.get("SANDBOX_MEMORY_LIMIT_MB", "512"))

MODEL_PROVIDER: str = os.environ.get("MODEL_PROVIDER", "anthropic")
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
