import os
from typing import Optional
import src.config.settings as settings

# מחירי טוקנים ל-1 מיליון (USD)
MODEL_PRICING = {
    # Google AI Studio (Gemini)
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    
    # Local (Ollama)
    "llama3": {"input": 0.00, "output": 0.00},
}

SUPPORTED_MODELS = list(MODEL_PRICING.keys())


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculates estimated USD cost for a given run based on token consumption."""
    pricing = MODEL_PRICING.get(model_name, {"input": 0.075, "output": 0.30})
    cost = (input_tokens / 1_000_000 * pricing["input"]) + (output_tokens / 1_000_000 * pricing["output"])
    return round(cost, 6)


def get_llm(model_name: Optional[str] = None, temperature: float = 0, api_key: Optional[str] = None):
    """Instantiates the LLM based on dynamic model selection."""
    chosen_model = model_name or getattr(settings, "LLM_MODEL", "gpt-4o-mini")
    model_lower = chosen_model.lower()

    if "gemini" in model_lower:
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or os.environ.get("GOOGLE_API_KEY") or None
        return ChatGoogleGenerativeAI(
            model=chosen_model,
            temperature=temperature,
            google_api_key=key,
        )

    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        from langchain_openai import ChatOpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY") or None
        return ChatOpenAI(
            model=chosen_model,
            temperature=temperature,
            api_key=key,
        )

    elif "llama" in model_lower:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=chosen_model,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=temperature,
        )

    else:
        from langchain_anthropic import ChatAnthropic
        timeout = getattr(settings, "LLM_TIMEOUT_SECONDS", 60)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or None
        return ChatAnthropic(
            model=chosen_model,
            timeout=timeout,
            temperature=temperature,
            api_key=key,
        )