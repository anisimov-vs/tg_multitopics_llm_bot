from .openai_compatible import OpenAICompatibleProvider
from storage import DatabaseManager
from typing import List, Optional, Any, Dict


class GroqProvider(OpenAICompatibleProvider):
    """Groq implementation using the OpenAI Compatible provider"""

    PROVIDER_NAME = "groq"

    AVAILABLE_MODELS = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-guard-4-12b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "moonshotai/kimi-k2-instruct-0905",
        "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3-32b",
    ]

    def __init__(
        self, storage: DatabaseManager, api_key: str, model: Optional[str] = None
    ):
        super().__init__(
            storage=storage,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model=model or "llama-3.3-70b-versatile",
            default_system_prompt="You are a helpful, fast, and precise AI assistant powered by Groq.",
        )

    @classmethod
    def create_config(cls, config: Any) -> Optional[Dict[str, Any]]:
        raw_key = config.GROQ_API_KEY
        if not raw_key:
            return None
        key = str(raw_key).strip()
        if (
            not key
            or key.lower() == "none"
            or not key.startswith("gsk_")
            or len(key) != 56
        ):
            return None
        return {
            "api_key": config.GROQ_API_KEY,
            "model": config.GROQ_MODEL or "llama-3.3-70b-versatile",
        }

    def get_available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS.copy()
