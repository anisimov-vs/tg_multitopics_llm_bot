from .openai_compatible import OpenAICompatibleProvider
from storage import DatabaseManager
from typing import List, Optional


class GroqProvider(OpenAICompatibleProvider):
    """Groq implementation using the OpenAI Compatible provider"""

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

    def get_available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS.copy()
