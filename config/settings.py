import os
from typing import Any, Type, Optional, Union
from dotenv import load_dotenv

load_dotenv()


class EnvVar:
    """
    Descriptor for environment variable loading.
    Handles fetching, type casting, validation, and caching.
    """

    def __init__(
        self,
        key: str,
        default: Any = None,
        cast: Type[Any] = str,
        required: bool = False,
    ):
        self.key = key
        self.default = default
        self.cast = cast
        self.required = required
        self._value: Any = None
        self._loaded = False

    def __get__(self, instance: Any, owner: Any) -> Any:
        # Lazy load validation
        if not self._loaded:
            val = os.getenv(self.key)

            if val is None:
                if self.required:
                    raise ValueError(
                        f"Missing required environment variable: {self.key}"
                    )
                self._value = self.default
            else:
                try:
                    if self.cast is bool:
                        self._value = val.lower() in ("true", "1", "yes", "on")
                    else:
                        self._value = self.cast(val)
                except ValueError as e:
                    raise ValueError(
                        f"Config '{self.key}' must be {self.cast.__name__}, got '{val}'"
                    ) from e

            self._loaded = True

        return self._value


class Config:
    """Centralized configuration management"""

    BOT_TOKEN = EnvVar("BOT_TOKEN", required=True)

    # Default Provider Selection
    PROVIDER_NAME = EnvVar("PROVIDER_NAME", default="perplexity")

    # Perplexity Config
    PERPLEXITY_COOKIES = EnvVar("PERPLEXITY_COOKIES")
    PERPLEXITY_MODEL = EnvVar("PERPLEXITY_MODEL")

    # Groq Config
    GROQ_API_KEY = EnvVar("GROQ_API_KEY")
    GROQ_MODEL = EnvVar("GROQ_MODEL", default="llama-3.3-70b-versatile")

    # Web Server & Tunneling
    WEB_HOST = EnvVar("WEB_HOST", default="127.0.0.1")
    WEB_PORT = EnvVar("WEB_PORT", default=5003, cast=int)
    NGROK_AUTH_TOKEN = EnvVar("NGROK_AUTH_TOKEN", default="")
    NGROK_DOMAIN = EnvVar("NGROK_DOMAIN")

    DATABASE_PATH = EnvVar("DATABASE_PATH", default="var/bot.sqlite3")
    LOG_LEVEL = EnvVar("LOG_LEVEL", default="INFO")

    MAX_MESSAGE_LENGTH = EnvVar("MAX_MESSAGE_LENGTH", default=4096, cast=int)
    SAFE_MESSAGE_LENGTH = EnvVar("SAFE_MESSAGE_LENGTH", default=4000, cast=int)

    MIN_UPDATE_INTERVAL = EnvVar("MIN_UPDATE_INTERVAL", default=3.0, cast=float)
    MAX_UPDATE_INTERVAL = EnvVar("MAX_UPDATE_INTERVAL", default=10.0, cast=float)
    INITIAL_RETRY_DELAY = EnvVar("INITIAL_RETRY_DELAY", default=1.0, cast=float)
    MAX_RETRY_DELAY = EnvVar("MAX_RETRY_DELAY", default=60.0, cast=float)
    MAX_RETRIES = EnvVar("MAX_RETRIES", default=5, cast=int)

    @classmethod
    def validate(cls) -> None:
        _ = cls.BOT_TOKEN

        provider = cls.PROVIDER_NAME
        if provider == "perplexity" and not cls.PERPLEXITY_COOKIES:
            raise ValueError(
                "PERPLEXITY_COOKIES required when PROVIDER_NAME is 'perplexity'"
            )

        if provider == "groq" and not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY required when PROVIDER_NAME is 'groq'")
