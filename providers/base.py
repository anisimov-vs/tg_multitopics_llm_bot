from storage import ProviderType, ProviderCapability, Conversation, DatabaseManager
from config import logger
from typing import (
    AsyncIterator,
    List,
    Protocol,
    runtime_checkable,
    Optional,
    TypedDict,
    Dict,
    Type,
    Any,
)
from abc import ABC, abstractmethod, ABCMeta
from aiogram.types import InlineKeyboardButton


class AttachmentInput(TypedDict):
    """
    Provider-agnostic attachment description.

    filename:     original file name (for display / MIME inference)
    content_type: MIME type (e.g. 'image/png', 'application/pdf')
    data:         raw bytes of the file
    """

    filename: str
    content_type: str
    data: bytes


class LLMProviderMeta(ABCMeta):
    """
    Metaclass for LLM Providers.
    1. Auto-registers classes into a central registry.
    2. Validates that required class attributes exist at definition time.
    """

    registry: Dict[str, Type["BaseLLMProvider"]] = {}

    def __new__(
        mcs, name: str, bases: tuple[Type[Any], ...], namespace: dict[str, Any]
    ) -> Type[Any]:
        cls: Type[Any] = super().__new__(mcs, name, bases, namespace)

        if name in ("BaseLLMProvider", "OpenAICompatibleProvider"):
            return cls

        provider_name = namespace.get("PROVIDER_NAME")

        if provider_name:
            if provider_name in mcs.registry:
                logger.warning(
                    f"Duplicate provider name detected: {provider_name}. Overwriting."
                )

            mcs.registry[provider_name] = cls
            logger.debug(
                f"Auto-registered provider '{provider_name}' from class '{name}'"
            )

        return cls


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining LLM provider interface"""

    @property
    def provider_type(self) -> ProviderType:
        ...

    @property
    def capabilities(self) -> List[ProviderCapability]:
        ...

    def generate_response(
        self,
        conversation: Conversation,
        stream: bool = True,
        attachments: Optional[List[AttachmentInput]] = None,
    ) -> AsyncIterator[str]:
        ...

    def create_extra_buttons(
        self, conversation: Conversation
    ) -> List[InlineKeyboardButton]:
        ...

    def get_available_models(self) -> List[str]:
        ...


class BaseLLMProvider(metaclass=LLMProviderMeta):
    """Abstract base class for LLM providers"""

    PROVIDER_NAME: str

    def __init__(self, storage: DatabaseManager):
        self.storage = storage

    @classmethod
    @abstractmethod
    def create_config(cls, config: Any) -> Optional[Dict[str, Any]]:
        """
        Check global config and return constructor args if enabled.
        Return None if dependencies (keys/cookies) are missing.
        """
        pass

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        pass

    @property
    def capabilities(self) -> List[ProviderCapability]:
        return [ProviderCapability.STREAMING]

    @abstractmethod
    def generate_response(
        self,
        conversation: Conversation,
        stream: bool = True,
        attachments: Optional[List[AttachmentInput]] = None,
    ) -> AsyncIterator[str]:
        """Generate a response for the given conversation."""
        pass

    def create_extra_buttons(
        self, conversation: Conversation
    ) -> List[InlineKeyboardButton]:
        return []

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return list of available models for this provider"""
        pass
