from storage import ProviderType, ProviderCapability, Conversation, DatabaseManager
from config import logger
from typing import (
    AsyncIterator,
    Dict,
    List,
    Protocol,
    Type,
    runtime_checkable,
    Optional,
    TypedDict,
)
from abc import ABC, abstractmethod
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


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, storage: DatabaseManager):
        self.storage = storage

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
