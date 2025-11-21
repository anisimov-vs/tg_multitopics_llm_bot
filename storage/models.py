from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from enum import Enum, auto
import json

from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    ForeignKey,
    JSON,
    DateTime,
    LargeBinary,
    BigInteger,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs


class MessageRole(Enum):
    """Message role enumeration"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ProviderType(Enum):
    """Provider type enumeration"""

    SERVER_HISTORY = "server_history"
    CLIENT_HISTORY = "client_history"


class ProviderCapability(Enum):
    """Provider capability flags"""

    ACCEPTS_IMAGES = auto()
    ACCEPTS_FILES = auto()
    STREAMING = auto()


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all ORM models"""

    pass


class ConversationMessage(Base):
    """Individual conversation message ORM model"""

    __tablename__ = "messages"

    message_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id"), nullable=False, index=True
    )

    _role: Mapped[str] = mapped_column("role", String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    meta_data_json: Mapped[Dict[str, Any]] = mapped_column(
        "meta_data", JSON, default=dict
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    @property
    def role(self) -> MessageRole:
        return MessageRole(self._role)

    @role.setter
    def role(self, value: MessageRole) -> None:
        self._role = value.value

    @property
    def meta_data(self) -> Dict[str, Any]:
        return self.meta_data_json

    @meta_data.setter
    def meta_data(self, value: Dict[str, Any]) -> None:
        self.meta_data_json = value

    def __init__(
        self,
        role: MessageRole,
        content: str,
        timestamp: Optional[datetime] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.meta_data = meta_data or {}


class Conversation(Base):
    """Complete conversation context ORM model"""

    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    topic_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    topic_name: Mapped[str] = mapped_column(String, nullable=False)
    meta_data_json: Mapped[Dict[str, Any]] = mapped_column(
        "meta_data", JSON, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    messages: Mapped[List[ConversationMessage]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.timestamp",
    )

    @property
    def meta_data(self) -> Dict[str, Any]:
        return self.meta_data_json

    @meta_data.setter
    def meta_data(self, value: Dict[str, Any]) -> None:
        self.meta_data_json = value

    def add_message(
        self,
        role: MessageRole,
        content: str,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        msg = ConversationMessage(role=role, content=content, meta_data=meta_data or {})
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)

    def __init__(
        self,
        conversation_id: str,
        chat_id: int,
        topic_id: Optional[int],
        topic_name: str,
        meta_data: Optional[Dict[str, Any]] = None,
    ):
        self.conversation_id = conversation_id
        self.chat_id = chat_id
        self.topic_id = topic_id
        self.topic_name = topic_name
        self.meta_data = meta_data or {}
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class WebPage(Base):
    """Web page meta_data for viewing answers"""

    __tablename__ = "web_pages"

    page_id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id"), nullable=False, index=True
    )
    message_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class Asset(Base):
    """File asset associated with a web page"""

    __tablename__ = "assets"

    asset_id: Mapped[str] = mapped_column(String, primary_key=True)
    page_id: Mapped[str] = mapped_column(
        ForeignKey("web_pages.page_id"), nullable=False, index=True
    )
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    language: Mapped[str] = mapped_column(String, nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    def __init__(
        self,
        asset_id: str,
        file_name: str,
        file_data: bytes,
        language: str,
        size: Optional[int] = None,
    ):
        self.asset_id = asset_id
        self.file_name = file_name
        self.file_data = file_data
        self.language = language
        self.size = size if size is not None else len(file_data)


class UserSetting(Base):
    """User specific settings"""

    __tablename__ = "user_settings"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    settings_json: Mapped[Dict[str, Any]] = mapped_column(
        "settings", JSON, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class KeyboardState(Base):
    """Temporary storage for keyboard states"""

    __tablename__ = "keyboard_states"

    page_id: Mapped[str] = mapped_column(String, primary_key=True)
    keyboard_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
