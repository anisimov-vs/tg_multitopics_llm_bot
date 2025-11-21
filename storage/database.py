from config import logger
from .models import (
    Base,
    Conversation,
    ConversationMessage,
    WebPage,
    Asset,
    UserSetting,
    KeyboardState,
    MessageRole,
)
from decorators import db_lock_retry

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, delete, update
from sqlalchemy.orm import selectinload


class DatabaseManager:
    """Async SQLAlchemy database manager"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        db_url = f"sqlite+aiosqlite:///{self.db_path}"
        self.engine = create_async_engine(db_url, echo=False, future=True)

        self.session_factory = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

        # Initialize schema asynchronously (managed by start-up routine or auto-run)
        asyncio.create_task(self._initialize_schema())

    async def _initialize_schema(self) -> None:
        """Initialize database schema using SQLAlchemy meta_data"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @db_lock_retry
    async def save_conversation(self, conversation: Conversation) -> None:
        async with self.session_factory() as session:
            await session.merge(conversation)
            await session.commit()

    @db_lock_retry
    async def load_conversation(
        self, chat_id: int, topic_id: Optional[int]
    ) -> Optional[Conversation]:
        async with self.session_factory() as session:
            stmt = (
                select(Conversation)
                .where(Conversation.chat_id == chat_id)
                .where(Conversation.topic_id == topic_id)
                .order_by(Conversation.updated_at.desc())
                .limit(1)
                .options(selectinload(Conversation.messages))
            )
            result = await session.execute(stmt)
            conversation = result.scalar_one_or_none()
            return conversation

    @db_lock_retry
    async def save_web_page(
        self, page_id: str, conversation_id: str, message_index: int
    ) -> None:
        async with self.session_factory() as session:
            page = WebPage(
                page_id=page_id,
                conversation_id=conversation_id,
                message_index=message_index,
            )
            await session.merge(page)
            await session.commit()

    @db_lock_retry
    async def load_web_page(self, page_id: str) -> Optional[str]:
        """Load content dynamically from messages table via WebPage linkage"""
        async with self.session_factory() as session:
            stmt_page = select(WebPage).where(WebPage.page_id == page_id)
            result_page = await session.execute(stmt_page)
            page = result_page.scalar_one_or_none()

            if not page:
                return None

            stmt_msg = (
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == page.conversation_id)
                .order_by(ConversationMessage.timestamp.asc())
                .limit(1)
                .offset(page.message_index)
            )

            result_msg = await session.execute(stmt_msg)
            msg = result_msg.scalar_one_or_none()

            return msg.content if msg else None

    @db_lock_retry
    async def save_asset(self, page_id: str, asset: Asset) -> None:
        async with self.session_factory() as session:
            asset.page_id = page_id
            await session.merge(asset)
            await session.commit()

    @db_lock_retry
    async def load_assets(self, page_id: str) -> List[Asset]:
        async with self.session_factory() as session:
            stmt = (
                select(Asset).where(Asset.page_id == page_id).order_by(Asset.asset_id)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @db_lock_retry
    async def load_asset(self, page_id: str, asset_id: str) -> Optional[Asset]:
        async with self.session_factory() as session:
            stmt = (
                select(Asset)
                .where(Asset.page_id == page_id)
                .where(Asset.asset_id == asset_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    @db_lock_retry
    async def save_keyboard_state(self, page_id: str, keyboard_json: str) -> None:
        async with self.session_factory() as session:
            state = KeyboardState(page_id=page_id, keyboard_json=keyboard_json)
            await session.merge(state)
            await session.commit()

    @db_lock_retry
    async def load_keyboard_state(self, page_id: str) -> Optional[str]:
        async with self.session_factory() as session:
            result = await session.get(KeyboardState, page_id)
            return result.keyboard_json if result else None

    @db_lock_retry
    async def delete_keyboard_state(self, page_id: str) -> None:
        async with self.session_factory() as session:
            stmt = delete(KeyboardState).where(KeyboardState.page_id == page_id)
            await session.execute(stmt)
            await session.commit()

    @db_lock_retry
    async def get_user_settings(self, user_id: int) -> Dict[str, Any]:
        async with self.session_factory() as session:
            result = await session.get(UserSetting, user_id)
            if result:
                return result.settings_json
            return {"default_provider": "perplexity", "default_model": "auto"}

    @db_lock_retry
    async def save_user_settings(self, user_id: int, settings: Dict[str, Any]) -> None:
        async with self.session_factory() as session:
            user_setting = UserSetting(user_id=user_id, settings_json=settings)
            await session.merge(user_setting)
            await session.commit()

    @db_lock_retry
    async def get_conversation_by_id(
        self, conversation_id: str
    ) -> Optional[Conversation]:
        async with self.session_factory() as session:
            stmt = (
                select(Conversation)
                .where(Conversation.conversation_id == conversation_id)
                .options(selectinload(Conversation.messages))
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    @db_lock_retry
    async def get_conversation_id_by_prefix(self, prefix: str) -> Optional[str]:
        async with self.session_factory() as session:
            stmt = (
                select(Conversation.conversation_id)
                .where(Conversation.conversation_id.like(f"{prefix}%"))
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
