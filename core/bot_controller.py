from storage import DatabaseManager
from providers import ProviderManager
from utils import WebServer, MessageFormatter
from .handlers import KeyboardHandler
from storage import Conversation, MessageRole, ProviderCapability, Asset
from config import Config, logger
from providers.base import AttachmentInput
from decorators import operation, resilient_request
import asyncio
import uuid
import secrets
from typing import Dict, Tuple, List, Optional, Union, cast, AsyncIterator, Any
import traceback
import time
import mimetypes
from io import BytesIO
from aiogram import Bot
from aiogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyParameters,
    PhotoSize,
    Document,
)
from aiogram.enums import ParseMode


class BotController:
    def __init__(
        self,
        bot: Bot,
        storage: DatabaseManager,
        provider_manager: ProviderManager,
        web_server: WebServer,
        formatter: MessageFormatter,
    ):
        self.bot = bot
        self.storage = storage
        self.provider_manager = provider_manager
        self.web_server = web_server
        self.formatter = formatter
        self.keyboard_handler = KeyboardHandler(storage, provider_manager)
        self._pending_attachments: Dict[int, List[AttachmentInput]] = {}
        self._media_group_buffer: Dict[str, Dict[str, Any]] = {}

    @resilient_request(scope="telegram", max_retries=3)
    async def _send_message(self, chat_id: int, **kwargs: Any) -> Message:
        return await self.bot.send_message(chat_id=chat_id, **kwargs)

    @resilient_request(scope="telegram", max_retries=3)
    async def _edit_message_text(self, chat_id: int, **kwargs: Any) -> bool:
        try:
            await self.bot.edit_message_text(chat_id=chat_id, **kwargs)
            return True
        except Exception:
            raise

    @resilient_request(scope="telegram", max_retries=3)
    async def _edit_message_reply_markup(self, chat_id: int, **kwargs: Any) -> None:
        await self.bot.edit_message_reply_markup(chat_id=chat_id, **kwargs)

    @operation(name="handle_user_message", notify_user=True)
    async def handle_user_message(self, message: Message) -> None:
        """Entry point for pure text messages"""
        await self._process_conversation_flow(message, message.text or "")

    @operation(name="handle_user_document", notify_user=True)
    async def handle_user_document(self, message: Message) -> None:
        await self._handle_media_upload(message, is_photo=False)

    @operation(name="handle_user_photo", notify_user=True)
    async def handle_user_photo(self, message: Message) -> None:
        await self._handle_media_upload(message, is_photo=True)

    async def _handle_media_upload(self, message: Message, is_photo: bool) -> None:
        """Unified handler for photos and documents with media group support"""

        try:
            file_data, filename, mime_type = await self._download_file(
                message, is_photo
            )
        except ValueError as e:
            logger.warning(f"Failed to download file: {e}")
            return

        attachment: AttachmentInput = {
            "filename": filename,
            "content_type": mime_type,
            "data": file_data,
        }

        if message.media_group_id:
            await self._buffer_media_group(message, attachment)
            return

        if message.caption:
            await self._process_conversation_flow(
                message, text=message.caption, immediate_attachments=[attachment]
            )
        else:
            if not message.from_user:
                return
            user_id = message.from_user.id
            self._pending_attachments.setdefault(user_id, []).append(attachment)
            await message.reply("File attached. Please send your prompt.", quote=True)

    async def _download_file(
        self, message: Message, is_photo: bool
    ) -> Tuple[bytes, str, str]:
        """Download file content from Telegram"""
        if is_photo:
            if not message.photo:
                raise ValueError("No photo in message")
            photo: PhotoSize = message.photo[-1]
            file_id = photo.file_id
            filename = "photo.jpg"
            mime_type = "image/jpeg"
        else:
            doc: Optional[Document] = message.document
            if not doc:
                raise ValueError("No document in message")
            file_id = doc.file_id
            filename = doc.file_name or "document"
            mime_type = self._guess_content_type(filename)

        file = await self.bot.get_file(file_id)
        if not file.file_path:
            raise ValueError("File path not available")

        buf = BytesIO()
        await self.bot.download_file(file.file_path, destination=buf)
        return buf.getvalue(), filename, mime_type

    async def _buffer_media_group(
        self, message: Message, attachment: AttachmentInput
    ) -> None:
        """Buffer media group items and process when complete"""
        mg_id = message.media_group_id
        if not mg_id:
            return

        if mg_id not in self._media_group_buffer:
            self._media_group_buffer[mg_id] = {
                "files": [],
                "caption": None,
                "message": message,
                "task": None,
            }

        group = self._media_group_buffer[mg_id]
        group["files"].append(attachment)

        if message.caption and not group["caption"]:
            group["caption"] = message.caption
            group["message"] = message

        if group["task"]:
            group["task"].cancel()

        group["task"] = asyncio.create_task(self._media_group_timer(mg_id))

    async def _media_group_timer(self, mg_id: str) -> None:
        """Wait for more files, then finalize"""
        try:
            await asyncio.sleep(1.5)
            await self._finalize_media_group(mg_id)
        except asyncio.CancelledError:
            pass

    async def _finalize_media_group(self, mg_id: str) -> None:
        """Process the complete media group"""
        if mg_id not in self._media_group_buffer:
            return

        group = self._media_group_buffer.pop(mg_id)
        message = group["message"]
        attachments = group["files"]
        caption = group["caption"]

        if caption:
            await self._process_conversation_flow(
                message, text=caption, immediate_attachments=attachments
            )
        else:
            if message.from_user:
                user_id = message.from_user.id
                self._pending_attachments.setdefault(user_id, []).extend(attachments)
                await message.reply(
                    f"Received {len(attachments)} files. Please send your prompt.",
                    quote=True,
                )

    async def _process_conversation_flow(
        self,
        message: Message,
        text: str,
        immediate_attachments: Optional[List[AttachmentInput]] = None,
    ) -> None:
        """Handle a user interaction (text or caption) that triggers a response generation"""
        if not message.from_user:
            return
        user_id = message.from_user.id
        chat_id = message.chat.id
        thread_id = message.message_thread_id

        pending = self._pending_attachments.pop(user_id, [])
        all_attachments = pending + (immediate_attachments or [])

        is_general_topic = thread_id in (1, None)
        if is_general_topic:
            topic_name = self._generate_topic_name(text)
            try:
                forum_topic = await self.bot.create_forum_topic(
                    chat_id=chat_id, name=topic_name
                )
                thread_id = forum_topic.message_thread_id

                user_settings = await self.storage.get_user_settings(user_id)
                default_provider = user_settings.get("default_provider", "perplexity")
                default_model = user_settings.get("default_model", "auto")

                transfer_keyboard = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text=f"Go to {topic_name}",
                                url=self._get_topic_url(chat_id, thread_id),
                            )
                        ],
                        [
                            InlineKeyboardButton(
                                text="Change Default Settings",
                                callback_data=f"settings:u:open:{user_id}",
                            )
                        ],
                    ]
                )

                await self._send_message(
                    chat_id=chat_id,
                    message_thread_id=message.message_thread_id,
                    text=f"New topic created: {topic_name}\nProvider: {default_provider}\nModel: {default_model}",
                    reply_parameters=ReplyParameters(message_id=message.message_id),
                    reply_markup=transfer_keyboard,
                )

                await self.bot.forward_message(
                    chat_id=chat_id,
                    from_chat_id=chat_id,
                    message_id=message.message_id,
                    message_thread_id=thread_id,
                )

                conversation = Conversation(
                    conversation_id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    topic_id=thread_id,
                    topic_name=topic_name,
                    meta_data={"provider": default_provider, "model": default_model},
                )
            except Exception as e:
                await message.answer(f"Failed to create topic: {e}")
                return
        else:
            conversation = await self.storage.load_conversation(
                chat_id=chat_id, topic_id=thread_id
            )
            if not conversation:
                await message.answer("No conversation found in this topic.")
                return

        conversation.add_message(
            role=MessageRole.USER,
            content=text,
            meta_data={
                "message_id": message.message_id,
                "attachment_count": len(all_attachments),
            },
        )
        await self.storage.save_conversation(conversation)

        thinking_msg = await self._send_message(
            chat_id=chat_id, message_thread_id=thread_id, text="Thinking..."
        )

        await self._generate_and_stream_response(
            conversation=conversation,
            thinking_msg=thinking_msg,
            attachments=all_attachments,
        )

    async def _generate_and_stream_response(
        self,
        conversation: Conversation,
        thinking_msg: Message,
        attachments: List[AttachmentInput],
    ) -> None:
        try:
            provider_name = conversation.provider
            model = conversation.model

            provider = self.provider_manager.get_provider(provider_name, model)

            accumulated_text = ""
            last_update_time = time.time()
            sent_messages: Dict[int, Tuple[int, str]] = {}

            caps = getattr(provider, "capabilities", [])
            supports_attachments = any(
                c in caps
                for c in (
                    ProviderCapability.ACCEPTS_FILES,
                    ProviderCapability.ACCEPTS_IMAGES,
                )
            )

            if supports_attachments and attachments:
                response_generator = provider.generate_response(
                    conversation=conversation, attachments=attachments, stream=True
                )
            elif attachments:
                await self._send_message(
                    chat_id=thinking_msg.chat.id,
                    message_thread_id=thinking_msg.message_thread_id,
                    text=f"Provider '{provider_name}' does not support attachments. Ignoring {len(attachments)} files.",
                )
                response_generator = provider.generate_response(
                    conversation=conversation, stream=True
                )
            else:
                response_generator = provider.generate_response(
                    conversation=conversation, stream=True
                )

            async for chunk in response_generator:
                accumulated_text += chunk
                if time.time() - last_update_time >= 2.0 and len(accumulated_text) > 50:
                    await self._update_messages(
                        accumulated_text, thinking_msg, sent_messages
                    )
                    last_update_time = time.time()

            if accumulated_text:
                await self._update_messages(
                    accumulated_text, thinking_msg, sent_messages
                )

            conversation.add_message(
                role=MessageRole.ASSISTANT,
                content=accumulated_text,
                meta_data={"message_id": thinking_msg.message_id},
            )
            await self.storage.save_conversation(conversation)

            await self._finalize(
                conversation, thinking_msg, sent_messages, accumulated_text
            )

        except Exception as e:
            logger.error(f"Error in generation: {e}\n{traceback.format_exc()}")
            error_msg = f"Error: {str(e)}"
            if len(error_msg) > Config.SAFE_MESSAGE_LENGTH:
                error_msg = error_msg[: Config.SAFE_MESSAGE_LENGTH] + "..."
            await self._edit_message_text(
                chat_id=thinking_msg.chat.id,
                message_id=thinking_msg.message_id,
                text=error_msg,
            )

    async def _update_messages(
        self,
        accumulated_text: str,
        thinking_msg: Message,
        sent_messages: Dict[int, Tuple[int, str]],
    ) -> None:
        messages, _ = await self.formatter.format_response_for_telegram(
            accumulated_text
        )
        if messages:
            for i, msg_content in enumerate(messages):
                if i == 0:
                    if 0 not in sent_messages or sent_messages[0][1] != msg_content:
                        try:
                            await self._edit_message_text(
                                chat_id=thinking_msg.chat.id,
                                message_id=thinking_msg.message_id,
                                text=msg_content,
                                parse_mode=ParseMode.MARKDOWN_V2,
                            )
                            sent_messages[0] = (thinking_msg.message_id, msg_content)
                        except:
                            pass
                else:
                    if i in sent_messages:
                        msg_id, last_content = sent_messages[i]
                        if msg_content != last_content:
                            try:
                                await self._edit_message_text(
                                    chat_id=thinking_msg.chat.id,
                                    message_id=msg_id,
                                    text=msg_content,
                                    parse_mode=ParseMode.MARKDOWN_V2,
                                )
                                sent_messages[i] = (msg_id, msg_content)
                            except:
                                pass
                    else:
                        new_msg = await self._send_message(
                            chat_id=thinking_msg.chat.id,
                            message_thread_id=thinking_msg.message_thread_id,
                            text=msg_content,
                            reply_to_message_id=thinking_msg.message_id,
                            parse_mode=ParseMode.MARKDOWN_V2,
                        )
                        sent_messages[i] = (new_msg.message_id, msg_content)
            await asyncio.sleep(0.3)

    async def _finalize(
        self,
        conversation: Conversation,
        first_message: Message,
        sent_messages: Dict[int, Tuple[int, str]],
        full_text: str,
    ) -> None:
        messages, assets = await self.formatter.format_response_for_telegram(full_text)
        page_id = secrets.token_urlsafe(16)

        await self.storage.save_web_page(
            page_id, conversation.conversation_id, len(conversation.messages) - 1
        )

        for asset in assets:
            await self.storage.save_asset(page_id, asset)

        model = conversation.model
        provider_name = conversation.provider

        provider = self.provider_manager.get_provider(provider_name, model)

        keyboard = self._create_keyboard(page_id, assets, conversation, provider)

        last_idx = len(messages) - 1
        if last_idx in sent_messages:
            msg_id = sent_messages[last_idx][0]
            try:
                await self._edit_message_text(
                    chat_id=first_message.chat.id,
                    message_id=msg_id,
                    text=messages[last_idx],
                    reply_markup=keyboard,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
            except:
                await self._edit_message_reply_markup(
                    chat_id=first_message.chat.id,
                    message_id=msg_id,
                    reply_markup=keyboard,
                )

    def _create_keyboard(
        self,
        page_id: str,
        assets: List[Asset],
        conversation: Conversation,
        provider: Any,
    ) -> InlineKeyboardMarkup:
        buttons = []

        buttons.append(
            [
                InlineKeyboardButton(
                    text="Web View", url=self.web_server.get_answer_url(page_id)
                )
            ]
        )

        prov_buttons = provider.create_extra_buttons(conversation)
        if prov_buttons:
            buttons.append(prov_buttons)

        if assets:
            buttons.append(
                [
                    InlineKeyboardButton(
                        text=f"Assets ({len(assets)}) ▼",
                        callback_data=f"assets_menu:{page_id}",
                    )
                ]
            )

        settings_buttons = self.keyboard_handler.create_settings_button(
            conversation.conversation_id
        )
        if settings_buttons:
            buttons.append(settings_buttons)

        return InlineKeyboardMarkup(inline_keyboard=buttons)

    def _guess_content_type(self, filename: str) -> str:
        mime, _ = mimetypes.guess_type(filename)
        return mime or "application/octet-stream"

    def _generate_topic_name(self, text: str) -> str:
        return (text[:47] + "...") if len(text) > 50 else text or "New Conversation"

    def _get_topic_url(self, chat_id: int, topic_id: int) -> str:
        return f"https://t.me/c/{str(chat_id).replace('-100', '')}/{topic_id}"

    async def _get_or_create_conversation_for_message(
        self, message: Message
    ) -> Optional[Conversation]:
        conv = await self.storage.load_conversation(
            chat_id=message.chat.id, topic_id=message.message_thread_id
        )
        if not conv:
            conv = Conversation(
                str(uuid.uuid4()),
                message.chat.id,
                message.message_thread_id,
                f"Chat {message.chat.id}",
            )
        return cast(Optional[Conversation], conv)
