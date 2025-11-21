from storage import DatabaseManager
from providers import ProviderManager
from decorators import operation, resilient_request
from config import logger

from aiogram.types import (
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile,
    Message,
)
import hashlib
import json
from typing import List, Optional, Dict, Any, cast
from abc import ABC, abstractmethod


class KeyboardStateManager:
    """Centralized keyboard state management"""

    def __init__(self, storage: DatabaseManager):
        self.storage = storage

    @staticmethod
    def serialize_keyboard(keyboard: InlineKeyboardMarkup) -> str:
        return json.dumps(
            {
                "inline_keyboard": [
                    [
                        {
                            "text": btn.text,
                            "url": btn.url,
                            "callback_data": btn.callback_data,
                        }
                        for btn in row
                    ]
                    for row in keyboard.inline_keyboard
                ]
            }
        )

    @staticmethod
    def deserialize_keyboard(keyboard_json: str) -> InlineKeyboardMarkup:
        keyboard_data = json.loads(keyboard_json)
        buttons = []
        for row in keyboard_data["inline_keyboard"]:
            button_row = []
            for btn_data in row:
                if "url" in btn_data and btn_data.get("url"):
                    button_row.append(
                        InlineKeyboardButton(text=btn_data["text"], url=btn_data["url"])
                    )
                elif "callback_data" in btn_data and btn_data.get("callback_data"):
                    button_row.append(
                        InlineKeyboardButton(
                            text=btn_data["text"],
                            callback_data=btn_data["callback_data"],
                        )
                    )
            buttons.append(button_row)
        return InlineKeyboardMarkup(inline_keyboard=buttons)

    async def save_keyboard_state(
        self, context_id: str, keyboard: InlineKeyboardMarkup
    ) -> None:
        if not keyboard:
            return
        keyboard_json = self.serialize_keyboard(keyboard)
        await self.storage.save_keyboard_state(f"settings_{context_id}", keyboard_json)

    async def restore_keyboard_state(
        self, context_id: str
    ) -> Optional[InlineKeyboardMarkup]:
        keyboard_json = await self.storage.load_keyboard_state(f"settings_{context_id}")
        if not keyboard_json:
            return None
        return self.deserialize_keyboard(keyboard_json)

    async def delete_keyboard_state(self, context_id: str) -> None:
        await self.storage.delete_keyboard_state(f"settings_{context_id}")


class SettingsStrategy(ABC):
    """Abstract strategy for handling settings logic (Conversation vs User)"""

    def __init__(self, storage: DatabaseManager, provider_manager: ProviderManager):
        self.storage = storage
        self.provider_manager = provider_manager

    @abstractmethod
    async def get_settings(self, context_id: str) -> Dict[str, str]:
        """Return dict with 'provider' and 'model'"""
        pass

    @abstractmethod
    async def update_settings(self, context_id: str, key: str, value: str) -> None:
        """Update a setting"""
        pass

    @abstractmethod
    async def get_available_providers(self, context_id: str) -> List[str]:
        pass


class ConversationStrategy(SettingsStrategy):
    async def get_settings(self, context_id: str) -> Dict[str, str]:
        conv = await self.storage.get_conversation_by_id(context_id)
        if not conv:
            return {"provider": "perplexity", "model": "auto"}
        return {
            "provider": str(conv.meta_data.get("provider", "perplexity")),
            "model": str(conv.meta_data.get("model", "auto")),
        }

    async def update_settings(self, context_id: str, key: str, value: str) -> None:
        conv = await self.storage.get_conversation_by_id(context_id)
        if conv:
            conv.meta_data[key] = value
            if key == "provider":
                conv.meta_data["model"] = self.provider_manager.get_default_model(value)
            await self.storage.save_conversation(conv)

    async def get_available_providers(self, context_id: str) -> List[str]:
        conv = await self.storage.get_conversation_by_id(context_id)
        in_active = len(conv.messages) > 0 if conv else False
        current = str(conv.meta_data.get("provider")) if conv else None
        return self.provider_manager.get_available_providers(in_active, current)


class UserStrategy(SettingsStrategy):
    async def get_settings(self, context_id: str) -> Dict[str, str]:
        user_id = int(context_id)
        settings = await self.storage.get_user_settings(user_id)
        return {
            "provider": str(settings.get("default_provider", "perplexity")),
            "model": str(settings.get("default_model", "auto")),
        }

    async def update_settings(self, context_id: str, key: str, value: str) -> None:
        user_id = int(context_id)
        settings = await self.storage.get_user_settings(user_id)

        db_key = f"default_{key}"
        settings[db_key] = value

        if key == "provider":
            settings["default_model"] = self.provider_manager.get_default_model(value)

        await self.storage.save_user_settings(user_id, settings)

    async def get_available_providers(self, context_id: str) -> List[str]:
        return list(self.provider_manager._provider_classes.keys())


class KeyboardHandler:
    """Unified Handler for Settings Menus"""

    def __init__(self, storage: DatabaseManager, provider_manager: ProviderManager):
        self.storage = storage
        self.provider_manager = provider_manager
        self.state_manager = KeyboardStateManager(storage)

        self.strategies = {
            "c": ConversationStrategy(storage, provider_manager),
            "u": UserStrategy(storage, provider_manager),
        }

    def _hash_val(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def _resolve_val(self, hashed: str, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if self._hash_val(c) == hashed:
                return c
        return None

    async def _resolve_context_id(self, scope: str, short_id: str) -> Optional[str]:
        if scope == "u":
            return short_id
        if scope == "c":
            return cast(
                Optional[str],
                await self.storage.get_conversation_id_by_prefix(short_id),
            )
        return None

    def build_root_menu(
        self, scope: str, short_id: str, settings: Dict[str, str]
    ) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=f"Provider: {settings['provider']}",
                        callback_data=f"settings:{scope}:nav:{short_id}:prov",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=f"Model: {settings['model']}",
                        callback_data=f"settings:{scope}:nav:{short_id}:mod",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="< Back",
                        callback_data=f"settings:{scope}:close:{short_id}",
                    )
                ],
            ]
        )

    def build_list_menu(
        self, scope: str, short_id: str, items: List[str], category: str
    ) -> InlineKeyboardMarkup:
        buttons = []
        for item in items:
            h = self._hash_val(item)
            buttons.append(
                [
                    InlineKeyboardButton(
                        text=item,
                        callback_data=f"settings:{scope}:pick:{short_id}:{category}:{h}",
                    )
                ]
            )

        buttons.append(
            [
                InlineKeyboardButton(
                    text="< Back", callback_data=f"settings:{scope}:nav:{short_id}:root"
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=buttons)

    def create_settings_button(
        self, conversation_id: str
    ) -> List[InlineKeyboardButton]:
        short_id = conversation_id[:12]
        return [
            InlineKeyboardButton(
                text="Conversation Settings",
                callback_data=f"settings:c:open:{short_id}",
            )
        ]

    @operation(name="settings_dispatch", validate_callback_prefix="settings:")
    @resilient_request(scope="telegram")
    async def handle_unified_callback(self, callback: CallbackQuery) -> None:
        # Format: settings:{scope}:{action}:{short_id}[:{extra}...]
        if not callback.data or not isinstance(callback.message, Message):
            return

        parts = callback.data.split(":")
        if len(parts) < 4:
            return

        scope, action, short_id = parts[1], parts[2], parts[3]
        strategy = self.strategies.get(scope)
        if not strategy:
            return

        full_id = await self._resolve_context_id(scope, short_id)
        if not full_id:
            await callback.answer("Context not found", show_alert=True)
            return

        if action == "open":
            if callback.message.reply_markup:
                await self.state_manager.save_keyboard_state(
                    short_id, callback.message.reply_markup
                )
            settings = await strategy.get_settings(full_id)
            kb = self.build_root_menu(scope, short_id, settings)
            await callback.message.edit_reply_markup(reply_markup=kb)
            await callback.answer()

        elif action == "close":
            restored = await self.state_manager.restore_keyboard_state(short_id)
            if restored:
                await callback.message.edit_reply_markup(reply_markup=restored)
                await self.state_manager.delete_keyboard_state(short_id)
                await callback.answer()
            else:
                await callback.message.delete()
                await callback.answer()

        elif action == "nav":
            view = parts[4]
            if view == "root":
                settings = await strategy.get_settings(full_id)
                kb = self.build_root_menu(scope, short_id, settings)
            elif view == "prov":
                items = await strategy.get_available_providers(full_id)
                kb = self.build_list_menu(scope, short_id, items, "prov")
            elif view == "mod":
                settings = await strategy.get_settings(full_id)
                current_prov = settings["provider"]
                items = self.provider_manager.get_available_models(current_prov)
                kb = self.build_list_menu(scope, short_id, items, "mod")

            await callback.message.edit_reply_markup(reply_markup=kb)
            await callback.answer()

        elif action == "pick":
            category, item_hash = parts[4], parts[5]

            candidates = []
            if category == "prov":
                candidates = await strategy.get_available_providers(full_id)
            elif category == "mod":
                settings = await strategy.get_settings(full_id)
                candidates = self.provider_manager.get_available_models(
                    settings["provider"]
                )

            value = self._resolve_val(item_hash, candidates)
            if not value:
                await callback.answer("Invalid selection", show_alert=True)
                return

            key = "provider" if category == "prov" else "model"
            await strategy.update_settings(full_id, key, value)

            settings = await strategy.get_settings(full_id)
            kb = self.build_root_menu(scope, short_id, settings)
            await callback.message.edit_reply_markup(reply_markup=kb)
            await callback.answer(f"Set {key} to {value}")


@operation(name="assets_menu", validate_callback_prefix="assets_menu:")
@resilient_request(scope="telegram")
async def handle_assets_menu(
    callback: CallbackQuery,
    storage: DatabaseManager,
    state_manager: KeyboardStateManager,
) -> None:
    if not callback.data or not isinstance(callback.message, Message):
        return
    page_id = callback.data.split(":")[1]
    assets = await storage.load_assets(page_id)
    if callback.message.reply_markup:
        await state_manager.save_keyboard_state(page_id, callback.message.reply_markup)

    buttons = []
    for asset in assets:
        size_str = (
            f"{asset.size/1024:.1f} KB"
            if asset.size < 1024 * 1024
            else f"{asset.size/1024/1024:.1f} MB"
        )
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{asset.file_name} ({size_str})",
                    callback_data=f"asset_dl:{page_id}:{asset.asset_id}",
                )
            ]
        )
    buttons.append(
        [InlineKeyboardButton(text="← Back", callback_data=f"assets_back:{page_id}")]
    )

    await callback.message.edit_reply_markup(
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
    )
    await callback.answer()


@operation(name="asset_download", validate_callback_prefix="asset_dl:")
@resilient_request(scope="telegram", max_retries=3)
async def handle_asset_download(
    callback: CallbackQuery, storage: DatabaseManager
) -> None:
    if not callback.data or not isinstance(callback.message, Message):
        return
    _, page_id, asset_id = callback.data.split(":")
    asset = await storage.load_asset(page_id, asset_id)
    if not asset:
        await callback.answer("Asset not found", show_alert=True)
        return
    await callback.message.reply_document(
        document=BufferedInputFile(asset.file_data, asset.file_name),
        caption=f"Code file: {asset.file_name}",
    )
    await callback.answer(f"Sent {asset.file_name}")


@operation(name="assets_back", validate_callback_prefix="assets_back:")
@resilient_request(scope="telegram")
async def handle_assets_back(
    callback: CallbackQuery,
    storage: DatabaseManager,
    state_manager: KeyboardStateManager,
) -> None:
    if not callback.data or not isinstance(callback.message, Message):
        return
    page_id = callback.data.split(":")[1]
    restored = await state_manager.restore_keyboard_state(page_id)
    if restored:
        await callback.message.edit_reply_markup(reply_markup=restored)
        await state_manager.delete_keyboard_state(page_id)
        await callback.answer()
    else:
        await callback.answer("Error restoring menu", show_alert=True)
