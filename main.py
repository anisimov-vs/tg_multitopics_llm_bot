from config import Config, logger
from storage import DatabaseManager
from providers import ProviderManager
from utils import MessageFormatter, WebServer
from core import (
    BotController,
    KeyboardStateManager,
    handle_assets_menu,
    handle_asset_download,
    handle_assets_back,
)

import sys
import asyncio
import traceback
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties


async def main() -> None:
    """Main entry point"""
    try:
        storage = DatabaseManager(Config.DATABASE_PATH)

        provider_manager = ProviderManager(storage)

        # Auto-load and configure all providers found in the registry
        provider_manager.load_providers(Config)

        formatter = MessageFormatter()
        web_server = WebServer(
            storage=storage, host=Config.WEB_HOST, port=Config.WEB_PORT
        )
        await web_server.start()

        bot = Bot(token=Config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))

        controller = BotController(
            bot=bot,
            storage=storage,
            provider_manager=provider_manager,
            web_server=web_server,
            formatter=formatter,
        )

        dp = Dispatcher(storage=MemoryStorage())
        router = Router()

        # Message Handlers
        @router.message(F.text)
        async def message_handler(message: Message) -> None:
            await controller.handle_user_message(message)

        @router.message(F.document)
        async def document_handler(message: Message) -> None:
            await controller.handle_user_document(message)

        @router.message(F.photo)
        async def photo_handler(message: Message) -> None:
            await controller.handle_user_photo(message)

        # Asset Handlers
        @router.callback_query(F.data.startswith("assets_menu:"))
        async def assets_menu_handler(callback: CallbackQuery) -> None:
            state_manager = KeyboardStateManager(storage)
            await handle_assets_menu(callback, storage, state_manager)

        @router.callback_query(F.data.startswith("asset_dl:"))
        async def asset_download_handler(callback: CallbackQuery) -> None:
            await handle_asset_download(callback, storage)

        @router.callback_query(F.data.startswith("assets_back:"))
        async def assets_back_handler(callback: CallbackQuery) -> None:
            state_manager = KeyboardStateManager(storage)
            await handle_assets_back(callback, storage, state_manager)

        # Unified Settings Handler
        @router.callback_query(F.data.startswith("settings:"))
        async def unified_settings_handler(callback: CallbackQuery) -> None:
            await controller.keyboard_handler.handle_unified_callback(callback)

        dp.include_router(router)

        logger.info("Bot started successfully")
        logger.info(
            f"Registered providers: {list(provider_manager._provider_classes.keys())}"
        )

        await dp.start_polling(bot)

    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
