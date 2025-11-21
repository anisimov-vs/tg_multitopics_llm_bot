import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from aiogram.types import Chat, User, Document, Message, CallbackQuery
from aiogram.enums import ChatType

from core.bot_controller import BotController
from storage.models import Conversation, MessageRole
from storage.database import DatabaseManager
from providers.provider_manager import ProviderManager
from utils.webserver import WebServer
from utils.formatter import MessageFormatter


@pytest.fixture
def mock_bot():
    bot = AsyncMock()
    bot.send_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    bot.edit_message_reply_markup = AsyncMock()
    bot.create_forum_topic = AsyncMock()
    bot.get_file = AsyncMock()
    bot.download_file = AsyncMock()
    return bot


@pytest.fixture
def mock_storage():
    storage = AsyncMock(spec=DatabaseManager)
    # Default mocks for common calls
    storage.get_user_settings.return_value = {
        "default_provider": "mock_prov",
        "default_model": "mock_model",
    }
    storage.save_conversation = AsyncMock()
    storage.load_conversation = AsyncMock(return_value=None)
    storage.save_web_page = AsyncMock()
    storage.save_asset = AsyncMock()
    storage.save_user_settings = AsyncMock()
    storage.save_keyboard_state = AsyncMock()
    storage.load_keyboard_state = AsyncMock(return_value=None)
    storage.delete_keyboard_state = AsyncMock()
    storage.get_conversation_by_id = AsyncMock()
    return storage


@pytest.fixture
def mock_provider_instance():
    """A mock LLM provider instance that yields a predictable stream."""
    provider = MagicMock()

    async def async_gen(*args, **kwargs):
        yield "Hello "
        yield "World"

    provider.generate_response = MagicMock(side_effect=async_gen)
    provider.create_extra_buttons.return_value = []
    provider.capabilities = []
    provider.get_available_models.return_value = ["mock_model"]
    return provider


@pytest.fixture
def mock_provider_manager(mock_provider_instance):
    manager = MagicMock(spec=ProviderManager)
    manager.get_provider.return_value = mock_provider_instance
    manager.get_available_providers.return_value = ["mock_prov", "other_prov"]
    manager.get_available_models.return_value = ["mock_model"]
    manager.get_default_model.return_value = "mock_model"
    # Mock internal dict for strategy access
    manager._provider_classes = {"mock_prov": MagicMock(), "other_prov": MagicMock()}
    return manager


@pytest.fixture
def mock_web_server():
    server = MagicMock(spec=WebServer)
    server.get_answer_url.return_value = "http://localhost/answer/123"
    return server


@pytest.fixture
def mock_formatter():
    formatter = MagicMock(spec=MessageFormatter)
    # Return simple tuple: (List[str messages], List[Asset assets])
    async def format_resp(text):
        return [text], []

    formatter.format_response_for_telegram = AsyncMock(side_effect=format_resp)
    return formatter


@pytest.fixture
def controller(
    mock_bot, mock_storage, mock_provider_manager, mock_web_server, mock_formatter
):
    return BotController(
        bot=mock_bot,
        storage=mock_storage,
        provider_manager=mock_provider_manager,
        web_server=mock_web_server,
        formatter=mock_formatter,
    )


# Helper to create a mock Message that works with await AND isinstance checks
def create_mock_message(text="test", user_id=123, chat_id=-100, thread_id=None):
    msg = AsyncMock(spec=Message)

    msg.message_id = 1
    msg.date = datetime.now()

    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = ChatType.SUPERGROUP

    # Assign explicitly to avoid attribute access errors with spec
    user = MagicMock(spec=User)
    user.id = user_id
    user.first_name = "TestUser"
    msg.from_user = user

    msg.text = text
    msg.message_thread_id = thread_id

    msg.caption = None
    msg.document = None
    msg.photo = None
    msg.media_group_id = None

    # Ensure methods are AsyncMock so they can be awaited
    msg.reply = AsyncMock()
    msg.answer = AsyncMock()
    msg.edit_reply_markup = AsyncMock()
    msg.delete = AsyncMock()
    msg.edit_message_text = AsyncMock()

    return msg


@pytest.mark.asyncio
async def test_handle_new_conversation_flow(controller, mock_bot, mock_storage):
    """
    Test handling a message in the 'General' topic (thread_id=1 or None).
    Expectation: Create a new forum topic, save conversation, generate response.
    """
    message = create_mock_message(text="Hello Bot", thread_id=None)

    # Mock ForumTopic return
    mock_topic = MagicMock()
    mock_topic.message_thread_id = 99
    mock_topic.name = "Hello Bot..."
    mock_bot.create_forum_topic.return_value = mock_topic

    # Mock send_message return
    sent_msg_mock = AsyncMock()
    sent_msg_mock.message_id = 50
    sent_msg_mock.chat.id = -100
    sent_msg_mock.message_thread_id = 99
    mock_bot.send_message.return_value = sent_msg_mock

    # Execute
    await controller.handle_user_message(message)

    # Assertions
    mock_bot.create_forum_topic.assert_called_once()
    assert mock_storage.save_conversation.called

    # Check arg was a Conversation object
    saved_conv = mock_storage.save_conversation.call_args[0][0]
    assert isinstance(saved_conv, Conversation)
    assert saved_conv.topic_id == 99
    assert saved_conv.messages[0].content == "Hello Bot"

    # Check final response edit
    mock_bot.edit_message_text.assert_called()
    _, kwargs = mock_bot.edit_message_text.call_args
    assert kwargs["text"] == "Hello World"


@pytest.mark.asyncio
async def test_handle_existing_conversation_flow(controller, mock_bot, mock_storage):
    """
    Test handling a message in an existing topic.
    Expectation: Load existing conversation, append message, generate response.
    """
    message = create_mock_message(text="Continue", thread_id=99)

    # Mock existing conversation in DB
    existing_conv = Conversation(
        conversation_id="uuid-1", chat_id=-100, topic_id=99, topic_name="Existing"
    )
    mock_storage.load_conversation.return_value = existing_conv

    sent_msg_mock = AsyncMock()
    sent_msg_mock.message_id = 51
    sent_msg_mock.chat.id = -100
    sent_msg_mock.message_thread_id = 99
    mock_bot.send_message.return_value = sent_msg_mock

    # Execute
    await controller.handle_user_message(message)

    # Assertions
    mock_bot.create_forum_topic.assert_not_called()

    # Logic appends USER message, then ASSISTANT message.
    # So existing_conv.messages should have 2 messages now.
    assert len(existing_conv.messages) == 2
    assert existing_conv.messages[0].role == MessageRole.USER
    assert existing_conv.messages[0].content == "Continue"
    assert existing_conv.messages[1].role == MessageRole.ASSISTANT
    assert existing_conv.messages[1].content == "Hello World"

    mock_storage.save_conversation.assert_called()


@pytest.mark.asyncio
async def test_handle_attachment_upload(controller, mock_bot):
    """Test handling a document upload (queuing it for the next prompt)."""
    message = create_mock_message(text=None)

    doc = MagicMock(spec=Document)
    doc.file_id = "file_123"
    doc.file_name = "test.py"
    doc.mime_type = "text/x-python"
    message.document = doc

    # Mock file download
    mock_file_info = MagicMock()
    mock_file_info.file_path = "path/to/file"
    mock_bot.get_file.return_value = mock_file_info

    # Simulate download_file writing to BytesIO
    async def side_effect_download(file_path, destination):
        destination.write(b"print('hello')")

    mock_bot.download_file.side_effect = side_effect_download

    await controller.handle_user_document(message)

    # Assert file was downloaded
    mock_bot.download_file.assert_called()

    # Assert attachment is queued in controller memory
    assert 123 in controller._pending_attachments
    assert len(controller._pending_attachments[123]) == 1
    assert controller._pending_attachments[123][0]["filename"] == "test.py"
    assert controller._pending_attachments[123][0]["data"] == b"print('hello')"

    # Assert confirmation reply
    message.reply.assert_called_with(
        "File attached. Please send your prompt.", quote=True
    )


@pytest.mark.asyncio
async def test_settings_open_menu(controller, mock_storage):
    """Test opening the settings menu for a user context."""
    handler = controller.keyboard_handler

    # Setup mock callback with spec=CallbackQuery for isinstance check
    callback = AsyncMock(spec=CallbackQuery)
    callback.id = "1"
    callback.data = "settings:u:open:123"

    # Explicitly assign user mock
    user = MagicMock(spec=User)
    user.id = 123
    callback.from_user = user

    # Explicitly assign answer to AsyncMock to avoid TypeError in await
    callback.answer = AsyncMock()

    # Setup mock message inside callback
    msg = create_mock_message()
    msg.reply_markup = MagicMock()
    callback.message = msg

    mock_storage.get_user_settings.return_value = {
        "default_provider": "perplexity",
        "default_model": "auto",
    }

    # Execute
    await handler.handle_unified_callback(callback)

    # Assertions
    # Should save old keyboard state
    handler.state_manager.storage.save_keyboard_state.assert_called()

    # Should edit message with new menu
    assert msg.edit_reply_markup.called
    kwargs = msg.edit_reply_markup.call_args[1]
    keyboard = kwargs["reply_markup"]

    # Check keyboard structure
    buttons = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert any("Provider: perplexity" in b for b in buttons)
    assert any("Model: auto" in b for b in buttons)

    # Should answer callback to stop loading animation
    callback.answer.assert_called()


@pytest.mark.asyncio
async def test_settings_navigation_provider_list(
    controller, mock_storage, mock_provider_manager
):
    """Test navigating to the provider selection list."""
    handler = controller.keyboard_handler

    callback = AsyncMock(spec=CallbackQuery)
    callback.id = "1"
    callback.data = "settings:u:nav:123:prov"

    user = MagicMock(spec=User)
    user.id = 123
    callback.from_user = user

    # Explicitly assign answer to AsyncMock
    callback.answer = AsyncMock()

    msg = create_mock_message()
    callback.message = msg

    # Execute
    await handler.handle_unified_callback(callback)

    # Assertions
    assert msg.edit_reply_markup.called
    kwargs = msg.edit_reply_markup.call_args[1]
    keyboard = kwargs["reply_markup"]

    # Should list providers from provider_manager
    buttons = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert "mock_prov" in buttons
    assert "other_prov" in buttons


@pytest.mark.asyncio
async def test_settings_pick_provider(controller, mock_storage):
    """Test actually selecting a new provider."""
    handler = controller.keyboard_handler

    callback = AsyncMock(spec=CallbackQuery)
    callback.id = "1"

    user = MagicMock(spec=User)
    user.id = 123
    callback.from_user = user

    # Explicitly assign answer to AsyncMock
    callback.answer = AsyncMock()

    # Calculate hash for "other_prov"
    prov_name = "other_prov"
    h = handler._hash_val(prov_name)
    callback.data = f"settings:u:pick:123:prov:{h}"

    msg = create_mock_message()
    callback.message = msg

    # Execute
    await handler.handle_unified_callback(callback)

    # Assertions
    # Should update settings in DB
    mock_storage.save_user_settings.assert_called()
    call_args = mock_storage.save_user_settings.call_args
    assert call_args[0][0] == 123
    assert call_args[0][1]["default_provider"] == "other_prov"

    # Should answer callback
    callback.answer.assert_called()
