import pytest
import pytest_asyncio
import asyncio
from sqlalchemy import text

from storage.database import DatabaseManager
from storage.models import (
    Conversation,
    MessageRole,
    Asset,
)


@pytest.fixture
def db_path(tmp_path):
    """Create a temp file path for the database."""
    d = tmp_path / "data"
    d.mkdir()
    return d / "test_bot.sqlite3"


@pytest_asyncio.fixture
async def storage(db_path):
    """Initialize DatabaseManager with a real temp database."""
    # Initialize manager
    manager = DatabaseManager(str(db_path))

    # Wait for schema initialization to complete
    await asyncio.sleep(0.1)

    # Ensure schema is definitely created for the test
    async with manager.engine.begin() as conn:
        from storage.models import Base

        await conn.run_sync(Base.metadata.create_all)

    yield manager

    # Teardown
    await manager.engine.dispose()


@pytest.mark.asyncio
async def test_schema_creation(storage):
    """Verify tables are created."""
    async with storage.session_factory() as session:
        # Check if conversations table exists
        result = await session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
            )
        )
        assert result.scalar() == "conversations"


@pytest.mark.asyncio
async def test_conversation_lifecycle(storage):
    """Test creating, saving, and loading a conversation with messages."""
    # 1. Create
    conv_id = "uuid-1"
    conv = Conversation(
        conversation_id=conv_id, chat_id=1001, topic_id=None, topic_name="Test Topic"
    )
    conv.add_message(MessageRole.USER, "Hello")
    conv.add_message(MessageRole.ASSISTANT, "Hi there")

    # 2. Save
    await storage.save_conversation(conv)

    # 3. Load
    loaded = await storage.load_conversation(chat_id=1001, topic_id=None)

    assert loaded is not None
    assert loaded.conversation_id == conv_id
    assert loaded.topic_name == "Test Topic"
    assert len(loaded.messages) == 2
    assert loaded.messages[0].content == "Hello"
    assert loaded.messages[0].role == MessageRole.USER
    assert loaded.messages[1].content == "Hi there"
    assert loaded.messages[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_json_properties_and_metadata(storage):
    """Test that JSON properties (provider, model) are correctly serialized/deserialized."""
    conv = Conversation("uuid-2", 2002, 5, "Meta Test")

    # Set via descriptor properties
    conv.provider = "groq"
    conv.model = "llama3"
    conv.perplexity_thread_id = "t-123"

    await storage.save_conversation(conv)

    # Reload
    loaded = await storage.get_conversation_by_id("uuid-2")

    assert loaded.meta_data["provider"] == "groq"
    assert loaded.provider == "groq"
    assert loaded.model == "llama3"
    assert loaded.perplexity_thread_id == "t-123"


@pytest.mark.asyncio
async def test_conversation_update(storage):
    """Test adding messages to an existing conversation."""
    conv = Conversation("uuid-3", 3003, None, "Update Test")
    conv.add_message(MessageRole.USER, "Msg 1")
    await storage.save_conversation(conv)

    # Load and update
    loaded = await storage.get_conversation_by_id("uuid-3")
    loaded.add_message(MessageRole.ASSISTANT, "Msg 2")
    await storage.save_conversation(loaded)

    # Verify persistence
    final = await storage.get_conversation_by_id("uuid-3")
    assert len(final.messages) == 2
    assert final.messages[1].content == "Msg 2"


@pytest.mark.asyncio
async def test_web_page_loading(storage):
    """Test linking a WebPage to a specific message index."""
    # Setup conversation
    conv = Conversation("uuid-web", 4004, None, "Web Test")
    conv.add_message(MessageRole.USER, "Prompt")
    conv.add_message(MessageRole.ASSISTANT, "# Valid Markdown\n\nResponse")
    await storage.save_conversation(conv)

    page_id = "page-123"
    # Link page_id to message index 1
    await storage.save_web_page(page_id, "uuid-web", 1)

    # Load content via page_id
    content = await storage.load_web_page(page_id)
    assert content == "# Valid Markdown\n\nResponse"

    # Test non-existent page
    assert await storage.load_web_page("invalid") is None


@pytest.mark.asyncio
async def test_assets(storage):
    """Test binary asset storage."""
    page_id = "page-123"

    data = b"\x89PNG\r\n\x1a\n..."
    asset = Asset(
        asset_id="asset-1", file_name="image.png", file_data=data, language="image"
    )

    conv = Conversation("uuid-asset", 5005, None, "Asset Test")
    await storage.save_conversation(conv)
    await storage.save_web_page(page_id, "uuid-asset", 0)

    # Save asset
    await storage.save_asset(page_id, asset)

    # Load assets
    assets = await storage.load_assets(page_id)
    assert len(assets) == 1
    assert assets[0].file_name == "image.png"
    assert assets[0].file_data == data

    # Load specific asset
    single = await storage.load_asset(page_id, "asset-1")
    assert single is not None
    assert single.asset_id == "asset-1"


@pytest.mark.asyncio
async def test_user_settings(storage):
    """Test user settings CRUD."""
    user_id = 999

    # 1. Default should be empty or dict
    initial = await storage.get_user_settings(user_id)
    assert initial.get("default_provider") == "perplexity"  # Default in code logic

    # 2. Update
    new_settings = {"default_provider": "groq", "custom_key": "custom_val"}
    await storage.save_user_settings(user_id, new_settings)

    # 3. Retrieve
    updated = await storage.get_user_settings(user_id)
    assert updated["default_provider"] == "groq"
    assert updated["custom_key"] == "custom_val"


@pytest.mark.asyncio
async def test_keyboard_state(storage):
    """Test temporary keyboard state storage."""
    ctx_id = "ctx-1"
    kb_json = '{"inline_keyboard": [[{"text": "btn"}]]}'

    await storage.save_keyboard_state(ctx_id, kb_json)

    # Load
    loaded = await storage.load_keyboard_state(ctx_id)
    assert loaded == kb_json

    # Delete
    await storage.delete_keyboard_state(ctx_id)
    assert await storage.load_keyboard_state(ctx_id) is None


@pytest.mark.asyncio
async def test_conversation_id_by_prefix(storage):
    """Test resolving short IDs."""
    full_id = "550e8400-e29b-41d4-a716-446655440000"
    conv = Conversation(full_id, 6006, None, "Prefix Test")
    await storage.save_conversation(conv)

    # Resolve using first 8 chars
    resolved = await storage.get_conversation_id_by_prefix("550e8400")
    assert resolved == full_id

    # Resolve non-existent
    assert await storage.get_conversation_id_by_prefix("ffffff") is None
