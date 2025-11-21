import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY

from storage.models import Conversation, MessageRole, ProviderType
from storage.database import DatabaseManager
from providers.provider_manager import ProviderManager
from providers.openai_compatible import OpenAICompatibleProvider
from providers.perplexity import PerplexityProvider


@pytest.fixture
def mock_storage():
    return AsyncMock(spec=DatabaseManager)


@pytest.fixture
def sample_conversation():
    conv = Conversation(
        conversation_id="test_conv",
        chat_id=123,
        topic_id=None,
        topic_name="Test",
        meta_data={"provider": "openai", "model": "gpt-4"},
    )
    conv.add_message(MessageRole.USER, "Hello")
    return conv


class TestProviderManager:
    def test_register_and_get_provider(self, mock_storage):
        manager = ProviderManager(mock_storage)

        # Mock a provider class
        MockClass = MagicMock()
        MockClass.return_value.provider_type = ProviderType.CLIENT_HISTORY

        config = {"api_key": "123"}
        manager.register("test_prov", MockClass, config)

        # Retrieve
        provider = manager.get_provider("test_prov", model="my-model")

        # Verify instantiation
        MockClass.assert_called_with(
            storage=mock_storage, api_key="123", model="my-model"
        )
        assert provider == MockClass.return_value

    def test_get_available_providers_filtering(self, mock_storage):
        manager = ProviderManager(mock_storage)

        # Setup Perplexity (Server History)
        pplx_mock = MagicMock()
        pplx_mock.provider_type = ProviderType.SERVER_HISTORY

        # Setup Groq (Client History)
        groq_mock = MagicMock()
        groq_mock.provider_type = ProviderType.CLIENT_HISTORY

        # Inject instances directly
        manager._provider_classes = {"perplexity": MagicMock(), "groq": MagicMock()}

        # Mock get_provider to return our typed mocks
        def get_prov_side_effect(name, model=None):
            return pplx_mock if name == "perplexity" else groq_mock

        manager.get_provider = MagicMock(side_effect=get_prov_side_effect)

        # Case 1: New conversation (not active) -> Show all
        available = manager.get_available_providers(in_active_conversation=False)
        assert "perplexity" in available
        assert "groq" in available

        # Case 2: Active conversation, current is 'groq' -> Hide 'perplexity'
        available = manager.get_available_providers(
            in_active_conversation=True, current_provider="groq"
        )
        assert "groq" in available
        assert "perplexity" not in available

        # Case 3: Active conversation, current is 'perplexity' -> Show both (current is kept)
        available = manager.get_available_providers(
            in_active_conversation=True, current_provider="perplexity"
        )
        assert "perplexity" in available
        assert "groq" in available


class TestOpenAIProvider:
    @pytest.fixture
    def provider(self, mock_storage):
        with patch("providers.openai_compatible.AsyncOpenAI") as mock_client_cls:
            prov = OpenAICompatibleProvider(
                storage=mock_storage,
                api_key="sk-test",
                base_url="http://test",
                model="gpt-test",
            )
            prov.client = mock_client_cls.return_value
            return prov

    def test_prepare_messages(self, provider, sample_conversation):
        """Test conversion of Conversation object to OpenAI message format."""
        msgs = provider._prepare_messages(sample_conversation, attachments=None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello"

    def test_prepare_messages_with_attachments(self, provider, sample_conversation):
        """Test text attachments are appended to content."""
        attachments = [
            {
                "filename": "test.py",
                "content_type": "text/x-python",
                "data": b"print('hi')",
            }
        ]

        msgs = provider._prepare_messages(sample_conversation, attachments)

        last_msg = msgs[-1]
        assert "print('hi')" in str(last_msg["content"])
        assert "test.py" in str(last_msg["content"])

    @pytest.mark.asyncio
    async def test_generate_response_stream(self, provider, sample_conversation):
        """Test streaming response."""
        # Mock the completion stream
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello "

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = "World"

        # Create an async generator
        async def async_iter():
            yield mock_chunk
            yield mock_chunk2

        provider.client.chat.completions.create = AsyncMock(return_value=async_iter())

        # Execute
        chunks = []
        async for chunk in provider.generate_response(sample_conversation):
            chunks.append(chunk)

        assert chunks == ["Hello ", "World"]


class TestPerplexityProvider:
    @pytest.fixture
    def provider(self, mock_storage):
        # Mock AsyncSession so it doesn't try to connect
        with patch("providers.perplexity.AsyncSession") as mock_session_cls:
            prov = PerplexityProvider(
                storage=mock_storage, cookies="uid=123; session=abc", model="auto"
            )
            prov.session = mock_session_cls.return_value
            return prov

    def test_cookie_parsing(self, mock_storage):
        """Test that cookie string is parsed into dict."""
        with patch("providers.perplexity.AsyncSession"):
            prov = PerplexityProvider(mock_storage, "key1=val1; key2=val2")
            assert prov.cookies_dict["key1"] == "val1"
            assert prov.cookies_dict["key2"] == "val2"

    @pytest.mark.asyncio
    async def test_generate_response_sse_parsing(self, provider, sample_conversation):
        """Test parsing of Perplexity's specific SSE format."""

        mock_resp = MagicMock()
        mock_resp.ok = True

        # Packet 1: Offset 0, chunks ["Hello"] -> Yields "Hello", local offset -> 1.
        # Packet 2: Offset 1, chunks [" World"] -> Yields " World", local offset -> 2.
        lines_fixed = [
            b'data: {"backend_uuid": "t1"}',
            b'data: {"blocks": [{"intended_usage": "ask_text", "markdown_block": {"chunks": ["Hello"], "chunk_starting_offset": 0}}]}',
            b'data: {"blocks": [{"intended_usage": "ask_text", "markdown_block": {"chunks": [" World"], "chunk_starting_offset": 1}}]}',
        ]

        async def line_iter_fixed():
            for line in lines_fixed:
                yield line

        # Set the return value correctly
        mock_resp.aiter_lines.return_value = line_iter_fixed()

        # Mock _establish_connection to return our fake response
        provider._establish_connection = AsyncMock(return_value=mock_resp)

        # Execute
        chunks = []
