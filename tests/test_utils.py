import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from aiohttp import web
import telegramify_markdown.type

from utils.formatter import MessageFormatter
from utils.webserver import WebServer
from storage.database import DatabaseManager


@pytest.fixture
def formatter():
    return MessageFormatter()


@pytest.fixture
def mock_storage():
    return AsyncMock(spec=DatabaseManager)


@pytest.fixture
def web_server(mock_storage):
    return WebServer(mock_storage, host="127.0.0.1", port=5000)


class TestMessageFormatter:
    def test_latex_to_unicode_simple(self, formatter):
        """Test basic LaTeX symbol conversion."""
        # Greek
        assert formatter._latex_to_unicode(r"\alpha") == "α"
        assert formatter._latex_to_unicode(r"\beta") == "β"
        # Math
        assert formatter._latex_to_unicode(r"\infty") == "∞"
        assert formatter._latex_to_unicode(r"\rightarrow") == "→"

    def test_latex_to_unicode_superscripts(self, formatter):
        """Test superscript conversion."""
        assert formatter._latex_to_unicode("x^2") == "x²"
        assert formatter._latex_to_unicode("x^{12}") == "x¹²"
        assert formatter._latex_to_unicode("y^n") == "yⁿ"

    def test_latex_to_unicode_subscripts(self, formatter):
        """Test subscript conversion."""
        assert formatter._latex_to_unicode("H_2O") == "H₂O"
        # Only testing supported chars (i, x) as per formatter.py implementation
        assert formatter._latex_to_unicode("x_{ix}") == "xᵢₓ"

    def test_latex_fractions(self, formatter):
        """Test fraction flattening."""
        latex = r"\frac{a}{b}"
        assert formatter._latex_to_unicode(latex) == "(a)/(b)"

    @pytest.mark.asyncio
    async def test_preprocess_latex_preserves_code(self, formatter):
        """Ensure LaTeX inside code blocks is NOT converted."""
        text = "Check `\\alpha` inside code."

        # The function is decorated with @cpu_bound, so it must be awaited
        processed = await formatter._preprocess_latex_in_markdown(text)
        assert "`\\alpha`" in processed

    @pytest.mark.asyncio
    async def test_format_response_splits_long_messages(self, formatter):
        """Test that multiple blocks are split into messages if combined len > limit."""
        # Create 2 chunks that fit individually but exceed limit together
        chunk_size = 3000
        text_1 = "a" * chunk_size
        text_2 = "b" * chunk_size

        # Mock telegramify to return two text boxes
        mock_box_1 = MagicMock(spec=telegramify_markdown.type.Text)
        mock_box_1.content = text_1

        mock_box_2 = MagicMock(spec=telegramify_markdown.type.Text)
        mock_box_2.content = text_2

        with patch(
            "telegramify_markdown.telegramify", new_callable=AsyncMock
        ) as mock_tg:
            mock_tg.return_value = [mock_box_1, mock_box_2]

            messages, assets = await formatter.format_response_for_telegram(
                "input ignored"
            )

            # Should be split into 2 messages
            assert len(messages) == 2
            assert messages[0] == text_1
            assert messages[1] == text_2
            assert len(assets) == 0

    @pytest.mark.asyncio
    async def test_format_response_extracts_code_assets(self, formatter):
        """Test that file boxes are converted to Assets and code blocks."""

        code_bytes = b"print('hello')"
        # Use MagicMock for File object
        mock_file_box = MagicMock(spec=telegramify_markdown.type.File)
        mock_file_box.file_name = "script.py"
        mock_file_box.file_data = code_bytes

        with patch(
            "telegramify_markdown.telegramify", new_callable=AsyncMock
        ) as mock_tg:
            mock_tg.return_value = [mock_file_box]

            messages, assets = await formatter.format_response_for_telegram("Some code")

            assert len(assets) == 1
            assert assets[0].file_name.endswith(".py")
            assert assets[0].file_data == code_bytes
            assert assets[0].language == "python"

            # The message should contain a code block representation
            assert "```python" in messages[0]
            assert "print('hello')" in messages[0]

    @pytest.mark.asyncio
    async def test_fallback_on_error(self, formatter):
        """Test fallback to simple splitting if formatting crashes."""
        with patch(
            "telegramify_markdown.telegramify", side_effect=Exception("Parse Error")
        ):
            messages, assets = await formatter.format_response_for_telegram(
                "Simple text"
            )

            assert messages == ["Simple text"]
            assert assets == []


class TestWebServer:
    @pytest.mark.asyncio
    async def test_view_answer_success(self, web_server, mock_storage):
        """Test rendering a found page."""
        page_id = "p1"
        content = "# Title\nContent"
        mock_storage.load_web_page.return_value = content

        # Create a mock request
        request = MagicMock(spec=web.Request)
        request.match_info = {"page_id": page_id}

        response = await web_server.view_answer(request)

        assert response.status == 200
        assert response.content_type == "text/html"

        # Verify content presence.
        assert "Title" in response.text
        assert "Content" in response.text

    @pytest.mark.asyncio
    async def test_view_answer_not_found(self, web_server, mock_storage):
        """Test 404 when page missing."""
        mock_storage.load_web_page.return_value = None

        request = MagicMock(spec=web.Request)
        request.match_info = {"page_id": "missing"}

        with pytest.raises(web.HTTPNotFound):
            await web_server.view_answer(request)

    def test_get_answer_url_logic(self, web_server):
        """Test URL generation logic (Public vs Local)."""
        page_id = "123"

        # Case 1: No ngrok (Localhost)
        web_server.public_url = None
        local_url = web_server.get_answer_url(page_id)
        assert local_url == f"http://127.0.0.1:5000/answer/{page_id}"

        # Case 2: Ngrok active
        web_server.public_url = "https://random.ngrok-free.app"
        public_url = web_server.get_answer_url(page_id)
        assert public_url == f"https://random.ngrok-free.app/answer/{page_id}"
