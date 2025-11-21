from .base import BaseLLMProvider, AttachmentInput
from storage import DatabaseManager, ProviderType, Conversation, ProviderCapability
from config import logger
from decorators import resilient_request
import asyncio
import json
import uuid
from typing import AsyncIterator, Optional, Dict, List, Any, Union, cast
from urllib.parse import unquote
from curl_cffi.requests import AsyncSession
from curl_cffi import CurlMime
from aiogram.types import InlineKeyboardButton


class PerplexityProvider(BaseLLMProvider):
    """Perplexity LLM provider"""

    AVAILABLE_MODELS = [
        "auto",
        "sonar",
        "research",
        "gpt-5.1",
        "gpt-5.1-reasoning",
        "claude-4.5-sonnet",
        "claude-4.5-sonnet-reasoning",
        "gemini-2.5-pro",
        "grok-4",
        "grok-4-reasoning",
        "o3-mini-reasoning",
    ]

    MODEL_CONFIG = {
        "auto": {"mode": "concise", "model_preference": "turbo"},
        "sonar": {"mode": "copilot", "model_preference": "experimental"},
        "research": {"mode": "copilot", "model_preference": "pplx_alpha"},
        "gpt-5.1": {"mode": "copilot", "model_preference": "gpt51"},
        "gpt-5.1-reasoning": {"mode": "copilot", "model_preference": "gpt51_thinking"},
        "claude-4.5-sonnet": {"mode": "copilot", "model_preference": "claude45sonnet"},
        "claude-4.5-sonnet-reasoning": {
            "mode": "copilot",
            "model_preference": "claude45sonnetthinking",
        },
        "gemini-2.5-pro": {"mode": "copilot", "model_preference": "gemini25pro"},
        "grok-4": {"mode": "copilot", "model_preference": "grok4nonthinking"},
        "grok-4-reasoning": {"mode": "copilot", "model_preference": "grok4"},
        "o3-mini-reasoning": {"mode": "copilot", "model_preference": "o3mini"},
    }

    def __init__(
        self, storage: DatabaseManager, cookies: str, model: Optional[str] = None
    ):
        super().__init__(storage)
        self.model = model or "auto"
        self.cookies_dict = self._parse_cookies(cookies)
        self.session: Optional[AsyncSession] = None
        self._init_session()
        logger.info(f"Perplexity provider initialized (model: {self.model})")

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.SERVER_HISTORY

    @property
    def capabilities(self) -> List[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.ACCEPTS_FILES,
            ProviderCapability.ACCEPTS_IMAGES,
        ]

    def get_available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS.copy()

    def _parse_cookies(self, cookies_str: str) -> Dict[str, str]:
        cookies_dict = {}
        if ";" in cookies_str:
            pairs = cookies_str.split(";")
        else:
            pairs = [cookies_str]
        for pair in pairs:
            pair = pair.strip()
            if "=" in pair:
                key, value = pair.split("=", 1)
                cookies_dict[unquote(key.strip())] = unquote(value.strip())
        return cookies_dict

    def _init_session(self) -> None:
        self.session = AsyncSession(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.perplexity.ai/",
                "Origin": "https://www.perplexity.ai",
                "Priority": "u=1, i",
            },
            cookies=self.cookies_dict,
            impersonate="chrome120",
            timeout=45,
        )

    @resilient_request(scope="perplexity", max_retries=2, use_circuit_breaker=True)
    async def _establish_connection(self, url: str, json_data: Dict[str, Any]) -> Any:
        if self.session is None:
            raise RuntimeError("Session not initialized")
        resp = await self.session.post(
            url,
            headers={
                "accept": "text/event-stream",
                "content-type": "application/json",
                "x-perplexity-request-reason": "perplexity-query-state-provider",
                "x-request-id": str(uuid.uuid4()),
            },
            json=json_data,
            stream=True,
        )

        if not resp.ok:
            raise RuntimeError(f"API returned {resp.status_code}: {resp.text[:100]}")
        return resp

    async def generate_response(
        self,
        conversation: Conversation,
        stream: bool = True,
        attachments: Optional[List[AttachmentInput]] = None,
    ) -> AsyncIterator[str]:
        query = conversation.messages[-1].content
        last_backend_uuid = conversation.meta_data.get("perplexity_thread_id")

        model_config = self.MODEL_CONFIG.get(self.model, self.MODEL_CONFIG["auto"])

        attachment_urls = []
        if attachments:
            try:
                attachment_urls = await self._upload_attachments(attachments)
            except Exception as e:
                yield f"Error uploading attachments: {e}"
                return

        json_data = {
            "query_str": query,
            "params": {
                "attachments": attachment_urls,
                "language": "en-US",
                "mode": model_config["mode"],
                "model_preference": model_config["model_preference"],
                "frontend_uuid": str(uuid.uuid4()),
                "frontend_context_uuid": str(uuid.uuid4()),
                "last_backend_uuid": last_backend_uuid,
                "source": "default",
                "search_focus": "internet",
                "sources": ["web"],
            },
        }

        ask_url = "https://www.perplexity.ai/rest/sse/perplexity_ask"

        has_yielded = False

        try:
            resp = await self._establish_connection(ask_url, json_data)
            chunk_offset = 0

            async for line in resp.aiter_lines():
                line = line.decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                if line.startswith("data: "):
                    json_str = line[6:]

                    try:
                        data = json.loads(json_str)

                        if (
                            "backend_uuid" in data
                            and "perplexity_thread_id" not in conversation.meta_data
                        ):
                            conversation.meta_data["perplexity_thread_id"] = data[
                                "backend_uuid"
                            ]
                            conversation.meta_data["perplexity_thread_url"] = data.get(
                                "thread_url_slug", ""
                            )

                        if "blocks" in data:
                            for block in data["blocks"]:
                                if block.get("intended_usage") == "ask_text":
                                    markdown_block = block.get("markdown_block", {})
                                    chunks = markdown_block.get("chunks", [])
                                    current_offset = markdown_block.get(
                                        "chunk_starting_offset", 0
                                    )

                                    if current_offset >= chunk_offset:
                                        for text_chunk in chunks:
                                            has_yielded = True
                                            yield text_chunk
                                        chunk_offset = current_offset + len(chunks)

                    except json.JSONDecodeError:
                        continue

                elif "end_of_stream" in line:
                    break

            if not has_yielded:
                yield "No response received (Stream ended empty)"

        except Exception as e:
            err_msg = str(e)

            if (
                "curl: (18)" in err_msg or "partial file" in err_msg.lower()
            ) and has_yielded:
                logger.warning(
                    f"Stream ended with partial file warning (ignored as content received): {e}"
                )
                return

            logger.error(f"Stream failed: {e}")
            yield f"Connection error: {err_msg}"

    @resilient_request(scope="perplexity_upload", max_retries=2)
    async def _upload_attachments(self, attachments: List[Dict[str, Any]]) -> List[str]:
        """Async upload of all attachments"""
        uploaded_urls = []
        for att in attachments:
            ticket = await self._create_upload_ticket(
                str(att["filename"]), str(att["content_type"]), len(att["data"])
            )
            await self._upload_to_s3(
                ticket,
                str(att["filename"]),
                str(att["content_type"]),
                att["data"],  # bytes
            )
            uploaded_urls.append(cast(str, ticket["s3_object_url"]))
        return uploaded_urls

    async def _create_upload_ticket(
        self, filename: str, content_type: str, file_size: int
    ) -> Dict[str, Any]:
        """Async ticket creation"""
        if self.session is None:
            raise RuntimeError("Session not initialized")
        file_id = str(uuid.uuid4())
        url = "https://www.perplexity.ai/rest/uploads/batch_create_upload_urls?version=2.18&source=default"
        body = {
            "files": {
                file_id: {
                    "filename": filename,
                    "content_type": content_type,
                    "file_size": file_size,
                    "source": "default",
                }
            }
        }

        resp = await self.session.post(url, json=body)
        if not resp.ok:
            raise RuntimeError(f"Ticket failed: {resp.status_code}")

        data = resp.json()
        result = data["results"][file_id]
        if result.get("error"):
            raise RuntimeError(result["error"])

        return cast(Dict[str, Any], result)

    async def _upload_to_s3(
        self, ticket: Dict[str, Any], filename: str, content_type: str, data: bytes
    ) -> None:
        """Async S3 upload using temporary session"""
        async with AsyncSession(verify=False) as s3_session:
            mp = CurlMime()

            fields = ticket.get("fields", {})
            for k, v in fields.items():
                if v is not None:
                    mp.addpart(name=str(k), data=str(v).encode("utf-8"))

            mp.addpart(
                name="file", filename=filename, content_type=content_type, data=data
            )

            bucket_url = cast(str, ticket["s3_bucket_url"])
            resp = await s3_session.post(bucket_url, multipart=mp, timeout=60)

            if not (200 <= resp.status_code < 300):
                raise RuntimeError(f"S3 upload failed: {resp.status_code} {resp.text}")

    def create_extra_buttons(
        self, conversation: Conversation
    ) -> List[InlineKeyboardButton]:
        thread_url = conversation.meta_data.get("perplexity_thread_url")
        if thread_url:
            return [
                InlineKeyboardButton(
                    text="View on Perplexity",
                    url=f"https://www.perplexity.ai/search/{thread_url}",
                )
            ]
        return []
