from .base import BaseLLMProvider, AttachmentInput
from storage import (
    DatabaseManager,
    ProviderType,
    Conversation,
    ProviderCapability,
    MessageRole,
)
from config import logger
from decorators import resilient_request

import base64
from typing import AsyncIterator, Optional, List, Dict, Any, cast
from openai import AsyncOpenAI, APIError


class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic provider for APIs compatible with the OpenAI SDK"""

    def __init__(
        self,
        storage: DatabaseManager,
        api_key: str,
        base_url: str,
        model: str,
        default_system_prompt: str = "You are a helpful AI assistant.",
    ):
        super().__init__(storage)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.default_system_prompt = default_system_prompt

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(
            f"OpenAI Compatible provider initialized (URL: {base_url}, Model: {model})"
        )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.CLIENT_HISTORY

    @property
    def capabilities(self) -> List[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.ACCEPTS_IMAGES,
        ]

    def get_available_models(self) -> List[str]:
        return [self.model]

    def _encode_image(self, image_data: bytes) -> str:
        return base64.b64encode(image_data).decode("utf-8")

    def _prepare_messages(
        self, conversation: Conversation, attachments: Optional[List[AttachmentInput]]
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        if (
            not conversation.messages
            or conversation.messages[0].role != MessageRole.SYSTEM
        ):
            messages.append({"role": "system", "content": self.default_system_prompt})

        for msg in conversation.messages:
            role = "user" if msg.role == MessageRole.USER else "assistant"
            if msg.role == MessageRole.SYSTEM:
                role = "system"

            messages.append({"role": role, "content": msg.content})

        if attachments:
            last_msg = messages[-1]
            if last_msg["role"] != "user":
                messages.append({"role": "user", "content": []})
                last_msg = messages[-1]

            # Ensure content is a list for mixed content
            if isinstance(last_msg["content"], str):
                last_msg["content"] = [{"type": "text", "text": last_msg["content"]}]
            elif not isinstance(last_msg["content"], list):
                last_msg["content"] = []

            content_list = cast(List[Dict[str, Any]], last_msg["content"])

            for att in attachments:
                if att["content_type"].startswith("image/"):
                    b64_img = self._encode_image(att["data"])
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{att['content_type']};base64,{b64_img}"
                            },
                        }
                    )
                else:
                    try:
                        text_content = att["data"].decode("utf-8", errors="ignore")
                        snippet = f"\n\n[Attached File: {att['filename']}]\n```\n{text_content}\n```"

                        found_text = False
                        for item in content_list:
                            if item.get("type") == "text":
                                item["text"] = str(item.get("text", "")) + snippet
                                found_text = True
                                break
                        if not found_text:
                            content_list.append({"type": "text", "text": snippet})
                    except Exception:
                        pass

        return messages

    @resilient_request(scope="openai_api", max_retries=2, use_circuit_breaker=True)
    async def _create_chat_completion(
        self, messages: List[Dict[str, Any]], model: str
    ) -> Any:
        return await self.client.chat.completions.create(
            model=model, messages=cast(Any, messages), stream=True, temperature=0.6
        )

    async def generate_response(
        self,
        conversation: Conversation,
        stream: bool = True,
        attachments: Optional[List[AttachmentInput]] = None,
    ) -> AsyncIterator[str]:

        model = self.model
        if (
            conversation.meta_data.get("model")
            and conversation.meta_data.get("model") != "auto"
        ):
            model = cast(str, conversation.meta_data.get("model"))

        messages = self._prepare_messages(conversation, attachments)

        try:
            completion_stream = await self._create_chat_completion(messages, model)

            async for chunk in completion_stream:
                content: Optional[str] = chunk.choices[0].delta.content
                if content:
                    yield content

        except APIError as e:
            logger.error(f"OpenAI API Error: {e}")
            yield f"Provider Error: {str(e)}"
        except Exception as e:
            logger.error(f"General Generation Error: {e}")
            yield f"Error: {str(e)}"
