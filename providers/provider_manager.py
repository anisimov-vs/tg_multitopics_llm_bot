from .base import BaseLLMProvider
from storage import DatabaseManager, ProviderType
from config import logger
from typing import Dict, Optional, List, Any, Type


class ProviderManager:
    """Manages multiple LLM providers with dynamic switching"""

    def __init__(self, storage: DatabaseManager):
        self.storage = storage
        self._provider_classes: Dict[str, Type[BaseLLMProvider]] = {}
        self._provider_configs: Dict[str, Dict[str, Any]] = {}
        self._provider_instances: Dict[str, BaseLLMProvider] = {}
        logger.info("ProviderManager initialized")

    def register(
        self, name: str, provider_class: Type[BaseLLMProvider], config: Dict[str, Any]
    ) -> None:
        """Register a provider with its configuration"""
        self._provider_classes[name] = provider_class
        self._provider_configs[name] = config
        logger.info(f"Registered provider: {name}")

    def get_provider(self, name: str, model: Optional[str] = None) -> BaseLLMProvider:
        """Get or create provider instance"""
        cache_key = f"{name}:{model}" if model else name

        if cache_key not in self._provider_instances:
            if name not in self._provider_classes:
                raise ValueError(f"Unknown provider: {name}")

            config = self._provider_configs[name].copy()
            if model:
                config["model"] = model

            self._provider_instances[cache_key] = self._provider_classes[name](
                storage=self.storage, **config
            )
            logger.info(f"Created provider instance: {cache_key}")

        return self._provider_instances[cache_key]

    def get_available_providers(
        self,
        in_active_conversation: bool = False,
        current_provider: Optional[str] = None,
    ) -> List[str]:
        """Get providers available in current context"""
        all_providers = list(self._provider_classes.keys())

        if not in_active_conversation:
            return all_providers

        filtered = []
        for name in all_providers:
            if name == current_provider:
                filtered.append(name)
                continue

            provider_instance = self.get_provider(name)
            if provider_instance.provider_type != ProviderType.SERVER_HISTORY:
                filtered.append(name)

        return filtered

    def get_available_models(self, provider_name: str) -> List[str]:
        """Get available models for a provider"""
        if provider_name not in self._provider_classes:
            return []

        provider = self.get_provider(provider_name)
        return provider.get_available_models()

    def get_default_model(self, provider_name: str) -> str:
        """Get default model for a provider"""
        models = self.get_available_models(provider_name)
        return models[0] if models else "auto"
