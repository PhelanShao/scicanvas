"""
LLM Manager - Orchestrates multiple providers with thinking depth strategies
"""

import logging
from typing import Dict, Optional, List, AsyncGenerator

from .config import get_settings, get_models_config, ThinkingStrategy
from .providers.base import (
    BaseLLMProvider,
    CompletionRequest,
    CompletionResponse,
    Message,
    ModelTier,
)
from .providers.openai_provider import OpenAIProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.deepseek_provider import DeepSeekProvider


logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages LLM providers and routes requests based on thinking depth
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.settings = get_settings()
        self.models_config = get_models_config()
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        settings = self.settings
        
        # OpenAI
        if settings.openai_api_key:
            try:
                self.providers["openai"] = OpenAIProvider(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                )
                logger.info("Initialized OpenAI provider")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Anthropic
        if settings.anthropic_api_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(
                    api_key=settings.anthropic_api_key,
                    base_url=settings.anthropic_base_url,
                )
                logger.info("Initialized Anthropic provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic provider: {e}")
        
        # DeepSeek
        if settings.deepseek_api_key:
            try:
                self.providers["deepseek"] = DeepSeekProvider(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                )
                logger.info("Initialized DeepSeek provider")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepSeek provider: {e}")
        
        if not self.providers:
            logger.warning("No LLM providers configured! Set API keys in environment.")
    
    def is_configured(self) -> bool:
        """Check if at least one provider is configured"""
        return len(self.providers) > 0
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())
    
    def _get_strategy(self, thinking_depth: str) -> ThinkingStrategy:
        """Get thinking strategy configuration"""
        return self.models_config.get_strategy(thinking_depth)
    
    def _select_provider_and_model(
        self, 
        tier: str,
        preferred_provider: Optional[str] = None,
    ) -> tuple[BaseLLMProvider, str]:
        """Select the best available provider and model for a tier"""
        
        # If preferred provider is specified and available
        if preferred_provider and preferred_provider in self.providers:
            provider = self.providers[preferred_provider]
            model = provider.get_default_model(ModelTier(tier))
            if model:
                return provider, model
        
        # Try providers from config priority
        tier_providers = self.models_config.get_tier_providers(tier)
        for provider_model in tier_providers:
            if provider_model.provider in self.providers:
                provider = self.providers[provider_model.provider]
                if provider.supports_model(provider_model.model):
                    return provider, provider_model.model
                # Fallback to default model for tier
                model = provider.get_default_model(ModelTier(tier))
                if model:
                    return provider, model
        
        # Fallback to first available provider
        if self.providers:
            provider_name = next(iter(self.providers))
            provider = self.providers[provider_name]
            model = provider.get_default_model(ModelTier(tier)) or list(provider.available_models.keys())[0]
            return provider, model
        
        raise ValueError("No LLM providers available")
    
    def _prepare_messages_with_thinking(
        self,
        messages: List[Message],
        strategy: ThinkingStrategy,
    ) -> List[Message]:
        """Prepare messages with thinking depth system prompt"""
        if not strategy.system_prompt_prefix:
            return messages
        
        # Find or create system message
        prepared = []
        has_system = False
        
        for msg in messages:
            if msg.role == "system":
                # Prepend thinking instruction to existing system message
                new_content = f"{strategy.system_prompt_prefix}\n\n{msg.content}"
                prepared.append(Message(role="system", content=new_content))
                has_system = True
            else:
                prepared.append(msg)
        
        # Add system message if none exists
        if not has_system and strategy.system_prompt_prefix:
            prepared.insert(0, Message(role="system", content=strategy.system_prompt_prefix))
        
        return prepared
    
    async def complete(
        self,
        messages: List[Message],
        thinking_depth: str = "standard",
        preferred_provider: Optional[str] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Generate a completion with specified thinking depth
        
        Args:
            messages: List of chat messages
            thinking_depth: One of "off", "quick", "standard", "deep"
            preferred_provider: Optional preferred provider name
        """
        if not self.is_configured():
            raise ValueError("No LLM providers configured")
        
        # Get strategy for thinking depth
        strategy = self._get_strategy(thinking_depth)
        
        # Select provider and model
        provider, model = self._select_provider_and_model(
            strategy.model_tier,
            preferred_provider,
        )
        
        # Prepare messages with thinking prompt
        prepared_messages = self._prepare_messages_with_thinking(messages, strategy)
        
        # Create request
        request = CompletionRequest(
            messages=prepared_messages,
            model=model,
            model_tier=ModelTier(strategy.model_tier),
            temperature=strategy.temperature,
            max_tokens=strategy.max_tokens,
            thinking_depth=thinking_depth,
            **kwargs,
        )
        
        logger.info(
            f"Completing with {provider.name}/{model}, "
            f"depth={thinking_depth}, tier={strategy.model_tier}"
        )
        
        return await provider.complete(request)
    
    async def stream_complete(
        self,
        messages: List[Message],
        thinking_depth: str = "standard",
        preferred_provider: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion with specified thinking depth
        
        Yields text chunks as they are generated
        """
        if not self.is_configured():
            raise ValueError("No LLM providers configured")
        
        # Get strategy for thinking depth
        strategy = self._get_strategy(thinking_depth)
        
        # Select provider and model
        provider, model = self._select_provider_and_model(
            strategy.model_tier,
            preferred_provider,
        )
        
        # Prepare messages with thinking prompt
        prepared_messages = self._prepare_messages_with_thinking(messages, strategy)
        
        # Create request
        request = CompletionRequest(
            messages=prepared_messages,
            model=model,
            model_tier=ModelTier(strategy.model_tier),
            temperature=strategy.temperature,
            max_tokens=strategy.max_tokens,
            thinking_depth=thinking_depth,
            stream=True,
            **kwargs,
        )
        
        logger.info(
            f"Streaming with {provider.name}/{model}, "
            f"depth={thinking_depth}, tier={strategy.model_tier}"
        )
        
        async for chunk in provider.stream_complete(request):
            yield chunk


# Singleton instance
_manager_instance: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get or create the singleton LLM manager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = LLMManager()
    return _manager_instance
