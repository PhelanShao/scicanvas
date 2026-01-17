"""
Anthropic Provider implementation
"""

import time
from typing import AsyncGenerator, Optional
import logging

from anthropic import AsyncAnthropic

from .base import (
    BaseLLMProvider,
    CompletionRequest,
    CompletionResponse,
    TokenUsage,
    ModelTier,
)


logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
    ):
        super().__init__("anthropic", api_key, base_url)
        
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
        )
        
        # Available models
        self.available_models = {
            "claude-3-5-haiku-20241022": {
                "tier": "small",
                "context_window": 200000,
                "max_tokens": 8192,
                "input_price_per_1k": 0.0008,
                "output_price_per_1k": 0.004,
            },
            "claude-3-5-sonnet-20241022": {
                "tier": "medium",
                "context_window": 200000,
                "max_tokens": 8192,
                "input_price_per_1k": 0.003,
                "output_price_per_1k": 0.015,
            },
            "claude-3-opus-20240229": {
                "tier": "large",
                "context_window": 200000,
                "max_tokens": 4096,
                "input_price_per_1k": 0.015,
                "output_price_per_1k": 0.075,
            },
        }
    
    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for token usage"""
        model_info = self.available_models.get(model, {})
        input_price = model_info.get("input_price_per_1k", 0.003)
        output_price = model_info.get("output_price_per_1k", 0.015)
        return (input_tokens * input_price + output_tokens * output_price) / 1000
    
    def _prepare_messages(self, messages: list) -> tuple:
        """Separate system message from other messages for Anthropic API"""
        system_content = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
            else:
                chat_messages.append(msg)
        
        return system_content.strip(), chat_messages
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion"""
        start_time = time.time()
        
        model = request.model or self.get_default_model(request.model_tier) or "claude-3-5-haiku-20241022"
        
        system_content, chat_messages = self._prepare_messages(
            request.to_messages_list()
        )
        
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=request.max_tokens,
                system=system_content if system_content else None,
                messages=chat_messages,
            )
            
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
            
            # Get usage
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            
            return CompletionResponse(
                content=content,
                model=model,
                provider=self.name,
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    estimated_cost=self._calculate_cost(model, input_tokens, output_tokens),
                ),
                finish_reason=response.stop_reason or "stop",
                latency_ms=int((time.time() - start_time) * 1000),
            )
            
        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise
    
    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion"""
        model = request.model or self.get_default_model(request.model_tier) or "claude-3-5-haiku-20241022"
        
        system_content, chat_messages = self._prepare_messages(
            request.to_messages_list()
        )
        
        try:
            async with self.client.messages.stream(
                model=model,
                max_tokens=request.max_tokens,
                system=system_content if system_content else None,
                messages=chat_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise
    
    def get_default_model(self, tier: ModelTier) -> Optional[str]:
        """Get default model for tier"""
        tier_models = {
            ModelTier.SMALL: "claude-3-5-haiku-20241022",
            ModelTier.MEDIUM: "claude-3-5-sonnet-20241022",
            ModelTier.LARGE: "claude-3-5-sonnet-20241022",
        }
        return tier_models.get(tier)
