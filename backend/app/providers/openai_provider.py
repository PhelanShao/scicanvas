"""
OpenAI Provider implementation
"""

import time
from typing import AsyncGenerator, Optional
import logging

from openai import AsyncOpenAI

from .base import (
    BaseLLMProvider,
    CompletionRequest,
    CompletionResponse,
    TokenUsage,
    ModelTier,
)


logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
    ):
        super().__init__("openai", api_key, base_url)
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        # Available models
        self.available_models = {
            "gpt-4o-mini": {
                "tier": "small",
                "context_window": 128000,
                "max_tokens": 16384,
                "input_price_per_1k": 0.00015,
                "output_price_per_1k": 0.0006,
            },
            "gpt-4o": {
                "tier": "medium",
                "context_window": 128000,
                "max_tokens": 16384,
                "input_price_per_1k": 0.0025,
                "output_price_per_1k": 0.01,
            },
            "gpt-4-turbo": {
                "tier": "medium",
                "context_window": 128000,
                "max_tokens": 4096,
                "input_price_per_1k": 0.01,
                "output_price_per_1k": 0.03,
            },
            "o1": {
                "tier": "large",
                "context_window": 200000,
                "max_tokens": 100000,
                "input_price_per_1k": 0.015,
                "output_price_per_1k": 0.06,
            },
            "o1-mini": {
                "tier": "medium",
                "context_window": 128000,
                "max_tokens": 65536,
                "input_price_per_1k": 0.003,
                "output_price_per_1k": 0.012,
            },
        }
    
    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for token usage"""
        model_info = self.available_models.get(model, {})
        input_price = model_info.get("input_price_per_1k", 0.001)
        output_price = model_info.get("output_price_per_1k", 0.002)
        return (input_tokens * input_price + output_tokens * output_price) / 1000
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion"""
        start_time = time.time()
        
        model = request.model or self.get_default_model(request.model_tier) or "gpt-4o-mini"
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=request.to_messages_list(),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )
            
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"
            
            # Calculate usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
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
                finish_reason=finish_reason,
                latency_ms=int((time.time() - start_time) * 1000),
            )
            
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise
    
    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion"""
        model = request.model or self.get_default_model(request.model_tier) or "gpt-4o-mini"
        
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=request.to_messages_list(),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    def get_default_model(self, tier: ModelTier) -> Optional[str]:
        """Get default model for tier"""
        tier_models = {
            ModelTier.SMALL: "gpt-4o-mini",
            ModelTier.MEDIUM: "gpt-4o",
            ModelTier.LARGE: "o1",
        }
        return tier_models.get(tier)
