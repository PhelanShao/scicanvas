"""
DeepSeek Provider implementation (OpenAI-compatible API)
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


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider (OpenAI-compatible)"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
    ):
        super().__init__("deepseek", api_key, base_url)
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        # Available models
        self.available_models = {
            "deepseek-chat": {
                "tier": "small",
                "context_window": 64000,
                "max_tokens": 8192,
                "input_price_per_1k": 0.00014,
                "output_price_per_1k": 0.00028,
            },
            "deepseek-reasoner": {
                "tier": "large",
                "context_window": 64000,
                "max_tokens": 8192,
                "input_price_per_1k": 0.00055,
                "output_price_per_1k": 0.00219,
            },
        }
    
    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for token usage"""
        model_info = self.available_models.get(model, {})
        input_price = model_info.get("input_price_per_1k", 0.00014)
        output_price = model_info.get("output_price_per_1k", 0.00028)
        return (input_tokens * input_price + output_tokens * output_price) / 1000
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion"""
        start_time = time.time()
        
        model = request.model or self.get_default_model(request.model_tier) or "deepseek-chat"
        
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
            
            # Handle reasoning content if present
            thinking_content = None
            if hasattr(response.choices[0].message, "reasoning_content"):
                thinking_content = response.choices[0].message.reasoning_content
            
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
                thinking_content=thinking_content,
            )
            
        except Exception as e:
            logger.error(f"DeepSeek completion failed: {e}")
            raise
    
    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion"""
        model = request.model or self.get_default_model(request.model_tier) or "deepseek-chat"
        
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
            logger.error(f"DeepSeek streaming failed: {e}")
            raise
    
    def get_default_model(self, tier: ModelTier) -> Optional[str]:
        """Get default model for tier"""
        tier_models = {
            ModelTier.SMALL: "deepseek-chat",
            ModelTier.MEDIUM: "deepseek-chat",
            ModelTier.LARGE: "deepseek-reasoner",
        }
        return tier_models.get(tier)
