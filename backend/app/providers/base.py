"""
Base classes for LLM providers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from enum import Enum


class ModelTier(str, Enum):
    """Model tier for routing requests"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class Message:
    """Chat message"""
    role: str  # system, user, assistant
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class TokenUsage:
    """Token usage statistics"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost=self.estimated_cost + other.estimated_cost,
        )


@dataclass
class CompletionRequest:
    """Request for chat completion"""
    messages: List[Message]
    model: Optional[str] = None
    model_tier: ModelTier = ModelTier.SMALL
    temperature: float = 0.7
    max_tokens: int = 4000
    stream: bool = False
    
    # Optional metadata
    session_id: Optional[str] = None
    thinking_depth: str = "standard"
    
    def to_messages_list(self) -> List[Dict[str, str]]:
        """Convert messages to list of dicts"""
        return [m.to_dict() for m in self.messages]


@dataclass 
class CompletionResponse:
    """Response from chat completion"""
    content: str
    model: str
    provider: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    latency_ms: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # For streaming - thinking content if available
    thinking_content: Optional[str] = None
    

class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, name: str, api_key: str, base_url: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.available_models: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion (non-streaming)"""
        pass
    
    @abstractmethod
    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion, yielding text chunks"""
        pass
    
    def is_available(self) -> bool:
        """Check if provider is properly configured"""
        return bool(self.api_key)
    
    def supports_model(self, model: str) -> bool:
        """Check if provider supports a specific model"""
        return model in self.available_models
    
    def get_default_model(self, tier: ModelTier) -> Optional[str]:
        """Get default model for a tier"""
        for model_name, info in self.available_models.items():
            if info.get("tier") == tier.value:
                return model_name
        return None
