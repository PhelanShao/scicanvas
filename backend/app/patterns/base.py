"""
Base classes and interfaces for thinking patterns.
Reasoning infrastructure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time
import uuid


class ThinkingDepth(Enum):
    """Thinking depth levels mapped to model tiers"""
    OFF = "off"
    QUICK = "quick"       # small tier, fast responses
    STANDARD = "standard" # medium tier, balanced
    DEEP = "deep"        # large tier, thorough analysis


@dataclass
class ThinkingStep:
    """A single step in the thinking process"""
    step_id: str
    step_type: str  # "reasoning", "action", "observation", "thought", "evaluation"
    content: str
    confidence: float = 0.5
    tokens_used: int = 0
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, step_type: str, content: str, **kwargs) -> "ThinkingStep":
        return cls(
            step_id=str(uuid.uuid4())[:8],
            step_type=step_type,
            content=content,
            **kwargs
        )


@dataclass
class AgentResult:
    """Result from an individual agent execution"""
    agent_id: str
    response: str
    success: bool = True
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_used: str = ""
    provider: str = ""
    duration_ms: int = 0
    tools_used: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternConfig:
    """Base configuration for all patterns"""
    model_tier: str = "medium"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 60
    enable_streaming: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_tier": self.model_tier,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "enable_streaming": self.enable_streaming,
        }


@dataclass
class PatternResult:
    """Result from executing a thinking pattern"""
    pattern_name: str
    final_response: str
    thinking_steps: List[ThinkingStep] = field(default_factory=list)
    agent_results: List[AgentResult] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: int = 0
    confidence: float = 0.5
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: ThinkingStep):
        self.thinking_steps.append(step)
        self.total_tokens += step.tokens_used
        
    def add_agent_result(self, result: AgentResult):
        self.agent_results.append(result)
        self.total_tokens += result.tokens_used


class BasePattern(ABC):
    """Abstract base class for all thinking patterns"""
    
    def __init__(self, llm_manager, config: Optional[PatternConfig] = None):
        self.llm_manager = llm_manager
        self.config = config or PatternConfig()
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[str]] = None,
        **kwargs
    ) -> PatternResult:
        """Execute the thinking pattern"""
        pass
    
    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_tier: Optional[str] = None,
        **kwargs
    ) -> AgentResult:
        """Call LLM through the manager"""
        start_time = time.time()
        tier = model_tier or self.config.model_tier
        
        try:
            response = await self.llm_manager.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                model_tier=tier,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return AgentResult(
                agent_id=kwargs.get("agent_id", "default"),
                response=response.get("content", ""),
                success=True,
                tokens_used=response.get("usage", {}).get("total_tokens", 0),
                input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
                output_tokens=response.get("usage", {}).get("completion_tokens", 0),
                model_used=response.get("model", ""),
                provider=response.get("provider", ""),
                duration_ms=duration_ms,
            )
        except Exception as e:
            return AgentResult(
                agent_id=kwargs.get("agent_id", "default"),
                response="",
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
            )
