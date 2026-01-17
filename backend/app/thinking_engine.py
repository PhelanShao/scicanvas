"""
Deep Thinking Engine - Main orchestration for multi-pattern reasoning
Implementation for intelligent thinking mode selection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator
from enum import Enum
import asyncio
import time

from .patterns.base import (
    PatternResult, ThinkingStep, AgentResult,
    ThinkingDepth, PatternConfig
)
from .patterns.react import ReActPattern, ReActConfig
from .patterns.chain_of_thought import ChainOfThoughtPattern, CoTConfig
from .patterns.tree_of_thoughts import TreeOfThoughtsPattern, ToTConfig
from .patterns.debate import DebatePattern, DebateConfig
from .patterns.reflection import ReflectionPattern, ReflectionConfig
from .patterns.parallel import ParallelExecutor, ParallelConfig, ExecutionStrategy
from .patterns.decomposition import TaskDecomposer, DecompositionConfig, decompose_task
from .patterns.synthesis import ResultSynthesizer, SynthesisConfig, synthesize_results


class ThinkingMode(Enum):
    """Available thinking modes"""
    DIRECT = "direct"           # Simple direct response
    REACT = "react"             # Reason-Act-Observe loop
    CHAIN_OF_THOUGHT = "cot"    # Step-by-step reasoning
    TREE_OF_THOUGHTS = "tot"    # Branching exploration
    DEBATE = "debate"           # Multi-perspective debate
    AUTO = "auto"               # Automatic mode selection


@dataclass
class ThinkingConfig:
    """Configuration for the thinking engine"""
    # General settings
    default_depth: ThinkingDepth = ThinkingDepth.STANDARD
    default_mode: ThinkingMode = ThinkingMode.AUTO
    enable_reflection: bool = True
    enable_decomposition: bool = True
    
    # Model tier mappings
    depth_to_tier: Dict[ThinkingDepth, str] = field(default_factory=lambda: {
        ThinkingDepth.OFF: "small",
        ThinkingDepth.QUICK: "small",
        ThinkingDepth.STANDARD: "medium",
        ThinkingDepth.DEEP: "large"
    })
    
    # Mode configurations
    react_config: Optional[ReActConfig] = None
    cot_config: Optional[CoTConfig] = None
    tot_config: Optional[ToTConfig] = None
    debate_config: Optional[DebateConfig] = None
    reflection_config: Optional[ReflectionConfig] = None
    
    # Execution settings
    max_parallel_agents: int = 5
    timeout_seconds: int = 120
    
    # Tools
    available_tools: List[str] = field(default_factory=list)


@dataclass
class ThinkingRequest:
    """Request for thinking engine"""
    query: str
    mode: ThinkingMode = ThinkingMode.AUTO
    depth: ThinkingDepth = ThinkingDepth.STANDARD
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[str]] = None
    stream: bool = False
    session_id: Optional[str] = None


@dataclass
class ThinkingResponse:
    """Response from thinking engine"""
    content: str
    thinking_steps: List[ThinkingStep] = field(default_factory=list)
    mode_used: ThinkingMode = ThinkingMode.DIRECT
    depth_used: ThinkingDepth = ThinkingDepth.STANDARD
    total_tokens: int = 0
    model_used: str = ""
    provider: str = ""
    confidence: float = 0.5
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThinkingEngine:
    """
    Main orchestration engine for deep thinking.
    
    Provides:
    - Automatic mode selection based on query complexity
    - Multiple reasoning patterns (ReAct, CoT, ToT, Debate)
    - Task decomposition for complex queries
    - Result synthesis from multiple agents
    - Reflection for quality improvement
    """
    
    def __init__(self, llm_manager, config: Optional[ThinkingConfig] = None):
        self.llm_manager = llm_manager
        self.config = config or ThinkingConfig()
        
        # Initialize pattern instances
        self._init_patterns()
        
    def _init_patterns(self):
        """Initialize all reasoning patterns"""
        # Get model tier for patterns
        tier = self.config.depth_to_tier.get(
            self.config.default_depth,
            "medium"
        )
        
        # Base config for all patterns
        base_config = PatternConfig(model_tier=tier)
        
        # ReAct
        react_config = self.config.react_config or ReActConfig(
            model_tier=tier,
            available_tools=self.config.available_tools
        )
        self.react = ReActPattern(self.llm_manager, react_config)
        
        # Chain of Thought
        cot_config = self.config.cot_config or CoTConfig(model_tier=tier)
        self.cot = ChainOfThoughtPattern(self.llm_manager, cot_config)
        
        # Tree of Thoughts
        tot_config = self.config.tot_config or ToTConfig(model_tier=tier)
        self.tot = TreeOfThoughtsPattern(self.llm_manager, tot_config)
        
        # Debate
        debate_config = self.config.debate_config or DebateConfig(model_tier=tier)
        self.debate = DebatePattern(self.llm_manager, debate_config)
        
        # Reflection
        reflection_config = self.config.reflection_config or ReflectionConfig()
        self.reflection = ReflectionPattern(self.llm_manager, reflection_config)
        
        # Decomposition
        self.decomposer = TaskDecomposer(
            self.llm_manager,
            DecompositionConfig(available_tools=self.config.available_tools)
        )
        
        # Synthesis
        self.synthesizer = ResultSynthesizer(self.llm_manager)
        
        # Parallel executor
        self.parallel_executor = ParallelExecutor(ParallelConfig(
            max_concurrency=self.config.max_parallel_agents
        ))
    
    async def think(self, request: ThinkingRequest) -> ThinkingResponse:
        """
        Main entry point for thinking.
        
        Args:
            request: ThinkingRequest with query and settings
            
        Returns:
            ThinkingResponse with result and metadata
        """
        start_time = time.time()
        
        # Determine model tier from depth
        tier = self.config.depth_to_tier.get(request.depth, "medium")
        
        # Update pattern configs with correct tier
        self._update_pattern_tiers(tier)
        
        # Select thinking mode
        mode = request.mode
        if mode == ThinkingMode.AUTO:
            mode = await self._select_mode(request.query, request.context)
        
        # Handle OFF depth - direct simple response
        if request.depth == ThinkingDepth.OFF:
            return await self._direct_response(request, start_time)
        
        # Execute the appropriate pattern
        result = await self._execute_pattern(mode, request)
        
        # Apply reflection if enabled and not in quick mode
        if self.config.enable_reflection and request.depth != ThinkingDepth.QUICK:
            result = await self._apply_reflection(request.query, result)
        
        # Build response
        response = ThinkingResponse(
            content=result.final_response,
            thinking_steps=result.thinking_steps,
            mode_used=mode,
            depth_used=request.depth,
            total_tokens=result.total_tokens,
            confidence=result.confidence,
            duration_ms=int((time.time() - start_time) * 1000),
            metadata={
                "pattern": result.pattern_name,
                "model_tier": tier,
                **result.metadata
            }
        )
        
        # Extract model info from agent results
        for agent_result in result.agent_results:
            if agent_result.model_used:
                response.model_used = agent_result.model_used
                response.provider = agent_result.provider
                break
        
        return response
    
    async def think_stream(
        self,
        request: ThinkingRequest
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming version of think.
        
        Yields events as thinking progresses.
        """
        start_time = time.time()
        
        # Determine tier
        tier = self.config.depth_to_tier.get(request.depth, "medium")
        self._update_pattern_tiers(tier)
        
        # Select mode
        mode = request.mode
        if mode == ThinkingMode.AUTO:
            mode = await self._select_mode(request.query, request.context)
            yield {
                "type": "mode_selected",
                "mode": mode.value,
                "depth": request.depth.value
            }
        
        # Handle OFF depth
        if request.depth == ThinkingDepth.OFF:
            result = await self._direct_response(request, start_time)
            yield {
                "type": "content",
                "content": result.content,
                "done": True
            }
            return
        
        # Yield thinking start
        yield {
            "type": "thinking_start",
            "mode": mode.value,
            "depth": request.depth.value
        }
        
        # Execute pattern with step callbacks
        result = await self._execute_pattern(mode, request)
        
        # Yield thinking steps
        for step in result.thinking_steps:
            yield {
                "type": "thinking_step",
                "step_type": step.step_type,
                "content": step.content,
                "confidence": step.confidence
            }
        
        # Yield final content
        yield {
            "type": "content",
            "content": result.final_response,
            "total_tokens": result.total_tokens,
            "confidence": result.confidence,
            "duration_ms": int((time.time() - start_time) * 1000),
            "done": True
        }
    
    async def _select_mode(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> ThinkingMode:
        """Automatically select the best thinking mode for the query"""
        # Analyze complexity
        complexity = await self.decomposer.analyze_complexity(query)
        
        # Check for specific indicators in query
        query_lower = query.lower()
        
        # Debate indicators
        if any(w in query_lower for w in ["compare", "versus", "vs", "pros and cons", "debate"]):
            return ThinkingMode.DEBATE
        
        # Tree of thoughts indicators (exploratory, multiple solutions)
        if any(w in query_lower for w in ["explore", "alternatives", "different approaches", "creative"]):
            return ThinkingMode.TREE_OF_THOUGHTS
        
        # ReAct indicators (needs tools/actions)
        if any(w in query_lower for w in ["search", "find", "look up", "calculate", "research"]):
            return ThinkingMode.REACT
        
        # Select based on complexity
        if complexity < 0.3:
            return ThinkingMode.DIRECT
        elif complexity < 0.5:
            return ThinkingMode.CHAIN_OF_THOUGHT
        elif complexity < 0.7:
            return ThinkingMode.REACT
        else:
            return ThinkingMode.TREE_OF_THOUGHTS
    
    async def _execute_pattern(
        self,
        mode: ThinkingMode,
        request: ThinkingRequest
    ) -> PatternResult:
        """Execute the selected thinking pattern"""
        context = request.context or {}
        history = request.history or []
        
        if mode == ThinkingMode.DIRECT:
            return await self._direct_pattern(request)
        elif mode == ThinkingMode.REACT:
            return await self.react.execute(request.query, context, history)
        elif mode == ThinkingMode.CHAIN_OF_THOUGHT:
            return await self.cot.execute(request.query, context, history)
        elif mode == ThinkingMode.TREE_OF_THOUGHTS:
            return await self.tot.execute(request.query, context, history)
        elif mode == ThinkingMode.DEBATE:
            return await self.debate.execute(request.query, context, history)
        else:
            return await self.cot.execute(request.query, context, history)
    
    async def _direct_pattern(self, request: ThinkingRequest) -> PatternResult:
        """Simple direct response without complex reasoning"""
        result = PatternResult(
            pattern_name="Direct",
            final_response=""
        )
        
        response = await self.llm_manager.complete(
            prompt=request.query,
            system_prompt="You are a helpful assistant. Provide clear, accurate responses.",
            model_tier=self.config.depth_to_tier.get(request.depth, "medium")
        )
        
        result.final_response = response.get("content", "")
        result.total_tokens = response.get("usage", {}).get("total_tokens", 0)
        result.confidence = 0.7
        result.agent_results.append(AgentResult(
            agent_id="direct",
            response=result.final_response,
            tokens_used=result.total_tokens,
            model_used=response.get("model", ""),
            provider=response.get("provider", "")
        ))
        
        return result
    
    async def _direct_response(
        self,
        request: ThinkingRequest,
        start_time: float
    ) -> ThinkingResponse:
        """Generate a direct response without deep thinking"""
        response = await self.llm_manager.complete(
            prompt=request.query,
            model_tier="small"
        )
        
        return ThinkingResponse(
            content=response.get("content", ""),
            mode_used=ThinkingMode.DIRECT,
            depth_used=ThinkingDepth.OFF,
            total_tokens=response.get("usage", {}).get("total_tokens", 0),
            model_used=response.get("model", ""),
            provider=response.get("provider", ""),
            confidence=0.7,
            duration_ms=int((time.time() - start_time) * 1000)
        )
    
    async def _apply_reflection(
        self,
        query: str,
        result: PatternResult
    ) -> PatternResult:
        """Apply reflection to improve the result"""
        if not result.final_response:
            return result
        
        reflection_result = await self.reflection.execute(
            query=query,
            context={"_initial_response": result.final_response}
        )
        
        # Update result with reflection improvements
        if reflection_result.success and reflection_result.confidence > result.confidence:
            result.final_response = reflection_result.final_response
            result.confidence = reflection_result.confidence
            result.thinking_steps.extend(reflection_result.thinking_steps)
            result.total_tokens += reflection_result.total_tokens
            result.metadata["reflection_applied"] = True
        
        return result
    
    def _update_pattern_tiers(self, tier: str):
        """Update all patterns to use the specified tier"""
        self.react.config.model_tier = tier
        self.cot.config.model_tier = tier
        self.tot.config.model_tier = tier
        self.debate.config.model_tier = tier
        self.reflection.config.model_tier = tier
        self.decomposer.config.model_tier = tier
        self.synthesizer.config.model_tier = tier


# Convenience functions for direct use

async def deep_think(
    llm_manager,
    query: str,
    depth: str = "standard",
    mode: str = "auto",
    context: Optional[Dict[str, Any]] = None
) -> ThinkingResponse:
    """
    Convenience function for deep thinking.
    
    Args:
        llm_manager: LLM manager instance
        query: The question or task
        depth: "off", "quick", "standard", or "deep"
        mode: "auto", "react", "cot", "tot", or "debate"
        context: Optional context dictionary
        
    Returns:
        ThinkingResponse with result
    """
    # Map string to enums
    depth_map = {
        "off": ThinkingDepth.OFF,
        "quick": ThinkingDepth.QUICK,
        "standard": ThinkingDepth.STANDARD,
        "deep": ThinkingDepth.DEEP
    }
    
    mode_map = {
        "auto": ThinkingMode.AUTO,
        "direct": ThinkingMode.DIRECT,
        "react": ThinkingMode.REACT,
        "cot": ThinkingMode.CHAIN_OF_THOUGHT,
        "chain_of_thought": ThinkingMode.CHAIN_OF_THOUGHT,
        "tot": ThinkingMode.TREE_OF_THOUGHTS,
        "tree_of_thoughts": ThinkingMode.TREE_OF_THOUGHTS,
        "debate": ThinkingMode.DEBATE
    }
    
    engine = ThinkingEngine(llm_manager)
    
    request = ThinkingRequest(
        query=query,
        mode=mode_map.get(mode.lower(), ThinkingMode.AUTO),
        depth=depth_map.get(depth.lower(), ThinkingDepth.STANDARD),
        context=context
    )
    
    return await engine.think(request)
