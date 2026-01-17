"""
Deep Thinking API endpoints
Complete API for multi-agent reasoning.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import json
import asyncio

from ..llm_manager import get_llm_manager, LLMManager
from ..thinking_engine import (
    ThinkingEngine, ThinkingConfig, ThinkingRequest,
    ThinkingMode, ThinkingResponse
)
from ..patterns.base import ThinkingDepth

router = APIRouter(prefix="/thinking", tags=["thinking"])


# Request/Response Models

class ThinkRequest(BaseModel):
    """Request for thinking endpoint"""
    query: str = Field(..., description="The question or task to think about")
    depth: str = Field(
        default="standard",
        description="Thinking depth: off, quick, standard, deep"
    )
    mode: str = Field(
        default="auto",
        description="Thinking mode: auto, react, cot, tot, debate"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for the query"
    )
    history: Optional[List[str]] = Field(
        default=None,
        description="Conversation history"
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming response"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context persistence"
    )


class ThinkingStepResponse(BaseModel):
    """A single thinking step"""
    step_type: str
    content: str
    confidence: float = 0.5
    metadata: Dict[str, Any] = {}


class ThinkResponse(BaseModel):
    """Response from thinking endpoint"""
    content: str
    thinking_steps: List[ThinkingStepResponse] = []
    mode_used: str
    depth_used: str
    total_tokens: int = 0
    model_used: str = ""
    provider: str = ""
    confidence: float = 0.5
    duration_ms: int = 0
    metadata: Dict[str, Any] = {}


class DecomposeRequest(BaseModel):
    """Request for task decomposition"""
    query: str
    context: Optional[Dict[str, Any]] = None
    available_tools: Optional[List[str]] = None


class DecomposeResponse(BaseModel):
    """Response from decomposition"""
    mode: str
    complexity_score: float
    subtasks: List[Dict[str, Any]]
    execution_strategy: str
    cognitive_strategy: str
    total_estimated_tokens: int


class SynthesizeRequest(BaseModel):
    """Request for result synthesis"""
    query: str
    results: List[Dict[str, Any]]
    style: str = "comprehensive"
    context: Optional[Dict[str, Any]] = None


class SynthesizeResponse(BaseModel):
    """Response from synthesis"""
    content: str
    total_tokens: int
    confidence: float


# Helper functions

def _get_depth(depth_str: str) -> ThinkingDepth:
    """Convert string to ThinkingDepth enum"""
    mapping = {
        "off": ThinkingDepth.OFF,
        "quick": ThinkingDepth.QUICK,
        "standard": ThinkingDepth.STANDARD,
        "deep": ThinkingDepth.DEEP
    }
    return mapping.get(depth_str.lower(), ThinkingDepth.STANDARD)


def _get_mode(mode_str: str) -> ThinkingMode:
    """Convert string to ThinkingMode enum"""
    mapping = {
        "auto": ThinkingMode.AUTO,
        "direct": ThinkingMode.DIRECT,
        "react": ThinkingMode.REACT,
        "cot": ThinkingMode.CHAIN_OF_THOUGHT,
        "chain_of_thought": ThinkingMode.CHAIN_OF_THOUGHT,
        "tot": ThinkingMode.TREE_OF_THOUGHTS,
        "tree_of_thoughts": ThinkingMode.TREE_OF_THOUGHTS,
        "debate": ThinkingMode.DEBATE
    }
    return mapping.get(mode_str.lower(), ThinkingMode.AUTO)


def _response_to_dict(response: ThinkingResponse) -> ThinkResponse:
    """Convert ThinkingResponse to API response"""
    return ThinkResponse(
        content=response.content,
        thinking_steps=[
            ThinkingStepResponse(
                step_type=step.step_type,
                content=step.content,
                confidence=step.confidence,
                metadata=step.metadata
            )
            for step in response.thinking_steps
        ],
        mode_used=response.mode_used.value,
        depth_used=response.depth_used.value,
        total_tokens=response.total_tokens,
        model_used=response.model_used,
        provider=response.provider,
        confidence=response.confidence,
        duration_ms=response.duration_ms,
        metadata=response.metadata
    )


# Endpoints

@router.post("/think", response_model=ThinkResponse)
async def think(
    request: ThinkRequest,
    llm_manager: LLMManager = Depends(get_llm_manager)
):
    """
    Main thinking endpoint - processes a query with configurable depth and mode.
    
    Depths:
    - off: Direct response, no deep thinking
    - quick: Fast response with minimal reasoning (small model)
    - standard: Balanced reasoning (medium model)
    - deep: Thorough analysis with multiple perspectives (large model)
    
    Modes:
    - auto: Automatically select best mode
    - react: Reason-Act-Observe loop (good for tasks needing tools)
    - cot: Chain of Thought (step-by-step reasoning)
    - tot: Tree of Thoughts (exploratory, creative tasks)
    - debate: Multi-perspective debate (comparing viewpoints)
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /thinking/think/stream for streaming responses"
        )
    
    engine = ThinkingEngine(llm_manager)
    
    think_request = ThinkingRequest(
        query=request.query,
        mode=_get_mode(request.mode),
        depth=_get_depth(request.depth),
        context=request.context,
        history=request.history,
        session_id=request.session_id
    )
    
    try:
        response = await engine.think(think_request)
        return _response_to_dict(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/think/stream")
async def think_stream(
    request: ThinkRequest,
    llm_manager: LLMManager = Depends(get_llm_manager)
):
    """
    Streaming thinking endpoint - returns Server-Sent Events with thinking progress.
    
    Events:
    - mode_selected: When auto mode selects a thinking mode
    - thinking_start: When thinking begins
    - thinking_step: Each reasoning step
    - content: Final response content
    """
    engine = ThinkingEngine(llm_manager)
    
    think_request = ThinkingRequest(
        query=request.query,
        mode=_get_mode(request.mode),
        depth=_get_depth(request.depth),
        context=request.context,
        history=request.history,
        stream=True,
        session_id=request.session_id
    )
    
    async def generate():
        try:
            async for event in engine.think_stream(think_request):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/decompose", response_model=DecomposeResponse)
async def decompose_task(
    request: DecomposeRequest,
    llm_manager: LLMManager = Depends(get_llm_manager)
):
    """
    Decompose a complex task into subtasks.
    
    Returns:
    - mode: Complexity level (simple, standard, complex)
    - subtasks: List of subtasks with dependencies
    - execution_strategy: Recommended execution (sequential, parallel, hybrid)
    - cognitive_strategy: Recommended thinking pattern
    """
    from ..patterns.decomposition import TaskDecomposer, DecompositionConfig
    
    decomposer = TaskDecomposer(
        llm_manager,
        DecompositionConfig(
            available_tools=request.available_tools or []
        )
    )
    
    try:
        result = await decomposer.execute(
            query=request.query,
            context=request.context
        )
        
        decomposition = result.metadata.get("decomposition", {})
        
        return DecomposeResponse(
            mode=decomposition.mode if hasattr(decomposition, 'mode') else "standard",
            complexity_score=decomposition.complexity_score if hasattr(decomposition, 'complexity_score') else 0.5,
            subtasks=[
                {
                    "id": st.id,
                    "description": st.description,
                    "dependencies": st.dependencies,
                    "suggested_tools": st.suggested_tools,
                    "estimated_tokens": st.estimated_tokens
                }
                for st in (decomposition.subtasks if hasattr(decomposition, 'subtasks') else [])
            ],
            execution_strategy=decomposition.execution_strategy if hasattr(decomposition, 'execution_strategy') else "sequential",
            cognitive_strategy=decomposition.cognitive_strategy if hasattr(decomposition, 'cognitive_strategy') else "cot",
            total_estimated_tokens=decomposition.total_estimated_tokens if hasattr(decomposition, 'total_estimated_tokens') else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_results(
    request: SynthesizeRequest,
    llm_manager: LLMManager = Depends(get_llm_manager)
):
    """
    Synthesize multiple results into a coherent response.
    
    Styles:
    - comprehensive: Detailed synthesis with all information
    - concise: Brief, focused synthesis
    """
    from ..patterns.synthesis import ResultSynthesizer, SynthesisConfig
    from ..patterns.base import AgentResult
    
    # Convert request results to AgentResult
    agent_results = [
        AgentResult(
            agent_id=r.get("agent_id", f"agent-{i}"),
            response=r.get("response", r.get("content", "")),
            success=r.get("success", True),
            tokens_used=r.get("tokens_used", 0)
        )
        for i, r in enumerate(request.results)
    ]
    
    synthesizer = ResultSynthesizer(
        llm_manager,
        SynthesisConfig(style=request.style)
    )
    
    try:
        result = await synthesizer.execute(
            query=request.query,
            context=request.context,
            agent_results=agent_results
        )
        
        return SynthesizeResponse(
            content=result.final_response,
            total_tokens=result.total_tokens,
            confidence=result.confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modes")
async def list_modes():
    """List available thinking modes with descriptions"""
    return {
        "modes": [
            {
                "id": "auto",
                "name": "Automatic",
                "description": "Automatically selects the best mode based on query complexity"
            },
            {
                "id": "direct",
                "name": "Direct",
                "description": "Simple direct response without complex reasoning"
            },
            {
                "id": "react",
                "name": "ReAct",
                "description": "Reason-Act-Observe loop, good for tasks needing tools or research"
            },
            {
                "id": "cot",
                "name": "Chain of Thought",
                "description": "Step-by-step reasoning for logical problems"
            },
            {
                "id": "tot",
                "name": "Tree of Thoughts",
                "description": "Explores multiple solution paths, good for creative tasks"
            },
            {
                "id": "debate",
                "name": "Debate",
                "description": "Multi-perspective debate for comparing viewpoints"
            }
        ],
        "depths": [
            {
                "id": "off",
                "name": "Off",
                "description": "No deep thinking, direct LLM response"
            },
            {
                "id": "quick",
                "name": "Quick",
                "description": "Fast response with minimal reasoning"
            },
            {
                "id": "standard",
                "name": "Standard",
                "description": "Balanced reasoning with medium depth"
            },
            {
                "id": "deep",
                "name": "Deep",
                "description": "Thorough analysis with multiple iterations"
            }
        ]
    }


@router.get("/status")
async def thinking_status():
    """Get thinking engine status and capabilities"""
    return {
        "status": "ready",
        "capabilities": {
            "patterns": ["react", "cot", "tot", "debate", "reflection"],
            "decomposition": True,
            "synthesis": True,
            "parallel_execution": True,
            "streaming": True
        },
        "version": "1.0.0"
    }
