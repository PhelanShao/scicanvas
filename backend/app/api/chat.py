"""
Chat API endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import logging

from ..llm_manager import get_llm_manager
from ..providers.base import Message
from ..config import get_settings


router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Chat message in request"""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat completion request"""
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    thinking_depth: str = Field(
        default="standard",
        description="Thinking depth: off, quick, standard, deep"
    )
    stream: bool = Field(default=False, description="Enable streaming response")
    provider: Optional[str] = Field(
        default=None,
        description="Preferred provider (openai, anthropic, deepseek)"
    )
    session_id: Optional[str] = Field(default=None, description="Session ID for tracking")


class UsageInfo(BaseModel):
    """Token usage information"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float


class ChatResponse(BaseModel):
    """Chat completion response"""
    content: str
    model: str
    provider: str
    thinking_depth: str
    usage: UsageInfo
    finish_reason: str
    latency_ms: Optional[int] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    }
)
async def chat_completion(request: ChatRequest):
    """
    Generate a chat completion with configurable thinking depth.
    
    Thinking depths:
    - **off**: Direct response without extended reasoning (fastest)
    - **quick**: Brief thinking process for simple queries
    - **standard**: Step-by-step reasoning (default)
    - **deep**: Comprehensive analysis with multiple perspectives
    """
    manager = get_llm_manager()
    
    if not manager.is_configured():
        raise HTTPException(
            status_code=503,
            detail="No LLM providers configured. Please set API keys."
        )
    
    # Validate thinking depth
    valid_depths = ["off", "quick", "standard", "deep"]
    if request.thinking_depth not in valid_depths:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid thinking_depth. Must be one of: {valid_depths}"
        )
    
    # If streaming requested, redirect
    if request.stream:
        return await chat_stream(request)
    
    try:
        # Convert messages
        messages = [
            Message(role=m.role, content=m.content)
            for m in request.messages
        ]
        
        # Generate completion
        response = await manager.complete(
            messages=messages,
            thinking_depth=request.thinking_depth,
            preferred_provider=request.provider,
            session_id=request.session_id,
        )
        
        return ChatResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            thinking_depth=request.thinking_depth,
            usage=UsageInfo(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                estimated_cost=response.usage.estimated_cost,
            ),
            finish_reason=response.finish_reason,
            latency_ms=response.latency_ms,
        )
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Generate a streaming chat completion.
    
    Returns Server-Sent Events (SSE) with text chunks.
    """
    manager = get_llm_manager()
    
    if not manager.is_configured():
        raise HTTPException(
            status_code=503,
            detail="No LLM providers configured. Please set API keys."
        )
    
    # Convert messages
    messages = [
        Message(role=m.role, content=m.content)
        for m in request.messages
    ]
    
    async def generate_stream():
        """Generate SSE stream"""
        try:
            async for chunk in manager.stream_complete(
                messages=messages,
                thinking_depth=request.thinking_depth,
                preferred_provider=request.provider,
                session_id=request.session_id,
            ):
                # Send as SSE data event
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# OpenAI-compatible endpoint
class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(default="deep-thinking")
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4000
    stream: Optional[bool] = False
    
    # Extension: thinking depth mapping via model name
    # model="deep-thinking-quick" → quick depth
    # model="deep-thinking-standard" → standard depth
    # model="deep-thinking-deep" → deep depth


@router.post("/v1/chat/completions")
async def openai_compatible_chat(request: OpenAIRequest):
    """
    OpenAI-compatible chat completion endpoint.
    
    Use model names to control thinking depth:
    - `deep-thinking-off` or `gpt-4o-mini` → off
    - `deep-thinking-quick` → quick
    - `deep-thinking-standard` or `deep-thinking` → standard
    - `deep-thinking-deep` or `o1` → deep
    """
    manager = get_llm_manager()
    
    if not manager.is_configured():
        raise HTTPException(
            status_code=503,
            detail="No LLM providers configured"
        )
    
    # Map model name to thinking depth
    model_depth_map = {
        "deep-thinking-off": "off",
        "deep-thinking-quick": "quick",
        "deep-thinking-standard": "standard",
        "deep-thinking": "standard",
        "deep-thinking-deep": "deep",
        "gpt-4o-mini": "off",
        "gpt-4o": "standard",
        "o1": "deep",
    }
    
    thinking_depth = model_depth_map.get(request.model, "standard")
    
    # Convert messages
    messages = [
        Message(role=m.role, content=m.content)
        for m in request.messages
    ]
    
    if request.stream:
        async def generate_openai_stream():
            """Generate OpenAI-compatible SSE stream"""
            try:
                async for chunk in manager.stream_complete(
                    messages=messages,
                    thinking_depth=thinking_depth,
                ):
                    data = {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                
                # Final chunk
                data = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"OpenAI stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_openai_stream(),
            media_type="text/event-stream",
        )
    
    # Non-streaming
    try:
        response = await manager.complete(
            messages=messages,
            thinking_depth=thinking_depth,
        )
        
        return {
            "id": "chatcmpl-deep-thinking",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content,
                },
                "finish_reason": response.finish_reason,
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
        
    except Exception as e:
        logger.error(f"OpenAI compatible chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
