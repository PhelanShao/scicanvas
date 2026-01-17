"""
Deep Thinking Backend - Main Application Entry Point

A complete LLM orchestration service with multi-agent reasoning.
Supports multiple thinking modes: ReAct, Chain of Thought, Tree of Thoughts, Debate.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import get_settings
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.thinking import router as thinking_router
from app.llm_manager import get_llm_manager
from app.tools.executor import initialize_tools


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Deep Thinking Backend...")
    
    # Initialize LLM manager (validates provider configurations)
    manager = get_llm_manager()
    if manager.is_configured():
        logger.info(f"Initialized providers: {manager.get_available_providers()}")
    else:
        logger.warning("No LLM providers configured! Set API keys in environment.")
    
    # Initialize tool system
    logger.info("Initializing tool system...")
    initialize_tools()
    logger.info("Tool system initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Deep Thinking Backend...")


# Create FastAPI application
app = FastAPI(
    title="Deep Thinking Backend",
    description="""
A complete LLM orchestration service with multi-agent reasoning.

## Thinking Depths

- **off**: Direct response without extended reasoning (fastest, cheapest)
- **quick**: Brief thinking process for simple queries (small model)
- **standard**: Step-by-step reasoning with clear explanations (medium model)
- **deep**: Comprehensive analysis with multiple perspectives (large model)

## Thinking Modes

- **auto**: Automatically select the best mode based on query complexity
- **react**: Reason-Act-Observe loop (good for tasks needing tools/research)
- **cot**: Chain of Thought step-by-step reasoning
- **tot**: Tree of Thoughts exploratory reasoning
- **debate**: Multi-agent debate for comparing viewpoints

## Features

- Multi-agent reasoning patterns (ReAct, CoT, ToT, Debate)
- Task decomposition for complex queries
- Result synthesis from multiple agents
- Reflection for quality improvement
- Parallel agent execution
- Tool integration (web search, calculator, code execution)

## Supported Providers

- OpenAI (gpt-4o-mini, gpt-4o, o1)
- Anthropic (claude-3-5-haiku, claude-3-5-sonnet)
- DeepSeek (deepseek-chat, deepseek-reasoner)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(thinking_router, prefix="/api")

# OpenAI-compatible endpoint at root level
app.include_router(chat_router, prefix="", tags=["openai-compatible"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Deep Thinking Backend",
        "version": "1.0.0",
        "description": "LLM orchestration with multi-agent reasoning",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "chat": "/api/chat",
            "stream": "/api/chat/stream",
            "think": "/api/thinking/think",
            "think_stream": "/api/thinking/think/stream",
            "decompose": "/api/thinking/decompose",
            "synthesize": "/api/thinking/synthesize",
            "openai_compatible": "/v1/chat/completions",
        },
        "thinking_depths": ["off", "quick", "standard", "deep"],
        "thinking_modes": ["auto", "react", "cot", "tot", "debate"],
        "capabilities": {
            "multi_agent": True,
            "reasoning_patterns": ["react", "cot", "tot", "debate", "reflection"],
            "task_decomposition": True,
            "result_synthesis": True,
            "parallel_execution": True,
            "tool_support": True,
        }
    }


if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
