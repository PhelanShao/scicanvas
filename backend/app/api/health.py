"""
Health check endpoints
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from ..llm_manager import get_llm_manager
from ..config import get_settings


router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    providers: List[str]
    configured: bool


class ProvidersResponse(BaseModel):
    providers: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    manager = get_llm_manager()
    
    return HealthResponse(
        status="healthy" if manager.is_configured() else "degraded",
        version="0.1.0",
        providers=manager.get_available_providers(),
        configured=manager.is_configured(),
    )


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers():
    """List available LLM providers and their models"""
    manager = get_llm_manager()
    
    providers_info = {}
    for name, provider in manager.providers.items():
        providers_info[name] = {
            "available": provider.is_available(),
            "models": list(provider.available_models.keys()),
        }
    
    return ProvidersResponse(providers=providers_info)


@router.get("/config")
async def get_config():
    """Get current configuration (non-sensitive)"""
    settings = get_settings()
    
    return {
        "default_thinking_depth": settings.default_thinking_depth,
        "thinking_depths": ["off", "quick", "standard", "deep"],
        "cors_origins": settings.get_cors_origins(),
    }
