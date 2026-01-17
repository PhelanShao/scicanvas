"""
Configuration management for Deep Thinking Backend
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml


class ThinkingStrategy(BaseModel):
    """Configuration for a thinking depth strategy"""
    description: str = ""
    model_tier: str = "small"
    max_tokens: int = 4000
    temperature: float = 0.7
    system_prompt_prefix: str = ""


class ProviderModel(BaseModel):
    """Model configuration for a provider"""
    provider: str
    model: str
    priority: int = 1


class ModelTierConfig(BaseModel):
    """Configuration for a model tier"""
    providers: List[ProviderModel] = Field(default_factory=list)


class ProviderSettings(BaseModel):
    """Settings for an LLM provider"""
    base_url: Optional[str] = None
    timeout: int = 300
    max_retries: int = 3


class ModelInfo(BaseModel):
    """Information about a specific model"""
    tier: str = "small"
    context_window: int = 128000
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    
    # Custom base URLs
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    
    # Optional Redis
    redis_url: Optional[str] = None
    
    # Defaults
    default_thinking_depth: str = "standard"
    
    # CORS
    cors_origins: str = "http://localhost:3000"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


class ModelsConfig:
    """Configuration loaded from models.yaml"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.thinking_strategies: Dict[str, ThinkingStrategy] = {}
        self.model_tiers: Dict[str, ModelTierConfig] = {}
        self.provider_settings: Dict[str, ProviderSettings] = {}
        self.model_catalog: Dict[str, Dict[str, ModelInfo]] = {}
        
        # Load config
        if config_path is None:
            # Try default paths
            paths = [
                Path("config/models.yaml"),
                Path(__file__).parent.parent / "config" / "models.yaml",
            ]
            for p in paths:
                if p.exists():
                    config_path = str(p)
                    break
        
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
        else:
            self._load_defaults()
    
    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        # Parse thinking strategies
        for name, data in config.get("thinking_strategies", {}).items():
            self.thinking_strategies[name] = ThinkingStrategy(**data)
        
        # Parse model tiers
        for tier_name, tier_data in config.get("model_tiers", {}).items():
            providers = [
                ProviderModel(**p) for p in tier_data.get("providers", [])
            ]
            self.model_tiers[tier_name] = ModelTierConfig(providers=providers)
        
        # Parse provider settings
        for provider, settings in config.get("provider_settings", {}).items():
            self.provider_settings[provider] = ProviderSettings(**settings)
        
        # Parse model catalog
        for provider, models in config.get("model_catalog", {}).items():
            self.model_catalog[provider] = {}
            for model_name, model_data in models.items():
                self.model_catalog[provider][model_name] = ModelInfo(**model_data)
    
    def _load_defaults(self):
        """Load default configuration when no config file found"""
        # Default thinking strategies
        self.thinking_strategies = {
            "off": ThinkingStrategy(
                description="Direct response",
                model_tier="small",
                max_tokens=2000,
                temperature=0.7,
            ),
            "quick": ThinkingStrategy(
                description="Brief thinking",
                model_tier="small",
                max_tokens=4000,
                temperature=0.5,
                system_prompt_prefix="Think briefly before answering.",
            ),
            "standard": ThinkingStrategy(
                description="Standard thinking",
                model_tier="medium",
                max_tokens=8000,
                temperature=0.3,
                system_prompt_prefix="Think step by step and explain your reasoning clearly.",
            ),
            "deep": ThinkingStrategy(
                description="Deep analysis",
                model_tier="large",
                max_tokens=16000,
                temperature=0.2,
                system_prompt_prefix="Engage in deep, thorough analysis with explicit reasoning steps.",
            ),
        }
        
        # Default model tiers (using OpenAI)
        self.model_tiers = {
            "small": ModelTierConfig(providers=[
                ProviderModel(provider="openai", model="gpt-4o-mini", priority=1),
            ]),
            "medium": ModelTierConfig(providers=[
                ProviderModel(provider="openai", model="gpt-4o", priority=1),
            ]),
            "large": ModelTierConfig(providers=[
                ProviderModel(provider="openai", model="gpt-4o", priority=1),
            ]),
        }
    
    def get_strategy(self, depth: str) -> ThinkingStrategy:
        """Get thinking strategy by depth name"""
        return self.thinking_strategies.get(
            depth, 
            self.thinking_strategies.get("standard", ThinkingStrategy())
        )
    
    def get_tier_providers(self, tier: str) -> List[ProviderModel]:
        """Get providers for a model tier, sorted by priority"""
        tier_config = self.model_tiers.get(tier)
        if not tier_config:
            return []
        return sorted(tier_config.providers, key=lambda x: x.priority)


# Singleton instances
_settings: Optional[Settings] = None
_models_config: Optional[ModelsConfig] = None


def get_settings() -> Settings:
    """Get or create settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_models_config() -> ModelsConfig:
    """Get or create models config singleton"""
    global _models_config
    if _models_config is None:
        _models_config = ModelsConfig()
    return _models_config
