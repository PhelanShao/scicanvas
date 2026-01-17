"""
Base classes for tool system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
import time


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of a tool"""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    dangerous: bool = False  # Requires confirmation
    cost_per_use: float = 0.0  # For budgeting


class Tool(ABC):
    """Base class for all tools"""
    
    name: str = "base_tool"
    description: str = "Base tool"
    parameters: List[ToolParameter] = []
    dangerous: bool = False
    cost_per_use: float = 0.0
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    def get_definition(self) -> ToolDefinition:
        """Get the tool definition"""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            dangerous=self.dangerous,
            cost_per_use=self.cost_per_use
        )
    
    def validate_parameters(self, **kwargs) -> Optional[str]:
        """Validate parameters, return error message if invalid"""
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return f"Missing required parameter: {param.name}"
        return None


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool"""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str):
        """Unregister a tool by name"""
        if name in self._tools:
            del self._tools[name]
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def list_tools(self, exclude_dangerous: bool = False) -> List[str]:
        """List all registered tool names"""
        if exclude_dangerous:
            return [name for name, tool in self._tools.items() if not tool.dangerous]
        return list(self._tools.keys())
    
    def get_definitions(self, exclude_dangerous: bool = False) -> List[ToolDefinition]:
        """Get definitions for all tools"""
        tools = self._tools.values()
        if exclude_dangerous:
            tools = [t for t in tools if not t.dangerous]
        return [tool.get_definition() for tool in tools]
    
    def get_tool_descriptions(self, tool_names: Optional[List[str]] = None) -> str:
        """Get formatted tool descriptions for prompts"""
        if tool_names:
            tools = [self._tools[n] for n in tool_names if n in self._tools]
        else:
            tools = list(self._tools.values())
        
        descriptions = []
        for tool in tools:
            params_str = ", ".join([
                f"{p.name}: {p.type}" + ("?" if not p.required else "")
                for p in tool.parameters
            ])
            descriptions.append(f"- {tool.name}({params_str}): {tool.description}")
        
        return "\n".join(descriptions) if descriptions else "No tools available"


# Global registry instance
_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry"""
    return _global_registry


def register_tool(tool: Tool):
    """Register a tool to the global registry"""
    _global_registry.register(tool)


def get_tool(name: str) -> Optional[Tool]:
    """Get a tool from the global registry"""
    return _global_registry.get(name)
