"""
Tool executor for managing tool invocations.
"""

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool, ToolResult, ToolRegistry, get_tool_registry


class ToolExecutor:
    """
    Tool executor with safety controls and parallel execution support.
    """
    
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        max_concurrent: int = 5,
        default_timeout: int = 30
    ):
        self.registry = registry or get_tool_registry()
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> ToolResult:
        """Execute a single tool"""
        tool = self.registry.get(tool_name)
        
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Tool not found: {tool_name}"
            )
        
        # Validate parameters
        validation_error = tool.validate_parameters(**parameters)
        if validation_error:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=validation_error
            )
        
        # Execute with semaphore for concurrency control
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    tool.execute(**parameters),
                    timeout=timeout or self.default_timeout
                )
                return result
            except asyncio.TimeoutError:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error=f"Tool execution timed out after {timeout or self.default_timeout}s"
                )
            except Exception as e:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output=None,
                    error=str(e)
                )
    
    async def execute_multiple(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """Execute multiple tools in parallel"""
        tasks = []
        
        for call in tool_calls:
            tool_name = call.get("tool") or call.get("name")
            parameters = call.get("parameters", {})
            timeout = call.get("timeout")
            
            task = self.execute(tool_name, parameters, timeout)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parse a tool call from text.
        Supports formats:
        - tool_name(param1=value1, param2=value2)
        - tool_name("value") for single parameter
        """
        # Pattern: tool_name(...)
        match = re.match(r'(\w+)\s*\((.*)\)', text.strip())
        if not match:
            return None
        
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        
        if not args_str:
            return tool_name, {}
        
        # Try to parse as kwargs
        params = {}
        
        # Simple single value case
        if '=' not in args_str:
            # Get first parameter name from tool definition
            tool = self.registry.get(tool_name)
            if tool and tool.parameters:
                first_param = tool.parameters[0].name
                # Remove quotes if present
                value = args_str.strip('"\'')
                params[first_param] = value
        else:
            # Parse key=value pairs
            # Simple parser (doesn't handle all edge cases)
            pairs = re.findall(r'(\w+)\s*=\s*([^,]+)', args_str)
            for key, value in pairs:
                value = value.strip().strip('"\'')
                # Try to convert to appropriate type
                try:
                    if value.lower() == 'true':
                        params[key] = True
                    elif value.lower() == 'false':
                        params[key] = False
                    elif '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
        
        return tool_name, params
    
    async def execute_from_text(self, text: str) -> Optional[ToolResult]:
        """Execute a tool call parsed from text"""
        parsed = self.parse_tool_call(text)
        if not parsed:
            return None
        
        tool_name, params = parsed
        return await self.execute(tool_name, params)
    
    def get_available_tools(self, exclude_dangerous: bool = False) -> List[str]:
        """Get list of available tool names"""
        return self.registry.list_tools(exclude_dangerous=exclude_dangerous)
    
    def get_tool_descriptions(self, tool_names: Optional[List[str]] = None) -> str:
        """Get formatted descriptions for tools"""
        return self.registry.get_tool_descriptions(tool_names)


# Global executor instance
_global_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    """Get the global tool executor"""
    global _global_executor
    if _global_executor is None:
        _global_executor = ToolExecutor()
    return _global_executor


def initialize_tools():
    """Initialize the tool system with built-in tools"""
    from .builtin import register_builtin_tools
    register_builtin_tools()
