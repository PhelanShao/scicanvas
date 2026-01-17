"""
Built-in tools for the Deep Thinking system.
"""

import asyncio
import re
import math
import time
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolParameter


class WebSearchTool(Tool):
    """Web search tool for information retrieval"""
    
    name = "web_search"
    description = "Search the web for information on any topic"
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query",
            required=True
        ),
        ToolParameter(
            name="num_results",
            type="number",
            description="Number of results to return",
            required=False,
            default=5
        )
    ]
    
    def __init__(self, search_provider: Optional[str] = None):
        self.search_provider = search_provider
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute web search"""
        start_time = time.time()
        query = kwargs.get("query", "")
        num_results = kwargs.get("num_results", 5)
        
        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error="Query is required"
            )
        
        try:
            # Placeholder implementation
            # In production, integrate with actual search APIs
            results = [
                {
                    "title": f"Result {i+1} for: {query}",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a sample search result for the query: {query}"
                }
                for i in range(min(num_results, 10))
            ]
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=results,
                duration_ms=int((time.time() - start_time) * 1000),
                metadata={"query": query, "num_results": len(results)}
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )


class CalculatorTool(Tool):
    """Calculator for mathematical expressions"""
    
    name = "calculator"
    description = "Evaluate mathematical expressions safely"
    parameters = [
        ToolParameter(
            name="expression",
            type="string",
            description="Mathematical expression to evaluate (e.g., '2 + 2 * 3')",
            required=True
        )
    ]
    
    # Safe math functions
    SAFE_FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Evaluate mathematical expression"""
        start_time = time.time()
        expression = kwargs.get("expression", "")
        
        if not expression:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error="Expression is required"
            )
        
        try:
            # Sanitize expression - only allow safe characters
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,\%\^a-zA-Z_]+$', expression):
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=None,
                    error="Invalid characters in expression"
                )
            
            # Replace ^ with ** for power
            expression = expression.replace('^', '**')
            
            # Evaluate with safe functions only
            result = eval(expression, {"__builtins__": {}}, self.SAFE_FUNCTIONS)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=result,
                duration_ms=int((time.time() - start_time) * 1000),
                metadata={"expression": expression}
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Calculation error: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            )


class CodeExecutorTool(Tool):
    """Safe Python code executor (sandbox)"""
    
    name = "code_executor"
    description = "Execute Python code in a safe sandbox"
    parameters = [
        ToolParameter(
            name="code",
            type="string",
            description="Python code to execute",
            required=True
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="Execution timeout in seconds",
            required=False,
            default=10
        )
    ]
    dangerous = True  # Requires confirmation
    
    # Allowed built-ins for sandbox
    SAFE_BUILTINS = {
        'abs': abs,
        'all': all,
        'any': any,
        'bool': bool,
        'dict': dict,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'int': int,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'print': print,
        'range': range,
        'reversed': reversed,
        'round': round,
        'set': set,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'zip': zip,
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute Python code in sandbox"""
        start_time = time.time()
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 10)
        
        if not code:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error="Code is required"
            )
        
        # Security checks
        forbidden = ['import', 'exec', 'eval', '__', 'open', 'file', 'os', 'sys']
        code_lower = code.lower()
        for f in forbidden:
            if f in code_lower:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=None,
                    error=f"Forbidden keyword: {f}"
                )
        
        try:
            # Capture output
            import io
            import sys
            
            output_buffer = io.StringIO()
            old_stdout = sys.stdout
            
            # Create sandbox environment
            sandbox = {
                '__builtins__': self.SAFE_BUILTINS,
                'math': math,
            }
            
            result = None
            
            async def run_code():
                nonlocal result
                sys.stdout = output_buffer
                try:
                    exec(code, sandbox)
                    result = output_buffer.getvalue()
                finally:
                    sys.stdout = old_stdout
            
            # Run with timeout
            await asyncio.wait_for(run_code(), timeout=timeout)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=result or "Code executed successfully (no output)",
                duration_ms=int((time.time() - start_time) * 1000),
                metadata={"code_length": len(code)}
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Execution timed out after {timeout} seconds",
                duration_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Execution error: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            )


class WebFetchTool(Tool):
    """Fetch content from a URL"""
    
    name = "web_fetch"
    description = "Fetch and extract content from a URL"
    parameters = [
        ToolParameter(
            name="url",
            type="string",
            description="URL to fetch",
            required=True
        ),
        ToolParameter(
            name="extract_text",
            type="boolean",
            description="Extract text content only",
            required=False,
            default=True
        )
    ]
    
    async def execute(self, **kwargs) -> ToolResult:
        """Fetch URL content"""
        start_time = time.time()
        url = kwargs.get("url", "")
        
        if not url:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error="URL is required"
            )
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        return ToolResult(
                            tool_name=self.name,
                            success=False,
                            output=None,
                            error=f"HTTP {response.status}"
                        )
                    
                    content = await response.text()
                    
                    # Basic text extraction
                    if kwargs.get("extract_text", True):
                        # Remove HTML tags
                        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
                        content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
                        content = re.sub(r'<[^>]+>', ' ', content)
                        content = re.sub(r'\s+', ' ', content).strip()
                        content = content[:5000]  # Limit content
                    
                    return ToolResult(
                        tool_name=self.name,
                        success=True,
                        output=content,
                        duration_ms=int((time.time() - start_time) * 1000),
                        metadata={"url": url, "content_length": len(content)}
                    )
                    
        except ImportError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error="aiohttp not installed"
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )


# Register built-in tools
def register_builtin_tools():
    """Register all built-in tools"""
    from .base import register_tool
    
    register_tool(WebSearchTool())
    register_tool(CalculatorTool())
    register_tool(CodeExecutorTool())
    register_tool(WebFetchTool())
