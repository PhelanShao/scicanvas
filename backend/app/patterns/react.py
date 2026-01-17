"""
ReAct Pattern - Reason, Act, Observe Loop
Implementation for iterative reasoning with tool use.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import time

from .base import BasePattern, PatternConfig, PatternResult, ThinkingStep, AgentResult


@dataclass
class ReActConfig(PatternConfig):
    """Configuration for ReAct pattern"""
    max_iterations: int = 5
    enable_tools: bool = True
    available_tools: List[str] = field(default_factory=list)
    stop_on_final_answer: bool = True
    reflection_enabled: bool = True


class ReActPattern(BasePattern):
    """
    Reason-Act-Observe pattern implementation.
    
    This pattern follows the ReAct paradigm:
    1. THOUGHT: Reason about the current state
    2. ACTION: Decide on an action to take (including tool use)
    3. OBSERVATION: Process the result of the action
    4. Repeat until a final answer is reached
    """
    
    REACT_SYSTEM_PROMPT = """You are a reasoning assistant that solves problems step by step using the ReAct framework.

For each step, you must output in this EXACT format:
THOUGHT: [Your reasoning about the current state and what to do next]
ACTION: [The action to take - either "Final Answer" or a tool name with parameters]
OBSERVATION: [What you observed from the action - leave empty if action is "Final Answer"]

Available actions:
- Final Answer: [your final response] - Use this when you have enough information
{tool_descriptions}

Rules:
1. Always start with THOUGHT
2. Be specific and logical in your reasoning
3. If you need more information, take an action to get it
4. When you have sufficient information, provide a Final Answer
5. Keep responses concise but complete"""

    REACT_PROMPT_TEMPLATE = """Question: {query}

{context}

Previous steps:
{history}

Now continue with your next THOUGHT, ACTION, and OBSERVATION:"""

    def __init__(self, llm_manager, config: Optional[ReActConfig] = None):
        super().__init__(llm_manager, config or ReActConfig())
        self.config: ReActConfig = self.config
        
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[str]] = None,
        **kwargs
    ) -> PatternResult:
        """Execute the ReAct reasoning loop"""
        result = PatternResult(
            pattern_name="ReAct",
            final_response="",
            metadata={"query": query, "iterations": 0}
        )
        
        start_time = time.time()
        react_history = []
        iteration = 0
        
        # Build tool descriptions
        tool_descriptions = self._build_tool_descriptions()
        system_prompt = self.REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions
        )
        
        context_str = self._format_context(context) if context else ""
        
        while iteration < self.config.max_iterations:
            iteration += 1
            
            # Build prompt with history
            history_str = "\n".join(react_history) if react_history else "None yet."
            prompt = self.REACT_PROMPT_TEMPLATE.format(
                query=query,
                context=context_str,
                history=history_str
            )
            
            # Call LLM
            agent_result = await self._call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                agent_id=f"react-iteration-{iteration}"
            )
            
            result.add_agent_result(agent_result)
            
            if not agent_result.success:
                result.success = False
                result.error = agent_result.error
                break
                
            # Parse the response
            thought, action, observation = self._parse_react_response(agent_result.response)
            
            # Record thinking step
            if thought:
                result.add_step(ThinkingStep.create(
                    step_type="reasoning",
                    content=thought,
                    tokens_used=agent_result.tokens_used // 3,
                    metadata={"iteration": iteration}
                ))
                react_history.append(f"THOUGHT: {thought}")
            
            # Check for final answer
            if action and action.lower().startswith("final answer"):
                final_answer = action.replace("Final Answer:", "").replace("final answer:", "").strip()
                result.final_response = final_answer
                result.add_step(ThinkingStep.create(
                    step_type="action",
                    content=f"Final Answer: {final_answer}",
                    metadata={"iteration": iteration, "is_final": True}
                ))
                result.confidence = self._calculate_confidence(result.thinking_steps)
                break
            
            # Record action
            if action:
                result.add_step(ThinkingStep.create(
                    step_type="action",
                    content=action,
                    metadata={"iteration": iteration}
                ))
                react_history.append(f"ACTION: {action}")
                
                # Execute tool if enabled
                if self.config.enable_tools:
                    tool_result = await self._execute_tool(action, context)
                    observation = tool_result or observation
            
            # Record observation
            if observation:
                result.add_step(ThinkingStep.create(
                    step_type="observation",
                    content=observation,
                    metadata={"iteration": iteration}
                ))
                react_history.append(f"OBSERVATION: {observation}")
        
        # If no final answer found after max iterations
        if not result.final_response:
            result.final_response = self._synthesize_from_history(react_history)
            result.metadata["max_iterations_reached"] = True
            
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        result.metadata["iterations"] = iteration
        
        return result
    
    def _parse_react_response(self, response: str) -> tuple:
        """Parse ReAct format response into thought, action, observation"""
        thought = ""
        action = ""
        observation = ""
        
        # Extract THOUGHT
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|OBSERVATION:|$)', response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
            
        # Extract ACTION
        action_match = re.search(r'ACTION:\s*(.+?)(?=OBSERVATION:|THOUGHT:|$)', response, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            
        # Extract OBSERVATION
        obs_match = re.search(r'OBSERVATION:\s*(.+?)(?=THOUGHT:|ACTION:|$)', response, re.DOTALL | re.IGNORECASE)
        if obs_match:
            observation = obs_match.group(1).strip()
            
        return thought, action, observation
    
    def _build_tool_descriptions(self) -> str:
        """Build tool descriptions for the system prompt"""
        if not self.config.enable_tools or not self.config.available_tools:
            return "- No additional tools available"
            
        descriptions = []
        tool_docs = {
            "web_search": "web_search(query) - Search the web for information",
            "calculator": "calculator(expression) - Evaluate mathematical expressions",
            "code_executor": "code_executor(code) - Execute Python code",
        }
        
        for tool in self.config.available_tools:
            desc = tool_docs.get(tool, f"{tool}(...) - Execute {tool}")
            descriptions.append(f"- {desc}")
            
        return "\n".join(descriptions)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary into a string"""
        if not context:
            return ""
        parts = ["Context:"]
        for key, value in context.items():
            if key not in ["internal", "system"]:
                parts.append(f"- {key}: {value}")
        return "\n".join(parts)
    
    async def _execute_tool(self, action: str, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Execute a tool based on the action string"""
        # Parse tool name and parameters from action
        # Format: tool_name(param1, param2, ...)
        match = re.match(r'(\w+)\s*\((.*)\)', action)
        if not match:
            return None
            
        tool_name = match.group(1)
        params = match.group(2)
        
        # TODO: Implement actual tool execution
        # For now, return a placeholder
        return f"[Tool {tool_name} executed with params: {params}]"
    
    def _calculate_confidence(self, steps: List[ThinkingStep]) -> float:
        """Calculate confidence score based on thinking steps"""
        if not steps:
            return 0.5
            
        # Base confidence
        confidence = 0.5
        
        # More reasoning steps = higher confidence (up to a point)
        reasoning_steps = [s for s in steps if s.step_type == "reasoning"]
        if len(reasoning_steps) >= 2:
            confidence += 0.1
        if len(reasoning_steps) >= 3:
            confidence += 0.1
            
        # Observations add confidence
        observations = [s for s in steps if s.step_type == "observation"]
        confidence += min(len(observations) * 0.05, 0.2)
        
        return min(confidence, 1.0)
    
    def _synthesize_from_history(self, history: List[str]) -> str:
        """Synthesize a final response from the history when max iterations reached"""
        if not history:
            return "Unable to determine a final answer."
            
        # Extract the last thought
        thoughts = [h for h in history if h.startswith("THOUGHT:")]
        if thoughts:
            last_thought = thoughts[-1].replace("THOUGHT:", "").strip()
            return f"Based on my analysis: {last_thought}"
            
        return "The analysis was incomplete. Please try rephrasing your question."
