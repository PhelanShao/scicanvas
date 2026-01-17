"""
Task Decomposition - Breaking complex tasks into subtasks
Implementation for cognitive task planning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import time

from .base import BasePattern, PatternConfig, PatternResult, ThinkingStep, AgentResult


@dataclass
class Subtask:
    """A decomposed subtask"""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    estimated_tokens: int = 0
    suggested_tools: List[str] = field(default_factory=list)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionResult:
    """Result of task decomposition"""
    mode: str  # "simple", "standard", "complex"
    complexity_score: float
    subtasks: List[Subtask]
    execution_strategy: str  # "sequential", "parallel", "hybrid"
    total_estimated_tokens: int
    agent_types: List[str] = field(default_factory=list)
    concurrency_limit: int = 5
    cognitive_strategy: str = ""  # "react", "cot", "tot", "debate"
    confidence: float = 0.5
    fallback_strategy: str = ""


@dataclass
class DecompositionConfig(PatternConfig):
    """Configuration for task decomposition"""
    enable_tools: bool = True
    available_tools: List[str] = field(default_factory=list)
    max_subtasks: int = 10
    min_subtasks: int = 1
    auto_select_strategy: bool = True


class TaskDecomposer(BasePattern):
    """
    Task decomposition engine for complex query planning.
    
    Features:
    - Analyzes query complexity
    - Breaks into manageable subtasks
    - Suggests execution strategy
    - Assigns tools and resources
    """
    
    DECOMPOSITION_PROMPT = """Analyze this task and break it down into subtasks:

Task: {query}

Context: {context}

Available tools: {tools}

For each subtask, provide:
1. A clear description
2. Dependencies on other subtasks (if any)
3. Suggested tools to use
4. Estimated complexity (1-10)

Also determine:
- Overall complexity score (0.0 - 1.0)
- Recommended execution strategy (sequential, parallel, or hybrid)
- Best cognitive approach (react, chain_of_thought, tree_of_thoughts, debate)

Format your response as:
COMPLEXITY: [0.0-1.0]
STRATEGY: [sequential/parallel/hybrid]
COGNITIVE_APPROACH: [react/chain_of_thought/tree_of_thoughts/debate]

SUBTASKS:
1. [Description] | Dependencies: [none/subtask_ids] | Tools: [tool1, tool2] | Complexity: [1-10]
2. [Description] | Dependencies: [1] | Tools: [tool3] | Complexity: [1-10]
..."""

    COMPLEXITY_PROMPT = """Quickly assess the complexity of this query on a scale of 0.0 to 1.0:

Query: {query}

Consider:
- Number of distinct aspects to address
- Domain expertise required
- Need for external information
- Reasoning depth required

Respond with just a number between 0.0 and 1.0:"""

    def __init__(self, llm_manager, config: Optional[DecompositionConfig] = None):
        super().__init__(llm_manager, config or DecompositionConfig())
        self.config: DecompositionConfig = self.config
        
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[str]] = None,
        **kwargs
    ) -> PatternResult:
        """Decompose a task into subtasks"""
        result = PatternResult(
            pattern_name="TaskDecomposition",
            final_response="",
            metadata={"query": query}
        )
        
        start_time = time.time()
        
        # Format context and tools
        context_str = self._format_context(context) if context else "No additional context."
        tools_str = ", ".join(self.config.available_tools) if self.config.available_tools else "No tools available"
        
        # Perform decomposition
        decomp_result = await self._call_llm(
            prompt=self.DECOMPOSITION_PROMPT.format(
                query=query,
                context=context_str,
                tools=tools_str
            ),
            agent_id="task-decomposer"
        )
        result.add_agent_result(decomp_result)
        
        if not decomp_result.success:
            result.success = False
            result.error = decomp_result.error
            result.total_duration_ms = int((time.time() - start_time) * 1000)
            return result
        
        # Parse decomposition result
        decomposition = self._parse_decomposition(decomp_result.response)
        
        # Record subtasks as thinking steps
        for subtask in decomposition.subtasks:
            result.add_step(ThinkingStep.create(
                step_type="reasoning",
                content=f"Subtask: {subtask.description}",
                metadata={
                    "subtask_id": subtask.id,
                    "dependencies": subtask.dependencies,
                    "tools": subtask.suggested_tools
                }
            ))
        
        # Build summary response
        result.final_response = self._format_decomposition_summary(decomposition)
        result.confidence = decomposition.confidence
        result.metadata.update({
            "complexity_score": decomposition.complexity_score,
            "execution_strategy": decomposition.execution_strategy,
            "cognitive_strategy": decomposition.cognitive_strategy,
            "num_subtasks": len(decomposition.subtasks),
            "decomposition": decomposition  # Include full result
        })
        
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        return result
    
    async def analyze_complexity(self, query: str) -> float:
        """Quick complexity analysis using heuristics and LLM"""
        # First, heuristic analysis
        heuristic_score = self._heuristic_complexity(query)
        
        # If clearly simple or complex, return heuristic
        if heuristic_score < 0.2 or heuristic_score > 0.8:
            return heuristic_score
        
        # Otherwise, use LLM for better assessment
        result = await self._call_llm(
            prompt=self.COMPLEXITY_PROMPT.format(query=query),
            agent_id="complexity-analyzer",
            max_tokens=50
        )
        
        if result.success:
            try:
                score_match = re.search(r'(\d+\.?\d*)', result.response)
                if score_match:
                    return float(score_match.group(1))
            except:
                pass
        
        return heuristic_score
    
    def _heuristic_complexity(self, query: str) -> float:
        """Heuristic-based complexity estimation"""
        score = 0.3  # Base score
        query_lower = query.lower()
        
        # Length indicators
        words = len(query.split())
        if words > 50:
            score += 0.2
        elif words > 20:
            score += 0.1
        
        # Multi-part indicators
        multi_part_words = ["and", "also", "additionally", "furthermore", "compare", "versus"]
        if any(w in query_lower for w in multi_part_words):
            score += 0.15
        
        # Research indicators
        research_words = ["analyze", "research", "investigate", "evaluate", "comprehensive"]
        if any(w in query_lower for w in research_words):
            score += 0.15
        
        # Technical indicators
        if "?" in query and query.count("?") > 1:
            score += 0.1  # Multiple questions
        
        # Simple task indicators (reduce score)
        simple_words = ["what is", "define", "explain briefly", "simply"]
        if any(w in query_lower for w in simple_words):
            score -= 0.1
        
        return max(0, min(1, score))
    
    def _parse_decomposition(self, response: str) -> DecompositionResult:
        """Parse LLM decomposition response"""
        result = DecompositionResult(
            mode="standard",
            complexity_score=0.5,
            subtasks=[],
            execution_strategy="sequential",
            total_estimated_tokens=0
        )
        
        # Extract complexity
        complexity_match = re.search(r'COMPLEXITY:\s*(\d+\.?\d*)', response, re.IGNORECASE)
        if complexity_match:
            result.complexity_score = float(complexity_match.group(1))
        
        # Determine mode based on complexity
        if result.complexity_score < 0.3:
            result.mode = "simple"
        elif result.complexity_score > 0.7:
            result.mode = "complex"
        else:
            result.mode = "standard"
        
        # Extract strategy
        strategy_match = re.search(r'STRATEGY:\s*(\w+)', response, re.IGNORECASE)
        if strategy_match:
            result.execution_strategy = strategy_match.group(1).lower()
        
        # Extract cognitive approach
        cognitive_match = re.search(r'COGNITIVE_APPROACH:\s*(\w+)', response, re.IGNORECASE)
        if cognitive_match:
            result.cognitive_strategy = cognitive_match.group(1).lower()
        
        # Extract subtasks
        subtask_pattern = r'(\d+)\.\s*(.+?)\s*\|\s*Dependencies:\s*(.+?)\s*\|\s*Tools:\s*(.+?)\s*\|\s*Complexity:\s*(\d+)'
        subtask_matches = re.findall(subtask_pattern, response, re.IGNORECASE)
        
        for match in subtask_matches:
            task_id, description, deps, tools, complexity = match
            
            # Parse dependencies
            dependencies = []
            if deps.lower() != "none":
                dep_nums = re.findall(r'\d+', deps)
                dependencies = [f"task-{d}" for d in dep_nums]
            
            # Parse tools
            suggested_tools = [t.strip() for t in tools.split(',') if t.strip()]
            
            subtask = Subtask(
                id=f"task-{task_id}",
                description=description.strip(),
                dependencies=dependencies,
                suggested_tools=suggested_tools,
                estimated_tokens=int(complexity) * 500,  # Rough estimate
                priority=int(task_id)
            )
            result.subtasks.append(subtask)
        
        # If no structured subtasks found, create a single task
        if not result.subtasks:
            result.subtasks.append(Subtask(
                id="task-1",
                description="Complete the requested task",
                estimated_tokens=2000
            ))
        
        # Calculate total tokens
        result.total_estimated_tokens = sum(s.estimated_tokens for s in result.subtasks)
        
        # Confidence based on parsing quality
        result.confidence = 0.8 if subtask_matches else 0.5
        
        return result
    
    def _format_decomposition_summary(self, decomposition: DecompositionResult) -> str:
        """Format decomposition result as readable summary"""
        lines = [
            f"Task Analysis:",
            f"- Complexity: {decomposition.complexity_score:.2f} ({decomposition.mode})",
            f"- Strategy: {decomposition.execution_strategy}",
            f"- Cognitive Approach: {decomposition.cognitive_strategy or 'standard'}",
            f"- Subtasks: {len(decomposition.subtasks)}",
            "",
            "Subtasks:"
        ]
        
        for subtask in decomposition.subtasks:
            deps = ", ".join(subtask.dependencies) if subtask.dependencies else "none"
            tools = ", ".join(subtask.suggested_tools) if subtask.suggested_tools else "none"
            lines.append(f"  {subtask.id}: {subtask.description}")
            lines.append(f"    Dependencies: {deps} | Tools: {tools}")
        
        return "\n".join(lines)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompts"""
        if not context:
            return ""
        parts = [f"- {k}: {v}" for k, v in context.items() if k not in ["internal", "system"]]
        return "\n".join(parts)


async def decompose_task(
    llm_manager,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    available_tools: Optional[List[str]] = None
) -> DecompositionResult:
    """
    Convenience function to decompose a task.
    
    Returns DecompositionResult with subtasks and strategy.
    """
    decomposer = TaskDecomposer(
        llm_manager,
        DecompositionConfig(
            available_tools=available_tools or []
        )
    )
    
    result = await decomposer.execute(query, context)
    return result.metadata.get("decomposition", DecompositionResult(
        mode="simple",
        complexity_score=0.5,
        subtasks=[],
        execution_strategy="sequential",
        total_estimated_tokens=0
    ))
