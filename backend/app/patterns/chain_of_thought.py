"""
Chain of Thought Pattern - Step-by-step reasoning
Implementation for explicit reasoning chains.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import time

from .base import BasePattern, PatternConfig, PatternResult, ThinkingStep, AgentResult


@dataclass
class CoTConfig(PatternConfig):
    """Configuration for Chain of Thought pattern"""
    num_reasoning_steps: int = 5
    extract_final_answer: bool = True
    show_work: bool = True
    step_by_step_prompt: bool = True


class ChainOfThoughtPattern(BasePattern):
    """
    Chain of Thought pattern implementation.
    
    This pattern guides the model through explicit step-by-step reasoning:
    1. Break down the problem
    2. Reason through each step
    3. Build up to a conclusion
    4. Extract the final answer
    """
    
    COT_SYSTEM_PROMPT = """You are a methodical problem solver who thinks step by step.

When given a question or problem:
1. First, understand what is being asked
2. Break down the problem into clear steps
3. Work through each step logically
4. Show your reasoning at each stage
5. Arrive at a well-supported conclusion

Format your response as:
## Understanding
[What the question is asking]

## Step-by-Step Reasoning
Step 1: [First step of reasoning]
Step 2: [Second step of reasoning]
...

## Conclusion
[Your final answer with supporting reasoning]"""

    COT_PROMPT_TEMPLATE = """Question: {query}

{context}

Please think through this step by step, showing your reasoning process clearly before arriving at your final answer."""

    EXTRACTION_PROMPT = """Based on the following reasoning process, extract ONLY the final answer in a concise form.

Reasoning:
{reasoning}

Final Answer (concise):"""

    def __init__(self, llm_manager, config: Optional[CoTConfig] = None):
        super().__init__(llm_manager, config or CoTConfig())
        self.config: CoTConfig = self.config
        
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[str]] = None,
        **kwargs
    ) -> PatternResult:
        """Execute Chain of Thought reasoning"""
        result = PatternResult(
            pattern_name="ChainOfThought",
            final_response="",
            metadata={"query": query}
        )
        
        start_time = time.time()
        
        # Format context
        context_str = ""
        if context:
            context_parts = [f"- {k}: {v}" for k, v in context.items() if k not in ["internal", "system"]]
            if context_parts:
                context_str = "Context:\n" + "\n".join(context_parts)
        
        # Build the main prompt
        prompt = self.COT_PROMPT_TEMPLATE.format(
            query=query,
            context=context_str
        )
        
        # Call LLM for reasoning
        agent_result = await self._call_llm(
            prompt=prompt,
            system_prompt=self.COT_SYSTEM_PROMPT,
            agent_id="cot-reasoning"
        )
        
        result.add_agent_result(agent_result)
        
        if not agent_result.success:
            result.success = False
            result.error = agent_result.error
            result.total_duration_ms = int((time.time() - start_time) * 1000)
            return result
        
        reasoning_response = agent_result.response
        
        # Parse the reasoning into steps
        steps = self._parse_reasoning_steps(reasoning_response)
        for step in steps:
            result.add_step(step)
        
        # Extract final answer if configured
        if self.config.extract_final_answer:
            conclusion = self._extract_conclusion(reasoning_response)
            if conclusion:
                result.final_response = conclusion
            else:
                # Use LLM to extract the final answer
                extraction_result = await self._call_llm(
                    prompt=self.EXTRACTION_PROMPT.format(reasoning=reasoning_response),
                    agent_id="cot-extraction",
                    max_tokens=500
                )
                result.add_agent_result(extraction_result)
                if extraction_result.success:
                    result.final_response = extraction_result.response.strip()
                else:
                    result.final_response = reasoning_response
        else:
            result.final_response = reasoning_response
        
        # Calculate confidence based on reasoning quality
        result.confidence = self._assess_reasoning_quality(steps)
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        
        return result
    
    def _parse_reasoning_steps(self, response: str) -> List[ThinkingStep]:
        """Parse the reasoning response into individual steps"""
        steps = []
        
        # Try to find structured steps (Step 1:, Step 2:, etc.)
        step_pattern = r'Step\s*(\d+)[:\.]?\s*(.+?)(?=Step\s*\d+|##|$)'
        step_matches = re.findall(step_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if step_matches:
            for num, content in step_matches:
                steps.append(ThinkingStep.create(
                    step_type="reasoning",
                    content=content.strip(),
                    metadata={"step_number": int(num)}
                ))
        else:
            # Try to find section-based structure
            sections = {
                "understanding": r'##\s*Understanding\s*\n(.+?)(?=##|$)',
                "reasoning": r'##\s*(?:Step-by-Step\s+)?Reasoning\s*\n(.+?)(?=##|$)',
                "conclusion": r'##\s*Conclusion\s*\n(.+?)(?=##|$)'
            }
            
            for section_type, pattern in sections.items():
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    steps.append(ThinkingStep.create(
                        step_type=section_type if section_type != "reasoning" else "reasoning",
                        content=match.group(1).strip(),
                        metadata={"section": section_type}
                    ))
        
        # If no structured steps found, treat the whole response as one step
        if not steps:
            steps.append(ThinkingStep.create(
                step_type="reasoning",
                content=response.strip()
            ))
        
        return steps
    
    def _extract_conclusion(self, response: str) -> Optional[str]:
        """Extract the conclusion section from the response"""
        # Try to find a conclusion section
        patterns = [
            r'##\s*Conclusion\s*\n(.+?)(?=##|$)',
            r'(?:Final\s+)?(?:Answer|Conclusion)[:\s]+(.+?)(?=\n\n|$)',
            r'(?:Therefore|Thus|In conclusion)[,:\s]+(.+?)(?=\n\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _assess_reasoning_quality(self, steps: List[ThinkingStep]) -> float:
        """Assess the quality of reasoning based on the steps"""
        if not steps:
            return 0.3
        
        confidence = 0.5
        
        # Multiple steps indicate thorough reasoning
        if len(steps) >= 3:
            confidence += 0.15
        elif len(steps) >= 2:
            confidence += 0.1
        
        # Check for logical connectors in reasoning
        logical_words = ["therefore", "because", "since", "thus", "hence", "so", "consequently"]
        for step in steps:
            content_lower = step.content.lower()
            if any(word in content_lower for word in logical_words):
                confidence += 0.05
                break
        
        # Check for structured reasoning (numbered points, etc.)
        for step in steps:
            if re.search(r'\d+[.)]\s', step.content):
                confidence += 0.05
                break
        
        return min(confidence, 1.0)
