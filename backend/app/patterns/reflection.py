"""
Reflection Pattern - Self-evaluation and iterative improvement
Implementation for quality enhancement through reflection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import time

from .base import BasePattern, PatternConfig, PatternResult, ThinkingStep, AgentResult


@dataclass
class ReflectionConfig(PatternConfig):
    """Configuration for Reflection pattern"""
    max_retries: int = 3
    confidence_threshold: float = 0.7
    criteria: List[str] = field(default_factory=lambda: [
        "accuracy", "completeness", "clarity", "relevance"
    ])
    enabled: bool = True
    timeout_ms: int = 30000


class ReflectionPattern(BasePattern):
    """
    Reflection pattern implementation.
    
    This pattern improves responses through self-evaluation:
    1. Generate initial response
    2. Evaluate response against criteria
    3. If below threshold, regenerate with feedback
    4. Repeat until quality threshold or max retries
    """
    
    INITIAL_RESPONSE_PROMPT = """Question: {query}

{context}

Please provide a comprehensive and accurate response."""

    EVALUATION_PROMPT = """Evaluate the following response for quality.

Question: {query}

Response to evaluate:
{response}

Evaluation criteria:
{criteria}

For each criterion, provide a score from 0.0 to 1.0 and brief feedback.
Then provide an overall score and specific suggestions for improvement.

Format:
Criterion Scores:
- accuracy: [score] - [feedback]
- completeness: [score] - [feedback]
- clarity: [score] - [feedback]
- relevance: [score] - [feedback]

Overall Score: [0.0-1.0]

Improvement Suggestions:
[List specific ways to improve the response]"""

    IMPROVEMENT_PROMPT = """Question: {query}

Previous response:
{previous_response}

Feedback for improvement:
{feedback}

Please provide an improved response that addresses the feedback while maintaining the strengths of the previous response."""

    def __init__(self, llm_manager, config: Optional[ReflectionConfig] = None):
        super().__init__(llm_manager, config or ReflectionConfig())
        self.config: ReflectionConfig = self.config
        
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[str]] = None,
        **kwargs
    ) -> PatternResult:
        """Execute reflection-based response improvement"""
        result = PatternResult(
            pattern_name="Reflection",
            final_response="",
            metadata={"query": query, "iterations": 0, "improvements_made": []}
        )
        
        start_time = time.time()
        
        # Format context
        context_str = ""
        if context:
            parts = [f"- {k}: {v}" for k, v in context.items() if k not in ["internal", "system"]]
            if parts:
                context_str = "Context:\n" + "\n".join(parts)
        
        # Generate initial response
        result.add_step(ThinkingStep.create(
            step_type="reasoning",
            content="Generating initial response",
            metadata={"phase": "initial"}
        ))
        
        initial_result = await self._call_llm(
            prompt=self.INITIAL_RESPONSE_PROMPT.format(query=query, context=context_str),
            agent_id="reflection-initial"
        )
        result.add_agent_result(initial_result)
        
        if not initial_result.success:
            result.success = False
            result.error = initial_result.error
            result.total_duration_ms = int((time.time() - start_time) * 1000)
            return result
        
        current_response = initial_result.response
        
        # If reflection is disabled, return initial response
        if not self.config.enabled:
            result.final_response = current_response
            result.confidence = 0.5
            result.total_duration_ms = int((time.time() - start_time) * 1000)
            return result
        
        # Reflection loop
        retry_count = 0
        last_score = 0.5
        
        while retry_count < self.config.max_retries:
            result.metadata["iterations"] = retry_count + 1
            
            # Evaluate current response
            eval_result = await self._evaluate_response(query, current_response)
            
            result.add_step(ThinkingStep.create(
                step_type="evaluation",
                content=f"Evaluation score: {eval_result['score']:.2f}",
                confidence=eval_result['score'],
                metadata={
                    "iteration": retry_count + 1,
                    "criteria_scores": eval_result.get('criteria_scores', {})
                }
            ))
            
            last_score = eval_result['score']
            
            # Check if meets threshold
            if last_score >= self.config.confidence_threshold:
                result.add_step(ThinkingStep.create(
                    step_type="reasoning",
                    content=f"Response meets quality threshold ({last_score:.2f} >= {self.config.confidence_threshold})",
                    metadata={"status": "passed"}
                ))
                break
            
            # Need improvement
            retry_count += 1
            if retry_count >= self.config.max_retries:
                result.add_step(ThinkingStep.create(
                    step_type="reasoning",
                    content="Max retries reached, using best effort response",
                    metadata={"status": "max_retries"}
                ))
                break
            
            # Generate improved response
            result.add_step(ThinkingStep.create(
                step_type="reasoning",
                content=f"Improving response based on feedback (iteration {retry_count})",
                metadata={"phase": "improvement", "iteration": retry_count}
            ))
            
            improvement_result = await self._call_llm(
                prompt=self.IMPROVEMENT_PROMPT.format(
                    query=query,
                    previous_response=current_response,
                    feedback=eval_result.get('feedback', 'Please improve the response.')
                ),
                agent_id=f"reflection-improvement-{retry_count}"
            )
            result.add_agent_result(improvement_result)
            
            if improvement_result.success:
                current_response = improvement_result.response
                result.metadata["improvements_made"].append({
                    "iteration": retry_count,
                    "feedback": eval_result.get('feedback', '')[:200]
                })
            else:
                # Keep previous response if improvement failed
                break
        
        result.final_response = current_response
        result.confidence = last_score
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        
        return result
    
    async def _evaluate_response(self, query: str, response: str) -> Dict[str, Any]:
        """Evaluate response quality"""
        criteria_str = "\n".join([f"- {c}" for c in self.config.criteria])
        
        eval_result = await self._call_llm(
            prompt=self.EVALUATION_PROMPT.format(
                query=query,
                response=response,
                criteria=criteria_str
            ),
            agent_id="reflection-evaluator",
            max_tokens=1000
        )
        
        if not eval_result.success:
            return {"score": 0.5, "feedback": "Evaluation failed"}
        
        return self._parse_evaluation(eval_result.response)
    
    def _parse_evaluation(self, evaluation: str) -> Dict[str, Any]:
        """Parse evaluation response into structured data"""
        result = {
            "score": 0.5,
            "feedback": "",
            "criteria_scores": {}
        }
        
        # Extract overall score
        score_match = re.search(r'Overall\s+Score:\s*(\d+\.?\d*)', evaluation, re.IGNORECASE)
        if score_match:
            try:
                result["score"] = float(score_match.group(1))
            except ValueError:
                pass
        
        # Extract criterion scores
        for criterion in self.config.criteria:
            pattern = rf'{criterion}:\s*(\d+\.?\d*)\s*[-–]\s*(.+?)(?=\n|$)'
            match = re.search(pattern, evaluation, re.IGNORECASE)
            if match:
                try:
                    result["criteria_scores"][criterion] = {
                        "score": float(match.group(1)),
                        "feedback": match.group(2).strip()
                    }
                except ValueError:
                    pass
        
        # Extract improvement suggestions
        suggestions_match = re.search(
            r'Improvement\s+Suggestions?:\s*(.+?)(?=\n\n|$)',
            evaluation,
            re.IGNORECASE | re.DOTALL
        )
        if suggestions_match:
            result["feedback"] = suggestions_match.group(1).strip()
        else:
            # Fallback: use the whole evaluation as feedback
            result["feedback"] = evaluation
        
        # Calculate average if no overall score found
        if result["score"] == 0.5 and result["criteria_scores"]:
            scores = [cs["score"] for cs in result["criteria_scores"].values()]
            result["score"] = sum(scores) / len(scores) if scores else 0.5
        
        return result


async def reflect_on_result(
    llm_manager,
    query: str,
    initial_result: str,
    agent_results: List[AgentResult],
    context: Optional[Dict[str, Any]] = None,
    config: Optional[ReflectionConfig] = None
) -> tuple:
    """
    Standalone reflection function for reflection results.
    
    Returns: (improved_result, final_score, total_tokens)
    """
    pattern = ReflectionPattern(llm_manager, config)
    
    # Create a modified execution that starts with the initial result
    result = await pattern.execute(
        query=query,
        context={
            **(context or {}),
            "_initial_response": initial_result
        }
    )
    
    return (
        result.final_response,
        result.confidence,
        result.total_tokens
    )
