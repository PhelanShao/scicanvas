"""
Result Synthesis - Combining multiple agent results into coherent output
Implementation for multi-agent result aggregation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import re
import hashlib
import time

from .base import BasePattern, PatternConfig, PatternResult, ThinkingStep, AgentResult


@dataclass
class SynthesisConfig(PatternConfig):
    """Configuration for result synthesis"""
    style: str = "comprehensive"  # "comprehensive", "concise", "academic"
    include_citations: bool = True
    max_sources: int = 10
    target_length_words: int = 800
    language_match: bool = True  # Match response language to query
    dedup_threshold: float = 0.85


@dataclass
class SynthesisInput:
    """Input for synthesis"""
    query: str
    agent_results: List[AgentResult]
    context: Optional[Dict[str, Any]] = None
    parent_workflow_id: str = ""
    collected_citations: Optional[List[Dict[str, Any]]] = None


class ResultSynthesizer(BasePattern):
    """
    Result synthesis engine for multi-agent outputs.
    
    Features:
    - Deduplication of similar results
    - Quality filtering
    - Citation extraction
    - Language-aware synthesis
    - Multiple output styles
    """
    
    SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis expert who combines multiple research findings into a coherent, well-structured response.

Guidelines:
1. Identify and highlight the most important findings
2. Resolve any contradictions between sources
3. Maintain factual accuracy
4. Use clear, professional language
5. Match the language of the user's query
6. Include relevant citations when available"""

    SYNTHESIS_PROMPT_COMPREHENSIVE = """Query: {query}

Agent Results ({num_results} sources):
{agent_results}

{citation_instructions}

Create a comprehensive response that:
1. Synthesizes all relevant findings
2. Organizes information logically
3. Highlights key insights and conclusions
4. Notes any limitations or uncertainties

Respond in the same language as the query."""

    SYNTHESIS_PROMPT_CONCISE = """Query: {query}

Agent Results ({num_results} sources):
{agent_results}

Create a concise, direct response that:
1. Answers the query directly
2. Includes only essential information
3. Is 2-3 paragraphs maximum

Respond in the same language as the query."""

    def __init__(self, llm_manager, config: Optional[SynthesisConfig] = None):
        super().__init__(llm_manager, config or SynthesisConfig())
        self.config: SynthesisConfig = self.config
        
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[str]] = None,
        **kwargs
    ) -> PatternResult:
        """Synthesize agent results into final response"""
        result = PatternResult(
            pattern_name="Synthesis",
            final_response="",
            metadata={"query": query}
        )
        
        start_time = time.time()
        
        # Get agent results from kwargs or context
        agent_results = kwargs.get("agent_results", [])
        if not agent_results and context:
            agent_results = context.get("agent_results", [])
        
        if not agent_results:
            result.final_response = "No results to synthesize."
            result.success = False
            result.total_duration_ms = int((time.time() - start_time) * 1000)
            return result
        
        # Preprocess results
        result.add_step(ThinkingStep.create(
            step_type="reasoning",
            content=f"Processing {len(agent_results)} agent results",
            metadata={"phase": "preprocessing"}
        ))
        
        processed_results = self._preprocess_results(agent_results)
        
        result.add_step(ThinkingStep.create(
            step_type="reasoning",
            content=f"After filtering: {len(processed_results)} unique results",
            metadata={"filtered_count": len(processed_results)}
        ))
        
        if not processed_results:
            result.final_response = "No valid results after filtering."
            result.success = False
            result.total_duration_ms = int((time.time() - start_time) * 1000)
            return result
        
        # Build synthesis prompt
        results_text = self._format_results_for_synthesis(processed_results)
        citation_instructions = self._build_citation_instructions(context)
        
        if self.config.style == "concise":
            prompt = self.SYNTHESIS_PROMPT_CONCISE.format(
                query=query,
                num_results=len(processed_results),
                agent_results=results_text
            )
        else:
            prompt = self.SYNTHESIS_PROMPT_COMPREHENSIVE.format(
                query=query,
                num_results=len(processed_results),
                agent_results=results_text,
                citation_instructions=citation_instructions
            )
        
        # Call LLM for synthesis
        synth_result = await self._call_llm(
            prompt=prompt,
            system_prompt=self.SYNTHESIS_SYSTEM_PROMPT,
            agent_id="synthesizer",
            model_tier="large"  # Use large tier for synthesis quality
        )
        result.add_agent_result(synth_result)
        
        if synth_result.success:
            result.final_response = synth_result.response
            result.success = True
            result.confidence = 0.8
        else:
            # Fallback to simple concatenation
            result.final_response = self._simple_synthesis(processed_results)
            result.success = True
            result.confidence = 0.5
            result.metadata["fallback_used"] = True
        
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        return result
    
    async def synthesize(self, synthesis_input: SynthesisInput) -> PatternResult:
        """Alternative entry point with SynthesisInput"""
        return await self.execute(
            query=synthesis_input.query,
            context=synthesis_input.context,
            agent_results=synthesis_input.agent_results
        )
    
    def _preprocess_results(self, results: List[AgentResult]) -> List[AgentResult]:
        """Preprocess and filter results"""
        # Filter failed results
        valid = [r for r in results if r.success and r.response.strip()]
        
        # Deduplicate exact matches
        seen_hashes: Set[str] = set()
        deduped = []
        
        for r in valid:
            normalized = r.response.lower().strip()
            h = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(r)
        
        # Deduplicate similar results
        if self.config.dedup_threshold < 1.0:
            deduped = self._deduplicate_similar(deduped)
        
        # Filter low quality results
        deduped = self._filter_low_quality(deduped)
        
        return deduped[:self.config.max_sources]
    
    def _deduplicate_similar(self, results: List[AgentResult]) -> List[AgentResult]:
        """Remove near-duplicate results using Jaccard similarity"""
        if len(results) <= 1:
            return results
        
        unique = []
        
        for candidate in results:
            is_dup = False
            candidate_tokens = self._tokenize(candidate.response)
            
            for existing in unique:
                existing_tokens = self._tokenize(existing.response)
                similarity = self._jaccard_similarity(candidate_tokens, existing_tokens)
                
                if similarity > self.config.dedup_threshold:
                    is_dup = True
                    break
            
            if not is_dup:
                unique.append(candidate)
        
        return unique
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text for similarity comparison"""
        # Remove punctuation and lowercase
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        return set(cleaned.split())
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _filter_low_quality(self, results: List[AgentResult]) -> List[AgentResult]:
        """Filter out low quality results"""
        # Patterns indicating poor quality
        low_quality_patterns = [
            "i cannot access",
            "unable to access",
            "unable to find",
            "no information available",
            "couldn't find",
            "failed to fetch",
            "error occurred",
        ]
        
        filtered = []
        for r in results:
            response_lower = r.response.lower()
            
            # Check for low quality patterns
            if any(pattern in response_lower for pattern in low_quality_patterns):
                continue
            
            # Check minimum length
            if len(r.response.strip()) < 50:
                continue
            
            filtered.append(r)
        
        return filtered
    
    def _format_results_for_synthesis(self, results: List[AgentResult]) -> str:
        """Format agent results for the synthesis prompt"""
        formatted = []
        
        max_chars_per_result = 2000  # Limit each result
        
        for i, r in enumerate(results, 1):
            content = r.response.strip()
            if len(content) > max_chars_per_result:
                content = content[:max_chars_per_result] + "..."
            
            # Clean up the content
            content = self._clean_agent_output(content)
            
            formatted.append(f"[Source {i}]\n{content}")
        
        return "\n\n".join(formatted)
    
    def _clean_agent_output(self, text: str) -> str:
        """Clean agent output for better synthesis"""
        # Remove XML-like tags that might have leaked
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove duplicate newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _build_citation_instructions(self, context: Optional[Dict[str, Any]]) -> str:
        """Build citation instructions based on context"""
        if not self.config.include_citations:
            return ""
        
        citations = []
        if context:
            citations = context.get("collected_citations", [])
            if not citations:
                citations = context.get("available_citations", [])
        
        if not citations:
            return "Note: Include references to specific sources when available."
        
        return f"""Citations available: {len(citations)}
Include inline citations [n] where appropriate to reference sources."""
    
    def _simple_synthesis(self, results: List[AgentResult]) -> str:
        """Fallback simple synthesis when LLM fails"""
        if not results:
            return "No results available."
        
        if len(results) == 1:
            return results[0].response
        
        # Combine results with separators
        parts = []
        for i, r in enumerate(results, 1):
            content = r.response.strip()
            if len(content) > 1000:
                content = content[:1000] + "..."
            parts.append(content)
        
        return "\n\n---\n\n".join(parts)


async def synthesize_results(
    llm_manager,
    query: str,
    agent_results: List[AgentResult],
    context: Optional[Dict[str, Any]] = None,
    style: str = "comprehensive"
) -> str:
    """
    Convenience function to synthesize results.
    
    Returns the synthesized response text.
    """
    synthesizer = ResultSynthesizer(
        llm_manager,
        SynthesisConfig(style=style)
    )
    
    result = await synthesizer.execute(
        query=query,
        context=context,
        agent_results=agent_results
    )
    
    return result.final_response
