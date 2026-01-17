"""
Debate Pattern - Multi-agent debate for exploring different perspectives
Implementation for adversarial reasoning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import time
import asyncio

from .base import BasePattern, PatternConfig, PatternResult, ThinkingStep, AgentResult


@dataclass
class DebatePosition:
    """Represents one agent's position in the debate"""
    agent_id: str
    perspective: str
    position: str
    arguments: List[str] = field(default_factory=list)
    confidence: float = 0.5
    tokens_used: int = 0


@dataclass
class DebateConfig(PatternConfig):
    """Configuration for Debate pattern"""
    num_debaters: int = 3
    max_rounds: int = 3
    perspectives: List[str] = field(default_factory=list)
    require_consensus: bool = False
    moderator_enabled: bool = False
    voting_enabled: bool = True


class DebatePattern(BasePattern):
    """
    Multi-agent Debate pattern implementation.
    
    This pattern simulates a debate between multiple perspectives:
    1. Each debater takes an initial position
    2. Debaters respond to each other's arguments
    3. Through rounds of debate, positions evolve
    4. A final synthesis or consensus is reached
    """
    
    INITIAL_POSITION_PROMPT = """You are a debate participant representing the {perspective} perspective.

Topic for debate: {query}

Context: {context}

Provide your initial position on this topic. Be specific and provide strong arguments.

Your position:"""

    DEBATE_ROUND_PROMPT = """You are a debate participant representing the {perspective} perspective.

Topic: {query}

Round {round_num}: Consider these other perspectives:
{other_positions}

As {perspective}, respond with:
1. Counter-arguments to opposing views
2. Strengthen your position with new evidence
3. Find any common ground

Your response:"""

    MODERATOR_PROMPT = """You are a debate moderator.

Topic: {query}

Debate summary:
{debate_history}

Final positions:
{final_positions}

Synthesize the debate into a balanced conclusion that:
1. Acknowledges the strongest arguments from each side
2. Identifies areas of consensus
3. Provides a nuanced final answer

Your synthesis:"""

    DEFAULT_PERSPECTIVES = ["optimistic", "skeptical", "practical", "innovative", "conservative"]

    def __init__(self, llm_manager, config: Optional[DebateConfig] = None):
        super().__init__(llm_manager, config or DebateConfig())
        self.config: DebateConfig = self.config
        
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[str]] = None,
        **kwargs
    ) -> PatternResult:
        """Execute multi-agent debate"""
        result = PatternResult(
            pattern_name="Debate",
            final_response="",
            metadata={
                "query": query,
                "rounds": 0,
                "consensus_reached": False
            }
        )
        
        start_time = time.time()
        
        # Initialize perspectives
        perspectives = self.config.perspectives if self.config.perspectives else \
            self.DEFAULT_PERSPECTIVES[:self.config.num_debaters]
        
        context_str = self._format_context(context) if context else "No additional context."
        debate_history = []
        
        # Phase 1: Initial positions
        result.add_step(ThinkingStep.create(
            step_type="reasoning",
            content="Phase 1: Gathering initial positions from all debaters",
            metadata={"phase": "initial_positions"}
        ))
        
        positions = await self._gather_initial_positions(query, context_str, perspectives)
        
        for pos in positions:
            result.add_step(ThinkingStep.create(
                step_type="thought",
                content=f"[{pos.perspective}] {pos.position[:200]}...",
                confidence=pos.confidence,
                metadata={"agent": pos.agent_id, "perspective": pos.perspective}
            ))
            debate_history.append(f"{pos.perspective}: {pos.position}")
            result.total_tokens += pos.tokens_used
        
        # Phase 2: Debate rounds
        current_positions = positions
        
        for round_num in range(1, self.config.max_rounds + 1):
            result.add_step(ThinkingStep.create(
                step_type="reasoning",
                content=f"Phase 2: Debate Round {round_num}",
                metadata={"phase": "debate", "round": round_num}
            ))
            
            round_positions = await self._conduct_debate_round(
                query, round_num, current_positions, perspectives
            )
            
            for pos in round_positions:
                result.add_step(ThinkingStep.create(
                    step_type="thought",
                    content=f"[Round {round_num} - {pos.perspective}] {pos.position[:150]}...",
                    confidence=pos.confidence,
                    metadata={"round": round_num, "perspective": pos.perspective}
                ))
                debate_history.append(f"Round {round_num} - {pos.perspective}: {pos.position}")
                result.total_tokens += pos.tokens_used
            
            current_positions = round_positions
            result.metadata["rounds"] = round_num
            
            # Check for consensus
            if self.config.require_consensus and self._check_consensus(round_positions):
                result.metadata["consensus_reached"] = True
                break
        
        # Phase 3: Resolution
        result.add_step(ThinkingStep.create(
            step_type="reasoning",
            content="Phase 3: Synthesizing final conclusion",
            metadata={"phase": "resolution"}
        ))
        
        if self.config.moderator_enabled:
            final_response = await self._moderate_debate(
                query, debate_history, current_positions
            )
        elif self.config.voting_enabled:
            final_response, votes = self._conduct_voting(current_positions)
            result.metadata["votes"] = votes
        else:
            final_response = self._synthesize_positions(query, current_positions)
        
        result.final_response = final_response
        
        # Calculate confidence based on debate quality
        result.confidence = self._calculate_debate_confidence(current_positions)
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        
        return result
    
    async def _gather_initial_positions(
        self,
        query: str,
        context_str: str,
        perspectives: List[str]
    ) -> List[DebatePosition]:
        """Gather initial positions from all debaters concurrently"""
        tasks = []
        
        for i, perspective in enumerate(perspectives):
            agent_id = f"debater-{i+1}-{perspective}"
            prompt = self.INITIAL_POSITION_PROMPT.format(
                perspective=perspective,
                query=query,
                context=context_str
            )
            tasks.append(self._call_llm(prompt=prompt, agent_id=agent_id))
        
        results = await asyncio.gather(*tasks)
        
        positions = []
        for i, (perspective, result) in enumerate(zip(perspectives, results)):
            if result.success:
                positions.append(DebatePosition(
                    agent_id=f"debater-{i+1}",
                    perspective=perspective,
                    position=result.response,
                    arguments=self._extract_arguments(result.response),
                    confidence=0.5,
                    tokens_used=result.tokens_used
                ))
        
        return positions
    
    async def _conduct_debate_round(
        self,
        query: str,
        round_num: int,
        current_positions: List[DebatePosition],
        perspectives: List[str]
    ) -> List[DebatePosition]:
        """Conduct a round of debate"""
        tasks = []
        
        for i, position in enumerate(current_positions):
            # Build other positions context
            others = []
            for j, other in enumerate(current_positions):
                if i != j:
                    others.append(f"{other.perspective}: {other.position}")
            
            prompt = self.DEBATE_ROUND_PROMPT.format(
                perspective=position.perspective,
                query=query,
                round_num=round_num,
                other_positions="\n\n".join(others)
            )
            
            tasks.append(self._call_llm(
                prompt=prompt,
                agent_id=f"debater-{i+1}-round-{round_num}"
            ))
        
        results = await asyncio.gather(*tasks)
        
        new_positions = []
        for i, (position, result) in enumerate(zip(current_positions, results)):
            if result.success:
                new_positions.append(DebatePosition(
                    agent_id=position.agent_id,
                    perspective=position.perspective,
                    position=result.response,
                    arguments=self._extract_arguments(result.response),
                    confidence=self._calculate_argument_strength(result.response),
                    tokens_used=result.tokens_used
                ))
        
        return new_positions
    
    async def _moderate_debate(
        self,
        query: str,
        debate_history: List[str],
        final_positions: List[DebatePosition]
    ) -> str:
        """Have a moderator synthesize the debate"""
        prompt = self.MODERATOR_PROMPT.format(
            query=query,
            debate_history="\n\n".join(debate_history[-10:]),  # Last 10 entries
            final_positions="\n\n".join([f"{p.perspective}: {p.position}" for p in final_positions])
        )
        
        result = await self._call_llm(
            prompt=prompt,
            agent_id="debate-moderator"
        )
        
        if result.success:
            return result.response
        return self._synthesize_positions(query, final_positions)
    
    def _conduct_voting(self, positions: List[DebatePosition]) -> tuple:
        """Conduct voting based on argument strength"""
        votes = {}
        
        # Vote based on confidence/strength scores
        winner = positions[0] if positions else None
        for pos in positions:
            votes[pos.perspective] = int(pos.confidence * 100)
            if pos.confidence > (winner.confidence if winner else 0):
                winner = pos
        
        return winner.position if winner else "No consensus reached", votes
    
    def _synthesize_positions(self, query: str, positions: List[DebatePosition]) -> str:
        """Synthesize multiple debate positions into a conclusion"""
        if not positions:
            return "No positions available to synthesize."
        
        # Find strongest arguments
        all_arguments = []
        for pos in positions:
            all_arguments.extend(pos.arguments)
        
        # Build synthesis
        synthesis = f"After debate on '{query}':\n\n"
        
        # Add strongest position
        strongest = max(positions, key=lambda p: p.confidence)
        synthesis += f"**Strongest Position ({strongest.perspective}):**\n{strongest.position}\n\n"
        
        # Add key arguments
        if all_arguments:
            synthesis += "**Key Arguments:**\n"
            for arg in all_arguments[:5]:
                synthesis += f"- {arg}\n"
        
        return synthesis
    
    def _extract_arguments(self, text: str) -> List[str]:
        """Extract key arguments from a position"""
        arguments = []
        
        # Look for numbered points or bullet points
        pattern = r'^\s*(?:\d+[.)]\s*|[-•*]\s*)(.+?)$'
        matches = re.findall(pattern, text, re.MULTILINE)
        
        for match in matches:
            if len(match) > 20:
                arguments.append(match.strip())
        
        # If no structured arguments, extract first few sentences
        if not arguments:
            sentences = text.split('. ')
            for sent in sentences[:3]:
                if len(sent) > 20:
                    arguments.append(sent.strip())
        
        return arguments
    
    def _calculate_argument_strength(self, response: str) -> float:
        """Calculate argument strength from response"""
        strength = 0.5
        response_lower = response.lower()
        
        # Evidence indicators
        if any(w in response_lower for w in ["evidence", "study", "data", "research"]):
            strength += 0.15
        
        # Logical structure
        if any(w in response_lower for w in ["therefore", "because", "consequently"]):
            strength += 0.1
        
        # Counter-arguments addressed
        if any(w in response_lower for w in ["however", "although", "counter"]):
            strength += 0.15
        
        # Specific examples
        if any(w in response_lower for w in ["for example", "such as", "instance"]):
            strength += 0.1
        
        return min(strength, 1.0)
    
    def _check_consensus(self, positions: List[DebatePosition]) -> bool:
        """Check if positions have converged to consensus"""
        if len(positions) < 2:
            return True
        
        agreement_count = 0
        for pos in positions:
            if any(w in pos.position.lower() for w in ["agree", "consensus", "common ground"]):
                agreement_count += 1
        
        return agreement_count > len(positions) / 2
    
    def _calculate_debate_confidence(self, positions: List[DebatePosition]) -> float:
        """Calculate overall debate confidence"""
        if not positions:
            return 0.5
        
        avg_confidence = sum(p.confidence for p in positions) / len(positions)
        
        # Bonus for multiple perspectives
        if len(positions) >= 3:
            avg_confidence += 0.1
        
        return min(avg_confidence, 1.0)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompts"""
        if not context:
            return ""
        parts = [f"- {k}: {v}" for k, v in context.items() if k not in ["internal", "system"]]
        return "\n".join(parts)
