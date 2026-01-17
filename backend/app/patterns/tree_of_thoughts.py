"""
Tree of Thoughts Pattern - Branching exploration of solution paths
Implementation for systematic thought exploration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import time

from .base import BasePattern, PatternConfig, PatternResult, ThinkingStep, AgentResult


@dataclass
class ThoughtNode:
    """A node in the thought tree"""
    id: str
    thought: str
    score: float = 0.5
    depth: int = 0
    parent_id: Optional[str] = None
    children: List["ThoughtNode"] = field(default_factory=list)
    tokens_used: int = 0
    is_terminal: bool = False
    explanation: str = ""


@dataclass
class ToTConfig(PatternConfig):
    """Configuration for Tree of Thoughts pattern"""
    max_depth: int = 3
    branching_factor: int = 3
    pruning_threshold: float = 0.3
    exploration_budget: int = 15
    evaluation_method: str = "scoring"  # "scoring", "voting", "llm"
    backtrack_enabled: bool = True


class TreeOfThoughtsPattern(BasePattern):
    """
    Tree of Thoughts pattern implementation.
    
    This pattern explores multiple reasoning paths:
    1. Generate multiple initial thoughts (branches)
    2. Evaluate each thought's promise
    3. Expand promising thoughts, prune weak ones
    4. Select the best path to a solution
    """
    
    BRANCH_PROMPT = """Given this problem: {query}

Current reasoning path: {current_thought}

Generate {num_branches} different next steps or approaches to explore. Each should be:
1. A distinct direction or method
2. Building on the current thought
3. Moving toward a solution

Format as a numbered list:
1. [First approach]
2. [Second approach]
3. [Third approach]"""

    EVALUATE_PROMPT = """Problem: {query}

Proposed thought/approach: {thought}

Evaluate this approach on a scale of 0.0 to 1.0:
- 0.0-0.3: Poor approach, unlikely to lead to solution
- 0.4-0.6: Moderate approach, could work but has issues
- 0.7-1.0: Strong approach, likely to lead to good solution

Score: """

    SYNTHESIZE_PROMPT = """Problem: {query}

Best reasoning path:
{path}

Based on this reasoning path, provide a clear and complete final answer."""

    def __init__(self, llm_manager, config: Optional[ToTConfig] = None):
        super().__init__(llm_manager, config or ToTConfig())
        self.config: ToTConfig = self.config
        
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[str]] = None,
        **kwargs
    ) -> PatternResult:
        """Execute Tree of Thoughts exploration"""
        result = PatternResult(
            pattern_name="TreeOfThoughts",
            final_response="",
            metadata={"query": query, "tree_depth": 0, "thoughts_explored": 0}
        )
        
        start_time = time.time()
        
        # Initialize root node
        root = ThoughtNode(
            id="root",
            thought=query,
            score=1.0,
            depth=0,
            explanation="Initial problem statement"
        )
        
        all_nodes = [root]
        queue = [root]  # Exploration queue
        thoughts_explored = 0
        
        # Main exploration loop
        while queue and thoughts_explored < self.config.exploration_budget:
            # Sort by score (best first)
            queue.sort(key=lambda n: n.score, reverse=True)
            current = queue.pop(0)
            
            # Check depth limit
            if current.depth >= self.config.max_depth:
                current.is_terminal = True
                continue
            
            # Record exploration step
            result.add_step(ThinkingStep.create(
                step_type="thought",
                content=f"Exploring: {current.thought[:100]}...",
                confidence=current.score,
                metadata={"depth": current.depth, "node_id": current.id}
            ))
            
            # Generate branches
            branches = await self._generate_branches(query, current)
            thoughts_explored += len(branches)
            
            # Evaluate and process branches
            for branch in branches:
                # Calculate score
                branch.score = await self._evaluate_thought(query, branch)
                
                # Prune low-scoring branches
                if branch.score < self.config.pruning_threshold:
                    result.add_step(ThinkingStep.create(
                        step_type="evaluation",
                        content=f"Pruned: {branch.thought[:50]}... (score: {branch.score:.2f})",
                        confidence=branch.score,
                        metadata={"pruned": True}
                    ))
                    continue
                
                # Add to tree
                current.children.append(branch)
                all_nodes.append(branch)
                result.total_tokens += branch.tokens_used
                
                # Check if terminal
                if self._is_terminal_thought(branch.thought):
                    branch.is_terminal = True
                else:
                    queue.append(branch)
            
            # Update tree depth
            if current.depth + 1 > result.metadata["tree_depth"]:
                result.metadata["tree_depth"] = current.depth + 1
        
        result.metadata["thoughts_explored"] = thoughts_explored
        
        # Find best path
        best_path = self._find_best_path(root)
        
        if best_path:
            # Synthesize final solution
            path_str = "\n".join([f"{i+1}. {node.thought}" for i, node in enumerate(best_path) if node.id != "root"])
            
            synth_result = await self._call_llm(
                prompt=self.SYNTHESIZE_PROMPT.format(query=query, path=path_str),
                agent_id="tot-synthesis"
            )
            result.add_agent_result(synth_result)
            
            if synth_result.success:
                result.final_response = synth_result.response
                result.confidence = self._calculate_path_confidence(best_path)
            else:
                result.final_response = best_path[-1].thought if best_path else "No solution found"
        else:
            result.final_response = "No viable solution path found"
            result.confidence = 0.0
        
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        return result
    
    async def _generate_branches(self, query: str, parent: ThoughtNode) -> List[ThoughtNode]:
        """Generate child thoughts from parent node"""
        prompt = self.BRANCH_PROMPT.format(
            query=query,
            current_thought=parent.thought,
            num_branches=self.config.branching_factor
        )
        
        agent_result = await self._call_llm(
            prompt=prompt,
            agent_id=f"tot-branch-{parent.id}"
        )
        
        if not agent_result.success:
            return []
        
        # Parse branches from response
        thoughts = self._parse_branches(agent_result.response)
        
        branches = []
        for i, thought in enumerate(thoughts[:self.config.branching_factor]):
            branch = ThoughtNode(
                id=f"{parent.id}-{i}",
                thought=thought,
                depth=parent.depth + 1,
                parent_id=parent.id,
                tokens_used=agent_result.tokens_used // max(len(thoughts), 1),
                explanation=f"Branch {i+1} from {parent.id}"
            )
            branches.append(branch)
        
        return branches
    
    def _parse_branches(self, response: str) -> List[str]:
        """Parse branches from LLM response"""
        thoughts = []
        
        # Look for numbered items
        pattern = r'^\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\n\n|$)'
        matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            thought = match.strip()
            if len(thought) > 10:
                thoughts.append(thought)
        
        # If no numbered format, split by newlines
        if not thoughts:
            lines = response.split('\n')
            for line in lines:
                line = line.strip().lstrip('-•*')
                if len(line) > 20:
                    thoughts.append(line)
        
        return thoughts
    
    async def _evaluate_thought(self, query: str, node: ThoughtNode) -> float:
        """Evaluate a thought node's promise"""
        if self.config.evaluation_method == "scoring":
            return self._heuristic_score(node.thought)
        elif self.config.evaluation_method == "llm":
            return await self._llm_score(query, node)
        else:
            return 0.5  # Default
    
    def _heuristic_score(self, thought: str) -> float:
        """Simple heuristic scoring"""
        score = 0.5
        thought_lower = thought.lower()
        
        # Solution indicators
        if any(w in thought_lower for w in ["therefore", "solution", "answer", "result"]):
            score += 0.2
        
        # Logical progression
        if any(w in thought_lower for w in ["because", "since", "thus", "hence"]):
            score += 0.1
        
        # Concrete steps
        if any(w in thought_lower for w in ["step", "first", "next", "then"]):
            score += 0.1
        
        # Uncertainty penalty
        if any(w in thought_lower for w in ["maybe", "perhaps", "might", "possibly"]):
            score -= 0.1
        
        return max(0, min(1, score))
    
    async def _llm_score(self, query: str, node: ThoughtNode) -> float:
        """Use LLM to score a thought"""
        prompt = self.EVALUATE_PROMPT.format(query=query, thought=node.thought)
        
        agent_result = await self._call_llm(
            prompt=prompt,
            agent_id=f"tot-eval-{node.id}",
            max_tokens=50
        )
        
        if not agent_result.success:
            return 0.5
        
        # Extract score from response
        try:
            score_match = re.search(r'(\d+\.?\d*)', agent_result.response)
            if score_match:
                score = float(score_match.group(1))
                return max(0, min(1, score))
        except:
            pass
        
        return 0.5
    
    def _is_terminal_thought(self, thought: str) -> bool:
        """Check if a thought represents a final solution"""
        thought_lower = thought.lower()
        
        solution_keywords = [
            "the answer is", "therefore", "in conclusion",
            "final answer", "solution:", "result:"
        ]
        
        dead_end_keywords = [
            "impossible", "cannot be solved", "no solution"
        ]
        
        return any(kw in thought_lower for kw in solution_keywords + dead_end_keywords)
    
    def _find_best_path(self, root: ThoughtNode) -> List[ThoughtNode]:
        """Find the highest scoring path through the tree"""
        best_path = []
        best_score = 0.0
        
        def dfs(node: ThoughtNode, current_path: List[ThoughtNode], current_score: float):
            nonlocal best_path, best_score
            
            current_path = current_path + [node]
            current_score += node.score
            
            # Check if leaf node
            if node.is_terminal or not node.children:
                avg_score = current_score / len(current_path)
                if avg_score > best_score:
                    best_score = avg_score
                    best_path = current_path.copy()
                return
            
            # Explore children
            for child in node.children:
                dfs(child, current_path, current_score)
        
        dfs(root, [], 0)
        return best_path
    
    def _calculate_path_confidence(self, path: List[ThoughtNode]) -> float:
        """Calculate confidence for a solution path"""
        if not path:
            return 0.0
        
        total_score = sum(node.score for node in path)
        avg_score = total_score / len(path)
        
        # Depth penalty
        depth_penalty = 1.0 / (1.0 + len(path) * 0.1)
        
        return avg_score * depth_penalty
