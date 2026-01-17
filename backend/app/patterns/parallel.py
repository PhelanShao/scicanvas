"""
Parallel Execution - Concurrent multi-agent execution
Implementation for parallel task processing.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Coroutine
import asyncio
import time
from enum import Enum

from .base import AgentResult, PatternResult, ThinkingStep


class ExecutionStrategy(Enum):
    """Execution strategy for parallel tasks"""
    ALL = "all"              # Wait for all tasks
    FIRST_SUCCESS = "first"  # Return on first success
    MAJORITY = "majority"    # Wait for majority to complete
    THRESHOLD = "threshold"  # Wait until threshold met


@dataclass
class ParallelTask:
    """A task to be executed in parallel"""
    task_id: str
    coroutine: Coroutine
    priority: int = 0
    timeout_seconds: int = 60
    required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelConfig:
    """Configuration for parallel execution"""
    max_concurrency: int = 5
    strategy: ExecutionStrategy = ExecutionStrategy.ALL
    timeout_seconds: int = 120
    fail_fast: bool = False
    success_threshold: float = 0.5  # For THRESHOLD strategy
    collect_all_results: bool = True


class ParallelExecutor:
    """
    Parallel execution engine for multi-agent tasks.
    
    Features:
    - Configurable concurrency limits
    - Multiple execution strategies
    - Timeout handling per task and overall
    - Result collection and aggregation
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        
    async def execute(
        self,
        tasks: List[ParallelTask],
        **kwargs
    ) -> PatternResult:
        """Execute tasks in parallel according to configuration"""
        result = PatternResult(
            pattern_name="ParallelExecution",
            final_response="",
            metadata={
                "total_tasks": len(tasks),
                "completed": 0,
                "failed": 0,
                "strategy": self.config.strategy.value
            }
        )
        
        start_time = time.time()
        
        if not tasks:
            result.final_response = "No tasks to execute"
            return result
        
        # Sort by priority (higher first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        # Execute based on strategy
        if self.config.strategy == ExecutionStrategy.FIRST_SUCCESS:
            agent_results = await self._execute_first_success(sorted_tasks, semaphore)
        elif self.config.strategy == ExecutionStrategy.MAJORITY:
            agent_results = await self._execute_majority(sorted_tasks, semaphore)
        elif self.config.strategy == ExecutionStrategy.THRESHOLD:
            agent_results = await self._execute_threshold(sorted_tasks, semaphore)
        else:  # ALL
            agent_results = await self._execute_all(sorted_tasks, semaphore)
        
        # Process results
        for agent_result in agent_results:
            result.add_agent_result(agent_result)
            if agent_result.success:
                result.metadata["completed"] += 1
            else:
                result.metadata["failed"] += 1
        
        # Build final response
        successful_results = [r for r in agent_results if r.success]
        if successful_results:
            result.final_response = self._aggregate_results(successful_results)
            result.success = True
            result.confidence = len(successful_results) / len(tasks)
        else:
            result.final_response = "All parallel tasks failed"
            result.success = False
            result.confidence = 0.0
        
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        return result
    
    async def _execute_all(
        self,
        tasks: List[ParallelTask],
        semaphore: asyncio.Semaphore
    ) -> List[AgentResult]:
        """Execute all tasks and wait for completion"""
        
        async def run_with_semaphore(task: ParallelTask) -> AgentResult:
            async with semaphore:
                return await self._run_task(task)
        
        try:
            # Run with overall timeout
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[run_with_semaphore(task) for task in tasks],
                    return_exceptions=True
                ),
                timeout=self.config.timeout_seconds
            )
            
            # Convert exceptions to failed results
            agent_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    agent_results.append(AgentResult(
                        agent_id=tasks[i].task_id,
                        response="",
                        success=False,
                        error=str(result)
                    ))
                else:
                    agent_results.append(result)
            
            return agent_results
            
        except asyncio.TimeoutError:
            return [AgentResult(
                agent_id="parallel",
                response="",
                success=False,
                error="Overall timeout exceeded"
            )]
    
    async def _execute_first_success(
        self,
        tasks: List[ParallelTask],
        semaphore: asyncio.Semaphore
    ) -> List[AgentResult]:
        """Return as soon as first task succeeds"""
        results = []
        
        async def run_with_semaphore(task: ParallelTask) -> AgentResult:
            async with semaphore:
                return await self._run_task(task)
        
        # Create tasks
        pending = {
            asyncio.create_task(run_with_semaphore(task)): task
            for task in tasks
        }
        
        try:
            while pending:
                done, pending_set = await asyncio.wait(
                    pending.keys(),
                    timeout=self.config.timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    result = task.result()
                    results.append(result)
                    
                    if result.success:
                        # Cancel remaining tasks
                        for p in pending_set:
                            p.cancel()
                        return results
                
                pending = {t: pending[t] for t in pending_set}
            
        except Exception as e:
            results.append(AgentResult(
                agent_id="parallel",
                response="",
                success=False,
                error=str(e)
            ))
        
        return results
    
    async def _execute_majority(
        self,
        tasks: List[ParallelTask],
        semaphore: asyncio.Semaphore
    ) -> List[AgentResult]:
        """Wait for majority of tasks to complete"""
        target = len(tasks) // 2 + 1
        return await self._execute_until_count(tasks, semaphore, target)
    
    async def _execute_threshold(
        self,
        tasks: List[ParallelTask],
        semaphore: asyncio.Semaphore
    ) -> List[AgentResult]:
        """Wait until success threshold is met"""
        target = int(len(tasks) * self.config.success_threshold)
        target = max(1, target)  # At least 1
        return await self._execute_until_count(tasks, semaphore, target)
    
    async def _execute_until_count(
        self,
        tasks: List[ParallelTask],
        semaphore: asyncio.Semaphore,
        target_count: int
    ) -> List[AgentResult]:
        """Execute until target number of successes"""
        results = []
        success_count = 0
        
        async def run_with_semaphore(task: ParallelTask) -> AgentResult:
            async with semaphore:
                return await self._run_task(task)
        
        pending = {
            asyncio.create_task(run_with_semaphore(task)): task
            for task in tasks
        }
        
        try:
            while pending and success_count < target_count:
                done, pending_set = await asyncio.wait(
                    pending.keys(),
                    timeout=self.config.timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if not done:  # Timeout
                    break
                
                for task in done:
                    try:
                        result = task.result()
                        results.append(result)
                        if result.success:
                            success_count += 1
                    except Exception as e:
                        results.append(AgentResult(
                            agent_id="unknown",
                            response="",
                            success=False,
                            error=str(e)
                        ))
                
                pending = {t: pending[t] for t in pending_set}
            
            # Cancel remaining if threshold met
            for task in pending.keys():
                task.cancel()
                
        except Exception as e:
            results.append(AgentResult(
                agent_id="parallel",
                response="",
                success=False,
                error=str(e)
            ))
        
        return results
    
    async def _run_task(self, task: ParallelTask) -> AgentResult:
        """Run a single task with timeout"""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                task.coroutine,
                timeout=task.timeout_seconds
            )
            
            # If result is already an AgentResult, return it
            if isinstance(result, AgentResult):
                return result
            
            # Otherwise wrap in AgentResult
            return AgentResult(
                agent_id=task.task_id,
                response=str(result),
                success=True,
                duration_ms=int((time.time() - start_time) * 1000),
                metadata=task.metadata
            )
            
        except asyncio.TimeoutError:
            return AgentResult(
                agent_id=task.task_id,
                response="",
                success=False,
                error=f"Task timeout after {task.timeout_seconds}s",
                duration_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            return AgentResult(
                agent_id=task.task_id,
                response="",
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )
    
    def _aggregate_results(self, results: List[AgentResult]) -> str:
        """Aggregate successful results into a combined response"""
        if not results:
            return "No results"
        
        if len(results) == 1:
            return results[0].response
        
        # Combine multiple results
        combined = []
        for i, result in enumerate(results):
            combined.append(f"[Result {i+1}]\n{result.response}")
        
        return "\n\n".join(combined)


async def execute_parallel_agents(
    agent_tasks: List[Callable[[], Coroutine]],
    max_concurrency: int = 5,
    timeout_seconds: int = 120
) -> List[AgentResult]:
    """
    Convenience function to execute agent tasks in parallel.
    
    Args:
        agent_tasks: List of async functions that return AgentResult
        max_concurrency: Maximum concurrent tasks
        timeout_seconds: Overall timeout
        
    Returns:
        List of AgentResult from all tasks
    """
    executor = ParallelExecutor(ParallelConfig(
        max_concurrency=max_concurrency,
        timeout_seconds=timeout_seconds,
        strategy=ExecutionStrategy.ALL
    ))
    
    tasks = [
        ParallelTask(
            task_id=f"agent-{i}",
            coroutine=task(),
            timeout_seconds=timeout_seconds
        )
        for i, task in enumerate(agent_tasks)
    ]
    
    result = await executor.execute(tasks)
    return result.agent_results
