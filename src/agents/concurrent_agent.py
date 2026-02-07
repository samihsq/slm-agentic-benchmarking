"""
Concurrent Agent for Benchmarking.

Multiple specialized agents work in parallel, then a synthesizer
combines their outputs into a final response.
"""

from typing import Optional, Dict, Any, List

from crewai import Agent, Task, Crew, Process
from .base_agent import BaseAgent, BenchmarkResponse
from ..config import get_llm
from ..utils.trace import TraceCapture


class ConcurrentAgent(BaseAgent):
    """
    Concurrent orchestration with parallel specialized agents.

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                 Concurrent Agents                       │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
    │  │   Analyst   │  │  Researcher │  │   Critic    │    │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
    │         └────────────────┼────────────────┘            │
    │                          ▼                             │
    │                   ┌──────────┐                         │
    │                   │Synthesizer│→ Output                │
    │                   └──────────┘                         │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        model: str = "phi-4",
        verbose: bool = False,
        max_iterations: int = 1,
    ):
        super().__init__(
            model=model,
            verbose=verbose,
            max_iterations=max_iterations,
        )

        self.llm = get_llm(model)

    def _setup_agents(self, benchmark_type: str = "general"):
        """Initialize the concurrent agents with benchmark-specific prompts."""
        
        # Get prompts for this benchmark type
        analyst_prompt = self.get_system_prompt(benchmark_type, "concurrent_analyst")
        researcher_prompt = self.get_system_prompt(benchmark_type, "concurrent_researcher")
        critic_prompt = self.get_system_prompt(benchmark_type, "concurrent_critic")
        synthesizer_prompt = self.get_system_prompt(benchmark_type, "concurrent_synthesizer")

        self.analyst = Agent(
            role="Task Analyst",
            goal="Analyze task requirements and identify key components",
            backstory=analyst_prompt,
            llm=self.llm,
            verbose=self.verbose,
        )

        self.researcher = Agent(
            role="Information Researcher",
            goal="Gather relevant information and context for the task",
            backstory=researcher_prompt,
            llm=self.llm,
            verbose=self.verbose,
        )

        self.critic = Agent(
            role="Critical Reviewer",
            goal="Identify potential issues and suggest improvements",
            backstory=critic_prompt,
            llm=self.llm,
            verbose=self.verbose,
        )

        self.synthesizer = Agent(
            role="Response Synthesizer",
            goal="Combine all inputs into a coherent final response",
            backstory=synthesizer_prompt,
            llm=self.llm,
            verbose=self.verbose,
        )

    def respond_to_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResponse:
        """Process task with concurrent agents."""
        
        # Determine benchmark type and setup agents with appropriate prompts
        benchmark_type = (context or {}).get("benchmark_type", "general")
        self._setup_agents(benchmark_type)

        context_str = ""
        if context:
            if "tools" in context:
                context_str += f"\n\nAvailable tools: {context['tools']}"
            if "patient_data" in context:
                context_str += f"\n\nPatient data: {context['patient_data']}"
            if "additional_info" in context:
                context_str += f"\n\n{context['additional_info']}"

        full_task = f"{task}{context_str}"

        # Concurrent tasks
        analyze = Task(
            description=f"Analyze this task following your role guidelines:\n\nTASK: {full_task}",
            expected_output="Analysis of task components",
            agent=self.analyst,
        )

        research = Task(
            description=f"Research relevant information following your role guidelines:\n\nTASK: {full_task}",
            expected_output="Relevant information and context",
            agent=self.researcher,
        )

        critique = Task(
            description=f"Provide critical analysis following your role guidelines:\n\nTASK: {full_task}",
            expected_output="Critical analysis and considerations",
            agent=self.critic,
        )

        # Synthesis task (depends on concurrent tasks) - use benchmark-specific output format
        synthesizer_prompt = self.get_system_prompt(benchmark_type, "concurrent_synthesizer")
        synthesize = Task(
            description=f"""Synthesize all inputs into a final response.

ORIGINAL TASK: {full_task}

Based on the analysis, research, and critical review, generate your final response.

{synthesizer_prompt}""",
            expected_output="Final synthesized response as JSON",
            agent=self.synthesizer,
            context=[analyze, research, critique],
        )

        crew = Crew(
            agents=[self.analyst, self.researcher, self.critic, self.synthesizer],
            tasks=[analyze, research, critique, synthesize],
            process=Process.sequential,  # CrewAI handles concurrent via context
            verbose=self.verbose,
        )

        result = crew.kickoff()
        response = self.parse_json_response(str(result))
        
        # Capture the parallel agent outputs
        task_descriptions = [
            analyze.description,
            research.description,
            critique.description,
            synthesize.description,
        ]
        parallel_outputs = []
        for i, task_obj in enumerate([analyze, research, critique, synthesize]):
            task_output = getattr(task_obj, 'output', None)
            output_str = ""
            if task_output:
                output_str = str(task_output.raw) if hasattr(task_output, 'raw') else str(task_output)
            
            step = {
                "agent": task_obj.agent.role if task_obj.agent else "unknown",
                "input": task_descriptions[i][:1000],
                "output": output_str,
            }
            parallel_outputs.append(step)
            
            # Record to trace if capture is active
            TraceCapture.record(
                role=step["agent"],
                input_prompt=step["input"],
                output_response=output_str,
            )
        
        # Add to metadata
        if response.metadata is None:
            response.metadata = {}
        response.metadata["parallel_outputs"] = parallel_outputs

        self.add_to_history(
            task=task,
            response=response.response,
            reasoning=response.reasoning,
            success=response.success,
        )

        return response
