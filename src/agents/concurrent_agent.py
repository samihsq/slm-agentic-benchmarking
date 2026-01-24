"""
Concurrent Agent for Benchmarking.

Multiple specialized agents work in parallel, then a synthesizer
combines their outputs into a final response.
"""

from typing import Optional, Dict, Any

from crewai import Agent, Task, Crew, Process
from .base_agent import BaseAgent, BenchmarkResponse
from ..config import get_llm


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
        self._setup_agents()

    def _setup_agents(self):
        """Initialize the concurrent agents."""

        self.analyst = Agent(
            role="Task Analyst",
            goal="Analyze task requirements and identify key components",
            backstory="Expert at breaking down complex tasks into components.",
            llm=self.llm,
            verbose=self.verbose,
        )

        self.researcher = Agent(
            role="Information Researcher",
            goal="Gather relevant information and context for the task",
            backstory="Expert at finding and organizing relevant information.",
            llm=self.llm,
            verbose=self.verbose,
        )

        self.critic = Agent(
            role="Critical Reviewer",
            goal="Identify potential issues and suggest improvements",
            backstory="Expert at critical analysis and quality assurance.",
            llm=self.llm,
            verbose=self.verbose,
        )

        self.synthesizer = Agent(
            role="Response Synthesizer",
            goal="Combine all inputs into a coherent final response",
            backstory="Expert at synthesizing multiple perspectives into clear responses.",
            llm=self.llm,
            verbose=self.verbose,
        )

    def respond_to_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResponse:
        """Process task with concurrent agents."""

        context_str = ""
        if context:
            if "tools" in context:
                context_str += f"\n\nAvailable tools: {context['tools']}"
            if "patient_data" in context:
                context_str += f"\n\nPatient data: {context['patient_data']}"

        full_task = f"{task}{context_str}"

        # Concurrent tasks
        analyze = Task(
            description=f"Analyze this task and identify key components:\n{full_task}",
            expected_output="Analysis of task components",
            agent=self.analyst,
        )

        research = Task(
            description=f"Research relevant information for:\n{full_task}",
            expected_output="Relevant information and context",
            agent=self.researcher,
        )

        critique = Task(
            description=f"Identify potential issues or considerations for:\n{full_task}",
            expected_output="Critical analysis and considerations",
            agent=self.critic,
        )

        # Synthesis task (depends on concurrent tasks)
        synthesize = Task(
            description="""Synthesize all inputs into a final response as JSON:
{{"reasoning": "<synthesis of all perspectives>", "confidence": <0.0-1.0>, "response": "<complete response>"}}

ONLY output the JSON object.""",
            expected_output="Final synthesized response",
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

        self.add_to_history(
            task=task,
            response=response.response,
            reasoning=response.reasoning,
            success=response.success,
        )

        return response
