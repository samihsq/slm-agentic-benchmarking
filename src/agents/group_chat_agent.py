"""
Group Chat Agent for Benchmarking.

Agents collaborate through discussion to reach a consensus response.
"""

from typing import Optional, Dict, Any

from crewai import Agent, Task, Crew, Process
from .base_agent import BaseAgent, BenchmarkResponse
from ..config import get_llm


class GroupChatAgent(BaseAgent):
    """
    Group chat orchestration where agents discuss and collaborate.

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                  Group Chat                              │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
    │  │ Proposer │←→│  Critic  │←→│ Advisor  │             │
    │  └──────────┘  └──────────┘  └──────────┘             │
    │       ↓              ↓             ↓                   │
    │              Shared Discussion                         │
    │                     ↓                                  │
    │              [Moderator] → Output                      │
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
        """Initialize the group chat agents."""

        self.proposer = Agent(
            role="Solution Proposer",
            goal="Propose initial solutions and approaches to tasks",
            backstory="Proactive problem solver who generates initial approaches.",
            llm=self.llm,
            verbose=self.verbose,
        )

        self.critic = Agent(
            role="Critical Analyst",
            goal="Challenge proposals and identify weaknesses",
            backstory="Skeptical analyst who ensures quality through criticism.",
            llm=self.llm,
            verbose=self.verbose,
        )

        self.advisor = Agent(
            role="Expert Advisor",
            goal="Provide expert guidance and refinements",
            backstory="Experienced advisor who improves solutions with expertise.",
            llm=self.llm,
            verbose=self.verbose,
        )

        self.moderator = Agent(
            role="Discussion Moderator",
            goal="Synthesize discussion into final response",
            backstory="Facilitator who captures consensus and produces final output.",
            llm=self.llm,
            verbose=self.verbose,
        )

    def respond_to_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResponse:
        """Process task through group discussion."""

        context_str = ""
        if context:
            if "tools" in context:
                context_str += f"\n\nAvailable tools: {context['tools']}"
            if "patient_data" in context:
                context_str += f"\n\nPatient data: {context['patient_data']}"

        full_task = f"{task}{context_str}"

        # Group discussion tasks
        propose = Task(
            description=f"Propose an initial solution for:\n{full_task}",
            expected_output="Initial proposed solution",
            agent=self.proposer,
        )

        critique = Task(
            description="Critically analyze the proposal. What are the weaknesses?",
            expected_output="Critical analysis of proposal",
            agent=self.critic,
            context=[propose],
        )

        advise = Task(
            description="Provide expert advice to improve the solution.",
            expected_output="Expert guidance and refinements",
            agent=self.advisor,
            context=[propose, critique],
        )

        moderate = Task(
            description="""Synthesize the discussion into a final response as JSON:
{{"reasoning": "<synthesis of discussion>", "confidence": <0.0-1.0>, "response": "<final consensus response>"}}

ONLY output the JSON object.""",
            expected_output="Final consensus response",
            agent=self.moderator,
            context=[propose, critique, advise],
        )

        crew = Crew(
            agents=[self.proposer, self.critic, self.advisor, self.moderator],
            tasks=[propose, critique, advise, moderate],
            process=Process.sequential,
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
