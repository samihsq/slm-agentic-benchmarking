"""
Group Chat Agent for Benchmarking.

Agents collaborate through discussion to reach a consensus response.
"""

from typing import Optional, Dict, Any, List

from crewai import Agent, Task, Crew, Process
from .base_agent import BaseAgent, BenchmarkResponse, kickoff_with_timeout
from ..config import get_llm
from ..utils.trace import TraceCapture


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

    def _setup_agents(self, benchmark_type: str = "general"):
        """Initialize the group chat agents with benchmark-specific prompts."""
        
        # Get prompts for this benchmark type
        proposer_prompt = self.get_system_prompt(benchmark_type, "groupchat_proposer")
        critic_prompt = self.get_system_prompt(benchmark_type, "groupchat_critic")
        advisor_prompt = self.get_system_prompt(benchmark_type, "groupchat_advisor")
        moderator_prompt = self.get_system_prompt(benchmark_type, "groupchat_moderator")

        self.proposer = Agent(
            role="Solution Proposer",
            goal="Propose initial solutions and approaches to tasks",
            backstory=proposer_prompt,
            llm=self.llm,
            verbose=self.verbose,
            max_iter=1,
        )

        self.critic = Agent(
            role="Critical Analyst",
            goal="Challenge proposals and identify weaknesses",
            backstory=critic_prompt,
            llm=self.llm,
            verbose=self.verbose,
            max_iter=1,
        )

        self.advisor = Agent(
            role="Expert Advisor",
            goal="Provide expert guidance and refinements",
            backstory=advisor_prompt,
            llm=self.llm,
            verbose=self.verbose,
            max_iter=1,
        )

        self.moderator = Agent(
            role="Discussion Moderator",
            goal="Synthesize discussion into final response",
            backstory=moderator_prompt,
            llm=self.llm,
            verbose=self.verbose,
            max_iter=1,
        )

    def respond_to_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResponse:
        """Process task through group discussion."""
        
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

        # Group discussion tasks
        propose = Task(
            description=f"Propose an initial solution following your role guidelines:\n\nTASK: {full_task}",
            expected_output="Initial proposed solution",
            agent=self.proposer,
        )

        critique = Task(
            description=f"Critically analyze the proposal following your role guidelines.\n\nORIGINAL TASK: {full_task}",
            expected_output="Critical analysis of proposal",
            agent=self.critic,
            context=[propose],
        )

        advise = Task(
            description=f"Provide expert advice following your role guidelines.\n\nORIGINAL TASK: {full_task}",
            expected_output="Expert guidance and refinements",
            agent=self.advisor,
            context=[propose, critique],
        )

        # Moderation task - use benchmark-specific output format
        moderator_prompt = self.get_system_prompt(benchmark_type, "groupchat_moderator")
        moderate = Task(
            description=f"""Synthesize the discussion into a final response.

ORIGINAL TASK: {full_task}

Based on all perspectives shared, generate your final response.

{moderator_prompt}""",
            expected_output="Final consensus response as JSON",
            agent=self.moderator,
            context=[propose, critique, advise],
        )

        crew = Crew(
            agents=[self.proposer, self.critic, self.advisor, self.moderator],
            tasks=[propose, critique, advise, moderate],
            process=Process.sequential,
            verbose=self.verbose,
            max_execution_time=600,
        )

        result, timed_out = kickoff_with_timeout(crew)
        result_str = "" if timed_out else str(result)
        response = self.parse_json_response(result_str)
        
        # Capture the full discussion from each task
        task_descriptions = [
            propose.description,
            critique.description,
            advise.description,
            moderate.description,
        ]
        discussion = []
        for i, task_obj in enumerate([propose, critique, advise, moderate]):
            task_output = getattr(task_obj, 'output', None)
            output_str = ""
            if task_output:
                output_str = str(task_output.raw) if hasattr(task_output, 'raw') else str(task_output)
            
            turn = {
                "agent": task_obj.agent.role if task_obj.agent else "unknown",
                "input": task_descriptions[i][:1000],
                "output": output_str,
            }
            discussion.append(turn)
            
            # Record to trace if capture is active
            TraceCapture.record(
                role=turn["agent"],
                input_prompt=turn["input"],
                output_response=output_str,
            )
        
        # Add discussion to metadata
        if response.metadata is None:
            response.metadata = {}
        response.metadata["discussion"] = discussion
        response.metadata["timed_out"] = timed_out

        self.add_to_history(
            task=task,
            response=response.response,
            reasoning=response.reasoning,
            success=response.success,
        )

        return response
