"""
Ollama-backed agentic architectures for benchmarking.

Sequential, Concurrent, and GroupChat agents using CrewAI
configured with an Ollama LLM backend via LiteLLM.
"""

from typing import Optional, Dict, Any

from crewai import Agent, Task, Crew, Process, LLM

from .base_agent import BaseAgent, BenchmarkResponse
from ..config.azure_llm_config import OLLAMA_MODELS
from ..utils.trace import TraceCapture


OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"


def _build_ollama_llm(model_key: str, ollama_base_url: str) -> LLM:
    """Return a CrewAI LLM pointed at a local/remote Ollama instance."""
    if model_key in OLLAMA_MODELS:
        model_name = OLLAMA_MODELS[model_key]["model"]
    else:
        model_name = model_key
    return LLM(model=f"ollama/{model_name}", base_url=ollama_base_url)


class OllamaSequentialAgent(BaseAgent):
    """
    Sequential pipeline (Analyzer → Evaluator → Responder) backed by Ollama.

    Drop-in Ollama equivalent of SequentialAgent; uses CrewAI with LiteLLM
    ollama/ routing instead of Azure get_llm().
    """

    def __init__(
        self,
        model: str = "dasd-4b",
        verbose: bool = False,
        max_iterations: int = 1,
        ollama_base_url: str = OLLAMA_BASE_URL_DEFAULT,
    ):
        super().__init__(model=model, verbose=verbose, max_iterations=max_iterations)
        self.ollama_base_url = ollama_base_url
        self.llm = _build_ollama_llm(model, ollama_base_url)

    def _setup_agents(self, benchmark_type: str = "general"):
        self.analyzer = Agent(
            role="Task Analyzer",
            goal="Analyze the nature and requirements of tasks",
            backstory=self.get_system_prompt(benchmark_type, "sequential_analyzer"),
            llm=self.llm,
            verbose=self.verbose,
        )
        self.evaluator = Agent(
            role="Approach Evaluator",
            goal="Determine the best approach to complete the task",
            backstory=self.get_system_prompt(benchmark_type, "sequential_evaluator"),
            llm=self.llm,
            verbose=self.verbose,
        )
        self.responder = Agent(
            role="Response Generator",
            goal="Execute the approach and provide the final response",
            backstory=self.get_system_prompt(benchmark_type, "sequential_responder"),
            llm=self.llm,
            verbose=self.verbose,
        )

    def respond_to_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> BenchmarkResponse:
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

        analyze_task = Task(
            description=f"Analyze this task:\n\nTASK: {full_task}\n\nProvide your analysis following your role guidelines.",
            expected_output="Analysis of the task's nature and requirements",
            agent=self.analyzer,
        )
        evaluate_task = Task(
            description=f"Evaluate how to approach this task:\n\nTASK: {full_task}\n\nBased on the analysis, provide your evaluation following your role guidelines.",
            expected_output="Evaluation of the best approach",
            agent=self.evaluator,
            context=[analyze_task],
        )
        respond_task = Task(
            description=f"Complete the task and provide your final answer.\n\nORIGINAL TASK: {full_task}\n\nBased on the analysis and evaluation, generate your final response.\n\n{self.get_system_prompt(benchmark_type, 'sequential_responder')}",
            expected_output="JSON with reasoning and the final answer",
            agent=self.responder,
            context=[analyze_task, evaluate_task],
        )

        crew = Crew(
            agents=[self.analyzer, self.evaluator, self.responder],
            tasks=[analyze_task, evaluate_task, respond_task],
            process=Process.sequential,
            verbose=self.verbose,
        )
        result = crew.kickoff()
        response = self.parse_json_response(str(result))

        for task_obj in [analyze_task, evaluate_task, respond_task]:
            task_output = getattr(task_obj, "output", None)
            output_str = str(task_output.raw) if task_output and hasattr(task_output, "raw") else str(task_output or "")
            TraceCapture.record(
                role=task_obj.agent.role if task_obj.agent else "unknown",
                input_prompt=task_obj.description[:1000],
                output_response=output_str,
            )

        if response.metadata is None:
            response.metadata = {}
        self.add_to_history(task=task, response=response.response, reasoning=response.reasoning, success=response.success)
        return response


class OllamaConcurrentAgent(BaseAgent):
    """
    Concurrent agents (Analyst + Researcher + Critic → Synthesizer) backed by Ollama.
    """

    def __init__(
        self,
        model: str = "dasd-4b",
        verbose: bool = False,
        max_iterations: int = 1,
        ollama_base_url: str = OLLAMA_BASE_URL_DEFAULT,
    ):
        super().__init__(model=model, verbose=verbose, max_iterations=max_iterations)
        self.ollama_base_url = ollama_base_url
        self.llm = _build_ollama_llm(model, ollama_base_url)

    def _setup_agents(self, benchmark_type: str = "general"):
        self.analyst = Agent(
            role="Task Analyst",
            goal="Analyze task requirements and identify key components",
            backstory=self.get_system_prompt(benchmark_type, "concurrent_analyst"),
            llm=self.llm,
            verbose=self.verbose,
        )
        self.researcher = Agent(
            role="Information Researcher",
            goal="Gather relevant information and context for the task",
            backstory=self.get_system_prompt(benchmark_type, "concurrent_researcher"),
            llm=self.llm,
            verbose=self.verbose,
        )
        self.critic = Agent(
            role="Critical Reviewer",
            goal="Identify potential issues and suggest improvements",
            backstory=self.get_system_prompt(benchmark_type, "concurrent_critic"),
            llm=self.llm,
            verbose=self.verbose,
        )
        self.synthesizer = Agent(
            role="Response Synthesizer",
            goal="Combine all inputs into a coherent final response",
            backstory=self.get_system_prompt(benchmark_type, "concurrent_synthesizer"),
            llm=self.llm,
            verbose=self.verbose,
        )

    def respond_to_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> BenchmarkResponse:
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
        synthesize = Task(
            description=f"Synthesize all inputs into a final response.\n\nORIGINAL TASK: {full_task}\n\nBased on the analysis, research, and critical review, generate your final response.\n\n{self.get_system_prompt(benchmark_type, 'concurrent_synthesizer')}",
            expected_output="Final synthesized response as JSON",
            agent=self.synthesizer,
            context=[analyze, research, critique],
        )

        crew = Crew(
            agents=[self.analyst, self.researcher, self.critic, self.synthesizer],
            tasks=[analyze, research, critique, synthesize],
            process=Process.sequential,
            verbose=self.verbose,
        )
        result = crew.kickoff()
        response = self.parse_json_response(str(result))

        for task_obj in [analyze, research, critique, synthesize]:
            task_output = getattr(task_obj, "output", None)
            output_str = str(task_output.raw) if task_output and hasattr(task_output, "raw") else str(task_output or "")
            TraceCapture.record(
                role=task_obj.agent.role if task_obj.agent else "unknown",
                input_prompt=task_obj.description[:1000],
                output_response=output_str,
            )

        if response.metadata is None:
            response.metadata = {}
        self.add_to_history(task=task, response=response.response, reasoning=response.reasoning, success=response.success)
        return response


class OllamaGroupChatAgent(BaseAgent):
    """
    Group chat (Proposer → Critic → Advisor → Moderator) backed by Ollama.
    """

    def __init__(
        self,
        model: str = "dasd-4b",
        verbose: bool = False,
        max_iterations: int = 1,
        ollama_base_url: str = OLLAMA_BASE_URL_DEFAULT,
    ):
        super().__init__(model=model, verbose=verbose, max_iterations=max_iterations)
        self.ollama_base_url = ollama_base_url
        self.llm = _build_ollama_llm(model, ollama_base_url)

    def _setup_agents(self, benchmark_type: str = "general"):
        self.proposer = Agent(
            role="Solution Proposer",
            goal="Propose initial solutions and approaches to tasks",
            backstory=self.get_system_prompt(benchmark_type, "groupchat_proposer"),
            llm=self.llm,
            verbose=self.verbose,
        )
        self.critic = Agent(
            role="Critical Analyst",
            goal="Challenge proposals and identify weaknesses",
            backstory=self.get_system_prompt(benchmark_type, "groupchat_critic"),
            llm=self.llm,
            verbose=self.verbose,
        )
        self.advisor = Agent(
            role="Expert Advisor",
            goal="Provide expert guidance and refinements",
            backstory=self.get_system_prompt(benchmark_type, "groupchat_advisor"),
            llm=self.llm,
            verbose=self.verbose,
        )
        self.moderator = Agent(
            role="Discussion Moderator",
            goal="Synthesize discussion into final response",
            backstory=self.get_system_prompt(benchmark_type, "groupchat_moderator"),
            llm=self.llm,
            verbose=self.verbose,
        )

    def respond_to_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> BenchmarkResponse:
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
        moderate = Task(
            description=f"Synthesize the discussion into a final response.\n\nORIGINAL TASK: {full_task}\n\nBased on all perspectives shared, generate your final response.\n\n{self.get_system_prompt(benchmark_type, 'groupchat_moderator')}",
            expected_output="Final consensus response as JSON",
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

        for task_obj in [propose, critique, advise, moderate]:
            task_output = getattr(task_obj, "output", None)
            output_str = str(task_output.raw) if task_output and hasattr(task_output, "raw") else str(task_output or "")
            TraceCapture.record(
                role=task_obj.agent.role if task_obj.agent else "unknown",
                input_prompt=task_obj.description[:1000],
                output_response=output_str,
            )

        if response.metadata is None:
            response.metadata = {}
        self.add_to_history(task=task, response=response.response, reasoning=response.reasoning, success=response.success)
        return response
