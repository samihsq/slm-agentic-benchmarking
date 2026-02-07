"""
Sequential Agent for Benchmarking.

Implements a linear pipeline pattern where specialized agents process
the task through multiple stages: Analyze → Evaluate → Respond

Based on Microsoft's Sequential Orchestration pattern:
https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns
"""

from typing import Optional, Dict, Any, List

from crewai import Agent, Task, Crew, Process
from .base_agent import BaseAgent, BenchmarkResponse
from ..config import get_llm
from ..utils.trace import TraceCapture


class SequentialAgent(BaseAgent):
    """
    Sequential orchestration agent using a linear task processing pipeline.

    Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                   Sequential Pipeline                     │
    │                                                          │
    │  Task → [Analyzer] → [Evaluator] → [Responder] → Response │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    The agent processes each task through three stages:
    1. Analyzer: Identifies the nature of the task and requirements
    2. Evaluator: Assesses the best approach to complete the task
    3. Responder: Formulates the final response
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
        """Initialize the sequential pipeline agents with benchmark-specific prompts."""
        
        # Get prompts for this benchmark type
        analyzer_prompt = self.get_system_prompt(benchmark_type, "sequential_analyzer")
        evaluator_prompt = self.get_system_prompt(benchmark_type, "sequential_evaluator")
        responder_prompt = self.get_system_prompt(benchmark_type, "sequential_responder")

        # Stage 1: Analyzer - Identifies task type and requirements
        self.analyzer = Agent(
            role="Task Analyzer",
            goal="Analyze the nature and requirements of tasks",
            backstory=analyzer_prompt,
            llm=self.llm,
            verbose=self.verbose,
        )

        # Stage 2: Evaluator - Assesses appropriate approach
        self.evaluator = Agent(
            role="Approach Evaluator",
            goal="Determine the best approach to complete the task",
            backstory=evaluator_prompt,
            llm=self.llm,
            verbose=self.verbose,
        )

        # Stage 3: Responder - Executes and provides final response
        self.responder = Agent(
            role="Response Generator",
            goal="Execute the approach and provide the final response",
            backstory=responder_prompt,
            llm=self.llm,
            verbose=self.verbose,
        )

    def respond_to_task(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResponse:
        """
        Process the task through the sequential pipeline.

        Args:
            task: The benchmark task or question
            context: Additional context (tools, data, etc.)

        Returns:
            BenchmarkResponse with reasoning and the response
        """
        
        # Determine benchmark type and setup agents with appropriate prompts
        benchmark_type = (context or {}).get("benchmark_type", "general")
        self._setup_agents(benchmark_type)

        # Add context information to task if provided
        context_str = ""
        if context:
            if "tools" in context:
                context_str += f"\n\nAvailable tools: {context['tools']}"
            if "patient_data" in context:
                context_str += f"\n\nPatient data: {context['patient_data']}"
            if "additional_info" in context:
                context_str += f"\n\n{context['additional_info']}"

        full_task = f"{task}{context_str}"

        # Stage 1: Analyze
        analyze_task = Task(
            description=f"""Analyze this task:

TASK: {full_task}

Provide your analysis following your role guidelines.""",
            expected_output="Analysis of the task's nature and requirements",
            agent=self.analyzer,
        )

        # Stage 2: Evaluate
        evaluate_task = Task(
            description=f"""Evaluate how to approach this task:

TASK: {full_task}

Based on the analysis, provide your evaluation following your role guidelines.""",
            expected_output="Evaluation of the best approach",
            agent=self.evaluator,
            context=[analyze_task],
        )

        # Stage 3: Respond - use benchmark-specific output format
        responder_prompt = self.get_system_prompt(benchmark_type, "sequential_responder")
        respond_task = Task(
            description=f"""Complete the task and provide your final answer.

ORIGINAL TASK: {full_task}

Based on the analysis and evaluation, generate your final response.

{responder_prompt}""",
            expected_output="JSON with reasoning and the final answer",
            agent=self.responder,
            context=[analyze_task, evaluate_task],
        )

        # Create and run the sequential crew
        crew = Crew(
            agents=[self.analyzer, self.evaluator, self.responder],
            tasks=[analyze_task, evaluate_task, respond_task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        result = crew.kickoff()

        # Parse the response using robust parser from base class
        response = self.parse_json_response(str(result))
        
        # Capture the pipeline steps for debugging and tracing
        pipeline_steps = []
        task_descriptions = [
            analyze_task.description,
            evaluate_task.description, 
            respond_task.description,
        ]
        for i, task_obj in enumerate([analyze_task, evaluate_task, respond_task]):
            task_output = getattr(task_obj, 'output', None)
            output_str = ""
            if task_output:
                output_str = str(task_output.raw) if hasattr(task_output, 'raw') else str(task_output)
            
            step = {
                "stage": task_obj.agent.role if task_obj.agent else "unknown",
                "input": task_descriptions[i][:1000],
                "output": output_str,
            }
            pipeline_steps.append(step)
            
            # Record to trace if capture is active
            TraceCapture.record(
                role=step["stage"],
                input_prompt=step["input"],
                output_response=output_str,
            )
        
        # Add pipeline steps to metadata
        if response.metadata is None:
            response.metadata = {}
        response.metadata["pipeline_steps"] = pipeline_steps

        # Add to history
        self.add_to_history(
            task=task,
            response=response.response,
            reasoning=response.reasoning,
            success=response.success,
        )

        return response
