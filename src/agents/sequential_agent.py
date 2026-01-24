"""
Sequential Agent for Benchmarking.

Implements a linear pipeline pattern where specialized agents process
the task through multiple stages: Analyze → Evaluate → Respond

Based on Microsoft's Sequential Orchestration pattern:
https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns
"""

from typing import Optional, Dict, Any

from crewai import Agent, Task, Crew, Process
from .base_agent import BaseAgent, BenchmarkResponse
from ..config import get_llm


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
        self._setup_agents()

    def _setup_agents(self):
        """Initialize the sequential pipeline agents."""

        # Stage 1: Analyzer - Identifies task type and requirements
        self.analyzer = Agent(
            role="Task Analyzer",
            goal="Analyze the nature and requirements of tasks",
            backstory="""You are an expert at understanding task requirements.
            
            Your job is to identify:
            - What is the core objective of the task
            - What domain or topic does this relate to
            - What information or tools are needed
            - What would constitute a successful response""",
            llm=self.llm,
            verbose=self.verbose,
        )

        # Stage 2: Evaluator - Assesses appropriate approach
        self.evaluator = Agent(
            role="Approach Evaluator",
            goal="Determine the best approach to complete the task",
            backstory="""You evaluate how to best approach tasks.
            
            Consider:
            - What strategy should be used?
            - Are there multiple steps required?
            - What information would be most relevant?
            - What potential issues might arise?""",
            llm=self.llm,
            verbose=self.verbose,
        )

        # Stage 3: Responder - Executes and provides final response
        self.responder = Agent(
            role="Response Generator",
            goal="Execute the approach and provide the final response",
            backstory="""You synthesize the analysis and execute the task.
            
            Guidelines:
            - Follow the recommended approach
            - Provide complete and accurate responses
            - Use available tools or data as needed
            - Output valid JSON with your response""",
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

Identify:
1. What is the core objective?
2. What domain/topic does this relate to?
3. What information or tools are needed?
4. What would constitute a successful response?

Provide your analysis.""",
            expected_output="Analysis of the task's nature and requirements",
            agent=self.analyzer,
        )

        # Stage 2: Evaluate
        evaluate_task = Task(
            description=f"""Evaluate how to approach this task:

TASK: {full_task}

Based on the analysis, determine:
1. What strategy should be used?
2. Are there multiple steps required?
3. What would be the most effective approach?
4. What potential challenges might arise?

Provide your evaluation.""",
            expected_output="Evaluation of the best approach",
            agent=self.evaluator,
            context=[analyze_task],
        )

        # Stage 3: Respond
        respond_task = Task(
            description="""Execute the task and provide the final response as JSON.

Based on the analysis and evaluation, complete the task:
{{"reasoning": "<summary of your approach>", "confidence": <0.0-1.0>, "response": "<your complete response>"}}

ONLY output the JSON object, nothing else.""",
            expected_output="JSON with reasoning, confidence, and response",
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

        # Add to history
        self.add_to_history(
            task=task,
            response=response.response,
            reasoning=response.reasoning,
            success=response.success,
        )

        return response
