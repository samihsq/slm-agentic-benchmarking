"""
Trace utilities for capturing LLM call inputs/outputs.

Used to record full traces of multi-agent interactions for debugging and analysis.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class LLMCall:
    """A single LLM call with input and output."""
    role: str  # e.g., "Analyzer", "Critic", "Proposer"
    input_prompt: str
    output_response: str
    timestamp: float = field(default_factory=time.time)
    tokens_in: int = 0
    tokens_out: int = 0
    latency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class QuestionTrace:
    """Full trace for a single question/task."""
    task_id: str
    agent_type: str
    model: str
    input_question: str
    calls: List[LLMCall] = field(default_factory=list)
    final_output: str = ""
    predicted: str = ""
    correct: str = ""
    match: bool = False
    confidence: float = 0.0
    total_latency: float = 0.0
    total_cost: float = 0.0
    reasoning: str = ""
    
    def add_call(self, call: LLMCall):
        """Add an LLM call to the trace."""
        self.calls.append(call)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "model": self.model,
            "input_question": self.input_question,
            "calls": [c.to_dict() for c in self.calls],
            "final_output": self.final_output,
            "predicted": self.predicted,
            "correct": self.correct,
            "match": self.match,
            "confidence": self.confidence,
            "total_latency": self.total_latency,
            "total_cost": self.total_cost,
            "reasoning": self.reasoning,
        }
    
    def save(self, output_dir: Path):
        """Save trace to a folder."""
        question_dir = output_dir / self.task_id
        question_dir.mkdir(parents=True, exist_ok=True)
        
        trace_file = question_dir / "trace.json"
        with open(trace_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return trace_file


class TraceCapture:
    """Context manager for capturing LLM calls during agent execution."""
    
    _current: Optional['TraceCapture'] = None
    
    def __init__(self, task_id: str, agent_type: str, model: str, input_question: str):
        self.trace = QuestionTrace(
            task_id=task_id,
            agent_type=agent_type,
            model=model,
            input_question=input_question,
        )
    
    def __enter__(self):
        TraceCapture._current = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        TraceCapture._current = None
        return False
    
    def record_call(self, role: str, input_prompt: str, output_response: str, 
                    tokens_in: int = 0, tokens_out: int = 0, latency: float = 0.0):
        """Record an LLM call."""
        call = LLMCall(
            role=role,
            input_prompt=input_prompt,
            output_response=output_response,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency=latency,
        )
        self.trace.add_call(call)
    
    @classmethod
    def get_current(cls) -> Optional['TraceCapture']:
        """Get the current trace capture context."""
        return cls._current
    
    @classmethod
    def record(cls, role: str, input_prompt: str, output_response: str, **kwargs):
        """Record a call to the current trace if one exists."""
        current = cls.get_current()
        if current:
            current.record_call(role, input_prompt, output_response, **kwargs)
