"""
Ollama-backed agentic architectures using direct API call chains.

Replaces the CrewAI-based implementations in ollama_agentic_agent.py with
explicit chained calls to the Ollama REST API. This eliminates the ReAct
format loop that causes 92-100% timeout rates on SLMs.

Each architecture reuses OllamaAgent._call_ollama() and _postprocess() for
HTTP/retry/thinking-model handling, and the same role prompts from
src/agents/prompts/bigbench/prompts.yaml.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

from .base_agent import BaseAgent, BenchmarkResponse
from .ollama_agent import OllamaAgent
from ..utils.trace import TraceCapture

OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
PER_CALL_TIMEOUT = 30  # seconds per individual Ollama call


def _build_ollama_backend(model: str, ollama_base_url: str, per_call_timeout: int) -> OllamaAgent:
    """Create an OllamaAgent instance to use as the LLM backend."""
    return OllamaAgent(
        model=model,
        ollama_base_url=ollama_base_url,
        max_tokens=512,
        temperature=0.7,
    )


def _call_and_postprocess(
    backend: OllamaAgent,
    system_prompt: str,
    user_message: str,
    timeout: int = PER_CALL_TIMEOUT,
) -> tuple[str, Dict[str, Any]]:
    """
    Make a single Ollama call and postprocess the result.

    Returns (cleaned_text, metadata_dict).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    start = time.time()
    try:
        response = backend._call_ollama(messages, max_tokens=512)
        latency = time.time() - start

        msg = response.get("message", {})
        thinking = msg.get("thinking", "") or ""
        raw_content = msg.get("content", "") or ""
        text = backend._postprocess(thinking, raw_content)

        meta = {
            "latency": latency,
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "thinking_length": len(thinking),
        }
        return text, meta

    except Exception as e:
        latency = time.time() - start
        return "", {"latency": latency, "error": str(e)[:200]}


def _build_context_str(context: Optional[Dict[str, Any]]) -> str:
    """Extract context fields into a string suffix for user messages."""
    if not context:
        return ""
    parts = []
    if "tools" in context:
        parts.append(f"\n\nAvailable tools: {context['tools']}")
    if "patient_data" in context:
        parts.append(f"\n\nPatient data: {context['patient_data']}")
    if "additional_info" in context:
        parts.append(f"\n\n{context['additional_info']}")
    return "".join(parts)


class OllamaSequentialAgent(BaseAgent):
    """
    Sequential pipeline (Analyzer -> Evaluator -> Responder) via direct Ollama calls.

    Three chained API calls, each feeding its output to the next stage.
    No CrewAI, no ReAct format requirements.
    """

    def __init__(
        self,
        model: str = "dasd-4b",
        verbose: bool = False,
        max_iterations: int = 1,
        ollama_base_url: str = OLLAMA_BASE_URL_DEFAULT,
        per_call_timeout: int = PER_CALL_TIMEOUT,
    ):
        super().__init__(model=model, verbose=verbose, max_iterations=max_iterations)
        self.ollama_base_url = ollama_base_url
        self.per_call_timeout = per_call_timeout
        self._backend = _build_ollama_backend(model, ollama_base_url, per_call_timeout)

    def respond_to_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> BenchmarkResponse:
        benchmark_type = (context or {}).get("benchmark_type", "general")
        full_task = f"{task}{_build_context_str(context)}"

        # Stage 1: Analyzer
        analyzer_prompt = self.get_system_prompt(benchmark_type, "sequential_analyzer")
        analysis, meta1 = _call_and_postprocess(
            self._backend, analyzer_prompt, f"TASK: {full_task}", self.per_call_timeout,
        )
        TraceCapture.record(
            role="Task Analyzer",
            input_prompt=f"TASK: {full_task}"[:1000],
            output_response=analysis[:1000],
            latency=meta1.get("latency", 0),
        )
        if self.verbose:
            print(f"  [Analyzer] {len(analysis)} chars, {meta1.get('latency', 0):.1f}s")

        # Stage 2: Evaluator
        evaluator_prompt = self.get_system_prompt(benchmark_type, "sequential_evaluator")
        strategy, meta2 = _call_and_postprocess(
            self._backend,
            evaluator_prompt,
            f"TASK: {full_task}\n\nANALYSIS:\n{analysis}",
            self.per_call_timeout,
        )
        TraceCapture.record(
            role="Approach Evaluator",
            input_prompt=f"TASK: {full_task}\n\nANALYSIS:\n{analysis}"[:1000],
            output_response=strategy[:1000],
            latency=meta2.get("latency", 0),
        )
        if self.verbose:
            print(f"  [Evaluator] {len(strategy)} chars, {meta2.get('latency', 0):.1f}s")

        # Stage 3: Responder
        responder_prompt = self.get_system_prompt(benchmark_type, "sequential_responder")
        final_text, meta3 = _call_and_postprocess(
            self._backend,
            responder_prompt,
            f"ORIGINAL TASK: {full_task}\n\nANALYSIS:\n{analysis}\n\nSTRATEGY:\n{strategy}",
            self.per_call_timeout,
        )
        TraceCapture.record(
            role="Response Generator",
            input_prompt=f"ORIGINAL TASK: {full_task}\n\nANALYSIS:\n{analysis}\n\nSTRATEGY:\n{strategy}"[:1000],
            output_response=final_text[:1000],
            latency=meta3.get("latency", 0),
        )
        if self.verbose:
            print(f"  [Responder] {len(final_text)} chars, {meta3.get('latency', 0):.1f}s")

        response = self.parse_json_response(final_text)
        if response.metadata is None:
            response.metadata = {}
        response.metadata["timed_out"] = False
        response.metadata["stage_meta"] = [meta1, meta2, meta3]
        total_lat = sum(m.get("latency", 0) for m in [meta1, meta2, meta3])
        response.metadata["total_latency"] = total_lat

        self.add_to_history(
            task=task, response=response.response,
            reasoning=response.reasoning, success=response.success,
        )
        return response


class OllamaConcurrentAgent(BaseAgent):
    """
    Concurrent agents (Analyst + Researcher + Critic -> Synthesizer) via direct Ollama calls.

    Three parallel API calls followed by one synthesis call.
    """

    def __init__(
        self,
        model: str = "dasd-4b",
        verbose: bool = False,
        max_iterations: int = 1,
        ollama_base_url: str = OLLAMA_BASE_URL_DEFAULT,
        per_call_timeout: int = PER_CALL_TIMEOUT,
    ):
        super().__init__(model=model, verbose=verbose, max_iterations=max_iterations)
        self.ollama_base_url = ollama_base_url
        self.per_call_timeout = per_call_timeout
        self._backend = _build_ollama_backend(model, ollama_base_url, per_call_timeout)

    def respond_to_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> BenchmarkResponse:
        benchmark_type = (context or {}).get("benchmark_type", "general")
        full_task = f"{task}{_build_context_str(context)}"

        # Parallel stage: analyst, researcher, critic
        roles = [
            ("Task Analyst", "concurrent_analyst"),
            ("Information Researcher", "concurrent_researcher"),
            ("Critical Reviewer", "concurrent_critic"),
        ]

        results: Dict[str, tuple[str, Dict]] = {}

        def _run_role(role_name: str, prompt_key: str) -> tuple[str, str, Dict]:
            prompt = self.get_system_prompt(benchmark_type, prompt_key)
            text, meta = _call_and_postprocess(
                self._backend, prompt, f"TASK: {full_task}", self.per_call_timeout,
            )
            return role_name, text, meta

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_run_role, rn, pk) for rn, pk in roles]
            for fut in as_completed(futures):
                try:
                    role_name, text, meta = fut.result(timeout=self.per_call_timeout + 5)
                except Exception as e:
                    role_name = "unknown"
                    text, meta = "", {"error": str(e)[:200]}
                results[role_name] = (text, meta)

        analyst_text = results.get("Task Analyst", ("", {}))[0]
        researcher_text = results.get("Information Researcher", ("", {}))[0]
        critic_text = results.get("Critical Reviewer", ("", {}))[0]

        for role_name in ["Task Analyst", "Information Researcher", "Critical Reviewer"]:
            text, meta = results.get(role_name, ("", {}))
            TraceCapture.record(
                role=role_name,
                input_prompt=f"TASK: {full_task}"[:1000],
                output_response=text[:1000],
                latency=meta.get("latency", 0) if isinstance(meta, dict) else 0,
            )
            if self.verbose:
                lat = meta.get("latency", 0) if isinstance(meta, dict) else 0
                print(f"  [{role_name}] {len(text)} chars, {lat:.1f}s")

        # Synthesis stage
        synthesizer_prompt = self.get_system_prompt(benchmark_type, "concurrent_synthesizer")
        synth_input = (
            f"TASK: {full_task}\n\n"
            f"ANALYSIS:\n{analyst_text}\n\n"
            f"RESEARCH:\n{researcher_text}\n\n"
            f"CRITIQUE:\n{critic_text}"
        )
        final_text, meta_synth = _call_and_postprocess(
            self._backend, synthesizer_prompt, synth_input, self.per_call_timeout,
        )
        TraceCapture.record(
            role="Response Synthesizer",
            input_prompt=synth_input[:1000],
            output_response=final_text[:1000],
            latency=meta_synth.get("latency", 0),
        )
        if self.verbose:
            print(f"  [Synthesizer] {len(final_text)} chars, {meta_synth.get('latency', 0):.1f}s")

        all_metas = [results.get(rn, ("", {}))[1] for rn, _ in roles] + [meta_synth]
        response = self.parse_json_response(final_text)
        if response.metadata is None:
            response.metadata = {}
        response.metadata["timed_out"] = False
        response.metadata["stage_meta"] = all_metas
        total_lat = sum(m.get("latency", 0) for m in all_metas if isinstance(m, dict))
        response.metadata["total_latency"] = total_lat

        self.add_to_history(
            task=task, response=response.response,
            reasoning=response.reasoning, success=response.success,
        )
        return response


class OllamaGroupChatAgent(BaseAgent):
    """
    Group chat (Proposer -> Critic -> Advisor -> Moderator) via direct Ollama calls.

    Four chained API calls simulating a discussion.
    """

    def __init__(
        self,
        model: str = "dasd-4b",
        verbose: bool = False,
        max_iterations: int = 1,
        ollama_base_url: str = OLLAMA_BASE_URL_DEFAULT,
        per_call_timeout: int = PER_CALL_TIMEOUT,
    ):
        super().__init__(model=model, verbose=verbose, max_iterations=max_iterations)
        self.ollama_base_url = ollama_base_url
        self.per_call_timeout = per_call_timeout
        self._backend = _build_ollama_backend(model, ollama_base_url, per_call_timeout)

    def respond_to_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> BenchmarkResponse:
        benchmark_type = (context or {}).get("benchmark_type", "general")
        full_task = f"{task}{_build_context_str(context)}"

        # Stage 1: Proposer
        proposer_prompt = self.get_system_prompt(benchmark_type, "groupchat_proposer")
        proposal, meta1 = _call_and_postprocess(
            self._backend, proposer_prompt, f"TASK: {full_task}", self.per_call_timeout,
        )
        TraceCapture.record(
            role="Solution Proposer",
            input_prompt=f"TASK: {full_task}"[:1000],
            output_response=proposal[:1000],
            latency=meta1.get("latency", 0),
        )
        if self.verbose:
            print(f"  [Proposer] {len(proposal)} chars, {meta1.get('latency', 0):.1f}s")

        # Stage 2: Critic
        critic_prompt = self.get_system_prompt(benchmark_type, "groupchat_critic")
        critique, meta2 = _call_and_postprocess(
            self._backend,
            critic_prompt,
            f"ORIGINAL TASK: {full_task}\n\nPROPOSAL:\n{proposal}",
            self.per_call_timeout,
        )
        TraceCapture.record(
            role="Critical Analyst",
            input_prompt=f"ORIGINAL TASK: {full_task}\n\nPROPOSAL:\n{proposal}"[:1000],
            output_response=critique[:1000],
            latency=meta2.get("latency", 0),
        )
        if self.verbose:
            print(f"  [Critic] {len(critique)} chars, {meta2.get('latency', 0):.1f}s")

        # Stage 3: Advisor
        advisor_prompt = self.get_system_prompt(benchmark_type, "groupchat_advisor")
        advice, meta3 = _call_and_postprocess(
            self._backend,
            advisor_prompt,
            f"ORIGINAL TASK: {full_task}\n\nPROPOSAL:\n{proposal}\n\nCRITIQUE:\n{critique}",
            self.per_call_timeout,
        )
        TraceCapture.record(
            role="Expert Advisor",
            input_prompt=f"ORIGINAL TASK: {full_task}\n\nPROPOSAL:\n{proposal}\n\nCRITIQUE:\n{critique}"[:1000],
            output_response=advice[:1000],
            latency=meta3.get("latency", 0),
        )
        if self.verbose:
            print(f"  [Advisor] {len(advice)} chars, {meta3.get('latency', 0):.1f}s")

        # Stage 4: Moderator
        moderator_prompt = self.get_system_prompt(benchmark_type, "groupchat_moderator")
        mod_input = (
            f"ORIGINAL TASK: {full_task}\n\n"
            f"PROPOSAL:\n{proposal}\n\n"
            f"CRITIQUE:\n{critique}\n\n"
            f"ADVICE:\n{advice}"
        )
        final_text, meta4 = _call_and_postprocess(
            self._backend, moderator_prompt, mod_input, self.per_call_timeout,
        )
        TraceCapture.record(
            role="Discussion Moderator",
            input_prompt=mod_input[:1000],
            output_response=final_text[:1000],
            latency=meta4.get("latency", 0),
        )
        if self.verbose:
            print(f"  [Moderator] {len(final_text)} chars, {meta4.get('latency', 0):.1f}s")

        all_metas = [meta1, meta2, meta3, meta4]
        response = self.parse_json_response(final_text)
        if response.metadata is None:
            response.metadata = {}
        response.metadata["timed_out"] = False
        response.metadata["stage_meta"] = all_metas
        total_lat = sum(m.get("latency", 0) for m in all_metas)
        response.metadata["total_latency"] = total_lat

        self.add_to_history(
            task=task, response=response.response,
            reasoning=response.reasoning, success=response.success,
        )
        return response
