"""
Ollama Agent for Benchmarking.

Calls a remote (or local) Ollama instance via its REST API.
Handles thinking-model quirks: combines thinking + content, strips <think> tags,
and post-processes output to extract structured answers.

Supports all benchmark types via the standard BaseAgent interface.
"""

import json
import re
import time
import random
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

from .base_agent import BaseAgent, BenchmarkResponse

# Retry configuration
MAX_RETRIES = 2
BASE_DELAY = 1.0
MAX_DELAY = 30.0


class OllamaAgent(BaseAgent):
    """
    Benchmark agent using a remote Ollama instance.

    Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                  Ollama One-Shot Agent                    │
    │                                                          │
    │  Task → [Ollama REST API] → Post-process → Response      │
    │                                                          │
    │  Handles:                                                │
    │    - Thinking models (<think> tokens / thinking field)   │
    │    - Empty content (extracts answer from thinking)       │
    │    - <think> tags leaked into content                    │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        model: str = "qwen3:0.6b",
        verbose: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: Optional[int] = 512,
        think: bool = True,
    ):
        # Resolve config key (e.g. "falcon-h1-90m") to the actual Ollama model
        # name (e.g. "hf.co/tiiuae/Falcon-H1-Tiny-R-90M-GGUF:Q4_K_M").
        try:
            from ..config.azure_llm_config import OLLAMA_MODELS
            if model in OLLAMA_MODELS:
                model_name = OLLAMA_MODELS[model]["model"]
            else:
                model_name = model
        except ImportError:
            model_name = model

        super().__init__(
            model=model,           # keep config key as the identity/display name
            verbose=verbose,
            max_iterations=1,
        )

        self._ollama_model_name = model_name  # actual name sent to Ollama API
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.temperature = temperature
        # None = no limit (Ollama uses num_predict: -1 for unlimited)
        self.max_tokens = max_tokens
        self.think = think

        # Model display name for results (strip HF prefix)
        self._display_name = model
        if "/" in model_name:
            parts = model_name.split("/")
            self._display_name = parts[-1].replace("-GGUF", "").replace(":Q4_K_M", "")

    def _call_ollama(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call the Ollama native API (not OpenAI-compatible) to get
        separated thinking and content fields.

        If the model doesn't support the `think` parameter (returns 400/404),
        automatically retries without it so GGUF and non-thinking models work.

        Returns:
            Dict with 'thinking', 'content', 'total_duration', 'eval_count', etc.
        """
        # num_predict: -1 = unlimited in Ollama; otherwise use requested or instance limit
        num_predict = max_tokens if max_tokens is not None else self.max_tokens
        if num_predict is None:
            num_predict = -1

        url = f"{self.ollama_base_url}/api/chat"

        def _make_request(include_think: bool) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "model": self._ollama_model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": num_predict,
                },
            }
            if include_think:
                payload["think"] = True
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        if self.think:
            try:
                return _make_request(include_think=True)
            except urllib.error.HTTPError as e:
                if e.code in (400, 404):
                    # Model doesn't support think parameter; disable and retry
                    self.think = False
                    return _make_request(include_think=False)
                raise

        data = json.dumps({
            "model": self._ollama_model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": num_predict,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _postprocess(self, thinking: str, content: str) -> str:
        """
        Combine thinking + content into a usable response string.

        Strategy:
        1. If content is non-empty and clean, use it directly.
        2. If content has <think> tags, strip them and extract the answer.
        3. If content is empty, extract the answer from thinking tokens.
        """
        # Clean content: strip <think>...</think> blocks
        if content:
            cleaned = re.sub(
                r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE
            ).strip()
            if cleaned:
                return cleaned
            # Content was all <think> tags -- fall through to thinking extraction

        # Content is empty or all thinking tags -- extract from thinking
        if thinking:
            return self._extract_answer_from_thinking(thinking)

        return content or ""

    def _extract_answer_from_thinking(self, thinking: str) -> str:
        """
        Extract the final answer from thinking/reasoning tokens.

        Looks for common patterns where the model states its answer:
        - "The answer is X"
        - "Answer: X"
        - "So the answer is X"
        - "I'll go with X"
        - "Therefore, X"
        - The last sentence that looks like a conclusion
        """
        # Pattern 1: Explicit "answer is/:" patterns (take the last one)
        answer_patterns = [
            r"(?:the\s+)?answer\s*(?:is|:)\s*[\"']?(.+?)(?:[\"']?\s*$|[\"']?\s*\.)",
            r"(?:so|therefore|thus)[,:]?\s+(?:the\s+)?answer\s*(?:is|:)\s*(.+?)(?:\s*$|\s*\.)",
            r"I(?:'ll| will)\s+(?:go with|choose|pick|select)\s+(.+?)(?:\s*$|\s*\.)",
            r"(?:final\s+)?(?:answer|response|output)\s*:\s*(.+?)(?:\s*$|\s*\.)",
        ]

        last_answer = None
        for pattern in answer_patterns:
            for match in re.finditer(pattern, thinking, re.IGNORECASE | re.MULTILINE):
                last_answer = match.group(1).strip()

        if last_answer:
            return last_answer

        # Pattern 2: JSON in thinking
        json_match = re.search(r"\{[^}]*\"(?:answer|response)\"[^}]*\}", thinking)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("answer", data.get("response", ""))
            except json.JSONDecodeError:
                pass

        # Pattern 3: Last meaningful sentence (often the conclusion)
        sentences = re.split(r"(?<=[.!?])\s+", thinking.strip())
        if sentences:
            # Return the last non-trivial sentence
            for sent in reversed(sentences):
                sent = sent.strip()
                if len(sent) > 5 and not sent.lower().startswith(("let me", "okay", "hmm", "wait")):
                    return sent

        # Fallback: return last 200 chars of thinking
        return thinking[-200:].strip()

    def respond_to_task(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResponse:
        """
        Generate a response via Ollama with post-processing.
        """
        benchmark_type = (context or {}).get("benchmark_type", "general")
        system_prompt = self.get_system_prompt(benchmark_type)

        # Add context
        context_str = ""
        if context:
            if "tools" in context:
                context_str += f"\n\nAvailable tools: {context['tools']}"
            if "patient_data" in context:
                context_str += f"\n\nPatient data: {context['patient_data']}"
            if "additional_info" in context:
                context_str += f"\n\n{context['additional_info']}"

        user_message = f"TASK: {task}{context_str}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        result_text = None
        thinking_text = ""
        last_error = None
        eval_count = 0
        prompt_eval_count = 0
        total_duration = 0

        for attempt in range(MAX_RETRIES):
            try:
                response = self._call_ollama(messages)

                msg = response.get("message", {})
                thinking_text = msg.get("thinking", "") or ""
                raw_content = msg.get("content", "") or ""

                # Post-process: combine thinking + content
                result_text = self._postprocess(thinking_text, raw_content)

                # Token counts from Ollama
                eval_count = response.get("eval_count", 0)
                prompt_eval_count = response.get("prompt_eval_count", 0)
                total_duration = response.get("total_duration", 0)

                break

            except urllib.error.HTTPError as e:
                last_error = e
                body = e.read().decode("utf-8", errors="replace")[:200]
                if self.verbose:
                    print(f"  HTTP {e.code}: {body}")
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                    time.sleep(delay)

            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"  Ollama error: {e}")
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                    time.sleep(delay)

        if result_text is None:
            result_text = f'{{"reasoning": "Error after {MAX_RETRIES} retries: {str(last_error)[:150]}", "confidence": 0.0, "response": "Error"}}'

        # Parse response using base class parser
        parsed = self.parse_json_response(result_text)

        # If parsing fell back but we have a clean non-JSON answer, use it directly
        if parsed.metadata and parsed.metadata.get("parse_fallback"):
            # For short answers (MCQ, numbers, etc.), the postprocessed text IS the answer
            if len(result_text) < 500:
                parsed.response = result_text

        # Add metadata
        parsed.metadata = parsed.metadata or {}
        parsed.metadata.update({
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count,
            "total_duration_ns": total_duration,
            "thinking_length": len(thinking_text),
            "raw_content_length": len(raw_content) if "raw_content" in dir() else 0,
            "ollama_model": self.model,
        })

        # Add to history
        self.add_to_history(
            task=task,
            response=parsed.response,
            reasoning=parsed.reasoning or thinking_text[:500],
            success=parsed.success,
        )

        return parsed
