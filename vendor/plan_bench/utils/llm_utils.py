from transformers import StoppingCriteriaList, StoppingCriteria
import openai
import os

if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.environ["OPENAI_API_KEY"]


def _strip_thinking_tags(text):
    """Strip <think>...</think> blocks that some models leak into content."""
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _task_prefill(query):
    """Return the assistant prefill string appropriate for this PlanBench query.

    Queries end with a task-specific marker that signals what the model should output:
      [PLAN]             → plan generation (t1,t2,t4,t5,t6,t8_x) — prefill [PLAN]
      [VERIFICATION]     → plan verification (t3)               — no prefill
      [RESULTING STATE]  → state prediction (t7)                — no prefill
    """
    stripped = query.rstrip()
    if stripped.endswith("[PLAN]"):
        return "[PLAN]\n"
    return ""


def _task_system_message(query):
    """Return a concise system message for non-Ollama (Azure) models based on task type."""
    stripped = query.rstrip()
    if stripped.endswith("[PLAN]"):
        return (
            "You are a planning assistant. After [PLAN], output only the action steps "
            "one per line, exactly as shown in the examples. Do not explain."
        )
    if stripped.endswith("[VERIFICATION]"):
        return (
            "You are a planning assistant. After [VERIFICATION], state whether the plan "
            "is valid or invalid and explain any unmet preconditions, exactly as shown in the examples."
        )
    if stripped.endswith("[RESULTING STATE]"):
        return (
            "You are a planning assistant. After [RESULTING STATE], describe the resulting "
            "state as a comma-separated list of facts, exactly as shown in the examples."
        )
    return "You are a planning assistant. Follow the format shown in the examples exactly."


def _ollama_chat_direct(model_name, query, max_tokens, stop_list):
    """Call the Ollama /api/chat endpoint directly.

    For plan-generation tasks the assistant turn is prefilled with '[PLAN]' so
    the model continues with plan actions instead of generating a verbose essay.
    For verification (t3) and state-prediction (t7) tasks no prefill is used.
    litellm silently drops assistant-role messages, hence the direct call.
    """
    import urllib.request, json

    model_tag = model_name[len("ollama/"):]
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    prefill = _task_prefill(query)
    messages = [{"role": "user", "content": query}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    extended_stop = list(stop_list)
    if "[PLAN END]" not in extended_stop:
        extended_stop.append("[PLAN END]")

    payload = {
        "model": model_tag,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "stop": extended_stop,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        result = json.loads(r.read())

    return (result.get("message") or {}).get("content") or ""


def send_query_litellm(query, engine, max_tokens, model=None, stop="[STATEMENT]"):
    """Call LiteLLM (Azure/Ollama/etc.) instead of OpenAI. Used when USE_LITELLM=1.

    Ollama: bypasses litellm to use /api/chat directly with assistant prefill.
    Azure/other: uses litellm with a task-appropriate system message so that
    instruction-tuned models follow the PlanBench output format.
    """
    import litellm
    litellm_model = os.environ.get("LITELLM_MODEL")
    if not litellm_model:
        print("[-]: USE_LITELLM=1 but LITELLM_MODEL not set")
        return ""
    stop_list = [stop] if isinstance(stop, str) else list(stop or [])

    is_ollama = litellm_model.startswith("ollama/")
    if is_ollama:
        try:
            return _ollama_chat_direct(litellm_model, query, max_tokens, stop_list)
        except Exception as e:
            print("[-]: Ollama direct call failed: {}".format(e))
            return ""

    # Azure / other providers — use litellm with a system message for format compliance
    system_msg = _task_system_message(query)
    extended_stop = list(stop_list)
    if "[PLAN END]" not in extended_stop:
        extended_stop.append("[PLAN END]")

    kwargs = {
        "model": litellm_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stop": extended_stop,
    }
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("AZURE_API_KEY")
    api_base = os.environ.get("LITELLM_API_BASE") or os.environ.get("AZURE_AI_ENDPOINT")
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    try:
        response = litellm.completion(**kwargs)
        text = _strip_thinking_tags(response.choices[0].message.content or "")
        # Some reasoning models (DeepSeek-R1, o1) surface reasoning in a separate field
        if not text:
            reasoning = getattr(response.choices[0].message, "reasoning_content", None) or ""
            text = _strip_thinking_tags(reasoning)
        return text
    except Exception as e:
        print("[-]: LiteLLM query failed: {}".format(e))
        return ""


def generate_from_bloom(model, tokenizer, query, max_tokens):
    encoded_input = tokenizer(query, return_tensors='pt')
    stop = tokenizer("[PLAN END]", return_tensors='pt')
    stoplist = StoppingCriteriaList([stop])
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), max_new_tokens=max_tokens,
                                      temperature=0, top_p=1)
    return tokenizer.decode(output_sequences[0], skip_special_tokes=True)


def send_query(query, engine, max_tokens, model=None, stop="[STATEMENT]"):
    if os.environ.get("USE_LITELLM") == "1":
        return send_query_litellm(query, engine, max_tokens, model=model, stop=stop)
    max_token_err_flag = False
    if engine == 'bloom':

        if model:
            response = generate_from_bloom(model['model'], model['tokenizer'], query, max_tokens)
            response = response.replace(query, '')
            resp_string = ""
            for line in response.split('\n'):
                if '[PLAN END]' in line:
                    break
                else:
                    resp_string += f'{line}\n'
            return resp_string
        else:
            assert model is not None
    elif engine == 'finetuned':
        if model:
            try:
                response = openai.Completion.create(
                    model=model['model'],
                    prompt=query,
                    temperature=0,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["[PLAN END]"])
            except Exception as e:
                max_token_err_flag = True
                print("[-]: Failed GPT3 query execution: {}".format(e))
            text_response = response["choices"][0]["text"] if not max_token_err_flag else ""
            return text_response.strip()
        else:
            assert model is not None
    elif '_chat' in engine:
        
        eng = engine.split('_')[0]
        messages=[
        {"role": "system", "content": "You are the planner assistant who comes up with correct plans."},
        {"role": "user", "content": query}
        ]
        try:
            response = openai.ChatCompletion.create(model=eng, messages=messages, temperature=0)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT3 query execution: {}".format(e))
        text_response = response['choices'][0]['message']['content'] if not max_token_err_flag else ""
        return text_response.strip()        
    else:
        try:
            response = openai.Completion.create(
                model=engine,
                prompt=query,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT3 query execution: {}".format(e))

        text_response = response["choices"][0]["text"] if not max_token_err_flag else ""
        return text_response.strip()
