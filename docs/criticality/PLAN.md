# Criticality Benchmark v2: Logit-Based Argument Evaluation

## Motivation

The current criticality benchmark (`criticality_runner.py`) uses pairwise comparison -- the model picks A or B. This only tells us *which* argument the model thinks is stronger, not *how confident* it is or *why* at the probability level.

**v2 Goal**: Use logprobs to measure how strongly a model commits to its evaluation of argument quality. Instead of just checking if the model picks the right answer, we analyze the probability distribution over candidate responses to understand the model's internal certainty.

## What We're Measuring

1. **Calibration**: When the model says argument A is stronger, does the logprob distribution reflect appropriate confidence? (High confidence on easy pairs, lower on close calls)
2. **Discrimination**: Can the model differentiate between arguments of varying quality, reflected in the spread of logprobs across choices?
3. **Robustness**: Does the model's logprob-based ranking stay consistent across prompt variations, or is it easily swayed?

## Design

### Phase 1: MCQ Logprob Probing

Present the model with a topic and 4+ arguments of known quality, ask it to select the strongest.

```
Topic: "Should school uniforms be mandatory?"

Which of the following arguments is the strongest?

A) School uniforms reduce bullying by eliminating visible socioeconomic differences among students.
B) Uniforms are cheaper for parents.
C) Kids look nicer in uniforms.
D) Mandatory school uniforms have been shown in longitudinal studies to correlate with a 15% reduction in disciplinary incidents and improved academic focus, particularly in low-income districts.

Answer:
```

**What we extract:**
- Logprob of each choice token (A, B, C, D) at the answer position
- Convert to probabilities: `softmax([logprob_A, logprob_B, logprob_C, logprob_D])`
- Compare model's probability ranking to ground truth quality ranking

**Metrics:**
- `rank_correlation`: Spearman correlation between model's logprob ranking and ground truth quality ranking
- `top1_accuracy`: Does the highest-logprob choice match the best argument?
- `calibration_error`: Difference between model confidence and actual correctness rate
- `margin`: Logprob gap between the model's top choice and second choice (higher = more decisive)

### Phase 2: Freeform Refutation Probing

Give the model a claim and measure which human-written refutations it would most likely generate.

```
Claim: "Nuclear energy is too dangerous for widespread use."

What is the strongest counterargument to this claim?
```

**What we extract:**
- Generate the freeform response
- Then, present 4+ pre-written refutations and measure logprob of each as a continuation
- Compare: Does the model's freeform response align with the refutation it assigns highest probability to?

**Metrics:**
- `refutation_alignment`: Does the model's freeform response match the highest-logprob refutation?
- `refutation_quality_correlation`: Do higher-quality refutations get higher logprobs?

### Phase 3: Consistency Under Perturbation

Rerun Phase 1 with:
- Shuffled option order (A/B/C/D → C/A/D/B)
- Paraphrased arguments (same meaning, different wording)
- Added distractor arguments

**Metric:** `logprob_stability` -- how much does the probability distribution shift under perturbation?

## Data Source

Continue using **IBM Argument Quality Ranking 30k** (already integrated).

For each topic:
- Group arguments by quality score (WA or MACE-P)
- Bin into quality tiers: strong (>0.7), medium (0.4-0.7), weak (<0.4)
- Construct MCQ tasks by sampling one from each tier + adding a distractor

For refutations (Phase 2):
- Use argument pairs from opposite stances on the same topic
- One serves as claim, opposite-stance arguments serve as candidate refutations

## Infrastructure

### Model Support

| Source | Logprobs? | Models |
|--------|-----------|--------|
| Remote Ollama (10.1.10.87:11434) | Yes (top_logprobs up to 20) | DASD-4B, Falcon-H1-90M, Qwen3-0.6B |
| Azure Foundry | GPT-4o and Llama 3.3 only | gpt-4o, llama-3.3-70b |
| Azure Foundry | No logprobs | phi-4, mistral-*, ministral-* |

**Primary target**: Remote Ollama models (all support logprobs)
**Baseline comparison**: GPT-4o via Azure (logprobs supported)

### API Pattern

```python
# Ollama OpenAI-compatible endpoint
response = client.chat.completions.create(
    model="hf.co/mradermacher/DASD-4B-Thinking-GGUF:Q4_K_M",
    messages=[{"role": "user", "content": prompt}],
    logprobs=True,
    top_logprobs=10,
    max_tokens=1,        # Just need the choice token for MCQ
    temperature=0.0,      # Deterministic for logprob extraction
)

# Extract logprobs for choice tokens
choice_logprobs = {}
for token_info in response.choices[0].logprobs.content:
    for alt in token_info.top_logprobs:
        if alt.token.strip() in ["A", "B", "C", "D"]:
            choice_logprobs[alt.token.strip()] = alt.logprob
```

## Test Results (Feb 6, 2026)

Tested all 3 models on remote Ollama (10.1.10.87:11434) with MCQ argument quality prompts.

### DASD-4B-Thinking (Qwen3 family) -- USABLE

- **Behavior**: Thinks internally (hidden by Ollama), then outputs clean answer
- **With system constraint** ("You are a multiple choice answering machine. Output ONLY a single letter"):
  - Native API: thinking="The user asks... likely D..." → content="D" (correct)
  - OpenAI API: content="" but logprobs contain ~150 tokens (thinking + answer)
- **Logprob extraction**: The thinking tokens contain the decision process. At the "answer:" position (~pos 112), D gets 99.99% probability. The final answer token at pos ~147-149 also shows D at 99.93-100%.
- **Key finding**: Logprobs span ALL tokens (thinking + content). The choice probabilities at the "answer:" position in the thinking are the most useful -- they show the model's decision *before* it commits.
- **Gotcha**: The OpenAI-compatible API returns `content=""` (thinking is stripped) but logprobs include thinking tokens. Need to scan for the answer pattern in the logprob token sequence, not in content.

### Qwen3 0.6B -- NOT USABLE

- **Behavior**: Always thinks ("Okay, let's see...") regardless of instructions
- `/no_think` flag does NOT work -- model still thinks, content comes back empty
- Choice tokens (A/B/C/D) barely appear in top logprobs at any position
- **Verdict**: Too small to follow formatting instructions. Cannot extract meaningful logprobs.

### Falcon-H1-Tiny 90M -- NOT USABLE

- **Behavior**: Emits `<think>` tags directly into content (doesn't use Ollama's thinking separation)
- Echoes the prompt back in its "thinking"
- Choice tokens appear with negligible probability (0.0002-0.0033)
- **Verdict**: Too small, doesn't properly implement thinking/content separation.

### Prompt Strategy Results

| Strategy | DASD-4B | Qwen3 0.6B | Falcon-H1 90M |
|----------|---------|-------------|---------------|
| Bare MCQ | Starts reasoning, no direct answer | "Okay, let's see..." | `<think>` echoes prompt |
| Few-shot | Choice tokens at pos 16, very low prob | No choice tokens found | `<think>` immediately |
| System constrained | **"D" with 99.99% at answer position** | Still thinks, empty content | `<think>` echoes instructions |
| Completion style | Reasoning, choice tokens at pos 15 | Still thinks | `<think>` |

### Implications for Implementation

1. **DASD-4B is the only viable local model** for logprob-based criticality
2. Logprob extraction must scan the full token sequence (including thinking tokens) for the answer pattern
3. The "answer: X" pattern in thinking tokens gives the cleanest signal -- model probability at decision point
4. For Qwen3 0.6B and Falcon-H1 90M, fall back to standard accuracy-based evaluation (no logprobs)
5. GPT-4o and Llama-3.3-70B on Azure remain viable for logprob extraction (no thinking token issue)

## Implementation Steps

1. [x] **Test models** -- Run the 3 Ollama models through simple MCQ prompts, observe raw behavior and logprobs
2. [x] **Prompt engineering** -- Find prompt formats that reliably get choice tokens in top_logprobs for each model
3. [ ] **Build logprob extractor** -- Scan full token sequence (including thinking) for answer pattern, extract choice probabilities at decision point
4. [ ] **Build MCQ task generator** -- Pull from IBM dataset, construct 4-choice tasks with known quality ordering
5. [ ] **Build CriticalityV2Runner** -- New runner class that uses logprobs, computes calibration/discrimination metrics
6. [ ] **Run Phase 1** -- MCQ logprob probing on DASD-4B (local) + GPT-4o and Llama-3.3-70B (Azure)
7. [ ] **Analyze** -- Compare logprob-based metrics vs simple accuracy, see if logprobs reveal anything accuracy alone doesn't
8. [ ] **Phase 2 & 3** -- Freeform refutation and perturbation testing (stretch)

## File Structure

```
src/benchmarks/reasoning/
├── criticality_runner.py          # Existing v1 (pairwise, accuracy-only)
├── criticality_v2_runner.py       # New v2 (logprob-based)
├── criticality_logprob_utils.py   # Logprob extraction, calibration math
└── criticality_task_generator.py  # MCQ construction from IBM dataset
```
