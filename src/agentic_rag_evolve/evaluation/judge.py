"""Provider-neutral LLM judge for document question answering."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence


class JudgeModel(Protocol):
    model_name: str

    def complete(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class JudgeResult:
    status: str
    score: int | None
    reasoning: str
    prompt_type: str = "Generic_0-4"
    model_name: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    raise ValueError("judge response does not contain a JSON object")


def judge_answer(
    model: JudgeModel,
    *,
    task_id: str,
    question: str,
    gold_answers: Sequence[str],
    answer: str,
) -> JudgeResult:
    system = "You are an expert evaluator scoring how well an AI-generated answer matches a gold standard."
    prompt = f"""Score Generated Answer vs Gold Answer (0-4).
The Gold Answer is provided as a JSON array (e.g. ["ans1", "ans2"]). Treat the array as a whole — it represents the complete set of acceptable answers.
RULE:
1: Strictly penalize core factual errors. DO NOT penalize verbosity, expanded lists, or non-contradictory redundancy.
2: Refusal Check: If Gold contains facts but Gen says "Not mentioned", score 0. If both say "Not mentioned", score 4.

[Rubric]
4: Fully captures Gold Answer. No factual errors. Extra valid info allowed.
3: Accurate but incomplete. No core errors.
2: Misses core facts but relevant, OR minor secondary errors.
1: Core factual errors (e.g., wrong dates/methods).
0: Completely wrong or hallucinates conflicting info.

Question: {question}
Gold Answer: {json.dumps(list(gold_answers), ensure_ascii=False)}
Generated Answer: {answer}

First, write a 1-sentence reasoning. Then output the integer score.
Respond ONLY with a JSON object: {{"score": 0 to 4, "reasoning": "string"}}"""
    payload = {
        "model": model.model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "stream": False,
    }
    try:
        response = model.complete(payload)
        message = (response.get("choices") or [{}])[0].get("message") or {}
        content = str(message.get("content") or "")
        parsed = _parse_json_object(content)
        score = max(0, min(4, int(parsed["score"])))
        usage = response.get("usage") or {}
        return JudgeResult(
            status="ok",
            score=score,
            reasoning=str(parsed.get("reasoning") or "No reasoning provided."),
            model_name=model.model_name,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )
    except Exception as exc:
        return JudgeResult(
            status="error",
            score=None,
            reasoning="Judge invocation or response parsing failed.",
            model_name=model.model_name,
            error=f"{type(exc).__name__}: {exc}",
        )
