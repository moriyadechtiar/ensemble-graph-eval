"""LLM judge: asks an OpenAI model to list the issues in a predicted graph. Needs the optional ``openai`` package."""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMJudgeConfig:
    model: str = os.getenv("OPENAI_LLM_JUDGE_MODEL", "gpt-5")
    timeout: float = float(os.getenv("OPENAI_LLM_TIMEOUT", "60"))
    max_retries: int = int(os.getenv("OPENAI_LLM_MAX_RETRIES", "1"))
    insist_json: bool = True


class LLMJudge:
    """Scores a predicted graph against a reference graph and the clause text.

    Returns ``score`` in [0, 1] (1 is perfectly correct), the number of issues,
    a list of issues, and a short explanation. Reads OPENAI_API_KEY.
    """

    def __init__(self, cfg: Optional[LLMJudgeConfig] = None, client=None):
        self.name = "openai_issues"
        self.cfg = cfg or LLMJudgeConfig()
        if client is None:
            from openai import OpenAI
            client = OpenAI(timeout=self.cfg.timeout, max_retries=self.cfg.max_retries)
        self._client = client

    def _build_prompt(self, pred: dict, gold: dict, clause_text: str, original_prompt: str) -> str:
        return (
            "You are a meticulous evaluator for legal graph extraction.\n"
            "Given a PRED graph (model prediction) and a GOLD graph (ground truth), "
            "identify concrete issues in PRED compared to GOLD, based only on the provided clause text and graphs.\n\n"
            "Return a strict JSON object with fields:\n"
            "  - score: number in [0,1] where 1=perfectly correct, 0=completely wrong\n"
            "  - issues_found: integer count of distinct issues\n"
            "  - issue_list: array of short issue strings (e.g., 'missing node TERM: Force Majeure', 'wrong edge type BETWEEN A->B')\n"
            "  - explanation: brief rationale (1-3 sentences)\n\n"
            f"CLAUSE_TEXT:\n{clause_text}\n\n"
            f"ORIGINAL_PROMPT:\n{original_prompt}\n\n"
            f"PRED_GRAPH(JSON):\n{json.dumps(pred, ensure_ascii=False)}\n\n"
            f"GOLD_GRAPH(JSON):\n{json.dumps(gold, ensure_ascii=False)}\n\n"
            "Respond with ONLY the JSON object. No prose."
        )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            score = data.get("score")
            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    score = None
            if score is not None:
                score = max(0.0, min(1.0, float(score)))
            issues_found = data.get("issues_found")
            if isinstance(issues_found, str):
                try:
                    issues_found = int(issues_found)
                except ValueError:
                    issues_found = None
            issue_list = data.get("issue_list") or []
            if not isinstance(issue_list, list):
                issue_list = [str(issue_list)]
            return {
                "score": score,
                "issues_found": issues_found if isinstance(issues_found, int) else (len(issue_list) if issue_list else None),
                "issue_list": [str(x) for x in issue_list][:20],
                "explanation": str(data.get("explanation") or "")[:1000],
            }
        except Exception as e:
            return {"score": None, "issues_found": None, "issue_list": [], "explanation": f"JSON parse error: {e}"}

    def score(self, pred: dict, gold: dict, clause_text: str = "", original_prompt: str = "") -> Dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for legal graph extraction."},
                    {"role": "user", "content": self._build_prompt(pred, gold, clause_text, original_prompt)},
                ],
                response_format={"type": "json_object"} if self.cfg.insist_json else None,
            )
            return self._parse_json(response.choices[0].message.content or "{}")
        except Exception as e:
            return {"score": None, "issues_found": None, "issue_list": [], "explanation": f"Error: {e}"}
