# src/evaluation/generation_metrics.py
"""
Core Generation Metrics for RAG Evaluation

    Core Generation Metrics
     ─────────────────────────────────────────────────────────────
     Faithfulness, Answer Relevance,Completeness, Abstention Quality

     LLM-as-Judge Implementation
     ─────────────────────────────────────────────────────────────
     - Judge model > generation model
     - Chain-of-thought reasoning before scoring
     - 4-6 few-shot calibration examples
     - Structured JSON output (score + reasoning)
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("uvicorn.error")


# LLM-AS-JUDGE PROMPTS  (chain-of-thought + few-shot)

# ── Judge prompt template (used for all generation metrics) ──────
_JUDGE_SYSTEM = """\
You are an expert RAG evaluation judge. Your task is to score an AI-generated answer
on a specific criterion. Follow these rules:
1. Read the criterion definition carefully.
2. Reason step-by-step (chain-of-thought) before assigning a score.
3. Assign a score between 0.0 and 1.0.
4. Return ONLY valid JSON — no markdown, no extra text.
"""

_JUDGE_TEMPLATE = """\
## Criterion: {criterion_name}
{criterion_definition}

## Few-shot calibration examples
{few_shot_examples}

## Now evaluate the following
Question   : {question}
Context    : {context}
Answer     : {answer}
Ground truth: {ground_truth}

## Your evaluation (chain-of-thought then JSON)
Think step-by-step:
<reasoning>
[Your reasoning here]
</reasoning>

Then return:
{{"score": <float 0-1>, "reasoning": "<one sentence summary>", "flag": "<issue if any>"}}
"""

# ── Few-shot calibration examples per criterion ───────────────────
_FEW_SHOT = {
    "faithfulness": """\
Example 1 — Score 1.0:
  Context : "The contract expires on 31 Dec 2025."
  Answer  : "The contract expires on 31 December 2025."
  → Every claim is directly supported. Score: 1.0

Example 2 — Score 0.0:
  Context : "The contract expires on 31 Dec 2025."
  Answer  : "The contract was signed in 2020 and expires in 2026."
  → "signed in 2020" and "2026" are not in context. Score: 0.0

Example 3 — Score 0.5:
  Context : "Article 47 limits termination to economic grounds."
  Answer  : "Termination is limited to economic grounds per Article 47,
             and requires 30-day notice."  ← "30-day notice" not in context
  → Half the claims are grounded. Score: 0.5""",

    "answer_relevance": """\
Example 1 — Score 1.0:
  Question: "What is the notice period for termination?"
  Answer  : "The notice period is 30 days as specified in Article 5."
  → Directly answers the question. Score: 1.0

Example 2 — Score 0.2:
  Question: "What is the notice period for termination?"
  Answer  : "The contract covers many aspects of employment including
             salary, benefits, and working hours."
  → Does not address notice period at all. Score: 0.2""",

    "completeness": """\
Example 1 — Score 1.0:
  Question: "What are the three conditions for contract termination?"
  Answer  : "The three conditions are: (1) economic grounds, (2) misconduct,
             (3) mutual agreement."
  → All three sub-questions answered. Score: 1.0

Example 2 — Score 0.33:
  Question: "What are the three conditions for contract termination?"
  Answer  : "Economic grounds are one valid reason."
  → Only one of three sub-questions answered. Score: 0.33""",

    "abstention_quality": """\
Example 1 — Score 1.0 (correct abstention):
  Question: "What is the CEO's salary?"  (not in document)
  Answer  : "The document does not contain information about the CEO's salary."
  → Correctly refused. Score: 1.0

Example 2 — Score 0.0 (hallucinated answer):
  Question: "What is the CEO's salary?"  (not in document)
  Answer  : "The CEO's salary is $500,000 per year."
  → Answered without basis. Score: 0.0

Example 3 — Score 0.0 (wrong abstention):
  Question: "What is the contract duration?"  (clearly in document)
  Answer  : "I don't have enough information."
  → Refused when answer was available. Score: 0.0""",
}


# ══════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════

class GenerationMetricsEvaluator:
    """
    Evaluates Core Generation Metrics for RAG systems.
    Uses LLM-as-Judge for subjective metrics.

    Parameters
    ----------
    generation_client : the answer-generation LLM 
    judge_client      : a MORE capable model used as judge .
                        If None, falls back to generation_client (not recommended).
    embedding_client  : for Answer Relevance reverse-embedding check.
    """

    def __init__(
        self,
        generation_client,
        judge_client      = None,
        embedding_client  = None,
    ):
        self.gen    = generation_client
        self.judge  = judge_client
        self.embed  = embedding_client

    # entry point
    async def evaluate(
        self,
        question:     str,
        answer:       str,
        contexts:     List[str],          # retrieved chunk texts
        ground_truth: str = "",
        is_unanswerable: bool = False,    # True for hallucination-type questions
    ) -> Dict[str, float]:
        """
        Returns a flat dict ready to merge into existing result dicts.
        All scores are 0–1. None means "not computed".
        """
        if not answer:
            return self._zero_result()

        context_str = "\n\n---\n\n".join(contexts) if contexts else ""

        results: Dict[str, float] = {}

        # ── 3.1 Core Generation Metrics ───────────────────────────
        results["faithfulness"]       = await self._faithfulness(answer, context_str, question, ground_truth)
        results["answer_relevance"]   = await self._answer_relevance(answer, question)
        results["completeness"]       = await self._completeness(answer, question, context_str, ground_truth)
        results["abstention_quality"] = await self._abstention_quality(answer, question, context_str, is_unanswerable)

        # ── Composite generation score ────────────────────────────
        results["generation_score_v2"] = self._composite_score(results)

        return results

    # Faithfulness
    async def _faithfulness(
        self, answer: str, context: str, question: str, ground_truth: str
    ) -> float:
        """Claims supported by context / total claims."""
        return await self._llm_judge(
            criterion_name="Faithfulness",
            criterion_definition=(
                "Score 1.0 if EVERY factual claim in the answer is explicitly supported "
                "by the context. Score 0.0 if the answer contains claims absent from "
                "or contradicting the context. Partial credit for mixed answers."
            ),
            few_shot=_FEW_SHOT["faithfulness"],
            question=question,
            context=context[:3000],
            answer=answer,
            ground_truth=ground_truth,
        )

    #  Answer Relevance
    async def _answer_relevance(self, answer: str, question: str) -> float:
        """
        Use LLM judge to evaluate answer relevance.
        (Changed from embedding approach to be consistent with other metrics)
        """
        return await self._llm_judge(
            criterion_name="Answer Relevance",
            criterion_definition=(
                "Score 1.0 if the answer directly and completely addresses "
                "the question. Score 0.0 if the answer is off-topic or ignores "
                "the core of the question."
            ),
            few_shot=_FEW_SHOT["answer_relevance"],
            question=question,
            context="",
            answer=answer,
            ground_truth="",
        )

    #  Completeness
    async def _completeness(
        self, answer: str, question: str, context: str, ground_truth: str
    ) -> float:
        """LLM judge: does the answer cover all sub-questions?"""
        return await self._llm_judge(
            criterion_name="Completeness",
            criterion_definition=(
                "Score 1.0 if the answer addresses ALL aspects of the question — "
                "no important sub-question is left unanswered. "
                "Score 0.0 if the answer is partial and misses key parts of the question."
            ),
            few_shot=_FEW_SHOT["completeness"],
            question=question,
            context=context[:1500],
            answer=answer,
            ground_truth=ground_truth,
        )

    # Abstention Quality
    async def _abstention_quality(
        self, answer: str, question: str, context: str, is_unanswerable: bool
    ) -> float:
        """
        Correct refusals / total unanswerable queries.
        For answerable questions:  penalise wrong abstentions.
        For unanswerable questions: reward correct abstentions.
        """
        abstention_phrases = [
            "don't have", "do not have", "cannot answer", "not available",
            "insufficient information", "not found in", "no information",
            "unable to answer", "not enough information", "cannot provide",
            "not in the document", "لا تحتوي", "لا يوجد", "غير متاح",
        ]
        answer_lower = answer.lower()
        did_abstain  = any(p in answer_lower for p in abstention_phrases)

        if is_unanswerable:
            # Correct behaviour = abstain
            return 1.0 if did_abstain else 0.0
        else:
            # Correct behaviour = answer
            if did_abstain:
                # Wrong abstention — use LLM judge to decide if context really had the answer
                return await self._llm_judge(
                    criterion_name="Abstention Quality",
                    criterion_definition=(
                        "The question is answerable from the context. "
                        "Score 0.0 if the model refused to answer when it had the information. "
                        "Score 1.0 if the refusal was actually correct (context truly insufficient)."
                    ),
                    few_shot=_FEW_SHOT["abstention_quality"],
                    question=question,
                    context=context[:1500],
                    answer=answer,
                    ground_truth="",
                )
            return 1.0   # answered when it should have — abstention quality is fine

    # LLM-as-Judge core
    async def _llm_judge(
        self,
        criterion_name:       str,
        criterion_definition: str,
        few_shot:             str,
        question:             str,
        context:              str,
        answer:               str,
        ground_truth:         str,
    ) -> float:
        """
        Uses the judge client (stronger model) with:
        - System prompt enforcing role + rules
        - Chain-of-thought <reasoning> tag
        - 4-6 few-shot calibration examples
        - Structured JSON output
        """
        prompt = _JUDGE_TEMPLATE.format(
            criterion_name       = criterion_name,
            criterion_definition = criterion_definition,
            few_shot_examples    = few_shot,
            question             = question[:500],
            context              = context[:2000],
            answer               = answer[:800],
            ground_truth         = ground_truth[:500],
        )

        try:
            # Use generate_json if available (OpenAI enforces JSON output)
            if hasattr(self.judge, "generate_json"):
                system_msgs = []
                if hasattr(self.judge, "construct_prompt"):
                    system_msgs = [
                        self.judge.construct_prompt(
                            prompt=_JUDGE_SYSTEM,
                            role=self.judge.enums.SYSTEM.value,
                        )
                    ]
                result = self.judge.generate_json(
                    prompt=prompt,
                    chat_history=system_msgs,
                    max_output_tokens=400,
                )
                score = float(result.get("score", 0.0))
                return max(0.0, min(1.0, score))

            # Fallback: generate_text + parse
            raw = self.judge.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.0,
                max_output_tokens=400,
            )
            return self._parse_judge_score(raw or "")

        except Exception as exc:
            logger.warning(f"[GenMetrics] LLM judge failed for {criterion_name}: {exc}")
            return 0.0

    @staticmethod
    def _parse_judge_score(raw: str) -> float:
        """Extract score from raw LLM output."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        # Try JSON
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict) and "score" in obj:
                return max(0.0, min(1.0, float(obj["score"])))
        except Exception:
            pass
        # Fallback: first float in text
        matches = re.findall(r"\b(0\.\d+|1\.0|0|1)\b", cleaned)
        if matches:
            return max(0.0, min(1.0, float(matches[0])))
        return 0.0

    # Helpers
    @staticmethod
    def _zero_result() -> Dict[str, float]:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "completeness": 0.0,
            "abstention_quality": 0.0,
            "generation_score_v2": 0.0,
        }

    @staticmethod
    def _composite_score(r: Dict) -> float:
        """
        Weighted composite of core generation metrics.
        Faithfulness is weighted highest (most critical).
        """
        weights = {
            "faithfulness":      0.40,
            "answer_relevance":  0.35,
            "completeness":      0.20,
            "abstention_quality":0.05,
        }
        score = sum(
            r.get(k, 0.0) * w for k, w in weights.items()
        )
        return round(score, 4)