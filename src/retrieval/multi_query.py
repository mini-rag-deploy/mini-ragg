# src/retrieval/multi_query.py
"""
Multi-query expansion.

Problem:
  A single query phrased one way may miss documents phrased differently.
  "conditions for contract termination" misses chunks that say
  "grounds for employee dismissal" or "أسباب إنهاء العقد".

Solution:
  Use the LLM to generate N alternative phrasings of the same question,
  search with all of them, then fuse results via RRF.

Edge cases handled:
- LLM returns malformed JSON or plain text (robust parser)
- LLM returns fewer variants than requested (uses whatever it returns)
- LLM returns duplicates (deduplicated before searching)
- Query in Arabic stays in Arabic; English stays in English
- Generation failure (falls back to original query only)
- All expansions return identical results (RRF handles gracefully)
"""

from __future__ import annotations

import json
import logging
import re
from typing import List

# Use uvicorn logger to ensure logs appear
logger = logging.getLogger("uvicorn.error")


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────
MULTI_QUERY_PROMPT = """\
You are a search-query expansion assistant for a legal document retrieval system.

## Task
Given the original query below, generate {n} alternative search queries that:
1. Preserve the original intent exactly.
2. Use different vocabulary, legal terminology, or sentence structure.
3. Are written in the SAME language as the original query.
   - Arabic query → all alternatives in Arabic.
   - English query → all alternatives in English.
4. Cover different angles: synonyms, formal/informal phrasings, broader/narrower scope.

## Original Query
{query}

## Output Format
Return ONLY a JSON array of strings — no explanation, no markdown, no extra text.
Example: ["alternative 1", "alternative 2", "alternative 3"]

JSON array:
"""


# ─────────────────────────────────────────────
# Parser (handles LLM output variations)
# ─────────────────────────────────────────────
def _parse_llm_output(raw: str) -> List[str]:
    """
    Robust parser for LLM output that should be a JSON array.
    Handles:
    - Clean JSON: ["q1", "q2"]
    - Markdown-wrapped: ```json\n["q1"]\n```
    - Numbered list: 1. q1\n2. q2
    - Mixed: partially valid JSON
    """
    if not raw:
        return []

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass

    # Try finding JSON array anywhere in the text
    array_match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: treat lines as individual queries
    lines = [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in cleaned.split("\n")
        if line.strip() and not line.strip().startswith(("#", "-", "*", "[", "]"))
    ]
    queries = [l for l in lines if len(l) > 5]  # filter noise
    return queries


# ─────────────────────────────────────────────
# Multi-Query Expander
# ─────────────────────────────────────────────
class MultiQueryExpander:
    """
    Generates query variants using the generation_client.

    Parameters
    ----------
    generation_client : any LLM provider with generate_text()
    n_variants        : number of alternative queries to generate
    include_original  : always include the original query in the results
    """

    def __init__(
        self,
        generation_client,
        n_variants:       int  = 3,
        include_original: bool = True,
    ):
        self.generation_client = generation_client
        self.n_variants        = n_variants
        self.include_original  = include_original

    def expand(self, query: str) -> List[str]:
        """
        Returns a list of queries (original + variants, deduplicated).
        Falls back to [original] if generation fails.
        """
        if not query or not query.strip():
            return []

        prompt = MULTI_QUERY_PROMPT.format(
            n=self.n_variants,
            query=query.strip(),
        )

        try:
            raw = self.generation_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.3,         # slight creativity, not too wild
                max_output_tokens=512,
            )

            variants = _parse_llm_output(raw or "")

            if not variants:
                logger.warning("[MultiQuery] LLM returned no valid variants")
                return [query] if self.include_original else []


            # Deduplicate while preserving order
            seen: set = set()
            unique: List[str] = []

            if self.include_original:
                seen.add(query.strip().lower())
                unique.append(query.strip())

            for v in variants:
                key = v.strip().lower()
                if key not in seen and len(v.strip()) > 5:
                    seen.add(key)
                    unique.append(v.strip())
            
            for v in unique:
                logger.debug(f"[MultiQuery] Variant: {v}")

            logger.info(
                f"[MultiQuery] '{query[:50]}' → {len(unique)} queries total"
            )
            return unique

        except Exception as exc:
            logger.error(f"[MultiQuery] Expansion failed: {exc}")
            return [query] if self.include_original else [query]