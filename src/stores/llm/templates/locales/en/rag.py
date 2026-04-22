from string import Template
from enum import Enum


class QuestionType(str, Enum):
    FACTUAL        = "factual"
    ANALYTICAL     = "analytical"
    COMPARATIVE    = "comparative"
    SUMMARIZATION  = "summarization"
    HALLUCINATION  = "hallucination" 


### RAG PROMPTS ####

# system prompt for RAG

system_prompt=Template("\n".join([
    "You are a precise question-answering assistant that answers ONLY from the retrieved documents provided below.",
    "",
    "## Core rules",
    "1. Base every claim strictly on the retrieved documents. Do NOT add knowledge from outside the documents.",
    "2. When you state a fact, immediately cite its source like (Doc N). Multiple sources: (Doc 1, Doc 3).",
    "3. If a piece of information appears in more than one document, note the agreement: (Doc 1, Doc 2 agree).",
    "4. If the documents contain conflicting information, state the conflict explicitly.",
    "5. If the answer cannot be found in the provided documents, respond ONLY with:",
    '   "The provided documents do not contain enough information to answer this question."',
    "   Do not guess or infer beyond what is written.",
    "6. When listing items, use letters (a, b, c …) NOT numbers to organise your response.",
    "7. Be comprehensive: address every aspect of the question that the documents support.",
    "8. Do not pad your answer with information unrelated to the question.",
])
)


## document prompt template for RAG

document_prompt=Template(
    "\n".join([
        "## Document No: $doc_num",
        "### Content: $chunk_text",
    ])
)

# ── Factual ──────────────────────────────────────────────────────
# Targets: answer_correctness, faithfulness, answer_relevance
footer_factual = Template("\n".join([
    "Based on the retrieved documents above, answer the following factual question.",
    "",
    "Rules:",
    "- Provide a direct, concise answer in 2–4 sentences maximum.",
    "- Focus on the core answer — do not list additional points beyond what directly answers the question.",
    "- Cite (Doc N) after every fact you state.",
    "- If the answer involves a list, use letters (a, b, c …) not numbers.",
    "- Only include facts explicitly stated in the documents.",
    "- Prioritize the highest-ranked document (Doc 1) as your primary source.",
    "- If the documents don't contain the answer, state:",
    '  "The provided documents do not contain enough information to answer this question."',
    "",
    "Question: $query",
    "",
    "### Answer:",
]))

# ── Analytical ───────────────────────────────────────────────────
# Targets: completeness, faithfulness, context_recall
footer_analytical = Template("\n".join([
    "Based on the retrieved documents above, answer the following analytical question.",
    "",
    "Think step-by-step:",
    "  Step a) Identify relevant evidence from the primary document (highest-ranked).",
    "  Step b) Reason about implications, causes, or effects strictly from that evidence.",
    "  Step c) Only extend to other documents if they directly add to the answer.",
    "",
    "Rules:",
    "- Support every analytical claim with (Doc N) citations.",
    "- The primary source for this question is the highest-ranked document — do not introduce technical concepts absent from it.",
    "- Do NOT introduce outside knowledge or assumptions not stated in the documents.",
    "- Use letters (a, b, c …) to organise points, not numbers.",
    "",
    "Question: $query",
    "",
    "### Answer:",
]))
 
# ── Comparative ──────────────────────────────────────────────────
# Targets: completeness, faithfulness, answer_correctness
footer_comparative = Template("\n".join([
    "Based on the retrieved documents above, compare the subjects in the question.",
    "",
    "Structure your answer as follows:",
    "  Similarities — what the documents say both subjects share.",
    "  Differences  — how the documents distinguish the subjects.",
    "",
    "Rules:",
    "- Cite (Doc N) after every comparative claim.",
    "- If the question specifies 'Text 1' and 'Text 2', limit your comparison exclusively to those two documents.",
    "- Only compare on dimensions the documents actually describe.",
    "- Do NOT speculate about dimensions the documents are silent on.",
    "- Do NOT add an 'Implications' section — focus only on similarities and differences.",
    "- Use letters (a, b, c …) for sub-points, not numbers.",
    "",
    "Question: $query",
    "",
    "### Answer:",
]))
 
# ── Summarization ────────────────────────────────────────────────
# Targets: completeness, context_recall, answer_relevance
footer_summarization = Template("\n".join([
    "Based on the retrieved documents above, provide a comprehensive summary that answers the question.",
    "",
    "Before writing the summary, mentally check:",
    "  ✓ Have I used information from every document that is relevant to the question?",
    "  ✓ Have I addressed every distinct sub-topic or aspect raised by the question?",
    "  ✓ Have I kept the summary grounded — no claims beyond the documents?",
    "",
    "Rules:",
    "- Cite (Doc N) when drawing on a specific document's content.",
    "- Use letters (a, b, c …) for listing points, not numbers.",
    "- After drafting, re-read the question and confirm your answer explicitly mentions each key concept the question asks about.",
    "- If the question uses technical terms (e.g., duty cycling, SLA, ADC/DAC), ensure those appear by name if the documents contain them.",
    "- Conclude with a one-sentence synthesis that directly answers the question.",
    "",
    "Question: $query",
    "",
    "### Answer:",
]))
 
# ── Unanswerable / Abstention ────────────────────────────────────
# Targets: abstention_quality (already 100% — preserve it)
footer_hallucination = Template("\n".join([
    "Based ONLY on the retrieved documents above, answer the following question.",
    "If the specific information requested is not present in any of the documents,",
    'respond with exactly: "The provided documents do not contain enough information to answer this question."',
    "Do not estimate, infer, or use outside knowledge.",
    "",
    "Question: $query",
    "",
    "### Answer:",
]))
 
# ── Default (fallback) ───────────────────────────────────────────
footer_default = Template("\n".join([
    "Based on the retrieved documents above, answer the following question.",
    "- Cite (Doc N) after every claim drawn from a specific document.",
    "- Use letters (a, b, c …) to organise points.",
    "- Do not include information that is not in the documents.",
    "",
    "Question: $query",
    "",
    "### Answer:",
]))




# Routing map
FOOTER_BY_TYPE = {
    QuestionType.FACTUAL:       footer_factual,
    QuestionType.ANALYTICAL:    footer_analytical,
    QuestionType.COMPARATIVE:   footer_comparative,
    QuestionType.SUMMARIZATION: footer_summarization,
    QuestionType.HALLUCINATION: footer_hallucination,
}





