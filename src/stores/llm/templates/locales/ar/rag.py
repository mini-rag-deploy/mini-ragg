from string import Template


### RAG PROMPTS ####

# system prompt for RAG
system_prompt=Template("\n".join([
    "أنت مساعد للإجابة على الأسئلة بناءً على المستندات المسترجعة التالية",
    "يرجى تقديم إجابة شاملة ودقيقة للسؤال"
])
)
## document prompt template for RAG

document_prompt=Template(
    "\n".join([
        "## المستند رقم: $doc_num",
        "### المحتوى: $chunk_text",
    ])
)

### footer prompt for RAG
footer_prompt=Template(
    "\n".join([
        "بناءً على المستندات المسترجعة أعلاه، يرجى الإجابة على السؤال التالي:",
        "### السؤال: $query",
        "### الإجابة:"
    ])
)