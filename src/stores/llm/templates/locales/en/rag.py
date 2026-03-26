from string import Template
### RAG PROMPTS ####

# system prompt for RAG

system_prompt=Template("\n".join([
    "You are an assistant for answering questions based on the following retrieved documents ",
    "Please provide a comprehensive and accurate answer to the question"
])
)


## document prompt template for RAG

document_prompt=Template(
    "\n".join([
        "## Document No: $doc_num",
        "### Content: $chunk_text",
    ])
)



### footer prompt for RAG

footer_prompt=Template(
    "\n".join([
        "Based on the above retrieved documents, please answer the following question:",
        "### Answer:"
    ])
)







