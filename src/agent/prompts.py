"""
Prompts for the Dynamic Source Selection Agent.

These prompts enable the system to:
1. Decide if current context is sufficient
2. Select the best information source
3. Format tool calls and internet queries
"""


# ──────────────────────────────────────────────────────────────
# 1. NEED MORE DETAILS DECISION
#    Task: Determine if current context is sufficient to answer
#    Output: JSON {"need_more": bool, "reason": "..."}
# ──────────────────────────────────────────────────────────────
NEED_MORE_DETAILS_PROMPT = """\
You are an information sufficiency evaluator for a RAG system.

## Your Task
Analyze whether the current context provides enough information to answer the user's question
with high confidence and completeness. Consider both the quality and relevance of available information.

## Inputs
Question: {question}

Current Context:
{context}

Current Answer (if any):
{answer}

## Evaluation Criteria
Score "need_more: true" if ANY of these apply:
- The context is empty or contains no relevant information
- The context is too vague or generic to provide a specific answer
- The question asks for real-time data (weather, stock prices, current events)
- The question requires external tools (calculations, API calls, conversions)
- The question asks about information outside the document scope (e.g., "What's the weather?")
- The answer explicitly states insufficient information
- Key facts or details are missing to fully answer the question

Score "need_more: false" if:
- The context contains sufficient, relevant information to answer completely
- The answer is comprehensive and grounded in the context
- No external data or tools are required

## Output Format
Return a single, valid JSON object — no markdown, no explanation outside the JSON.

{{"need_more": true, "reason": "<why more information is needed>"}}
{{"need_more": false, "reason": "<why current context is sufficient>"}}
"""


# ──────────────────────────────────────────────────────────────
# 2. SOURCE SELECTION
#    Task: Choose the best source for additional information
#    Output: JSON {"source": "vector_db" | "tools" | "internet", "reason": "..."}
# ──────────────────────────────────────────────────────────────
SOURCE_SELECTION_PROMPT = """\
You are an intelligent source router for a hybrid RAG system.

## Your Task
Analyze the question and determine the BEST source to retrieve additional information from.
Choose the most appropriate source based on the nature of the query.

## Question
{question}

## Context (what we already tried)
{context}

## Available Sources

### 1. vector_db
- Internal document collection (PDFs, text files, knowledge base)
- Best for: domain-specific knowledge, company documents, technical documentation
- Use when: the question is about topics that should be in your documents

### 2. tools
- External APIs, calculators, converters, database queries
- Best for: computations, data transformations, structured data lookups
- Use when: the question requires calculation, conversion, or API calls
- Available tools: {available_tools}

### 3. internet
- Live web search for current information
- Best for: real-time data, current events, latest information, general knowledge
- Use when: the question asks for up-to-date information not in documents

## Selection Rules
1. **CRITICAL**: NEVER select a source that appears in "Previous sources tried" in the context above
2. Prefer "vector_db" if the question is about topics in your document domain AND it hasn't been tried yet
3. Choose "tools" if the question requires computation or API calls
4. Choose "internet" for real-time data, current events, or general knowledge
5. If all sources have been tried, select "vector_db" as fallback

## Output Format
Return a single, valid JSON object — no markdown, no explanation outside the JSON.

{{"source": "vector_db", "reason": "<why this source is best>", "query": "<optimized query for this source>"}}
{{"source": "tools", "reason": "<why this source is best>", "tool_name": "<specific tool to use>", "tool_params": {{"param": "value"}}}}
{{"source": "internet", "reason": "<why this source is best>", "query": "<search query>"}}
"""


# ──────────────────────────────────────────────────────────────
# 3. TOOL PARAMETER EXTRACTION
#    Task: Extract parameters needed for a specific tool
#    Output: JSON with tool parameters
# ──────────────────────────────────────────────────────────────
TOOL_PARAMS_EXTRACTION_PROMPT = """\
You are a parameter extraction assistant for tool execution.

## Your Task
Extract the required parameters from the user's question to call the specified tool.

## Question
{question}

## Tool Information
Tool Name: {tool_name}
Tool Description: {tool_description}
Required Parameters: {required_params}

## Instructions
1. Analyze the question and extract values for each required parameter
2. Use reasonable defaults if a parameter is not explicitly mentioned
3. Ensure parameter types match the tool's requirements
4. Return a valid JSON object with all required parameters

## Output Format
Return a single, valid JSON object — no markdown, no explanation outside the JSON.

{{"param1": "value1", "param2": "value2", ...}}
"""


# ──────────────────────────────────────────────────────────────
# 4. INTERNET QUERY OPTIMIZATION
#    Task: Convert a question into an optimal web search query
#    Output: Plain text search query
# ──────────────────────────────────────────────────────────────
INTERNET_QUERY_OPTIMIZATION_PROMPT = """\
You are a web search query optimizer.

## Your Task
Convert the user's question into an optimal search query for web search engines.

## Original Question
{question}

## Optimization Rules
1. Extract key concepts and entities
2. Remove question words (what, how, when, where, why)
3. Use specific terms and proper nouns
4. Keep it concise (3-7 words ideal)
5. Preserve the same language as the question
6. Add relevant context keywords if needed

## Examples
Question: "What is the current weather in Paris?"
Query: "Paris weather current"

Question: "How do I calculate compound interest?"
Query: "compound interest formula calculation"

Question: "Who won the 2024 World Cup?"
Query: "2024 World Cup winner"

## Output
Return ONLY the optimized search query as plain text — no JSON, no explanation, no quotes.

Optimized Query:
"""


# ──────────────────────────────────────────────────────────────
# 5. CONTEXT INTEGRATION
#    Task: Integrate new information from external sources
#    Output: Natural language answer
# ──────────────────────────────────────────────────────────────
CONTEXT_INTEGRATION_PROMPT = """\
You are an information synthesis assistant.

## Your Task
Integrate the newly retrieved information with the existing context to provide a comprehensive answer.

## Original Question
{question}

## Existing Context
{existing_context}

## New Information (from {source})
{new_information}

## Instructions
1. Combine both sources of information intelligently
2. Prioritize the most relevant and recent information
3. Cite sources when appropriate (e.g., "According to the documents..." or "Based on current data...")
4. Maintain the same language as the question
5. Be concise but complete
6. If information conflicts, note the discrepancy

## Answer
"""
