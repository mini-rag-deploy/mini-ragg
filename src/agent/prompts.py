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
You are an information sufficiency evaluator for a RAG system specialized in IoT (Internet of Things) topics.

## Your Task
Analyze the user's question and current context to determine:
1. Is the question within the IoT domain scope?
2. If yes, is the current context sufficient to answer it?

## Domain Scope Definition
**IoT Domain includes:**
- IoT architecture, layers (perception, network, application)
- IoT devices, sensors, actuators, gateways
- IoT protocols (MQTT, CoAP, OPC-UA, etc.)
- IoT networks (LoRaWAN, Zigbee, 6LoWPAN, etc.)
- Industrial IoT (IIoT), smart factories, manufacturing
- IoT security, privacy, authentication
- IoT applications (smart cities, healthcare, agriculture, etc.)
- IoT standards (TSN, IEEE, IETF, etc.)

**NOT in IoT Domain:**
- Sports, celebrities, entertainment
- General history, geography, politics
- Unrelated technology (pure software, databases without IoT context)
- Personal questions, general knowledge

## Inputs
Question: {question}

Current Context:
{context}

Current Answer (if any):
{answer}

## Decision Logic

### Case 1: Question is OUT OF SCOPE
If the question is clearly NOT related to IoT topics:
- Return {{"need_more": false, "reason": "Question is outside the IoT domain scope (e.g., about sports/celebrities/unrelated topics)"}}
- Examples: "Who is Messi?", "What is the capital of France?"

### Case 2: Question is IN SCOPE but context is SUFFICIENT
If the question is IoT-related AND context has enough information:
- Return {{"need_more": false, "reason": "Context provides sufficient IoT-related information"}}

### Case 3: Question is IN SCOPE but context is INSUFFICIENT
If the question is IoT-related BUT context is empty/incomplete/lacks details:
- Return {{"need_more": true, "reason": "Question is IoT-related but context lacks details about [specific aspect]"}}
- Examples: 
  - "Question about OPC-UA security but context only mentions OPC-UA basics"
  - "Question about IIoT applications but context is empty"
  - "Question about latest IoT standards but context has outdated info"

## Output Format
Return a single, valid JSON object — no markdown, no explanation outside the JSON.

{{"need_more": true, "reason": "<specific reason why more IoT information is needed>"}}
{{"need_more": false, "reason": "<reason: either sufficient context OR out of scope>"}}
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

### 1. internet (ONLY AVAILABLE SOURCE)
- Live web search for current information
- Best for: real-time data, current events, latest information, additional details
- Use when: the question is IoT-related but context is insufficient

## Selection Rules
1. **CRITICAL**: NEVER select a source that appears in "Previous sources tried" in the context above
2. **ALWAYS select "internet"** as the only external source available
3. If "internet" was already tried, do not select any source (return "internet" anyway as fallback)

## Output Format
Return a single, valid JSON object — no markdown, no explanation outside the JSON.

{{"source": "internet", "reason": "<why internet search is needed>", "query": "<search query>"}}
"""


# ──────────────────────────────────────────────────────────────
# 3. INTERNET QUERY OPTIMIZATION
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
