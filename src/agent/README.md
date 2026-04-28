# Agentic RAG with Dynamic Source Selection

This module extends the RAG system with intelligent decision-making capabilities to dynamically select information sources based on query requirements.

## 🎯 Overview

The Agentic RAG system adds a **Dynamic Source Selection** layer that:

1. **Evaluates** if current context is sufficient to answer the question
2. **Decides** which source to use: Vector DB, Tools, or Internet
3. **Fetches** information from the selected source
4. **Integrates** new information with existing context
5. **Generates** a comprehensive answer

## 🏗️ Architecture

```
User Query
    ↓
Retrieve from Vector DB
    ↓
Grade Documents
    ↓
Generate Answer
    ↓
Need More Details? ──→ NO ──→ Audit Answer ──→ END
    ↓ YES
Select Source
    ↓
    ├─→ Vector DB (retry with better query)
    ├─→ Tools (calculator, APIs, converters)
    └─→ Internet (web search)
    ↓
Fetch Information
    ↓
Re-generate Answer (with new context)
    ↓
Audit Answer ──→ END
```

## 📁 Module Structure

```
src/agent/
├── __init__.py                 # Module exports
├── source_router.py            # Core routing logic
├── tools_registry.py           # Tool management
├── internet_retriever.py       # Web search
├── prompts.py                  # LLM prompts
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Basic Integration

```python
from agent.source_router import SourceRouter
from agent.tools_registry import ToolsRegistry
from agent.internet_retriever import InternetRetriever
from graph.rag_graph_agentic import build_agentic_rag_graph

# Initialize components
tools_registry = ToolsRegistry()
internet_retriever = InternetRetriever(backend="mock")

source_router = SourceRouter(
    generation_client=nlp_controller.generation_client,
    tools_registry=tools_registry,
    internet_retriever=internet_retriever,
)

# Build agentic graph
graph = build_agentic_rag_graph(
    nlp_controller=nlp_controller,
    project=project,
    source_router=source_router,
    enable_source_selection=True,
)

# Execute
result = await graph.ainvoke({
    "question": "What is 25 * 4 + 10?",
    "documents": [],
    "answer": None,
    "iterations": 0,
    "sources_tried": [],
})

print(result["answer"])
```

### 2. Using the Agentic Controller Wrapper

```python
from agent.example_integration import AgenticNLPController

# Wrap existing NLP controller
agentic_controller = AgenticNLPController(
    nlp_controller=nlp_controller,
    enable_tools=True,
    enable_internet=True,
    internet_backend="mock",
)

# Use it
answer, metadata, _ = await agentic_controller.answer_agentic_rag_question(
    project=project,
    query="What is the current weather in Paris?",
    enable_source_selection=True,
)

print(f"Answer: {answer}")
print(f"Sources used: {metadata['sources_tried']}")
```

## 🔧 Components

### 1. SourceRouter

The brain of the system. Makes decisions about:
- Whether more information is needed
- Which source to use
- How to query each source

**Key Methods:**
```python
# Decide if more details are needed
need_more, reason = await source_router.decide_need_more_details(
    question="What is IoT?",
    context="Current context...",
    answer="Current answer...",
)

# Select best source
selection = await source_router.select_source(
    question="What is IoT?",
    context="Current context...",
    previous_sources=["vector_db"],
)

# Complete workflow
result = await source_router.route_and_fetch(
    question="What is IoT?",
    context="Current context...",
    nlp_controller=nlp_controller,
    project=project,
)
```

### 2. ToolsRegistry

Manages callable tools (APIs, calculators, utilities).

**Built-in Tools:**
- `calculator`: Evaluate mathematical expressions
- `get_current_time`: Get current date/time
- `unit_converter`: Convert between units (length, weight, temperature)
- `json_parser`: Parse and validate JSON

**Register Custom Tools:**
```python
def weather_api(city: str) -> dict:
    # Call weather API
    return {"temperature": 22, "condition": "sunny"}

tools_registry.register_tool(
    name="weather_api",
    description="Get current weather for a city",
    function=weather_api,
    parameters={
        "city": {
            "type": "string",
            "required": True,
            "description": "City name",
        }
    },
)
```

### 3. InternetRetriever

Handles web search for real-time information.

**Supported Backends:**
- `mock`: Simulated search (for testing)
- `duckduckgo`: Privacy-focused search (no API key)
- `google`: Google Custom Search (requires API key)
- `bing`: Bing Search (requires API key)

**Usage:**
```python
retriever = InternetRetriever(backend="mock")

results = await retriever.search("Python tutorials", top_k=5)

for result in results:
    print(f"{result.title}: {result.snippet}")
```

## 📊 Graph Flow

### Classic RAG (enable_source_selection=False)
```
Retrieve → Grade → Generate → Audit → END
```

### Agentic RAG (enable_source_selection=True)
```
Retrieve → Grade → Generate → Need More? 
                                    ↓ YES
                              Select Source
                                    ↓
                              Fetch Data
                                    ↓
                              Re-generate
                                    ↓
                                 Audit → END
```

## 🎨 Customization

### Add Custom Tools

```python
# Define tool function
async def database_query(sql: str) -> dict:
    # Execute SQL query
    results = await db.execute(sql)
    return {"rows": results}

# Register
tools_registry.register_tool(
    name="database_query",
    description="Execute SQL queries on the database",
    function=database_query,
    parameters={
        "sql": {
            "type": "string",
            "required": True,
            "description": "SQL query to execute",
        }
    },
    is_async=True,
)
```

### Add Custom Internet Backend

```python
class CustomSearchBackend(InternetRetriever):
    async def _custom_search(self, query: str, top_k: int):
        # Implement your search logic
        results = []
        # ... fetch from your API
        return results
```

### Customize Decision Prompts

Edit `src/agent/prompts.py` to modify:
- `NEED_MORE_DETAILS_PROMPT`: Decision criteria
- `SOURCE_SELECTION_PROMPT`: Source selection logic
- `TOOL_PARAMS_EXTRACTION_PROMPT`: Parameter extraction
- `INTERNET_QUERY_OPTIMIZATION_PROMPT`: Query optimization

## 🧪 Testing

### Test Source Router
```python
# Test decision making
need_more, reason = await source_router.decide_need_more_details(
    question="What is 2+2?",
    context="",
    answer="",
)
assert need_more == True  # No context, needs more info

# Test source selection
selection = await source_router.select_source(
    question="What is 2+2?",
    context="",
)
assert selection["source"] == "tools"  # Should select calculator
```

### Test Tools
```python
# Test calculator
result = await tools_registry.execute_tool(
    "calculator",
    expression="2 + 2",
)
assert result["result"] == 4

# Test unit converter
result = await tools_registry.execute_tool(
    "unit_converter",
    value=100,
    from_unit="cm",
    to_unit="m",
)
assert result["result"] == 1.0
```

### Test Internet Retriever
```python
retriever = InternetRetriever(backend="mock")
results = await retriever.search("test query", top_k=3)
assert len(results) <= 3
assert all(hasattr(r, "title") for r in results)
```

## 📈 Performance Considerations

### Source Selection Strategy
- **Vector DB first**: Always try internal documents first
- **Tools for computation**: Use tools for calculations, conversions
- **Internet for real-time**: Use internet for current events, weather, etc.

### Iteration Limits
- Default: `max_iterations=2`
- Prevents infinite loops
- Balances quality vs. latency

### Source Limits
- Max 3 sources per query
- Prevents excessive API calls
- Ensures reasonable response time

## 🔒 Security

### Tool Execution
- Tools run in isolated context
- Input validation required
- No arbitrary code execution

### Internet Search
- Results are sanitized
- No direct code execution from web content
- Rate limiting recommended

### API Keys
- Store in environment variables
- Never commit to version control
- Use secrets management in production

## 📝 Example Queries

### Queries that use Vector DB
```
"What is IoT according to the documents?"
"Explain the concept from chapter 3"
"Summarize the key points"
```

### Queries that use Tools
```
"What is 25 * 4 + 10?"
"Convert 100 km to miles"
"What time is it?"
"Parse this JSON: {...}"
```

### Queries that use Internet
```
"What is the current weather in Paris?"
"Latest news about AI"
"Who won the 2024 World Cup?"
```

## 🚦 Status Indicators

The system tracks:
- `sources_tried`: List of sources attempted
- `selected_source`: Current source being used
- `need_more_details`: Whether more info is needed
- `source_reason`: Why a source was selected

## 🎯 Best Practices

1. **Enable source selection for complex queries**: Use `enable_source_selection=True` for queries that might need external data

2. **Disable for simple document queries**: Use `enable_source_selection=False` for pure document retrieval

3. **Register domain-specific tools**: Add tools relevant to your use case

4. **Monitor source usage**: Track which sources are used most frequently

5. **Tune iteration limits**: Adjust `max_iterations` based on your latency requirements

6. **Cache tool results**: Implement caching for expensive tool calls

7. **Rate limit internet searches**: Prevent API quota exhaustion

## 🔄 Migration from Classic RAG

### Before (Classic RAG)
```python
answer, metadata, _ = await nlp_controller.answer_rag_question(
    project=project,
    query=query,
    use_self_correction=True,
)
```

### After (Agentic RAG)
```python
agentic_controller = AgenticNLPController(nlp_controller)

answer, metadata, _ = await agentic_controller.answer_agentic_rag_question(
    project=project,
    query=query,
    enable_source_selection=True,
)
```

## 📚 Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Agentic AI Patterns](https://www.anthropic.com/research/building-effective-agents)

## 🤝 Contributing

To add new features:

1. **New Tools**: Add to `tools_registry.py`
2. **New Sources**: Extend `SourceRouter` in `source_router.py`
3. **New Prompts**: Add to `prompts.py`
4. **New Backends**: Extend `InternetRetriever` in `internet_retriever.py`

## 📄 License

Same as parent project.
