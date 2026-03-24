from enum import Enum

class LLMEnums(Enum):
    """
    Enum class for LLM related constants.
    """
    # LLM Types
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    HUGGINGFACE = "HUGGINGFACE"
    CUSTOM = "CUSTOM"
    GEMINI = "GEMINI"

    # LLM Models
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    BERT = "bert-base-uncased"
    ROBERTA = "roberta-base"

    # LLM Providers
    OPENAI_API = "OpenAI API"
    HUGGINGFACE_HUB = "Hugging Face Hub"

class OpenAIEnums(Enum):
    SYSTEM= "system"
    USER = "user"
    ASSISTANT = "assistant"

class CoHereEnums(Enum):
    SYSTEM= "SYSTEM"
    USER = "user"
    ASSISTANT = "CHATBOT"
    
    DOCUMENT= "search_document"
    QUERY = "search_query"

class DocumentTypeEnums(Enum):
    DOCUMENT= "document"
    QUERY = "query"



