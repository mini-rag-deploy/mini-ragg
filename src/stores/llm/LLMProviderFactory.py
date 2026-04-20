from .LLMEnums import LLMEnums, CoHereEnums, DocumentTypeEnums
from .providers import CoHereProvider, OpenAIProvider, GemmaProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                api_url=self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.OUTPUT_DEFAULT_MAX_CHARACTERS,
                default_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )

        if provider == LLMEnums.COHERE.value:
            
            return CoHereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.OUTPUT_DEFAULT_MAX_CHARACTERS,
                default_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )

        if provider == LLMEnums.GEMINI.value:
            return GeminiProvider(
                api_key=self.config.GEMINI_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.OUTPUT_DEFAULT_MAX_CHARACTERS,
                default_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )

        if provider == LLMEnums.GEMMA.value:
            return GemmaProvider(
                api_key=self.config.GEMMA_API_KEY if hasattr(self.config, 'GEMMA_API_KEY') and self.config.GEMMA_API_KEY else self.config.OPENAI_API_KEY,
                api_url=self.config.GEMMA_API_URL if hasattr(self.config, 'GEMMA_API_URL') and self.config.GEMMA_API_URL else self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.OUTPUT_DEFAULT_MAX_CHARACTERS,
                default_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )

        return None
       