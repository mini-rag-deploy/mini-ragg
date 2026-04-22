from .LLMEnums import LLMEnums, CoHereEnums, DocumentTypeEnums
from .providers import CoHereProvider, OpenAIProvider

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
                backup_api_key=getattr(self.config, 'COHERE_API_KEY_BACKUP', None),  # Add backup key
                backup_api_key2=getattr(self.config, 'COHERE_API_KEY_BACKUP2', None),
                backup_api_key3=getattr(self.config, 'COHERE_API_KEY_BACKUP3', None),
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.OUTPUT_DEFAULT_MAX_CHARACTERS,
                default_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )

        return None
       