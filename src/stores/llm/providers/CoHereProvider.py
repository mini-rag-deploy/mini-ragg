from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums,DocumentTypeEnums
import cohere
import logging

class CoHereProvider(LLMInterface):
    def __init__(self, api_key: str,
                    default_input_max_characters: int = 1000,
                    default_output_max_tokens: int = 2048 ,
                    default_temperature: float = 0.7
                     ):
        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_output_max_tokens = default_output_max_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)


    def set_generation_mode(self, model_id: str):
        self.generation_model_id = model_id
    
    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str , chat_history:list=[], max_output_tokens:int=None,
                       temperature: float = None):
        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return None
        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set.")
            return None
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_output_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        response = self.client.chat(
            model=self.generation_model_id,
            chat_history=chat_history,
            message= self.process_text(prompt),
            temperature=temperature,
            max_tokens=max_output_tokens
        )

        if not response or not response.text:
            self.logger.error("No response data received from Cohere.")
            return None
        
        return response.text

    def embed_text(self, text: str, document_type: str=None):
        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set.")
            return None

        input_type = CoHereEnums.DOCUMENT
        if document_type== DocumentTypeEnums.QUERY:
            input_type = CoHereEnums.QUERY
        
        response = self.client.embed(
            texts=[self.process_text(text)],
            model=self.embedding_model_id,
            input_type=input_type,
            embedding_types=['float']
        )

        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("No embedding data received from Cohere.")
            return None
        return response.embeddings.float[0]

    def construct_prompt(self, prompt: str , role: str):
        return{
            "role": role,
            "text": self.process_text(prompt)
        }