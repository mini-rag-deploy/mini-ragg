from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import logging
import json
from typing import List, Union

class OpenAIProvider(LLMInterface):

    def __init__(self, api_key: str, api_url: str = None,
                    default_input_max_characters: int = 1000,
                    default_output_max_tokens: int = 2048 ,
                    default_temperature: float = 0.7
                     ):
        
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_output_max_tokens = default_output_max_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url if self.api_url else None
        )
        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)
    
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
    
    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str, is_prompt: bool = False):
        limit = max(self.default_input_max_characters, 20000)
        
        if len(text) <= limit:
            return text.strip()
            
        if is_prompt:
            # Smart Truncation: Keep 70% from the start (Instructions + Context) 
            # and 30% from the end (Question + Final Formatting)
            top_part = int(limit * 0.7)
            bottom_part = int(limit * 0.3)
            truncated_text = text[:top_part] + "\n\n...[MIDDLE CONTENT TRUNCATED DUE TO LENGTH]...\n\n" + text[-bottom_part:]
            return truncated_text.strip()
            
        # Regular truncation for plain text (just cut from the end)
        return text[:limit].strip()

    def generate_text(self, prompt: str , chat_history:list=[], max_output_tokens:int=None,
                       temperature: float = None):
        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None
        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set.")
            return None
        self.logger.info(f"Generation model ID openai is set")
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_output_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        
        chat_history.append(self.construct_prompt(prompt, role=OpenAIEnums.USER.value))

        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=chat_history,
            max_tokens=max_output_tokens,
            temperature=temperature
        )

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("No response data received from OpenAI.")
            return None
        
        return response.choices[0].message.content
    
    # ============ NEW: for graders that return JSON ============
    def generate_json(self, prompt: str, chat_history: list = [],
                      max_output_tokens: int = 256) -> dict:
        """
        Uses response_format=json_object to ensure valid JSON always
        """
        if not self.client or not self.generation_model_id:
            return {}

        messages = list(chat_history)
        messages.append(self.construct_prompt(prompt, role=OpenAIEnums.USER.value))

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=0,  # ← always 0 for graders to be deterministic
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            self.logger.error(f"generate_json failed: {e}")
            return {}



    def embed_text(self, text: Union[str, List[str]], document_type: str=None):
        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None

        if isinstance(text, str):
            text = [text]

        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set.")
            return None
        
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model_id
        )

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("No embedding data received from OpenAI.")
            return None
        
        return [f.embedding for f in response.data]

    def construct_prompt(self, prompt: str , role: str):
        return{
            "role": role,
            "content": prompt
        }