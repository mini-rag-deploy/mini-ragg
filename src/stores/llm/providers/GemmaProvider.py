from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums, DocumentTypeEnums
from openai import OpenAI
import logging
import json
import time
from typing import List, Union

class GemmaProvider(LLMInterface):
    """
    Gemma provider using OpenAI-compatible API endpoint.
    Works exactly like OpenAI provider but connects to custom Gemma endpoint.
    """

    def __init__(self, api_key: str, api_url: str,
                    default_input_max_characters: int = 1000,
                    default_output_max_tokens: int = 2048,
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

        # Use OpenAI client with custom base URL for Gemma endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )
        self.enums = OpenAIEnums  # Use OpenAI enums since it's compatible
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

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                       temperature: float = None):
        if not self.client:
            self.logger.error("Gemma client is not initialized.")
            return None
        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set.")
            return None
        
        self.logger.info(f"Using Gemma model: {self.generation_model_id}")
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_output_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        
        messages = list(chat_history)
        messages.append(self.construct_prompt(prompt, role=OpenAIEnums.USER.value))

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.generation_model_id,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=temperature
                )

                if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
                    self.logger.error("No response data received from Gemma.")
                    return None
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    self.logger.warning(f"Rate limit hit. Retrying in 10 seconds... (Attempt {attempt+1}/3)")
                    time.sleep(10)
                elif "timeout" in error_msg.lower():
                    self.logger.warning(f"Timeout error. Retrying... (Attempt {attempt+1}/3)")
                    time.sleep(5)
                else:
                    self.logger.error(f"generate_text failed: {e}")
                    return None
        
        return None
    
    def generate_json(self, prompt: str, chat_history: list = [],
                      max_output_tokens: int = 256) -> dict:
        """
        Generate JSON response. Since Gemma might not support response_format,
        we'll request JSON in the prompt and parse manually (like Cohere).
        """
        if not self.client or not self.generation_model_id:
            return {}

        json_prompt = f"{prompt}\n\nIMPORTANT: You must respond with ONLY a valid JSON object. No explanations, no additional text. Start with {{ and end with }}."

        messages = list(chat_history)
        messages.append(self.construct_prompt(json_prompt, role=OpenAIEnums.USER.value))

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.generation_model_id,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=0.0  # Use 0 for deterministic JSON responses
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to extract JSON from response
                try:
                    # First try direct parsing
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON block using regex
                    import re
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(0))
                        except json.JSONDecodeError:
                            pass
                    
                    # Fallback to heuristic parsing (like Cohere)
                    self.logger.warning(f"Gemma returned non-JSON text. Using heuristic parsing. Content: {content[:100]}")
                    content_lower = content.lower()
                    
                    # Heuristic parsing for hallucination check
                    if "grounded" in prompt.lower() or "hallucinat" in prompt.lower():
                        if "not grounded" in content_lower or "hallucinat" in content_lower or "contradict" in content_lower:
                            return {"score": "no", "reason": content[:200]}
                        else:
                            return {"score": "yes", "reason": content[:200]}
                    
                    # Heuristic parsing for relevance/answer check
                    if "not relevant" in content_lower or "does not answer" in content_lower or "does not resolve" in content_lower:
                        return {"score": "no"}
                    if "yes" in content_lower or "relevant" in content_lower or "resolves" in content_lower:
                        return {"score": "yes"}
                    return {"score": "no"}

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    self.logger.warning(f"Rate limit hit. Retrying in 10 seconds... (Attempt {attempt+1}/3)")
                    time.sleep(10)
                else:
                    self.logger.error(f"generate_json failed: {e}")
                    return {}
        
        return {}

    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
        """
        Embedding functionality. If Gemma endpoint doesn't support embeddings,
        this will fall back gracefully or you can use a different embedding provider.
        """
        if not self.client:
            self.logger.error("Gemma client is not initialized.")
            return None

        if isinstance(text, str):
            text = [text]

        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set.")
            return None
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model_id
            )

            if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
                self.logger.error("No embedding data received from Gemma.")
                return None
            
            return [f.embedding for f in response.data]
            
        except Exception as e:
            self.logger.error(f"Gemma embedding failed: {e}")
            self.logger.warning("Gemma endpoint might not support embeddings. Consider using a different embedding provider.")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt
        }