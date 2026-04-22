from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums,DocumentTypeEnums
import cohere
import logging
import json
import re
import time
from typing import List, Union
from threading import Lock
from collections import deque

class RateLimiter:
    """Simple rate limiter using sliding window"""
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the time window
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            
            # If at limit, wait until oldest request expires
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now + 0.1  # +0.1 for safety
                if sleep_time > 0:
                    logging.getLogger(__name__).info(f"[RateLimiter] Waiting {sleep_time:.1f}s to respect rate limit...")
                    time.sleep(sleep_time)
                    # Clean up again after waiting
                    now = time.time()
                    while self.requests and self.requests[0] < now - self.time_window:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now)

class CoHereProvider(LLMInterface):
    def __init__(self, api_key: str,
                    backup_api_key: str = None,  # NEW: Backup API key
                    backup_api_key2: str = None,
                    backup_api_key3: str = None,
                    default_input_max_characters: int = 1000,
                    default_output_max_tokens: int = 2048 ,
                    default_temperature: float = 0.1
                     ):
        self.api_key = api_key
        self.backup_api_key = backup_api_key
        self.backup_api_key2 = backup_api_key2
        self.backup_api_key3 = backup_api_key3
        self.current_api_key = api_key  # Track which key is active

        self.using_backup = False
        self.using_backup2 = False
        self.using_backup3 = False

        self.default_input_max_characters = default_input_max_characters
        self.default_output_max_tokens = default_output_max_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(api_key=self.api_key)
        self.enums = CoHereEnums
        self.logger = logging.getLogger(__name__)
        
        # Rate limiters for Cohere Trial API limits
        # Chat API: 20 requests/minute = 60 seconds / 20 = 3 seconds between requests
        self.chat_rate_limiter = RateLimiter(max_requests=20, time_window=60.0)
        # Embed API: 2000 inputs/minute (very generous, unlikely to hit)
        self.embed_rate_limiter = RateLimiter(max_requests=2000, time_window=60.0)
        self.logger.info("[CoHereProvider] Rate limiters initialized: Chat(20/min), Embed(2000/min)")
    
    def _switch_to_backup(self):
        """Switch to backup API key when primary fails"""
        if self.backup_api_key and not self.using_backup:
            self.logger.warning("⚠️  PRIMARY API KEY EXHAUSTED - Switching to BACKUP API key...")
            self.current_api_key = self.backup_api_key
            self.client = cohere.Client(api_key=self.backup_api_key)
            self.using_backup = True
            self.logger.info("✅ Successfully switched to BACKUP API key")
            return True
        
        if self.backup_api_key2 and not self.using_backup2:
            self.logger.warning("⚠️  PRIMARY API KEY EXHAUSTED - Switching to BACKUP API key2...")
            self.current_api_key = self.backup_api_key2
            self.client = cohere.Client(api_key=self.backup_api_key2)
            self.using_backup2 = True
            self.logger.info("✅ Successfully switched to BACKUP API key2")
            return True
        
        if self.backup_api_key3 and not self.using_backup3:
            self.logger.warning("⚠️  PRIMARY API KEY EXHAUSTED - Switching to BACKUP API key3...")
            self.current_api_key = self.backup_api_key3
            self.client = cohere.Client(api_key=self.backup_api_key3)
            self.using_backup3 = True
            self.logger.info("✅ Successfully switched to BACKUP API key3")
            return True

        return False
    
    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if error is a rate limit / quota exhausted error"""
        rate_limit_indicators = [
            "429", "TooManyRequests", "rate limit", "quota", 
            "limit exceeded", "too many requests"
        ]
        return any(indicator.lower() in error_msg.lower() for indicator in rate_limit_indicators)


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
            self.logger.error("Cohere client is not initialized.")
            return None
        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set.")
            return None
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_output_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        for attempt in range(3):
            try:
                # Wait if needed to respect rate limit
                self.chat_rate_limiter.wait_if_needed()
                
                response = self.client.chat(
                    model=self.generation_model_id,
                    chat_history=chat_history,
                    message= self.process_text(prompt, is_prompt=True),
                    temperature=temperature,
                    max_tokens=max_output_tokens
                )
                if not response or not response.text:
                    self.logger.error("No response data received from Cohere.")
                    return None
                return response.text
            except Exception as e:
                error_msg = str(e)
                if self._is_rate_limit_error(error_msg):
                    # Try to switch to backup key
                    if self._switch_to_backup():
                        self.logger.info("Retrying with backup API key...")
                        continue  # Retry immediately with backup key
                    else:
                        # No backup available, wait and retry
                        self.logger.warning(f"Rate limit hit (429). Retrying in 20 seconds... (Attempt {attempt+1}/3)")
                        time.sleep(20)
                else:
                    self.logger.error(f"generate_text failed: {e}")
                    return None
        return None
    
    # ============ NEW: for graders ============
    def generate_json(self, prompt: str, chat_history: list = [],
                      max_output_tokens: int = 256) -> dict:
        """
        Cohere does not have response_format — we will request JSON in the prompt
        and parse it manually
        """
        if not self.client or not self.generation_model_id:
            return {}

        json_prompt = f"{prompt}\n\nCRITICAL INSTRUCTION: You must return ONLY a raw JSON object. Do not add any conversational text, explanation, or introductions. Just start with {{ and end with }}."

        for attempt in range(3):
            try:
                # Wait if needed to respect rate limit
                self.chat_rate_limiter.wait_if_needed()
                
                response = self.client.chat(
                    model=self.generation_model_id,
                    chat_history=chat_history,
                    message=self.process_text(json_prompt, is_prompt=True),
                    temperature=0.0,
                    max_tokens=max_output_tokens
                )
                content = response.text.strip()
                
                # Try to extract json block using regex
                json_str = content
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Cohere returned non-JSON text. Falling back to heuristic parsing. Content snippet: {content[:100]}")
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
                if self._is_rate_limit_error(error_msg):
                    # Try to switch to backup key
                    if self._switch_to_backup():
                        self.logger.info("Retrying with backup API key...")
                        continue  # Retry immediately with backup key
                    else:
                        # No backup available, wait and retry
                        self.logger.warning(f"Rate limit hit (429). Retrying in 20 seconds... (Attempt {attempt+1}/3)")
                        time.sleep(20)
                else:
                    self.logger.error(f"generate_json failed: {e}")
                    return {}
        
        return {}
        

    def embed_text(self, text: Union[str, List[str]], document_type: str=None):
        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return None
        
        if isinstance(text, str):
            text = [text]

        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set.")
            return None

        input_type = CoHereEnums.DOCUMENT
        if document_type== DocumentTypeEnums.QUERY:
            input_type = CoHereEnums.QUERY
        
        for attempt in range(3):
            try:
                # Wait if needed to respect rate limit
                self.embed_rate_limiter.wait_if_needed()
                
                response = self.client.embed(
                    texts=[self.process_text(t) for t in text],
                    model=self.embedding_model_id,
                    input_type=input_type,
                    embedding_types=['float']
                )

                if not response or not response.embeddings or not response.embeddings.float:
                    self.logger.error("No embedding data received from Cohere.")
                    return None
                return [f for f in response.embeddings.float]
            
            except Exception as e:
                error_msg = str(e)
                if self._is_rate_limit_error(error_msg):
                    # Try to switch to backup key
                    if self._switch_to_backup():
                        self.logger.info("Retrying with backup API key...")
                        continue  # Retry immediately with backup key
                    else:
                        # No backup available, wait and retry
                        self.logger.warning(f"Rate limit hit (429). Retrying in 20 seconds... (Attempt {attempt+1}/3)")
                        time.sleep(20)
                else:
                    self.logger.error(f"embed_text failed: {e}")
                    return None
        
        return None

    def construct_prompt(self, prompt: str , role: str):
        return{
            "role": role,
            "text": prompt
        }