"""
LLM Client

Unified LLM calling interface, supporting OpenAI and Ollama
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from openai import OpenAI

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """LLM Response"""
    content: str
    model: str
    tokens_used: int = 0
    finish_reason: str = "stop"
    
    def __str__(self):
        return self.content


class LLMClient:
    """
    LLM Client
    
    Supports OpenAI and Ollama, provides a unified interface
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Initialize LLM client
        
        Args:
            provider: Provider ("openai" or "ollama")
            model: Model name
            api_key: API key
            base_url: API base URL
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
            
            if not self.api_key:
                raise ValueError("OpenAI API key not set")
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None
            )
            logger.info(f"Initialized OpenAI client: model={self.model}")

        elif provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            # Ollama uses OpenAI-compatible API
            self.client = OpenAI(
                api_key="ollama",  # Ollama doesn't need a real key
                base_url=f"{self.base_url}/v1"
            )
            logger.info(f"Initialized Ollama client: model={self.model}")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Send chat request
        
        Args:
            messages: Message list [{"role": "user", "content": "..."}]
            temperature: Temperature parameter (overrides default)
            max_tokens: Maximum number of tokens (overrides default)
            
        Returns:
            LLM response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason
            
            logger.info(f"LLM call successful: tokens={tokens_used}, reason={finish_reason}")
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text (simplified interface)
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            
        Returns:
            LLM response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature, max_tokens)

