"""
LLM Integration Module

Provides a unified LLM interface, supporting OpenAI, Ollama, etc.
"""
from .llm_client import LLMClient, LLMResponse

__all__ = ["LLMClient", "LLMResponse"]

