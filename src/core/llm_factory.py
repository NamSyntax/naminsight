import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMFactory:
    """Dynamic unified LLM factory from .env."""
    
    @staticmethod
    def create_llm(
        provider: str,
        model_name: str,
        temperature: float = 0.0,
        api_base: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> BaseChatModel:
        provider = provider.lower()
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_base=api_base,
                max_tokens=max_tokens
            )
        elif provider == "vllm":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_base=api_base or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
                openai_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
                max_tokens=max_tokens
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens or 4096
            )
        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=model_name,
                temperature=temperature
            )
        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=api_base or "http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def get_architect_llm(cls, temperature: float = 0.0) -> BaseChatModel:
        provider = os.getenv("ARCHITECT_PROVIDER", "openai")
        model = os.getenv("ARCHITECT_MODEL_NAME", os.getenv("ARCHITECT_MODEL", "gpt-4-turbo"))
        api_base = os.getenv("ARCHITECT_API_BASE", os.getenv("VLLM_API_BASE"))
        
        logger.info(f"Initialized Architect LLM: {provider} - {model}")
        return cls.create_llm(provider, model, temperature, api_base)

    @classmethod
    def get_dispatcher_llm(cls, temperature: float = 0.0) -> BaseChatModel:
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        
        if use_local:
            provider = os.getenv("LOCAL_PROVIDER", "ollama")
            model = os.getenv("LOCAL_MODEL", "llama3")
            api_base = os.getenv("LOCAL_API_BASE", "http://localhost:11434")
            logger.info("Local LLM Override is ENABLED for Dispatcher")
        else:
            provider = os.getenv("DISPATCHER_PROVIDER", "vllm")
            model = os.getenv("DISPATCHER_MODEL_NAME", os.getenv("DISPATCHER_MODEL", "Qwen2.5-7B"))
            api_base = os.getenv("DISPATCHER_API_BASE", os.getenv("VLLM_API_BASE"))
            
        logger.info(f"Initialized Dispatcher LLM: {provider} - {model}")
        return cls.create_llm(provider, model, temperature, api_base)
