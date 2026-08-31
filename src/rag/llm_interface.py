"""
llm_interface.py

Abstracts the LLM layer so the rest of the application doesn't care
which provider generates the actual text. Currently backed by Groq's
free, fast cloud API (OpenAI-compatible).
"""

import logging
import os
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMInterface(ABC):
    """Abstract base class all LLM providers must implement."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        raise NotImplementedError


class GroqLLM(LLMInterface):
    """Cloud LLM provider using Groq's OpenAI-compatible API (free tier, fast LPU inference)."""

    def __init__(self, model_name: str = "openai/gpt-oss-20b", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
        self.base_url = "https://api.groq.com/openai/v1"

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Groq API request failed: {e}") from e


def get_llm(provider: str = "groq", model_name: str = None) -> LLMInterface:
    """
    Factory function to get an LLM instance based on config.
    This is the single place that needs to change to add new providers.
    """
    if provider == "groq":
        return GroqLLM(model_name=model_name or "openai/gpt-oss-20b")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


if __name__ == "__main__":
    llm = get_llm(provider="groq")
    print("Sending test prompt to Groq...")
    result = llm.generate("Say hello in one short sentence.")
    print(f"\nResponse: {result}")
