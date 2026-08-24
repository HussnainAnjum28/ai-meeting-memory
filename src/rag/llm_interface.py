"""
llm_interface.py

Abstracts the LLM layer so the rest of the application doesn't care
whether answers come from a local model (Ollama) or an external API.
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


class OllamaLLM(LLMInterface):
    """Local LLM provider using Ollama HTTP API."""

    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "repeat_penalty": 1.3, "repeat_last_n": 64, "num_predict": 300},
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                "Could not connect to Ollama. Is it installed and running? "
                "Try 'ollama serve' or check https://ollama.com"
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError("Ollama request timed out. The model may be too slow for this machine.") from e
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}") from e


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


def get_llm(provider: str = "ollama", model_name: str = None) -> LLMInterface:
    """
    Factory function to get an LLM instance based on config.
    This is the single place that needs to change to add new providers.
    """
    if provider == "ollama":
        return OllamaLLM(model_name=model_name or "llama3.2")
    elif provider == "groq":
        return GroqLLM(model_name=model_name or "openai/gpt-oss-20b")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


if __name__ == "__main__":
    llm = get_llm(provider="ollama", model_name="llama3.2")
    print("Sending test prompt to Ollama...")
    result = llm.generate("Say hello in one short sentence.")
    print(f"\nResponse: {result}")


