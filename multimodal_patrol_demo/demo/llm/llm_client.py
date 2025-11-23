import os
from abc import ABC, abstractmethod
from typing import Optional

import requests


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class DummyLLMClient(BaseLLMClient):
    """Fallback implementation used when no provider is configured."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return f"[模板输出] {user_prompt[:200]}"


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, config) -> None:
        self.config = config
        self.api_key: Optional[str] = os.environ.get(config.api_key_env, "")
        self.api_base = config.api_base.rstrip("/")
        self.model_name = config.model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            return f"[未配置API_KEY，使用模板] {user_prompt[:200]}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }

        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            return f"[调用LLM失败，使用模板] {user_prompt[:200]}，错误：{exc}"


def create_llm_client(config) -> BaseLLMClient:
    if not getattr(config, "enabled", False):
        return DummyLLMClient()
    if getattr(config, "provider", "") == "openai":
        return OpenAILLMClient(config)
    return DummyLLMClient()
