"""Language-model providers for call analysis, tried in order: Gemini, Anthropic, then a local model.

Gemini is used when GEMINI_API_KEY is set, Anthropic when ANTHROPIC_API_KEY is set; the local model
(Ollama, CPU only) is always last, so analysis works with no keys at all. If a provider fails (quota,
network, refusal), the next one in the chain is tried.
Every provider takes a system prompt, the user text and a JSON schema, and returns the parsed JSON.
"""
import json
import logging
from typing import Protocol

import httpx

from app.config import Settings

logger = logging.getLogger("voicebridge")

LOCAL_TIMEOUT_S = 600.0  # a long call on CPU can take minutes
CLOUD_MAX_TOKENS = 16000


class LLM(Protocol):
    name: str

    async def json(self, system: str, user: str, schema: dict) -> dict: ...


class OllamaLLM:
    """Local open-weights model through Ollama (https://ollama.com), forced onto the CPU."""

    name = "ollama"

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._url = settings.ollama_url.rstrip("/") + "/api/chat"
        self._model = settings.llm_model
        self._options = {"temperature": 0, "num_ctx": settings.llm_num_ctx}
        if settings.llm_cpu_only:
            self._options["num_gpu"] = 0  # no layers on the GPU
        self._transport = transport

    async def json(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": self._model,
            "stream": False,
            "think": False,
            "format": schema,
            "options": self._options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        async with httpx.AsyncClient(transport=self._transport, timeout=LOCAL_TIMEOUT_S) as client:
            response = await client.post(self._url, json=body)
            response.raise_for_status()
            return json.loads(response.json()["message"]["content"])


class GeminiLLM:
    name = "gemini"

    def __init__(self, settings: Settings):
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    async def json(self, system: str, user: str, schema: dict) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        response = await self._client.aio.models.generate_content(model=self._model, contents=user, config=config)
        return json.loads(response.text)


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, settings: Settings):
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    async def json(self, system: str, user: str, schema: dict) -> dict:
        response = await self._client.beta.messages.create(
            model=self._model,
            max_tokens=CLOUD_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Anthropic declined the request")
        return json.loads(next(block.text for block in response.content if block.type == "text"))


class LlmChain:
    def __init__(self, providers: list[LLM]):
        self._providers = providers

    @property
    def names(self) -> list[str]:
        return [p.name for p in self._providers]

    async def json(self, system: str, user: str, schema: dict) -> tuple[dict, str]:
        """The first provider that succeeds: (parsed JSON, provider name)."""
        errors = []
        for provider in self._providers:
            try:
                return await provider.json(system, user, schema), provider.name
            except Exception as exc:
                logger.warning("LLM provider %s failed: %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
        raise RuntimeError("All LLM providers failed: " + "; ".join(errors))


def make_llm_chain(settings: Settings) -> LlmChain:
    providers: list[LLM] = []
    for key, factory in ((settings.gemini_api_key, GeminiLLM), (settings.anthropic_api_key, AnthropicLLM)):
        if not key:
            continue
        try:
            providers.append(factory(settings))
        except Exception as exc:  # e.g. the SDK is missing
            logger.warning("Cannot use %s for analysis: %s", factory.name, exc)
    providers.append(OllamaLLM(settings))
    return LlmChain(providers)
