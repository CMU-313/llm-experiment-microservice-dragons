from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol, Any, Tuple


class ChatMessage(Protocol):
	@property
	def content(self) -> str: ...


class ChatResponse(Protocol):
	@property
	def message(self) -> ChatMessage: ...


class OllamaLikeClient(Protocol):
	def chat(self, model: str, messages: list[dict[str, str]]) -> ChatResponse: ...


def _default_model_name() -> str:
	return os.getenv("MODEL_NAME", "mistral:7b")


@dataclass
class LLMService:
	"""Thin wrapper around an Ollama-like chat client to enable mocking in tests."""
	client: OllamaLikeClient
	model_name: str = _default_model_name()

	def get_translation(self, post: str, context: str) -> str:
		response = self.client.chat(
			model=self.model_name,
			messages=[
				{"role": "system", "content": context},
				{"role": "user", "content": post},
			],
		)
		return response.message.content.strip()

	def get_language(self, post: str, context: str) -> str:
		response = self.client.chat(
			model=self.model_name,
			messages=[
				{"role": "system", "content": context},
				{"role": "user", "content": post},
			],
		)
		result = response.message.content.strip()
		# Normalize common prefixes the model may include
		for prefix in ("OUTPUT:", "Language:", "The language is"):
			if result.lower().startswith(prefix.lower()):
				result = result[len(prefix):].strip()
		if "\n" in result:
			result = result.split("\n", 1)[0].strip()
		return result or "Unknown"

	def query_llm(self, post: str, translation_context: str, classification_context: str) -> Tuple[bool, str]:
		if not post or not post.strip():
			return (False, post or "")
		lang = self.get_language(post, classification_context)
		is_english = lang.strip().lower() == "english"
		if is_english:
			return (True, post)
		translation = self.get_translation(post, translation_context)
		return (False, translation)

	def query_llm_robust(self, post: str, translation_context: str, classification_context: str) -> Tuple[bool, str]:
		if not post or not post.strip():
			return (False, post or "")

		max_post_length = 10000
		if len(post) > max_post_length:
			return (False, post[:max_post_length] + "...")

		is_english = False
		try:
			lang = self.get_language(post, classification_context)
			if not lang or lang.strip() == "":
				lang = "Unknown"
			if "error" in lang.lower() or "unknown" in lang.lower():
				is_english = False
			else:
				is_english = lang.strip().lower() == "english"
		except Exception:
			is_english = False

		if is_english:
			return (True, post)

		try:
			translation = self.get_translation(post, translation_context)
			if not translation or translation.strip() == "":
				return (False, post)
			if translation.lower().startswith("error") or translation.lower().startswith("i cannot"):
				return (False, post)
			return (False, translation)
		except Exception:
			return (False, post)


