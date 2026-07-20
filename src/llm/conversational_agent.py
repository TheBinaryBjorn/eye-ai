"""LLM module.

Sole responsibility: hold a conversation with the user, grounded in what the
vision pipeline currently sees. Knows nothing about audio hardware,
transcription, or speech synthesis -- it only deals in text plus the plain
ObjectDescription records already produced by the description module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import ollama

from src.description.object_describer import ObjectDescription

_SYSTEM_PROMPT = (
    "You are a helpful voice assistant with live access to a camera feed. "
    "Before each user message you are given a short list of objects "
    "currently visible to the camera. Use it to answer questions about what "
    "you see, but keep answers brief and conversational -- you are being "
    "spoken out loud through a speaker, not read as text."
)


@dataclass(frozen=True)
class ChatMessage:
    speaker: str  # "you" or "assistant"
    text: str


class ConversationalAgent:
    """Wraps a local Ollama model and keeps a running, vision-grounded chat history."""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._client = ollama.Client(host=host)
        # Full history sent to Ollama: system prompt + scene-context-prefixed turns.
        self._history: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        # Clean, display-friendly transcript: no system prompt, no scene-context prefix.
        self._chat_log: List[ChatMessage] = []

    @property
    def history(self) -> List[ChatMessage]:
        return list(self._chat_log)

    def ask(self, user_message: str, visible_objects: List[ObjectDescription]) -> str:
        self._history.append({"role": "user", "content": self._with_scene_context(user_message, visible_objects)})

        response = self._client.chat(model=self._model, messages=self._history)
        reply = response["message"]["content"].strip()

        self._history.append({"role": "assistant", "content": reply})
        self._chat_log.append(ChatMessage("you", user_message))
        self._chat_log.append(ChatMessage("assistant", reply))
        return reply

    def _with_scene_context(self, user_message: str, visible_objects: List[ObjectDescription]) -> str:
        if not visible_objects:
            scene = "Currently visible: nothing recognizable."
        else:
            items = "; ".join(
                f"{obj.label} ({obj.size_label}, {obj.color_name})" for obj in visible_objects
            )
            scene = f"Currently visible: {items}."
        return f"[{scene}]\n{user_message}"
