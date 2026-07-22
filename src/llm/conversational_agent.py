"""LLM module.

Sole responsibility: hold a conversation with the user, grounded in what the
vision pipeline currently sees. Knows nothing about audio hardware,
transcription, or speech synthesis -- it only deals in text plus the plain
ObjectDescription records already produced by the description module.

Also owns making sure its own backend (a local Ollama server) is actually
up, since "the LLM isn't reachable" is squarely this module's problem, not
something the rest of the pipeline should need to know about.
"""

from __future__ import annotations

import shutil
import subprocess
import time
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

_STARTUP_TIMEOUT_SECONDS = 20
_POLL_INTERVAL_SECONDS = 0.5


class OllamaUnavailableError(RuntimeError):
    """Ollama couldn't be reached, even after trying to start it."""


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
        self._ensure_ollama_running()

    @property
    def history(self) -> List[ChatMessage]:
        return list(self._chat_log)

    def ask(self, user_message: str, visible_objects: List[ObjectDescription]) -> str:
        self._history.append({"role": "user", "content": self._with_scene_context(user_message, visible_objects)})

        try:
            response = self._client.chat(model=self._model, messages=self._history)
        except ConnectionError:
            # Ollama may have been closed or crashed after startup -- try
            # once to bring it back before giving up on this turn.
            self._ensure_ollama_running()
            response = self._client.chat(model=self._model, messages=self._history)

        reply = response["message"]["content"].strip()

        self._history.append({"role": "assistant", "content": reply})
        self._chat_log.append(ChatMessage("you", user_message))
        self._chat_log.append(ChatMessage("assistant", reply))
        return reply

    def _ensure_ollama_running(self) -> None:
        if self._is_reachable():
            return

        print("[llm] Ollama isn't responding -- trying to start it...")
        self._start_ollama()

        deadline = time.time() + _STARTUP_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self._is_reachable():
                print("[llm] Ollama is up.")
                return
            time.sleep(_POLL_INTERVAL_SECONDS)

        raise OllamaUnavailableError(
            "Could not reach Ollama, even after trying to start it. "
            "Install/start it manually: https://ollama.com/download"
        )

    def _is_reachable(self) -> bool:
        try:
            self._client.list()
            return True
        except ConnectionError:
            return False

    def _start_ollama(self) -> None:
        """Launch the local Ollama server in the background. Windows-only
        (CREATE_NO_WINDOW); there's no cross-platform requirement here."""
        ollama_exe = shutil.which("ollama")
        if ollama_exe is None:
            raise OllamaUnavailableError(
                "Ollama doesn't appear to be installed (no 'ollama' on PATH). "
                "Download it from https://ollama.com/download"
            )
        subprocess.Popen(
            [ollama_exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def _with_scene_context(self, user_message: str, visible_objects: List[ObjectDescription]) -> str:
        if not visible_objects:
            scene = "Currently visible: nothing recognizable."
        else:
            items = "; ".join(
                f"{obj.label} ({obj.size_label}, {obj.color_name})" for obj in visible_objects
            )
            scene = f"Currently visible: {items}."
        return f"[{scene}]\n{user_message}"
