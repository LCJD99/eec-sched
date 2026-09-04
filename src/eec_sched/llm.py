"""Model adapters shared by diagnosis and Scheduler Candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OpenAICompatibleChatModel:
    """A minimal adapter for an OpenAI-compatible Chat Completions endpoint."""

    base_url: str
    token: str
    model: str
    timeout_seconds: float = 360.0

    def __post_init__(self) -> None:
        for name in ("base_url", "token", "model"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"model {name} must not be empty")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    def complete(
        self,
        *,
        system: str,
        user: Mapping[str, object],
        json_output: bool = False,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, sort_keys=True)},
            ],
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - endpoint is configured by the experiment owner.
            body = json.loads(response.read())
        try:
            content = body["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("model endpoint returned no completion content") from exc
        if not isinstance(content, str):
            raise ValueError("model endpoint returned non-text completion content")
        return content
