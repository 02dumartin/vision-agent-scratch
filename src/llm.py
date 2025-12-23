# src/vision_agent/lmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union, TypedDict, cast
from openai import OpenAI
import anthropic
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam

from src.media import encode_media  # 새 유틸

class Message(TypedDict, total=False):
    role: str
    content: str
    media: Sequence[Union[str, Path]]

ReturnType = str | Iterator[str | None]

class LMM(ABC):
    @abstractmethod
    def generate(self, prompt: str, media: Optional[Sequence[Union[str, Path]]] = None, **kwargs: Any) -> ReturnType: ...
    @abstractmethod
    def chat(self, chat: Sequence[Message], **kwargs: Any) -> ReturnType: ...
    def __call__(self, input: str | Sequence[Message], **kwargs: Any) -> ReturnType:
        return self.generate(input, **kwargs) if isinstance(input, str) else self.chat(input, **kwargs)

class OpenAILMM(LMM):
    def __init__(self, model_name="gpt-4o-mini", api_key=None, max_tokens=4096, json_mode=False, image_size=768, image_detail="low", **kwargs: Any):
        self.client = OpenAI() if not api_key else OpenAI(api_key=api_key)
        self.model_name = model_name
        self.image_size = image_size
        self.image_detail = image_detail
        if "max_tokens" not in kwargs and not (model_name.startswith("o1") or model_name.startswith("o3")):
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self.kwargs = kwargs

    def generate(self, prompt: str, media=None, **kwargs: Any):
        chat = [{"role": "user", "content": prompt}]
        if media:
            chat[0]["media"] = media
        return self.chat(chat, **kwargs)

    def chat(self, chat, **kwargs: Any):
        fixed = []
        for msg in chat:
            content = [{"type": "text", "text": msg["content"]}]
            if msg.get("media") and self.model_name != "o3-mini":
                for m in msg["media"]:
                    encoded = encode_media(cast(str, m), resize=kwargs.get("resize", self.image_size))
                    content.append({"type": "image_base64", "image_base64": encoded, "detail": kwargs.get("image_detail", self.image_detail)})
            fixed.append({"role": msg["role"], "content": content})
        tmp = self.kwargs | kwargs
        resp = self.client.chat.completions.create(model=self.model_name, messages=fixed, **tmp)
        if tmp.get("stream"):
            return (chunk.choices[0].delta.content for chunk in resp)
        return resp.choices[0].message.content

class AnthropicLMM(LMM):
    def __init__(self, api_key=None, model_name="claude-sonnet-4-5-20250929", max_tokens=4096, image_size=768, **kwargs: Any):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.image_size = image_size
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_tokens
        self.kwargs = kwargs

    def generate(self, prompt: str, media=None, **kwargs: Any):
        chat = [{"role": "user", "content": prompt}]
        if media:
            chat[0]["media"] = media
        return self.chat(chat, **kwargs)

    def chat(self, chat, **kwargs: Any):
        tmp = self.kwargs | kwargs
        thinking_enabled = tmp.get("thinking", {}).get("type") == "enabled"
        if thinking_enabled:
            tmp["temperature"] = 1.0

        msgs: list[MessageParam] = []
        for msg in chat:
            content: list[TextBlockParam | ImageBlockParam] = [TextBlockParam(type="text", text=cast(str, msg["content"]))]
            for m in msg.get("media", []) or []:
                encoded = encode_media(cast(str, m), resize=kwargs.get("resize", self.image_size))
                content.append(ImageBlockParam(type="image", source={"type": "base64", "media_type": "image/png", "data": encoded}))
            msgs.append({"role": msg["role"], "content": content})

        resp = self.client.messages.create(model=self.model_name, messages=msgs, **tmp)
        if thinking_enabled:
            return "".join(block.text for block in resp.content if hasattr(block, "text"))
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

AnthropicLLMClient = AnthropicLMM