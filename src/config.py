# src/vision_agent/config.py
from typing import Type
from pydantic import BaseModel, Field
from src.llm import LMM, AnthropicLMM, OpenAILMM

class Config(BaseModel):
    vqa: Type[LMM] = Field(default=OpenAILMM)
    vqa_kwargs: dict = Field(default_factory=lambda: {
        "model_name": "gpt-4o-mini", "temperature": 0.0, "image_size": 768,
    })

    planner: Type[LMM] = Field(default=AnthropicLMM)
    planner_kwargs: dict = Field(default_factory=lambda: {
        "model_name": "claude-sonnet-4-5-20250929", "temperature": 0.0, "image_size": 768,
    })

    coder: Type[LMM] = Field(default=AnthropicLMM)
    coder_kwargs: dict = Field(default_factory=lambda: {
        "model_name": "claude-sonnet-4-5-20250929", "temperature": 0.0, "image_size": 768,
    })

    def create_vqa(self) -> LMM: return self.vqa(**self.vqa_kwargs)
    def create_planner(self) -> LMM: return self.planner(**self.planner_kwargs)
    def create_coder(self) -> LMM: return self.coder(**self.coder_kwargs)
