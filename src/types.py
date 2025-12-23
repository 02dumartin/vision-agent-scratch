from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class AgentState:
    user_request: str
    img_b64: Optional[str] = None
    vqa_struct: dict = field(default_factory=dict)
    vqa_log: str = ""
    tool_desc: str = ""
    observations: list = field(default_factory=list)
    code_plan: Optional[list] = None