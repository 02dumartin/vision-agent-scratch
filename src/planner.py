import json, re
from typing import Any, Dict, List, Optional
from src.media import b64_to_np
from src.config import Config 
from src.prompt import PROMPT_PLAN_TEMPLATE, PROMPT_FINAL_PLAN_TEMPLATE
from .types import AgentState
from src.display import print_code_plan

cfg = Config()

def render_prompt(
    user_request: str, 
    vqa_log: str, 
    vqa_struct: dict, 
    tool_desc: str, 
    observations: Optional[List] = None  # 매개변수로 추가
) -> str:
    vqa_struct_json = json.dumps(vqa_struct, ensure_ascii=False, indent=2)

    # observations 매개변수 사용
    if observations:
        obs_text = json.dumps(observations, ensure_ascii=False, indent=2)
    else:
        obs_text = "(none)"

    return PROMPT_PLAN_TEMPLATE.format(
        user_request=user_request,
        vqa_log=vqa_log,
        vqa_struct_json=vqa_struct_json,
        tool_desc=tool_desc,
        observations=obs_text,
    )


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def plan_once(
    user_request: str, 
    vqa_log: str, 
    vqa_struct: dict, 
    tool_desc: str, 
    img_b64: Optional[str],
    observations: Optional[List] = None  # 매개변수로 추가
):
    llm = cfg.create_planner()
    prompt_text = render_prompt(user_request, vqa_log, vqa_struct, tool_desc, observations)  # observations 전달
    media = [b64_to_np(img_b64)] if img_b64 else None
    raw = llm.generate(prompt_text, media=media)

    analysis_log = _extract_tag(raw, "analysis_log")
    plan_str = _extract_tag(raw, "plan_json")
    plan_json = json.loads(plan_str)
    return analysis_log, plan_json

def generate_final_plan(
    state: AgentState,
    prompt_template: str = PROMPT_FINAL_PLAN_TEMPLATE,
) -> Dict[str, Any]:
    llm = cfg.create_planner()
    prompt = prompt_template.format(
        user_request=state.user_request,
        vqa_log=state.vqa_log,
        vqa_struct_json=json.dumps(state.vqa_struct, ensure_ascii=False, indent=2),
        observations=json.dumps(state.observations, ensure_ascii=False, indent=2) if state.observations else "(none)",
        tool_desc=state.tool_desc,
    )
    media = [b64_to_np(state.img_b64)] if state.img_b64 else None
    raw = llm.generate(prompt, media=media)

    final_answer = _extract_tag(raw, "final_answer")
    code_plan_str = _extract_tag(raw, "code_plan")
    code_plan = json.loads(code_plan_str)
    
    # 코드 플랜 출력 추가
    if code_plan:
        print_code_plan(code_plan)
    
    return {"final_answer": final_answer, "code_plan": code_plan, "raw": raw}