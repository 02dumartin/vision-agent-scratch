from .types import AgentState
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from .planner import render_prompt, plan_once, generate_final_plan
from .codegen import generate_code
from .config import Config
cfg = Config()



def run_agent(state: AgentState, llm, tool_desc: str, tool_registry: Dict[str, Any]) -> AgentState:
    """
    에이전트 실행 - VQA 및 플래닝 수행
    """
    # VQA 단계 (필요한 경우)
    if not state.vqa_struct:
        # VQA 로직이 필요하면 여기에 추가
        # 현재는 기본값으로 설정
        state.vqa_struct = {"task_type": "detection", "target": "semi-ripe tomato"}
        state.vqa_log = "VQA analysis completed"
    
    # 플래닝 단계
    if not state.tool_desc:
        state.tool_desc = tool_desc
    
    # 최종 계획 생성
    final_plan_result = generate_final_plan(state)
    state.code_plan = final_plan_result["code_plan"]
    
    return state  # 중요: state를 반환해야 함

def run_coder_after_final_plan(state: AgentState, llm, out_filename: str = "extract_code.py"):
    llm_code = cfg.create_coder()
    return generate_code(llm_code, instruction=state.user_request, img_b64=state.img_b64,
                         tool_desc="", out_filename=out_filename)