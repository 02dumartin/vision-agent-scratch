from typing import Any, Dict
from src.media import b64_to_np
from src.pipeline import AgentState 

def run_tool_call(tool_call: dict, tool_registry: dict):
    """
    Execute a single tool call safely.

    Input:
      - tool_call: {"tool": str, "parameters": dict, ...}
      - tool_registry: {tool_name: callable}

    Output:
      {
        "tool": str,
        "ok": bool,
        "result": Any | None,
        "error": str | None,
      }
    """
    tool_name = tool_call.get("tool")
    params = tool_call.get("parameters", {})

    # 1) tool 존재 여부 확인
    if tool_name not in tool_registry:
        return {
            "tool": tool_name,
            "ok": False,
            "result": None,
            "error": f"Unknown tool: {tool_name}",
        }

    fn = tool_registry[tool_name]

    # 2) 실행 + 예외 처리
    try:
        result = fn(**params)
        return {
            "tool": tool_name,
            "ok": True,
            "result": result,
            "error": None,
        }
    except Exception as e:
        return {
            "tool": tool_name,
            "ok": False,
            "result": None,
            "error": repr(e),
        }

def validate_plan(plan: dict, tools_meta: list[dict]) -> None:
    names = {t["name"] for t in tools_meta}
    mode = plan.get("mode")
    if mode not in ("tool_calls", "final"):
        raise ValueError(f"Invalid mode: {mode}")

    if mode == "tool_calls":
        for tc in plan.get("tool_calls", []):
            if tc.get("tool") not in names:
                raise ValueError(f"Unknown tool in plan: {tc.get('tool')}")

def execute_plan(
    state: AgentState,
    plan_json: Dict[str, Any],
    verbose: bool = False,
) -> AgentState:
    tool_calls = plan_json.get("tool_calls", []) or []

    for tc in tool_calls:
        params = tc.get("parameters", {}) or {}

        # placeholder 치환 (LLM이 "image"라고 써둔 경우)
        if params.get("image") == "image":
            params["image"] = b64_to_np(state.img_b64)

        if params.get("images") == ["image"]:
            params["images"] = [b64_to_np(state.img_b64)]

        # agentic_object_detection 전용 prompt 정리 (추가)
        tool_name = tc.get("tool")
        if tool_name in ["agentic_object_detection", "agentic_sam2_instance_segmentation"]:
            prompt = params.get("prompt")
            if isinstance(prompt, str):
                # 콤마 기준으로 첫 번째만 사용
                prompt = prompt.split(",")[0]
                # 마침표 제거
                prompt = prompt.replace(".", " ")
                # 공백 정리
                params["prompt"] = prompt.strip()

        # detected_image_crop 전용 보정
        if tool_name == "detected_image_crop":
            params["image_np"] = b64_to_np(state.img_b64)
            params["bbox_format"] = "xyxy_norm"
            if "full_image_path" in params:
                params["full_image_path"] = None

        tc["parameters"] = params

        exec_result = run_tool_call(tc, state.tool_registry, verbose=verbose)
        state.all_execs.append(exec_result)
        state.observations.append(exec_result)

    return state