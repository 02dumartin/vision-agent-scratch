import argparse
import os
from dotenv import load_dotenv
from src.llm import AnthropicLMM
from src.pipeline import AgentState, run_agent, run_coder_after_final_plan

def main():
    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True, help="사용자 요청")
    p.add_argument("--image", help="이미지 base64 문자열 또는 파일 경로")
    p.add_argument("--out", default="extract_code.py")
    args = p.parse_args()

    img_b64 = None
    if args.image and not args.image.strip().startswith("data:"):
        import base64, pathlib
        img_b64 = base64.b64encode(pathlib.Path(args.image).read_bytes()).decode()
    elif args.image:
        img_b64 = args.image

    # API 키 확인
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        return
    
    llm = AnthropicLMM()  # AnthropicLLMClient() → AnthropicLMM()
    state = AgentState(user_request=args.request, img_b64=img_b64)
    state = run_agent(state, llm, tool_desc="", tool_registry={})
    coder_result = run_coder_after_final_plan(state, llm, out_filename=args.out)
    print("saved:", coder_result["file"])

if __name__ == "__main__":
    main()
