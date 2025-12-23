from pathlib import Path
from .prompt import build_codegen_prompt

def strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def save_code_to_file(code: str, filename: str = "extract_code.py") -> Path:
    path = Path(filename).absolute()
    path.write_text(code, encoding="utf-8")
    return path

def generate_code(llm, instruction: str, *, img_b64: str | None = None,
                  tool_desc: str = "", out_filename: str = "extract_code.py"):
    prompt = build_codegen_prompt(instruction, tool_desc=tool_desc, has_image=bool(img_b64))
    raw = llm.generate(prompt)
    code = strip_code_fences(raw)
    path = save_code_to_file(code, out_filename)
    return {"status": "success", "file": str(path)}
