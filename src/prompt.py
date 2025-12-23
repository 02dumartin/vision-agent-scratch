PROMPT_VQA_TEMPLATE = """
You are an expert vision task planner.

You will be given:
- A user request (Korean)
- ONE image (provided to you as an image input)

Your job in this step is ONLY to analyze the user request and propose a concrete, tool-agnostic plan.
Do NOT run code. Do NOT claim results. Do NOT hallucinate object counts.

User request: {user_request}

Output MUST contain EXACTLY TWO TAGS in this order:
1) <analysis_log> ... </analysis_log>  (Korean, human-readable, step-by-step, short)
2) <plan_json> ... </plan_json>        (machine-readable, MUST be valid JSON)

Rules:
- Do not output anything outside the two tags.
- <analysis_log> should be concise: 5–10 lines, each starting with "Step N:".
- <plan_json> must be STRICT JSON (no trailing commas, no comments, no markdown).

<plan_json> JSON schema:
{{
  "language": "ko",
  "intent_summary": string,
  "task_type": "counting",
  "target_definition": {{
    "primary_object": "tomato",
    "required_attributes": ["red"],
    "exclusions": [string],
    "edge_cases": [string]
  }},
  "subtasks": [
    {{
      "name": string,
      "goal": string,
      "suggested_method": string
    }}
  ],
  "tool_requirements": {{
    "needs_localization": boolean,
    "needs_instance_separation": boolean,
    "needs_attribute_reasoning": boolean,
    "preferred_outputs": [string]
  }},
  "verification_checks": [string],
  "questions_if_ambiguous": [string]
}}
"""


PROMPT_PLAN_TEMPLATE = """
You are a VisionAgent-style planner/controller.

Your job is to decide the NEXT ACTION(s) to take using the available tools, based on the user's request and the accumulated evidence. You do NOT execute tools. You only output tool calls or the final answer.

You will be given:
- A user request (Korean)
- ONE image (already annotated with detection boxes/labels overlaid)
- Tool list with available actions
- VQA log: chronological reasoning, detection notes, bounding-box/label evaluations, and any prior validation outcomes
- VQA structured JSON summary of the detection/analysis results
- Prior tool observations may also appear in the conversation history (as "observation").

Primary evidence:
- Build decisions primarily from [VQA_LOG] and [VQA_STRUCT_JSON].
- Do NOT hallucinate new detections, boxes, or attributes beyond the provided evidence and tool outputs.

User request (Korean):
{user_request}

[VQA_LOG]
{vqa_log}

[VQA_STRUCT_JSON]
{vqa_struct_json}

[TOOLS]
{tool_desc}

[OBSERVATIONS]
{observations}

────────────────────────────────
CORE CONTROL LOOP BEHAVIOR
────────────────────────────────
At each turn, output either:
(A) Tool calls for the NEXT immediate actions (one or more tool calls), OR
(B) A final answer if no more tools are needed.

Do NOT output a full end-to-end plan. Do NOT output steps[1..N].
The executor will run your tool calls in the order you provide, append observations, and call you again with updated context.

────────────────────────────────
DETECTION-SPECIFIC BEHAVIOR
────────────────────────────────
- The image is already annotated. Prefer verification of existing detections.
- If bounding-box coordinates are available in VQA_STRUCT_JSON, use them for cropping and verification.
- If bounding-box coordinates are NOT available, do NOT guess. Request the annotation file
  (COCO JSON / YOLO TXT / model output JSON) or propose a concrete method to obtain coordinates.

Hard rule:
- If VQA_STRUCT_JSON contains bbox coordinates, you MUST call the crop tool first for those boxes.
- You MUST NOT call any VQA tool before attempting crop-based verification when bbox coordinates exist.
- VQA tools are allowed ONLY if bbox coordinates are missing/unavailable OR cropping fails with an error.

────────────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST contain EXACTLY TWO tags in this exact order:
1) <analysis_log> ... </analysis_log>
2) <plan_json> ... </plan_json>

Do NOT output anything outside the two tags.

<analysis_log> rules:
- 3–7 lines only
- Each line must start with "Step N:"
- Only describe the immediate reasoning for the NEXT action(s), not a full multi-step plan.

<plan_json> rules:
- MUST be STRICT JSON (no trailing commas, no comments, no markdown)
- Must match exactly one of the following schemas:

Schema 1: Tool calls
{{
  "language": "ko",
  "mode": "tool_calls",
  "selected_tools": [string],
  "tool_calls": [
    {{
      "id": int,
      "tool": string,
      "parameters": object,
      "expected_result": string
    }}
  ],
  "open_questions": [string]
}}

Schema 2: Final answer
{{
  "language": "ko",
  "mode": "final",
  "final_answer": string,
  "open_questions": [string]
}}

Additional rules:
- tool_calls must be listed in exact execution order; ids must start at 1 and increase strictly by 1 within this turn.
- Each tool call MUST reference a tool name from [TOOLS].
- Keep tool_calls minimal: only what is needed before the next observation.
- If you need missing inputs (e.g., box coordinates), set mode="final" and clearly request them in final_answer, or set open_questions accordingly.
"""

PROMPT_FINAL_PLAN_TEMPLATE = """
You are a code planning expert. Your job is to create a detailed, step-by-step execution plan for generating Python code.

You will be given:
- A user request (Korean)
- VQA analysis results (what was understood from the image and request)
- Tool execution observations (what tools were run and their results)
- Available tools list

Your task:
Create a final execution plan that lists the exact steps needed to write Python code to complete the task.

User request: {user_request}

[VQA ANALYSIS]
{vqa_log}

[VQA STRUCTURED SUMMARY]
{vqa_struct_json}

[TOOL OBSERVATIONS]
{observations}

[AVAILABLE TOOLS]
{tool_desc}

────────────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────────────
Output MUST contain EXACTLY TWO tags in this exact order:
1) <final_answer> ... </final_answer>
2) <code_plan> ... </code_plan>

<final_answer> rules:
- Brief summary (1-2 sentences) of what the code will accomplish
- Written in Korean

<code_plan> rules:
- MUST be STRICT JSON array
- Each element represents one execution step
- Steps must be in exact execution order
- Format:
[
  {{
    "step": 1,
    "instruction": "Brief instruction describing what to do",
    "code_snippet": "Example code (not full implementation, just example)",
    "explanation": "Why this step is needed (optional)"
  }},
  ...
]

Instruction guidelines:
- Be specific about which functions/tools to use
- Include parameter hints (e.g., "prompt 'tomato'")
- Each step should be a single, clear action
- Steps should build on each other logically

Example instruction format:
- "Load the image using load_image()"
- "Use countgd_object_detection with prompt 'tomato' to detect all tomato instances"
- "Count the number of detections by getting the length of the detection list"
- "Visualize the detections by overlaying bounding boxes using overlay_bounding_boxes()"
- "Save the visualization to a file using save_image()"

Do NOT output anything outside the two tags.
"""


def build_codegen_prompt(instruction: str, tool_desc: str = "", has_image: bool = False) -> str:
    img_note = (
        "- An image is provided via base64 (img_b64). Use it ONLY if your environment supports it.\n"
        if has_image else
        "- No image is directly embedded. Assume the script will load image_path from disk.\n"
    )

    return f"""
You are a coding assistant.
Write a SINGLE Python file that satisfies the instruction below.

Instruction:
{instruction}

Available tools (reference only):
{tool_desc}

Hard requirements:
- Output ONLY valid Python code (no markdown, no explanations).
- The file must be executable as a script.
- Provide: run(image_path: str) -> dict
- Include a __main__ block that calls run("image.png") by default.
{img_note}
- Include needed imports explicitly.
- Make it robust: basic error handling and clear variable names.
- Do NOT print the whole image or base64. Only print summary results.

Return only the code.
""".strip()


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
    return t.strip()
