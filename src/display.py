from typing import List, Dict, Any

def format_code_plan_display(
    code_plan: List[Dict[str, Any]],
    width: int = 90,
    separator_token: str = "---",
) -> str:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.rule import Rule
        from rich import box
        from rich.console import Group
        from rich.table import Table

        console = Console(
            width=width,
            record=True,
            force_terminal=True,
            color_system=None,
        )

        # Vision Agent 스타일의 연속된 테두리를 위한 Table 사용
        table = Table(
            title="Plan",
            title_style="bold cyan",
            box=box.SQUARE,  # 연속된 사각형 테두리
            show_header=True,
            header_style="bold magenta",
            padding=(0, 1),
            width=width,
            collapse_padding=True,  # 패딩 축소로 더 깔끔한 모양
        )
        
        # 헤더 추가
        table.add_column("Instructions", style="cyan", overflow="fold", no_wrap=False)

        # 각 instruction을 테이블 행으로 추가
        for step in code_plan:
            inst = str(step.get("instruction", "")).strip()
            if not inst:
                continue

            # separator_token으로 분리된 부분들을 처리
            parts = [p.strip() for p in inst.split(separator_token)]
            
            # 각 부분을 별도 행으로 추가하되, 구분선은 텍스트로 처리
            for i, part in enumerate(parts):
                if part:
                    table.add_row(part)
                # 마지막이 아니면 구분선 추가
                if i < len(parts) - 1:
                    table.add_row("─" * (width - 4))  # 구분선

        # 테이블을 문자열로 렌더링
        with console.capture() as capture:
            console.print(table)
        
        return capture.get()

    except ImportError:
        # Rich가 없는 경우 fallback
        lines = ["Plan", "=" * width]
        lines.append("Instructions")
        lines.append("─" * width)
        
        for step in code_plan:
            inst = str(step.get("instruction", "")).strip()
            if inst:
                parts = [p.strip() for p in inst.split(separator_token)]
                for i, part in enumerate(parts):
                    if part:
                        lines.append(part)
                    if i < len(parts) - 1:
                        lines.append("─" * width)
        
        lines.append("─" * width)
        return "\n".join(lines)


def print_code_plan(
    code_plan: List[Dict[str, Any]],
    width: int = 100,
    separator_token: str = "---",
    truncate: bool = False,  # Vision Agent 스타일에서는 truncate하지 않음
):
    """
    Vision Agent 스타일의 연속된 테두리로 code plan 출력
    """
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console(width=width, force_terminal=True, color_system=None)

    # Vision Agent와 동일한 스타일의 테이블
    table = Table(
        title="Plan",
        title_style="bold cyan",
        box=box.SQUARE,  # 핵심: 연속된 사각형 테두리
        show_header=True,
        header_style="bold magenta", 
        padding=(0, 1),
        width=width,
        collapse_padding=True,
        show_lines=False,  # 행 간 구분선 제거로 연속성 유지
    )
    
    table.add_column("Instructions", style="cyan", overflow="fold", no_wrap=False)

    # 모든 instruction을 하나의 연속된 텍스트로 처리
    all_instructions = []
    for step in code_plan:
        inst = str(step.get("instruction", "")).strip()
        if inst:
            # separator_token 기준으로 분리하되, 구분선으로 연결
            parts = [p.strip() for p in inst.split(separator_token) if p.strip()]
            all_instructions.extend(parts)

    # 각 instruction을 별도 행으로 추가
    for instruction in all_instructions:
        if instruction:
            table.add_row(instruction)

    # 출력
    console.print(table)