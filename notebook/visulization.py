from pathlib import Path

def visualize_bbox(
    image_path: str,
    bbox: list,
    out_path: str = "semi_ripe_visualization.png",
    label: str = "semi-ripe tomato",
    conf: float = 1.0,
):
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    x1, y1, x2, y2 = map(int, bbox)

    # bbox 그리기
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # 라벨 텍스트
    text = f"{label} ({conf:.3f})"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

    # 텍스트 배경 박스 (이미지 밖으로 나가지 않게)
    tx = x1
    ty = max(y1 - 10, th + 10)
    cv2.rectangle(img, (tx, ty - th - baseline), (tx + tw + 10, ty + baseline), (0, 255, 0), -1)
    cv2.putText(img, text, (tx + 5, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # 저장
    cv2.imwrite(out_path, img)
    print(f"[Saved] {out_path}")
    return out_path


if __name__ == "__main__":
    # 네 결과를 그대로 넣으면 됨
    visualize_bbox(
        image_path="data/tomato_farm.jpg",
        bbox=[486, 1812, 666, 1952],
        out_path="semi_ripe_visualization.png",
        label="semi-ripe tomato",
        conf=1.0,
    )
