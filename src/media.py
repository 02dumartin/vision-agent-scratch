# src/vision_agent/media.py
import base64, numpy as np
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from PIL import Image

def image_to_base64(image: Image.Image, resize: Optional[int] = None) -> str:
    if resize is not None:
        image.thumbnail((resize, resize))
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def encode_media(media: Union[str, Path, np.ndarray, Image.Image], resize: Optional[int] = None) -> str:
    if isinstance(media, np.ndarray):
        return image_to_base64(Image.fromarray(media), resize)
    if isinstance(media, Image.Image):
        return image_to_base64(media, resize)
    if isinstance(media, (str, Path)):
        path = Path(media)
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return image_to_base64(Image.open(path), resize)
    raise ValueError(f"Unsupported media type: {media}")

def b64_to_np(b64: str) -> np.ndarray:
    return np.array(Image.open(BytesIO(base64.b64decode(b64))))

def np_to_b64(arr: np.ndarray) -> str:
    return image_to_base64(Image.fromarray(arr))

def load_image(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)
