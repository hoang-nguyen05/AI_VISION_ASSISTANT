import math
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image, ImageOps

def fix_exif_orientation(image: Image.Image) -> Image.Image:
    """Corrects image orientation based on EXIF metadata."""
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image

def resize_image_budget(
    image: Image.Image,
    max_pixels: int = 524288,  # ~724x724
    min_pixels: int = 3136     # ~56x56
) -> Image.Image:
    """
    Resizes image maintaining aspect ratio so total pixels <= max_pixels.
    Prevents out-of-memory errors on GPU during visual encoding.
    """
    width, height = image.size
    total_pixels = width * height
    
    if total_pixels > max_pixels:
        scale = math.sqrt(max_pixels / float(total_pixels))
        new_width = max(int(round(width * scale)), 28)
        new_height = max(int(round(height * scale)), 28)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    elif total_pixels < min_pixels:
        scale = math.sqrt(min_pixels / float(total_pixels))
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        
    return image

def validate_image(image_input: Union[str, Path, Image.Image]) -> Tuple[bool, Optional[str]]:
    """
    Validates an image file or PIL Image object.
    Returns (is_valid, error_message).
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            return False, f"Tập tin ảnh không tồn tại: {path}"
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            return False, f"Ảnh bị hỏng hoặc không thể đọc: {e}"
            
    elif isinstance(image_input, Image.Image):
        if image_input.size[0] <= 0 or image_input.size[1] <= 0:
            return False, "Kích thước ảnh không hợp lệ (chiều dài/rộng <= 0)"
    else:
        return False, "Định dạng đầu vào không hợp lệ (yêu cầu file path hoặc PIL Image)"
        
    return True, None

def load_and_preprocess_image(
    image_input: Union[str, Path, Image.Image],
    max_pixels: int = 524288,
    min_pixels: int = 3136
) -> Image.Image:
    """
    Loads, corrects orientation, ensures RGB format, and scales image within pixel budget.
    """
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input)
    else:
        image = image_input

    # Correct EXIF orientation
    image = fix_exif_orientation(image)
    
    # Convert RGBA/Palette/Grayscale to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize within budget
    image = resize_image_budget(image, max_pixels=max_pixels, min_pixels=min_pixels)
    
    return image
