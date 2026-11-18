from PIL import Image
from src.utils.image_utils import (
    fix_exif_orientation,
    resize_image_budget,
    validate_image,
    load_and_preprocess_image
)

def test_validate_image_valid():
    img = Image.new("RGB", (100, 100), color="red")
    valid, err = validate_image(img)
    assert valid is True
    assert err is None

def test_validate_image_invalid_size():
    img = Image.new("RGB", (0, 0))
    valid, err = validate_image(img)
    assert valid is False

def test_resize_image_budget():
    # Large image > 524288 pixels (e.g. 1000x1000 = 1,000,000 pixels)
    large_img = Image.new("RGB", (1000, 1000), color="blue")
    resized = resize_image_budget(large_img, max_pixels=524288)
    total_pixels = resized.size[0] * resized.size[1]
    assert total_pixels <= 524288

def test_load_and_preprocess_image_rgba_to_rgb():
    rgba_img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
    processed = load_and_preprocess_image(rgba_img)
    assert processed.mode == "RGB"
