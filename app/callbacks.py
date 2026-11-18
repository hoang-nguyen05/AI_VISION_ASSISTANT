import time
from typing import Dict, List, Optional, Tuple
from PIL import Image
from src.utils.image_utils import validate_image

# Global engine cache
CURRENT_ENGINE = None
CURRENT_MODEL_KEY: str = ""

def get_engine(model_name: str, adapter_path: Optional[str] = None):
    global CURRENT_ENGINE, CURRENT_MODEL_KEY
    key = f"{model_name}::{adapter_path or 'base'}"
    if CURRENT_ENGINE is None or CURRENT_MODEL_KEY != key:
        print(f"[App] Nạp mới engine cho: {key}")
        from src.models.inference import VisionAssistantEngine
        from src.models.factory import ModelFactory
        if CURRENT_ENGINE is not None:
            ModelFactory.cleanup_memory()
        CURRENT_ENGINE = VisionAssistantEngine(
            model_name=model_name,
            adapter_path=adapter_path
        )
        CURRENT_MODEL_KEY = key
    return CURRENT_ENGINE

def handle_vqa(
    image: Image.Image,
    question: str,
    model_choice: str,
    use_adapter: bool,
    adapter_name: str
) -> Tuple[str, str, str]:
    """Handles VQA submission from Gradio."""
    if image is None:
        return "⚠️ Vui lòng tải ảnh lên trước khi đặt câu hỏi.", "0.0s", "0.0 MB"
    if not question or not question.strip():
        return "⚠️ Vui lòng nhập câu hỏi về bức ảnh.", "0.0s", "0.0 MB"

    adapter_path = f"adapters/{adapter_name}" if use_adapter and adapter_name else None
    try:
        engine = get_engine(model_name=model_choice, adapter_path=adapter_path)
        res = engine.answer_question(image, question)
        latency_str = f"{res['latency_sec']}s"
        vram_str = f"{res['peak_vram_mb']} MB"
        return res["text"], latency_str, vram_str
    except Exception as e:
        return f"❌ Lỗi suy luận: {e}", "0.0s", "0.0 MB"

def handle_caption(
    image: Image.Image,
    mode: str,
    model_choice: str,
    use_adapter: bool,
    adapter_name: str
) -> Tuple[str, str, str]:
    """Handles Caption generation submission."""
    if image is None:
        return "⚠️ Vui lòng tải ảnh lên trước.", "0.0s", "0.0 MB"

    adapter_path = f"adapters/{adapter_name}" if use_adapter and adapter_name else None
    try:
        engine = get_engine(model_name=model_choice, adapter_path=adapter_path)
        detailed = (mode == "Chi tiết")
        res = engine.generate_caption(image, detailed=detailed)
        latency_str = f"{res['latency_sec']}s"
        vram_str = f"{res['peak_vram_mb']} MB"
        return res["text"], latency_str, vram_str
    except Exception as e:
        return f"❌ Lỗi sinh mô tả: {e}", "0.0s", "0.0 MB"

def handle_compare(
    image: Image.Image,
    question: str,
    model_choice: str,
    adapter_name: str
) -> Tuple[str, str, str, str]:
    """
    Handles side-by-side comparison between Base Model and Fine-tuned Model.
    Runs sequentially to avoid GPU OOM on 4GB VRAM!
    """
    if image is None:
        return "Vui lòng tải ảnh", "", "Vui lòng tải ảnh", ""

    # 1. Base Model
    try:
        base_engine = get_engine(model_name=model_choice, adapter_path=None)
        q = question if question.strip() else "Hãy mô tả bức ảnh này bằng tiếng Việt."
        res_base = base_engine.answer_question(image, q)
        base_text = res_base["text"]
        base_meta = f"Độ trễ: {res_base['latency_sec']}s | VRAM: {res_base['peak_vram_mb']}MB"
    except Exception as e:
        base_text = f"Lỗi Base: {e}"
        base_meta = "0.0s"

    # 2. Fine-tuned Model
    adapter_path = f"adapters/{adapter_name}" if adapter_name else None
    try:
        ft_engine = get_engine(model_name=model_choice, adapter_path=adapter_path)
        res_ft = ft_engine.answer_question(image, q)
        ft_text = res_ft["text"]
        ft_meta = f"Độ trễ: {res_ft['latency_sec']}s | VRAM: {res_ft['peak_vram_mb']}MB"
    except Exception as e:
        ft_text = f"Lỗi LoRA: {e}"
        ft_meta = "0.0s"

    return base_text, base_meta, ft_text, ft_meta
