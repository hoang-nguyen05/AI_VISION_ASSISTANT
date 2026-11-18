import os
from pathlib import Path
from typing import List
import gradio as gr
from src.config import PROJECT_ROOT

def get_available_adapters() -> List[str]:
    """Finds all available LoRA adapter directories inside adapters/."""
    adapters_dir = PROJECT_ROOT / "adapters"
    if not adapters_dir.exists():
        return []
    res = [d.name for d in adapters_dir.iterdir() if d.is_dir()]
    return res if res else ["smolvlm_vietnamese_lora"]

def create_model_selector():
    """Creates model and adapter selection widgets."""
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["smolvlm", "qwen"],
            value="smolvlm",
            label="Chọn Mô hình Nền (Base VLM)",
            info="SmolVLM-500M tối ưu cho GPU 4GB; Qwen2-VL-2B mạnh mẽ hơn."
        )
        use_adapter_checkbox = gr.Checkbox(
            value=False,
            label="Kích hoạt LoRA Adapter",
            info="Áp dụng trọng số đã fine-tune tiếng Việt"
        )
        available_adapters = get_available_adapters()
        adapter_dropdown = gr.Dropdown(
            choices=available_adapters,
            value=available_adapters[0] if available_adapters else "",
            label="Chọn Adapter",
            interactive=True
        )
    return model_dropdown, use_adapter_checkbox, adapter_dropdown
