import gc
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
from transformers import AutoProcessor
from peft import PeftModel
from ..config import PROJECT_ROOT

MODEL_ALIASES = {
    "smolvlm": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "qwen": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}

class ModelFactory:
    """
    Factory for loading Vision-Language Models, Processors, and LoRA adapters.
    Optimized for low VRAM consumption (4GB - 8GB GPUs).
    """

    @staticmethod
    def resolve_model_id(model_name_or_path: str) -> str:
        key = model_name_or_path.lower().strip()
        return MODEL_ALIASES.get(key, model_name_or_path)

    @classmethod
    def load_processor(cls, model_name_or_path: str) -> Any:
        model_id = cls.resolve_model_id(model_name_or_path)
        print(f"Đang nạp processor cho model: {model_id}...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        return processor

    @classmethod
    def load_model(
        cls,
        model_name_or_path: str,
        adapter_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        torch_dtype: str = "float16",
        load_in_4bit: bool = False,
        use_cpu_offload: bool = False
    ) -> Any:
        model_id = cls.resolve_model_id(model_name_or_path)
        print("=" * 60)
        print(f"Đang nạp mô hình thị giác: {model_id}")
        print(f"  Device: {device} | Dtype: {torch_dtype} | 4-bit: {load_in_4bit}")
        print("=" * 60)

        # Parse dtype
        if torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Device mapping
        if device == "auto":
            device_map = "auto" if torch.cuda.is_available() else "cpu"
        else:
            device_map = device

        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": True
        }

        # 4-bit quantization setup
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                print("  Đã kích hoạt BitsAndBytes 4-bit NF4 Quantization!")
            except Exception as e:
                print(f"  Cảnh báo: Không thể tải cấu hình 4-bit ({e}). Nạp theo dtype thông thường.")

        # Load Base Model using ImageTextToText or AutoModel fallback
        model = None
        for class_name in ["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"]:
            try:
                cls_obj = getattr(__import__("transformers", fromlist=[class_name]), class_name, None)
                if cls_obj is not None:
                    model = cls_obj.from_pretrained(model_id, **kwargs)
                    break
            except Exception:
                continue

        if model is None:
            raise RuntimeError(f"Không thể nạp mô hình {model_id} qua các AutoModel classes.")

        # Attach LoRA adapter if specified
        if adapter_path:
            p = Path(adapter_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                print(f"Đang gắn LoRA adapter từ: {p}")
                model = PeftModel.from_pretrained(model, str(p))
                print("Đã gắn LoRA adapter thành công!")
            else:
                print(f"Cảnh báo: Đường dẫn adapter không tồn tại: {p}. Sử dụng Base model.")

        model.eval()
        return model

    @staticmethod
    def cleanup_memory():
        """Cleans up GPU VRAM and Python garbage collector."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
