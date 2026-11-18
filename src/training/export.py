import json
from pathlib import Path
from typing import Any, Dict
from ..config import PROJECT_ROOT

def save_lora_checkpoint(
    model: Any,
    output_dir: Path,
    metadata: Dict[str, Any]
):
    """
    Saves PEFT LoRA adapter weights, adapter_config.json, and metadata manifest.
    """
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Đang lưu LoRA adapter vào thư mục: {out_path}...")
    # Saves adapter_model.safetensors and adapter_config.json
    model.save_pretrained(str(out_path))

    # Save training metadata
    meta_file = out_path / "training_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Đã xuất LoRA adapter thành công! File metadata: {meta_file}")
