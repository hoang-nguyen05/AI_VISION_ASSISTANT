import sys
from typing import Any, List, Optional
import torch

# Prevent torchao version check conflict in PEFT across Colab and Linux
try:
    import torchao
    from packaging import version
    if version.parse(getattr(torchao, "__version__", "0.0.0")) < version.parse("0.16.0"):
        sys.modules["torchao"] = None
except Exception:
    sys.modules["torchao"] = None

from peft import LoraConfig, get_peft_model, TaskType

def get_default_target_modules(model_type: str = "smolvlm") -> List[str]:
    """Returns default LoRA target modules for the given model architecture."""
    if "smolvlm" in model_type.lower():
        # Idefics3 / SmolVLM architecture (Llama-like text backbone)
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "qwen" in model_type.lower():
        # Qwen2-VL architecture
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        return ["q_proj", "v_proj"]

def freeze_vision_modules(model: Any) -> int:
    """
    Freezes all parameters in vision tower / encoder.
    Returns number of frozen parameters.
    """
    frozen_count = 0
    vision_keywords = ["vision_model", "visual", "vision_tower", "vision_encoder"]
    
    for name, param in model.named_parameters():
        if any(vk in name.lower() for vk in vision_keywords):
            param.requires_grad = False
            frozen_count += param.numel()
            
    print(f"Đã đóng băng (freeze) {frozen_count:,} tham số của Vision Encoder.")
    return frozen_count

def setup_lora_model(
    model: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    freeze_vision: bool = True
) -> Any:
    """
    Applies PEFT LoRA adapter to model.
    """
    if freeze_vision:
        freeze_vision_modules(model)
        
    if target_modules is None:
        model_name = getattr(model.config, "model_type", "smolvlm")
        target_modules = get_default_target_modules(model_name)
        
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
