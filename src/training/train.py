import os
import argparse
from pathlib import Path
from transformers import Trainer, TrainingArguments

from ..config import load_yaml_config, PROJECT_ROOT
from ..models.factory import ModelFactory
from ..models.lora import setup_lora_model
from ..data.dataset import MultimodalVisionDataset
from ..data.collator import MultimodalDataCollator
from .callbacks import MetricsTrackerCallback
from .export import save_lora_checkpoint

def train(config_path: str, smoke_test: bool = False):
    config = load_yaml_config(config_path)
    print("=" * 60)
    print(f"KHỞI ĐỘNG HUẤN LUYỆN SFT LORA / QLORA")
    print(f"Config: {config_path} | Smoke Test: {smoke_test}")
    print("=" * 60)

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    model_name = model_cfg.get("name", "HuggingFaceTB/SmolVLM-500M-Instruct")
    load_in_4bit = model_cfg.get("load_in_4bit", False)
    torch_dtype = model_cfg.get("torch_dtype", "bfloat16")
    freeze_vision = model_cfg.get("freeze_vision", True)

    # 1. Load processor & base model
    processor = ModelFactory.load_processor(model_name)
    base_model = ModelFactory.load_model(
        model_name_or_path=model_name,
        device="auto",
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit
    )

    # 2. Setup LoRA
    r = lora_cfg.get("r", 16)
    alpha = lora_cfg.get("alpha", 32)
    dropout = lora_cfg.get("dropout", 0.05)
    target_modules = lora_cfg.get("target_modules", None)

    model = setup_lora_model(
        model=base_model,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        freeze_vision=freeze_vision
    )

    # 3. Load datasets
    train_file = data_cfg.get("train_file", "data/processed/train.jsonl")
    val_file = data_cfg.get("val_file", "data/processed/val.jsonl")
    max_pixels = data_cfg.get("max_pixels", 262144)
    max_samples = 8 if smoke_test else None

    # Fallback to vqa_train / caption_train if train.jsonl doesn't exist yet
    train_paths = [train_file]
    if not (PROJECT_ROOT / train_file).exists():
        train_paths = ["data/processed/vqa_train.jsonl", "data/processed/caption_train.jsonl"]

    val_paths = [val_file]
    if not (PROJECT_ROOT / val_file).exists():
        val_paths = ["data/processed/vqa_val.jsonl", "data/processed/caption_val.jsonl"]

    train_dataset = MultimodalVisionDataset(
        jsonl_paths=train_paths,
        max_pixels=max_pixels,
        max_samples=max_samples
    )
    val_dataset = MultimodalVisionDataset(
        jsonl_paths=val_paths,
        max_pixels=max_pixels,
        max_samples=4 if smoke_test else None
    )

    # 4. Data Collator
    data_collator = MultimodalDataCollator(
        processor=processor,
        max_length=data_cfg.get("max_length", 512)
    )

    # 5. Training Arguments
    out_dir = PROJECT_ROOT / train_cfg.get("output_dir", "checkpoints")
    max_steps = 4 if smoke_test else -1
    epochs = 1 if smoke_test else train_cfg.get("num_train_epochs", 3)
    grad_accum = 1 if smoke_test else train_cfg.get("gradient_accumulation_steps", 8)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_steps=1 if smoke_test else int(train_cfg.get("warmup_steps", 10)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        bf16=(torch_dtype == "bfloat16"),
        fp16=(torch_dtype == "float16"),
        logging_steps=1 if smoke_test else train_cfg.get("logging_steps", 10),
        eval_strategy="no" if smoke_test else train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy="no" if smoke_test else train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        remove_unused_columns=False,
        report_to="none"
    )

    # Callback for experiment tracking
    log_file = PROJECT_ROOT / "results" / "metrics" / "training_history.json"
    tracker = MetricsTrackerCallback(log_file)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if not smoke_test else None,
        data_collator=data_collator,
        callbacks=[tracker]
    )

    # Train
    print("Bắt đầu huấn luyện...")
    train_result = trainer.train()
    print("Huấn luyện hoàn tất!")

    # 6. Export Adapter
    adapter_dir = PROJECT_ROOT / train_cfg.get("adapter_dir", "adapters/best_lora")
    metadata = {
        "base_model": model_name,
        "config_path": config_path,
        "smoke_test": smoke_test,
        "train_loss": train_result.training_loss,
        "metrics": train_result.metrics
    }
    save_lora_checkpoint(model, adapter_dir, metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện LoRA / QLoRA cho VLM")
    parser.add_argument("--config", type=str, default="configs/smolvlm_lora.yaml",
                        help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Chạy thử nghiệm nhanh vài step kiểm tra luồng")
    args = parser.parse_args()
    train(config_path=args.config, smoke_test=args.smoke_test)
