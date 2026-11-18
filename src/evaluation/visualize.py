import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from ..config import PROJECT_ROOT

def plot_training_curves(history_file: str, output_image: str = "results/plots/loss_curve.png"):
    """Plots training and validation loss curves from JSON log history."""
    p = Path(history_file)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        print(f"Không tìm thấy file lịch sử training: {p}")
        return

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = [d["step"] for d in data if d.get("loss") is not None]
    losses = [d["loss"] for d in data if d.get("loss") is not None]

    if not steps or not losses:
        print("Không đủ dữ liệu loss để vẽ biểu đồ.")
        return

    plt.figure(figsize=(9, 5))
    plt.plot(steps, losses, label="Training Loss", color="#1f77b4", linewidth=2)
    plt.title("Tiến trình Loss Huấn luyện SFT LoRA", fontsize=14, fontweight="bold")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_p = Path(output_image)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ loss tại: {out_p}")

def plot_model_comparison(
    metrics_base: Dict[str, float],
    metrics_finetuned: Dict[str, float],
    output_image: str = "results/plots/model_comparison.png"
):
    """Generates comparison bar chart between Base Model and Fine-tuned Model."""
    keys = [k for k in metrics_base.keys() if k in metrics_finetuned and isinstance(metrics_base[k], (int, float))]
    if not keys:
        return

    base_vals = [metrics_base[k] for k in keys]
    ft_vals = [metrics_finetuned[k] for k in keys]

    x = range(len(keys))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], base_vals, width=width, label="Base Model (Pretrained)", color="#7f7f7f")
    plt.bar([i + width/2 for i in x], ft_vals, width=width, label="Fine-tuned (LoRA)", color="#2ca02c")

    plt.title("So sánh Hiệu năng Trước & Sau Fine-tuning", fontsize=14, fontweight="bold")
    plt.ylabel("Điểm số (%)", fontsize=12)
    plt.xticks(x, [k.replace("_", " ").title() for k in keys], fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_p = Path(output_image)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ so sánh mô hình tại: {out_p}")
