import json
import time
from pathlib import Path
from typing import Dict, List
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

class MetricsTrackerCallback(TrainerCallback):
    """
    Custom callback to log training loss, validation loss, learning rate,
    and GPU VRAM consumption to JSON/CSV for experiment reporting.
    """

    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict] = []
        self.step_start_time = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start_time = time.time()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict = None, **kwargs):
        if not logs:
            return

        step_duration = round(time.time() - self.step_start_time, 3) if self.step_start_time else 0.0
        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0.0, 3),
            "loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "learning_rate": logs.get("learning_rate"),
            "step_duration_sec": step_duration,
            "peak_vram_mb": vram_mb
        }
        self.history.append(entry)

        # Write to JSON
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
