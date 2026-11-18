import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
from torch.utils.data import Dataset
from ..utils.image_utils import load_and_preprocess_image
from ..config import PROJECT_ROOT

class MultimodalVisionDataset(Dataset):
    """
    Unified PyTorch Dataset for Multi-task Multimodal SFT (VQA & Image Captioning).
    Supports loading from one or multiple JSONL files with optional task weighting.
    """
    
    def __init__(
        self,
        jsonl_paths: Union[str, Path, List[Union[str, Path]]],
        max_pixels: int = 262144, # 512x512
        min_pixels: int = 3136,
        task_sample_ratio: Optional[Dict[str, float]] = None,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        if isinstance(jsonl_paths, (str, Path)):
            jsonl_paths = [jsonl_paths]
            
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
        # Load all records
        all_records = []
        for p in jsonl_paths:
            path = Path(p)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_records.append(json.loads(line))
                            
        if not all_records:
            print(f"Cảnh báo: Không có bản ghi nào được tìm thấy trong các file: {jsonl_paths}")
            
        # Optional task-ratio rebalancing
        if task_sample_ratio and all_records:
            vqa_records = [r for r in all_records if r.get("task") == "vqa"]
            cap_records = [r for r in all_records if r.get("task") == "caption"]
            other_records = [r for r in all_records if r.get("task") not in ["vqa", "caption"]]
            
            # Rebalance if both exist
            vqa_w = task_sample_ratio.get("vqa", 0.7)
            cap_w = task_sample_ratio.get("caption", 0.3)
            
            n_target = min(len(all_records), max_samples) if max_samples else len(all_records)
            n_vqa = int(n_target * vqa_w)
            n_cap = int(n_target * cap_w)
            
            sampled_vqa = random.choices(vqa_records, k=n_vqa) if vqa_records else []
            sampled_cap = random.choices(cap_records, k=n_cap) if cap_records else []
            
            all_records = sampled_vqa + sampled_cap + other_records
            random.shuffle(all_records)
            
        if max_samples and len(all_records) > max_samples:
            all_records = all_records[:max_samples]
            
        self.records = all_records
        
    def __len__(self) -> int:
        return len(self.records)
        
    def __getitem__(self, idx: int) -> Dict:
        item = self.records[idx]
        image_path = item["image"]
        p = Path(image_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
            
        # Load image or generate fallback if not found
        if p.exists():
            image = load_and_preprocess_image(p, max_pixels=self.max_pixels, min_pixels=self.min_pixels)
        else:
            # Fallback blank image to prevent crashes during dry-runs
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            
        question = item.get("question", "Hãy mô tả bức ảnh.")
        answer = item.get("answer", "")
        
        # Standard multimodal chat message structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        
        return {
            "id": item.get("id", str(idx)),
            "image": image,
            "messages": messages,
            "task": item.get("task", "vqa")
        }
