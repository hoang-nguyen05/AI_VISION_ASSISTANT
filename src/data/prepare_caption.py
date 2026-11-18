import os
import json
from pathlib import Path
from typing import Dict, List
from ..config import ConfigManager
from ..utils.text_utils import normalize_vietnamese_text

def process_caption_record(record_id: str, image_path: str, caption: str) -> Dict:
    """Creates a unified schema record for Image Captioning."""
    return {
        "id": f"cap_{record_id}",
        "image": str(image_path),
        "task": "caption",
        "question": "Hãy mô tả bức ảnh này bằng tiếng Việt.",
        "answer": normalize_vietnamese_text(caption)
    }

def convert_caption_to_jsonl(
    data_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Converts UIT-OpenViIC dataset records into caption_train.jsonl and caption_val.jsonl.
    Enforces image-level grouping.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Tiền xử lý Captioning và chuẩn hóa sang schema JSONL...")
    print("=" * 60)
    
    records = []
    ann_file = data_dir / "annotations.json"
    
    if ann_file.exists():
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Support MS-COCO style or custom style
            for item in data.get("annotations", data):
                records.append({
                    "id": item.get("id", len(records)),
                    "image": item.get("image", ""),
                    "caption": item.get("caption", "")
                })
    else:
        print(f"Chưa tìm thấy {ann_file}. Tạo dữ liệu mẫu caption chuẩn để kiểm thử pipeline...")
        sample_records = [
            ("001", "data/samples/sample_table.jpg", "Một chiếc bàn gỗ có đĩa hoa quả gồm ba quả táo đỏ và một cốc nước."),
            ("002", "data/samples/sample_street.jpg", "Một người đàn ông đang đạp xe trên con đường rợp bóng cây xanh vào ban ngày."),
            ("003", "data/samples/sample_cats.jpg", "Hai chú mèo dễ thương đang nằm sưởi nắng và thư giãn cạnh nhau.")
        ]
        for rid, img_p, cap in sample_records:
            records.append({
                "id": rid,
                "image": img_p,
                "caption": cap
            })
            
    # Group by image
    image_to_records = {}
    for r in records:
        img = r["image"]
        if img not in image_to_records:
            image_to_records[img] = []
        image_to_records[img].append(r)
        
    all_images = list(image_to_records.keys())
    import random
    random.seed(seed)
    random.shuffle(all_images)
    
    num_val = max(1, int(len(all_images) * val_ratio))
    val_images = set(all_images[:num_val])
    
    train_records = []
    val_records = []
    
    for img, recs in image_to_records.items():
        target_list = val_records if img in val_images else train_records
        for r in recs:
            formatted = process_caption_record(r["id"], r["image"], r["caption"])
            target_list.append(formatted)
            
    train_out = output_dir / "caption_train.jsonl"
    val_out = output_dir / "caption_val.jsonl"
    
    with open(train_out, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(val_out, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Đã lưu {len(train_records)} bản ghi Captioning vào: {train_out}")
    print(f"Đã lưu {len(val_records)} bản ghi Captioning vào: {val_out}")

if __name__ == "__main__":
    cfg = ConfigManager()
    convert_caption_to_jsonl(
        data_dir=cfg.raw_data_dir / "uit_openviic",
        output_dir=cfg.processed_data_dir
    )
