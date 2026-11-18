import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set
from ..config import ConfigManager
from ..utils.text_utils import normalize_vietnamese_text
from ..utils.image_utils import validate_image

def process_vqa_record(record_id: str, image_path: str, question: str, answer: str) -> Dict:
    """Creates a unified schema record for VQA."""
    return {
        "id": f"vqa_{record_id}",
        "image": str(image_path),
        "task": "vqa",
        "question": normalize_vietnamese_text(question),
        "answer": normalize_vietnamese_text(answer)
    }

def convert_openvivqa_to_jsonl(
    data_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Converts OpenViVQA dataset records into train.jsonl and val.jsonl.
    Enforces image-level grouping to completely avoid data leakage!
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Tiền xử lý OpenViVQA và chuẩn hóa sang schema JSONL...")
    print("=" * 60)
    
    records = []
    
    # Check for VLSP json annotations in data_dir
    vlsp_files = list(data_dir.glob("vlsp2023_*_data.json"))
    images_dir = data_dir / "images"
    
    if vlsp_files:
        print(f"Tìm thấy {len(vlsp_files)} file dữ liệu VLSP OpenViVQA trong {data_dir}...")
        for vf in vlsp_files:
            split_name = vf.stem.replace("vlsp2023_", "").replace("_data", "")
            with open(vf, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            images_map = content.get("images", {})
            annotations_map = content.get("annotations", {})
            print(f"  Đọc {vf.name}: {len(images_map)} ảnh, {len(annotations_map)} câu hỏi-đáp.")
            
            for ann_id, ann in annotations_map.items():
                img_key = str(ann.get("image_id", ""))
                img_name = images_map.get(img_key, f"{img_key}.jpg")
                
                # Check potential image locations
                img_path = images_dir / img_name
                if not img_path.exists():
                    img_path = images_dir / f"{split_name}-images" / img_name
                if not img_path.exists():
                    img_path = images_dir / "training-images" / img_name
                if not img_path.exists():
                    img_path = images_dir / "dev-images" / img_name
                
                records.append({
                    "id": f"{split_name}_{ann_id}",
                    "image": str(img_path) if img_path.exists() else str(data_dir / "images" / img_name),
                    "question": ann.get("question", ""),
                    "answer": ann.get("answer", ""),
                    "split_source": split_name
                })
    else:
        print(f"Chưa tìm thấy VLSP json trong {data_dir}. Tạo dữ liệu mẫu chuẩn (Sample) để kiểm thử...")
        sample_records = [
            ("001", "data/samples/sample_table.jpg", "Trên bàn có bao nhiêu quả táo và chúng màu gì?", "Có 3 quả táo màu đỏ trên bàn."),
            ("002", "data/samples/sample_table.jpg", "Có cốc nước nào trên bàn không?", "Có một cốc nước trong suốt ở góc phải bàn."),
            ("003", "data/samples/sample_street.jpg", "Người đi xe đạp đang mặc áo màu gì?", "Người đi xe đạp đang mặc áo màu xanh da trời."),
            ("004", "data/samples/sample_street.jpg", "Thời tiết trong ảnh như thế nào?", "Trời có vẻ quang đãng và có nắng nhẹ."),
            ("005", "data/samples/sample_cats.jpg", "Có bao nhiêu con mèo trong ảnh?", "Có 2 con mèo đang nằm cạnh nhau."),
            ("006", "data/samples/sample_cats.jpg", "Màu lông của con mèo bên trái là gì?", "Con mèo bên trái có bộ lông màu vàng cam sọc vằn.")
        ]
        for idx, (img_id, img_path, q, a) in enumerate(sample_records):
            records.append({
                "id": f"sample_{idx:04d}",
                "image": img_path,
                "question": q,
                "answer": a,
                "split_source": "train" if idx < 4 else "val"
            })

    print(f"Tổng số bản ghi VQA thu thập được: {len(records)}")
    
    # Group by image to prevent leakage
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
    train_images = set(all_images[num_val:])
    
    train_records = []
    val_records = []
    
    for img, recs in image_to_records.items():
        target_list = val_records if img in val_images else train_records
        for r in recs:
            formatted = process_vqa_record(r["id"], r["image"], r["question"], r["answer"])
            target_list.append(formatted)
            
    # Save to JSONL
    train_out = output_dir / "vqa_train.jsonl"
    val_out = output_dir / "vqa_val.jsonl"
    
    with open(train_out, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(val_out, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Đã lưu {len(train_records)} bản ghi VQA vào: {train_out}")
    print(f"Đã lưu {len(val_records)} bản ghi VQA vào: {val_out}")

if __name__ == "__main__":
    cfg = ConfigManager()
    convert_openvivqa_to_jsonl(
        data_dir=cfg.raw_data_dir / "openvivqa",
        output_dir=cfg.processed_data_dir
    )
