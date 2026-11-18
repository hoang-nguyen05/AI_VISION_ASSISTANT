import os
import argparse
from pathlib import Path
from datasets import load_dataset
from ..config import ConfigManager

def download_openvivqa(output_dir: Path):
    """
    Downloads OpenViVQA dataset from Hugging Face Hub (uitnlp/OpenViVQA-dataset).
    """
    print("=" * 60)
    print("Bắt đầu tải bộ dữ liệu OpenViVQA (uitnlp/OpenViVQA-dataset)...")
    print("=" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset("uitnlp/OpenViVQA-dataset", trust_remote_code=True)
        print(f"Đã tải thành công OpenViVQA. Các split: {list(dataset.keys())}")
        for split in dataset.keys():
            print(f"  Split '{split}': {len(dataset[split])} mẫu.")
            
        save_path = output_dir / "openvivqa_hf"
        dataset.save_to_disk(str(save_path))
        print(f"Đã lưu dataset OpenViVQA vào: {save_path}")
        return dataset
    except Exception as e:
        print(f"Lỗi khi tải OpenViVQA: {e}")
        print("Lưu ý: Bạn có thể tải thủ công file zip hoặc clone dataset từ HuggingFace vào data/raw/openvivqa/")
        return None

def download_datasets(args):
    cfg = ConfigManager()
    raw_dir = cfg.raw_data_dir
    
    if args.dataset in ["all", "vqa"]:
        download_openvivqa(raw_dir / "openvivqa")
        
    if args.dataset in ["all", "caption"]:
        print("\n" + "=" * 60)
        print("Thông tin bộ dữ liệu UIT-OpenViIC (Image Captioning tiếng Việt):")
        print("Nguồn: https://uit-together.github.io/datasets/UIT-OpenViIC/")
        print("Nếu cần tải tự động hoặc thủ công, hãy đặt file annotations và ảnh vào: data/raw/uit_openviic/")
        print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải các bộ dữ liệu VQA và Captioning tiếng Việt")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "vqa", "caption"],
                        help="Bộ dữ liệu cần tải (all, vqa, caption)")
    args = parser.parse_args()
    download_datasets(args)
