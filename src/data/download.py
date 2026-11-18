import os
import zipfile
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from ..config import ConfigManager

REPO_ID = "uitnlp/OpenViVQA-dataset"

def download_openvivqa(output_dir: Path, split: str = "dev"):
    """
    Downloads OpenViVQA dataset files from Hugging Face Hub (uitnlp/OpenViVQA-dataset)
    using huggingface_hub directly, avoiding datasets library JSON parsing issues.
    """
    print("=" * 60)
    print(f"Bắt đầu tải bộ dữ liệu OpenViVQA ({split}) từ Hugging Face Hub...")
    print("=" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    json_file = f"vlsp2023_{split}_data.json"
    zip_file = f"{split}-images.zip"

    # 1. Download metadata json
    print(f"Đang tải metadata: {json_file}...")
    local_json = hf_hub_download(
        repo_id=REPO_ID,
        filename=json_file,
        repo_type="dataset",
        local_dir=str(output_dir)
    )
    print(f"Đã lưu metadata tại: {local_json}")

    # 2. Download and unzip images
    print(f"Đang tải tệp nén hình ảnh: {zip_file}...")
    local_zip = hf_hub_download(
        repo_id=REPO_ID,
        filename=zip_file,
        repo_type="dataset",
        local_dir=str(output_dir)
    )
    print(f"Đang giải nén {zip_file} vào {images_dir}...")
    with zipfile.ZipFile(local_zip, "r") as z:
        z.extractall(str(images_dir))
    print(f"Giải nén hoàn tất! Các ảnh đã sẵn sàng tại: {images_dir}")

def download_datasets(args):
    cfg = ConfigManager()
    raw_dir = cfg.raw_data_dir
    
    if args.dataset in ["all", "vqa"]:
        download_openvivqa(raw_dir / "openvivqa", split=args.split)
        
    if args.dataset in ["all", "caption"]:
        print("\n" + "=" * 60)
        print("Thông tin bộ dữ liệu UIT-OpenViIC (Image Captioning tiếng Việt):")
        print("Nguồn: https://uit-together.github.io/datasets/UIT-OpenViIC/")
        print("Nếu cần tải tự động hoặc thủ công, hãy đặt file annotations và ảnh vào: data/raw/uit_openviic/")
        print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải các bộ dữ liệu VQA và Captioning tiếng Việt")
    parser.add_argument("--dataset", type=str, default="vqa", choices=["all", "vqa", "caption"],
                        help="Bộ dữ liệu cần tải (all, vqa, caption)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "train", "test"],
                        help="Split dữ liệu cần tải (dev: 165MB, train: 1.34GB)")
    args = parser.parse_args()
    download_datasets(args)
