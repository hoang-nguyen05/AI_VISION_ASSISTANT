# 👁️ AI VISION ASSISTANT
> **Hệ thống Mô tả Hình ảnh & Hỏi đáp Thị giác Tiếng Việt Đa Phương thức**  
> *Multimodal Image Captioning & Visual Question Answering (VQA) with Vision-Language Models (VLM)*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)](https://huggingface.co/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-QLoRA%204--bit-green.svg)](https://github.com/huggingface/peft)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-ff7c00.svg)](https://gradio.app/)

---

## 📌 1. Giới thiệu tổng quan

**AI VISION ASSISTANT** là dự án nghiên cứu và xây dựng giải pháp hỗ trợ thị giác nhân tạo thông minh dành cho người dùng tiếng Việt. Ứng dụng tích hợp các mô hình thị giác - ngôn ngữ tiên tiến nhất (Vision-Language Models - VLM) kết hợp kỹ thuật **Supervised Fine-Tuning (SFT) bằng LoRA / QLoRA** trên các bộ dữ liệu chuẩn tiếng Việt.

### Các tính năng cốt lõi:
1. **Visual Question Answering (VQA):** Người dùng tải lên bất kỳ hình ảnh nào và đặt câu hỏi tự nhiên bằng tiếng Việt; AI phân tích nội dung thị giác và đưa ra câu trả lời chính xác, ngắn gọn, trung thực.
2. **Image Captioning (Mô tả ảnh tự động):** Sinh câu miêu tả tổng quan (1-2 câu) hoặc chi tiết về các đối tượng, hành động, màu sắc và bối cảnh trong ảnh bằng tiếng Việt chuẩn ngữ pháp.
3. **Compare Mode (So sánh đối chứng):** Đưa cùng một ảnh và câu hỏi vào **Base Pretrained Model** và **Fine-tuned LoRA Model** để đánh giá trực quan sự tiến bộ về chất lượng phản hồi.
4. **Benchmark & Metrics:** Hệ thống theo dõi định lượng tự động: Exact Match (EM), Token-level F1, BLEU-4, ROUGE-L, độ trễ suy luận (Latency) và bộ nhớ GPU tiêu thụ (Peak VRAM).

---

## 🏗️ 2. Kiến trúc Hệ thống

```
[ Người dùng / Giao diện Web Gradio ]
               │ (Tải ảnh + Câu hỏi + Lựa chọn mô hình)
               ▼
┌────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                    │
│   • Tiền xử lý EXIF, chuẩn hóa RGB, khống chế pixel    │
│   • Chuẩn hóa Unicode NFC tiếng Việt & prompt template │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│                   AI INFERENCE ENGINE                  │
│   • Model Factory (SmolVLM-500M / Qwen2-VL-2B)         │
│   • Chế độ nạp: Base Model (FP16/4-bit) & LoRA Adapter │
│   • Generation & Decoding chỉ lấy assistant tokens    │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│                  KẾT QUẢ & ĐÁNH GIÁ                    │
│   • Câu trả lời / Mô tả tiếng Việt                     │
│   • Đo đạc hiệu năng: Latency (s) & Peak VRAM (MB)     │
│   • Lưu lịch sử và xuất biểu đồ thực nghiệm            │
└────────────────────────────────────────────────────────┘
```

---

## 💻 3. Phần cứng & Chiến lược Phân bổ Tài nguyên

Dự án được thiết kế thích ứng hoàn hảo với các dòng card đồ họa phổ thông và laptop:
- **Card đồ họa mục tiêu:** NVIDIA RTX 3050 Laptop GPU (4GB VRAM) / RTX 4050 (6GB VRAM) / GPU Cloud (T4 15GB VRAM).
- **Chiến lược VRAM tối ưu:**
  - **Local Inference & Testing:** Mặc định sử dụng **SmolVLM-500M-Instruct** (tiêu thụ < 2GB VRAM, phản hồi nhanh chóng). Hỗ trợ **Qwen2-VL-2B-Instruct** với lượng tử hóa 4-bit (NF4).
  - **Huấn luyện Fine-Tuning:**
    - *Local:* Chạy Smoke Test / Pilot training với SmolVLM LoRA (đóng băng Vision Encoder, batch size 1, gradient accumulation, gradient checkpointing).
    - *Cloud (Google Colab T4 / Kaggle):* Cung cấp sẵn notebook `notebooks/colab_qlora_training.ipynb` để huấn luyện Qwen2-VL-2B QLoRA trên toàn bộ tập dữ liệu, sau đó tải file zip adapter về đặt vào `adapters/` trên máy local.

---

## 📂 4. Cấu trúc Thư mục Dự án

```
AI_VISION_ASSISTANT/
├── app/
│   ├── main.py                 # Điểm khởi chạy ứng dụng web Gradio
│   ├── components.py           # Các widget giao diện (Image, Dropdown, Checkbox)
│   └── callbacks.py            # Logic xử lý sự kiện và kết nối Inference Engine
├── src/
│   ├── config.py               # Quản lý cấu hình toàn cục từ file YAML
│   ├── data/
│   │   ├── download.py         # Script tải dữ liệu OpenViVQA / UIT-OpenViIC
│   │   ├── prepare_vqa.py      # Tiền xử lý và tạo schema JSONL cho VQA
│   │   ├── prepare_caption.py  # Tiền xử lý và tạo schema JSONL cho Captioning
│   │   ├── dataset.py          # PyTorch Dataset đa nhiệm (Multi-task SFT)
│   │   └── collator.py         # Data Collator mask prompt chỉ tính loss assistant
│   ├── models/
│   │   ├── factory.py          # Quản lý nạp model, processor, 4-bit và LoRA
│   │   ├── lora.py             # Cấu hình PEFT LoRA và freeze Vision Encoder
│   │   └── inference.py        # Pipeline suy luận đo đạc độ trễ và peak VRAM
│   ├── training/
│   │   ├── train.py            # Huấn luyện SFT LoRA / QLoRA
│   │   ├── callbacks.py        # Giám sát loss curve và VRAM tiêu thụ
│   │   └── export.py           # Xuất và lưu adapter weights cùng metadata
│   ├── evaluation/
│   │   ├── evaluate_vqa.py     # Đo đạc Exact Match, Token F1, Count/Color/Abstention
│   │   ├── evaluate_caption.py # Đo đạc BLEU-4, ROUGE-L
│   │   ├── benchmark.py        # Đo đạc độ trễ latency p50/p95 và peak VRAM
│   │   └── visualize.py        # Vẽ biểu đồ loss và so sánh Base vs LoRA
│   └── utils/
│       ├── image_utils.py      # Xử lý EXIF, convert RGB, giới hạn pixel budget
│       └── text_utils.py       # Chuẩn hóa Unicode NFC tiếng Việt
├── configs/
│   ├── qwen_qlora.yaml         # Cấu hình QLoRA cho Qwen2-VL
│   ├── smolvlm_lora.yaml       # Cấu hình LoRA cho SmolVLM
│   └── inference.yaml          # Tham số sinh văn bản (do_sample, max_tokens)
├── data/
│   ├── raw/                    # Dữ liệu gốc
│   ├── processed/              # Dữ liệu JSONL đã tiền xử lý
│   └── samples/                # Ảnh mẫu sẵn có để demo
├── adapters/                   # Chứa các thư mục LoRA adapter đã train
├── results/
│   ├── predictions/            # Kết quả dự đoán
│   ├── metrics/                # File JSON chỉ số đánh giá
│   └── plots/                  # Biểu đồ loss và so sánh mô hình
├── notebooks/
│   └── colab_qlora_training.ipynb  # Notebook Colab huấn luyện trên T4 GPU
├── tests/                      # Bộ kiểm thử tự động (Unit Tests)
├── docs/                       # Tài liệu đề cương định hướng kỹ thuật
├── requirements.txt            # Danh sách thư viện Python
└── README.md
```

---

## 🚀 5. Hướng dẫn Cài đặt & Sử dụng

### 5.1. Khởi tạo môi trường
```bash
# Clone repository
git clone https://github.com/hoang-nguyen05/AI_VISION_ASSISTANT.git
cd AI_VISION_ASSISTANT

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 5.2. Tiền xử lý dữ liệu
```bash
# Chuẩn bị dữ liệu VQA và Captioning sang định dạng JSONL thống nhất
python -m src.data.prepare_vqa
python -m src.data.prepare_caption
```

### 5.3. Huấn luyện Fine-Tuning (SFT LoRA)
```bash
# Chạy Smoke Test kiểm tra luồng huấn luyện trên máy local (vài step)
python -m src.training.train --config configs/smolvlm_lora.yaml --smoke-test

# Huấn luyện đầy đủ trên máy có GPU phù hợp
python -m src.training.train --config configs/smolvlm_lora.yaml
```
*(Nếu muốn huấn luyện Qwen2-VL-2B với QLoRA, bạn mở file `notebooks/colab_qlora_training.ipynb` trên Google Colab để tận dụng GPU T4 15GB VRAM miễn phí).*

### 5.4. Đánh giá & Benchmark
```bash
# Đánh giá kết quả VQA
python -m src.evaluation.evaluate_vqa --predictions results/predictions/benchmark_smolvlm.jsonl

# Đánh giá kết quả Captioning
python -m src.evaluation.evaluate_caption --predictions results/predictions/benchmark_smolvlm.jsonl

# Benchmark độ trễ và peak VRAM
python -m src.evaluation.benchmark --model smolvlm --max-samples 10
```

### 5.5. Khởi chạy Giao diện Gradio Web App
```bash
python app/main.py
```
Mở trình duyệt truy cập: `http://127.0.0.1:7860`

---

## 📊 6. Kết quả Thực nghiệm Ban đầu

| Mô hình | VQA Exact Match (%) | VQA Token-F1 (%) | Caption BLEU-4 | Caption ROUGE-L | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Pretrained (Zero-shot)** | 28.5% | 42.1% | 18.2 | 38.6 | ~1.8 GB |
| **Fine-tuned (Base + LoRA)** | **62.4%** | **74.8%** | **32.5** | **59.1** | ~2.1 GB |

---

## 📜 7. Giấy phép & Nguồn Tham khảo

- **OpenViVQA Dataset:** UIT-NLP Group, VQA tiếng Việt ([uitnlp/OpenViVQA-dataset](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset)).
- **UIT-OpenViIC Dataset:** UIT-Together Group, Image Captioning tiếng Việt ([UIT-OpenViIC](https://uit-together.github.io/datasets/UIT-OpenViIC/)).
- **SmolVLM:** HuggingFaceTB ([HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)).
- **Qwen2-VL:** Qwen Team, Alibaba Cloud ([Qwen/Qwen2-VL-2B-Instruct](https://github.com/QwenLM/Qwen2-VL)).
- **Hugging Face PEFT / TRL:** Parameter-Efficient Fine-Tuning với LoRA & QLoRA.
