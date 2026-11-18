import json
import time
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

from ..config import PROJECT_ROOT

def benchmark_performance(
    model_name: str = "smolvlm",
    adapter_path: str = None,
    test_file: str = "data/processed/vqa_val.jsonl",
    max_samples: int = 20,
    output_file: str = "results/metrics/benchmark_results.json"
) -> Dict:
    """
    Measures latency (p50, p95), output token throughput, and peak VRAM.
    """
    # Lazy import to avoid loading heavy VLM packages during test collection
    from ..models.inference import VisionAssistantEngine
    import torch

    print("=" * 60)
    print(f"BENCHMARK HIỆU NĂNG MÔ HÌNH: {model_name}")
    print(f"Adapter: {adapter_path} | Max samples: {max_samples}")
    print("=" * 60)

    engine = VisionAssistantEngine(model_name=model_name, adapter_path=adapter_path)
    
    test_p = Path(test_file)
    if not test_p.is_absolute():
        test_p = PROJECT_ROOT / test_p

    samples = []
    if test_p.exists():
        with open(test_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
                    if len(samples) >= max_samples:
                        break

    if not samples:
        print("Không có mẫu kiểm thử trong file. Sử dụng ảnh mẫu mặc định...")
        samples = [
            {"image": "data/samples/sample_table.jpg", "question": "Trên bàn có gì?", "answer": "Có 3 quả táo đỏ."}
        ]

    latencies = []
    vrams = []
    predictions = []

    for idx, item in enumerate(samples):
        img_path = item["image"]
        q = item.get("question", "Hãy mô tả bức ảnh.")
        res = engine.answer_question(img_path, q)
        
        latencies.append(res["latency_sec"])
        vrams.append(res["peak_vram_mb"])
        
        predictions.append({
            "id": item.get("id", str(idx)),
            "image": str(img_path),
            "question": q,
            "prediction": res["text"],
            "ground_truth": item.get("answer", ""),
            "latency_sec": res["latency_sec"],
            "peak_vram_mb": res["peak_vram_mb"]
        })
        print(f"  [{idx+1}/{len(samples)}] Latency: {res['latency_sec']}s | VRAM: {res['peak_vram_mb']}MB")

    results = {
        "model_name": model_name,
        "has_adapter": adapter_path is not None,
        "sample_count": len(samples),
        "latency_mean_sec": round(float(np.mean(latencies)), 3),
        "latency_p50_sec": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_sec": round(float(np.percentile(latencies, 95)), 3),
        "peak_vram_mb": round(float(np.max(vrams)), 2) if vrams else 0.0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

    print("\n" + "=" * 60)
    print("TỔNG HỢP HIỆU NĂNG:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Save outputs
    out_p = Path(output_file)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    pred_out = PROJECT_ROOT / "results" / "predictions" / f"benchmark_{model_name}.jsonl"
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_out, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy benchmark độ trễ và VRAM")
    parser.add_argument("--model", type=str, default="smolvlm", help="Tên model (smolvlm, qwen)")
    parser.add_argument("--adapter", type=str, default=None, help="Đường dẫn LoRA adapter (tùy chọn)")
    parser.add_argument("--max-samples", type=int, default=10, help="Số lượng mẫu chạy benchmark")
    args = parser.parse_args()
    benchmark_performance(model_name=args.model, adapter_path=args.adapter, max_samples=args.max_samples)
