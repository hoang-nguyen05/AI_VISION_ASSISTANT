import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from ..utils.text_utils import clean_text_for_eval, tokenize_vietnamese
from ..config import PROJECT_ROOT

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Computes exact match score between normalized prediction and ground truth."""
    return 1.0 if clean_text_for_eval(prediction) == clean_text_for_eval(ground_truth) else 0.0

def compute_f1(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 score."""
    pred_tokens = tokenize_vietnamese(prediction)
    gold_tokens = tokenize_vietnamese(ground_truth)
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
        
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
        
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_vqa_metrics(predictions: List[Dict]) -> Dict[str, float]:
    """
    Computes Exact Match, Token F1, Count accuracy, Color accuracy, and Abstention accuracy.
    """
    if not predictions:
        return {}

    em_scores = []
    f1_scores = []
    
    count_scores = []
    color_scores = []
    abstention_scores = []

    count_keywords = ["bao nhiêu", "mấy", "số lượng", "đếm"]
    color_keywords = ["màu gì", "màu sắc", "màu"]
    abstention_keywords = ["không xác định", "không rõ", "không thể", "không có thông tin"]

    for item in predictions:
        q = item.get("question", "").lower()
        pred = item.get("prediction", "")
        gold = item.get("answer", item.get("ground_truth", ""))

        em = compute_exact_match(pred, gold)
        f1 = compute_f1(pred, gold)
        
        em_scores.append(em)
        f1_scores.append(f1)

        # Count subtest
        if any(kw in q for kw in count_keywords):
            count_scores.append(em)

        # Color subtest
        if any(kw in q for kw in color_keywords):
            color_scores.append(em)

        # Abstention subtest
        if any(kw in gold.lower() for kw in abstention_keywords):
            abstention_scores.append(1.0 if any(kw in pred.lower() for kw in abstention_keywords) else 0.0)

    results = {
        "total_samples": len(predictions),
        "exact_match": round(sum(em_scores) / len(em_scores) * 100, 2),
        "f1_score": round(sum(f1_scores) / len(f1_scores) * 100, 2),
        "count_accuracy": round(sum(count_scores) / len(count_scores) * 100, 2) if count_scores else None,
        "color_accuracy": round(sum(color_scores) / len(color_scores) * 100, 2) if color_scores else None,
        "abstention_accuracy": round(sum(abstention_scores) / len(abstention_scores) * 100, 2) if abstention_scores else None,
    }
    return results

def evaluate_vqa_predictions(prediction_file: str, output_metric_file: Optional[str] = None):
    pred_path = Path(prediction_file)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / pred_path

    if not pred_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dự đoán: {pred_path}")

    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    metrics = compute_vqa_metrics(predictions)
    print("=" * 60)
    print(f"KẾT QUẢ ĐÁNH GIÁ VQA: {pred_path.name}")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if output_metric_file:
        out_p = Path(output_metric_file)
        if not out_p.is_absolute():
            out_p = PROJECT_ROOT / out_p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Đã lưu metrics vào: {out_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá kết quả VQA")
    parser.add_argument("--predictions", type=str, required=True, help="Đường dẫn file predictions.jsonl")
    parser.add_argument("--output", type=str, default="results/metrics/vqa_metrics.json", help="Đường dẫn lưu metrics")
    args = parser.parse_args()
    evaluate_vqa_predictions(args.predictions, args.output)
