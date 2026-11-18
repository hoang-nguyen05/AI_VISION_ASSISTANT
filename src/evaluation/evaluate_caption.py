import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import evaluate
from ..utils.text_utils import clean_text_for_eval
from ..config import PROJECT_ROOT

def compute_caption_metrics(predictions: List[Dict]) -> Dict[str, float]:
    """
    Computes BLEU-1, BLEU-4, and ROUGE-L for generated captions.
    """
    if not predictions:
        return {}

    preds = []
    refs = []

    for item in predictions:
        p = clean_text_for_eval(item.get("prediction", ""))
        # Can have one or multiple reference captions
        r = item.get("answer", item.get("ground_truth", item.get("caption", "")))
        if isinstance(r, list):
            r = [clean_text_for_eval(x) for x in r]
        else:
            r = [clean_text_for_eval(r)]
            
        preds.append(p)
        refs.append(r)

    results = {"total_samples": len(predictions)}

    # BLEU
    try:
        bleu_metric = evaluate.load("sacrebleu")
        bleu_res = bleu_metric.compute(predictions=preds, references=refs)
        results["bleu_score"] = round(bleu_res["score"], 2)
    except Exception as e:
        print(f"Cảnh báo sacrebleu: {e}")

    # ROUGE
    try:
        rouge_metric = evaluate.load("rouge")
        # Flatten refs for rouge
        flat_refs = [r[0] if isinstance(r, list) else r for r in refs]
        rouge_res = rouge_metric.compute(predictions=preds, references=flat_refs)
        results["rouge_1"] = round(rouge_res["rouge1"] * 100, 2)
        results["rouge_2"] = round(rouge_res["rouge2"] * 100, 2)
        results["rouge_l"] = round(rouge_res["rougeL"] * 100, 2)
    except Exception as e:
        print(f"Cảnh báo rouge: {e}")

    return results

def evaluate_caption_predictions(prediction_file: str, output_metric_file: Optional[str] = None):
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

    metrics = compute_caption_metrics(predictions)
    print("=" * 60)
    print(f"KẾT QUẢ ĐÁNH GIÁ CAPTIONING: {pred_path.name}")
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
    parser = argparse.ArgumentParser(description="Đánh giá kết quả Image Captioning")
    parser.add_argument("--predictions", type=str, required=True, help="Đường dẫn file predictions.jsonl")
    parser.add_argument("--output", type=str, default="results/metrics/caption_metrics.json", help="Đường dẫn lưu metrics")
    args = parser.parse_args()
    evaluate_caption_predictions(args.predictions, args.output)
