from src.evaluation.evaluate_vqa import compute_exact_match, compute_f1, compute_vqa_metrics

def test_compute_exact_match():
    assert compute_exact_match("Có 3 quả táo", "có 3 quả táo.") == 1.0
    assert compute_exact_match("Có 2 quả táo", "có 3 quả táo") == 0.0

def test_compute_f1():
    pred = "có 3 quả táo màu đỏ"
    gold = "có 3 quả táo màu vàng"
    f1 = compute_f1(pred, gold)
    assert 0.0 < f1 < 1.0

def test_compute_vqa_metrics():
    predictions = [
        {
            "question": "Có bao nhiêu quả táo trên bàn?",
            "prediction": "Có 3 quả táo",
            "answer": "Có 3 quả táo"
        },
        {
            "question": "Chiếc áo màu gì?",
            "prediction": "Màu xanh",
            "answer": "Màu xanh"
        },
        {
            "question": "Người này tên gì?",
            "prediction": "Không xác định được từ hình ảnh",
            "answer": "Không xác định được"
        }
    ]
    res = compute_vqa_metrics(predictions)
    assert res["exact_match"] == 66.67
    assert res["count_accuracy"] == 100.0
    assert res["color_accuracy"] == 100.0
    assert res["abstention_accuracy"] == 100.0
