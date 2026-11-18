import json
from pathlib import Path
from src.evaluation.visualize import plot_training_curves, plot_model_comparison

def test_visualize_plots(tmp_path):
    history_file = tmp_path / "history.json"
    dummy_history = [
        {"step": 1, "loss": 2.5},
        {"step": 2, "loss": 2.1},
        {"step": 3, "loss": 1.7},
    ]
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(dummy_history, f)

    loss_img = tmp_path / "loss.png"
    plot_training_curves(str(history_file), str(loss_img))
    assert loss_img.exists()
    assert loss_img.stat().st_size > 0

    comp_img = tmp_path / "comp.png"
    metrics_base = {"Exact Match (%)": 30.0, "Token F1 (%)": 45.0}
    metrics_ft = {"Exact Match (%)": 65.0, "Token F1 (%)": 78.0}
    plot_model_comparison(metrics_base, metrics_ft, str(comp_img))
    assert comp_img.exists()
    assert comp_img.stat().st_size > 0
