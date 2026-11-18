from .evaluate_vqa import evaluate_vqa_predictions, compute_vqa_metrics
from .visualize import plot_training_curves, plot_model_comparison

def benchmark_performance(*args, **kwargs):
    from .benchmark import benchmark_performance as _fn
    return _fn(*args, **kwargs)

def compute_caption_metrics(*args, **kwargs):
    from .evaluate_caption import compute_caption_metrics as _fn
    return _fn(*args, **kwargs)

__all__ = [
    "evaluate_vqa_predictions",
    "compute_vqa_metrics",
    "compute_caption_metrics",
    "benchmark_performance",
    "plot_training_curves",
    "plot_model_comparison"
]
