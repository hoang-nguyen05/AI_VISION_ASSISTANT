from pathlib import Path
from src.data.dataset import MultimodalVisionDataset

def test_dataset_loading():
    p = Path("data/processed/vqa_train.jsonl")
    if p.exists():
        ds = MultimodalVisionDataset(jsonl_paths=[p], max_samples=4)
        assert len(ds) <= 4
        item = ds[0]
        assert "image" in item
        assert "messages" in item
        assert item["messages"][0]["role"] == "user"
        assert item["messages"][1]["role"] == "assistant"
