import os
import yaml
from pathlib import Path
from typing import Any, Dict

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}

class ConfigManager:
    """Centralized configuration manager for AI Vision Assistant."""
    
    def __init__(self, inference_config_path: str = "configs/inference.yaml"):
        self.inference_config = load_yaml_config(inference_config_path)
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.samples_dir = self.data_dir / "samples"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.adapters_dir = self.project_root / "adapters"
        self.results_dir = self.project_root / "results"
        
        # Ensure directories exist
        for d in [self.raw_data_dir, self.processed_data_dir, self.samples_dir,
                  self.checkpoints_dir, self.adapters_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self.inference_config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val
