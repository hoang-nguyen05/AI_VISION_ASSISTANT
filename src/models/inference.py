import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from PIL import Image

from .factory import ModelFactory
from ..utils.image_utils import load_and_preprocess_image, validate_image
from ..utils.text_utils import normalize_vietnamese_text
from ..config import ConfigManager

class VisionAssistantEngine:
    """
    Inference Engine for Multimodal VQA and Image Captioning.
    Tracks execution latency, GPU memory usage, and formats Vietnamese responses.
    """

    def __init__(
        self,
        model_name: str = "smolvlm",
        adapter_path: Optional[str] = None,
        config_path: str = "configs/inference.yaml"
    ):
        self.cfg = ConfigManager(config_path)
        self.model_name = model_name
        self.adapter_path = adapter_path
        
        # Load configs
        model_cfg = self.cfg.get("model", {})
        self.device = model_cfg.get("device", "auto")
        self.torch_dtype = model_cfg.get("torch_dtype", "float16")
        self.load_in_4bit = model_cfg.get("load_in_4bit", False)
        
        self.processor = None
        self.model = None
        self._load()

    def _load(self):
        """Loads processor and model instances."""
        self.processor = ModelFactory.load_processor(self.model_name)
        self.model = ModelFactory.load_model(
            model_name_or_path=self.model_name,
            adapter_path=self.adapter_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
            load_in_4bit=self.load_in_4bit
        )

    def answer_question(
        self,
        image_input: Union[str, Path, Image.Image],
        question: str,
        max_new_tokens: int = 128
    ) -> Dict[str, Any]:
        """Answers a question about the image in Vietnamese."""
        return self._generate(
            image_input=image_input,
            text_prompt=question,
            task="vqa",
            max_new_tokens=max_new_tokens
        )

    def generate_caption(
        self,
        image_input: Union[str, Path, Image.Image],
        detailed: bool = False,
        max_new_tokens: int = 160
    ) -> Dict[str, Any]:
        """Generates a Vietnamese caption for the image."""
        if detailed:
            prompt = self.cfg.get("prompts.caption_detailed", "Hãy mô tả chi tiết các đối tượng, hành động và bối cảnh bức ảnh này bằng tiếng Việt.")
        else:
            prompt = self.cfg.get("prompts.caption_short", "Hãy mô tả ngắn gọn nội dung bức ảnh này bằng tiếng Việt trong 1-2 câu.")
            
        return self._generate(
            image_input=image_input,
            text_prompt=prompt,
            task="caption",
            max_new_tokens=max_new_tokens
        )

    def _generate(
        self,
        image_input: Union[str, Path, Image.Image],
        text_prompt: str,
        task: str = "vqa",
        max_new_tokens: int = 128
    ) -> Dict[str, Any]:
        """Core multimodal generation method."""
        is_valid, err = validate_image(image_input)
        if not is_valid:
            return {
                "success": False,
                "text": f"Lỗi ảnh: {err}",
                "latency_sec": 0.0,
                "peak_vram_mb": 0.0
            }

        max_pixels = self.cfg.get("image.max_pixels", 524288)
        min_pixels = self.cfg.get("image.min_pixels", 3136)
        image = load_and_preprocess_image(image_input, max_pixels=max_pixels, min_pixels=min_pixels)
        
        # Reset peak VRAM tracker
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.time()
        
        # Construct multimodal chat message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": normalize_vietnamese_text(text_prompt)}
                ]
            }
        ]

        if hasattr(self.processor, "apply_chat_template"):
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = f"<|im_start|>user\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Prepare inputs
        try:
            inputs = self.processor(
                text=prompt_text,
                images=[image],
                return_tensors="pt"
            )
        except Exception:
            inputs = self.processor(
                text=prompt_text,
                images=image,
                return_tensors="pt"
            )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generation configuration
        gen_cfg = self.cfg.get(f"generation.{task}", {})
        do_sample = gen_cfg.get("do_sample", False)
        temperature = gen_cfg.get("temperature", 0.1)
        top_p = gen_cfg.get("top_p", 0.9)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None
            )

        # Slice generated tokens only
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        decoded = normalize_vietnamese_text(decoded)
        
        latency = round(time.time() - start_time, 3)
        
        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            
        return {
            "success": True,
            "text": decoded,
            "latency_sec": latency,
            "peak_vram_mb": peak_vram_mb,
            "model_name": self.model_name,
            "has_adapter": self.adapter_path is not None
        }
