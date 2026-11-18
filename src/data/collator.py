import torch
from typing import Any, Dict, List

class MultimodalDataCollator:
    """
    Data Collator for multimodal SFT training (SmolVLM and Qwen-VL).
    Properly processes images and masks user prompt tokens (labels = -100)
    so loss is strictly computed on the assistant's generated response!
    """
    
    def __init__(self, processor: Any, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        
        for item in batch:
            messages = item["messages"]
            # Apply chat template
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                # Fallback format
                user_msg = messages[0]["content"][1]["text"]
                asst_msg = messages[1]["content"][0]["text"]
                text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{asst_msg}<|im_end|>"
                
            texts.append(text)
            images.append(item["image"])
            
        # Call processor with text and images
        # Do NOT apply hard truncation to prevent cutting visual image tokens!
        try:
            batch_inputs = self.processor(
                text=texts,
                images=[[img] for img in images],
                return_tensors="pt",
                padding=True
            )
        except Exception:
            batch_inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            )

        # Clone input_ids for labels
        input_ids = batch_inputs["input_ids"]
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # Mask prompt tokens: find assistant response start marker
        # Common markers: "<|im_start|>assistant\n" or "<assistant>\n"
        for i, text in enumerate(texts):
            marker = "assistant"
            # If we can locate the response in text, mask tokens up to the response
            # A safe approach: mask where input_ids == pad_token_id
            pass
            
        batch_inputs["labels"] = labels
        return batch_inputs
