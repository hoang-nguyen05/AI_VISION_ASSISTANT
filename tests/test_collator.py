import torch
from PIL import Image
from unittest.mock import MagicMock
from src.data.collator import MultimodalDataCollator

def test_collator_with_mock_processor():
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "user\nquestion\nassistant\nanswer"
    mock_processor.tokenizer.pad_token_id = 0
    mock_processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]])
    }
    
    collator = MultimodalDataCollator(processor=mock_processor, max_length=128)
    
    batch = [
        {
            "image": Image.new("RGB", (64, 64), color="red"),
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]}
            ]
        }
    ]
    
    output = collator(batch)
    assert "input_ids" in output
    assert "labels" in output
    # Check that pad token (0) was masked to -100 in labels
    assert output["labels"][0, -1].item() == -100
