import pytest
from src.utils.text_utils import normalize_vietnamese_text, clean_text_for_eval, tokenize_vietnamese

def test_normalize_vietnamese_text():
    # Test multiple whitespaces
    raw = "  Hôm   nay trời    đẹp quá!   "
    expected = "Hôm nay trời đẹp quá!"
    assert normalize_vietnamese_text(raw) == expected

def test_clean_text_for_eval():
    # Test punctuation removal and lowercasing while preserving accents
    raw = "Có 3 quả táo màu đỏ trên bàn!"
    cleaned = clean_text_for_eval(raw)
    assert cleaned == "có 3 quả táo màu đỏ trên bàn"

def test_tokenize_vietnamese():
    raw = "Một người đang đi xe đạp"
    tokens = tokenize_vietnamese(raw)
    assert tokens == ["một", "người", "đang", "đi", "xe", "đạp"]
