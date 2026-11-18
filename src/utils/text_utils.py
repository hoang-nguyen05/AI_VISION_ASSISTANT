import re
import unicodedata
import string
from typing import List

def normalize_vietnamese_text(text: str) -> str:
    """
    Normalizes Vietnamese text to Unicode NFC and collapses excess whitespaces.
    Does NOT remove Vietnamese tone marks or accents.
    """
    if not text:
        return ""
    # Normalize to Unicode NFC standard
    text = unicodedata.normalize("NFC", str(text))
    # Replace multiple whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_text_for_eval(text: str) -> str:
    """
    Cleans text for quantitative evaluation (Exact Match, Token-level F1).
    Lowercases, removes punctuation, but keeps Vietnamese diacritics intact.
    """
    text = normalize_vietnamese_text(text).lower()
    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    # Collapse whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_vietnamese(text: str) -> List[str]:
    """Tokenizes normalized text into words."""
    cleaned = clean_text_for_eval(text)
    return cleaned.split() if cleaned else []
