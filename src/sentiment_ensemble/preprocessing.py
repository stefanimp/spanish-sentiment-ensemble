"""Text normalization used by the three base models and the binary cascade."""

from __future__ import annotations

import re
from typing import Literal

CleanMode = Literal["cased", "robertuito_uncased"]
EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]")


def clean_text(text: str, mode: CleanMode) -> str:
    """Keep emoji and punctuation signals while normalizing URLs and user mentions."""
    if mode not in {"cased", "robertuito_uncased"}:
        raise ValueError(f"Unknown normalization mode: {mode}")
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "url" if mode == "robertuito_uncased" else "URL", text)
    text = re.sub(r"@\w+", "@usuario", text)
    replacement = r"hashtag \1" if mode == "robertuito_uncased" else r"\1"
    text = re.sub(r"#(\w+)", replacement, text)
    text = EMOJI_RE.sub(lambda match: f" {match.group(0)} ", text)
    text = re.sub(r"([!?])\1{2,}", r"\1\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip() if mode == "robertuito_uncased" else text.strip()
