import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from equimo.language import SentencePieceTokenizer


def _require_language_extra():
    for module in ("tensorflow", "tensorflow_text"):
        if os.environ.get("EQUIMO_TEST_OPTIONAL_EXTRA") == "language":
            importlib.import_module(module)
        else:
            pytest.importorskip(module)


def test_sentencepiece_token_ids_truncation_and_padding():
    _require_language_extra()
    model_path = Path(__file__).parent / "data" / "tiny_sentencepiece.model"
    tokenizer = SentencePieceTokenizer(path=str(model_path))

    token_ids, padding_mask = tokenizer.encode(["Hello world"], max_length=16)
    expected_ids = [17, 11, 15, 14, 14, 16, 17, 3, 16, 5, 14, 12]
    assert token_ids.tolist() == [expected_ids + [0, 0, 0, 0]]
    assert padding_mask.tolist() == [[0] * len(expected_ids) + [1, 1, 1, 1]]
    assert token_ids.dtype == np.int32
    assert padding_mask.dtype == np.int32

    truncated_ids, truncated_mask = tokenizer.encode(["Hello world"], max_length=4)
    assert truncated_ids.tolist() == [expected_ids[:4]]
    assert truncated_mask.tolist() == [[0, 0, 0, 0]]
