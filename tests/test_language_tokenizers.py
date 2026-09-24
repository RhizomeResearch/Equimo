from pathlib import Path

import numpy as np

from equimo.language import SentencePieceTokenizer
from equimo.language.tokenizers import DEFAULT_TOKENIZER_REPOSITORY
from equimo.serialization import DEFAULT_REPOSITORY_REVISION, DEFAULT_REPOSITORY_URL
from _optional import require_extra


def _require_language_extra():
    for module in ("tensorflow", "tensorflow_text"):
        require_extra(module, "language")


def test_default_repositories_use_the_same_immutable_revision():
    revision_path = f"/resolve/{DEFAULT_REPOSITORY_REVISION}/"

    assert len(DEFAULT_REPOSITORY_REVISION) == 40
    assert revision_path in DEFAULT_REPOSITORY_URL
    assert revision_path in DEFAULT_TOKENIZER_REPOSITORY
    assert "/resolve/main/" not in DEFAULT_REPOSITORY_URL
    assert "/resolve/main/" not in DEFAULT_TOKENIZER_REPOSITORY


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


def test_sentencepiece_unknown_id_is_not_treated_as_padding():
    _require_language_extra()
    model_path = Path(__file__).parent / "data" / "tiny_sentencepiece.model"
    tokenizer = SentencePieceTokenizer(path=str(model_path))

    token_ids, padding_mask = tokenizer.encode(["☃"], max_length=8, lowercase=False)
    valid_length = int(np.sum(padding_mask[0] == 0))

    assert valid_length > 0
    assert 0 in token_ids[0, :valid_length]
    assert padding_mask[0, :valid_length].tolist() == [0] * valid_length
