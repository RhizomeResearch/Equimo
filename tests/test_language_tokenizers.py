from pathlib import Path

import numpy as np
import pytest

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


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def _downloader():
    # ``download`` does not use instance state; skip the TensorFlow-backed constructor.
    return object.__new__(SentencePieceTokenizer)


def test_tokenizer_cache_is_keyed_by_full_url(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    contents = {
        "https://example.test/rev-a/tok.model": b"revision a",
        "https://example.test/rev-b/tok.model": b"revision b",
    }
    monkeypatch.setattr(
        "equimo.language.tokenizers.requests.get",
        lambda url, **kwargs: _FakeResponse(contents[url]),
    )

    paths = [_downloader().download(url) for url in contents]

    assert paths[0] != paths[1]
    assert [path.read_bytes() for path in paths] == list(contents.values())
    assert all(path.is_relative_to(tmp_path / ".cache" / "equimo") for path in paths)


def test_missing_language_extra_fails_before_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    def missing_extra():
        raise ImportError("language extra missing")

    def unexpected_download(url, **kwargs):
        raise AssertionError("download attempted without the language extra")

    monkeypatch.setattr(
        "equimo.language.tokenizers._require_tensorflow_text", missing_extra
    )
    monkeypatch.setattr("equimo.language.tokenizers.requests.get", unexpected_download)

    with pytest.raises(ImportError, match="language extra missing"):
        SentencePieceTokenizer(identifier="tok")
