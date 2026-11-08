import gzip
from pathlib import Path

import numpy as np
import pytest

from equimo.language import ClipBpeTokenizer, SentencePieceTokenizer
from equimo.language.tokenizers import DEFAULT_TOKENIZER_REPOSITORY
from equimo.serialization import DEFAULT_REPOSITORY_REVISION, DEFAULT_REPOSITORY_URL
from _optional import require_extra


def _require_language_extra():
    for module in ("tensorflow", "tensorflow_text"):
        require_extra(module, "language")


def _require_clip_extra():
    for module in ("ftfy", "regex"):
        require_extra(module, "clip")


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


CLIP_START = 49406
CLIP_END = 49407
A_PHOTO_OF_A_CAT = [CLIP_START, 320, 1125, 539, 320, 2368, CLIP_END]


def test_clip_tokenizer_token_ids_and_padding_mask():
    _require_clip_extra()

    token_ids, padding_mask = ClipBpeTokenizer().encode(["a photo of a cat"])

    assert token_ids.shape == padding_mask.shape == (1, 32)
    assert token_ids.dtype == np.int32
    assert padding_mask.dtype == np.bool_
    assert token_ids[0].tolist() == A_PHOTO_OF_A_CAT + [0] * 25
    assert padding_mask[0].tolist() == [False] * 7 + [True] * 25


@pytest.mark.parametrize(
    "text",
    ["A PHOTO OF A CAT", "  a\tphoto\n of   a cat ", "a photo of a &#99;at"],
)
def test_clip_tokenizer_cleans_case_whitespace_and_html(text):
    _require_clip_extra()

    token_ids, _ = ClipBpeTokenizer().encode([text])

    assert token_ids[0, :7].tolist() == A_PHOTO_OF_A_CAT


def test_clip_tokenizer_encodes_accents_and_emoji():
    _require_clip_extra()

    token_ids, _ = ClipBpeTokenizer().encode(["Tom's Café &amp; 😀"])

    assert token_ids[0, :7].tolist() == [
        CLIP_START,
        2435,
        568,
        15304,
        261,
        7334,
        CLIP_END,
    ]


def test_clip_tokenizer_truncation_keeps_the_end_token():
    _require_clip_extra()
    tokenizer = ClipBpeTokenizer()

    token_ids, padding_mask = tokenizer.encode(["a " * 40, "cat"])

    assert token_ids[0].tolist() == [CLIP_START] + [320] * 30 + [CLIP_END]
    assert not padding_mask[0].any()
    assert padding_mask[1].tolist() == [False] * 3 + [True] * 29


def test_clip_tokenizer_marks_real_token_id_zero_as_valid():
    _require_clip_extra()

    token_ids, padding_mask = ClipBpeTokenizer().encode(["!@#$%^&*()"])

    valid = token_ids[0, ~padding_mask[0]]
    assert 0 in valid.tolist()
    assert valid[-1] == CLIP_END


def test_clip_tokenizer_reads_merges_from_path_and_honors_context_length(tmp_path):
    _require_clip_extra()
    path = tmp_path / "merges.txt.gz"
    path.write_bytes(gzip.compress(b"#version: 0.2\nc a\nca t</w>"))

    token_ids, padding_mask = ClipBpeTokenizer(context_length=5, path=str(path)).encode(
        ["cat"]
    )

    # 512 byte symbols, then the merges "ca" and "cat</w>", then the special tokens.
    assert token_ids[0].tolist() == [514, 513, 515, 0, 0]
    assert padding_mask[0].tolist() == [False, False, False, True, True]


def test_clip_tokenizer_rejects_a_single_string():
    _require_clip_extra()

    with pytest.raises(TypeError, match="sequence of strings"):
        ClipBpeTokenizer().encode("a photo of a cat")
