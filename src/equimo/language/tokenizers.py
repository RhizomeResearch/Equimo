import gzip
import hashlib
import html
import importlib.resources
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import requests

from equimo._io import atomic_file
from equimo._pretrained import DEFAULT_REPOSITORY_REVISION

DEFAULT_TOKENIZER_REPOSITORY = (
    "https://huggingface.co/poiretclement/equimo/resolve/"
    f"{DEFAULT_REPOSITORY_REVISION}/models/tokenizers"
)

_CLIP_VOCABULARY_FILE = "bpe_simple_vocab_16e6.txt.gz"
# The vocabulary file lists more merges than the 49,408 tokens CLIP uses:
# 512 byte symbols, 48,894 merges and the two special tokens.
_CLIP_MAX_MERGES = 49152 - 256 - 2
_CLIP_START_TOKEN = "<|startoftext|>"
_CLIP_END_TOKEN = "<|endoftext|>"
_CLIP_TOKEN_PATTERN = (
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"
    r"|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
)


def _require_tensorflow_text():
    try:
        import tensorflow as tf
        import tensorflow_text
    except ImportError as exc:
        raise ImportError(
            "`tensorflow` and `tensorflow_text` are required for language "
            "tokenizers. Install Equimo with the 'language' extra (for example, "
            'pip install "equimo[language]").'
        ) from exc
    return tf, tensorflow_text


def _require_clip():
    try:
        import ftfy
        import regex
    except ImportError as exc:
        raise ImportError(
            "`ftfy` and `regex` are required for the CLIP tokenizer. Install "
            "Equimo with the 'clip' extra (for example, "
            'pip install "equimo[clip]").'
        ) from exc
    return ftfy, regex


class SentencePieceTokenizer:
    """SentencePiece tokenizer wrapper used by Equimo language checkpoints."""

    def __init__(
        self,
        identifier: Optional[str] = None,
        url: Optional[str] = None,
        path: Optional[str] = None,
        repository: str = DEFAULT_TOKENIZER_REPOSITORY,
    ):
        if not identifier and not path and not url:
            raise ValueError("At least one of identifier, path, or url is required.")
        _, tensorflow_text = _require_tensorflow_text()

        if identifier:
            url = f"{repository}/{identifier}.model"
        if url:
            model_file_path = self.download(url)
        else:
            assert path
            model_file_path = Path(path).expanduser()
            if not model_file_path.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {path}")

        with open(model_file_path, "rb") as f:
            model = f.read()
        self.tokenizer = tensorflow_text.SentencepieceTokenizer(
            model=model, add_eos=False, add_bos=False
        )

    def download(self, url: str) -> Path:
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("Tokenizer URL must start with http:// or https://.")

        fname = Path(urlparse(url).path).name
        # Key the cache by the full URL so a pinned repository revision never reuses
        # a same-named file downloaded from another revision or host.
        url_digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        model_file_path = (
            Path("~/.cache/equimo/tokenizers").expanduser() / url_digest / fname
        )

        if not model_file_path.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with atomic_file(model_file_path) as temporary:
                temporary.write_bytes(response.content)

        return model_file_path

    def encode(self, input_text, max_length: int = 64, lowercase: bool = True):
        """Return ``(token_ids, padding_mask)`` arrays.

        ``padding_mask`` uses ``1`` for padding tokens and ``0`` for valid
        tokens, matching :class:`TextTransformerEncoder`.
        """
        tf, _ = _require_tensorflow_text()
        text = tf.strings.lower(input_text) if lowercase else input_text
        ragged_tokens = self.tokenizer.tokenize(text)
        token_lengths = ragged_tokens.row_lengths()
        tokens = ragged_tokens.to_tensor(default_value=0)
        curr_len = tokens.shape[1]
        if curr_len > max_length:
            tokens = tokens[:, :max_length]
        else:
            padding_len = max_length - curr_len
            tokens = tf.pad(tokens, [[0, 0], [0, padding_len]], constant_values=0)
        valid_lengths = tf.minimum(token_lengths, max_length)
        padding_mask = 1 - tf.sequence_mask(
            valid_lengths,
            maxlen=max_length,
            dtype=tf.int32,
        )
        return tokens.numpy(), padding_mask.numpy()


# The CLIP helpers below and ClipBpeTokenizer are adapted from OpenAI CLIP's
# clip/simple_tokenizer.py (https://github.com/openai/CLIP), MIT License,
# Copyright (c) 2021 OpenAI.
def _clip_byte_encoder() -> dict[int, str]:
    """Map every byte to a printable character, so BPE never sees whitespace."""
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    characters = printable[:]
    remapped = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + remapped)
            remapped += 1
    return dict(zip(printable, map(chr, characters), strict=True))


def _clip_symbol_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:], strict=False))


def _clip_merge_pair(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    first, second = pair
    merged = []
    index = 0
    while index < len(word):
        if word[index : index + 2] == (first, second):
            merged.append(first + second)
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return tuple(merged)


class ClipBpeTokenizer:
    """CLIP byte-level BPE tokenizer producing fixed-length token sequences.

    Reads the CLIP vocabulary shipped with Equimo, or the gzipped merges file at
    ``path``. Requires the ``clip`` extra.
    """

    def __init__(self, context_length: int = 32, path: Optional[str] = None):
        _, regex = _require_clip()
        vocabulary = (
            importlib.resources.files("equimo.language")
            / "data"
            / _CLIP_VOCABULARY_FILE
            if path is None
            else Path(path).expanduser()
        )
        lines = gzip.decompress(vocabulary.read_bytes()).decode("utf-8").split("\n")
        merges = [tuple(line.split()) for line in lines[1 : _CLIP_MAX_MERGES + 1]]

        self.context_length = context_length
        self._byte_encoder = _clip_byte_encoder()
        symbols = list(self._byte_encoder.values())
        symbols += [symbol + "</w>" for symbol in symbols]
        symbols += ["".join(merge) for merge in merges]
        symbols += [_CLIP_START_TOKEN, _CLIP_END_TOKEN]
        self._token_ids = {symbol: index for index, symbol in enumerate(symbols)}
        self._merge_ranks = {merge: rank for rank, merge in enumerate(merges)}
        self._bpe_cache = {
            _CLIP_START_TOKEN: [_CLIP_START_TOKEN],
            _CLIP_END_TOKEN: [_CLIP_END_TOKEN],
        }
        self._pattern = regex.compile(_CLIP_TOKEN_PATTERN, regex.IGNORECASE)

    def _bpe(self, token: str) -> list[str]:
        if token in self._bpe_cache:
            return self._bpe_cache[token]

        word = (*token[:-1], token[-1] + "</w>")
        pairs = _clip_symbol_pairs(word)
        while pairs:
            best = min(pairs, key=lambda pair: self._merge_ranks.get(pair, math.inf))
            if best not in self._merge_ranks:
                break
            word = _clip_merge_pair(word, best)
            pairs = _clip_symbol_pairs(word)

        self._bpe_cache[token] = list(word)
        return self._bpe_cache[token]

    def _encode_text(self, text: str) -> list[int]:
        ftfy, regex = _require_clip()
        text = html.unescape(html.unescape(ftfy.fix_text(text))).strip()
        text = regex.sub(r"\s+", " ", text).strip().lower()

        token_ids = []
        for token in self._pattern.findall(text):
            encoded = "".join(self._byte_encoder[b] for b in token.encode("utf-8"))
            token_ids.extend(self._token_ids[symbol] for symbol in self._bpe(encoded))
        return token_ids

    def encode(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(token_ids, padding_mask)`` arrays of shape ``(N, context_length)``.

        ``padding_mask`` uses ``True`` for padding. Sequences carry the start and
        end tokens; longer texts are truncated with the end token kept.
        """
        if isinstance(texts, str):
            raise TypeError("encode expects a sequence of strings, not a single str.")

        start_id = self._token_ids[_CLIP_START_TOKEN]
        end_id = self._token_ids[_CLIP_END_TOKEN]
        token_ids = np.zeros((len(texts), self.context_length), dtype=np.int32)
        padding_mask = np.ones((len(texts), self.context_length), dtype=bool)
        for row, text in enumerate(texts):
            sequence = [start_id, *self._encode_text(text), end_id]
            if len(sequence) > self.context_length:
                sequence = [*sequence[: self.context_length - 1], end_id]
            token_ids[row, : len(sequence)] = sequence
            padding_mask[row, : len(sequence)] = False
        return token_ids, padding_mask
