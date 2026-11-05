# equimo.language

## Overview

Language modality: a transformer text encoder that turns token ids into a pooled sequence embedding, the
modality-neutral block stack it is built on, a padding-aware average pool, and a SentencePiece tokenizer wrapper that
fetches Equimo tokenizer files and produces the `(token_ids, padding_mask)` pair the encoder expects.

## Interface

- `TextTransformerEncoder(dim, mlp_ratio, depth, num_heads, vocab_size, *, key, scale_sqrt_depth=True, act_layer=jax.nn.gelu, temperature=-1.0)`
  \[models.py:TextTransformerEncoder\]: registered as model `"text_transformer_encoder"` with `modality="language"`.
  - `__call__(ids, padding_mask, key=None, inference=None)` returns a pooled `(dim,)` embedding
    [models.py:TextTransformerEncoder.\_\_call\_\_].
  - `features(...)` returns final-LayerNorm tokens `(seqlen, dim)` [models.py:TextTransformerEncoder.features].
  - `intermediate_features(..., indices=None, n_last_blocks=None)` returns selected block outputs
    [models.py:TextTransformerEncoder.intermediate_features].
  - `posemb(min_timescale=1, max_timescale=10000, seq_len=None, position=None)` returns sinusoidal embeddings
    `(seq_len, dim)` [models.py:TextTransformerEncoder.posemb].
- `TransformerEncoderStack(dim, depth, num_heads, mlp_ratio, *, key, act_layer=jax.nn.gelu)` called as
  `(x, key=None, inference=None, mask=None, *, attn_mask=None, ffn_mask=None)`, plus `intermediate_features`; a tuple of
  core `AttentionBlock`s [models.py:TransformerEncoderStack].
- `global_avg_pooling(inputs, compatible_paddings, pooling_dims, epsilon=1e-8)` [models.py:global_avg_pooling].
- `SentencePieceTokenizer(identifier=None, url=None, path=None, repository=DEFAULT_TOKENIZER_REPOSITORY)` with
  `encode(input_text, max_length=64, lowercase=True) -> (token_ids, padding_mask)` and `download(url) -> Path`
  [tokenizers.py:SentencePieceTokenizer].
- `DEFAULT_TOKENIZER_REPOSITORY`: Hugging Face `poiretclement/equimo` tokenizer folder pinned to
  `DEFAULT_REPOSITORY_REVISION` [tokenizers.py:DEFAULT_TOKENIZER_REPOSITORY].

## Invariants

- `padding_mask` uses `1` for padding and `0` for valid tokens, both in the tokenizer output and in the encoder
  [tokenizers.py:SentencePieceTokenizer.encode, models.py:TextTransformerEncoder.\_prepare_tokens]. Padding is excluded
  from attention, the FFN mask, and the final pool \[models.py:TextTransformerEncoder.features,
  models.py:TextTransformerEncoder.\_\_call\_\_\].
- Token embeddings are multiplied by `sqrt(dim)` when `scale_sqrt_depth` is set, then sinusoidal position embeddings are
  added in the embedding dtype [models.py:TextTransformerEncoder.\_prepare_tokens]. Odd `dim` zero-pads the last
  channel; `posemb` raises `ValueError` without `seq_len` or with a non-1D `position`
  [models.py:TextTransformerEncoder.posemb].
- `global_avg_pooling` accumulates in float32 for `bfloat16` and `float16` inputs and returns the input dtype
  [models.py:global_avg_pooling].
- Block keys are split with `split_for_mode`, so a missing key is allowed only under `inference=True`
  [models.py:TransformerEncoderStack.\_\_call\_\_].
- The tokenizer requires at least one of `identifier`, `url`, or `path` (`ValueError`); `identifier` takes precedence
  and builds `{repository}/{identifier}.model`; URLs must use `http://` or `https://`; a missing local path raises
  `FileNotFoundError` \[tokenizers.py:SentencePieceTokenizer.\_\_init\_\_,
  tokenizers.py:SentencePieceTokenizer.download\]. It never adds BOS/EOS tokens and pads or truncates to `max_length`
  with id `0` [tokenizers.py:SentencePieceTokenizer.encode].

## Side Effects

- `SentencePieceTokenizer.download` performs an HTTP GET with `requests` when the file is not cached and writes it to
  `~/.cache/equimo/tokenizers/` (creating the directory) [tokenizers.py:SentencePieceTokenizer.download]. The cache is
  keyed only by the URL's file basename, so a cached file with the same name from another repository, revision, or host
  is reused without a download. The constructor reads the model file from disk
  [tokenizers.py:SentencePieceTokenizer.\_\_init\_\_].
- The constructor and `encode` lazily import `tensorflow` and `tensorflow_text`, raising `ImportError` that points to
  the `language` extra when missing [tokenizers.py:\_require_tensorflow_text]. The constructor checks this after any
  download, so without the extra it fails after the HTTP GET and cache write
  [tokenizers.py:SentencePieceTokenizer.\_\_init\_\_].
- Importing `models.py` registers `TextTransformerEncoder` in the global model registry
  [models.py:TextTransformerEncoder].

## Verification

- Run: `devenv shell -- uv run pytest tests/test_language.py tests/test_language_tokenizers.py -q`
