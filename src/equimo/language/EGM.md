# equimo.language

## Overview

Language modality: a transformer text encoder that turns token ids into a pooled sequence embedding, a CLIP-style causal
text transformer that returns per-token features, the modality-neutral block stack both are built on, a padding-aware
average pool, and two tokenizers that produce the `(token_ids, padding_mask)` pair the encoders expect: a SentencePiece
wrapper that fetches Equimo tokenizer files and a CLIP BPE tokenizer with a packaged vocabulary.

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
- `CausalTextTransformer(dim, mlp_ratio, depth, num_heads, vocab_size, context_length, *, key, out_dim=None, act_layer=jax.nn.gelu)`
  \[models.py:CausalTextTransformer\]: registered as model `"causal_text_transformer"` with `modality="language"`. Both
  `__call__` and `features` take `(token_ids, padding_mask, key=None, inference=True)` of length `context_length` and
  return per-token features `(context_length, out_dim)`, or `(context_length, dim)` without a projection
  [models.py:CausalTextTransformer.features].
- `sam31_text(pretrained=False, inference_mode=True, **kwargs)`: the 32-token, 1024-wide, 24-block text encoder with a
  projection to 256 and exact GELU, built from `_CAUSAL_TEXT_REGISTRY` [models.py:sam31_text]. `pretrained=True` with
  any config override raises `ValueError` [src/equimo/core/factory.py:build_model_variant].
- `global_avg_pooling(inputs, compatible_paddings, pooling_dims, epsilon=1e-8)` [models.py:global_avg_pooling].
- `SentencePieceTokenizer(identifier=None, url=None, path=None, repository=DEFAULT_TOKENIZER_REPOSITORY)` with
  `encode(input_text, max_length=64, lowercase=True) -> (token_ids, padding_mask)` and `download(url) -> Path`
  [tokenizers.py:SentencePieceTokenizer].
- `ClipBpeTokenizer(context_length=32, path=None)` with `encode(texts) -> (token_ids, padding_mask)`, both NumPy arrays
  of shape `(len(texts), context_length)`, `int32` and `bool` [tokenizers.py:ClipBpeTokenizer].
- `DEFAULT_TOKENIZER_REPOSITORY`: Hugging Face `poiretclement/equimo` tokenizer folder pinned to
  `DEFAULT_REPOSITORY_REVISION` [tokenizers.py:DEFAULT_TOKENIZER_REPOSITORY].

## Invariants

- `padding_mask` uses `1` for padding and `0` for valid tokens, both in the tokenizer output and in the encoder
  [tokenizers.py:SentencePieceTokenizer.encode, models.py:TextTransformerEncoder.\_prepare_tokens]. Padding is excluded
  from attention, the FFN mask, and the final pool \[models.py:TextTransformerEncoder.features,
  models.py:TextTransformerEncoder.\_\_call\_\_\].
- `CausalTextTransformer` lets each token attend only to itself and earlier tokens, so changing a later token leaves
  earlier outputs bit-exact [models.py:CausalTextTransformer.\_attention_mask]. Tokens are padded on the right, so the
  causal mask alone keeps padding from reaching real tokens: `padding_mask` is not consulted, and only tells callers
  which output rows to ignore. It uses `True` for padding (`0`/`1` integers also work); padded rows are not zeroed.
- `ClipBpeTokenizer` lower-cases and cleans text with `ftfy` and HTML unescaping, adds the start and end tokens, pads
  with id `0` and truncates with the end token kept [tokenizers.py:ClipBpeTokenizer.encode]. Id `0` is a real CLIP
  token, so only the padding mask identifies padding. `encode` raises `TypeError` for a single `str`.
- Token embeddings are multiplied by `sqrt(dim)` when `scale_sqrt_depth` is set, then sinusoidal position embeddings are
  added in the embedding dtype [models.py:TextTransformerEncoder.\_prepare_tokens]. Odd `dim` zero-pads the last
  channel; `posemb` raises `ValueError` without `seq_len` or with a non-1D `position`
  [models.py:TextTransformerEncoder.posemb].
- `TextTransformerEncoder` stores `temperature` but never reads it, so the value has no effect
  [models.py:TextTransformerEncoder.\_\_init\_\_].
- `global_avg_pooling` accumulates in float32 for `bfloat16` and `float16` inputs and returns the input dtype
  [models.py:global_avg_pooling].
- Block keys are split with `split_for_mode`, so a missing key is allowed only under `inference=True`
  [models.py:TransformerEncoderStack.\_\_call\_\_].
- The tokenizer requires at least one of `identifier`, `url`, or `path` (`ValueError`); `identifier` takes precedence
  and builds `{repository}/{identifier}.model`, and `url` takes precedence over `path`; URLs must use `http://` or
  `https://`; a missing local path raises `FileNotFoundError` \[tokenizers.py:SentencePieceTokenizer.\_\_init\_\_,
  tokenizers.py:SentencePieceTokenizer.download\]. It never adds BOS/EOS tokens and pads or truncates to `max_length`
  with id `0` [tokenizers.py:SentencePieceTokenizer.encode].

## Side Effects

- `SentencePieceTokenizer.download` performs an HTTP GET with `requests` (60 s timeout) when the file is not cached and
  writes it atomically to `~/.cache/equimo/tokenizers/<sha256(url)[:16]>/<basename>`, so each repository revision has
  its own cache entry [tokenizers.py:SentencePieceTokenizer.download]. The constructor reads the model file from disk
  [tokenizers.py:SentencePieceTokenizer.\_\_init\_\_].
- The constructor and `encode` lazily import `tensorflow` and `tensorflow_text`, raising `ImportError` that points to
  the `language` extra when missing [tokenizers.py:\_require_tensorflow_text]. The constructor checks this before any
  download [tokenizers.py:SentencePieceTokenizer.\_\_init\_\_].
- `ClipBpeTokenizer` reads the MIT CLIP vocabulary from `data/bpe_simple_vocab_16e6.txt.gz` or from the gzip-compressed
  file at `path` [tokenizers.py:ClipBpeTokenizer.\_\_init\_\_], and never touches the network. It lazily imports `ftfy`
  and `regex`, raising `ImportError` that points to the `clip` extra when missing [tokenizers.py:\_require_clip].
- Importing `models.py` registers `TextTransformerEncoder` and `CausalTextTransformer` in the global model registry
  [models.py:TextTransformerEncoder, models.py:CausalTextTransformer].

## Verification

- Run: `devenv shell -- uv run pytest tests/test_language.py tests/test_language_tokenizers.py -q`
