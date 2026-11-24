# OpenAI Privacy Filter: Architecture Notes

Every detail in this document was verified directly against the upstream source code and checkpoint files rather than inferred from naming conventions or related models.

| Artifact | Pinned at | Used for |
|---|---|---|
| `github.com/openai/privacy-filter` | commit `f7f00ca7fb869683eb732c010299d901457f19c3` | Model, runtime, and decoding source (`opf/_model`, `opf/_core`, `opf/_common/label_space.py`, `opf/_api.py`) |
| `huggingface.co/openai/privacy-filter`, directory `original/` | revision `7ffa9a043d54d1be65afb281eddf0ffbe629385b` | `config.json`, `model.safetensors`, `viterbi_calibration.json` |
| `original/model.safetensors` | 2,798,984,088 bytes, SHA-256 `9c262cbe68a0c8a50590a648ef8341a2b7d3be1fa11dfb79893fe0b03ce57b5c` | Weights: 107 tensors, 1,399,486,832 parameters |

The total parameter count derived from the module definitions matches the safetensors header exactly, accounting for every tensor in the checkpoint.

- **Sections 1–15**: Specification of the upstream model and inference pipeline.
- **Sections 16–17**: Design and benchmark/parity measurements for `opf-mlx`.
- **Sections 18–19**: Two additions built on top of `opf-mlx` that are not part of the upstream repo: the local detection service (`opf-mlx serve`) and the Pi coding agent extension (`pi-privacy-screen`).

---

## 1. Configuration (`original/config.json`)

| Key | Value | Meaning |
|---|---|---|
| `num_hidden_layers` | 8 | Number of transformer blocks |
| `hidden_size` | 640 | Residual stream width |
| `vocab_size` | 200064 | Embedding table rows (tokenizer IDs span `0..200018`) |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 14 / 2 / 64 | Grouped-query attention (7 query heads per KV head) |
| `bidirectional_left_context` / `_right_context` | 128 / 128 | Attention band width on each side |
| `sliding_window` | 257 | Total band width including self (`128 + 128 + 1`); validated only, never used directly |
| `num_experts` / `experts_per_token` / `intermediate_size` | 128 / 4 / 640 | Mixture-of-Experts (MoE) feed-forward layer |
| `rope_theta`, `rope_scaling_factor`, `rope_ntk_alpha`, `rope_ntk_beta`, `initial_context_length` | 150000, 32, 1, 32, 4096 | YaRN rotary position encoding parameters |
| `num_labels` | 33 | Output token classes |
| `default_n_ctx` / `max_position_embeddings` | 128000 / 131072 | Default GPU window length / initial RoPE table length |
| `encoding` | `o200k_base` | `tiktoken` encoding name |
| `param_dtype` | `bfloat16` | Storage and primary compute dtype of the weights |

Additional hyperparameters are not stored in `config.json` and come from upstream code defaults: `swiglu_limit = 7.0`, `packed_geglu = False`, RMSNorm `eps = 1e-5`, and SwiGLU `alpha = 1.702`.

## 2. Checkpoint tensors

All tensors are stored in `bfloat16` except `sinks`, which is `float32`. `n` is the block index (`0..7`).

| Name | Shape | Role |
|---|---|---|
| `embedding.weight` | `[200064, 640]` | Token embedding table |
| `block.n.attn.norm.scale` | `[640]` | Pre-attention RMSNorm scale |
| `block.n.attn.qkv.weight`, `.bias` | `[1152, 640]`, `[1152]` | Fused Q/K/V projection (`y = x Wᵀ + b`); `1152 = 64 × (14 + 2 + 2)` |
| `block.n.attn.sinks` | `[14]` | Learned attention-sink logit per query head (stored in log2 units) |
| `block.n.attn.out.weight`, `.bias` | `[640, 896]`, `[640]` | Attention output projection |
| `block.n.mlp.norm.scale` | `[640]` | Pre-MLP RMSNorm scale |
| `block.n.mlp.gate.weight`, `.bias` | `[128, 640]`, `[128]` | MoE expert router |
| `block.n.mlp.swiglu.weight`, `.bias` | `[128, 640, 1280]`, `[128, 1280]` | Expert layer 1, layout `[expert, in, out]`: `y = x W[e] + b[e]` |
| `block.n.mlp.out.weight`, `.bias` | `[128, 640, 640]`, `[128, 640]` | Expert layer 2, layout `[expert, in, out]` |
| `norm.scale` | `[640]` | Final RMSNorm scale |
| `unembedding.weight` | `[33, 640]` | Linear token classifier (no bias) |

## 3. Forward pass

For a single window of `T` tokens with batch size 1 (matching the reference runtime):

```text
ids                                                     [1, T]         int32
x = embedding[ids]                                      [1, T, 640]

repeat for 8 blocks:
  h       = rmsnorm(x)
  qkv     = h Wqkvᵀ + bqkv                              [1, T, 1152]
  q, k, v = qkv[..., :896], [..., 896:1024], [..., 1024:]
            q -> [1, T, 14, 64]    k, v -> [1, T, 2, 64]
  q, k    = rope(q), rope(k)            then each multiplied by 64^(-1/4)
  q       -> [1, T, 2, 7, 64]           query head i reads KV head i // 7
  s[t, g, j, w] = q[t, g, j] · k[t + w - 128, g]        [1, T, 2, 7, 257]
            -inf where t + w - 128 is outside [0, T)
  s       = concat(s, sinks[g, j] * ln 2)               [1, T, 2, 7, 258]
  p       = softmax(s)[..., :257]                       sink column dropped
  a[t, g, j] = sum_w p[t, g, j, w] * v[t + w - 128, g]  -> [1, T, 896]
  x       = x + a Woutᵀ + bout

  h       = rmsnorm(x)                                  [T, 640]
  r       = h Wgateᵀ + bgate                            [T, 128]
  e, val  = top4(r);   wts = softmax(val)               [T, 4]
  for each selected expert e:
      u = h W1[e] + b1[e]                               [1280]
      z = min(u[:640], 7) * sigmoid(1.702 * min(u[:640], 7)) * (clip(u[640:], -7, 7) + 1)
      y = z W2[e] + b2[e]                               [640]
  x       = x + sum_e wts[e] * y[e]

x      = rmsnorm(x)
logits = x Wunembedᵀ                                    [1, T, 33]
logp   = log_softmax(logits)                            [T, 33]
```

## 4. Attention behavior

- **Bidirectional banded attention in every layer**: Token `t` attends to tokens `t - 128 .. t + 128` within the current window. There is no causal mask.
- **Attention sink**: A learned per-head scalar is appended to the attention logits before softmax and discarded afterward, allowing the weights over real tokens to sum to less than 1. It is not a real token and has no value vector. Because the sink logit is always finite, the softmax denominator is never empty.
- **Query/key scaling**: Queries and keys are each multiplied by `64^(-1/4)` after RoPE; no separate `1 / sqrt(d)` factor is applied to the dot products.
- **Head grouping**: Query heads `0..6` attend to KV head `0`, and query heads `7..13` attend to KV head `1`. Sink scalars follow the same head ordering.
- **Padding**: The upstream runtime supports masking padded keys in batched inference, though single-text inference never pads.
- **Receptive field**: Eight layers of `±128` give a total receptive field of `±1024` tokens. Because no other operation mixes information across positions, the logits for token `t` depend strictly on tokens within 1,024 positions of it.

## 5. Position encoding

Rotary position embeddings (RoPE) with YaRN frequency blending are applied to queries and keys only. With `d = 64` and pair index `i = 0..31`:

```text
freq[i]   = 150000 ^ (2i / 64)
low, high = 32 * ln(4096 / (32 * 2π)) / ln(150000),  32 * ln(4096 / (1 * 2π)) / ln(150000)
          = 8.0928, 17.3980
mask[i]   = 1 - clip((i - low) / (high - low), 0, 1)
inv[i]    = mask[i] / freq[i] + (1 - mask[i]) / (32 * freq[i])
c         = 0.1 * ln(32) + 1 = 1.346574
cos[p, i] = c * cos(p * inv[i]),   sin[p, i] = c * sin(p * inv[i])
```

Pairs are interleaved: `(x[2i], x[2i+1])` rotates to `(x[2i] cos - x[2i+1] sin, x[2i+1] cos + x[2i] sin)`. Because the YaRN concentration factor `c` is baked into both the `cos` and `sin` tables, it scales queries and keys by `c` and attention scores by `c²`. Pairs `0..8` keep their base frequency, pairs `18..31` are slowed by 32x, and pairs `9..17` interpolate between them. The same tables are shared across all heads. Position `p` starts at `0` in each window. Tables are initialized for 131,072 positions and extended on demand if a longer window is requested.

### Floating-point details in the RoPE tables

The tables are constructed in `float32` on the CPU. Whereas PyTorch computes `freq[i]` with `float32` `pow`, `opf-mlx` computes it in `float64` and rounds once to `float32` so frequency errors do not accumulate linearly with position. On our test machine, PyTorch is off by 1 ULP on 2 of the 32 frequencies, creating a table difference of up to `1.2e-7 * p` (`4e-5` at position 4,096; `1.3e-3` at position 131,071). Because attention only depends on angle differences between tokens at most 128 positions apart, the maximum angular difference is `1.5e-5` radians—invisible in the final logits (Section 17).

However, the angle `p * inv[i]` is computed in `float32` (matching the reference), so rounding becomes coarser at large positions: `1.2e-4` radians between 2,048 and 4,096 radians, and `4e-3` radians near 100,000. As a result, a passage's logits depend slightly on its absolute offset in a window. Shifting the same 2,430 tokens forward by 1,024 to 120,000 positions changes `float32` logits by `0.01–0.11` RMS (up to `5.7` on individual tokens), though no top label changed. Computing angles in `float64` eliminates this shift down to normal round-off (`3e-5`), confirming table rounding is the sole cause. We keep the reference `float32` formula to maintain parity. In `bfloat16`, `cos` and `sin` are rounded to 8 bits of significand before rotation, shifting logits between offsets by ~`0.26` RMS (typical of any two `bfloat16` runs). This explains why windowed inference cannot match a single pass bit-for-bit (Section 16).

## 6. Expert routing

The router projects the normalized residual stream to 128 logits per token, selects the top 4 experts, and computes mixing weights via softmax over only those 4 logits. (The reference divides the weights by 4 and multiplies the summed expert output by 4; since powers of two are exact in binary floating point, those factors cancel.)

Each expert is a two-layer MLP with the clamped SwiGLU activation from Section 3. The gate half occupies the first 640 channels of layer 1's output and the linear half occupies the remaining 640 channels (contiguous halves, not interleaved). The `+1` offset on the linear half is unconditional.

Running 4 of 128 experts per token accounts for the model's "1.5B total, 50M active" headline: exactly 50,571,632 parameters touch a given token (excluding the embedding table).

Top-4 expert selection is the single discrete operation in the network. If a token's 4th and 5th router logits are closer than floating-point rounding noise, two valid implementations can pick different experts, producing logit differences well above ordinary round-off around that token (see Section 17 for a measured example). Because `bfloat16` has larger rounding noise, expert flips occur more often and contribute to overall `bfloat16` drift.

## 7. Normalization

Pre-norm RMSNorm is applied before attention, before the MoE MLP, and once before the classifier:

```text
y = x * rsqrt(mean(x², last axis) + 1e-5) * scale
```

Normalization runs in `float32` and casts back to the residual stream dtype. There are no biases and no mean subtraction.

## 8. Classifier output

A bias-free linear projection maps `640 -> 33` (`[1, T, 33]`), followed by a `float32` log-softmax:

- Class `0` is `O` (background).
- Classes `1..32` are four boundary tags (`B`, `I`, `E`, `S`) for each of eight categories in alphabetical order:
  1. `account_number`
  2. `private_address`
  3. `private_date`
  4. `private_email`
  5. `private_person`
  6. `private_phone`
  7. `private_url`
  8. `secret`
- Class `1 + 4c + b` corresponds to boundary `b` of category `c`.

`config.json` does not store label names; the upstream runtime infers this "v2" label space from `num_labels = 33`, matching `id2label` in the Hugging Face Transformers config.

## 9. Precision

The checkpoint is stored in `bfloat16`, and the reference implementation keeps the residual stream in `bfloat16` while promoting specific operations to `float32`:

| Stage | Reference dtype |
|---|---|
| Embedding, residual stream, QKV & output projections, RoPE rotation | `bfloat16` |
| `q · k` dot products | `bfloat16`, then upcast to `float32` |
| Band mask, attention sink, softmax | `float32` |
| `p · v` weighted sum | `bfloat16` |
| RMSNorm internals | `float32`, cast back to `bfloat16` |
| Router logits, top-k, softmax | `float32` |
| Expert MLPs | `float32` on CPU (weights upcast per use); `bfloat16` in CUDA Triton kernels |
| Output logits | `bfloat16`, upcast to `float32` for log-softmax |

Consequently, the upstream CPU and CUDA paths are not numerically identical. Because only the CPU path runs on macOS, it serves as our parity target.

The reference also includes an off-by-default flag `OPF_ATTN_LOW_PRECISION` (`opf eval --attn-low-precision`) that keeps rotary tables, attention scores, sinks, and softmax in `bfloat16`. `opf-mlx` does not implement this flag.

### Rounding behavior in `bfloat16`

Because `bfloat16` has only 8 bits of significand (including the implicit leading bit), *where* rounding occurs matters as much as the compute dtype. Measured on PyTorch 2.15 CPU against exact arithmetic rounded once:

| Operation in the reference | Rounding behavior |
|---|---|
| `F.linear(x, W, b)` in `bfloat16` | Rounds once, *after* adding bias (not after the matrix product) |
| `einsum` / `matmul` in `bfloat16` | Rounds once, after `float32` accumulation |
| `x1 * cos - x2 * sin` (RoPE) | Rounds three times (each product, then the difference); `cos`/`sin` are cast to `bfloat16` first |
| `q * qk_scale` (tensor × Python float) | Rounds once; scalar stays `float32` during multiplication |
| `scores.float()` | Scores are already rounded to `bfloat16` before entering softmax |
| `W[..., :-1].to(V.dtype)` | Attention weights are rounded to `bfloat16` before multiplying `V` |
| `x + proj`, `x + t` | Rounds once per residual addition |

MLX matches every row except scalar multiplication: MLX casts a Python float down to the array's dtype before multiplying, turning `q * 0.35355339` into `q * 0.35351563` and flipping the last bit in 2.6% of query/key entries. `opf_mlx.model.scale` works around this by multiplying in `float32` and rounding once.

### Disabling reduced-precision `float32` (TF32) on Apple GPUs

On Apple GPUs with neural accelerators, MLX 0.32.4 defaults to reduced-precision `float32` matrix multiplications (~`1e-3` relative error instead of `1e-6`), controlled by `MLX_ENABLE_TF32` (read once on the first GPU matmul). Because the reference disables the equivalent PyTorch mode and runs both the router and expert MLPs in `float32`, `opf_mlx` sets `MLX_ENABLE_TF32=0` on import and probes both `matmul` and `gather_mm` when loading a model (since `gather_mm` uses separate kernels that exhibit ~`8e-4` error under TF32). Loading fails immediately if reduced precision is detected.

## 10. Tokenization and text offsets

- Uses `tiktoken.get_encoding("o200k_base").encode(text, allowed_special="all")`. Literal `<|endoftext|>` and `<|endofprompt|>` strings map to their special token IDs (`199999` and `200018`). Empty input returns no spans.
- Character offsets are reconstructed from token bytes: token bytes are concatenated and decoded as UTF-8 with replacement (`errors="replace"`). A token's character span starts at the character containing its first byte and ends after the character containing its last byte, widening outward if a token splits a multi-byte character. Offsets are Python string indices (Unicode code points).
- If the decoded string differs from the original input (for example, a lone surrogate becomes `U+FFFD`), returned spans index the decoded text, `decoded_mismatch` is set to `true`, and the result includes a `warning`.

## 11. Windowing

Inputs longer than `n_ctx` are split into consecutive, non-overlapping windows of `n_ctx` tokens. Each window runs as an independent forward pass with positions starting at `0`, and log-probabilities are stitched back by absolute token index. Input is never truncated.

If the caller does not specify `n_ctx`, the reference defaults to `4,096` on CPU and `default_n_ctx` (`128,000`) on GPU. Tokens near a window cut lose context from across the boundary. (Section 16 describes `opf-mlx`'s optional overlapping `halo` window policy.)

## 12. Decoding

By default, token labels are decoded with constrained Viterbi over the entire token sequence (across all windows combined) in `float32`, maximizing:

```text
start[y0] + sum_t logp[t, y_t] + sum_t trans[y_(t-1), y_t] + end[y_last]
```

| From | Allowed next | Calibration bias name |
|---|---|---|
| Start of text | `O`, `B-*`, `S-*` | none |
| `O` | `O` | `background_stay` |
| `O` | `B-*`, `S-*` | `background_to_start` |
| `B-x`, `I-x` | `I-x` | `inside_to_continue` |
| `B-x`, `I-x` | `E-x` | `inside_to_end` |
| `E-x`, `S-x` | `O` | `end_to_background` |
| `E-x`, `S-x` | `B-*`, `S-*` | `end_to_start` |
| End of text (last token) | must be `O`, `E-*`, `S-*` | none |

All disallowed transitions receive `-1e9`. The six transition biases come from `viterbi_calibration.json` (`operating_points.default.biases`); in the released checkpoint, all six are `0.0`, so only the structural BIES constraints apply. Ties break toward the lowest class index. If no finite path exists or the decoded path length is invalid, the decoder falls back to per-token argmax (which can also be selected directly).

## 13. Spans and redaction

1. Convert decoded token labels to token spans using a tolerant state machine (which also handles ill-formed sequences produced by argmax).
2. Map token spans to character spans and drop empty spans.
3. Trim leading and trailing whitespace from each span (enabled by default).
4. Optionally drop overlapping spans within the same label (disabled by default).
5. Select a left-to-right non-overlapping set of spans, ordered by start offset ascending, then length descending, then label ascending.
6. Apply `output_mode`: `typed` preserves original labels; `redacted` collapses every label to `redacted` and every placeholder to `<REDACTED>`.
7. Format placeholders by uppercasing the label, replacing non-alphanumeric runs with `_`, and wrapping in angle brackets (e.g., `private_person` -> `<PRIVATE_PERSON>`).
8. Build `redacted_text` by substituting each kept span with its placeholder.

**Result schema (version 1)**: `schema_version`, `summary` (`output_mode`, `span_count`, `by_label` sorted alphabetically, `decoded_mismatch`), `text`, `detected_spans` (`label`, `start`, `end`, `text`, `placeholder`), `redacted_text`, and an optional `warning` field. The reference only emits `warning` on a decode mismatch; `opf-mlx` also warns when an input is split across multiple windows (Section 16).

## 14. Upstream ambiguities and how they were resolved

| Question | Resolution | Evidence |
|---|---|---|
| The Hugging Face repo has two checkpoints. Which is original? | `original/`. Top-level files are a Transformers re-layout with split Q/K/V, renamed tensors, and an extra `score.bias`. | `checkpoint_download.py` fetches `original/*`; safetensors headers compared |
| What does `sliding_window` mean? | `257 = 128 + 128 + 1`, used only as a validation check. (The Transformers config sets 128 per side; `gpt-oss` uses 128 as causal lookback.) | `_validate_encoder_artifact_contract`, `sdpa` |
| Are all layers banded? | Yes (`gpt-oss` bands only even layers). | `AttentionBlock.__init__` |
| Expert layer 2 is `640 × 640`: `[in, out]` or `[out, in]`? | `[in, out]`: `y = x W[e]` (ambiguous from shape alone). | `MLPBlock.__init__`, `_batched_linear_with_parity` |
| Are SwiGLU halves interleaved or contiguous? | Contiguous (`gpt-oss` interleaves them). | `swiglu(packed=False)`; no `packed_geglu` in config; Triton kernel matches |
| Is the `+1` on the linear half conditional? | No, despite an upstream comment mentioning the packed variant. | `swiglu`, Triton kernel line 221 |
| Router softmax: over all 128 experts or top 4? | Over only the top 4 values. | `MLPBlock.forward` |
| Where is `1/sqrt(d)` applied? | Split as `d^(-1/4)` on `q` and `k` individually, after RoPE. | `AttentionBlock.forward` |
| Which RoPE pairing and frequency schedule? | Interleaved pairs, YaRN blend, magnitude `c` baked into both tables. | `RotaryEmbedding`, `_apply_rotary_emb` |
| `dtypes.json` omits norm scales. What dtype are they? | Stored as `bfloat16`, computed in `float32`. | Safetensors header; `RMSNorm` |
| Does the classifier have a bias? | No. | `Transformer.__init__`; safetensors header |
| What precision does the reference use? | `bfloat16` residual stream with `float32` islands; CPU and CUDA differ in the expert MLPs. | Section 9 |
| Does a `bfloat16` linear layer round before adding bias? | No; matmul and bias are accumulated in `float32` and rounded once. | Measured on PyTorch vs. exact arithmetic |
| Is `q * qk_scale` computed with the scale cast to `bfloat16`? | No; PyTorch keeps the Python scalar in `float32`. | Measured: attention outputs match 57–82% of entries if cast, 97–99.9% if kept in `float32` |
| What is the default context length? | Device-dependent: 4,096 on CPU, 128,000 otherwise. | `_resolve_n_ctx` |
| Are long inputs truncated? | Never; they are split into non-overlapping windows. | `example_to_windows` |
| Which tokenizer is used? | `tiktoken` `o200k_base` (not the repo's `tokenizer.json`). | `load_inference_runtime` |
| Where do label names come from? | Inferred from `num_labels`. | `resolve_label_space_from_config` |

## 15. Key invariants tested

- Total parameter count and every tensor's name, shape, and dtype match Section 2.
- Modifying a token more than 1,024 positions away leaves a token's logits unchanged (Section 4).
- Attention weights plus the sink's share sum to 1.
- Outputs are independent of `n_ctx` whenever the input fits within a single window.
- A token's expert MLP outputs are bit-identical regardless of which other tokens are processed alongside it.
- Under every window policy, every token is scored exactly once; under `halo`, every token sees either its full receptive field or the start/end of the text.
- Logits, decoded labels, spans, and redacted text match the PyTorch CPU reference for the same input and `n_ctx`.

All of these invariants are enforced by automated tests: `tests/test_model.py` validates the MLX model against an independent NumPy implementation of this spec, `tests/test_golden.py` checks against recorded reference outputs, and `tests/test_parity.py` runs live comparisons against PyTorch when installed in `reference/`.

---

## 16. The `opf-mlx` implementation

```text
text -> Tokenizer -> token IDs -> plan_windows -> Transformer (MLX) -> logits
     -> log-softmax (NumPy) -> Viterbi / argmax -> token spans
     -> character spans -> placeholders
```

| Module | Responsibility |
|---|---|
| `config`, `labels` | Parse and validate `config.json`; define the BIES label space |
| `weights` | Safetensors reader; validates every tensor name, shape, and dtype (no missing or extra tensors) |
| `model` | Transformer and MoE network using `mlx.core` and `mlx.nn` only |
| `tokenizer` | `tiktoken` encoding and byte-to-character offset mapping |
| `windows` | Windowing policies for inputs longer than `n_ctx` |
| `decoding`, `spans` | Viterbi/argmax decoding, span extraction, and placeholder substitution (NumPy + stdlib) |
| `pipeline` | `PrivacyFilter` orchestrating the full pipeline |
| `checkpoint` | Locating, downloading, and SHA-256 verifying the pinned checkpoint |
| `env`, `precision` | Machine hardware/memory detection and `float32` matmul precision checks |
| `cli` | The `opf-mlx` command-line interface |
| `serve` | Local detection HTTP/Unix-socket server (stdlib only; Section 18) |

### Design choices beyond the reference

- **Device selection**: Defaults to `gpu`, supports `cpu` on request, and never switches silently. Default `n_ctx` is `128,000` on GPU and `4,096` on CPU (matching upstream CUDA/CPU defaults).
- **Precision**: Supports stored `bfloat16` (default) or upcast `float32` (`dtype="float32"`, which uses 2x memory without slowing down inference). No quantization.
- **Overlapping `halo` windows**: The default policy matches the reference's consecutive windows and emits a `warning` / `PrivacyFilterWarning` whenever splitting occurs. The optional `halo` policy pads each window with 1,024 tokens of context on each side (configurable via `halo`), keeping only the interior tokens so every token sees its full receptive field (no boundary warning emitted). It uses the minimum number of windows required: `1` if the input fits, otherwise:
  ```text
  2 + ceil(max(0, tokens - 2 * (n_ctx - halo)) / (n_ctx - 2 * halo))
  ```
  For long documents, this adds ~2x compute at `n_ctx = 4,096` and under 2% at `n_ctx = 128,000`. Initializing `PrivacyFilter` with a window too small to retain at least one interior token between two halos raises an error. Because RoPE positions restart at `0` in each window (Section 5), halo windows do not match a single pass bit-for-bit; Section 17 shows their measured accuracy.
- **Grouped expert matmul**: The PyTorch CPU reference gathers each token's four expert weight matrices and multiplies one token at a time (`torch.bmm`). `opf-mlx` instead sorts `(token, expert)` pairs by expert ID and uses `mx.gather_mm`, which multiplies rows by their assigned expert matrix in place and batches contiguous runs sharing an expert. Each row's output is bit-identical whether computed alone or with other tokens (tested), ensuring no cross-token leakage. On CPU, outputs match PyTorch bit-for-bit across all 9,720 rows of both layers; on GPU, they match to `float32` accuracy (different summation order). The router still runs once across all tokens.
- **Memory checks**: Weight size is checked against available system memory before loading—raising an error if weights cannot fit and warning if headroom is tight.
- **Verified downloads**: Downloads only the three files from the pinned revision, verifying size and SHA-256 for each. Existing files that do not match the pinned release are left untouched unless `--force` is specified.
- **Memory-bounded chunking**: Attention runs in chunks of 128 queries against only their visible keys, and expert MLPs run in chunks of 16,384 tokens, evaluating each chunk before starting the next. Expert chunking does not change a single bit; attention chunking affects at most `float32` summation rounding.

## 17. Parity and performance measurements

All measurements were taken on November 23, 2026 on an Apple M5 Max (128 GiB RAM, macOS 26.6.2, Python 3.13.13, MLX 0.32.4), compared against PyTorch 2.15.0 on the same machine's CPU.

### Parity on recorded test cases

Logits were recorded for 7 texts ranging from 15 to 2,430 tokens (3,349 tokens total; logit values ~5 RMS, up to 28), and full redaction results for 13 cases:

| Weights | Device | Max logit diff | Per-text RMS diff | Tokens with different top label | Identical redaction results |
|---|---|---:|---:|---:|---:|
| `float32` | GPU | `3.7e-5` | `< 6e-6` | 0 / 3,349 | 13 / 13 |
| `float32` | CPU | `3.2e-5` | `< 6e-6` | 0 / 3,349 | 13 / 13 |
| `bfloat16` | GPU | `7.1` | `0.09–0.33` | 1 / 3,349 | 13 / 13 |
| `bfloat16` | CPU | `7.0` | `0.09–0.19` | 1 / 3,349 | 13 / 13 |
| Reference `bfloat16` vs. `float32` | CPU | `7.2` | `0.07–0.45` | 1 / 3,349 | 13 / 13 |

In `bfloat16`, a 1-ULP difference early in the network propagates through subsequent layers—comparable to the gap between PyTorch's own `bfloat16` and `float32` runs. The sole differing token was a near-tie in a 20-token text (`O` at 5.97 vs. `I-private_person` at 5.91 on GPU; 6.09 vs. 5.84 on CPU; 5.94 vs. 6.16 in PyTorch), where PyTorch's own two precisions disagreed on the preceding token. Neither changed the final redaction.

Testing each sublayer in isolation on identical reference inputs confirms close numerical agreement: attention matches PyTorch bit-for-bit in 96.8–100% of entries, expert layers in ≥99.95%, and the classifier in ≥99.86%.

### Parity on a larger sample

To test beyond the 7 recorded texts, we scored 60 additional 200-token passages (12 dense with private data, 48 from the Python Language Reference; 12,000 tokens total). Zero top labels differed from PyTorch in either precision or on either device:

- **`bfloat16`**: Median per-passage RMS logit difference was `0.61` on GPU and `0.57` on CPU (max `0.85`), compared to `0.81` median (max `1.11`) between PyTorch's own `bfloat16` and `float32` runs.
- **`float32`**: All 60 passages agreed within `6e-5` on GPU, and 59 of 60 on CPU. In the 1 remaining passage, a single token had 4th and 5th router scores of `0.110512368` and `0.110512152` in PyTorch; normal round-off caused CPU to pick the other expert, shifting nearby logits by up to `3.2` (Section 6)—one near-tie out of 96,000 expert selections.

### Throughput and latency

Timed end-to-end (`redact` from input text to final result) using `uv run python scripts/benchmark.py` on text from the Python Language Reference (`bfloat16`, fastest of 3 runs):

| Tokens | GPU latency | GPU tokens/s | GPU peak memory | CPU latency | CPU tokens/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 29 ms | 4,500 | 3.2 GiB | 127 ms | 1,000 |
| 2,048 | 63 ms | 32,700 | 3.7 GiB | 869 ms | 2,400 |
| 4,096 | — | — | — | 1.66 s | 2,500 |
| 8,192 | 186 ms | 44,100 | 4.2 GiB | — | — |
| 32,768 | 730 ms | 44,900 | 5.0 GiB | — | — |
| 128,000 | 2.88 s | 44,400 | 5.8 GiB | — | — |

Run-to-run variance is ~5% (up to ~15% at 128 tokens on CPU). Model loading takes ~0.2 s. At 128,000 tokens, the forward pass accounts for ~2.5 s of the 2.88 s total; most of the remainder is Viterbi decoding.

- **`float32` weights**: Faster on short inputs and similar from 8,192 tokens upward—14 ms at 128 tokens, 49 ms at 2,048, and 2.92 s at 128,000 on GPU (5.2–9.0 GiB peak); 65 ms at 128 tokens and 0.8 s at 2,048 on CPU. The extra latency of `bfloat16` on short inputs comes from upcasting each layer's expert weights to `float32` (~5 GB written per pass).
- **Content independence**: Repeating a single paragraph (activating fewer distinct experts) runs within 3% of varied text on GPU and within 8% on CPU.
- **Grouped vs. per-expert matmul**: Running a separate matmul per active expert (the simpler alternative to `mx.gather_mm`, Section 16) requires ~100 small dispatches per layer—taking 94 ms vs. 29 ms at 128 tokens and 142 ms vs. 63 ms at 2,048 tokens on GPU. On CPU, the per-expert loop is slower at 128 tokens (176 ms vs. 124 ms forward pass) but faster on longer inputs (689 ms vs. 858 ms at 2,048 tokens; 1.19 s vs. 1.65 s at 4,096 tokens).
- **Comparison against PyTorch**: On a 2,430-token text, PyTorch takes 11.5 s on CPU (`6.2 s` in `float32`), while `opf-mlx` takes 1.0 s on CPU and 0.07 s on GPU.
- **Impact of TF32 (Section 9)**: Enabling MLX's reduced-precision `float32` mode gives no speedup at 128 tokens, +8% throughput at 2,048 tokens, and +23–25% from 8,192 tokens upward, while shifting `bfloat16` logits by ~`0.6` RMS (zero top-label changes on a 2,048-token sample). We keep it disabled to match the reference.

### Windowing accuracy (`original` vs. `halo`)

Two ~12,000-token documents were scored in a single pass and compared against 4,096-token windows (3 consecutive windows vs. 5 windows with a 1,024-token halo). GPU results below (CPU RMS values were within 10%):

| Document | Weights | Policy | Log-prob RMS diff (max) | Differing labels | Differing spans |
|---|---|---|---:|---:|---:|
| Synthetic records (12,150 tokens, 720 spans) | `float32` | `original` | `0.36` (`17`) | 2 | 3 |
| | `float32` | `halo` | `0.08` (`7.4`) | 0 | 0 |
| | `bfloat16` | `original` | `0.68` (`17`) | 0 | 0 |
| | `bfloat16` | `halo` | `0.67` (`8.9`) | 0 | 0 |
| Python docs (11,681 tokens, 2 spans) | `float32` | `original` | `0.96` (`17.5`) | 0 | 0 |
| | `float32` | `halo` | `0.25` (`9.9`) | 0 | 0 |
| | `bfloat16` | `original` | `1.89` (`19`) | 0 | 0 |
| | `bfloat16` | `halo` | `1.82` (`19`) | 0 | 0 |

With `halo`, no token loses context; the remaining difference comes solely from RoPE positions restarting at `0` in each window (Section 5). The `float32` rows isolate the effect of lost context: consecutive windows altered 3 of 720 spans, whereas `halo` altered none (at 1.6x runtime: 0.45 s vs. 0.28 s). Tokens in the first window beyond reach of the cut (3,072 of 12,150) match the single pass bit-for-bit under both policies.

In `bfloat16`, `halo` does not guarantee single-pass equivalence on borderline entities. On a synthetic third document containing a repeated borderline phrase ("9 April" repeated 40 times across 4,240 tokens, split at 2,500 tokens, flagged as a date in 2 of 40 spots in one pass), spans differing from the 322 single-pass spans were:

| Weights | Device | `original` | `halo` |
|---|---|---:|---:|
| `float32` | GPU, CPU | 5 | 0 |
| `bfloat16` | GPU | 3 | 1 |
| `bfloat16` | CPU | 4 | 6 |

In `float32`, `halo` matched single-pass spans across all three documents (while consecutive windows falsely flagged "9 April" in five paragraphs after the cut). In `bfloat16`, near-ties tip according to position-dependent rounding, making single-pass and halo outputs equally valid within `bfloat16` noise.

---

## 18. The local detection service (`opf-mlx serve`)

`opf-mlx serve` exposes span detection to other local programs over HTTP: text in, spans out. The caller decides how to mask or redact.

```text
GET  /health     -> {"status": "ok", "api": 1, "labels": [...], "device", "dtype", "n_ctx"}
POST /v1/detect  {"texts": ["...", ...], "offsets": "codepoint" | "utf16"}
                 -> {"results": [{"spans": [{"start", "end", "label"}, ...], "warning"?}, ...]}
```

`serve.py` handles transport only and uses nothing outside the Python standard library (`http.server` over `socketserver`). It accepts a detector callable rather than importing the model directly; `cli.py` wires it up to a `PrivacyFilter`.

- **Local-only listening**: Defaults to a Unix socket at `~/.opf/opf-mlx.sock`, created under a restrictive `umask` with mode `0600` so it is never accessible to other users even briefly. `--port` binds only to loopback TCP addresses and rejects non-loopback hosts, because the service has no authentication and relies on OS permissions for access control.
- **Exclusive socket ownership**: Claims a companion lock file *before* loading weights, so if two instances launch simultaneously, the second exits immediately instead of wasting memory loading the model. Stale socket files left by dead processes are cleaned up automatically.
- **Sequential execution**: Processes one request at a time on the serving thread, avoiding concurrent calls into MLX.
- **Zero content logging**: Does not log per-request traffic. Error messages report only the item index and exception type (e.g., `"detection failed for text 2 of 5 (ValueError)"`), and default `http.server` error pages—which echo the request line—are overridden.
- **Strict error responses**: Rejects payloads over `--max-request-bytes` (default 64 MiB) with `413`, non-JSON content types with `415`, malformed payloads with `400`, and per-text detection failures with `500`. Input is never silently truncated. Any text that does not round-trip cleanly through the tokenizer is rejected because its span offsets would refer to a different string.
- **Offset units**: Supports Unicode code points (`"codepoint"`, default) or UTF-16 code units (`"utf16"`, matching JavaScript string indices). Unpaired surrogates map 1-to-1 to `U+FFFD`, keeping offsets aligned with the caller's string.
- **Lifecycle**: `--idle-exit SECONDS` shuts the server down after the specified period without requests (active detection time does not count as idle). `SIGTERM` and `SIGINT` trigger a clean shutdown after the current request finishes.
- **Protocol versioning**: `/health` advertises `"api": 1`; clients expecting a different version must refuse to connect.

### Service latency

Because each text runs in its own forward pass (Section 16), batch latency is the sum of per-text latencies. Measured over the Unix socket on the M5 Max from Section 17 using files from this repo:

- Single texts: 31 ms (500 chars), 46 ms (4,000 chars), 118 ms (20,000 chars), 0.56 s (100,000 chars), 5.9 s (1,000,000 chars).
- Batches of 4,000-char texts: 44–47 ms per text for 64 texts (2.8 s) and 500 texts (23 s).

Because fixed per-pass overhead dominates short texts (Section 17), many short strings take longer than a single long string of the same total length.

---

## 19. The privacy screen for Pi (`pi-privacy-screen/`)

`pi-privacy-screen/` is a zero-dependency TypeScript extension for the Pi coding agent. It masks sensitive values with numbered placeholders before requests leave your machine and restores the real values in incoming replies and tool calls. (See `pi-privacy-screen/README.md` for user documentation; this section covers architecture and rationale.)

### Core invariant

Everything stored, displayed, and executed locally uses real values; everything sent to a remote model provider uses placeholders.

![Diagram showing how a request is screened](privacy-screen-flow.svg)

### Pi hook integration

The extension hooks into Pi's request lifecycle in the following order:

| Hook | Responsibility |
|---|---|
| `session_start` | Load config and session placeholder table; start the detector; begin background pre-scanning of existing session messages |
| `input` | In `mask` mode, verify the detector is reachable before sending; otherwise block submission and restore text to the editor |
| `before_agent_start` | Inject concise placeholder instructions into the system prompt; re-read `.env` files |
| `context_with_system` | Mask every outgoing text; withhold any text that cannot be checked; abort the request if the detector goes down |
| `before_provider_request` | Final sweep over the serialized payload to replace known values in fields not rewritten message-by-message (e.g., tool descriptions) |
| `message_end` | Restore real values in the assistant reply and its tool-call arguments before Pi persists the message |
| `tool_call` | Restore any remaining placeholders right before tool execution; block shell commands that cannot be quoted safely |
| Markdown transformer | Unmask streaming assistant replies and thinking blocks for terminal display |
| `session_before_compact`, `session_before_tree` | Run compaction and branch summaries via Pi's `compact()` / `generateBranchSummary()` with a custom masking `streamFn`, since summary calls bypass standard hooks |

**Fail-closed error handling**: Because Pi continues with the unmodified request if a `context` or `before_provider_request` handler throws, no outgoing handler is allowed to throw. Every exception is caught, unchecked text is withheld, and the request is aborted.

**Hook ordering**: Pi invokes multiple extensions' handlers in directory-listing load order, chaining each return value into the next. Masking runs in `context_with_system`—the final message-level hook—so it sees modifications made by other extensions in `context`. Text injected later in `context_with_system` or `before_provider_request` by extensions loaded after `privacy-screen` is not scanned.

### Module structure

- `vault.ts`: Session placeholder table.
- `patterns.ts`, `spans.ts`: Deterministic key-format rules and ML span normalization.
- `screen.ts`: Core masking/unmasking logic, per-session caching, and cross-turn consistency.
- `messages.ts`: Traversal of Pi messages and serialized provider payloads.
- `shell.ts`: Context-aware shell quoting.
- `detector.ts`: Client and process manager for `opf-mlx serve` (Section 18).
- `dotenv.ts`, `config.ts`: `.env` secret pre-seeding and config parsing.
- `index.ts`: The only module that interacts with Pi's extension API; it has zero runtime imports from Pi so all modules run and test under plain Node.

### Placeholders and persistence

Placeholders take the form `<KIND_n>` where `KIND` is one of `SECRET`, `EMAIL`, `PERSON`, `PHONE`, `ADDRESS`, `ACCOUNT`, `URL`, `DATE`, or `LITERAL`. Each unique value receives one placeholder for the entire session. Mappings are persisted as custom entries in Pi's session file and loaded across *all* branches on open, ensuring a placeholder number can never be reassigned to a different value after `/resume`, `/fork`, or `/tree`.

### Detection sources

Masking combines three sources:

1. **Local ML detector**: Scans for enabled labels (excluding URLs and dates by default). Returned spans are trimmed to exact values so restoration is lossless (e.g., `=support@example.com,` becomes `support@example.com`, `NAME=value` keeps only `value`, and multi-line spans split per line). Implausible spans (such as phone numbers with fewer than 7 digits) are discarded.
2. **Regex patterns**: Matches structured credential formats and assignments to sensitive variable names. Patterns are written to avoid catastrophic backtracking—never restarting greedily at every word, and expressing "20+" length checks as 20 fixed characters followed by a linear tail—keeping 1 MB of adversarial input under 150 ms per rule.
3. **Known values**: Includes all values already in the vault plus secrets pre-seeded from `.env` files. Known values are replaced everywhere regardless of whether the detector flags them again. Values shorter than 3 characters are only replaced at detected spans; word-like values match only on word boundaries ("Lena" will not match inside "Lenard"); long mixed-character keys match anywhere.

Because a value discovered in a later turn must also be masked in earlier turns, masking iterates until the placeholder table reaches a fixed point.

### Origin-based handling

How a piece of text is processed depends on where it came from:

| Origin | Source | Handling |
|---|---|---|
| `input` | User messages, tool results, shell output, files, other extensions' messages | ML detector + regex patterns + known values |
| `system` | System prompt | Regex patterns + known values only |
| `provider` | Prior assistant replies going back to the *same* provider | Sent verbatim as originally received, preserving placeholders and prompt caching |
| `foreign` | Replies from a *different* model/provider, and all summaries | Checked like `input`, except existing screen placeholders keep their meaning |

Remembering assistant replies in their original masked form (keyed in the session file by a digest of the unmasked text) ensures byte-for-byte consistency for the originating provider's prompt cache. Switching models treats earlier replies from the previous provider as `foreign` so they are re-screened before the new provider sees them (including prior thinking blocks, in case Pi ever includes them).

### Preventing placeholder spoofing

Any text in untrusted input (files, tool output, user messages) that already looks like a placeholder (e.g., `<EMAIL_1>`) is escaped to `<LITERAL_n>` on the way out and restored verbatim on the way back. Without this, a file or web page containing `<EMAIL_1>` could trick the model into echoing it back and leaking another session value. Placeholders only resolve to vault entries when written by a model replying to a masked request.

### Conditional unmasking

Replies are only unmasked if the corresponding outgoing request was actually masked. In `report` mode, `off` mode, or when talking to a local model, the model saw the real text—so any placeholder-like text in its reply is a literal quote and is left untouched (both in streaming display and final storage). Once a reply is saved, its values are already restored, so any remaining placeholder-like strings are displayed literally.

### Tool calls and safe shell quoting

Tool-call arguments are unmasked in `message_end` before storage (since edits in `tool_call` are not written back to Pi's session log). Each tool call's restored arguments are recorded per call ID so `tool_call` does not double-unmask values that happen to contain placeholder-like strings.

Because a masked value could contain arbitrary untrusted characters—such as `$(rm -rf ~)`—that an unmasked model would have known to quote, `shell.ts` parses shell syntax far enough to identify each placeholder's exact quoting context and inserts the value strictly as literal data (single-quoted outside quotes; properly escaped inside double quotes, `$'...'`, and heredocs). Whenever safe quoting cannot be guaranteed—inside backticks, `${...}` expansions, arithmetic contexts, immediately after `$` or `\`, where a value could be parsed as a CLI flag or terminate a heredoc, or after unsupported syntax—the command is blocked with an actionable refusal. Refusal messages omit the angle brackets around placeholder names so Pi doesn't re-escape them as `<LITERAL_n>` when feeding the error back to the model.

### Failure modes

| Situation | Behavior in `mask` mode |
|---|---|
| Detector unreachable before sending | Message blocked; input restored to the editor |
| Detector fails mid-request | Request aborted; already-cached texts remain valid |
| Single text fails (over 8 MiB, non-tokenizable, pattern abort) | That text is replaced with a withheld notice; request continues |
| Unexpected error inside extension | Request aborted; nothing sent |
| Summary request cannot be screened or generated | Compaction or branch summary cancelled |
| Shell command cannot be safely quoted | Tool call blocked with a reason |

In `report` mode, nothing is modified in either direction. When `off` or using a local model (loopback URL or listed in `localProviders`), screening is bypassed completely.

### Detector process management

All Pi sessions share a single detached `opf-mlx serve` process launched on demand with `--no-download` and `--idle-exit`. If two sessions start simultaneously, both race for the socket lock; the loser detects the live owner and waits for it to become ready. Failed launches back off for 30 seconds so a missing checkpoint doesn't trigger repeated startup attempts on every keypress. Unexpected processes on the socket path or incompatible API versions are surfaced as errors and never bypassed.

### Runtime overhead and background pre-scanning

Each unique text is sent to `opf-mlx serve` at most once per session and cached by content digest. Re-masking a cached 2 MB conversation takes ~1 ms, plus ~63 ms for the final payload sweep; the ~45 ms per unseen text in the detector is the dominant cost. When resuming a long session (`pi -c` or `/resume`), `session_start` kicks off background pre-scanning while the user types, and the first outgoing request awaits that in-flight promise rather than duplicating work.

### Verification

- **Unit tests**: 177 tests under Node, including executing shell-quoted commands with adversarial values in real `bash` 5, macOS `bash` 3.2 (which parses `$( )` differently), `zsh`, and `sh`.
- **Mutation testing**: `test/mutations.py` injects 183 single-line defects into copies of the source and verifies the unit tests catch all 183.
- **End-to-end tests**: `test/e2e/run.ts` drives real `pi` 1.6.0 in both RPC mode and TUI mode (`tmux`) with isolated home/agent directories, the real `opf-mlx` detector, and a local mock provider recording every request (73 checks across 11 scenarios, verifying zero leaked values, real tool outputs, safe shell quoting of hostile values, and fail-closed behavior when the detector is down).
- **Known gaps**: TypeScript type-checking has not been run yet (no `tsc` in the current environment), branch summaries are only unit-tested, and real-world detection recall has not been measured.

### Design boundaries

Full limitations are listed in `pi-privacy-screen/README.md` under *What it does not cover*. Structurally, the extension governs only the text of outgoing Pi model requests—not what tools do with real values, not encoded/hashed/split representations, not names appearing solely in the system prompt, and not signed provider thinking blocks.
