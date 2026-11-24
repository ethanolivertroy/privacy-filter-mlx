# privacy-filter-mlx

An independent Apple Silicon implementation of [OpenAI Privacy Filter](https://github.com/openai/privacy-filter), built from scratch in [MLX](https://github.com/ml-explore/mlx). It detects sensitive information in text—names, addresses, emails, phone numbers, dates, URLs, account numbers, and secrets—and replaces it with placeholders, running entirely locally on your Mac.

```console
$ uv run opf-mlx "Hi, I'm Dana Whitfield. Reach me at dana.whitfield@example.com or 415-555-0199 before 12 May 2025."
Hi, I'm <PRIVATE_PERSON>. Reach me at <PRIVATE_EMAIL> or <PRIVATE_PHONE> before <PRIVATE_DATE>.
```

## What's in this repository

- **`opf-mlx`**: A CLI tool, Python library, and local detection service for running OpenAI's pretrained Privacy Filter weights on Apple Silicon. The model architecture, tokenization offset tracking, decoding, and redaction are implemented directly in MLX and NumPy—not wrapped around PyTorch, `mlx-embeddings`, OpenMed, or another port. (PyTorch is only used in an optional `reference/` environment for parity testing against the original implementation.)
- **[`pi-privacy-screen`](pi-privacy-screen/README.md)**: An extension for the [Pi coding agent](https://pi.dev) that uses the local `opf-mlx` service to mask secrets and personal data before Pi sends requests to remote model providers.

> [!IMPORTANT]
> **This reduces what leaves your machine—it is not a guarantee.**
> Detection relies on a machine learning model and pattern rules, both of which can miss sensitive values. The Pi privacy screen covers model requests sent by Pi, not what local tools do with real values when they run. Read [Limitations](#limitations) before relying on either tool.
>
> This is an independent, unofficial implementation. It is not affiliated with or endorsed by OpenAI or the makers of Pi.

## Setup

Requires an Apple Silicon Mac, Python 3.11+, and [uv](https://docs.astral.sh/uv/).

```console
$ git clone https://github.com/ethanolivertroy/privacy-filter-mlx
$ cd privacy-filter-mlx
$ uv sync                 # install dependencies (mlx, numpy, tiktoken)
$ uv run opf-mlx env      # print chip, memory, macOS, Python, and MLX info
$ uv run opf-mlx download # download the 2.8 GB checkpoint to ~/.opf/privacy_filter
$ uv run opf-mlx verify   # verify files and check every tensor against the model architecture
```

`opf-mlx download` fetches three files from the `original/` directory of `huggingface.co/openai/privacy-filter` (pinned to a specific revision) and verifies each file's size and SHA-256 hash. Running `opf-mlx "..."` will also download the checkpoint automatically if it is missing. Set `OPF_CHECKPOINT` or pass `--checkpoint` to store the checkpoint in a different directory.

## Command line

```console
$ uv run opf-mlx "some text"                 # redact a single string
$ uv run opf-mlx -f notes.txt -f letter.txt  # redact files (one example per file)
$ cat lines.txt | uv run opf-mlx             # redact stdin (one example per line)
$ uv run opf-mlx                             # interactive prompt (type /exit to quit)
$ uv run opf-mlx --format json "some text"   # output detected spans, labels, and offsets as JSON
```

CLI flags follow the original `opf` command. Common options:

| Flag | Description |
|---|---|
| `--format text\|json` | Print redacted text or the full structured JSON result |
| `--output-mode typed\|redacted` | Use typed placeholders (`<PRIVATE_EMAIL>`, etc.) or `<REDACTED>` for all spans |
| `--decode-mode viterbi\|argmax` | Constrained Viterbi decoding (default) or independent per-token argmax |
| `--device gpu\|cpu` | Hardware device to run inference on (default: `gpu`) |
| `--dtype bfloat16\|float32` | Run with stored `bfloat16` weights (default) or upcast to `float32` |
| `--n-ctx N` | Maximum tokens processed per forward pass |
| `--window-policy original\|halo` | How inputs longer than `--n-ctx` are split: consecutive windows (default) or overlapping windows so edge tokens keep full context |
| `--halo N` | Context tokens on each side when using `--window-policy halo` (default: 1,024, the model's receptive field) |
| `--checkpoint DIR`, `--no-download` | Custom checkpoint path; disable automatic downloading |

Run `uv run opf-mlx redact --help` for the full list of options.

## Python API

```python
from opf_mlx import PrivacyFilter

redactor = PrivacyFilter()  # loads the model once (~0.2 s)
result = redactor.redact("Wire it to account 4532 9981 0042 7716.")

result.redacted_text        # 'Wire it to account <ACCOUNT_NUMBER>.'
result.detected_spans[0]    # DetectedSpan(label='account_number', start=19, end=38, ...)
result.to_dict()            # matches the original result schema (version 1)
```

`PrivacyFilter` accepts the same options as the CLI as keyword arguments (`device`, `dtype`, `n_ctx`, `decode_mode`, `output_mode`, etc.).

## Local detection service

For other programs that want to detect sensitive spans and handle redaction themselves, `opf-mlx serve` keeps the model loaded in memory behind a local HTTP interface:

```console
$ uv run opf-mlx serve                       # listens on ~/.opf/opf-mlx.sock until stopped
listening on unix:/Users/you/.opf/opf-mlx.sock
$ curl -s --unix-socket ~/.opf/opf-mlx.sock http://x/v1/detect \
    --json '{"texts": ["Mail lena.fischer@example.com about the invoice.", "Nothing here."]}'
{"results": [{"spans": [{"start": 5, "end": 29, "label": "private_email"}]}, {"spans": []}]}
```

Key behaviors:

- **Local-only access**: Listens on a user-only (`0600`) Unix socket by default, or on a loopback address with `--port`. It refuses to bind to external network interfaces.
- **Endpoints**: `GET /health` reports the loaded model configuration. `POST /v1/detect` accepts `{"texts": [...]}` (`application/json`) and returns a list of spans per input string. Offsets default to Unicode code points; pass `"offsets": "utf16"` for UTF-16 code units (matching JavaScript string indexing).
- **No content logging**: Never prints, logs, or echoes back input text.
- **No silent truncation**: Requests are either processed in full or rejected with an HTTP status code (such as `413` if a payload exceeds the size limit). Requests are processed one at a time.
- **Lifecycle and model flags**: Supports `--idle-exit SECONDS` to shut down automatically after inactivity, along with all model and decoding flags from `opf-mlx redact`.

## Privacy screen for the Pi coding agent

[`pi-privacy-screen/`](pi-privacy-screen/README.md) is an extension for [Pi](https://pi.dev) built on top of `opf-mlx serve`. Before any request leaves your Mac, it replaces private values with numbered placeholders (`<EMAIL_1>`, `<SECRET_2>`). When the model responds, it restores the real values in the reply and in tool arguments—so everything stored, displayed, and executed locally stays real while remote providers only see placeholders.

![Diagram showing how a request is screened. Pi works on real text locally. Each outgoing request passes through the extension, which queries the local detector for private spans, replaces found values with numbered placeholders, and sends the masked request to the model provider. When the reply returns with placeholders, the extension restores the real values before Pi stores, displays, or runs anything.](docs/privacy-screen-flow.svg)

See the [extension README](pi-privacy-screen/README.md) for installation instructions, performance overhead, and [what it does not cover](pi-privacy-screen/README.md#what-it-does-not-cover).

## Limitations

### The model

- **False negatives and false positives**: The model can miss sensitive values and flag non-sensitive ones. OpenAI's [model card](https://huggingface.co/openai/privacy-filter#bias-risks-and-limitations) describes it as "a redaction and data minimization aid, not an anonymization, compliance, or a safety guarantee" and notes weaker accuracy on uncommon or regional names, non-English or non-Latin text, unfamiliar credential formats, and strings that merely resemble secrets.
- **Fixed label set**: The model only detects eight built-in categories: `account_number`, `private_address`, `private_date`, `private_email`, `private_person`, `private_phone`, `private_url`, and `secret`. Changing categories requires fine-tuning, which is outside the scope of this repository.
- **Parity tested, not real-world recall**: This repository verifies numerical and behavioral parity with OpenAI's original implementation ([Accuracy and performance](#accuracy-and-performance)), not detection recall on real-world datasets.

### This implementation

- **Hardware and testing environment**: Apple Silicon only. So far it has only been tested on one machine (an M5 Max running macOS 26.6, Python 3.13, and MLX 0.32.4).
- **bfloat16 rounding drift**: With the default `bfloat16` weights, borderline token classifications can occasionally differ from the original PyTorch run. Using `--dtype float32` matches the original logits within `4e-5` at twice the memory cost.
- **Window boundaries**: Inputs longer than the context window are split across passes, so tokens near a boundary see less surrounding context. Splits are always surfaced via warnings ([No silent fallbacks](#no-silent-fallbacks)).
- **No cross-text batching**: Each input string runs in its own forward pass, so many short strings incur ~30–45 ms of per-call overhead each on an M5 Max.
- **Resource requirements**: Requires the 2.8 GB checkpoint on disk and 3–6 GiB of memory in `bfloat16` (depending on input length).
- **Inference only**: No quantization, training, or evaluation tooling is included.

### The local service

- **No authentication**: Access control relies entirely on Unix socket file permissions, meaning any process running under your user account can query it. The server refuses to listen on non-loopback network addresses.
- **Single-threaded execution**: Handles one request at a time.

### The Pi privacy screen

![Diagram showing what the privacy screen covers and what bypasses it. Model requests pass through the screen and leave masked, except for values missed by the detector, images, encoded/hashed/split values, names appearing only in the system prompt, and text added later by other extensions. Local tools run with real values and can send data externally. Direct model calls made by other extensions are not screened.](docs/privacy-screen-limits.svg)

- **Covers model requests, not tool side effects**: Tools receive real values so they can work properly, which means any command that uploads a file, runs `git push`, or fetches a URL sends whatever it sends. Direct model calls made by other extensions are also unscreened.
- **Unmasked blind spots**: Values missed by the detector, transformed values (base64, URL-encoded, hashed, or split across lines), images, and names that appear only in the system prompt are not masked.
- **Over-masking common words**: A detected name that is also a regular word (such as "Mark") will be replaced wherever it appears as a standalone word.
- **Local session storage**: Pi's session file on disk still stores the real conversation, along with the placeholder mapping table.
- **Pi version coupling**: Written and tested only against Pi 1.6.0, relying on its specific extension hooks. After upgrading Pi, verify masking still works by pasting a fake key and asking the model how many characters it has—if masked properly, the model won't know.

See [What it does not cover](pi-privacy-screen/README.md#what-it-does-not-cover) for the complete list.

### Testing and review status

- Verified only by its own automated test suites ([Tests](#tests)) on a single machine.
- No independent code review or security audit has been performed.
- The Pi extension is written in TypeScript and runs directly in Node without having been type-checked yet.
- The project is new and has not seen long-term daily use.

Open decisions and remaining work are tracked in [TODO.md](TODO.md).

## No silent fallbacks

- **No silent truncation**: Inputs longer than the context window are split into consecutive windows (matching the original implementation) so every token is scored. Splits are reported via `result.warning` and `PrivacyFilterWarning`, since tokens near a boundary lose context. Default window sizes are 128,000 tokens on GPU and 4,096 on CPU.
  - Passing `--window-policy halo` overlaps windows so every kept token sees the model's full receptive field (1,024 tokens per side). This adds up to 2x compute at 4,096-token windows and under 2% at 128,000-token windows. With `--dtype float32`, halo windows matched single-pass output across every tested text; in `bfloat16`, borderline tokens can still flip due to floating-point rounding.
- **Strict checkpoint validation**: Every tensor's name, shape, and dtype is validated against the model architecture. Missing, extra, or mismatched tensors raise an error.
- **No silent device or precision changes**: Runs strictly on the requested device and dtype (`bfloat16` by default, or `float32` when requested). MLX's reduced-precision `float32` matmul mode (`MLX_ENABLE_TF32`, enabled by default on Apple GPUs with neural accelerators) is explicitly disabled, and loading fails if reduced precision is still detected.
- **Pre-load memory check**: Weight size is checked against available system memory before loading to avoid running out of memory partway through.
- **No accidental file overwrites**: `opf-mlx download` refuses to overwrite existing files that do not match the pinned release unless `--force` is passed.

## Accuracy and performance

Measured on an Apple M5 Max with MLX 0.32.4 against the original PyTorch implementation on the same machine (full methodology and tables in [docs/architecture.md](docs/architecture.md), Section 17):

- **`float32` parity**: Logits match PyTorch within `4e-5` (on values up to 28) on both GPU and CPU across all recorded inputs (up to 2,430 tokens).
- **`bfloat16` parity**: Given identical inputs, each individual sublayer reproduces PyTorch's output in 97–100% of entries. Across the full network, single-bit differences accumulate—just as they do between PyTorch's own `bfloat16` and `float32` runs—yielding an RMS logit difference of `0.09–0.33` (compared to `0.07–0.45` between PyTorch's two precisions).
  - The top predicted label matched on 3,348 of 3,349 recorded tokens; the single difference was a near-tie (`5.97` vs `5.91`), where PyTorch's own precisions also disagreed on the adjacent token.
  - Across 60 additional passages (12,000 tokens), zero labels differed.
- **Redaction parity**: Final redaction outputs were identical across all 13 recorded test cases, in both precisions and on both devices.
- **Latency and throughput** (end-to-end on varied prose):
  - **GPU (`bfloat16`)**: 29 ms for 128 tokens, 63 ms for 2,048 tokens, and 2.9 s for 128,000 tokens (~44,000 tokens/s, 5.8 GiB peak memory).
  - **CPU (`bfloat16`)**: 0.9 s for 2,048 tokens.
  - **GPU (`float32`)**: 14 ms for 128 tokens (5.2 GiB weights instead of 2.6 GiB).
  - **Comparison vs. PyTorch**: On a 2,430-token document, the original PyTorch implementation takes 11.5 s on CPU, while `opf-mlx` takes 1.0 s on CPU and 0.07 s on GPU.

Run `uv run python scripts/benchmark.py` to benchmark your own Mac (supports `--device cpu`, `--dtype float32`, `--sizes`, and `--text-file`).

## Tests

```console
$ uv run pytest                                                # 515 tests (~30 s, no PyTorch required)
$ uv run --project reference pytest                            # same suite + 82 live PyTorch parity tests
$ uv run --project reference python reference/make_golden.py   # regenerate tests/golden/ fixtures
```

The default test suite does not require PyTorch. It verifies the MLX model against an independent NumPy reference implementation and against outputs recorded from OpenAI's original implementation (`tests/golden/`). Tests requiring the 2.8 GB checkpoint are skipped automatically if it isn't downloaded. `reference/` is an isolated `uv` project containing PyTorch and the pinned upstream package; nothing under `src/` imports from it.

The Pi extension has its own test suite (including end-to-end tests against the real `pi` CLI); see [`pi-privacy-screen/README.md#tests`](pi-privacy-screen/README.md#tests).

## Repository layout

```text
src/opf_mlx/
  model.py         Transformer and MoE network (mlx.core, mlx.nn)
  weights.py       Safetensors loader and tensor validation
  config.py        config.json parsing and validation
  labels.py        BIES label space definitions
  tokenizer.py     tiktoken wrapper and byte-to-character offset mapping
  windows.py       Window splitting policies (original and halo)
  decoding.py      Constrained Viterbi and argmax decoders
  spans.py         Span extraction, trimming, and placeholder formatting
  pipeline.py      High-level PrivacyFilter class
  cli.py           opf-mlx command-line interface
  checkpoint.py    Checkpoint resolution, downloading, and SHA-256 verification
  env.py           Hardware, memory, and runtime detection
  precision.py     Float32 matmul precision checks (TF32 guard)
  serve.py         Local detection HTTP/Unix-socket server (stdlib only)

docs/
  architecture.md            Detailed specification of the model, service, and Pi extension
  privacy-screen-flow.svg    Request/response masking flow diagram
  privacy-screen-limits.svg  Scope and blind-spots diagram

scripts/
  benchmark.py     Local latency and memory benchmark

reference/         Optional PyTorch environment and golden fixture generator
tests/             Unit, golden, and parity tests
pi-privacy-screen/ TypeScript extension for the Pi coding agent (with its own tests)
```

## License and credits

All code in this repository is released under the [MIT License](LICENSE).

The model architecture, pretrained weights, and original reference implementation are by OpenAI under the Apache License 2.0 ([GitHub repository](https://github.com/openai/privacy-filter), [Hugging Face model card](https://huggingface.co/openai/privacy-filter)). Pretrained weights are not committed to this repository; `opf-mlx download` fetches them directly from Hugging Face. `tests/golden/` contains reference outputs recorded from OpenAI's implementation for testing.
