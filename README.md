# privacy-filter-mlx

[OpenAI Privacy Filter](https://github.com/openai/privacy-filter) for Apple
Silicon, implemented from scratch in [MLX](https://github.com/ml-explore/mlx).
It finds private information in text (names, addresses, emails, phone numbers,
dates, URLs, account numbers, secrets) and replaces it with placeholders,
entirely on your Mac.

```console
$ uv run opf-mlx "Hi, I'm Dana Whitfield. Reach me at dana.whitfield@example.com or 415-555-0199 before 12 May 2025."
Hi, I'm <PRIVATE_PERSON>. Reach me at <PRIVATE_EMAIL> or <PRIVATE_PHONE> before <PRIVATE_DATE>.
```

Two things are here:

- **`opf-mlx`**, the model as a command, a Python class and a local service.
  It runs the original pretrained weights with its own implementation of the
  network, tokenization bookkeeping, decoding and redaction. It does not wrap
  PyTorch, mlx-embeddings, OpenMed or another port. PyTorch appears only in an
  optional environment used to check this implementation against the original.
- **[pi-privacy-screen](pi-privacy-screen/README.md)**, an extension for the
  [Pi coding agent](https://pi.dev) that uses the service to keep secrets and
  personal data out of what Pi sends to a model provider.

> [!IMPORTANT]
> **This reduces what leaves your machine. It does not guarantee it.**
> Detection is a model and a set of patterns, and both miss things. The
> privacy screen covers the requests Pi sends to a model, not what tools do
> with the values they are given. Read [Limitations](#limitations) before you
> rely on either.
>
> This is an independent, unofficial implementation. It is not affiliated
> with or endorsed by OpenAI or by the makers of Pi.

## Setup

Requires an Apple Silicon Mac, Python 3.11 or later, and [uv](https://docs.astral.sh/uv/).

```console
$ git clone https://github.com/ethanolivertroy/privacy-filter-mlx
$ cd privacy-filter-mlx
$ uv sync                 # create the environment (mlx, numpy, tiktoken)
$ uv run opf-mlx env      # what this machine is: chip, memory, macOS, Python, MLX
$ uv run opf-mlx download # fetch the checkpoint (2.8 GB) into ~/.opf/privacy_filter
$ uv run opf-mlx verify   # check the files and every tensor against the architecture
```

`download` fetches three files from `huggingface.co/openai/privacy-filter`
(`original/`, pinned to one revision) and checks each by size and SHA-256. The
first `opf-mlx "..."` downloads them too if they are missing. Set
`OPF_CHECKPOINT` or pass `--checkpoint` to keep the checkpoint elsewhere.

## Command line

```console
$ uv run opf-mlx "some text"                 # one text; prints the redacted text
$ uv run opf-mlx -f notes.txt -f letter.txt  # each file is one text
$ cat lines.txt | uv run opf-mlx             # one text per line
$ uv run opf-mlx                             # a prompt; type /exit to leave
$ uv run opf-mlx --format json "some text"   # spans with labels and offsets
```

The flags follow the original `opf` command. The ones you are likely to want:

| Flag | Meaning |
|---|---|
| `--format text\|json` | redacted text, or the full result as JSON |
| `--output-mode typed\|redacted` | `<PRIVATE_EMAIL>` and friends, or `<REDACTED>` for everything |
| `--decode-mode viterbi\|argmax` | constrained decoding (default) or the best label per token |
| `--device gpu\|cpu` | where the model runs (default `gpu`) |
| `--dtype bfloat16\|float32` | weights as stored (default) or upcast to float32 |
| `--n-ctx N` | tokens per forward pass |
| `--window-policy original\|halo` | how a longer input is split: consecutive windows (default), or windows that overlap so that no token loses context |
| `--halo N` | context tokens per side for `halo` (default: the model's reach, 1,024) |
| `--checkpoint DIR`, `--no-download` | where the weights are; never download |

`uv run opf-mlx redact --help` lists the rest.

## Python

```python
from opf_mlx import PrivacyFilter

redactor = PrivacyFilter()            # loads the model once (0.2 s)
result = redactor.redact("Wire it to account 4532 9981 0042 7716.")

result.redacted_text                  # 'Wire it to account <ACCOUNT_NUMBER>.'
result.detected_spans[0]              # DetectedSpan(label='account_number', start=19, end=38, ...)
result.to_dict()                      # the original's result schema, version 1
```

`PrivacyFilter` takes the same options as the command line as keyword
arguments (`device`, `dtype`, `n_ctx`, `decode_mode`, `output_mode`, ...).

## As a local service

```console
$ uv run opf-mlx serve                       # listens on ~/.opf/opf-mlx.sock until stopped
listening on unix:/Users/you/.opf/opf-mlx.sock
$ curl -s --unix-socket ~/.opf/opf-mlx.sock http://x/v1/detect \
    --json '{"texts": ["Mail lena.fischer@example.com about the invoice.", "Nothing here."]}'
{"results": [{"spans": [{"start": 5, "end": 29, "label": "private_email"}]}, {"spans": []}]}
```

Text in, spans out, for programs that want to decide themselves what to do
with a finding. The model is loaded once and stays loaded.

- It listens on a Unix socket that only your user can open, or with `--port`
  on a loopback address. It never listens on the network.
- `GET /health` says what is loaded. `POST /v1/detect` takes
  `{"texts": [...]}` as `application/json` and answers with one list of spans
  per text. Offsets count code points; `"offsets": "utf16"` counts UTF-16 code
  units, which is what a JavaScript string indexes by.
- It never prints, logs or repeats the text it is sent.
- A request is answered in full or refused with a reason: too large is a 413,
  not a truncated result. One request is handled at a time.
- `--idle-exit SECONDS` stops it after that long without a request. The model
  and decoding flags are those of the command line above.

## A privacy screen for the Pi coding agent

[pi-privacy-screen/](pi-privacy-screen/README.md) is an extension for
[Pi](https://pi.dev) built on that service. Before a request leaves the
machine it replaces private values by numbered placeholders (`<EMAIL_1>`,
`<SECRET_2>`), and it puts the values back into the reply and into the
arguments of tool calls, so that what is stored, shown and run locally is
real and what a provider receives is masked.

![How a request is screened. Pi works on real text. Each request passes the extension, which asks the local detector for private spans, replaces what is found by numbered placeholders, and sends the masked request to the model provider. The reply comes back with placeholders, and the extension puts the values back before Pi stores, shows or runs anything.](docs/privacy-screen-flow.svg)

Its README says how to install it, what it costs, and
[what it does not cover](pi-privacy-screen/README.md#what-it-does-not-cover).

## Limitations

**The model**

- It misses things, and it flags things that are not private. OpenAI's
  [model card](https://huggingface.co/openai/privacy-filter#bias-risks-and-limitations)
  calls it "a redaction and data minimization aid, not an anonymization,
  compliance, or a safety guarantee", and names where it is weaker: uncommon
  names and regional naming conventions, text that is not English or not in
  Latin script, new credential formats, and strings that only look like
  secrets.
- What it looks for is fixed: the eight kinds of span above. Changing that
  takes fine-tuning, which is not in this repository.
- How much private data it finds in real material has not been measured
  here. What was measured is how closely this implementation follows the
  original ([How faithful, how fast](#how-faithful-how-fast)).

**This implementation**

- Apple Silicon only. It has been run on one machine: an M5 Max with macOS
  26.6, Python 3.13 and MLX 0.32.4.
- With the stored bfloat16 weights, the default, a borderline decision can
  fall differently than in the original. `--dtype float32` follows the
  original to 4e-5 on the logits and takes twice the memory.
- A text longer than the context window is split, and tokens next to a cut
  see less context. That is reported, never hidden
  ([What it will not do quietly](#what-it-will-not-do-quietly)).
- Each text is one forward pass. There is no batching across texts, so many
  short texts cost 30 to 45 ms each on the machine above.
- It needs the 2.8 GB checkpoint and, in bfloat16, 3 to 6 GiB of memory
  depending on the length of the text.
- No quantization, training or evaluation tooling.

**The service**

- It has no authentication. Access is the socket file's permissions: any
  program running as your user can ask it. It refuses to listen on anything
  but a Unix socket or a loopback address.
- It answers one request at a time.

**The privacy screen for Pi**

![What the privacy screen covers, and what gets past it. Requests to the model provider pass the screen and leave masked, except for what the detector misses, images, values that are encoded, hashed or split, names that stand only in the system prompt, and text another extension adds afterwards. Tools run with the real values and send whatever they send. Model calls that another extension makes itself are not screened.](docs/privacy-screen-limits.svg)

- It screens what Pi sends in model requests. It does not screen what tools
  do: a command that uploads a file, a `git push`, a tool that fetches a URL
  send what they send, and tools are given the real values. It does not see
  model calls that another extension makes itself.
- What the detector misses goes out. So does a value in another form (base64,
  URL-encoded, hashed, split over lines), an image, and a name that stands
  only in the system prompt.
- It sometimes hides too much: a name that is also a word ("Mark") becomes a
  placeholder wherever it stands alone.
- The session file on disk holds the real conversation, as it always does,
  and now also the table of placeholders.
- It was written for Pi 1.6.0, tested with that version only, and depends on
  how its extension hooks behave. After a Pi update, check that masking still
  happens: paste a made-up key and ask the model how many characters it has.
  A model that was sent a placeholder cannot know.

The full list is in
[the extension's README](pi-privacy-screen/README.md#what-it-does-not-cover).

**How far it has been checked**

- By its own tests ([Tests](#tests) here and in the extension's README), on
  one machine. Nobody else has reviewed it, and it has had no security audit.
- The extension is TypeScript that has never been type-checked.
- It is new and has not been in daily use for any length of time.

## What it will not do quietly

- **Truncate.** An input longer than the context window is split into
  consecutive windows, exactly as the original does, and every token is
  scored. The split is reported in `result.warning` and as a
  `PrivacyFilterWarning`, because tokens next to a cut see less context. On the
  gpu the window is the checkpoint's 128,000 tokens; on the cpu it is 4,096,
  the original's CPU default. `--window-policy halo` overlaps the windows
  instead, so that every token sees all the context the model can use; that
  costs up to twice the work with 4,096-token windows and under 2 % with
  128,000. With `--dtype float32` it gave the result of a single pass in
  every text measured. In bfloat16 a borderline decision can still fall
  either way, as it can between any two bfloat16 runs.
- **Load a wrong checkpoint.** Every tensor's name, shape and dtype is checked
  against the architecture. A missing, extra or misshapen tensor is an error.
- **Change device or precision.** The model runs where you say, in bfloat16 as
  stored unless you ask for float32. MLX's reduced-precision float32 products
  (on by default on GPUs with neural accelerators) are turned off, and loading
  fails if they are found on.
- **Run out of memory halfway.** The weights' size is compared with what this
  machine has before anything is loaded.
- **Overwrite files.** `download` leaves an existing file alone if it is not
  the pinned release, unless you pass `--force`.

## How faithful, how fast

Measured on an Apple M5 Max with MLX 0.32.4 against the original PyTorch
implementation on the same machine; details in
[docs/architecture.md](docs/architecture.md), section 17.

- With float32 weights the logits agree to 4e-5 (on values up to 28), on gpu
  and cpu, for the recorded inputs of up to 2,430 tokens.
- With the stored bfloat16 weights, each sublayer reproduces the original's
  output in 97 to 100 % of its entries when given the same input. Over the
  whole network one differing last bit grows, as it does between the
  original's own bfloat16 and float32 runs, and logits end up 0.09 to 0.33
  apart (rms; the original's two precisions differ by 0.07 to 0.45). The
  highest-scoring label was the same for 3,348 of the 3,349 recorded tokens.
  The exception is a near-tie (5.97 against 5.91), and the original's own two
  precisions disagree on the token next to it. On 60 further passages (12,000
  tokens) no label differed.
- The redaction results were identical in all 13 recorded cases, in both
  precisions, on both devices.
- Text in, result out, on varied text: 128 tokens in 29 ms, 2,048 in 63 ms
  and 128,000 in 2.9 s on the gpu (44,000 tokens/s, 5.8 GiB peak); 2,048 tokens
  in 0.9 s on the cpu. With float32 weights, which take 5.2 GiB instead of
  2.6, 128 tokens take 14 ms. For one 2,430-token text the original takes
  11.5 s on this CPU; opf-mlx takes 1.0 s on the cpu and 0.07 s on the gpu.

`uv run python scripts/benchmark.py` measures your machine (`--device cpu`,
`--dtype float32`, `--sizes`, `--text-file` for a text of your own).

## Tests

```console
$ uv run pytest                               # 515 tests, about 30 s
$ uv run --project reference pytest           # the same plus 82 that run PyTorch side by side
$ uv run --project reference python reference/make_golden.py   # re-record tests/golden/
```

The plain suite needs no PyTorch. It checks the network against an independent
NumPy implementation and against outputs recorded from the original
(`tests/golden/`). Tests that need the 2.8 GB checkpoint are skipped when it is
not installed. `reference/` is a separate uv project holding PyTorch and the
original package pinned to one commit; nothing in `src/` imports from it.

The extension has tests of its own, among them an end-to-end run against the
real `pi`; see [its README](pi-privacy-screen/README.md#tests).

## Layout

```
src/opf_mlx/
  model.py        the network (mlx.core, mlx.nn)
  weights.py      safetensors reader and tensor validation
  config.py       config.json        labels.py     label space
  tokenizer.py    tiktoken + offsets windows.py    splitting long inputs
  decoding.py     Viterbi, argmax    spans.py      spans and placeholders
  pipeline.py     PrivacyFilter      cli.py        the opf-mlx command
  checkpoint.py   download, verify   env.py, precision.py   this machine
  serve.py        the local service (standard library only)
docs/architecture.md   what the model is, read from the original source and checkpoint;
                       how the service and the privacy screen are built
docs/*.svg             the two diagrams of the privacy screen
scripts/benchmark.py   how fast it is on this machine
reference/             optional PyTorch environment and the golden recorder
tests/
pi-privacy-screen/     the extension for the Pi coding agent (TypeScript, its own tests)
```

## License and credits

The model, its weights and the reference implementation are OpenAI's,
published under the Apache License 2.0
([repository](https://github.com/openai/privacy-filter),
[model card](https://huggingface.co/openai/privacy-filter)). The weights are
not part of this repository: `opf-mlx download` fetches them from Hugging
Face.

No license has been chosen yet for the code in this repository.
