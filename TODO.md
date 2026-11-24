# To Do

Open design decisions and remaining work. Known project limitations are documented in the READMEs: [Limitations](README.md#limitations) and [What it does not cover](pi-privacy-screen/README.md#what-it-does-not-cover).

## Review the privacy screen's default settings

The Pi extension's defaults were chosen during initial development and have not yet been revisited after extended daily use. Each involves a trade-off between stronger protection and convenience:

- [ ] **Mask on first start**: Currently enabled by default. The alternative is starting in `report` mode (count findings without modifying requests) until the user confirms the output looks right. Already configurable via `"mode": "report"` or `/privacy report`.
- [ ] **Fail closed when the detector is down**: Currently blocks outgoing messages if the detector cannot be reached. The alternative is sending unscreened with a warning. There is no setting for fail-open today; users must run `/privacy off` manually.
- [ ] **Exclude URLs and dates by default**: Secrets, emails, phone numbers, person names, addresses, and account numbers are masked, while `private_url` and `private_date` are off by default because codebases contain many benign URLs and dates. Users can opt in by adding them to `"labels"`.
- [ ] **Skip local models by default**: Requests to local models are not screened unless `"screenLocalModels": true` is set.
- [ ] **Escape user-typed placeholders**: Typing `<SECRET_1>` in a message (or encountering it in a file or tool result) sends `<LITERAL_n>` and restores it verbatim. This prevents untrusted files or web pages from extracting stored secrets, though it also prevents users from referencing placeholders manually. Not currently configurable.

## Optional faster `bfloat16` path for expert layers

`opf-mlx serve` takes 35–45 ms for texts up to 4,000 characters, which the privacy screen pays once per unseen text. A large portion of that fixed cost comes from upcasting every layer's `bfloat16` expert weights to `float32` on each forward pass (~5 GB written per pass). This matches OpenAI's PyTorch CPU path—the only reference path we could run locally on macOS to verify parity against.

OpenAI's CUDA path instead multiplies the expert layers directly in `bfloat16`. A prototype of that path on an M5 Max GPU showed significant speedups:

| Tokens | `float32` experts (current) | `bfloat16` experts (prototype) |
|---:|---:|---:|
| 128 | 29 ms | 9 ms |
| 2,048 | 58 ms | 27 ms |
| 8,192 | 173 ms | 94 ms |

(Both columns were measured during the same prototype run, which was not committed. Across 60 passages, median RMS logit difference against PyTorch CPU was `0.64` vs. `0.61` currently, with zero label changes.)

- [ ] Add this as an opt-in flag on `opf-mlx` (and `opf-mlx serve`) and a corresponding parameter on `PrivacyFilter`, so `pi-privacy-screen` can opt in via `server.command`.
- [ ] Test it against the recorded golden outputs within an appropriate tolerance (since CUDA reference outputs cannot be recorded on macOS), and keep the invariance test verifying a token's expert output does not depend on other tokens in the batch.
- [ ] Re-benchmark the committed implementation, including end-to-end `opf-mlx serve` latency per text.
- [ ] Document the trade-offs between the two paths in `README.md`.

In the meantime, running `opf-mlx serve --dtype float32` avoids the per-pass weight conversion at ~2x memory usage (14 ms vs. 29 ms at 128 tokens in standalone benchmarks; not yet tested with `pi-privacy-screen`).

## Add TypeScript type-checking for `pi-privacy-screen`

`pi-privacy-screen` is written in TypeScript, which Node executes directly by stripping type annotations without validating them. Because no TypeScript compiler is currently configured, type errors that don't break a test go unnoticed.

- [ ] Add standalone type-checking: install `typescript` and `@types/node` as dev dependencies (the extension's first dependencies), add a `tsconfig.json` compatible with Node type-stripping (`noEmit`, `allowImportingTsExtensions`, `erasableSyntaxOnly`, `strict`), add a `typecheck` npm script, and fix any reported errors.
- [ ] Type-check against Pi's real declarations: replace the hand-written `Pi` interface in `src/index.ts` (which currently types hook events as `any`) with type-only imports of `ExtensionAPI` from `@earendil-works/pi-coding-agent`. Because type-only imports are stripped at runtime, the extension will still run with zero runtime dependencies.
