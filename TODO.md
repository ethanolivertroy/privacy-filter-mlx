# To do

Open decisions and open work. What the project does not do is in the
READMEs: [Limitations](README.md#limitations) and
[What it does not cover](pi-privacy-screen/README.md#what-it-does-not-cover).

## Decide whether the privacy screen's defaults stay

The extension's defaults were set when it was built. None has been looked at
again after real use, and each trades protection against getting in the way.

- [ ] **It masks from its first start.** The other way: start in `report`
  mode, which counts what it would mask and changes nothing, and switch to
  `mask` once the findings look right. Possible today with
  `"mode": "report"` in the settings, or `/privacy report` for one session.
- [ ] **A message is not sent while the detector cannot be reached.** The
  other way: send it unscreened, with a warning. There is no setting for
  that; today the way around is `/privacy off`, by hand.
- [ ] **URLs and dates are not masked.** Secrets, e-mail addresses, phone
  numbers, names of people, addresses and account numbers are. The other
  way: add `private_url` and `private_date` to `"labels"` in the settings.
  They are left out because code is full of both.
- [ ] **Requests to a model on this machine are not screened.** The other
  way: `"screenLocalModels": true` in the settings.
- [ ] **A placeholder that someone typed is not a placeholder.**
  `<SECRET_1>` in a message, a file or a tool result goes out as
  `<LITERAL_n>` and comes back as the text it was, so a file or a web page
  cannot call up a stored value. Nor can you, by typing one. There is no
  setting for this.

## A faster path through the expert layers, as an option

The detector takes 35 to 45 ms for a text of up to 4,000 characters, and the
privacy screen pays that for every text it has not seen before. Part of it
is the same for every pass, however short the text: with bfloat16 weights,
each pass converts the expert weights of every layer to float32, about 5 GB
written. That follows the original's CPU path, the only one whose output
could be recorded on a Mac to test against.

The original's CUDA path multiplies the expert layers in bfloat16. Doing the
same here was tried once, on the gpu of an M5 Max:

| Tokens | float32 experts, as now | bfloat16 experts |
|---:|---:|---:|
| 128 | 29 ms | 9 ms |
| 2,048 | 58 ms | 27 ms |
| 8,192 | 173 ms | 94 ms |

Both columns are from that trial, whose code was not kept. Against the
original's CPU output on 60 passages the logits differed by 0.64 rms at the
median instead of 0.61, and no label changed.

- [ ] Build it as an option that is off by default: a flag of `opf-mlx`,
  and so of `opf-mlx serve`, and an argument of `PrivacyFilter`. The
  privacy screen can then ask for it through `server.command`.
- [ ] Test it against the recorded outputs with a tolerance, since outputs
  of its own cannot be recorded here, and keep the test that a row's result
  is the same alone or in company.
- [ ] Measure it again with the code in the repository, and measure what it
  does to the service's time per text, which the trial did not.
- [ ] Say in the README which path gives what.

Until then float32 weights avoid the conversion another way, at about twice
the memory: `opf-mlx serve --dtype float32`. The benchmark has 14 ms instead
of 29 at 128 tokens for the model alone. This has not been tried with the
privacy screen.

## Type-check the extension

`pi-privacy-screen` is TypeScript that no compiler has checked. Node runs it
by stripping the types without checking them, and there is no compiler in
this project's tooling. A type error that fails no test goes unnoticed.

- [ ] Check the extension by itself: `typescript` and `@types/node` as
  development dependencies (its first dependencies of any kind), a
  `tsconfig.json` for code that runs with its types stripped (`noEmit`,
  `allowImportingTsExtensions`, `erasableSyntaxOnly`, `strict`), a
  `typecheck` script, and fixes for what it reports.
- [ ] Check it against Pi's types. The extension describes the part of Pi it
  uses in an interface of its own (`Pi` in `src/index.ts`), with every event
  typed `any`, so the first step cannot show a hook that is used differently
  from how Pi declares it. Pi ships declarations (`ExtensionAPI` from
  `@earendil-works/pi-coding-agent`). A type-only import of them is stripped
  like the rest, so the extension would still need nothing installed to run.
