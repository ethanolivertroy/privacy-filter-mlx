"""Mutation check for the privacy screen: do the unit tests notice a defect?

    python3 test/mutations.py [part of a name, or a file name such as screen.ts]

Each entry below is one deliberate defect: a piece of a source file and what to put in its place. For each
one the package is copied to a temporary directory, the defect is made in the copy, and the unit tests that
cover that file are run. They have to fail. A defect that the tests do not notice ("survived") is a behaviour
nothing checks; one whose text is no longer found ("could not be applied") has to be brought up to date with
the source. The files of the package itself are never changed.

Exit status 0 only if every defect was noticed. MUT_VERBOSE=1 also lists, for each one, a test that caught it.
"""
import os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

package = Path(__file__).resolve().parent.parent
scratch = Path(tempfile.mkdtemp(prefix="pi-privacy-screen-mutations-"))
only = sys.argv[1] if len(sys.argv) > 1 else ""

TESTS = {
    "screen.ts": ["screen", "messages", "extension"],
    "messages.ts": ["messages", "extension"],
    "index.ts": ["extension"],
    "config.ts": ["config", "extension"],
    "detector.ts": ["detector"],
    "dotenv.ts": ["dotenv", "extension"],
    "patterns.ts": ["patterns", "screen", "dotenv"],
    "spans.ts": ["spans", "screen"],
    "shell.ts": ["shell", "shell-fuzz", "messages"],
    "vault.ts": ["vault", "screen", "extension"],
}

M = []
def m(name, file, old, new):
    M.append((name, file, old, new))

# ---------------------------------------------------------------- screen.ts
m("one round only", "screen.ts", "const MAX_ROUNDS = 8;", "const MAX_ROUNDS = 1;")
m("short values spread", "screen.ts", "const SPREAD_FROM = 3;", "const SPREAD_FROM = 1;")
m("system prompt goes to the detector", "screen.ts", 'detect: origin !== "system" };', "detect: true };")
m("kept foreign text is not detected", "screen.ts", 'mode: "kept", detect: foreign };', 'mode: "kept", detect: false };')
m("foreign text is trusted like the provider's", "screen.ts",
  'return foreign ? { key: digest("input", text), text, mode: "input", detect: true } : { key: memory, text, mode: "provider", detect: false };',
  'return { key: memory, text, mode: "provider", detect: false };')
m("a placeholder in what was sent or read is a reference", "screen.ts", 'const reference = mode === "kept" ? this.vault.lookup(match[0]) : undefined;',
  'const reference = mode === "kept" || mode === "input" ? this.vault.lookup(match[0]) : undefined;')
m("a placeholder the model was given is no reference", "screen.ts", 'const reference = mode === "kept" ? this.vault.lookup(match[0]) : undefined;',
  'const reference = undefined as Entry | undefined;')
m("a placeholder in what was sent or read is kept as it is", "screen.ts", 'else if (mode === "kept" || mode === "payload") claim(start, end, match[0]);',
  'else if (mode === "kept" || mode === "payload" || mode === "input") claim(start, end, match[0]);')
m("an unknown placeholder in kept text is a literal", "screen.ts", 'else if (mode === "kept" || mode === "payload") claim(start, end, match[0]);',
  'else if (mode === "payload") claim(start, end, match[0]);')
m("known values inside words", "screen.ts", "if (!free(at, end) || !(known.anywhere || standsAlone(text, at, end))) continue;", "if (!free(at, end)) continue;")
m("keys only where they stand alone", "screen.ts", "if (!free(at, end) || !(known.anywhere || standsAlone(text, at, end))) continue;",
  "if (!free(at, end) || !standsAlone(text, at, end)) continue;")
m("a fragment is left sticking out", "screen.ts", "if (holders.size === 0 || covered) continue;", "continue;")
m("a revealed detection is masked", "screen.ts", "			if (entry.revealed) continue;\n", "")
m("revealed values stay known", "screen.ts", "if (!entry.revealed && entry.value.length >= SPREAD_FROM)", "if (entry.value.length >= SPREAD_FROM)")
m("a revealed reference stays a placeholder", "screen.ts", "const shown = (entry: Entry) => (entry.revealed ? entry.value : entry.token);", "const shown = (entry: Entry) => entry.token;")
m("unchanged text is never remembered", "screen.ts", "if (real === masked && !tokenPattern().test(masked)) return;", "if (real === masked) return;")
m("remember does not report", "screen.ts", "		this.onRemember?.(key, kept);\n", "")
m("kept checks are shared between providers", "screen.ts", '`${foreign ? "kept+" : "kept:"}${digest("provider", written)}`', '`kept:${digest("provider", written)}`')
m("unmask does nothing", "screen.ts", "return text.replace(tokenPattern(), (token) => this.vault.lookup(token)?.value ?? token);", "return text;")
m("the last look misses JSON forms", "screen.ts", 'if (mode === "payload") {\n				const inJson', 'if (false) {\n				const inJson')
m("the last look counts its own placeholders", "screen.ts", "return { text: out + text.slice(at), hits: made.length - kept };", "return { text: out + text.slice(at), hits: made.length };")
m("a failed text is sent unchecked", "screen.ts", "const found = answer instanceof Error ? answer : search(wait.way.text, answer);",
  "const found = answer instanceof Error ? search(wait.way.text, []) : search(wait.way.text, answer);")
m("detector down fails open", "screen.ts", "				answers = undefined;\n				fail(asError(error), true,", "				answers = asked.map(() => []);\n				fail(asError(error), true,")
m("down is not reported as down", "screen.ts", "fail(asError(error), true, asked.reduce", "fail(asError(error), false, asked.reduce")
m("a wrong count of answers is used", "screen.ts", '				if (answers.length !== asked.length) throw new Error("the detector answered for a different number of texts");\n', "")
m("checks are forgotten at once", "screen.ts", "if (check.pass < this.pass - 1) this.checked.delete(key);", "if (check.pass <= this.pass) this.checked.delete(key);")
m("checks are never forgotten", "screen.ts", "if (check.pass < this.pass - 1) this.checked.delete(key);", "")
m("seeds do not bump the epoch", "screen.ts", "			if (value.length >= SPREAD_FROM) this.seeds.add(value);\n		}\n		this.epoch++;", "			if (value.length >= SPREAD_FROM) this.seeds.add(value);\n		}")
m("reveal does not bump the epoch", "screen.ts", "		if (!this.vault.setRevealed(token, revealed)) return false;\n		this.epoch++;", "		if (!this.vault.setRevealed(token, revealed)) return false;")
m("every value is key-like", "screen.ts", "return value.length >= 12 && /[A-Za-z]/.test(value) && /[0-9]/.test(value) && !/\\s/.test(value);", "return true;")
m("overlaps are not merged", "screen.ts", "if (last && range.start < last.end) last.end = Math.max(last.end, range.end);", "if (false) last.end = Math.max(last.end, range.end);")
m("pattern errors escape mask", "screen.ts", "				return new Error(`a text could not be searched for secrets (${asError(error).message})`);", "				throw error;")
m("patterns are skipped for tool text", "screen.ts", "return spans ? [...findPatterns(text), ...cleanSpans(text, spans, this.labels)] : findPatterns(text);",
  "return spans ? cleanSpans(text, spans, this.labels) : findPatterns(text);")
m("patterns are skipped for the system prompt", "screen.ts", 'const found = way.mode === "system" ? search(way.text, undefined) : [];', "const found: Found[] | Error = [];")
m("seeds are not replaced", "screen.ts", "			if (!this.vault.find(value)) list.push({ value, anywhere: keyLike(value) });\n", "")
m("a detection touching a reference is kept", "screen.ts", "const detected = merged([...found, ...literals]).filter((range) => free(range.start, range.end));", "const detected = merged([...found, ...literals]);")

# -------------------------------------------------------------- messages.ts
m("every reply is trusted", "messages.ts", 'const origin: Origin = typeof message.provider === "string" && message.provider === target.provider ? "provider" : "foreign";', 'const origin: Origin = "provider";')
m("a reply without a provider is trusted", "messages.ts", 'typeof message.provider === "string" && message.provider === target.provider ? "provider" : "foreign";', 'message.provider === target.provider ? "provider" : "foreign";')
m("summaries are trusted", "messages.ts", 'slot(copy.summary, "foreign", (text) => (copy.summary = text));', 'slot(copy.summary, "provider", (text) => (copy.summary = text));')
m("excluded shell runs are screened", "messages.ts", "				if (message.excludeFromContext) break;\n", "")
m("the system prompt is skipped", "messages.ts", '				content(copy, "content", "system");\n', "")
m("user text is not screened", "messages.ts", '			case "user":\n			case "toolResult":', '			case "toolResult":')
m("tool results are not screened", "messages.ts", '			case "user":\n			case "toolResult":\n			case "custom": // from an extension', '			case "user":\n			case "custom": // from an extension')
m("a typed command is not screened", "messages.ts", '				slot(copy.command, "input", (text) => (copy.command = text));\n', "")
m("shell output is not screened", "messages.ts", '				slot(copy.output, "input", (text) => (copy.output = text));\n', "")
m("sections are skipped", "messages.ts", 'for (const name of Object.keys(sections)) slot(sections[name], "system", (text) => (sections[name] = text));', "")
m("unchecked text is sent as it is", "messages.ts", "put(texts[index] ?? WITHHELD);", "if (texts[index] !== undefined) put(texts[index]!);")
m("tool call arguments are not screened", "messages.ts", "eachString(call.arguments, (text, put) => slot(text, origin, put));", "")
m("the transcript changes the stored message", "messages.ts", "const call = { ...block, arguments: structuredClone(block.arguments) as Holder };", "const call = { ...block, arguments: block.arguments as Holder };")
m("a refused command is changed anyway", "messages.ts", "	if (refusal !== undefined) return refusal;\n	for (const change of changes) {", "	for (const change of changes) {")
m("a refusal is not reported", "messages.ts", "	if (refusal !== undefined) return refusal;\n	for (const change of changes) {", "	if (refusal !== undefined) return undefined;\n	for (const change of changes) {")
m("shell commands get values unquoted", "messages.ts", "if (shell.includes(path[0]!)) {", "if (false) {")
m("values are put in twice", "messages.ts", "		if (options.settled?.get(where) === text) return;\n", "")
m("code tools get values", "messages.ts", "if (!rules.code.includes(name)) {", "if (true) {")
m("tool calls are not remembered", "messages.ts", "			eachString(written, (text) => screen.remember(text, text));\n", "")
m("reply text is not remembered", "messages.ts", "		screen.remember(real, block.text);\n		screen.remember(block.text, block.text);", "")
m("input changes are not remembered", "messages.ts", "		screen.remember(change.real, change.masked);\n", "")
m("the last look enters signatures", "messages.ts", "			if (typeof child === \"string\" && OPAQUE_KEYS.has(key)) continue;\n", "")
m("the last look enters thinking", "messages.ts", 'if (!isRecord(node) || (typeof node.type === "string" && OPAQUE_TYPES.has(node.type))) return node;', "if (!isRecord(node)) return node;")
m("the last look enters images", "messages.ts", '			if (node.startsWith("data:")) return node;\n', "")
m("withheld texts are not counted", "messages.ts", "				if (texts[index] === undefined) withheld++;\n", "")

# ----------------------------------------------------------------- index.ts
m("local models are screened", "index.ts", "const isLocal = (s: Session, ctx: Ctx) => !s.config.screenLocalModels && isLocalModel(ctx.model, s.config);", "const isLocal = (s: Session, ctx: Ctx) => false && isLocalModel(ctx.model, s.config);")
m("screenLocalModels is ignored", "index.ts", "const isLocal = (s: Session, ctx: Ctx) => !s.config.screenLocalModels && isLocalModel(ctx.model, s.config);", "const isLocal = (s: Session, ctx: Ctx) => isLocalModel(ctx.model, s.config);")
m("off still screens", "index.ts", '(s.mode === "off" || isLocal(s, ctx) ? undefined : s.mode);', "(isLocal(s, ctx) ? undefined : s.mode === \"off\" ? \"mask\" : s.mode);")
m("the door stays open", "index.ts", '			return { action: "handled" };', '			return { action: "continue" };')
m("the text is not given back", "index.ts", '				if (typeof event.text === "string") ctx.ui.setEditorText?.(event.text);\n', "")
m("no abort when the detector is down", "index.ts", "				say(ctx, `${problem.error.message}. The request was stopped, and what could not be checked was not sent. /privacy retry tries the detector again; /privacy off sends without screening.`, \"error\");\n				ctx.abort();", "				say(ctx, `${problem.error.message}. The request was stopped, and what could not be checked was not sent. /privacy retry tries the detector again; /privacy off sends without screening.`, \"error\");")
m("an exception sends the request", "index.ts", "			return withholdAll(ctx, error);", "			return undefined;")
m("withholdAll does not abort", "index.ts", "		try {\n			ctx.abort();\n		} catch {\n			// the messages below hold nothing either way\n		}", "")
m("report mode masks", "index.ts", "				refresh(ctx, s);\n				return undefined;\n			}\n\n			const built", "				refresh(ctx, s);\n			}\n\n			const built")
m("the last look is skipped", "index.ts", "			if (swept.hits === 0) return undefined;", "			if (swept.hits >= 0) return undefined;")
m("code tools are unmasked at the hook", "index.ts", "		if (s.config.codeTools.includes(event.toolName)) return undefined; // the calls such code makes come here themselves\n", "")
m("a refusal does not block", "index.ts", "return refusal === undefined ? undefined : { block: true, reason: refusal };", "return undefined;")
m("settled is ignored at the hook", "index.ts", "const settled = s.settled.get(event.toolCallId);", "const settled = undefined;")
m("settled is not recorded", "index.ts", "			for (const [id, record] of unmasked.settled) s.settled.set(id, record);\n", "")
m("summary with a problem is sent", "index.ts", "			if (masked.problem) throw masked.problem.error;\n			if (masked.texts.includes(undefined)) throw new Error(\"a text could not be checked\");\n", "")
m("summary prompt is not masked", "index.ts", "			if (prompt.length > 0) request.systemPrompt = masked.texts.at(-1);\n", "")
m("failed compaction falls back to pi's own", "index.ts", "			if (!event.signal?.aborted) say(ctx, `compaction was cancelled, because its summary request could not be screened or made (${reasonOf(error)}).`, \"error\");\n			return { cancel: true };", "			return undefined;")
m("the compaction summary stays masked", "index.ts", "return { compaction: { ...result, summary: restored(s, result.summary) } };", "return { compaction: { ...result } };")
m("the summary is not remembered", "index.ts", "		const real = s.screen.unmask(summary);\n		s.screen.remember(real, summary);", "		const real = s.screen.unmask(summary);")
m("failed branch summary falls back", "index.ts", "			if (!event.signal?.aborted) say(ctx, `the branch summary was cancelled, because its request could not be screened or made (${reasonOf(error)}).`, \"error\");\n			return { cancel: true };", "			return undefined;")
m("values are not replayed", "index.ts", 'if (kind === "value" && typeof token === "string" && typeof value === "string") vault.restore(token, value);', 'if (kind === "value" && typeof token === "string" && typeof value === "string") void 0;')
m("remembered text is not replayed", "index.ts", "screen.restore(key, masked);", "void 0;")
m("reveals are not replayed", "index.ts", 'else if (kind === "reveal" && typeof token === "string") screen.reveal(token, revealed === true);', "")
m("values are not saved", "index.ts", '		vault.onAllocate = (entry) => save({ kind: "value", token: entry.token, value: entry.value });\n', "")
m("remembered text is not saved", "index.ts", '		screen.onRemember = (key, masked) => save({ kind: "remember", key, masked });\n', "")
m("reveals are not saved", "index.ts", '					save({ kind: "reveal", token, revealed: name === "reveal" });\n', "")
m("no head start", "index.ts", '		warm(ctx, s);\n		headStart(ctx, s);\n', '		warm(ctx, s);\n')
m("a request does not wait for the head start", "index.ts", '			await s.ahead; // most of these texts are being checked already: that is waited for, not asked for again\n', '')
m("a head start for models that are not screened", "index.ts", '		if (!screening(s, ctx)) return;\n		warm(ctx, s);', '		warm(ctx, s);')
m("an ended session writes the status line", "index.ts", '		if (s !== session) return; // that session has ended; the line belongs to the next one\n', '')
m("a head start that cannot be made stops the session's start", "index.ts", '		} catch {\n			// then the first request does all of it\n		}', '		} finally {\n			// nothing\n		}')
m("the head start does not bring the count up to date", "index.ts", 's.ahead = s.screen.mask(plan.items).then(() => refresh(ctx, s));', 's.ahead = s.screen.mask(plan.items);')
m("the head start is made for another provider", "index.ts", 'const plan = transcript(messages as Message[], { provider: ctx.model?.provider });\n			s.ahead', 'const plan = transcript(messages as Message[]);\n			s.ahead')
m("reveal does not take the value itself", "index.ts", '?.token ?? s.screen.vault.find(what)?.token;', '?.token;')
m("reveal does not take a placeholder", "index.ts", 'const token = s.screen.vault.lookup(`<${bare}>`)?.token ?? s.screen.vault.find(what)?.token;', 'const token = s.screen.vault.find(what)?.token;')
m("conceal reveals", "index.ts", '					s.screen.reveal(token, name === "reveal");\n					save(', '					s.screen.reveal(token, true);\n					save(')
m("the reveal command changes nothing", "index.ts", '					s.screen.reveal(token, name === "reveal");\n					save(', '					save(')
m("what was typed is repeated in the notice", "index.ts", 'else if (token === undefined && /^[A-Z]+_[0-9]+$/.test(bare)) say(', 'else if (token === undefined) say(')
m("an unknown placeholder is not named", "index.ts", 'else if (token === undefined && /^[A-Z]+_[0-9]+$/.test(bare)) say(', 'else if (false) say(')
m("reveal without an argument is not usage", "index.ts", '				if (what === "") say(ctx, USAGE, "warning");\n				else if (token === undefined &&', '				if (token === undefined &&')
m("the note is always in the prompt", "index.ts", "			else delete sections.privacy_screen;\n", "")
m("the note is never in the prompt", "index.ts", '			if (screening(s, ctx) === "mask") sections.privacy_screen = PROMPT_NOTE;\n			else delete sections.privacy_screen;', "			delete sections.privacy_screen;")
m("replies are not unmasked", "index.ts", "			return { message: unmasked.message };", "			return undefined;")
m("extension input is checked at the door", "index.ts", '|| event.source === "extension" || event.streamingBehavior !== undefined) return { action: "continue" };', ') return { action: "continue" };')
m("list always shows values", "index.ts", 'argument === "full" ? entry.value : preview(entry.value)', "entry.value")
m("mode switch keeps the detector's failure", "index.ts", "				if (name === \"mask\") {\n					s.detector.retry();", "				if (name === \"mask\") {")
m("the transformer unmasks nothing", "index.ts", "|| stillComing ? session.screen.unmask(markdown) : markdown;", "|| stillComing ? markdown : markdown;")
m("replies to unscreened requests get values", "index.ts", "			if (!s.answersMasked) return undefined;\n			const unmasked", "			const unmasked")
m("the tool hook gives values after unscreened requests", "index.ts", "		if (!s.answersMasked) return undefined;\n		if (s.config.codeTools", "		if (s.config.codeTools")
m("whether a request was masked is not noted", "index.ts", '			s.answersMasked = how === "mask";\n', "")
m("every request counts as masked", "index.ts", 's.answersMasked = how === "mask";', "s.answersMasked = true;")
m("finished text is given values on screen", "index.ts", "const stillComing = context?.isStreaming === true && session.answersMasked;", "const stillComing = session.answersMasked;")
m("streaming text of an unscreened reply is given values", "index.ts", "const stillComing = context?.isStreaming === true && session.answersMasked;", "const stillComing = context?.isStreaming === true;")
m("thinking is shown as written", "index.ts", 'return context?.messageType === "assistant-thinking" || stillComing ?', "return stillComing ?")
m("another model's thinking is not screened", "messages.ts", '} else if (block.type === "thinking" && origin === "foreign") {', '} else if (false) {')
m("a provider's own thinking is screened", "messages.ts", '} else if (block.type === "thinking" && origin === "foreign") {', '} else if (block.type === "thinking") {')
m("thinking is not remembered", "messages.ts", '		if (block.type === "thinking" && typeof block.thinking === "string") screen.remember(block.thinking, block.thinking);\n', "")
m("config problems are not said", "index.ts", "		if (problems.length > 0) {", "		if (false) {")
m("env files are read once", "index.ts", "			const s = ready(ctx);\n			await learn(ctx, s);\n			const sections", "			const s = ready(ctx);\n			const sections")
m("the branch summary ignores userWantsSummary", "index.ts", "!preparation?.userWantsSummary || ", "")
m("the provider is not told to transcript", "index.ts", "transcript(event.messages, { provider: ctx.model?.provider })", "transcript(event.messages)")

# ---------------------------------------------------------------- config.ts
m("127 anywhere is local", "config.ts", "/^127(\\.\\d{1,3}){3}$/.test(host)", "/127(\\.\\d{1,3}){3}/.test(host)")
m("localhost prefix is local", "config.ts", 'host === "localhost" ||', 'host.startsWith("localhost") ||')
m("an unknown mode is taken", "config.ts", "const isMode = (value: unknown): value is Mode => MODES.includes(value as Mode);", "const isMode = (value: unknown): value is Mode => typeof value === \"string\";")
m("unknown labels are taken", "config.ts", "isNames(value) && value.every((label) => ALL_LABELS.includes(label));", "isNames(value);")
m("unknown settings are silent", "config.ts", "(settings[key] ?? (() => problems.push(`${key}: unknown setting; it is ignored`)))();", "settings[key]?.();")
m("a listed provider is not local", "config.ts", "	if (model.provider !== undefined && config.localProviders.includes(model.provider)) return true;\n", "")
m("the config path ignores the agent dir", "config.ts", 'const agentDir = env.PI_CODING_AGENT_DIR ? expand(env.PI_CODING_AGENT_DIR, home) : join(home, ".pi", "agent");', 'const agentDir = join(home, ".pi", "agent");')

# -------------------------------------------------------------- detector.ts
m("spans outside the text are used", "detector.ts", "start < 0 || end <= start || end > texts[index]!.length) return unusable();", "start < 0 || end <= start) return unusable();")
m("a wrong number of results is used", "detector.ts", "|| body.results.length !== texts.length) return texts.map(unusable);", ") return texts.map(unusable);")
m("too large texts are sent", "detector.ts", "			if (size > maxTextBytes) {", "			if (false) {")
m("a failed request is not split", "detector.ts", "if (texts.length > 1 && (answer.status === 500 || answer.status === 413)) {", "if (false) {")
m("any API version is fine", "detector.ts", "		if (body.api !== API_VERSION) {", "		if (false) {")
m("any listener is the detector", "detector.ts", 'if (answer.status !== 200 || !isRecord(body) || body.status !== "ok") return', 'if (false) return')
m("an error status is read as an answer", "detector.ts", "		if (answer.status === 200) return readResults(texts, answer.body);", "		if (answer.status >= 200) return readResults(texts, answer.body);")
m("detect does not ensure", "detector.ts", "		await this.ensure();\n		const { batchTexts", "		const { batchTexts")
m("autoStart off still starts", "detector.ts", "		if (!autoStart || !command) {", "		if (!command) {")
m("no cool-down", "detector.ts", "		if (this.failed && Date.now() < this.failed.until) throw this.failed.error;\n", "")
m("batches ignore the byte limit", "detector.ts", "if (batch.length >= batchTexts || (batch.length > 0 && bytes + size > batchBytes)) await send();", "if (batch.length >= batchTexts) await send();")
m("offsets are code points", "detector.ts", '{ texts, offsets: "utf16" }', '{ texts, offsets: "codepoint" }')
m("a transport failure is a per-text error", "detector.ts", "			throw new Error(`the detector at ${this.options.socket} did not answer (${(error as Error).message})`);", "			return texts.map(() => new Error(\"x\"));")

# ---------------------------------------------------------------- dotenv.ts
m("example files are read", "dotenv.ts", "return names.filter((name) => !EXAMPLE_FILE.test(name) && usable.some", "return names.filter((name) => usable.some")
m("public names are taken", "dotenv.ts", "if (!SECRET_NAME.test(name) || PUBLIC_NAME.test(name)) continue;", "if (!SECRET_NAME.test(name)) continue;")
m("every name is taken", "dotenv.ts", "if (!SECRET_NAME.test(name) || PUBLIC_NAME.test(name)) continue;", "if (PUBLIC_NAME.test(name)) continue;")
m("short values are taken", "dotenv.ts", "const MIN_LENGTH = 8;", "const MIN_LENGTH = 1;")
m("settings are taken", "dotenv.ts", "if (value.length >= MIN_LENGTH && !isSetting(value)) seeds.add(value);", "if (value.length >= MIN_LENGTH) seeds.add(value);")
m("key formats in the file are skipped", "dotenv.ts", "	for (const found of findPatterns(content)) seeds.add(content.slice(found.start, found.end));\n", "")
m("large files are read", "dotenv.ts", "			if (info.size > MAX_BYTES) {", "			if (false) {")
m("escapes are not read", "dotenv.ts", "		return read === written ? [written] : [written, read];", "		return [written];")
m("comments stay on values", "dotenv.ts", "	return [text.replace(/\\s+#.*$/, \"\")];", "	return [text];")
m("a star matches a leading dot", "dotenv.ts", '	if (pattern.startsWith("*") && name.startsWith(".")) return false;\n', "")

# -------------------------------------------------------------- patterns.ts
m("assigned values need not be plausible", "patterns.ts", "		check: plausibleSecret,\n", "")
m("a bare value ends in a backslash", "patterns.ts", "[^\\\\s\"'\\`,;]{6}[^\\\\s\"'\\`,;]*(?<!\\\\\\\\)))`,", "[^\\\\s\"'\\`,;]{6}[^\\\\s\"'\\`,;]*))`,")
m("a key block is its first line", "patterns.ts", "				if (!last) break; // nothing ends after this point, so nothing does after a later start either\n				end = past = last.index + last[0].length;", "				if (!last) break;")
m("sk- needs no digit", "patterns.ts", "check: (value) => /[0-9]/.test(value) && /[A-Z]/.test(value) },", "check: (value) => /[A-Z]/.test(value) },")
m("the URL password may be a template", "patterns.ts", "groups: [1], check: (value) => !/[${}<>%*]/.test(value) },", "groups: [1] },")
m("overlapping finds are all returned", "patterns.ts", "if (found.length === 0 || candidate.start >= found[found.length - 1]!.end) found.push(candidate);", "found.push(candidate);")
m("a plausible secret needs no digit", "patterns.ts", "	return /[0-9]/.test(value);\n}", "	return true;\n}")
m("paths are plausible secrets", "patterns.ts", "	if (/^(?:\\/|\\.\\.?\\/|~)/.test(value)) return false; // a path\n", "")

# ----------------------------------------------------------------- spans.ts
m("a trailing backslash stays", "spans.ts", '(trailing.test(text[end - 1]!) || text[end - 1] === "\\\\")) end--;', "trailing.test(text[end - 1]!)) end--;")
m("a leading escape stays", "spans.ts", '		else if (start + 1 < end && text[start] === "\\\\" && leading.test(text[start + 1]!)) start += 2;\n', "")
m("spans run across lines", "spans.ts", '		if (index === end || text[index] === "\\n" || text[index] === "\\r") {', "		if (index === end) {")
m("every label is screened", "spans.ts", "if (kind === undefined || !labels.has(span.label)) continue;", "if (kind === undefined) continue;")
m("spans outside the text are used", "spans.ts", "		if (span.start < 0 || span.end > text.length || span.start >= span.end) continue;\n", "")
m("the assigned name stays in a secret", "spans.ts", "			if (name) [start, end] = trim(text, [start + name[0].length, end]);\n", "")
m("short secrets are kept", "spans.ts", "return end - start >= 6 ? [start, end] : undefined;", "return end > start ? [start, end] : undefined;")
m("the address pattern is not applied", "spans.ts", "return email ? [start + email.index, start + email.index + email[0].length] : undefined;", "return [start, end];")

# ----------------------------------------------------------------- shell.ts
m("single quotes in a value are not escaped", "shell.ts", "const inSingle = (value: string) => value.replaceAll(\"'\", `'\\\\''`);", "const inSingle = (value: string) => value;")
m("braces are not checked", "shell.ts", "		if (!affected) return;\n", "		return;\n")
m("NUL is accepted", "shell.ts", '		if (spot.value.includes("\\0")) refuse(spot, "its value contains a NUL character");\n', "")
m("backquotes are accepted", "shell.ts", '			if (spots.has(close)) refuse(spots.get(close)!, "it stands inside backquotes; use $( ) instead");\n', "")
m("after a dollar is accepted", "shell.ts", '		if (spots.has(i + 1)) refuse(spots.get(i + 1)!, "it stands right after a $");\n', "")
m("arithmetic is accepted", "shell.ts", '			if (spots.has(at)) refuse(spots.get(at)!, "it stands inside an arithmetic expression");\n', "")
m("an expansion is accepted", "shell.ts", '			if (spots.has(at)) refuse(spots.get(at)!, "it stands inside a ${ } expansion");\n', "")
m("comments in a substitution are followed", "shell.ts", '		if (insideSubstitution()) return lose("a comment inside $( )");\n', "")
m("a here-document in a substitution is followed", "shell.ts", """				if (nested && /['"()`\\\\]/.test(written)) return lose("a here-document inside $( ) with quotes or parentheses in its text");\n""", "")
m("a dash value is accepted", "shell.ts", '			refuse(spot, "its value starts with a dash and would be read as an option here; put -- before it, or attach it to its option as in --name=…");\n', "")

# ----------------------------------------------------------------- vault.ts
m("a value gets a new placeholder each time", "vault.ts", "		const existing = this.byValue.get(value);\n		if (existing) return existing;\n", "")
m("a contradicting pair is loaded", "vault.ts", "			if (byToken !== byValue) return false;\n", "")
m("a malformed token is loaded", "vault.ts", '		if (!match || typeof value !== "string" || value === "") return false;', '		if (!match) return true;')
m("numbers start over after a restore", "vault.ts", "		this.highest.set(kind, Math.max(this.highest.get(kind) ?? 0, number));", "		this.highest.set(kind, 0);")
m("allocation is not reported", "vault.ts", "		this.onAllocate?.(entry);\n", "")

survived, killed, broken = [], [], []
started = time.time()
for name, file, old, new in M:
    if only and only not in name and only != file:
        continue
    work = scratch / "pkg"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shutil.copytree(package / "src", work / "src")
    (work / "test").mkdir()
    for test in (package / "test").glob("*.test.ts"):
        shutil.copy(test, work / "test" / test.name)
    shutil.copy(package / "package.json", work / "package.json")
    shutil.copy(package / "index.ts", work / "index.ts")
    target = work / "src" / file
    text = target.read_text()
    if text.count(old) != 1:
        broken.append(f"{file}: {name} (the text to change occurs {text.count(old)} times)")
        continue
    target.write_text(text.replace(old, new))
    tests = [f"test/{t}.test.ts" for t in TESTS[file]]
    env = dict(os.environ, SHELL_FUZZ="40")
    try:
        run = subprocess.run(["node", "--test", *tests], cwd=work, capture_output=True, text=True, timeout=240, env=env)
        out = run.stdout + run.stderr
        failed = re.findall(r"^✖ (.+?) \(\d", out, re.M)
        if run.returncode == 0:
            survived.append(f"{file}: {name}")
        else:
            first = failed[0] if failed else out.strip().splitlines()[-1][:100]
            killed.append(f"{file}: {name}  <-  {len(set(failed))} failing, e.g. {first[:90]}")
    except subprocess.TimeoutExpired:
        killed.append(f"{file}: {name}  <-  timed out")
    sys.stdout.write("S" if survived and survived[-1].endswith(": " + name) else ".")
    sys.stdout.flush()
print()
print(f"{len(killed)} killed, {len(survived)} survived, {len(broken)} could not be applied, in {time.time() - started:.0f} s")
for line in survived:
    print("SURVIVED  ", line)
for line in broken:
    print("NOT APPLIED", line)
if os.environ.get("MUT_VERBOSE"):
    for line in killed:
        print("killed    ", line)
shutil.rmtree(scratch, ignore_errors=True)
sys.exit(1 if survived or broken else 0)
