import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";

import { PROMPT_NOTE, install } from "../src/index.ts";
import { OWN_TYPE, WITHHELD } from "../src/messages.ts";
import type { Span } from "../src/spans.ts";
import { tokenPattern } from "../src/vault.ts";

const scratch = mkdtempSync(join(tmpdir(), "pi-privacy-ext-"));
after(() => rmSync(scratch, { recursive: true, force: true }));

const EMAIL = "lena@example.com";
const REMOTE = { provider: "acme", id: "big", baseUrl: "https://api.acme.example/v1" };
const LOCAL = { provider: "llama", id: "small", baseUrl: "http://127.0.0.1:8080/v1" };

type Handler = (event: any, ctx: any) => unknown;

interface Options {
	config?: unknown;
	entries?: any[];
	model?: object;
	values?: Record<string, string>;
	detect?: (text: string) => Span[];
	cwd?: string;
	hasUI?: boolean;
	importPi?: () => Promise<any>;
	history?: any[] | (() => unknown);
	gate?: () => Promise<void>;
}

function harness(options: Options = {}) {
	const dir = mkdtempSync(join(scratch, "h-"));
	const configFile = join(dir, "privacy-screen.json");
	if (options.config !== undefined) writeFileSync(configFile, typeof options.config === "string" ? options.config : JSON.stringify(options.config));

	const values = options.values ?? { [EMAIL]: "private_email" };
	const detector = {
		down: undefined as string | undefined,
		ensures: 0,
		retries: 0,
		calls: 0,
		asked: [] as string[],
		async ensure() {
			this.ensures++;
			if (this.down) throw new Error(this.down);
			return { status: "ok", api: 1 };
		},
		async health() {
			return this.down ? undefined : { status: "ok", api: 1 };
		},
		retry() {
			this.retries++;
		},
		async detect(texts: readonly string[]): Promise<(Span[] | Error)[]> {
			detector.calls++;
			await options.gate?.();
			if (detector.down) throw new Error(detector.down);
			detector.asked.push(...texts);
			return texts.map((text) => {
				if (text.includes("UNCHECKABLE")) return new Error("the text does not survive tokenization");
				if (options.detect) return options.detect(text);
				const spans: Span[] = [];
				for (const [value, label] of Object.entries(values)) {
					for (let at = text.indexOf(value); at !== -1; at = text.indexOf(value, at + 1)) spans.push({ start: at, end: at + value.length, label });
				}
				return spans;
			});
		},
	};

	const handlers = new Map<string, Handler[]>();
	const commands = new Map<string, { handler: (args: string, ctx: any) => unknown; getArgumentCompletions?: (prefix: string) => unknown }>();
	const entries: any[] = options.entries ?? [];
	const notes: [level: string, message: string][] = [];
	const status: Record<string, string | undefined> = {};
	const seen = { aborted: 0, editor: undefined as string | undefined, streamed: [] as any[], errors: [] as string[] };
	let transformer: { fn: (markdown: string, context?: any) => string; options: any } | undefined;

	const pi = {
		on: (event: string, handler: Handler) => void handlers.set(event, [...(handlers.get(event) ?? []), handler]),
		registerCommand: (name: string, spec: any) => void commands.set(name, spec),
		appendEntry: (customType: string, data: unknown) => void entries.push({ type: "custom", customType, data }),
		registerMarkdownTransformer: (fn: any, transformerOptions: any) => void (transformer = { fn, options: transformerOptions }),
	};
	const ctx = {
		hasUI: options.hasUI ?? true,
		mode: "tui",
		cwd: options.cwd ?? dir,
		model: options.model ?? REMOTE,
		thinkingLevel: "medium",
		ui: {
			notify: (message: string, level = "info") => void notes.push([level, message]),
			setStatus: (key: string, text: string | undefined) => void (status[key] = text),
			setEditorText: (text: string) => void (seen.editor = text),
		},
		sessionManager: {
			getEntries: () => entries,
			...(options.history === undefined ? {} : { buildSessionProjection: () => (typeof options.history === "function" ? options.history() : { entries: [], messages: options.history }) }),
		},
		abort: () => void seen.aborted++,
		modelRegistry: {
			streamSimple: (model: unknown, context: unknown, streamOptions: unknown) => {
				seen.streamed.push({ model, context, options: streamOptions });
				return { result: async () => ({ role: "assistant", content: [{ type: "text", text: "ok" }] }) };
			},
		},
	};

	install(pi, {
		env: { PI_PRIVACY_SCREEN_CONFIG: configFile },
		home: dir,
		detector: () => detector,
		importPi: options.importPi,
		log: (message: string) => void seen.errors.push(message),
	});

	const emit = async (event: string, payload: object = {}) => {
		let result: any;
		for (const handler of handlers.get(event) ?? []) {
			const one = await handler({ type: event, ...payload }, ctx);
			if (one !== undefined) result = one;
		}
		return result;
	};
	const send = async (...messages: any[]) => (await emit("context_with_system", { messages: [{ role: "system", content: "You are helpful.", timestamp: 0 }, ...messages] }))?.messages?.slice(1);
	const command = (args: string) => commands.get("privacy")!.handler(args, ctx);
	const ownEntries = () => entries.filter((entry) => entry.type === "custom" && entry.customType === OWN_TYPE).map((entry) => entry.data);
	return { ctx, emit, send, command, commands, entries, ownEntries, notes, status, detector, seen, dir, transformer: () => transformer!, start: () => emit("session_start", { reason: "startup" }) };
}

const idle = () => new Promise<void>((resolve) => setImmediate(resolve));

const user = (text: string) => ({ role: "user", content: text, timestamp: 1 });
const toolResult = (text: string) => ({ role: "toolResult", toolCallId: "c1", toolName: "read", content: [{ type: "text", text }], isError: false, timestamp: 2 });
const assistant = (...content: object[]) => ({ role: "assistant", content, provider: "acme", model: "big", stopReason: "stop", timestamp: 3 });
const text = (value: string) => ({ type: "text", text: value });

test("what goes to the provider is masked, and what comes back is real again", async () => {
	const h = harness();
	await h.start();

	const out = await h.send(user(`mail ${EMAIL}`));
	const reply = await h.emit("message_end", { message: assistant(text("I will write to <EMAIL_1>.")) });
	const again = await h.send(user(`mail ${EMAIL}`), reply.message, user("thanks"));

	assert.equal(out[0].content, "mail <EMAIL_1>");
	assert.deepEqual(reply.message.content, [text(`I will write to ${EMAIL}.`)]);
	assert.deepEqual(again.map((message: any) => message.content), ["mail <EMAIL_1>", [text("I will write to <EMAIL_1>.")], "thanks"]);
	assert.equal(h.seen.aborted, 0);
	assert.match(h.status.privacy!, /mask.*1 hidden/);
});

test("the system prompt keeps its place and its tools", async () => {
	const h = harness();
	await h.start();
	const tool = { name: "bash" };
	const system = { role: "system", content: "", sections: { preamble: `Ask ${EMAIL}.` }, toolsAdded: [tool], timestamp: 0 };

	const { messages } = await h.emit("context_with_system", { messages: [system, user(`mail ${EMAIL}`)] });

	assert.equal(messages[0].role, "system");
	assert.equal(messages[0].sections.preamble, "Ask <EMAIL_1>.");
	assert.equal(messages[0].toolsAdded[0], tool);
});

test("a tool call gets its values before it runs, quoted for the shell where it is a command", async () => {
	const h = harness();
	await h.start();
	await h.send(user(`mail ${EMAIL}`));
	const write = { path: "to.txt", content: "<EMAIL_1>\n" };
	const bash = { command: "echo <EMAIL_1> >> log.txt" };
	const tricky = { command: "echo `echo <EMAIL_1>`" };
	const code = { code: "await tools.write({ path: 'a', content: '<EMAIL_1>' })" };

	assert.equal(await h.emit("tool_call", { toolName: "write", toolCallId: "1", input: write }), undefined);
	assert.equal(await h.emit("tool_call", { toolName: "bash", toolCallId: "2", input: bash }), undefined);
	const blocked = await h.emit("tool_call", { toolName: "bash", toolCallId: "3", input: tricky });
	assert.equal(await h.emit("tool_call", { toolName: "codemode", toolCallId: "4", input: code }), undefined);

	assert.deepEqual(write, { path: "to.txt", content: `${EMAIL}\n` });
	assert.deepEqual(bash, { command: `echo '${EMAIL}' >> log.txt` });
	assert.equal(blocked.block, true);
	assert.match(blocked.reason, /backquotes/);
	assert.deepEqual(tricky, { command: "echo `echo <EMAIL_1>`" });
	assert.deepEqual(code, { code: "await tools.write({ path: 'a', content: '<EMAIL_1>' })" }, "code is left to the calls it makes");
});

test("a reply's tool calls are stored as they will run, and are not given their values twice", async () => {
	const h = harness();
	await h.start();
	await h.send(user(`mail ${EMAIL}`));
	const call = (id: string, name: string, args: object) => ({ type: "toolCall", id, name, arguments: args });
	const reply = assistant(text("On it."), call("a", "bash", { command: "echo <EMAIL_1>" }), call("b", "bash", { command: "echo `echo <EMAIL_1>`" }), call("c", "codemode", { code: "'<EMAIL_1>'" }));

	const { message } = await h.emit("message_end", { message: reply });
	const asStored = structuredClone(message.content[1].arguments);
	const asWritten = { command: "echo <EMAIL_1>" };
	const first = await h.emit("tool_call", { toolName: "bash", toolCallId: "a", input: asStored });
	const second = await h.emit("tool_call", { toolName: "bash", toolCallId: "a2", input: asWritten });
	const refused = await h.emit("tool_call", { toolName: "bash", toolCallId: "b", input: structuredClone(message.content[2].arguments) });
	const out = await h.send(user(`mail ${EMAIL}`), message);

	assert.deepEqual(message.content.slice(1).map((block: any) => block.arguments), [{ command: `echo '${EMAIL}'` }, { command: "echo `echo <EMAIL_1>`" }, { code: "'<EMAIL_1>'" }]);
	assert.deepEqual([first, second], [undefined, undefined]);
	assert.deepEqual(asStored, { command: `echo '${EMAIL}'` });
	assert.deepEqual(asWritten, { command: `echo '${EMAIL}'` });
	assert.equal(refused.block, true);
	assert.deepEqual(out[1].content.slice(1).map((block: any) => block.arguments), [{ command: "echo <EMAIL_1>" }, { command: "echo `echo <EMAIL_1>`" }, { code: "'<EMAIL_1>'" }], "and they go back out as the model wrote them");
});

test("a value that itself reads like a placeholder is put in once, and is still that text when the tool runs", async () => {
	const h = harness();
	await h.start();
	const out = await h.send(user(`mail ${EMAIL}`), toolResult("the template says: Dear <EMAIL_1>,"));
	const reply = assistant({ type: "toolCall", id: "w", name: "write", arguments: { path: "copy.txt", content: "Dear <LITERAL_1>," } });

	const { message } = await h.emit("message_end", { message: reply });
	const input = structuredClone(message.content[0].arguments);
	await h.emit("tool_call", { toolName: "write", toolCallId: "w", input });

	assert.deepEqual(out[1].content, [text("the template says: Dear <LITERAL_1>,")]);
	assert.deepEqual(input, { path: "copy.txt", content: "Dear <EMAIL_1>," }, "the file's own text, not the address");
});

test("with the detector down nothing unchecked leaves: the message is held back at the door", async () => {
	const h = harness();
	await h.start();
	h.detector.down = "the detector is not running at /x/opf.sock";

	const typed = await h.emit("input", { text: `mail ${EMAIL}`, source: "interactive" });

	assert.deepEqual(typed, { action: "handled" });
	assert.equal(h.seen.editor, `mail ${EMAIL}`, "the text is given back to the editor");
	assert.match(h.notes.at(-1)![1], /not running at \/x\/opf\.sock[\s\S]*not sent[\s\S]*\/privacy off/);
	assert.equal(h.notes.at(-1)![0], "error");
	assert.match(h.status.privacy!, /detector down/i);
});

test("the door is for what the user sends: an extension's text, or text typed while a reply streams, is checked with the request", async () => {
	const h = harness();
	await h.start();
	h.detector.down = "the detector is not running at /x/opf.sock";

	const fromExtension = await h.emit("input", { text: `mail ${EMAIL}`, source: "extension" });
	const whileStreaming = await h.emit("input", { text: `mail ${EMAIL}`, source: "interactive", streamingBehavior: "steer" });
	const out = await h.send(user(`mail ${EMAIL}`));

	assert.deepEqual([fromExtension, whileStreaming], [{ action: "continue" }, { action: "continue" }]);
	assert.equal(h.seen.editor, undefined, "nothing is put into the editor that the user did not just send from it");
	assert.equal(out[0].content, WITHHELD, "and the request is where it is stopped");
	assert.equal(h.seen.aborted, 1);
});

test("with the detector down a request that is already under way is stopped, and its new text withheld", async () => {
	const h = harness();
	await h.start();
	const first = await h.send(user(`mail ${EMAIL}`));
	h.detector.down = "the detector at /x/opf.sock did not answer (ECONNREFUSED)";

	const out = await h.send(user(`mail ${EMAIL}`), toolResult(`the file says: call ${EMAIL} or 555 0100`));

	assert.equal(first[0].content, "mail <EMAIL_1>");
	assert.equal(out[0].content, "mail <EMAIL_1>", "what was checked before still goes");
	assert.deepEqual(out[1].content, [text(WITHHELD)]);
	assert.equal(h.seen.aborted, 1);
	assert.match(h.notes.at(-1)![1], /did not answer/);
});

test("a text that cannot be checked is withheld by itself, and the request goes on", async () => {
	const h = harness();
	await h.start();

	const out = await h.send(user(`mail ${EMAIL}`), toolResult("UNCHECKABLE bytes"));

	assert.equal(out[0].content, "mail <EMAIL_1>");
	assert.deepEqual(out[1].content, [text(WITHHELD)]);
	assert.equal(h.seen.aborted, 0);
	assert.deepEqual(h.notes.at(-1)![0], "warning");
	assert.match(h.notes.at(-1)![1], /1 text.*withheld/);
});

test("a failure inside the screen never lets a request through as it was", async () => {
	const h = harness();
	await h.start();
	const poisoned = { role: "user", timestamp: 1 };
	Object.defineProperty(poisoned, "content", { enumerable: true, get: () => { throw new Error("boom"); } });

	const result = await h.emit("context_with_system", { messages: [{ role: "system", content: "x", timestamp: 0 }, poisoned] });

	assert.deepEqual(result.messages.map((message: any) => [message.role, message.content]), [["system", WITHHELD], ["user", WITHHELD]]);
	assert.equal(h.seen.aborted, 1);
	assert.match(h.notes.at(-1)![1], /boom/);
});

test("the conversation a session starts with is checked at once, so that its first request finds the work done", async () => {
	const history = [user(`mail ${EMAIL}`), assistant(text("I will look at the log.")), toolResult("the log says: all quiet")];
	const h = harness({ history });

	await h.start();
	await idle();
	const before = [...h.detector.asked];
	const hidden = h.status.privacy;
	const sent = await h.send(...history, user("and now?"));

	assert.deepEqual(before, [`mail ${EMAIL}`, "the log says: all quiet"], "asked while the user was still typing; not for what this provider wrote itself");
	assert.equal(hidden, "privacy: mask (1 hidden)");
	assert.deepEqual(h.detector.asked.slice(before.length), ["and now?"], "the request asks only for what is new");
	assert.equal(sent[0].content, "mail <EMAIL_1>");
});

test("a request that comes while that check is still running waits for it, rather than asking for the same texts again", async () => {
	let release!: () => void;
	const gate = new Promise<void>((resolve) => (release = resolve));
	const history = [user(`mail ${EMAIL}`), toolResult("the log says: all quiet")];
	const h = harness({ history, gate: () => gate });

	await h.start();
	await idle();
	const sending = h.send(...history);
	await idle();
	const calls = h.detector.calls;
	release();
	const sent = await sending;

	assert.equal(calls, 1, "one question to the detector while it was busy, not two");
	assert.deepEqual(h.detector.asked, [`mail ${EMAIL}`, "the log says: all quiet"]);
	assert.equal(sent[0].content, "mail <EMAIL_1>");
});

test("the early check is a head start and nothing more: when it cannot be made it says nothing, and the request is screened as ever", async () => {
	const history = [user(`mail ${EMAIL}`)];
	const down = harness({ history });
	down.detector.down = "connect ENOENT";
	await down.start();
	await idle();
	const told = down.notes.length;
	down.detector.down = undefined;
	const sent = await down.send(...history);

	assert.equal(told, 1, "the one notice is that the detector is not running");
	assert.equal(down.seen.aborted, 0);
	assert.equal(sent[0].content, "mail <EMAIL_1>");

	const broken = harness({ history: () => { throw new Error("no projection"); } });
	await broken.start();
	await idle();
	assert.deepEqual([broken.notes, broken.seen.errors], [[], []]);
	assert.equal((await broken.send(...history))[0].content, "mail <EMAIL_1>");
});

test("nothing is checked early for a model that is not screened; in report mode it is, for the count", async () => {
	const history = [user(`mail ${EMAIL}`)];
	const local = harness({ history, model: LOCAL });
	const report = harness({ history, config: { mode: "report" } });

	await local.start();
	await report.start();
	await idle();

	assert.deepEqual(local.detector.asked, []);
	assert.deepEqual(report.detector.asked, [`mail ${EMAIL}`]);
	assert.equal(report.status.privacy, "privacy: report only (1 found)");
});

test("a session that has ended no longer writes to the status line", async () => {
	let release!: () => void;
	const gate = new Promise<void>((resolve) => (release = resolve));
	const h = harness({ history: [user(`mail ${EMAIL}`)], gate: () => gate });

	await h.start();
	await idle();
	await h.emit("session_shutdown");
	release();
	await idle();

	assert.equal(h.status.privacy, undefined);
});

test("replies written by another model are checked before a new provider sees them", async () => {
	const h = harness({ values: { [EMAIL]: "private_email", "Omar Haddad": "private_person" } });
	await h.start();
	const local = { ...assistant(text("Omar Haddad asked for it.")), provider: "llama", model: "small" };
	const own = assistant(text("Omar Haddad again, in my own words."));

	const out = await h.send(local, own);

	assert.deepEqual(out[0].content, [text("<PERSON_1> asked for it.")]);
	assert.deepEqual(out[1].content, [text("<PERSON_1> again, in my own words.")], "a value once known is replaced everywhere");
	assert.deepEqual(h.detector.asked, ["Omar Haddad asked for it."], "its own words are not sent to the detector");
});

test("report mode looks, counts, and changes nothing", async () => {
	const h = harness({ config: { mode: "report" } });
	await h.start();
	h.ctx.ui.setStatus("privacy", undefined);

	assert.equal(await h.send(user(`mail ${EMAIL}`)), undefined);
	assert.equal(await h.emit("before_provider_request", { payload: { messages: [{ role: "user", content: `mail ${EMAIL}` }] } }), undefined);
	assert.match(h.status.privacy!, /report.*1 found/);

	h.detector.down = "gone";
	assert.deepEqual(await h.emit("input", { text: "hello", source: "interactive" }), { action: "continue" });
	assert.equal(await h.send(user("another text")), undefined);
	assert.equal(h.seen.aborted, 0);
});

test("off screens nothing; the reply that was still on its way gets its values, later ones are left as they are", async () => {
	const h = harness();
	await h.start();
	await h.send(user(`mail ${EMAIL}`));

	await h.command("off");
	const onItsWay = await h.emit("message_end", { message: assistant(text("To <EMAIL_1>")) });
	const unscreened = await h.send(user(`mail ${EMAIL}`));
	const payload = await h.emit("before_provider_request", { payload: { x: EMAIL } });
	const later = await h.emit("message_end", { message: assistant(text("The file says <EMAIL_1>")) });

	assert.deepEqual(onItsWay.message.content, [text(`To ${EMAIL}`)], "it answers a request that went out masked");
	assert.equal(unscreened, undefined);
	assert.equal(payload, undefined);
	assert.equal(later, undefined, "this model was sent the real text: a placeholder in its reply is text it quotes");
	assert.match(h.status.privacy!, /OFF/);
	assert.equal(h.notes.at(-1)![0], "warning");
});

test("report mode changes nothing on the way in either", async () => {
	const h = harness({ config: { mode: "report" } });
	await h.start();
	await h.send(user(`mail ${EMAIL}`));
	const call = { type: "toolCall", id: "w", name: "write", arguments: { path: "doc.md", content: "Write <EMAIL_1> for the address." } };

	const ended = await h.emit("message_end", { message: assistant(text("The docs say <EMAIL_1>."), call) });
	const input = structuredClone(call.arguments);
	const hook = await h.emit("tool_call", { toolName: "write", toolCallId: "w", input });
	const shown = h.transformer().fn("The docs say <EMAIL_1>.", { messageType: "assistant", isStreaming: true, availableWidth: 80 });

	assert.match(h.status.privacy!, /report.*1 found/);
	assert.equal(ended, undefined);
	assert.equal(hook, undefined);
	assert.deepEqual(input, { path: "doc.md", content: "Write <EMAIL_1> for the address." }, "the file gets what the model wrote");
	assert.equal(shown, "The docs say <EMAIL_1>.");
});

test("after a switch to a model on this machine, its replies are left as they are", async () => {
	const h = harness();
	await h.start();
	await h.send(user(`mail ${EMAIL}`));

	h.ctx.model = LOCAL;
	await h.emit("model_select", { model: LOCAL, source: "set" });
	const unscreened = await h.send(user(`mail ${EMAIL}`));
	const reply = await h.emit("message_end", { message: { ...assistant(text("The template has <EMAIL_1> in it.")), provider: "llama", model: "small" } });
	h.ctx.model = REMOTE;
	const back = await h.send(user(`mail ${EMAIL}`), { ...assistant(text("The template has <EMAIL_1> in it.")), provider: "llama", model: "small" });

	assert.equal(unscreened, undefined);
	assert.equal(reply, undefined);
	assert.deepEqual(back[1].content, [text("The template has <LITERAL_1> in it.")], "and to the next provider it is the text it was");
});

test("a model on this machine is not screened, unless the settings ask for it", async () => {
	const plain = harness({ model: LOCAL });
	await plain.start();
	const asked = harness({ model: LOCAL, config: { screenLocalModels: true } });
	await asked.start();

	assert.equal(await plain.send(user(`mail ${EMAIL}`)), undefined);
	assert.deepEqual(await plain.emit("input", { text: "hi", source: "interactive" }), { action: "continue" });
	assert.equal(plain.detector.ensures, 0, "the detector is not even started");
	assert.match(plain.status.privacy!, /local model/);
	assert.equal((await asked.send(user(`mail ${EMAIL}`)))[0].content, "mail <EMAIL_1>");
});

test("placeholders and what the model wrote survive a restart of the session", async () => {
	const first = harness();
	await first.start();
	await first.send(user(`mail ${EMAIL}`));
	const reply = await first.emit("message_end", { message: assistant(text("Sure: mailto:<EMAIL_1>x")) });

	const second = harness({ entries: structuredClone(first.entries), values: { [EMAIL]: "private_email", "omar@example.com": "private_email" } });
	await second.start();
	const out = await second.send(user("and omar@example.com"), reply.message, user(`mail ${EMAIL}`));

	assert.deepEqual(first.ownEntries().map((data) => data.kind), ["value", "remember", "remember"], "the text as it reads here, and as it was written");
	assert.deepEqual(first.ownEntries()[0], { v: 1, kind: "value", token: "<EMAIL_1>", value: EMAIL });
	assert.deepEqual(out.map((message: any) => message.content), [
		"and <EMAIL_2>",
		[text("Sure: mailto:<EMAIL_1>x")],
		"mail <EMAIL_1>",
	]);
});

test("the last look replaces a known value wherever the request still carries it, and says so once", async () => {
	const h = harness();
	await h.start();
	await h.send(user(`mail ${EMAIL}`));
	const payload = { messages: [{ role: "user", content: "mail <EMAIL_1>" }], tools: [{ name: "notify", description: `Notifies ${EMAIL}` }] };

	const clean = await h.emit("before_provider_request", { payload: { messages: payload.messages } });
	const swept = await h.emit("before_provider_request", { payload });
	await h.emit("before_provider_request", { payload });

	assert.equal(clean, undefined);
	assert.deepEqual(swept, { messages: payload.messages, tools: [{ name: "notify", description: "Notifies <EMAIL_1>" }] });
	assert.equal(h.notes.filter(([, message]) => /last step/.test(message)).length, 1);
});

test("/privacy shows the state and switches the mode for the session", async () => {
	const h = harness();
	await h.start();
	await h.send(user(`mail ${EMAIL}`));

	await h.command("");
	const shown = h.notes.at(-1)![1];
	await h.command("report");
	const reported = await h.send(user(`mail ${EMAIL}`));
	await h.command("mask");
	const masked = await h.send(user(`mail ${EMAIL}`));
	await h.command("sideways");

	assert.match(shown, /mode: mask/);
	assert.match(shown, /1 value hidden/);
	assert.match(shown, /detector: running/);
	assert.equal(reported, undefined);
	assert.equal(masked[0].content, "mail <EMAIL_1>");
	assert.equal(h.detector.retries, 1, "switching to mask tries the detector again");
	assert.match(h.notes.at(-1)![1], /usage/i);
	assert.deepEqual(h.commands.get("privacy")!.getArgumentCompletions!("re"), [
		{ value: "report", label: "report" },
		{ value: "reveal", label: "reveal" },
		{ value: "retry", label: "retry" },
	]);
});

test("/privacy list shows what is hidden without spelling it out", async () => {
	const h = harness();
	await h.start();
	await h.send(user(`mail ${EMAIL}`));

	await h.command("list");
	const brief = h.notes.at(-1)![1];
	await h.command("list full");
	const full = h.notes.at(-1)![1];

	assert.match(brief, /<EMAIL_1>/);
	assert.ok(!brief.includes(EMAIL));
	assert.match(brief, /16 characters/);
	assert.ok(full.includes(EMAIL));
});

test("/privacy reveal lets the model see one value, and conceal hides it again", async () => {
	const h = harness();
	await h.start();
	await h.send(user(`mail ${EMAIL}`));

	await h.command("reveal EMAIL_1");
	const revealed = await h.send(user(`mail ${EMAIL}`));
	await h.command("conceal <EMAIL_1>");
	const concealed = await h.send(user(`mail ${EMAIL}`));
	await h.command("reveal <EMAIL_7>");

	assert.equal(revealed[0].content, `mail ${EMAIL}`);
	assert.equal(concealed[0].content, "mail <EMAIL_1>");
	assert.deepEqual(h.ownEntries().filter((data) => data.kind === "reveal"), [
		{ v: 1, kind: "reveal", token: "<EMAIL_1>", revealed: true },
		{ v: 1, kind: "reveal", token: "<EMAIL_1>", revealed: false },
	]);
	assert.match(h.notes.at(-1)![1], /no placeholder <EMAIL_7>/);
	await h.command("reveal");
	assert.match(h.notes.at(-1)![1], /^Privacy screen: Usage: \/privacy /, "without saying what, it is a question of how");

	const later = harness({ entries: [...h.entries, { type: "custom", customType: OWN_TYPE, data: { v: 1, kind: "reveal", token: "<EMAIL_1>", revealed: true } }] });
	await later.start();
	assert.equal((await later.send(user(`mail ${EMAIL}`)))[0].content, `mail ${EMAIL}`, "a reveal is remembered with the session");
});

test("/privacy reveal also takes the value itself: the model's advice to reveal a placeholder reaches the user with the value in it", async () => {
	const h = harness({ values: { "Lena Fischer": "private_person" } });
	await h.start();
	await h.send(user("ask Lena Fischer about it"));

	await h.command("reveal Lena Fischer");
	const told = h.notes.at(-1)![1];
	const revealed = await h.send(user("ask Lena Fischer about it"));
	await h.command("conceal Lena Fischer");
	const concealed = await h.send(user("ask Lena Fischer about it"));
	await h.command("reveal Nobody Known");

	assert.match(told, /<PERSON_1> is now sent as it is/);
	assert.equal(revealed[0].content, "ask Lena Fischer about it");
	assert.equal(concealed[0].content, "ask <PERSON_1> about it");
	assert.deepEqual(h.ownEntries().filter((data) => data.kind === "reveal"), [
		{ v: 1, kind: "reveal", token: "<PERSON_1>", revealed: true },
		{ v: 1, kind: "reveal", token: "<PERSON_1>", revealed: false },
	]);
	assert.match(h.notes.at(-1)![1], /nothing is hidden under that/);
	assert.ok(!/nobody known/i.test(h.notes.at(-1)![1]), "what was typed may be a value: it is not repeated in the notice");
});

test("a settings file with a mistake is reported, and the screen stays on", async () => {
	const h = harness({ config: { mode: "of", labels: ["secret", "private_email"] } });
	await h.start();

	assert.equal((await h.send(user(`mail ${EMAIL}`)))[0].content, "mail <EMAIL_1>");
	assert.ok(h.notes.some(([level, message]) => level === "error" && /privacy-screen\.json[\s\S]*mode: must be/.test(message)));
});

test("without a screen to show notices on, problems go to the log", async () => {
	const h = harness({ config: "{ not json", hasUI: false });
	await h.start();

	assert.equal(h.notes.length, 0);
	assert.ok(h.seen.errors.some((message) => /not valid JSON/.test(message)));
});

test("the prompt explains placeholders while masking, and only then", async () => {
	const h = harness();
	await h.start();
	const sections: Record<string, string> = { other: "kept" };

	await h.emit("before_agent_start", { systemPromptOptions: { sections } });
	const whileMasking = { ...sections };
	await h.command("off");
	await h.emit("before_agent_start", { systemPromptOptions: { sections } });

	assert.deepEqual(whileMasking, { other: "kept", privacy_screen: PROMPT_NOTE });
	assert.deepEqual(sections, { other: "kept" });
	assert.equal(tokenPattern().test(PROMPT_NOTE), false, "the note itself must not look like a placeholder");
	assert.equal(tokenPattern().test(WITHHELD), false);
});

test("secrets in the project's .env are replaced without the detector having to find them", async () => {
	const project = mkdtempSync(join(scratch, "project-"));
	writeFileSync(join(project, ".env"), "API_TOKEN=tok_9f8e7d6c5b4a\nPORT=3000\n");
	const h = harness({ cwd: project });
	await h.start();

	const out = await h.send(toolResult("config: port 3000, token tok_9f8e7d6c5b4a"));
	writeFileSync(join(project, ".env.local"), "DB_PASSWORD=correct-horse-battery\n");
	await h.emit("before_agent_start", { systemPromptOptions: { sections: {} } });
	const later = await h.send(toolResult("the password is correct-horse-battery"));

	assert.deepEqual(out[0].content, [text("config: port 3000, token <SECRET_1>")]);
	assert.deepEqual(later[0].content, [text("the password is <SECRET_2>")], "files are read again before each run");
});

test("a reply is shown with its values while it streams; once it is stored it has them, and is shown as it is", async () => {
	const h = harness();
	await h.start();
	await h.send(user(`mail ${EMAIL}`));
	const show = (markdown: string, messageType: string, isStreaming: boolean) => h.transformer().fn(markdown, { messageType, isStreaming, availableWidth: 80 });

	assert.equal(show("Writing to <EMAIL_1> and <EMAIL_9>", "assistant", true), `Writing to ${EMAIL} and <EMAIL_9>`);
	assert.equal(show("The file's own text is <EMAIL_1>", "assistant", false), "The file's own text is <EMAIL_1>", "what is stored is real: a placeholder left in it is text");
	assert.equal(show("I will use <EMAIL_1>", "assistant-thinking", true), `I will use ${EMAIL}`);
	assert.equal(show("I will use <EMAIL_1>", "assistant-thinking", false), `I will use ${EMAIL}`, "thinking is stored as the model wrote it, so it is shown with its values whenever it is shown");
	assert.deepEqual(h.transformer().options, { messageTypes: ["assistant", "assistant-thinking"] });
});

function fakeCompaction(calls: any[], answers: { prompt?: string; summary?: string; branch?: object } = {}) {
	const written = (messages: any[]) => messages.map((message) => `[User]: ${message.content}`).join("\n");
	return async () => ({
		compact: async (preparation: any, model: any, apiKey: any, headers: any, customInstructions: any, signal: any, thinkingLevel: any, streamFn: any) => {
			calls.push({ preparation, model, apiKey, headers, customInstructions, signal, thinkingLevel });
			const context = {
				systemPrompt: answers.prompt ?? "Summarize the conversation.",
				messages: [{ role: "user", content: [text(`<conversation>\n${written(preparation.messagesToSummarize)}\n</conversation>\n${customInstructions ?? ""}`)], timestamp: 1 }],
			};
			await (await streamFn(model, context, { apiKey, maxTokens: 100 })).result();
			return { summary: answers.summary ?? "The user wants to mail <EMAIL_1>.", firstKeptEntryId: "entry-7", tokensBefore: 1234, usage: { input: 1 } };
		},
		generateBranchSummary: async (entries: any[], summaryOptions: any) => {
			calls.push({ entries, options: summaryOptions });
			const context = { messages: [{ role: "user", content: [text(written(entries.map((entry) => entry.message)))], timestamp: 1 }] };
			await (await summaryOptions.streamFn(summaryOptions.model, context, {})).result();
			return answers.branch ?? { summary: "A side branch about <EMAIL_1>.", readFiles: ["a.txt"], modifiedFiles: [] };
		},
	});
}

test("compaction summarizes masked text, and the summary is stored with its values", async () => {
	const calls: any[] = [];
	const h = harness({ importPi: fakeCompaction(calls, { prompt: `Summarize. Never name ${EMAIL}.` }) });
	await h.start();
	const preparation = { firstKeptEntryId: "entry-7", messagesToSummarize: [user(`mail ${EMAIL}`)], turnPrefixMessages: [], tokensBefore: 1234 };
	const signal = new AbortController().signal;

	const result = await h.emit("session_before_compact", { preparation, branchEntries: [], customInstructions: `keep ${EMAIL}`, reason: "manual", signal });

	assert.deepEqual(result, { compaction: { summary: `The user wants to mail ${EMAIL}.`, firstKeptEntryId: "entry-7", tokensBefore: 1234, usage: { input: 1 } } });
	assert.equal(calls[0].preparation, preparation);
	assert.deepEqual([calls[0].model, calls[0].apiKey, calls[0].signal, calls[0].thinkingLevel], [REMOTE, undefined, signal, "medium"]);
	const sent = JSON.stringify(h.seen.streamed[0].context);
	assert.ok(!sent.includes(EMAIL), "the summary request must not carry the value");
	assert.match(sent, /mail <EMAIL_1>[\s\S]*keep <EMAIL_1>/);
	assert.equal(h.seen.streamed[0].context.systemPrompt, "Summarize. Never name <EMAIL_1>.", "its prompt is screened for known values, like any prompt");
	assert.deepEqual(h.seen.streamed[0].options, { apiKey: undefined, maxTokens: 100 });
	const stored = { role: "compactionSummary", summary: result.compaction.summary, tokensBefore: 1234, timestamp: 9 };
	assert.equal((await h.send(stored))[0].summary, "The user wants to mail <EMAIL_1>.", "and goes back out as the model wrote it");
});

test("compaction that cannot be screened is cancelled rather than done in the clear", async () => {
	const calls: any[] = [];
	const h = harness({ importPi: fakeCompaction(calls) });
	await h.start();
	h.detector.down = "the detector is not running at /x/opf.sock";
	const preparation = { firstKeptEntryId: "e", messagesToSummarize: [], turnPrefixMessages: [], tokensBefore: 1 };

	const result = await h.emit("session_before_compact", { preparation, branchEntries: [], reason: "threshold", signal: new AbortController().signal });

	assert.deepEqual(result, { cancel: true });
	assert.equal(h.seen.streamed.length, 0);
	assert.match(h.notes.at(-1)![1], /compaction was cancelled[\s\S]*not running/i);
});

test("compaction is also cancelled when one text cannot be checked: a summary without it would lose it for good", async () => {
	const h = harness({ importPi: fakeCompaction([]) });
	await h.start();
	const preparation = { firstKeptEntryId: "e", messagesToSummarize: [user("UNCHECKABLE bytes")], turnPrefixMessages: [], tokensBefore: 1 };

	const result = await h.emit("session_before_compact", { preparation, branchEntries: [], reason: "manual", signal: new AbortController().signal });

	assert.deepEqual(result, { cancel: true });
	assert.equal(h.seen.streamed.length, 0);
	assert.match(h.notes.at(-1)![1], /compaction was cancelled[\s\S]*does not survive tokenization/);
});

test("in a summary request the conversation is material to check: a placeholder in it is text, not a reference", async () => {
	const h = harness({ importPi: fakeCompaction([]) });
	await h.start();
	await h.send(user(`mail ${EMAIL}`));
	const preparation = { firstKeptEntryId: "e", messagesToSummarize: [user("the docs say: write <EMAIL_1> for the address")], turnPrefixMessages: [], tokensBefore: 1 };

	await h.emit("session_before_compact", { preparation, branchEntries: [], reason: "manual", signal: new AbortController().signal });

	assert.match(h.seen.streamed[0].context.messages[0].content[0].text, /write <LITERAL_1> for the address/);
});

test("a summary goes back out as the model wrote it, also where nothing would find the value in it again", async () => {
	const detect = (value: string): Span[] => {
		const at = value.indexOf("colleague Xu");
		return at === -1 ? [] : [{ start: at + 10, end: at + 12, label: "private_person" }];
	};
	const h = harness({ detect, importPi: fakeCompaction([], { summary: "<PERSON_1> is to be asked first." }) });
	await h.start();
	const preparation = { firstKeptEntryId: "e", messagesToSummarize: [user("ask my colleague Xu")], turnPrefixMessages: [], tokensBefore: 1 };

	const result = await h.emit("session_before_compact", { preparation, branchEntries: [], reason: "manual", signal: new AbortController().signal });
	const out = await h.send({ role: "compactionSummary", summary: result.compaction.summary, tokensBefore: 1, timestamp: 9 });

	assert.equal(result.compaction.summary, "Xu is to be asked first.");
	assert.equal(out[0].summary, "<PERSON_1> is to be asked first.");
});

test("compaction is left to Pi when the screen is not masking", async () => {
	const calls: any[] = [];
	const h = harness({ importPi: fakeCompaction(calls), config: { mode: "report" } });
	await h.start();

	assert.equal(await h.emit("session_before_compact", { preparation: {}, branchEntries: [], reason: "manual", signal: new AbortController().signal }), undefined);
	assert.equal(calls.length, 0);
});

test("a branch summary is made from masked text, too", async () => {
	const calls: any[] = [];
	const h = harness({ importPi: fakeCompaction(calls) });
	await h.start();
	const signal = new AbortController().signal;
	const entries = [{ type: "message", id: "1", message: user(`mail ${EMAIL}`) }];
	const preparation = { targetId: "t", entriesToSummarize: entries, userWantsSummary: true, customInstructions: "short", replaceInstructions: false };

	const result = await h.emit("session_before_tree", { preparation, signal });
	const unwanted = await h.emit("session_before_tree", { preparation: { ...preparation, userWantsSummary: false }, signal });

	assert.deepEqual(result, { summary: { summary: `A side branch about ${EMAIL}.`, usage: undefined, details: { readFiles: ["a.txt"], modifiedFiles: [] } } });
	assert.equal(calls[0].entries, entries);
	assert.deepEqual(
		{ ...calls[0].options, streamFn: undefined },
		{ model: REMOTE, signal, thinkingLevel: "medium", customInstructions: "short", replaceInstructions: false, streamFn: undefined },
	);
	assert.ok(!JSON.stringify(h.seen.streamed[0].context).includes(EMAIL));
	assert.equal(unwanted, undefined);
	assert.equal(calls.length, 1);
});

test("a branch summary that cannot be screened, or that fails, is cancelled rather than left to be made in the clear", async () => {
	const signal = new AbortController().signal;
	const preparation = { targetId: "t", entriesToSummarize: [{ type: "message", id: "1", message: user(`mail ${EMAIL}`) }], userWantsSummary: true };
	const down = harness({ importPi: fakeCompaction([]) });
	await down.start();
	down.detector.down = "the detector is not running at /x/opf.sock";
	const failing = harness({ importPi: fakeCompaction([], { branch: { error: "the model refused" } }) });
	await failing.start();
	const stopped = harness({ importPi: fakeCompaction([], { branch: { aborted: true } }) });
	await stopped.start();

	assert.deepEqual(await down.emit("session_before_tree", { preparation, signal }), { cancel: true });
	assert.equal(down.seen.streamed.length, 0);
	assert.match(down.notes.at(-1)![1], /branch summary was cancelled[\s\S]*not running/);
	assert.deepEqual(await failing.emit("session_before_tree", { preparation, signal }), { cancel: true });
	assert.match(failing.notes.at(-1)![1], /branch summary was cancelled[\s\S]*the model refused/);
	assert.deepEqual(await stopped.emit("session_before_tree", { preparation, signal }), { cancel: true });
	assert.ok(!stopped.notes.some(([, message]) => /cancelled/.test(message)), "stopping it is the user's own doing and needs no notice");
});
