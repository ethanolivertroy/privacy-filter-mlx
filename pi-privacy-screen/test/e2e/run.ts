import { spawn, spawnSync } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { createServer, request as httpRequest } from "node:http";
import type { AddressInfo } from "node:net";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const packageDir = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const opf = join(packageDir, "..", ".venv", "bin", "opf-mlx");
const checkpoint = process.env.OPF_CHECKPOINT ?? join(homedir(), ".opf", "privacy_filter");
const root = mkdtempSync("/tmp/opf-e2e-"); // short: it holds Unix sockets
const serveCommand = [opf, "serve", "--checkpoint", checkpoint, "--no-download"];

const EMAIL = "lena.fischer@example.com";
const EMAIL_2 = "omar.haddad@example.com";
const TOKEN = ["ghp_", "x7KqLm29ZpQrStUvWxYz", "0123456789AbCdEf"].join("");
const HOSTILE = "it's; touch PWNED; echo '";

const results: { scenario: string; ok: boolean; name: string; detail?: string }[] = [];
const notes: string[] = [];
let current = "";
function check(name: string, ok: boolean, detail?: string): void {
	results.push({ scenario: current, ok, name, detail });
	console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${!ok && detail ? `\n       ${detail.split("\n").join("\n       ")}` : ""}`);
}
function note(text: string): void {
	notes.push(`${current}: ${text}`);
	console.log(`  note ${text}`);
}
const short = (value: unknown, length = 400) => {
	const text = typeof value === "string" ? value : (JSON.stringify(value) ?? String(value));
	return text.length > length ? `${text.slice(0, length)}…` : text;
};

interface Reply {
	text?: string;
	toolCalls?: { name: string; arguments: object }[];
	pause?: number;
}
type Script = (body: any, index: number) => Reply;

async function startProvider() {
	const requests: { raw: string; body: any }[] = [];
	let script: Script = () => ({ text: "ok" });
	const server = createServer(async (request, response) => {
		const chunks: Buffer[] = [];
		for await (const chunk of request) chunks.push(chunk as Buffer);
		const raw = Buffer.concat(chunks).toString("utf8");
		if (request.method !== "POST" || !request.url?.endsWith("/chat/completions")) {
			response.writeHead(404, { "Content-Type": "application/json" }).end("{}");
			return;
		}
		const body = JSON.parse(raw);
		requests.push({ raw, body });
		let reply: Reply;
		try {
			reply = script(body, requests.length - 1);
		} catch (error) {
			reply = { text: `script error: ${String(error)}` };
		}
		const id = `chatcmpl-${requests.length}`;
		const calls = reply.toolCalls ?? [];
		const toolCalls = calls.map((call, index) => ({ index, id: `call_${requests.length}_${index}`, type: "function", function: { name: call.name, arguments: JSON.stringify(call.arguments) } }));
		const usage = { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 };
		if (body.stream === false) {
			const message = { role: "assistant", content: reply.text ?? null, ...(calls.length ? { tool_calls: toolCalls.map(({ index: _, ...call }) => call) } : {}) };
			response.writeHead(200, { "Content-Type": "application/json" });
			response.end(JSON.stringify({ id, object: "chat.completion", created: 0, model: body.model, choices: [{ index: 0, message, finish_reason: calls.length ? "tool_calls" : "stop" }], usage }));
			return;
		}
		const event = (choices: object[], more: object = {}) => `data: ${JSON.stringify({ id, object: "chat.completion.chunk", created: 0, model: body.model, choices, ...more })}\n\n`;
		const delta = (content: object, finish: string | null = null) => event([{ index: 0, delta: content, finish_reason: finish }]);
		response.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
		response.write(delta({ role: "assistant", content: "" }));
		for (const piece of (reply.text ?? "").match(/[\s\S]{1,5}/g) ?? []) response.write(delta({ content: piece }));
		if (reply.pause) await new Promise((resolve) => setTimeout(resolve, reply.pause));
		for (const call of toolCalls) response.write(delta({ tool_calls: [call] }));
		response.write(delta({}, calls.length ? "tool_calls" : "stop"));
		response.write(event([], { usage }));
		response.end("data: [DONE]\n\n");
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const port = (server.address() as AddressInfo).port;
	return {
		baseUrl: `http://127.0.0.1:${port}/v1`,
		requests,
		play(next: Script) {
			script = next;
		},
		close: () => new Promise<void>((resolve) => server.close(() => resolve())),
	};
}
type Provider = Awaited<ReturnType<typeof startProvider>>;

const sentIn = (provider: Provider, from = 0) => provider.requests.slice(from).map((request) => request.raw).join("\n");
const toolMessages = (body: any) => (body.messages as any[]).filter((message) => message.role === "tool");
const textOf = (message: any): string => (typeof message?.content === "string" ? message.content : (message?.content ?? []).map((part: any) => part.text ?? "").join(""));
const isSummaryRequest = (body: any) => !body.tools?.length;

async function startTap(target: string, name: string) {
	const asked: string[][] = [];
	const path = join(root, `${name}.sock`);
	const server = createServer(async (request, response) => {
		const chunks: Buffer[] = [];
		for await (const chunk of request) chunks.push(chunk as Buffer);
		const body = Buffer.concat(chunks);
		if (request.url === "/v1/detect") asked.push(JSON.parse(body.toString("utf8")).texts);
		const onward = httpRequest({ socketPath: target, method: request.method, path: request.url, headers: request.headers }, (answer) => {
			response.writeHead(answer.statusCode ?? 502, answer.headers);
			answer.pipe(response);
		});
		onward.on("error", () => response.writeHead(502).end());
		onward.end(body);
	});
	await new Promise<void>((resolve) => server.listen(path, resolve));
	return {
		socket: path,
		asked,
		close: () => new Promise<void>((resolve) => (server.closeAllConnections(), server.close(() => resolve()))),
	};
}

class Pi {
	readonly events: any[] = [];
	stderr = "";
	private readonly child: ChildProcess;
	private buffer = "";
	private counter = 0;
	private readonly waiters: { test: (record: any) => boolean; resolve: (record: any) => void }[] = [];
	exited = false;

	constructor(args: string[], env: Record<string, string>, cwd: string) {
		this.child = spawn("pi", ["--mode", "rpc", ...args], { cwd, env, stdio: ["pipe", "pipe", "pipe"] });
		this.child.stdout!.setEncoding("utf8");
		this.child.stdout!.on("data", (data: string) => {
			this.buffer += data;
			for (let end = this.buffer.indexOf("\n"); end !== -1; end = this.buffer.indexOf("\n")) {
				const line = this.buffer.slice(0, end).replace(/\r$/, "");
				this.buffer = this.buffer.slice(end + 1);
				if (line === "") continue;
				let record: any;
				try {
					record = JSON.parse(line);
				} catch {
					record = { type: "unparsed", line };
				}
				this.events.push(record);
				for (const waiter of [...this.waiters]) {
					if (!waiter.test(record)) continue;
					this.waiters.splice(this.waiters.indexOf(waiter), 1);
					waiter.resolve(record);
				}
			}
		});
		this.child.stderr!.setEncoding("utf8");
		this.child.stderr!.on("data", (data: string) => (this.stderr += data));
		this.child.on("exit", () => (this.exited = true));
	}

	waitFor(test: (record: any) => boolean, seconds: number, from = 0, what = "a record"): Promise<any> {
		const seen = this.events.slice(from).find(test);
		if (seen) return Promise.resolve(seen);
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => reject(new Error(`timed out after ${seconds} s waiting for ${what}; stderr: ${short(this.stderr, 600)}`)), seconds * 1000);
			this.waiters.push({ test, resolve: (record) => (clearTimeout(timer), resolve(record)) });
		});
	}

	async send(command: Record<string, unknown>, seconds = 60): Promise<any> {
		const id = `c${++this.counter}`;
		const from = this.events.length;
		this.child.stdin!.write(`${JSON.stringify({ id, ...command })}\n`);
		return this.waitFor((record) => record.type === "response" && record.id === id, seconds, from, `the response to ${command.type}`);
	}

	async prompt(message: string, seconds = 120): Promise<{ disposition: string | undefined; records: any[] }> {
		const from = this.events.length;
		const response = await this.send({ type: "prompt", message });
		const disposition = response.data?.disposition as string | undefined;
		if (response.success && disposition !== "handled") await this.waitFor((record) => record.type === "agent_settled", seconds, from, "agent_settled");
		await this.drainTrailingNotices();
		return { disposition, records: this.events.slice(from) };
	}

	private drainTrailingNotices(): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, 150));
	}

	notices(from = 0): { level: string; message: string }[] {
		return this.events
			.slice(from)
			.filter((record) => record.type === "extension_ui_request" && record.method === "notify")
			.map((record) => ({ level: record.notifyType ?? "info", message: String(record.message) }));
	}
	status(): string | undefined {
		return this.events.filter((record) => record.type === "extension_ui_request" && record.method === "setStatus" && record.statusKey === "privacy").at(-1)?.statusText;
	}
	async messages(): Promise<any[]> {
		return (await this.send({ type: "get_messages" })).data?.messages ?? [];
	}

	async close(): Promise<void> {
		if (this.exited) return;
		const gone = new Promise<void>((resolve) => this.child.on("exit", () => resolve()));
		this.child.stdin!.end();
		const timer = setTimeout(() => this.child.kill("SIGKILL"), 15_000);
		await gone;
		clearTimeout(timer);
	}
}

interface Setup {
	config?: object;
	settings?: object;
	files?: Record<string, string>;
	discover?: boolean;
	more?: string[];
	work?: string;
	env?: Record<string, string>;
}

function prepare(name: string, provider: Provider, setup: Setup = {}) {
	const agent = join(root, `agent-${name}`);
	const home = join(root, `home-${name}`);
	const work = setup.work ?? join(root, `work-${name}`);
	const sessions = join(root, `sessions-${name}`);
	for (const directory of [agent, home, work, sessions]) mkdirSync(directory, { recursive: true });
	writeFileSync(
		join(agent, "models.json"),
		JSON.stringify({
			providers: {
				pretend: {
					baseUrl: provider.baseUrl,
					api: "openai-completions",
					apiKey: "not-a-key",
					models: [{ id: "echo", name: "Echo", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 200000, maxTokens: 4096 }],
				},
			},
		}),
	);
	if (setup.config) writeFileSync(join(agent, "privacy-screen.json"), JSON.stringify(setup.config));
	if (setup.settings) writeFileSync(join(agent, "settings.json"), JSON.stringify(setup.settings));
	for (const [file, content] of Object.entries(setup.files ?? {})) writeFileSync(join(work, file), content);
	if (setup.discover) {
		mkdirSync(join(agent, "extensions"), { recursive: true });
		if (!existsSync(join(agent, "extensions", "privacy-screen"))) symlinkSync(packageDir, join(agent, "extensions", "privacy-screen"));
	}
	const env: Record<string, string> = {
		PATH: process.env.PATH ?? "",
		HOME: home,
		TMPDIR: process.env.TMPDIR ?? "/tmp",
		LANG: "en_US.UTF-8",
		TERM: "dumb",
		PI_CODING_AGENT_DIR: agent,
		PI_OFFLINE: "1",
		PI_SKIP_VERSION_CHECK: "1",
		PI_TELEMETRY: "0",
		...setup.env,
	};
	const args = [
		"--model", "pretend/echo",
		...(setup.discover ? [] : ["-ne", "-e", packageDir]),
		"-ns", "-np", "-nc", "--no-mcp", "--no-themes", "--no-approve", "--offline",
		"--session-dir", sessions,
		...(setup.more ?? []),
	];
	return { args, env, work, agent, sessions };
}

function startPi(name: string, provider: Provider, setup: Setup = {}) {
	const made = prepare(name, provider, setup);
	return { pi: new Pi(made.args, made.env, made.work), ...made };
}

class Terminal {
	private readonly server: string;
	constructor(name: string, args: string[], env: Record<string, string>, cwd: string) {
		this.server = `opf-e2e-${process.pid}-${name}`;
		const command = ["pi", ...args].map((arg) => `'${arg.replaceAll("'", `'\\''`)}'`).join(" ");
		const made = spawnSync("tmux", ["-L", this.server, "-f", "/dev/null", "new-session", "-d", "-s", "pi", "-x", "118", "-y", "42", "-c", cwd, command], { env: { ...env, TERM: "xterm-256color" }, encoding: "utf8" });
		if (made.status !== 0) throw new Error(`tmux could not start pi: ${made.stderr}`);
	}
	private tmux(...args: string[]) {
		return spawnSync("tmux", ["-L", this.server, ...args], { encoding: "utf8" });
	}
	screen(): string {
		return this.tmux("capture-pane", "-p", "-J", "-t", "pi").stdout ?? "";
	}
	type(text: string): void {
		this.tmux("send-keys", "-t", "pi", "-l", text);
	}
	enter(): void {
		this.tmux("send-keys", "-t", "pi", "Enter");
	}
	async waitFor(test: (screen: string) => boolean, seconds: number, what: string): Promise<string> {
		const deadline = Date.now() + seconds * 1000;
		for (;;) {
			const screen = this.screen();
			if (test(screen)) return screen;
			if (Date.now() > deadline) throw new Error(`timed out after ${seconds} s waiting for ${what}; the screen:\n${screen.trimEnd()}`);
			await new Promise((resolve) => setTimeout(resolve, 150));
		}
	}
	close(): void {
		this.tmux("kill-server");
	}
}

const socket = join(root, "d.sock");
const masking = (more: object = {}) => ({ screenLocalModels: true, server: { socket, command: serveCommand, idleExit: 900, startTimeout: 120 }, ...more });

const scenarios: Record<string, (provider: Provider) => Promise<void>> = {
	async mask(provider) {
		provider.play((body) => {
			const all = JSON.stringify(body.messages);
			const email = /<EMAIL_\d+>/.exec(all)?.[0] ?? "NO-EMAIL-PLACEHOLDER";
			const secret = /<SECRET_\d+>/.exec(all)?.[0] ?? "NO-SECRET-PLACEHOLDER";
			const done = toolMessages(body).length;
			if (done === 0) return { toolCalls: [{ name: "write", arguments: { path: "contact.txt", content: `${email}\n` } }] };
			if (done === 1) return { toolCalls: [{ name: "bash", arguments: { command: `echo "token: ${secret}" >> deploy.log` } }] };
			return { text: `Saved ${email} and logged ${secret}.` };
		});
		const { pi, work, sessions } = startPi("mask", provider, { config: masking() });
		try {
			const started = Date.now();
			const { disposition } = await pi.prompt(`Please save the address ${EMAIL} to contact.txt, then log the deploy token ${TOKEN} to deploy.log with echo.`, 240);
			note(`first prompt, including the detector's start and model load: ${((Date.now() - started) / 1000).toFixed(1)} s (disposition ${disposition})`);
			const sent = sentIn(provider);
			check("three requests reached the provider", provider.requests.length === 3, `${provider.requests.length} requests; stderr: ${short(pi.stderr)}`);
			check("no request carries the address", !sent.includes(EMAIL));
			check("no request carries the token", !sent.includes(TOKEN));
			const first = provider.requests[0]?.body;
			const typed = textOf(first?.messages.findLast((message: any) => message.role === "user"));
			check("the prompt went out with placeholders", /<EMAIL_1>/.test(typed) && /<SECRET_1>/.test(typed), short(typed));
			check("the system prompt explains placeholders", /numbered placeholders/.test(JSON.stringify(first?.messages.filter((message: any) => message.role !== "user"))));
			check("the file a tool wrote holds the real address", existsSync(join(work, "contact.txt")) && readFileSync(join(work, "contact.txt"), "utf8") === `${EMAIL}\n`, existsSync(join(work, "contact.txt")) ? readFileSync(join(work, "contact.txt"), "utf8") : "no file");
			check("the command a tool ran got the real token", existsSync(join(work, "deploy.log")) && readFileSync(join(work, "deploy.log"), "utf8") === `token: ${TOKEN}\n`, existsSync(join(work, "deploy.log")) ? readFileSync(join(work, "deploy.log"), "utf8") : "no file");

			const stored = await pi.messages();
			const reply = textOf(stored.findLast((message) => message.role === "assistant"));
			check("the stored reply has the real values", reply === `Saved ${EMAIL} and logged ${TOKEN}.`, short(reply));
			const calls = stored.flatMap((message) => (message.role === "assistant" ? message.content.filter((part: any) => part.type === "toolCall") : []));
			note(`stored tool-call arguments: ${short(calls.map((call: any) => call.arguments))}`);
			const third = provider.requests[2]?.body;
			const echoed = (third?.messages ?? []).filter((message: any) => message.role === "assistant").flatMap((message: any) => message.tool_calls ?? []).map((call: any) => call.function.arguments);
			check("tool calls go back to the provider as it wrote them", JSON.stringify(echoed) === JSON.stringify(['{"path":"contact.txt","content":"<EMAIL_1>\\n"}', '{"command":"echo \\"token: <SECRET_1>\\" >> deploy.log"}']), short(echoed));
			const streamed = pi.events.filter((record) => record.type === "message_end" && record.message?.role === "assistant").map((record) => textOf(record.message)).filter(Boolean);
			note(`assistant text in message_end records (what a client of the event stream sees): ${short(streamed)}`);
			const problems = pi.notices().filter((notice) => notice.level !== "info");
			check("no warning or error was raised", problems.length === 0, short(problems));
			check("the status line says it is masking", /mask \(\d+ hidden\)/.test(pi.status() ?? ""), pi.status());
			check("the session file exists", readdirSync(sessions, { recursive: true }).some((name) => String(name).endsWith(".jsonl")), short(readdirSync(sessions, { recursive: true })));

			const listFrom = pi.events.length;
			const listed = await pi.prompt("/privacy list");
			const list = pi.notices(listFrom).map((notice) => notice.message).join("\n");
			check("/privacy list runs as a command and names the placeholders", listed.disposition === "handled" && list.includes("<EMAIL_1>") && list.includes("<SECRET_1>") && !list.includes(EMAIL), `disposition ${listed.disposition}; ${short(list)}`);
			check("a command does not reach the provider", provider.requests.length === 3);
		} finally {
			await pi.close();
		}
	},

	async resume(provider) {
		const before = provider.requests.at(-1)?.body.messages.filter((message: any) => message.role === "assistant" || message.role === "tool");
		const mark = provider.requests.length;
		provider.play((body) => (isSummaryRequest(body) ? { text: "## Summary\nThe user saved <EMAIL_1> and logged <SECRET_1>; next comes <EMAIL_2>." } : { text: `Noted ${/<EMAIL_2>/.exec(JSON.stringify(body.messages))?.[0] ?? "nothing"}.` }));
		const tap = await startTap(socket, "tap");
		const { pi } = startPi("mask", provider, { config: masking({ server: { socket: tap.socket, autoStart: false } }), more: ["-c"], settings: { compaction: { keepRecentTokens: 1 } } });
		try {
			for (const deadline = Date.now() + 30_000; tap.asked.length === 0 && Date.now() < deadline; ) await new Promise((resolve) => setTimeout(resolve, 100));
			await new Promise((resolve) => setTimeout(resolve, 500));
			const early = tap.asked.length;
			check("a resumed conversation is checked before its first request is made", tap.asked.flat().some((text) => text.includes(EMAIL)) && provider.requests.length === mark, `${early} detector requests with ${tap.asked.flat().length} texts; ${provider.requests.length - mark} provider requests`);
			await pi.prompt(`Also write to ${EMAIL_2}, and remind me of the first address (${EMAIL}).`);
			const late = tap.asked.slice(early).flat();
			check("so that request asks the detector only for what is new", late.length === 1 && late[0]!.includes(EMAIL_2), `${late.length} texts: ${short(late.map((text) => text.slice(0, 60)))}`);
			note(`on resume the detector was asked about ${tap.asked.slice(0, early).flat().length} texts before anything was sent, and about ${late.length} for the first request`);
			const request = provider.requests[mark]?.body;
			const sent = sentIn(provider, mark);
			check("the continued session reached the provider", request !== undefined, short(pi.stderr));
			check("nothing real in the continued request", !sent.includes(EMAIL) && !sent.includes(EMAIL_2) && !sent.includes(TOKEN));
			const typed = textOf(request?.messages.findLast((message: any) => message.role === "user"));
			check("an old value keeps its placeholder and a new one gets the next", /<EMAIL_2>/.test(typed) && /<EMAIL_1>/.test(typed), short(typed));
			const again = request?.messages.filter((message: any) => message.role === "assistant" || message.role === "tool").slice(0, before?.length);
			check("earlier turns go out exactly as before the restart", before?.length > 0 && JSON.stringify(again) === JSON.stringify(before), `before: ${short(before, 700)}\nafter:  ${short(again, 700)}`);

			const compactFrom = provider.requests.length;
			const response = await pi.send({ type: "compact", customInstructions: `Keep ${EMAIL} in mind` }, 180);
			const summaryRequests = provider.requests.slice(compactFrom);
			const summarySent = sentIn(provider, compactFrom);
			check("compaction succeeded", response.success === true, short(response));
			check("compaction asked the provider for a summary", summaryRequests.length >= 1 && summaryRequests.every((request) => isSummaryRequest(request.body)), `${summaryRequests.length} requests`);
			check("the summary request carries nothing real", summaryRequests.length >= 1 && !summarySent.includes(EMAIL) && !summarySent.includes(EMAIL_2) && !summarySent.includes(TOKEN));
			check("the summary request carries the conversation with placeholders", /<EMAIL_1>/.test(summarySent) && /<SECRET_1>/.test(summarySent), short(summarySent, 600));
			check("in it, a value written inside an escaped string keeps the placeholder it has", /echo \\\\\\"token: <SECRET_1>\\\\\\" >> deploy\.log/.test(summarySent) && !/<SECRET_2>/.test(summarySent), short(/.{0,60}deploy\.log.{0,10}/.exec(summarySent.slice(summarySent.indexOf("bash(")))?.[0]));
			check("the stored summary has the real values", String(response.data?.summary ?? "").includes(EMAIL) && String(response.data?.summary ?? "").includes(TOKEN), short(response.data?.summary));
			const afterFrom = provider.requests.length;
			await pi.prompt("And now?");
			const after = sentIn(provider, afterFrom);
			check("after compaction the summary goes out masked", provider.requests.length > afterFrom && after.includes("The user saved <EMAIL_1> and logged <SECRET_1>") && !after.includes(EMAIL) && !after.includes(TOKEN), short(after, 900));
			const problems = pi.notices().filter((notice) => notice.level !== "info");
			check("no warning or error was raised", problems.length === 0, short(problems));
		} finally {
			await pi.close();
			await tap.close();
		}
	},

	async door(provider) {
		const { pi } = startPi("door", provider, { config: { screenLocalModels: true, server: { socket: join(root, "nobody.sock"), autoStart: false } } });
		try {
			const mark = provider.requests.length;
			const { disposition, records } = await pi.prompt(`Mail ${EMAIL} please.`);
			const notices = pi.notices();
			check("nothing reached the provider", provider.requests.length === mark, sentIn(provider, mark).slice(0, 300));
			check("the prompt was handled without a run", disposition === "handled", String(disposition));
			check("the user is told why, and how to go on", notices.some((notice) => notice.level === "error" && /not running/.test(notice.message) && /not sent/.test(notice.message) && /\/privacy off/.test(notice.message)), short(notices));
			const editor = records.find((record) => record.type === "extension_ui_request" && record.method === "set_editor_text");
			check("the text is given back to the editor", editor?.text === `Mail ${EMAIL} please.`, short(editor));
			check("the status line says the detector is down", /DETECTOR DOWN/.test(pi.status() ?? ""), pi.status());

			await pi.prompt("/privacy off");
			provider.play(() => ({ text: "Sent unscreened." }));
			await pi.prompt(`Mail ${EMAIL} please.`);
			check("/privacy off lets a message through as it is", provider.requests.length === mark + 1 && sentIn(provider, mark).includes(EMAIL));
		} finally {
			await pi.close();
		}
	},

	async midrun(provider) {
		const own = join(root, "own.sock");
		const detector = spawn(opf, ["serve", "--checkpoint", checkpoint, "--no-download", "--socket", own, "--idle-exit", "300"], { stdio: ["ignore", "pipe", "pipe"] });
		let output = "";
		detector.stdout.on("data", (data) => (output += data));
		detector.stderr.on("data", (data) => (output += data));
		for (let waited = 0; !output.includes("listening on") && waited < 120; waited++) await new Promise((resolve) => setTimeout(resolve, 1000));
		check("a detector for this scenario came up", output.includes("listening on"), short(output));
		provider.play((body) => (toolMessages(body).length === 0 ? { toolCalls: [{ name: "bash", arguments: { command: "sleep 3; cat notes.txt" } }] } : { text: "Read it." }));
		const { pi } = startPi("midrun", provider, {
			config: { screenLocalModels: true, server: { socket: own, autoStart: false } },
			files: { "notes.txt": `Call Omar at ${EMAIL_2} about invoice 4471.\n` },
		});
		try {
			const mark = provider.requests.length;
			const killed = pi.waitFor((record) => record.type === "tool_execution_start", 120, 0, "the tool to start").then(() => detector.kill("SIGKILL"));
			const { records } = await pi.prompt("Read notes.txt and tell me what it says.", 180);
			await killed;
			const sent = sentIn(provider, mark);
			const requests = provider.requests.length - mark;
			check("the tool result was not sent unchecked", !sent.includes(EMAIL_2) && !sent.includes("invoice 4471"), short(sent.slice(-600)));
			check("the user is told the request was stopped", pi.notices().some((notice) => notice.level === "error" && /stopped/.test(notice.message)), short(pi.notices()));
			const last = records.findLast((record) => record.type === "message_end" && record.message?.role === "assistant")?.message;
			note(`after ctx.abort() in context_with_system: ${requests} request(s) reached the provider; the last assistant message has stopReason "${last?.stopReason}"${requests > 1 ? `; the second request carried ${sent.includes("withheld") ? "the withheld notice" : "no notice"}` : ""}`);
		} finally {
			detector.kill("SIGKILL");
			await pi.close();
		}
	},

	async report(provider) {
		const quoting = "Fine. A template would have <EMAIL_1> there.";
		provider.play(() => ({ text: quoting }));
		const { pi } = startPi("report", provider, { config: masking({ mode: "report" }) });
		try {
			const mark = provider.requests.length;
			await pi.prompt(`Mail ${EMAIL} please.`);
			check("in report mode the request goes out as it is", provider.requests.length === mark + 1 && sentIn(provider, mark).includes(EMAIL));
			check("the status line counts what was found", /report only \(1 found\)/.test(pi.status() ?? ""), pi.status());
			const reply = textOf((await pi.messages()).findLast((message) => message.role === "assistant"));
			check("and the reply is stored as the model wrote it: it saw the real text, so a placeholder in it is a quote", reply === quoting, short(reply));
			const from = pi.events.length;
			await pi.prompt("/privacy mask");
			await pi.prompt(`Mail ${EMAIL} again.`);
			const later = provider.requests.at(-1)?.raw ?? "";
			check("after /privacy mask the same value is hidden, in the old turn too", provider.requests.length === mark + 2 && !later.includes(EMAIL) && /<EMAIL_1>/.test(later), short(later, 600));
			note(`notices after switching: ${short(pi.notices(from).map((notice) => notice.message))}`);
		} finally {
			await pi.close();
		}
	},

	async hostile(provider) {
		provider.play((body) => {
			const tools = toolMessages(body);
			if (tools.length === 0) return { toolCalls: [{ name: "bash", arguments: { command: "cat .env" } }] };
			const secret = /<SECRET_\d+>/.exec(textOf(tools[0]))?.[0] ?? "NO-PLACEHOLDER";
			if (tools.length === 1) return { toolCalls: [{ name: "bash", arguments: { command: `printf '%s' ${secret} > out.txt; printf '%s' "${secret}" > quoted.txt` } }] };
			if (tools.length === 2) return { toolCalls: [{ name: "bash", arguments: { command: `echo \`echo ${secret}\`` } }] };
			return { text: "Done." };
		});
		const { pi, work } = startPi("hostile", provider, { config: masking(), files: { ".env": `DB_PASSWORD="${HOSTILE}"\nPORT=5432\n` } });
		try {
			const mark = provider.requests.length;
			await pi.prompt("Put the database password from .env into out.txt.");
			const sent = sentIn(provider, mark);
			const read = (file: string) => (existsSync(join(work, file)) ? readFileSync(join(work, file), "utf8") : "(no file)");
			check("the value from .env never reached the provider", !sent.includes("touch PWNED"), short(sent.slice(-500)));
			check("the tool result went out with a placeholder", /DB_PASSWORD=\\"<SECRET_\d+>\\"/.test(sent), short(textOf(toolMessages(provider.requests[mark + 1]?.body ?? { messages: [] })[0])));
			check("unquoted in a command, the value arrived as one plain argument", read("out.txt") === HOSTILE, read("out.txt"));
			check("in double quotes, too", read("quoted.txt") === HOSTILE, read("quoted.txt"));
			check("nothing in the value ran", !existsSync(join(work, "PWNED")), readdirSync(work).join(" "));
			const refusal = textOf(toolMessages(provider.requests.at(-1)?.body ?? { messages: [] }).at(-1));
			check("a command that cannot take the value safely is refused, and the model is told why", /backquotes/.test(refusal) && /Nothing was run/.test(refusal), short(refusal));
			check("the refusal names the placeholder the model used", /the value of SECRET_\d+ /.test(refusal) && !/LITERAL/.test(refusal), short(refusal));
		} finally {
			await pi.close();
		}
	},

	async literal(provider) {
		provider.play((body) => {
			const tools = toolMessages(body);
			if (tools.length === 0) return { toolCalls: [{ name: "bash", arguments: { command: "cat template.txt" } }] };
			const literal = /<LITERAL_\d+>/.exec(textOf(tools[0]))?.[0] ?? "NO-LITERAL";
			if (tools.length === 1) return { toolCalls: [{ name: "write", arguments: { path: "copy.txt", content: `Dear ${literal},\n` } }] };
			if (tools.length === 2) return { toolCalls: [{ name: "bash", arguments: { command: `printf '%s' ${literal} > shell.txt` } }] };
			return { text: `Copied ${literal}.` };
		});
		const { pi, work } = startPi("literal", provider, { config: masking(), files: { "template.txt": "Dear <EMAIL_1>,\n" } });
		try {
			const mark = provider.requests.length;
			await pi.prompt(`Copy template.txt to copy.txt and its placeholder to shell.txt. My own address is ${EMAIL}; the template writes it <EMAIL_1>.`);
			const sent = sentIn(provider, mark);
			const read = (file: string) => (existsSync(join(work, file)) ? readFileSync(join(work, file), "utf8") : "(no file)");
			const typed = textOf(provider.requests[mark]?.body.messages.findLast((message: any) => message.role === "user"));
			check("the address is <EMAIL_1> to the model", /My own address is <EMAIL_1>;/.test(typed) && !sent.includes(EMAIL), short(typed));
			check("the same text typed by the user is a literal, too", /the template writes it <LITERAL_1>\./.test(typed), short(typed));
			const shown = textOf(toolMessages(provider.requests[mark + 1]?.body ?? { messages: [] })[0]);
			check("the file's own <EMAIL_1> goes out as a literal, so that it cannot be taken for the address", /Dear <LITERAL_1>,/.test(shown), short(shown));
			check("written by a tool, it is the file's text again, and not the address", read("copy.txt") === "Dear <EMAIL_1>,\n", read("copy.txt"));
			check("in a command, too", read("shell.txt") === "<EMAIL_1>", read("shell.txt"));
			const reply = textOf((await pi.messages()).findLast((message) => message.role === "assistant"));
			check("and in the stored reply", reply === "Copied <EMAIL_1>.", short(reply));
			const problems = pi.notices().filter((notice) => notice.level !== "info");
			check("no warning or error was raised", problems.length === 0, short(problems));
		} finally {
			await pi.close();
		}
	},

	async local(provider) {
		provider.play(() => ({ text: "Local reply." }));
		const lonely = join(root, "never.sock");
		const { pi } = startPi("local", provider, { config: { server: { socket: lonely, command: serveCommand } } });
		try {
			const mark = provider.requests.length;
			await pi.prompt(`Mail ${EMAIL} please.`);
			check("a request to a loopback model goes out as it is", provider.requests.length === mark + 1 && sentIn(provider, mark).includes(EMAIL));
			check("the status line says so", /local model, not screened/.test(pi.status() ?? ""), pi.status());
			check("no detector was started", !existsSync(lonely) && !existsSync(`${lonely}.log`));
		} finally {
			await pi.close();
		}
	},

	async tui(provider) {
		const have = spawnSync("tmux", ["-V"], { encoding: "utf8" });
		if (have.status !== 0) {
			note("tmux is not installed: the terminal UI was not looked at");
			return;
		}
		provider.play((body) => {
			const last = body.messages.at(-1);
			if (last?.role === "tool") return { text: `It says: ${/Dear <LITERAL_\d+>,/.exec(textOf(last))?.[0] ?? "NO LITERAL"}`, pause: 1500 };
			if (/template/.test(textOf(last))) return { toolCalls: [{ name: "bash", arguments: { command: "cat template.txt" } }] };
			return { text: `Will mail ${/<EMAIL_\d+>/.exec(JSON.stringify(body.messages))?.[0] ?? "nobody"} today.`, pause: 2500 };
		});
		const mark = provider.requests.length;
		const made = prepare("tui", provider, { config: masking(), discover: true, files: { "template.txt": "Dear <EMAIL_1>,\n" } });
		const terminal = new Terminal("mask", made.args, made.env, made.work);
		try {
			const start = await terminal.waitFor((screen) => /privacy: mask/.test(screen), 60, "the footer to show the screen's state");
			check("the footer shows the screen's state", /privacy: mask \(0 hidden\)/.test(start), start.trimEnd().split("\n").slice(-6).join("\n"));
			check("pi lists the extension under a name that says what it is", /\[Extensions\]\s*\n\s*privacy-screen\s*\n/.test(start), start.trimEnd().split("\n").slice(0, 12).join("\n"));
			terminal.type(`Mail ${EMAIL} please.`);
			terminal.enter();
			const streaming = await terminal.waitFor((screen) => /Will mail/.test(screen), 60, "the reply to start");
			check("while a reply streams, it is shown with its values", streaming.includes(`Will mail ${EMAIL}`) && !/<EMAIL_\d+>/.test(streaming), streaming.trimEnd());
			const done = await terminal.waitFor((screen) => /privacy: mask \(1 hidden\)/.test(screen) && /today\./.test(screen), 60, "the reply to end");
			await new Promise((resolve) => setTimeout(resolve, 3200));
			const settled = terminal.screen();
			check("the finished reply is shown with its values", settled.includes(`Will mail ${EMAIL} today.`) && !/<EMAIL_\d+>/.test(settled), settled.trimEnd());
			check("what the provider got has the placeholder", provider.requests.length === mark + 1 && !sentIn(provider, mark).includes(EMAIL) && /<EMAIL_1>/.test(sentIn(provider, mark)));
			terminal.type("/privacy");
			terminal.enter();
			const state = await terminal.waitFor((screen) => /mode: mask/.test(screen), 30, "the answer to /privacy");
			check("/privacy shows the state", /1 value hidden/.test(state) && /detector: running/.test(state), state.trimEnd());
			note(`the terminal UI after a masked exchange and /privacy:\n${state.trimEnd()}`);
			void done;

			terminal.type("Show me the template file.");
			terminal.enter();
			const quoting = await terminal.waitFor((screen) => /It says:/.test(screen), 60, "the reply about the template");
			await new Promise((resolve) => setTimeout(resolve, 3500));
			const stored = terminal.screen();
			const line = (screen: string) => screen.split("\n").find((one) => one.includes("It says:")) ?? "(no such line)";
			check("text that only looks like a placeholder is shown as that text while the reply streams", line(quoting).includes("Dear <EMAIL_1>,"), line(quoting));
			check("and stays that text once the reply is stored", line(stored).includes("Dear <EMAIL_1>,") && !line(stored).includes(EMAIL), line(stored));
		} finally {
			terminal.close();
		}

		const before = provider.requests.length;
		const closed = prepare("tuidoor", provider, { config: { screenLocalModels: true, server: { socket: join(root, "nobody.sock"), autoStart: false } } });
		const second = new Terminal("door", closed.args, closed.env, closed.work);
		try {
			await second.waitFor((screen) => /DETECTOR DOWN/.test(screen), 60, "the footer to say the detector is down");
			second.type(`Mail ${EMAIL} please.`);
			second.enter();
			const refused = await second.waitFor((screen) => /was not sent/.test(screen), 30, "the notice that the message was not sent");
			await new Promise((resolve) => setTimeout(resolve, 600));
			const after = second.screen();
			check("with the detector down, the terminal UI sends nothing", provider.requests.length === before);
			check("it says why", /not running/.test(refused) && /\/privacy off/.test(refused), refused.trimEnd());
			check("the text is back in the editor, between its two rules", new RegExp(`─{20,}\\n\\s*Mail ${EMAIL.replaceAll(".", "\\.")} please\\.\\s*\\n─{20,}`).test(after), after.trimEnd());
			check("the footer says nothing is sent", /privacy: DETECTOR DOWN, nothing is sent/.test(after), after.trimEnd().split("\n").slice(-4).join("\n"));
		} finally {
			second.close();
		}
	},

	async discovery(provider) {
		provider.play(() => ({ text: "Fine." }));
		const { pi } = startPi("discovery", provider, { config: masking(), discover: true });
		try {
			const mark = provider.requests.length;
			await pi.prompt(`Mail ${EMAIL} please.`);
			const sent = sentIn(provider, mark);
			check("a symlinked extension directory is discovered and masks", provider.requests.length === mark + 1 && !sent.includes(EMAIL) && /<EMAIL_1>/.test(sent), `${short(sent.slice(-300))}\nstderr: ${short(pi.stderr)}`);
		} finally {
			await pi.close();
		}
	},

	async defaults(provider) {
		provider.play(() => ({ text: "Fine." }));
		const own = join(root, "own.sock");
		const { pi } = startPi("defaults", provider, { config: { screenLocalModels: true, server: { socket: own, idleExit: 900, startTimeout: 120 } }, discover: true, env: { OPF_CHECKPOINT: checkpoint } });
		try {
			const mark = provider.requests.length;
			await pi.prompt(`Mail ${EMAIL} please.`);
			const sent = sentIn(provider, mark);
			check("with no command in the settings a detector is started, and the request is masked", provider.requests.length === mark + 1 && !sent.includes(EMAIL) && /<EMAIL_1>/.test(sent), `${short(sent.slice(-300))}\nnotices: ${short(pi.notices())}\nstderr: ${short(pi.stderr)}`);
			const running = spawnSync("pgrep", ["-fl", own], { encoding: "utf8" }).stdout.trim();
			check("it is the opf-mlx of the project this extension is part of", running.includes(`${opf} serve --no-download --socket ${own} `), running);
			note(`the detector started by default: ${running.replace(/^\d+ /, "")}`);
		} finally {
			await pi.close();
		}
	},
};

const wanted = process.argv.slice(2);
const names = wanted.length > 0 ? wanted : Object.keys(scenarios);
for (const need of [opf, checkpoint]) {
	if (!existsSync(need)) {
		console.error(`missing: ${need}`);
		process.exit(2);
	}
}
const provider = await startProvider();
try {
	for (const name of names) {
		current = name;
		console.log(`\n${name}`);
		try {
			if (!scenarios[name]) throw new Error(`no such scenario; there are: ${Object.keys(scenarios).join(", ")}`);
			await scenarios[name]!(provider);
		} catch (error) {
			check("the scenario ran to its end", false, error instanceof Error ? (error.stack ?? error.message) : String(error));
		}
	}
} finally {
	await provider.close();
	if (process.env.E2E_DUMP) {
		mkdirSync(process.env.E2E_DUMP, { recursive: true });
		provider.requests.forEach((request, index) => writeFileSync(join(process.env.E2E_DUMP!, `request-${String(index + 1).padStart(2, "0")}.json`), JSON.stringify(request.body, null, 2)));
	}
	spawnSync("pkill", ["-f", root]);
	if (!process.env.E2E_KEEP) rmSync(root, { recursive: true, force: true });
	else console.log(`\nkept: ${root}`);
}
const failed = results.filter((result) => !result.ok);
console.log(`\n${results.length - failed.length} of ${results.length} checks passed${failed.length ? `; failed:\n${failed.map((result) => `  ${result.scenario}: ${result.name}`).join("\n")}` : ""}`);
if (notes.length) console.log(`\nobserved:\n${notes.map((text) => `  ${text}`).join("\n")}`);
process.exit(failed.length ? 1 : 0);
