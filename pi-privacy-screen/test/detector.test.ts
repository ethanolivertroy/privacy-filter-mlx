import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import { join } from "node:path";
import { after, test } from "node:test";

import { Detector } from "../src/detector.ts";

// Unix socket paths are limited to about 100 bytes, so these live right under /tmp.
const scratch = mkdtempSync("/tmp/opf-pi-");
const servers: Server[] = [];
const started: number[] = [];
after(() => {
	for (const server of servers) server.close();
	for (const pid of started) {
		try {
			process.kill(pid, "SIGKILL");
		} catch {
			// already gone
		}
	}
	rmSync(scratch, { recursive: true, force: true });
});

let counter = 0;
const socketPath = () => join(scratch, `s${counter++}.sock`);

interface Request {
	method: string;
	url: string;
	type: string | undefined;
	body: any;
}
/** An answer, or no answer at all: the connection is dropped. */
type Reply = [status: number, body: unknown] | "hang up";

/** A stand-in for `opf-mlx serve`: answers /health, and /v1/detect with whatever `detect` says. */
async function fake(detect: (request: Request) => Reply | Promise<Reply>, health: unknown = { status: "ok", api: 1, labels: ["secret"], device: "gpu" }) {
	const socket = socketPath();
	const requests: Request[] = [];
	const server = createServer(async (incoming: IncomingMessage, response: ServerResponse) => {
		const chunks: Buffer[] = [];
		for await (const chunk of incoming) chunks.push(chunk as Buffer);
		const text = Buffer.concat(chunks).toString("utf8");
		const request: Request = { method: incoming.method!, url: incoming.url!, type: incoming.headers["content-type"], body: text ? JSON.parse(text) : undefined };
		const reply: Reply = request.url === "/health" ? [200, health] : (requests.push(request), await detect(request));
		if (reply === "hang up") return void incoming.socket.destroy();
		response.writeHead(reply[0], { "Content-Type": "application/json" }).end(JSON.stringify(reply[1]));
	});
	servers.push(server);
	await new Promise<void>((resolve) => server.listen(socket, resolve));
	return { socket, requests };
}

/** Flags every "SECRET" in a text, as the real detector would report it. */
const flagSecrets = (request: Request): Reply => [
	200,
	{
		results: (request.body.texts as string[]).map((text) => ({
			spans: [...text.matchAll(/SECRET/g)].map((match) => ({ start: match.index, end: match.index + 6, label: "secret" })),
		})),
	},
];

const options = (socket: string, more: object = {}) => ({ socket, command: undefined, autoStart: false, idleExit: 60, startTimeout: 5, ...more });

test("a running detector reports what it is", async () => {
	const { socket } = await fake(flagSecrets);

	assert.deepEqual(await new Detector(options(socket)).health(), { status: "ok", api: 1, labels: ["secret"], device: "gpu" });
});

test("no detector, or one that speaks another version, is no detector", async () => {
	const other = await fake(flagSecrets, { status: "ok", api: 2 });

	assert.equal(await new Detector(options(socketPath())).health(), undefined);
	assert.equal(await new Detector(options(other.socket)).health(), undefined);
	await assert.rejects(new Detector(options(other.socket)).detect(["x"]), /API version 2/);
});

test("something else that listens on the socket is not taken for the detector, and none is started over it", async () => {
	const stranger = await fake(flagSecrets, { status: "loading", api: 1 });
	const detector = new Detector(options(stranger.socket, { command: starter(0), autoStart: true }), (pid) => started.push(pid));

	assert.equal(await detector.health(), undefined);
	await assert.rejects(detector.detect(["x"]), /is not the detector/);
	assert.equal(existsSync(`${stranger.socket}.starts`), false);
	assert.equal(stranger.requests.length, 0, "and it is sent no text");
});

test("texts go in, spans come out, in UTF-16 offsets", async () => {
	const { socket, requests } = await fake(flagSecrets);

	const spans = await new Detector(options(socket)).detect(["a SECRET", "none", "🙂 SECRET SECRET"]);

	assert.deepEqual(spans, [
		[{ start: 2, end: 8, label: "secret" }],
		[],
		[
			{ start: 3, end: 9, label: "secret" },
			{ start: 10, end: 16, label: "secret" },
		],
	]);
	assert.deepEqual(requests, [{ method: "POST", url: "/v1/detect", type: "application/json", body: { texts: ["a SECRET", "none", "🙂 SECRET SECRET"], offsets: "utf16" } }]);
});

test("nothing to check is not a request", async () => {
	const { socket, requests } = await fake(flagSecrets);

	assert.deepEqual(await new Detector(options(socket)).detect([]), []);
	assert.equal(requests.length, 0);
});

test("many texts are sent in several requests and come back in order", async () => {
	const { socket, requests } = await fake(flagSecrets);
	const texts = ["SECRET one", "two", "three SECRET", "four", "SECRET five"];

	const spans = await new Detector(options(socket, { batchTexts: 2 })).detect(texts);

	assert.deepEqual(spans.map((one) => (one as unknown[]).length), [1, 0, 1, 0, 1]);
	assert.deepEqual(requests.map((request) => request.body.texts), [["SECRET one", "two"], ["three SECRET", "four"], ["SECRET five"]]);
});

test("a request is also cut by size", async () => {
	const { socket, requests } = await fake(flagSecrets);

	await new Detector(options(socket, { batchBytes: 25 })).detect(["x".repeat(20), "y".repeat(20), "z"]);

	assert.deepEqual(requests.map((request) => request.body.texts.length), [1, 2]);
});

test("when one text makes a request fail, only that text goes unchecked", async () => {
	const { socket, requests } = await fake((request) => {
		const texts = request.body.texts as string[];
		const bad = texts.findIndex((text) => text.includes("BAD"));
		return bad === -1 ? flagSecrets(request) : [500, { error: `detection failed for text ${bad + 1} of ${texts.length} (the text does not survive tokenization)` }];
	});

	const spans = await new Detector(options(socket)).detect(["SECRET", "BAD text", "fine"]);

	assert.deepEqual(spans[0], [{ start: 0, end: 6, label: "secret" }]);
	assert.ok(spans[1] instanceof Error);
	assert.match((spans[1] as Error).message, /does not survive tokenization/);
	assert.deepEqual(spans[2], []);
	assert.deepEqual(requests.map((request) => request.body.texts.length), [3, 1, 1, 1]);
});

test("a text too large to check is not sent, and not cut short", async () => {
	const { socket, requests } = await fake(flagSecrets);

	const spans = await new Detector(options(socket, { maxTextBytes: 10 })).detect(["small", "a text that is larger than ten bytes"]);

	assert.deepEqual(spans[0], []);
	assert.match((spans[1] as Error).message, /too large/);
	assert.deepEqual(requests.map((request) => request.body.texts), [["small"]]);
});

test("an answer that cannot be right is not used", async () => {
	const answers: Reply[] = [
		[200, { results: [{ spans: [] }] }], // two texts, one result
		[200, { results: [{ spans: [{ start: 2, end: 99, label: "secret" }] }] }],
		[200, { results: [{ spans: [{ start: 3, end: 3, label: "secret" }] }] }],
		[200, { results: [{ spans: [{ start: 0, end: 1 }] }] }],
		[200, { results: [{ spans: "none" }] }],
		[200, "nonsense"],
		[404, { error: "not found" }],
	];
	let next = 0;
	const { socket } = await fake(() => answers[next++]!);
	const detector = new Detector(options(socket));

	const first = await detector.detect(["one", "two"]);
	assert.ok(first.every((one) => one instanceof Error && /could not be used/.test(one.message)), "wrong count");
	for (let index = 1; index < answers.length; index++) {
		const [only] = await detector.detect(["text"]);
		assert.ok(only instanceof Error, `answer ${index}`);
	}
});

test("without a detector, and without leave to start one, checking fails as a whole", async () => {
	const socket = socketPath();
	const detector = new Detector(options(socket, { command: starter(0), autoStart: false }), (pid) => started.push(pid));

	await assert.rejects(detector.detect(["x"]), (error: Error) => {
		assert.match(error.message, /not running/);
		assert.ok(error.message.includes(socket));
		return true;
	});
	assert.equal(existsSync(`${socket}.starts`), false, "the command is known, but it is not run");
});

test("a detector that goes away in the middle of a request fails the check as a whole, not text by text", async () => {
	const { socket } = await fake(() => "hang up");

	await assert.rejects(new Detector(options(socket)).detect(["one", "two"]), (error: Error) => {
		assert.match(error.message, /did not answer/);
		assert.ok(error.message.includes(socket));
		return true;
	});
});

/** A script that behaves like a detector being started: it listens on --socket after `delay` ms. */
function starter(delay: number): string[] {
	const script = join(scratch, `starter${counter++}.mjs`);
	writeFileSync(
		script,
		`import { createServer } from "node:http";
import { appendFileSync } from "node:fs";
const socket = process.argv[process.argv.indexOf("--socket") + 1];
appendFileSync(socket + ".starts", process.argv.slice(2).join(" ") + "\\n");
setTimeout(() => {
	createServer((request, response) => {
		request.resume();
		request.on("end", () => {
			response.writeHead(200, { "Content-Type": "application/json" });
			response.end(JSON.stringify(request.url === "/health" ? { status: "ok", api: 1 } : { results: [{ spans: [] }] }));
		});
	}).listen(socket, () => console.log("listening on " + socket));
}, ${delay});
`,
	);
	return [process.execPath, script, "serve"];
}

test("a detector that is not running is started, once, and waited for", async () => {
	const socket = socketPath();
	const detector = new Detector(options(socket, { command: starter(300), autoStart: true, idleExit: 77 }), (pid) => started.push(pid));

	const [first, second] = await Promise.all([detector.detect(["x"]), detector.detect(["y"])]);

	assert.deepEqual([first, second], [[[]], [[]]]);
	assert.deepEqual(readFileSync(`${socket}.starts`, "utf8"), `serve --socket ${socket} --idle-exit 77\n`);
	assert.match(readFileSync(`${socket}.log`, "utf8"), /listening on/);
	assert.equal(started.length, 1);
});

test("a detector that fails to start says why, and is not started again right away", async () => {
	const socket = socketPath();
	const script = join(scratch, "failing.mjs");
	writeFileSync(
		script,
		`import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(`${socket}.starts`)}, "start\\n");
console.error("error: no checkpoint at /somewhere; run: opf-mlx download");
process.exit(2);
`,
	);
	const detector = new Detector(options(socket, { command: [process.execPath, script], autoStart: true }));

	await assert.rejects(detector.detect(["x"]), /could not be started.*no checkpoint at \/somewhere/s);
	await assert.rejects(detector.detect(["x"]), /no checkpoint at \/somewhere/);
	assert.equal(readFileSync(`${socket}.starts`, "utf8"), "start\n");

	detector.retry();
	await assert.rejects(detector.detect(["x"]), /no checkpoint/);
	assert.equal(readFileSync(`${socket}.starts`, "utf8"), "start\nstart\n");
});

test("a detector that does not come up in time is given up on", async () => {
	const socket = socketPath();
	const detector = new Detector(options(socket, { command: starter(60_000), autoStart: true, startTimeout: 0.6 }), (pid) => started.push(pid));
	const before = Date.now();

	await assert.rejects(detector.detect(["x"]), /did not come up within 0\.6 s/);

	assert.ok(Date.now() - before < 5000);
});

test("a command that does not exist is reported, not thrown", async () => {
	const socket = socketPath();
	const detector = new Detector(options(socket, { command: [join(scratch, "no-such-program")], autoStart: true }));

	await assert.rejects(detector.detect(["x"]), /could not be started/);
	assert.equal(existsSync(socket), false);
});
