import { spawn } from "node:child_process";
import { closeSync, mkdirSync, openSync, readFileSync } from "node:fs";
import { request } from "node:http";
import { dirname } from "node:path";

import type { Detect } from "./screen.ts";
import type { Span } from "./spans.ts";

export const API_VERSION = 1;
const PROBE_MS = 1500;
const POLL_MS = 150;
const COOL_DOWN_MS = 30_000;

export interface DetectorOptions {
	socket: string;
	command: string[] | undefined;
	autoStart: boolean;
	idleExit: number;
	startTimeout: number;
	batchTexts?: number;
	batchBytes?: number;
	maxTextBytes?: number;
}

export interface Health {
	status: "ok";
	api: number;
	[more: string]: unknown;
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function call(socket: string, method: "GET" | "POST", path: string, body: unknown, timeoutMs: number): Promise<{ status: number; body: unknown }> {
	return new Promise((resolve, reject) => {
		const payload = body === undefined ? undefined : Buffer.from(JSON.stringify(body), "utf8");
		const outgoing = request(
			{ socketPath: socket, method, path, headers: payload ? { "Content-Type": "application/json", "Content-Length": payload.length } : {} },
			(response) => {
				const chunks: Buffer[] = [];
				response.on("data", (chunk: Buffer) => chunks.push(chunk));
				response.on("error", reject);
				response.on("end", () => {
					let parsed: unknown;
					try {
						parsed = JSON.parse(Buffer.concat(chunks).toString("utf8"));
					} catch {
						parsed = undefined;
					}
					resolve({ status: response.statusCode ?? 0, body: parsed });
				});
			},
		);
		outgoing.setTimeout(timeoutMs, () => outgoing.destroy(new Error(`no answer within ${Math.round(timeoutMs / 1000)} s`)));
		outgoing.on("error", reject);
		outgoing.end(payload);
	});
}

function readResults(texts: readonly string[], body: unknown): (Span[] | Error)[] {
	const unusable = () => new Error("the detector's answer could not be used");
	if (!isRecord(body) || !Array.isArray(body.results) || body.results.length !== texts.length) return texts.map(unusable);
	return body.results.map((result: unknown, index: number) => {
		const spans = isRecord(result) ? result.spans : undefined;
		if (!Array.isArray(spans)) return unusable();
		const out: Span[] = [];
		for (const span of spans as unknown[]) {
			if (!isRecord(span)) return unusable();
			const { start, end, label } = span;
			if (typeof start !== "number" || typeof end !== "number" || typeof label !== "string") return unusable();
			if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > texts[index]!.length) return unusable();
			out.push({ start, end, label });
		}
		return out;
	});
}

function tail(logPath: string): string {
	try {
		const lines = readFileSync(logPath, "utf8").slice(-2000).split("\n").map((line) => line.trim()).filter(Boolean);
		return lines.length > 0 ? `: ${lines.slice(-3).join(" / ")}` : "";
	} catch {
		return "";
	}
}

function lostSocketOwnershipRace(logPath: string): boolean {
	return /already owns/.test(tail(logPath));
}

export class Detector {
	readonly options: DetectorOptions;
	private readonly onStart?: (pid: number) => void;
	private starting?: Promise<Health>;
	private failed?: { error: Error; until: number };

	constructor(options: DetectorOptions, onStart?: (pid: number) => void) {
		this.options = options;
		this.onStart = onStart;
	}

	async health(): Promise<Health | undefined> {
		return (await this.probe()).health;
	}

	async ensure(): Promise<Health> {
		const { health, reason } = await this.probe();
		if (health) {
			this.failed = undefined;
			return health;
		}
		if (reason) throw new Error(reason);
		const { socket, command, autoStart } = this.options;
		if (!autoStart || !command) {
			throw new Error(`the detector is not running at ${socket}${autoStart ? " and no command to start it is known" : ""}; start it with: opf-mlx serve`);
		}
		if (this.failed && Date.now() < this.failed.until) throw this.failed.error;
		this.starting ??= this.launch(command)
			.catch((error: Error) => {
				this.failed = { error, until: Date.now() + COOL_DOWN_MS };
				throw error;
			})
			.finally(() => (this.starting = undefined));
		return this.starting;
	}

	retry(): void {
		this.failed = undefined;
	}

	readonly detect: Detect = async (texts) => {
		if (texts.length === 0) return [];
		await this.ensure();
		const { batchTexts = 64, batchBytes = 1024 * 1024, maxTextBytes = 8 * 1024 * 1024 } = this.options;
		const results = new Array<Span[] | Error>(texts.length);
		let batch: number[] = [];
		let bytes = 0;
		const send = async () => {
			if (batch.length === 0) return;
			const answers = await this.ask(batch.map((index) => texts[index]!), bytes);
			batch.forEach((index, at) => (results[index] = answers[at]!));
			batch = [];
			bytes = 0;
		};
		for (const [index, text] of texts.entries()) {
			const size = Buffer.byteLength(text, "utf8");
			if (size > maxTextBytes) {
				results[index] = new Error(`the text is too large to check (${size} bytes; the limit is ${maxTextBytes})`);
				continue;
			}
			if (batch.length >= batchTexts || (batch.length > 0 && bytes + size > batchBytes)) await send();
			batch.push(index);
			bytes += size;
		}
		await send();
		return results;
	};

	private async ask(texts: string[], bytes: number): Promise<(Span[] | Error)[]> {
		const timeoutMs = 120_000 + bytes / 4;
		let answer: { status: number; body: unknown };
		try {
			answer = await call(this.options.socket, "POST", "/v1/detect", { texts, offsets: "utf16" }, timeoutMs);
		} catch (error) {
			throw new Error(`the detector at ${this.options.socket} did not answer (${(error as Error).message})`);
		}
		if (answer.status === 200) return readResults(texts, answer.body);
		if (texts.length > 1 && (answer.status === 500 || answer.status === 413)) {
			const each: (Span[] | Error)[] = [];
			for (const text of texts) each.push((await this.ask([text], Buffer.byteLength(text, "utf8")))[0]!);
			return each;
		}
		const said = isRecord(answer.body) && typeof answer.body.error === "string" ? answer.body.error : `HTTP ${answer.status}`;
		return texts.map(() => new Error(`the detector could not check this text (${said})`));
	}

	private async probe(): Promise<{ health?: Health; reason?: string }> {
		let answer: { status: number; body: unknown };
		try {
			answer = await call(this.options.socket, "GET", "/health", undefined, PROBE_MS);
		} catch {
			return {};
		}
		const body = answer.body;
		if (answer.status !== 200 || !isRecord(body) || body.status !== "ok") return { reason: `what listens at ${this.options.socket} is not the detector` };
		if (body.api !== API_VERSION) {
			return { reason: `the detector at ${this.options.socket} speaks API version ${String(body.api)}, and this extension speaks ${API_VERSION}; update one of them` };
		}
		return { health: body as Health };
	}

	private async launch(command: readonly string[]): Promise<Health> {
		const { socket, idleExit, startTimeout } = this.options;
		const [program, ...args] = [...command, "--socket", socket, "--idle-exit", String(idleExit)] as [string, ...string[]];
		const logPath = `${socket}.log`;
		let ended: string | undefined;
		try {
			mkdirSync(dirname(socket), { recursive: true, mode: 0o700 });
			const log = openSync(logPath, "w", 0o600);
			try {
				const child = spawn(program, args, { detached: true, stdio: ["ignore", log, log] });
				child.on("error", (error) => (ended = error.message));
				child.on("exit", (code, signal) => (ended ??= code === null ? `signal ${signal}` : `exit status ${code}`));
				child.unref();
				if (child.pid !== undefined) this.onStart?.(child.pid);
			} finally {
				closeSync(log);
			}
		} catch (error) {
			throw new Error(`the detector could not be started (${(error as Error).message})`);
		}
		const deadline = Date.now() + startTimeout * 1000;
		for (;;) {
			const { health, reason } = await this.probe();
			if (health) return health;
			if (reason) throw new Error(reason);
			if (ended !== undefined && !lostSocketOwnershipRace(logPath)) {
				throw new Error(`the detector could not be started (${ended})${tail(logPath)}`);
			}
			if (Date.now() >= deadline) throw new Error(`the detector was started but did not come up within ${startTimeout} s; see ${logPath}`);
			await sleep(POLL_MS);
		}
	}
}
