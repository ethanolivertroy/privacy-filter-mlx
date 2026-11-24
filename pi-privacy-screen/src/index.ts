/**
 * Privacy screen for the Pi coding agent.
 *
 * Private values (names, addresses, secrets...) are replaced by numbered
 * placeholders before a request leaves this machine, and put back when the
 * reply and its tool calls come in. What is stored and shown here is real;
 * what a provider receives is masked.
 *
 *   session_start          the conversation a session starts with is checked
 *                          right away, while the user is still typing
 *   input                  the detector must be up, or the message is not sent
 *   before_agent_start     the prompt gets a short note about placeholders
 *   context_with_system    every text of the request is masked      (screen.ts)
 *   before_provider_request  a last look at the finished payload
 *   message_end            a reply gets its values back: its text, and the
 *                          arguments of its tool calls               (shell.ts)
 *   tool_call              the same for arguments that are still masked
 *                          (both only for a reply to a request that went out
 *                          masked: a model that was sent the real text can
 *                          only be quoting a placeholder, not using one)
 *   session_before_compact / session_before_tree
 *                          summaries are made from masked text
 *
 * Pi carries on with the unchanged request when a handler throws. So nothing
 * here may throw on the way out: whatever goes wrong, text that has not been
 * checked is withheld and the request is stopped.
 *
 * Modes: "mask" (the default), "report" (look and count, change nothing) and
 * "off". `/privacy` shows the state and switches. Requests to a model on this
 * machine are not screened unless the settings ask for it.
 */

import { existsSync, readFileSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { configPath, isLocalModel, parseConfig } from "./config.ts";
import type { Config, Mode } from "./config.ts";
import { Detector } from "./detector.ts";
import { readSeeds } from "./dotenv.ts";
import { OWN_TYPE, WITHHELD, sweepPayload, transcript, unmaskAssistant, unmaskInput } from "./messages.ts";
import type { Message } from "./messages.ts";
import { Screen } from "./screen.ts";
import type { Detect, Item } from "./screen.ts";
import { Vault } from "./vault.ts";

/** Told to the model while masking. It must not itself contain anything shaped like a placeholder. */
export const PROMPT_NOTE = [
	"Some private values in this conversation (names, addresses, secrets and the like) have been replaced by numbered placeholders: a kind and a number in angle brackets, in the form <KIND_n>. Each stands for one exact value that you cannot see.",
	"- Use a placeholder exactly as written wherever its value is needed: in replies, in file contents and in tool arguments. The real value is put in before a tool runs and before the user reads your reply.",
	"- Do not guess, alter, split, encode or describe a hidden value, and do not make up placeholders you were not given.",
	"- In a shell command, put a placeholder where one word or one quoted string goes. It is quoted for the shell automatically. It cannot be used inside backquotes, inside a ${ } expansion or arithmetic, or right after a $ or a backslash; assign it to a variable first if you need that.",
	"- You cannot inspect a hidden value (its length, its characters, its format). If the task needs that, say so; the user can reveal one value with /privacy reveal followed by the placeholder.",
	"- A placeholder of the kind LITERAL stands for text that only looked like a placeholder. Treat it as ordinary text.",
].join("\n");

const SUBCOMMANDS = ["mask", "report", "off", "list", "reveal", "conceal", "retry"];
const USAGE = "Usage: /privacy [mask | report | off | list [full] | reveal <placeholder> | conceal <placeholder> | retry]";

type Level = "info" | "warning" | "error";

// The parts of Pi's extension API that are used here (docs/extensions.md), so that this file needs nothing from Pi to load.
interface Ctx {
	hasUI: boolean;
	cwd: string;
	model?: { provider?: string; id?: string; baseUrl?: string };
	thinkingLevel?: unknown;
	ui: { notify(message: string, level?: Level): void; setStatus(key: string, text: string | undefined): void; setEditorText?(text: string): void };
	sessionManager: { getEntries(): unknown[]; buildSessionProjection?(): { messages?: unknown } };
	abort(): void;
	modelRegistry: { streamSimple(model: unknown, context: unknown, options?: unknown): unknown };
}
interface Pi {
	on(event: string, handler: (event: any, ctx: Ctx) => unknown): void;
	registerCommand(name: string, options: object): void;
	appendEntry(customType: string, data?: unknown): void;
	registerMarkdownTransformer?(transformer: (markdown: string, context?: { messageType?: string; isStreaming?: boolean }) => string, options?: object): void;
}
interface DetectorLike {
	detect: Detect;
	ensure(): Promise<unknown>;
	health(): Promise<unknown>;
	retry(): void;
}

/** What a test replaces; Pi uses the defaults. */
export interface Dependencies {
	env?: Record<string, string | undefined>;
	home?: string;
	detector?: (config: Config) => DetectorLike;
	/** Pi's own package, for its summary functions; loaded only when a summary is needed. */
	importPi?: () => Promise<any>;
	/** Where notices go when there is no screen to show them on. */
	log?: (message: string) => void;
}

/** One session's state. Built at `session_start`, and again after /new, /resume, /fork and /reload. */
interface Session {
	config: Config;
	configFile: string;
	mode: Mode;
	screen: Screen;
	detector: DetectorLike;
	/** Why the detector cannot be used, in words; undefined while it works. */
	trouble?: string;
	/** Whether the request the model is answering went out masked. Only then do the placeholders it writes stand for values. */
	answersMasked: boolean;
	/** The check of the conversation this session started with, begun ahead of its first request. It never rejects. */
	ahead?: Promise<unknown>;
	seeded: Set<string>;
	envFiles: string[];
	/** Notices that are given once. */
	told: Set<string>;
	withheld: number;
	/** Tool calls whose arguments got their values when the reply came in, so that the tool hook does not do it again. */
	settled: Map<string, ReadonlyMap<string, string>>;
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const plural = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : "s"}`;
const reasonOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** `opf-mlx serve` from the project this extension is part of, else the one on the PATH. */
function defaultCommand(): string[] {
	try {
		const here = realpathSync(dirname(fileURLToPath(import.meta.url)));
		const beside = join(here, "..", "..", ".venv", "bin", "opf-mlx");
		if (existsSync(beside)) return [beside, "serve", "--no-download"];
	} catch {
		// fall through
	}
	return ["opf-mlx", "serve", "--no-download"];
}

/** Enough of a value to recognise it by, without spelling it out on screen. */
function preview(value: string): string {
	const shown = value.length >= 12 ? `${value.slice(0, 2)}…${value.slice(-2)}` : value.length >= 6 ? `${value[0]}…` : "…";
	return `${shown.replace(/\s/g, " ")}  (${plural(value.length, "character")})`;
}

export function install(pi: Pi, dependencies: Dependencies = {}): void {
	const env = dependencies.env ?? process.env;
	const home = dependencies.home ?? homedir();
	const log = dependencies.log ?? ((message: string) => console.error(message));
	const importPi = dependencies.importPi ?? (() => import("@earendil-works/pi-coding-agent" as string));
	let session: Session | undefined;

	function say(ctx: Ctx, message: string, level: Level = "info"): void {
		const text = `Privacy screen: ${message}`;
		try {
			if (ctx.hasUI) ctx.ui.notify(text, level);
			else log(text);
		} catch {
			log(text);
		}
	}
	function sayOnce(s: Session, ctx: Ctx, about: string, message: string, level: Level): void {
		if (s.told.has(about)) return;
		s.told.add(about);
		say(ctx, message, level);
	}

	/** Save one fact with the session: a placeholder, what a provider wrote, or a reveal. Never throws. */
	function save(data: Record<string, unknown>): void {
		try {
			pi.appendEntry(OWN_TYPE, { v: 1, ...data });
		} catch (error) {
			log(`Privacy screen: could not save to the session (${reasonOf(error)})`);
		}
	}

	function open(ctx: Ctx): Session {
		const configFile = configPath(env, home);
		let text: string | undefined;
		let unreadable: string | undefined;
		try {
			text = readFileSync(configFile, "utf8");
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") unreadable = reasonOf(error);
		}
		const { config, problems } = parseConfig(text, home);
		if (unreadable) problems.push(`the file could not be read (${unreadable}); the defaults are used`);

		const vault = new Vault();
		const detector = dependencies.detector?.(config) ?? new Detector({ ...config.server, command: config.server.command ?? defaultCommand() });
		const screen = new Screen(detector.detect, { vault, labels: config.labels });

		// Everything from every branch of the session: a number, once given, must never mean a second value.
		for (const entry of ctx.sessionManager.getEntries()) {
			if (!isRecord(entry) || entry.type !== "custom" || entry.customType !== OWN_TYPE || !isRecord(entry.data)) continue;
			const { kind, token, value, key, masked, revealed } = entry.data;
			if (kind === "value" && typeof token === "string" && typeof value === "string") vault.restore(token, value);
			else if (kind === "remember" && typeof key === "string" && (typeof masked === "string" || masked === null)) screen.restore(key, masked);
			else if (kind === "reveal" && typeof token === "string") screen.reveal(token, revealed === true);
		}
		vault.onAllocate = (entry) => save({ kind: "value", token: entry.token, value: entry.value });
		screen.onRemember = (key, masked) => save({ kind: "remember", key, masked });

		const made: Session = { config, configFile, mode: config.mode, screen, detector, answersMasked: false, seeded: new Set(), envFiles: [], told: new Set(), withheld: 0, settled: new Map() };
		if (problems.length > 0) {
			say(ctx, `${configFile} has ${problems.length === 1 ? "a mistake" : "mistakes"}. The screen is on; what is wrong keeps its default.\n- ${problems.join("\n- ")}`, "error");
		}
		return made;
	}

	const ready = (ctx: Ctx): Session => (session ??= open(ctx));

	const isLocal = (s: Session, ctx: Ctx) => !s.config.screenLocalModels && isLocalModel(ctx.model, s.config);
	/** How requests to the current model are treated: masked, only looked at, or left alone. */
	const screening = (s: Session, ctx: Ctx): "mask" | "report" | undefined => (s.mode === "off" || isLocal(s, ctx) ? undefined : s.mode);

	function refresh(ctx: Ctx, s: Session): void {
		if (s !== session) return; // that session has ended; the line belongs to the next one
		const hidden = s.screen.vault.entries.filter((entry) => !entry.revealed).length;
		let text: string;
		if (s.mode === "off") text = "privacy: OFF";
		else if (isLocal(s, ctx)) text = "privacy: local model, not screened";
		else if (s.mode === "report") text = `privacy: report only (${hidden} found)`;
		else if (s.trouble) text = "privacy: DETECTOR DOWN, nothing is sent";
		else text = `privacy: mask (${hidden} hidden)`;
		try {
			ctx.ui.setStatus("privacy", text);
		} catch {
			// a status line is a nicety
		}
	}

	/** Start the detector ahead of the first request, without waiting for it. */
	function warm(ctx: Ctx, s: Session): void {
		void s.detector.ensure().then(
			() => {
				s.trouble = undefined;
				refresh(ctx, s);
			},
			(error) => {
				s.trouble = reasonOf(error);
				refresh(ctx, s);
				if (s.mode === "mask") sayOnce(s, ctx, `down:${s.trouble}`, `${s.trouble}. Until the detector runs, messages are not sent. /privacy retry tries again; /privacy off sends without screening.`, "warning");
			},
		);
	}

	/**
	 * Check the conversation a session starts with, without waiting for it. The detector takes a moment for
	 * each text and a long session has hundreds of them: this way the time passes while the user is typing
	 * and not after they have sent. It is a head start and nothing more. What it finds is what the first
	 * request then finds already checked, and whatever fails here fails again there, where it is dealt with.
	 */
	function headStart(ctx: Ctx, s: Session): void {
		try {
			const messages = ctx.sessionManager.buildSessionProjection?.().messages;
			if (!Array.isArray(messages) || messages.length === 0) return;
			const plan = transcript(messages as Message[], { provider: ctx.model?.provider });
			s.ahead = s.screen.mask(plan.items).then(() => refresh(ctx, s));
		} catch {
			// then the first request does all of it
		}
	}

	/** Read the project's `.env` files, again each time: they change. */
	async function learn(ctx: Ctx, s: Session): Promise<void> {
		const { seeds, files } = await readSeeds(ctx.cwd, s.config.envFiles);
		const fresh = seeds.filter((seed) => !s.seeded.has(seed));
		for (const seed of fresh) s.seeded.add(seed);
		if (fresh.length > 0) s.screen.seed(fresh);
		s.envFiles = files;
	}

	/** Stop the request and send nothing of it: the answer to any failure on the way out. */
	function withholdAll(ctx: Ctx, error: unknown): { messages: Message[] } {
		say(ctx, `the request could not be screened (${reasonOf(error)}). It was stopped and nothing of it was sent.`, "error");
		try {
			ctx.abort();
		} catch {
			// the messages below hold nothing either way
		}
		const timestamp = Date.now();
		return { messages: [{ role: "system", content: WITHHELD, timestamp }, { role: "user", content: WITHHELD, timestamp }] }; // fmt: skip
	}

	/**
	 * A stream function for Pi's own summary requests (compaction, branch summaries), which do not pass
	 * the hooks above: the request is masked here, on its way to the model.
	 */
	function maskedStream(ctx: Ctx, s: Session) {
		return async (model: unknown, context: unknown, options: unknown) => {
			const request = isRecord(context) ? { ...context } : {};
			const plan = transcript(Array.isArray(request.messages) ? (request.messages as Message[]) : []);
			const prompt: Item[] = typeof request.systemPrompt === "string" ? [{ text: request.systemPrompt, origin: "system" }] : [];
			const masked = await s.screen.mask([...plan.items, ...prompt]);
			if (masked.problem) throw masked.problem.error;
			if (masked.texts.includes(undefined)) throw new Error("a text could not be checked");
			request.messages = plan.build(masked.texts.slice(0, plan.items.length)).messages;
			if (prompt.length > 0) request.systemPrompt = masked.texts.at(-1);
			return ctx.modelRegistry.streamSimple(model, request, options);
		};
	}

	/** A summary as the model wrote it, with its values put back, and remembered so that it goes out the same again. */
	function restored(s: Session, summary: string): string {
		const real = s.screen.unmask(summary);
		s.screen.remember(real, summary);
		return real;
	}

	// ------------------------------------------------------------- lifecycle

	pi.on("session_start", async (_event, ctx) => {
		session = undefined;
		const s = ready(ctx);
		await learn(ctx, s);
		refresh(ctx, s);
		if (!screening(s, ctx)) return;
		warm(ctx, s);
		headStart(ctx, s);
	});

	pi.on("session_shutdown", (_event, ctx) => {
		session = undefined;
		try {
			ctx.ui.setStatus("privacy", undefined);
		} catch {
			// nothing to clear
		}
	});

	pi.on("model_select", (_event, ctx) => {
		const s = ready(ctx);
		refresh(ctx, s);
		if (screening(s, ctx)) warm(ctx, s);
	});

	// ------------------------------------------------------------ on the way out

	pi.on("input", async (event, ctx) => {
		const s = ready(ctx);
		// Text that arrives while a reply is streaming is checked with the next request; so is an extension's.
		if (screening(s, ctx) !== "mask" || event.source === "extension" || event.streamingBehavior !== undefined) return { action: "continue" };
		try {
			await s.detector.ensure();
			s.trouble = undefined;
			return { action: "continue" };
		} catch (error) {
			s.trouble = reasonOf(error);
			refresh(ctx, s);
			try {
				if (typeof event.text === "string") ctx.ui.setEditorText?.(event.text);
			} catch {
				// then it has to be typed again
			}
			say(ctx, `${s.trouble}. Your message was not sent. /privacy retry tries the detector again; /privacy off sends without screening.`, "error");
			return { action: "handled" };
		}
	});

	pi.on("before_agent_start", async (event, ctx) => {
		try {
			const s = ready(ctx);
			await learn(ctx, s);
			const sections = event.systemPromptOptions?.sections;
			if (!isRecord(sections)) return;
			if (screening(s, ctx) === "mask") sections.privacy_screen = PROMPT_NOTE;
			else delete sections.privacy_screen;
		} catch (error) {
			log(`Privacy screen: ${reasonOf(error)}`);
		}
	});

	pi.on("context_with_system", async (event, ctx) => {
		try {
			const s = ready(ctx);
			const how = screening(s, ctx);
			s.answersMasked = how === "mask";
			if (!how) return undefined;
			await s.ahead; // most of these texts are being checked already: that is waited for, not asked for again
			const plan = transcript(event.messages, { provider: ctx.model?.provider });
			const masked = await s.screen.mask(plan.items);
			s.screen.endPass();
			const problem = masked.problem;
			if (problem) s.trouble = problem.down ? problem.error.message : s.trouble;
			else s.trouble = undefined;

			if (how === "report") {
				if (problem) sayOnce(s, ctx, "report", `${plural(problem.count, "text")} could not be checked (${problem.error.message}), so the report is incomplete. Report mode changes nothing: the request was sent as it is.`, "warning");
				refresh(ctx, s);
				return undefined;
			}

			const built = plan.build(masked.texts);
			s.withheld += built.withheld;
			if (problem?.down) {
				say(ctx, `${problem.error.message}. The request was stopped, and what could not be checked was not sent. /privacy retry tries the detector again; /privacy off sends without screening.`, "error");
				ctx.abort();
			} else if (problem) {
				say(ctx, `${plural(problem.count, "text")} could not be checked and ${problem.count === 1 ? "was" : "were"} withheld from the model (${problem.error.message}).`, "warning");
			}
			refresh(ctx, s);
			return { messages: built.messages };
		} catch (error) {
			return withholdAll(ctx, error);
		}
	});

	pi.on("before_provider_request", (event, ctx) => {
		try {
			const s = ready(ctx);
			if (screening(s, ctx) !== "mask") return undefined;
			const swept = sweepPayload(event.payload, s.screen);
			if (swept.hits === 0) return undefined;
			sayOnce(s, ctx, "sweep", `${plural(swept.hits, "known value")} stood in a part of the request that is not rewritten text by text (a tool description, for example) and ${swept.hits === 1 ? "was" : "were"} replaced at the last step before sending. This is said once per session.`, "warning");
			return swept.payload;
		} catch (error) {
			say(ctx, `the last look at the request failed (${reasonOf(error)}). The request was stopped.`, "error");
			try {
				ctx.abort();
			} catch {
				// nothing more can be done from here
			}
			return undefined;
		}
	});

	// ------------------------------------------------------------- on the way in

	pi.on("message_end", (event, ctx) => {
		try {
			const s = ready(ctx);
			if (!s.answersMasked) return undefined;
			const unmasked = unmaskAssistant(event.message, s.screen, { shell: s.config.shellTools, code: s.config.codeTools });
			if (!unmasked) return undefined;
			for (const [id, record] of unmasked.settled) s.settled.set(id, record);
			return { message: unmasked.message };
		} catch (error) {
			say(ctx, `a reply could not be given its values back (${reasonOf(error)}); it is shown with its placeholders.`, "warning");
			return undefined;
		}
	});

	// Whatever is still masked when a tool is about to run: a call that was stored as the model wrote it, or one
	// made by another tool. An error thrown here blocks the tool, which is the safe way to fail.
	pi.on("tool_call", (event, ctx) => {
		const s = ready(ctx);
		if (!s.answersMasked) return undefined;
		if (s.config.codeTools.includes(event.toolName)) return undefined; // the calls such code makes come here themselves
		const settled = s.settled.get(event.toolCallId);
		s.settled.delete(event.toolCallId);
		const refusal = unmaskInput(event.input, s.screen, s.config.shellTools[event.toolName] ?? [], { settled });
		return refusal === undefined ? undefined : { block: true, reason: refusal };
	});

	// What is on screen. Thinking is stored as the model wrote it, so it is shown with its values whenever it is
	// shown. The text of a reply is stored with its values: only one that is still coming in has placeholders,
	// and what looks like one afterwards is text.
	pi.registerMarkdownTransformer?.(
		(markdown: string, context?: { messageType?: string; isStreaming?: boolean }) => {
			if (!session) return markdown;
			const stillComing = context?.isStreaming === true && session.answersMasked;
			return context?.messageType === "assistant-thinking" || stillComing ? session.screen.unmask(markdown) : markdown;
		},
		{ messageTypes: ["assistant", "assistant-thinking"] },
	);

	// --------------------------------------------------------------- summaries

	pi.on("session_before_compact", async (event, ctx) => {
		const s = ready(ctx);
		if (screening(s, ctx) !== "mask") return undefined;
		try {
			const { compact } = await importPi();
			const result = await compact(event.preparation, ctx.model, undefined, undefined, event.customInstructions, event.signal, ctx.thinkingLevel, maskedStream(ctx, s));
			return { compaction: { ...result, summary: restored(s, result.summary) } };
		} catch (error) {
			if (!event.signal?.aborted) say(ctx, `compaction was cancelled, because its summary request could not be screened or made (${reasonOf(error)}).`, "error");
			return { cancel: true };
		}
	});

	pi.on("session_before_tree", async (event, ctx) => {
		const s = ready(ctx);
		const preparation = event.preparation;
		if (screening(s, ctx) !== "mask" || !preparation?.userWantsSummary || !(preparation.entriesToSummarize?.length > 0)) return undefined;
		try {
			const { generateBranchSummary } = await importPi();
			const result = await generateBranchSummary(preparation.entriesToSummarize, {
				model: ctx.model,
				signal: event.signal,
				thinkingLevel: ctx.thinkingLevel,
				customInstructions: preparation.customInstructions,
				replaceInstructions: preparation.replaceInstructions,
				streamFn: maskedStream(ctx, s),
			});
			if (result.aborted) return { cancel: true };
			if (result.error || typeof result.summary !== "string") throw new Error(result.error ?? "no summary came back");
			return { summary: { summary: restored(s, result.summary), usage: result.usage, details: { readFiles: result.readFiles ?? [], modifiedFiles: result.modifiedFiles ?? [] } } };
		} catch (error) {
			if (!event.signal?.aborted) say(ctx, `the branch summary was cancelled, because its request could not be screened or made (${reasonOf(error)}).`, "error");
			return { cancel: true };
		}
	});

	// ----------------------------------------------------------------- command

	async function describe(ctx: Ctx, s: Session): Promise<string> {
		const entries = s.screen.vault.entries;
		const revealed = entries.filter((entry) => entry.revealed).length;
		const health = await s.detector.health().catch(() => undefined);
		const model = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "none selected";
		const treated = s.mode === "off" ? "not screened (the screen is off)" : isLocal(s, ctx) ? "on this machine, not screened" : s.mode === "mask" ? "masked" : "looked at, sent as it is";
		const about = isRecord(health) ? [health.device, health.dtype].filter((one) => typeof one === "string").join(", ") : "";
		return [
			`mode: ${s.mode}${s.mode === s.config.mode ? "" : ` (for this session; the settings say ${s.config.mode})`}`,
			`model: ${model}, ${treated}`,
			`detector: ${health ? `running${about ? ` (${about})` : ""}` : `not running${s.trouble ? ` (${s.trouble})` : ""}`}, ${s.config.server.socket}`,
			`${plural(entries.length - revealed, "value")} hidden, ${revealed} revealed; ${plural(s.withheld, "text")} withheld because they could not be checked`,
			`.env: ${s.envFiles.length > 0 ? `${plural(s.seeded.size, "value")} known from ${s.envFiles.join(", ")}` : "no file read"}`,
			`looks for: ${s.config.labels.join(", ")}, and known key formats`,
			`settings: ${s.configFile}`,
			USAGE,
		].join("\n");
	}

	pi.registerCommand("privacy", {
		description: "Privacy screen: show its state, switch between mask, report and off, list or reveal hidden values",
		getArgumentCompletions: (prefix: string) => {
			const items = SUBCOMMANDS.filter((name) => name.startsWith(prefix.trim())).map((name) => ({ value: name, label: name }));
			return items.length > 0 ? items : null;
		},
		handler: async (args: string, ctx: Ctx) => {
			const s = ready(ctx);
			const [name = "", argument = ""] = args.trim().split(/\s+/);
			if (name === "") {
				say(ctx, `\n${await describe(ctx, s)}`);
			} else if (name === "mask" || name === "report" || name === "off") {
				s.mode = name;
				s.told.clear();
				if (name === "mask") {
					s.detector.retry();
					say(ctx, "masking. Private values are replaced by placeholders before anything is sent.");
				} else if (name === "report") {
					say(ctx, "report only. Requests go out as they are; /privacy list shows what would have been hidden.", "warning");
				} else {
					say(ctx, "OFF. Requests go out unscreened until /privacy mask, or until the session is reloaded.", "warning");
				}
				if (screening(s, ctx)) warm(ctx, s);
			} else if (name === "retry") {
				s.detector.retry();
				s.told.clear();
				warm(ctx, s);
				say(ctx, "trying the detector again.");
			} else if (name === "list") {
				const entries = s.screen.vault.entries;
				const lines = entries.map((entry) => `${entry.token}  ${argument === "full" ? entry.value : preview(entry.value)}${entry.revealed ? "  (revealed: sent as it is)" : ""}`);
				say(ctx, entries.length > 0 ? `${plural(entries.length, "value")} known in this session. This list is shown here only.\n${lines.join("\n")}` : "no value has been hidden in this session yet.");
			} else if (name === "reveal" || name === "conceal") {
				// By its placeholder, or by the value itself: a reply that says which placeholder to reveal is shown
				// with the value in its place.
				const what = args.trim().slice(name.length).trim();
				const bare = what.replace(/^<|>$/g, "").toUpperCase();
				const token = s.screen.vault.lookup(`<${bare}>`)?.token ?? s.screen.vault.find(what)?.token;
				if (what === "") say(ctx, USAGE, "warning");
				else if (token === undefined && /^[A-Z]+_[0-9]+$/.test(bare)) say(ctx, `there is no placeholder <${bare}> in this session; /privacy list shows them.`, "warning");
				else if (token === undefined) say(ctx, "nothing is hidden under that in this session; /privacy list shows the placeholders.", "warning");
				else {
					s.screen.reveal(token, name === "reveal");
					save({ kind: "reveal", token, revealed: name === "reveal" });
					if (name === "reveal") say(ctx, `${token} is now sent as it is, also where it stood hidden before. /privacy conceal ${token} hides it again.`, "warning");
					else say(ctx, `${token} is hidden again. What a provider has already received stays received.`);
				}
			} else {
				say(ctx, USAGE, "warning");
			}
			refresh(ctx, s);
		},
	});
}

export default function privacyScreen(pi: Pi): void {
	install(pi);
}
