/**
 * Pi's messages, as far as the screen needs to know them: which of their
 * texts reach the provider, and where each came from.
 *
 *   transcript       the conversation on its way out: every text as an item
 *                    for `Screen.mask`, and a copy of the messages to put the
 *                    answers in
 *   unmaskAssistant  a reply on its way in: its text and the arguments of
 *                    its tool calls get their values back
 *   unmaskInput      the arguments of one tool call; also used when the
 *                    tool is about to run, for whatever is still masked
 *   sweepPayload     the last look at a provider's request body
 *
 * The shapes are Pi's (docs/message-types.md) but nothing is imported from
 * Pi, so all of this runs, and is tested, under plain Node. Fields that are
 * not text for the provider are passed on untouched: images, tool `details`,
 * a provider's own (signed) thinking, tool declarations.
 *
 * What a model wrote is known to the provider that wrote it and to nobody
 * else. So a reply goes back unchecked only to its own provider; a reply by
 * another model (after a switch from a local model, say) and every summary
 * is checked like a file would be. That includes the other model's thinking:
 * Pi 1.6 leaves it out of a request to a different model, but whether it is
 * sent is Pi's decision, and what is sent has to have been checked.
 */

import type { Item, Origin, Screen } from "./screen.ts";
import { embed } from "./shell.ts";

/** The `customType` of this extension's session entries. */
export const OWN_TYPE = "privacy-screen";

/** Sent in place of a text that could not be checked. */
export const WITHHELD = "[privacy screen: this text was withheld because it could not be checked for private data]";

export type Message = { role: string } & Record<string, unknown>;

/** Where the messages are going. */
export interface Target {
	/** The provider they are sent to: what it wrote itself is not new to it. */
	provider?: string;
}

export interface Plan {
	/** Every text that would reach the provider, in order. */
	items: Item[];
	/** The messages with those texts replaced; a text given as undefined is withheld. */
	build(texts: readonly (string | undefined)[]): { messages: Message[]; withheld: number };
}

type Holder = Record<string, unknown> | unknown[];

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Every string inside `holder`, however deep, with a way to replace it. */
function eachString(holder: Holder, visit: (text: string, put: (text: string) => void, path: readonly string[]) => void, path: readonly string[] = []): void {
	for (const [key, value] of Object.entries(holder)) {
		const here = [...path, key];
		if (typeof value === "string") visit(value, (text) => ((holder as Record<string, unknown>)[key] = text), here);
		else if (isRecord(value) || Array.isArray(value)) eachString(value, visit, here);
	}
}

// ------------------------------------------------------------- on the way out

export function transcript(messages: readonly Message[], target: Target = {}): Plan {
	const items: Item[] = [];
	const puts: ((text: string) => void)[] = [];
	const slot = (text: unknown, origin: Origin, put: (text: string) => void) => {
		if (typeof text !== "string") return;
		items.push({ text, origin });
		puts.push(put);
	};
	/** `holder[key]` is a text, or a list of blocks of which some are text. */
	const content = (holder: Record<string, unknown>, key: string, origin: Origin) => {
		const value = holder[key];
		if (!Array.isArray(value)) return slot(value, origin, (text) => (holder[key] = text));
		const blocks = [...value];
		holder[key] = blocks;
		blocks.forEach((block, index) => {
			if (!isRecord(block) || block.type !== "text") return;
			const copy = { ...block };
			blocks[index] = copy;
			slot(copy.text, origin, (text) => (copy.text = text));
		});
	};

	const out = messages.map((message) => {
		const copy: Message = { ...message };
		switch (message.role) {
			case "system":
				content(copy, "content", "system");
				if (isRecord(message.sections)) {
					const sections = { ...message.sections };
					copy.sections = sections;
					for (const name of Object.keys(sections)) slot(sections[name], "system", (text) => (sections[name] = text));
				}
				break;
			case "user":
			case "toolResult":
			case "custom": // from an extension
				content(copy, "content", "input");
				break;
			case "bashExecution": // a command the user ran with "!"
				if (message.excludeFromContext) break;
				slot(copy.command, "input", (text) => (copy.command = text));
				slot(copy.output, "input", (text) => (copy.output = text));
				break;
			case "compactionSummary":
			case "branchSummary": // whichever model wrote it
				slot(copy.summary, "foreign", (text) => (copy.summary = text));
				break;
			case "assistant": {
				if (!Array.isArray(message.content)) break;
				const origin: Origin = typeof message.provider === "string" && message.provider === target.provider ? "provider" : "foreign";
				const blocks = [...message.content];
				copy.content = blocks;
				blocks.forEach((block, index) => {
					if (!isRecord(block)) return;
					if (block.type === "text") {
						const text = { ...block };
						blocks[index] = text;
						slot(text.text, origin, (masked) => (text.text = masked));
					} else if (block.type === "toolCall" && (isRecord(block.arguments) || Array.isArray(block.arguments))) {
						const call = { ...block, arguments: structuredClone(block.arguments) as Holder };
						blocks[index] = call;
						eachString(call.arguments, (text, put) => slot(text, origin, put));
					} else if (block.type === "thinking" && origin === "foreign") {
						const thought = { ...block };
						blocks[index] = thought;
						slot(thought.thinking, origin, (masked) => (thought.thinking = masked));
					}
				});
				break;
			}
		}
		return copy;
	});

	return {
		items,
		build(texts) {
			let withheld = 0;
			puts.forEach((put, index) => {
				if (texts[index] === undefined) withheld++;
				put(texts[index] ?? WITHHELD);
			});
			return { messages: out, withheld };
		},
	};
}

// -------------------------------------------------------------- on the way in

/** How the arguments of tools are to be read. */
export interface ToolRules {
	/** For each tool, the arguments that are shell commands. */
	shell: Readonly<Record<string, readonly string[]>>;
	/** Tools whose arguments are code. They are left as they are: the calls such code makes are tool calls themselves. */
	code: readonly string[];
}

/** Where in a tool call's arguments a value was put (a path), and the text that now stands there. */
export type Settled = Map<string, string>;

export interface Unmasked {
	message: Message;
	/** For each tool call (by id) whose arguments were given their values: what was put where. */
	settled: Map<string, Settled>;
}

/**
 * A finished reply with the placeholders in its text and in the arguments of its tool calls replaced
 * by their values, or undefined if there were none. What the provider wrote is remembered, so that
 * the reply goes back out as it came. Thinking is never touched.
 *
 * A tool call whose arguments cannot take their values (a shell command the value cannot be put into
 * safely) is left as it was written; `unmaskInput` refuses it when the tool is about to run.
 */
export function unmaskAssistant(message: Message, screen: Screen, rules: ToolRules = { shell: {}, code: [] }): Unmasked | undefined {
	if (message.role !== "assistant" || !Array.isArray(message.content)) return undefined;
	const settled = new Map<string, Settled>();
	let changed = false;
	const content = message.content.map((block) => {
		if (!isRecord(block)) return block;
		if (block.type === "toolCall" && (isRecord(block.arguments) || Array.isArray(block.arguments))) {
			const written = block.arguments as Holder;
			const name = typeof block.name === "string" ? block.name : "";
			let result = block;
			if (!rules.code.includes(name)) {
				const real = structuredClone(written);
				const record: Settled = new Map();
				if (unmaskInput(real, screen, rules.shell[name] ?? [], { record }) === undefined && record.size > 0) {
					result = { ...block, arguments: real };
					if (typeof block.id === "string") settled.set(block.id, record);
					changed = true;
				}
			}
			// Where a call is kept as it was written, it also has to go back out as it was written.
			eachString(written, (text) => screen.remember(text, text));
			return result;
		}
		// Thinking stays as it was written. Its placeholders are the screen's own, should another provider ever be sent it.
		if (block.type === "thinking" && typeof block.thinking === "string") screen.remember(block.thinking, block.thinking);
		if (block.type !== "text" || typeof block.text !== "string") return block;
		const real = screen.unmask(block.text);
		screen.remember(real, block.text);
		screen.remember(block.text, block.text); // should the text be kept as it was written after all
		if (real === block.text) return block;
		changed = true;
		return { ...block, text: real };
	});
	return changed ? { message: { ...message, content }, settled } : undefined;
}

/**
 * Give the arguments of a tool call their values, in place. The arguments named in `shell` are shell
 * commands and get them quoted for the shell (`embed`); all others are data and get them as they are.
 * Returns the reason if a command cannot take a value safely; then nothing has been changed.
 *
 * A value is put in once. What `settled` says was put in before is not looked at again: a value may
 * itself contain text shaped like a placeholder, and that text must stay text. `record` is filled
 * with what is put in now.
 */
export function unmaskInput(input: unknown, screen: Screen, shell: readonly string[], options: { settled?: ReadonlyMap<string, string>; record?: Settled } = {}): string | undefined {
	if (!isRecord(input) && !Array.isArray(input)) return undefined;
	const changes: { put: (text: string) => void; real: string; masked: string; where: string }[] = [];
	let refusal: string | undefined;
	eachString(input, (text, put, path) => {
		const where = path.join("\0");
		if (options.settled?.get(where) === text) return;
		let real: string;
		if (shell.includes(path[0]!)) {
			const placed = embed(text, (token) => screen.vault.lookup(token)?.value);
			if ("refused" in placed) {
				refusal ??= placed.refused;
				return;
			}
			real = placed.command;
		} else {
			real = screen.unmask(text);
		}
		if (real !== text) changes.push({ put, real, masked: text, where });
	});
	if (refusal !== undefined) return refusal;
	for (const change of changes) {
		change.put(change.real);
		screen.remember(change.real, change.masked);
		options.record?.set(change.where, change.real);
	}
	return undefined;
}

// ------------------------------------------------------------- the last look

/** Parts of a provider's request that are identifiers, signed, encrypted or binary: never text to replace in. */
const OPAQUE_KEYS = new Set([
	"id", "call_id", "tool_use_id", "tool_call_id", "name", "type", "role", "model",
	"data", "signature", "thought_signature", "thoughtSignature", "encrypted_content", "encryptedContent",
]); // fmt: skip
const OPAQUE_TYPES = new Set(["thinking", "redacted_thinking", "reasoning"]);

/**
 * Replace known values anywhere in a request body about to be sent. Returns the same object if
 * there was nothing to replace, and otherwise a copy that shares what did not change.
 */
export function sweepPayload(payload: unknown, screen: Screen): { payload: unknown; hits: number } {
	let hits = 0;
	const visit = (node: unknown): unknown => {
		if (typeof node === "string") {
			if (node.startsWith("data:")) return node;
			const swept = screen.sweep(node);
			hits += swept.hits;
			return swept.text;
		}
		if (Array.isArray(node)) {
			let copy: unknown[] | undefined;
			node.forEach((child, index) => {
				const next = visit(child);
				if (next !== child) (copy ??= [...node])[index] = next;
			});
			return copy ?? node;
		}
		if (!isRecord(node) || (typeof node.type === "string" && OPAQUE_TYPES.has(node.type))) return node;
		let copy: Record<string, unknown> | undefined;
		for (const [key, child] of Object.entries(node)) {
			if (typeof child === "string" && OPAQUE_KEYS.has(key)) continue;
			const next = visit(child);
			if (next !== child) (copy ??= { ...node })[key] = next;
		}
		return copy ?? node;
	};
	return { payload: visit(payload), hits };
}
