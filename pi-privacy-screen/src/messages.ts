import type { Item, Origin, Screen } from "./screen.ts";
import { embed } from "./shell.ts";

export const OWN_TYPE = "privacy-screen";

export const WITHHELD = "[privacy screen: this text was withheld because it could not be checked for private data]";

export type Message = { role: string } & Record<string, unknown>;

export interface Target {
	provider?: string;
}

export interface Plan {
	items: Item[];
	build(texts: readonly (string | undefined)[]): { messages: Message[]; withheld: number };
}

type Holder = Record<string, unknown> | unknown[];

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function eachString(holder: Holder, visit: (text: string, put: (text: string) => void, path: readonly string[]) => void, path: readonly string[] = []): void {
	for (const [key, value] of Object.entries(holder)) {
		const here = [...path, key];
		if (typeof value === "string") visit(value, (text) => ((holder as Record<string, unknown>)[key] = text), here);
		else if (isRecord(value) || Array.isArray(value)) eachString(value, visit, here);
	}
}

export function transcript(messages: readonly Message[], target: Target = {}): Plan {
	const items: Item[] = [];
	const puts: ((text: string) => void)[] = [];
	const slot = (text: unknown, origin: Origin, put: (text: string) => void) => {
		if (typeof text !== "string") return;
		items.push({ text, origin });
		puts.push(put);
	};
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
			case "custom":
				content(copy, "content", "input");
				break;
			case "bashExecution":
				if (message.excludeFromContext) break;
				slot(copy.command, "input", (text) => (copy.command = text));
				slot(copy.output, "input", (text) => (copy.output = text));
				break;
			case "compactionSummary":
			case "branchSummary":
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

export interface ToolRules {
	shell: Readonly<Record<string, readonly string[]>>;
	code: readonly string[];
}

export type ArgPathKey = string;
export type UnmaskedPaths = Map<ArgPathKey, string>;
export type UnmaskResult = { ok: true } | { ok: false; refusal: string };

const argumentPathKey = (path: readonly string[]): ArgPathKey => path.join("\0");

export interface Unmasked {
	message: Message;
	settled: Map<string, UnmaskedPaths>;
}

export function unmaskAssistant(message: Message, screen: Screen, rules: ToolRules = { shell: {}, code: [] }): Unmasked | undefined {
	if (message.role !== "assistant" || !Array.isArray(message.content)) return undefined;
	const settled = new Map<string, UnmaskedPaths>();
	let changed = false;
	const content = message.content.map((block) => {
		if (!isRecord(block)) return block;
		if (block.type === "toolCall" && (isRecord(block.arguments) || Array.isArray(block.arguments))) {
			const written = block.arguments as Holder;
			const name = typeof block.name === "string" ? block.name : "";
			let result = block;
			if (!rules.code.includes(name)) {
				const real = structuredClone(written);
				const record: UnmaskedPaths = new Map();
				if (unmaskInputInPlace(real, screen, rules.shell[name] ?? [], { record }).ok && record.size > 0) {
					result = { ...block, arguments: real };
					if (typeof block.id === "string") settled.set(block.id, record);
					changed = true;
				}
			}
			eachString(written, (text) => screen.remember(text, text));
			return result;
		}
		if (block.type === "thinking" && typeof block.thinking === "string") screen.remember(block.thinking, block.thinking);
		if (block.type !== "text" || typeof block.text !== "string") return block;
		const real = screen.unmask(block.text);
		screen.remember(real, block.text);
		screen.remember(block.text, block.text);
		if (real === block.text) return block;
		changed = true;
		return { ...block, text: real };
	});
	return changed ? { message: { ...message, content }, settled } : undefined;
}

export function unmaskInputInPlace(
	input: unknown,
	screen: Screen,
	shell: readonly string[],
	options: { settled?: ReadonlyMap<ArgPathKey, string>; record?: UnmaskedPaths } = {},
): UnmaskResult {
	if (!isRecord(input) && !Array.isArray(input)) return { ok: true };
	const changes: { put: (text: string) => void; real: string; masked: string; where: ArgPathKey }[] = [];
	let refusal: string | undefined;
	eachString(input, (text, put, path) => {
		const where = argumentPathKey(path);
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
	if (refusal !== undefined) return { ok: false, refusal };
	for (const change of changes) {
		change.put(change.real);
		screen.remember(change.real, change.masked);
		options.record?.set(change.where, change.real);
	}
	return { ok: true };
}

const OPAQUE_KEYS = new Set([
	"id", "call_id", "tool_use_id", "tool_call_id", "name", "type", "role", "model",
	"data", "signature", "thought_signature", "thoughtSignature", "encrypted_content", "encryptedContent",
]);
const OPAQUE_TYPES = new Set(["thinking", "redacted_thinking", "reasoning"]);

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
