import { createHash } from "node:crypto";

import { findPatterns } from "./patterns.ts";
import type { Found } from "./patterns.ts";
import { DEFAULT_LABELS, cleanSpans } from "./spans.ts";
import type { Span } from "./spans.ts";
import { Vault, tokenPattern } from "./vault.ts";
import type { Entry } from "./vault.ts";

export type Origin = "input" | "system" | "provider" | "foreign";

export interface Item {
	text: string;
	origin: Origin;
}

export type Detect = (texts: readonly string[]) => Promise<(Span[] | Error)[]>;

export interface Problem {
	down: boolean;
	error: Error;
	count: number;
}

export interface Masked {
	texts: (string | undefined)[];
	problem?: Problem;
}

const MIN_GLOBAL_REPLACE_LENGTH = 3;
const WORD = /[\p{L}\p{N}]/u;
const MAX_ROUNDS = 8;
const UNCHANGED_TEXT = null;
const UNCLAIMED = -1;

type RememberedMask = string | typeof UNCHANGED_TEXT;
type Mode = Exclude<Origin, "foreign"> | "kept" | "payload";

interface Way {
	key: string;
	text: string;
	mode: Mode;
	detect: boolean;
}

interface Checked {
	found: readonly Found[];
	text: string;
	mode: Mode;
	masked?: string;
	stamp?: string;
	pass: number;
}

interface Known {
	value: string;
	entry?: Entry;
	anywhere: boolean;
}

interface Claim {
	start: number;
	end: number;
	put: string;
}

export class Screen {
	readonly vault: Vault;
	readonly labels: ReadonlySet<string>;
	onRemember?: (key: string, masked: RememberedMask) => void;

	private readonly detect: Detect;
	private readonly checked = new Map<string, Checked>();
	private readonly remembered = new Map<string, RememberedMask>();
	private readonly seeds = new Set<string>();
	private epoch = 0;
	private pass = 0;
	private known?: { stamp: string; list: Known[] };

	constructor(detect: Detect, options: { vault?: Vault; labels?: Iterable<string> } = {}) {
		this.detect = detect;
		this.vault = options.vault ?? new Vault();
		this.labels = new Set(options.labels ?? DEFAULT_LABELS);
	}

	async mask(items: readonly Item[]): Promise<Masked> {
		const texts: (string | undefined)[] = new Array(items.length).fill(undefined);
		const checks: (Checked | undefined)[] = new Array(items.length).fill(undefined);
		const waiting = new Map<string, { way: Way; indexes: number[] }>();
		let problem: Problem | undefined;
		const fail = (error: Error, down: boolean, count: number) => {
			problem = { down, error, count: (problem?.count ?? 0) + count };
		};
		const search = (text: string, spans: readonly Span[] | undefined): Found[] | Error => {
			try {
				return spans ? [...findPatterns(text), ...cleanSpans(text, spans, this.labels)] : findPatterns(text);
			} catch (error) {
				return new Error(`a text could not be searched for secrets (${asError(error).message})`);
			}
		};

		items.forEach((item, index) => {
			if (item.text === "") {
				texts[index] = "";
				return;
			}
			const way = this.wayFor(item);
			let check = this.checked.get(way.key);
			if (!check && !way.detect) {
				const found = way.mode === "system" ? search(way.text, undefined) : [];
				if (found instanceof Error) return fail(found, false, 1);
				check = this.keep(way, found);
			}
			if (check) checks[index] = check;
			else if (waiting.has(way.key)) waiting.get(way.key)!.indexes.push(index);
			else waiting.set(way.key, { way, indexes: [index] });
		});

		if (waiting.size > 0) {
			const asked = [...waiting.entries()];
			let answers: (Span[] | Error)[] | undefined;
			try {
				answers = await this.detect(asked.map(([, wait]) => wait.way.text));
				if (answers.length !== asked.length) throw new Error("the detector answered for a different number of texts");
			} catch (error) {
				answers = undefined;
				fail(asError(error), true, asked.reduce((sum, [, wait]) => sum + wait.indexes.length, 0));
			}
			answers?.forEach((answer, position) => {
				const [, wait] = asked[position]!;
				const found = answer instanceof Error ? answer : search(wait.way.text, answer);
				if (found instanceof Error) return fail(found, false, wait.indexes.length);
				const check = this.keep(wait.way, found);
				for (const index of wait.indexes) checks[index] = check;
			});
		}

		for (let round = 0, size = -1; round < MAX_ROUNDS && size !== this.vault.entries.length; round++) {
			size = this.vault.entries.length;
			checks.forEach((check, index) => {
				if (!check) return;
				try {
					texts[index] = this.render(check);
				} catch (error) {
					texts[index] = checks[index] = undefined;
					fail(new Error(`a text could not be rewritten (${asError(error).message})`), false, 1);
				}
			});
		}
		return problem ? { texts, problem } : { texts };
	}

	endPass(): void {
		for (const [key, check] of this.checked) {
			if (check.pass < this.pass - 1) this.checked.delete(key);
		}
		this.pass++;
	}

	unmask(text: string): string {
		if (!text.includes("<")) return text;
		return text.replace(tokenPattern(), (token) => this.vault.lookup(token)?.value ?? token);
	}

	remember(real: string, masked: string): void {
		if (real === masked && !tokenPattern().test(masked)) return;
		const key = digest("provider", real);
		const kept: RememberedMask = real === masked ? UNCHANGED_TEXT : masked;
		if (this.remembered.has(key) && this.remembered.get(key) === kept) return;
		this.restore(key, kept);
		this.onRemember?.(key, kept);
	}

	restore(key: string, masked: RememberedMask): void {
		this.remembered.set(key, masked);
	}

	sweep(text: string): { text: string; hits: number } {
		if (this.knownValues().length === 0) return { text, hits: 0 };
		return this.replace(text, "payload", []);
	}

	seed(values: Iterable<string>): void {
		for (const value of values) {
			if (value.length >= MIN_GLOBAL_REPLACE_LENGTH) this.seeds.add(value);
		}
		this.epoch++;
	}

	reveal(token: string, revealed: boolean): boolean {
		if (!this.vault.setRevealed(token, revealed)) return false;
		this.epoch++;
		return true;
	}

	private wayFor(item: Item): Way {
		const { text, origin } = item;
		if (origin !== "provider" && origin !== "foreign") return { key: digest(origin, text), text, mode: origin, detect: origin !== "system" };
		const foreign = origin === "foreign";
		const memory = digest("provider", text);
		if (this.remembered.has(memory)) {
			const written = this.remembered.get(memory) ?? text;
			return { key: `${foreign ? "kept+" : "kept:"}${digest("provider", written)}`, text: written, mode: "kept", detect: foreign };
		}
		return foreign ? { key: digest("input", text), text, mode: "input", detect: true } : { key: memory, text, mode: "provider", detect: false };
	}

	private keep(way: Way, found: readonly Found[]): Checked {
		const check: Checked = { found, text: way.text, mode: way.mode, pass: this.pass };
		this.checked.set(way.key, check);
		return check;
	}

	private stamp(): string {
		return `${this.vault.version}:${this.epoch}`;
	}

	private render(check: Checked): string {
		check.pass = this.pass;
		if (check.masked === undefined || check.stamp !== this.stamp()) {
			check.masked = this.replace(check.text, check.mode, check.found).text;
			check.stamp = this.stamp();
		}
		return check.masked;
	}

	private knownValues(): Known[] {
		const stamp = this.stamp();
		if (this.known?.stamp !== stamp) {
			const list: Known[] = [];
			for (const entry of this.vault.entries) {
				if (!entry.revealed && entry.value.length >= MIN_GLOBAL_REPLACE_LENGTH) list.push({ value: entry.value, entry, anywhere: keyLike(entry.value) });
			}
			for (const value of this.seeds) {
				if (!this.vault.find(value)) list.push({ value, anywhere: keyLike(value) });
			}
			list.sort((a, b) => b.value.length - a.value.length || (a.value < b.value ? -1 : 1));
			this.known = { stamp, list };
		}
		return this.known.list;
	}

	private replace(text: string, mode: Mode, found: readonly Found[]): { text: string; hits: number } {
		const claims: (Claim | undefined)[] = [];
		const owner = new Int32Array(text.length).fill(UNCLAIMED);
		const claim = (start: number, end: number, put: string) => {
			const id = claims.length;
			claims.push({ start, end, put });
			owner.fill(id, start, end);
		};
		const free = (start: number, end: number) => {
			for (let at = start; at < end; at++) if (owner[at] !== UNCLAIMED) return false;
			return true;
		};
		const shown = (entry: Entry) => (entry.revealed ? entry.value : entry.token);

		const claimPlaceholders = (): Found[] => {
			const literals: Found[] = [];
			for (const match of text.matchAll(tokenPattern())) {
				const [start, end] = [match.index, match.index + match[0].length];
				const reference = mode === "kept" ? this.vault.lookup(match[0]) : undefined;
				if (reference) claim(start, end, shown(reference));
				else if (mode === "kept" || mode === "payload") claim(start, end, match[0]);
				else literals.push({ start, end, kind: "LITERAL" });
			}
			return literals;
		};

		const claimKnownValues = (): void => {
			for (const known of this.knownValues()) {
				const forms = [known.value];
				if (mode === "payload") {
					const inJson = JSON.stringify(known.value).slice(1, -1);
					if (inJson !== known.value) forms.push(inJson);
				}
				for (const form of forms) {
					for (let at = text.indexOf(form); at !== -1; at = text.indexOf(form, at + 1)) {
						const end = at + form.length;
						if (!free(at, end) || !(known.anywhere || standsAlone(text, at, end))) continue;
						claim(at, end, (known.entry ?? this.vault.allocate("SECRET", known.value)).token);
						at = end - 1;
					}
				}
			}
		};

		const claimDetectedSpans = (detected: readonly Found[]): void => {
			for (const range of detected) {
				const entry = this.vault.find(text.slice(range.start, range.end))!;
				if (entry.revealed) continue;
				const holders = new Set<number>();
				let covered = true;
				for (let at = range.start; at < range.end; at++) {
					if (owner[at] === UNCLAIMED) covered = false;
					else holders.add(owner[at]!);
				}
				if (holders.size === 0) claim(range.start, range.end, entry.token);
				if (holders.size === 0 || covered) continue;
				let { start, end } = range;
				for (const holder of holders) {
					const other = claims[holder]!;
					[start, end] = [Math.min(start, other.start), Math.max(end, other.end)];
					claims[holder] = undefined;
				}
				claim(start, end, shown(this.vault.allocate(range.kind, text.slice(start, end))));
			}
		};

		const literals = claimPlaceholders();
		const kept = claims.length;
		const detected = merged([...found, ...literals]).filter((range) => free(range.start, range.end));
		for (const range of detected) this.vault.allocate(range.kind, text.slice(range.start, range.end));
		claimKnownValues();
		claimDetectedSpans(detected);

		let out = "";
		let at = 0;
		const made = claims.filter((made) => made !== undefined);
		for (const one of [...made].sort((a, b) => a.start - b.start)) {
			out += text.slice(at, one.start) + one.put;
			at = one.end;
		}
		return { text: out + text.slice(at), hits: made.length - kept };
	}
}

function digest(origin: Origin, text: string): string {
	return createHash("sha256").update(origin).update("\0").update(text).digest("base64url");
}

function asError(error: unknown): Error {
	return error instanceof Error ? error : new Error(String(error));
}

function keyLike(value: string): boolean {
	return value.length >= 12 && /[A-Za-z]/.test(value) && /[0-9]/.test(value) && !/\s/.test(value);
}

function standsAlone(text: string, start: number, end: number): boolean {
	if (start > 0 && WORD.test(text[start]!) && WORD.test(text[start - 1]!)) return false;
	if (end < text.length && WORD.test(text[end - 1]!) && WORD.test(text[end]!)) return false;
	return true;
}

function merged(ranges: readonly Found[]): Found[] {
	const sorted = [...ranges].sort((a, b) => a.start - b.start || b.end - a.end);
	const out: Found[] = [];
	for (const range of sorted) {
		const last = out[out.length - 1];
		if (last && range.start < last.end) last.end = Math.max(last.end, range.end);
		else out.push({ ...range });
	}
	return out;
}
