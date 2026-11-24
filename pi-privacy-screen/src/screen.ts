/**
 * The screen itself: real text in, text that is safe to send out, and back.
 *
 * Everything kept on this machine is real; everything sent to a provider is
 * masked. So each piece of text is masked on the way out according to where
 * it came from:
 *
 *   input     what the user sent (typed, pasted, a file attached to the
 *             message) and what tools returned (results, shell output,
 *             files). Screened by the detector.
 *   system    the prompt and project instructions. Known values and key
 *             formats only; it is not sent to the detector.
 *   provider  what the model wrote, going back to the provider that wrote
 *             it. It was unmasked for local use, so it goes back as it
 *             arrived (`remember`), its placeholders still meaning their
 *             values. Text nobody remembered is treated like input, minus
 *             the detector: this provider has seen it already.
 *   foreign   what some other model wrote, and every summary: this provider
 *             has not seen it. Screened by the detector like input, except
 *             that placeholders the screen put there itself keep their meaning.
 *
 * Only the screen makes placeholders. Text that merely looks like one, in a
 * file, in a tool's output or in what the user sent, is passed as a literal
 * (`<LITERAL_n>`) and comes back as the text it was. Otherwise a file that
 * happens to say `<EMAIL_1>` would be given somebody's address when the model
 * writes it back. A placeholder refers to a value only in text the model wrote
 * in answer to a masked request, which is what `remember` records.
 *
 * `sweep` is a last look at whatever is about to be sent, in any shape: it
 * replaces known values and reports how many it still found.
 *
 * Nothing here knows about Pi or about how the detector is reached.
 */

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

/**
 * Spans for each text (UTF-16 offsets), or an Error for a text that could not
 * be checked. Rejects when the detector cannot be reached at all.
 */
export type Detect = (texts: readonly string[]) => Promise<(Span[] | Error)[]>;

export interface Problem {
	/** True when the detector could not be reached; false when it answered but failed on some texts. */
	down: boolean;
	error: Error;
	/** How many texts could not be checked. */
	count: number;
}

export interface Masked {
	/** For each item the text to send, or undefined when it could not be checked and must not be sent. */
	texts: (string | undefined)[];
	problem?: Problem;
}

/** A value is replaced wherever it appears only if it is at least this long; shorter ones only where detected. */
const SPREAD_FROM = 3;
const WORD = /[\p{L}\p{N}]/u;
const MAX_ROUNDS = 8;

/** How placeholder-shaped text is read, and what else is replaced: by origin, or one of two special cases. */
type Mode = Exclude<Origin, "foreign"> | "kept" | "payload";

/** How one text is to be screened. */
interface Way {
	key: string;
	/** The text to work on: the item's, or for remembered text what the provider wrote. */
	text: string;
	mode: Mode;
	/** Whether the detector has to see it. */
	detect: boolean;
}

interface Checked {
	/** What the patterns and the detector found in this text. */
	found: readonly Found[];
	text: string;
	mode: Mode;
	masked?: string;
	/** The state of the vault `masked` was made with. */
	stamp?: string;
	pass: number;
}

interface Known {
	value: string;
	/** Missing for a seed that has no placeholder yet. */
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
	/** Called with what `remember` stored, so that it can be saved with the session and given to `restore` later. */
	onRemember?: (key: string, masked: string | null) => void;

	private readonly detect: Detect;
	private readonly checked = new Map<string, Checked>();
	private readonly remembered = new Map<string, string | null>(); // null: goes back as it is
	private readonly seeds = new Set<string>();
	private epoch = 0;
	private pass = 0;
	private known?: { stamp: string; list: Known[] };

	constructor(detect: Detect, options: { vault?: Vault; labels?: Iterable<string> } = {}) {
		this.detect = detect;
		this.vault = options.vault ?? new Vault();
		this.labels = new Set(options.labels ?? DEFAULT_LABELS);
	}

	// ------------------------------------------------------------------ out

	/**
	 * What to send in place of each text. Never rejects: a text that could not be checked comes back
	 * undefined, whether it was the detector that failed on it or this code.
	 */
	async mask(items: readonly Item[]): Promise<Masked> {
		const texts: (string | undefined)[] = new Array(items.length).fill(undefined);
		const checks: (Checked | undefined)[] = new Array(items.length).fill(undefined);
		const waiting = new Map<string, { way: Way; indexes: number[] }>();
		let problem: Problem | undefined;
		const fail = (error: Error, down: boolean, count: number) => {
			problem = { down, error, count: (problem?.count ?? 0) + count };
		};
		/** The patterns, or the Error for a text they cannot get through (the pattern engine has its limits). */
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

		// A value found in one text has to be replaced in all of them, also in those rendered before it was found.
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

	/** Forget the checks of texts that were not screened in the last passes. Call after each full pass over a conversation. */
	endPass(): void {
		for (const [key, check] of this.checked) {
			if (check.pass < this.pass - 1) this.checked.delete(key);
		}
		this.pass++;
	}

	// ------------------------------------------------------------------- in

	/** Put the values back: every known placeholder becomes what it stands for. One pass; values are not looked into. */
	unmask(text: string): string {
		if (!text.includes("<")) return text;
		return text.replace(tokenPattern(), (token) => this.vault.lookup(token)?.value ?? token);
	}

	/**
	 * `masked` is text exactly as the provider wrote it, `real` what it is on this machine (the same text, if
	 * it held no placeholders). From now on `real` goes back as `masked`, whatever else is learned.
	 */
	remember(real: string, masked: string): void {
		if (real === masked && !tokenPattern().test(masked)) return; // nothing in it that could be misread later
		const key = digest("provider", real);
		const kept = real === masked ? null : masked;
		if (this.remembered.has(key) && this.remembered.get(key) === kept) return;
		this.restore(key, kept);
		this.onRemember?.(key, kept);
	}

	/** Load what an earlier session remembered. */
	restore(key: string, masked: string | null): void {
		this.remembered.set(key, masked);
	}

	// ------------------------------------------------------------ last look

	/**
	 * Replace known values in text that is about to be sent, whatever it is part of. A value is also found in
	 * the form it has inside a JSON string. `hits` above zero means something got past `mask`.
	 */
	sweep(text: string): { text: string; hits: number } {
		if (this.knownValues().length === 0) return { text, hits: 0 };
		return this.replace(text, "payload", []);
	}

	// ------------------------------------------------------------- settings

	/** Values to replace wherever they occur, even though nothing has detected them (from `.env` files). */
	seed(values: Iterable<string>): void {
		for (const value of values) {
			if (value.length >= SPREAD_FROM) this.seeds.add(value);
		}
		this.epoch++;
	}

	/** Let the model see one value (or stop letting it). False if there is no such placeholder. */
	reveal(token: string, revealed: boolean): boolean {
		if (!this.vault.setRevealed(token, revealed)) return false;
		this.epoch++;
		return true;
	}

	// ------------------------------------------------------------- internal

	private wayFor(item: Item): Way {
		const { text, origin } = item;
		if (origin !== "provider" && origin !== "foreign") return { key: digest(origin, text), text, mode: origin, detect: origin !== "system" };
		const foreign = origin === "foreign";
		const memory = digest("provider", text);
		if (this.remembered.has(memory)) {
			// What the provider wrote is what gets sent; for another provider it is first looked at by the detector.
			// The check is kept under what was written, so remembering something else for the same text starts afresh.
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
		return `${this.vault.entries.length}:${this.epoch}`;
	}

	private render(check: Checked): string {
		check.pass = this.pass;
		if (check.masked === undefined || check.stamp !== this.stamp()) {
			check.masked = this.replace(check.text, check.mode, check.found).text;
			check.stamp = this.stamp(); // after replacing, which may have added values
		}
		return check.masked;
	}

	/** Every value that is replaced wherever it occurs, longest first. */
	private knownValues(): Known[] {
		const stamp = this.stamp();
		if (this.known?.stamp !== stamp) {
			const list: Known[] = [];
			for (const entry of this.vault.entries) {
				if (!entry.revealed && entry.value.length >= SPREAD_FROM) list.push({ value: entry.value, entry, anywhere: keyLike(entry.value) });
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
		const owner = new Int32Array(text.length); // which claim holds each position, plus one; 0 is free
		const claim = (start: number, end: number, put: string) => {
			claims.push({ start, end, put });
			owner.fill(claims.length, start, end);
		};
		const free = (start: number, end: number) => {
			for (let at = start; at < end; at++) if (owner[at] !== 0) return false;
			return true;
		};
		const shown = (entry: Entry) => (entry.revealed ? entry.value : entry.token);

		// 1. Text shaped like a placeholder. Kept as the provider wrote it, it refers to a value. In a payload
		//    it is one that `mask` put there. Anywhere else it is a literal, which must never turn into
		//    somebody's value.
		const literals: Found[] = [];
		for (const match of text.matchAll(tokenPattern())) {
			const [start, end] = [match.index, match.index + match[0].length];
			const reference = mode === "kept" ? this.vault.lookup(match[0]) : undefined;
			if (reference) claim(start, end, shown(reference));
			else if (mode === "kept" || mode === "payload") claim(start, end, match[0]);
			else literals.push({ start, end, kind: "LITERAL" });
		}
		const kept = claims.length;

		// 2. What was detected. A detection that touches a reference is dropped; overlapping ones are one value.
		const detected = merged([...found, ...literals]).filter((range) => free(range.start, range.end));
		for (const range of detected) this.vault.allocate(range.kind, text.slice(range.start, range.end));

		// 3. Every known value, wherever it occurs.
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

		// 4. Detections that step 3 did not cover completely.
		for (const range of detected) {
			const entry = this.vault.find(text.slice(range.start, range.end))!;
			if (entry.revealed) continue;
			const holders = new Set<number>();
			let covered = true;
			for (let at = range.start; at < range.end; at++) {
				if (owner[at] === 0) covered = false;
				else holders.add(owner[at]!);
			}
			if (holders.size === 0) claim(range.start, range.end, entry.token);
			if (holders.size === 0 || covered) continue;
			// It sticks out of other replacements: together they become one value, so that no fragment is left.
			let { start, end } = range;
			for (const holder of holders) {
				const other = claims[holder - 1]!;
				[start, end] = [Math.min(start, other.start), Math.max(end, other.end)];
				claims[holder - 1] = undefined;
			}
			claim(start, end, shown(this.vault.allocate(range.kind, text.slice(start, end))));
		}

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

/** Long, without spaces, letters and digits mixed: a key, which may be glued to other characters. */
function keyLike(value: string): boolean {
	return value.length >= 12 && /[A-Za-z]/.test(value) && /[0-9]/.test(value) && !/\s/.test(value);
}

/** Not a piece of a longer word or number: "Lena" in "Lena," but not in "Lenard". */
function standsAlone(text: string, start: number, end: number): boolean {
	if (start > 0 && WORD.test(text[start]!) && WORD.test(text[start - 1]!)) return false;
	if (end < text.length && WORD.test(text[end - 1]!) && WORD.test(text[end]!)) return false;
	return true;
}

/** In order, with overlapping ranges joined into one (of the kind that starts first). */
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
