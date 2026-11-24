/**
 * Values the screen should know before anyone types them: the secrets in the
 * project's `.env` files.
 *
 * A detector guesses; a value read from `.env` is known. Each one is given to
 * `Screen.seed`, and from then on it is replaced wherever it turns up in
 * outbound text, whether or not the detector would have noticed it. Nothing
 * is sent anywhere by reading the files, and a value that never turns up never
 * gets a placeholder.
 *
 * Only values that are meant to be secret are taken: those assigned to a name
 * that says so (`*_TOKEN`, `*_SECRET`, `DB_PASSWORD`...), and anything in a
 * known key format. A region, a port or a bucket name is left alone, or the
 * model would see placeholders where it needs to read plain settings.
 */

import { readFile, readdir, stat } from "node:fs/promises";
import { join } from "node:path";

import { findPatterns } from "./patterns.ts";

const MAX_BYTES = 1024 * 1024;
const MIN_LENGTH = 8;

const SECRET_NAME =
	/SECRET|TOKEN|PASSWORD|PASSWD|PASSPHRASE|CREDENTIAL|PRIVATE|SIGNING|ENCRYPTION|SALT|HMAC|API_?KEY|ACCESS_?KEY|AUTH_?KEY|(^|_)(KEY|PAT|PWD|PASS)$/i;
/** Names whose values are shipped to browsers, or are public by definition. */
const PUBLIC_NAME = /^(NEXT_PUBLIC_|NUXT_PUBLIC_|EXPO_PUBLIC_|PUBLIC_|VITE_|REACT_APP_|GATSBY_)|PUBLIC_?KEY|PUBLISHABLE/i;
/** Files that hold examples rather than values, by the usual names. */
const EXAMPLE_FILE = /\.(example|sample|template|dist|defaults)$/i;

const ASSIGNMENT = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(.*)$/;

/** A setting rather than a secret: a number, a switch, a path, an address, or a reference to another variable. */
function isSetting(value: string): boolean {
	return (
		/^[-+]?[0-9.,_]+$/.test(value) ||
		/^(true|false|yes|no|on|off|null|none|undefined)$/i.test(value) ||
		/^(~|\.{0,2})\//.test(value) ||
		/^[a-z][a-z0-9+.-]*:\/\//i.test(value) ||
		value.includes("$")
	);
}

/** The value of one line: unquoted, and for double quotes also with its escapes read. Undefined if the quote never closes. */
function valuesOf(raw: string): string[] | undefined {
	const text = raw.trim();
	const quote = text[0];
	if (quote === '"' || quote === "'") {
		let end = 1;
		while (end < text.length && text[end] !== quote) end += quote === '"' && text[end] === "\\" ? 2 : 1;
		if (end >= text.length) return undefined;
		const written = text.slice(1, end);
		if (quote === "'") return [written];
		const read = written.replace(/\\(.)/g, (_, c: string) => (c === "n" ? "\n" : c === "t" ? "\t" : c === "r" ? "\r" : c));
		return read === written ? [written] : [written, read];
	}
	return [text.replace(/\s+#.*$/, "")];
}

/** The secrets in the content of one `.env` file. */
export function seedsFrom(content: string): string[] {
	const seeds = new Set<string>();
	for (const line of content.split(/\r?\n/)) {
		const assignment = ASSIGNMENT.exec(line);
		if (!assignment) continue;
		const [, name, raw] = assignment as unknown as [string, string, string];
		if (!SECRET_NAME.test(name) || PUBLIC_NAME.test(name)) continue;
		for (const value of valuesOf(raw) ?? []) {
			if (value.length >= MIN_LENGTH && !isSetting(value)) seeds.add(value);
		}
	}
	for (const found of findPatterns(content)) seeds.add(content.slice(found.start, found.end));
	return [...seeds];
}

/** `pattern` with `*` for any run of characters; like a shell, a leading `*` does not match a leading dot. */
function matches(name: string, pattern: string): boolean {
	const body = pattern
		.split("*")
		.map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
		.join(".*");
	if (pattern.startsWith("*") && name.startsWith(".")) return false;
	return new RegExp(`^${body}$`).test(name);
}

/** Of the names in a directory, those the patterns select, without the files that only hold examples. */
export function selectEnvFiles(names: readonly string[], patterns: readonly string[]): string[] {
	const usable = patterns.filter((pattern) => !pattern.includes("/") && !pattern.includes("\\"));
	return names.filter((name) => !EXAMPLE_FILE.test(name) && usable.some((pattern) => matches(name, pattern)));
}

/** Read the `.env` files of `directory`. Never rejects: a file that cannot be read contributes nothing. */
export async function readSeeds(directory: string, patterns: readonly string[]): Promise<{ seeds: string[]; files: string[]; skipped: string[] }> {
	const seeds = new Set<string>();
	const files: string[] = [];
	const skipped: string[] = [];
	if (patterns.length === 0) return { seeds: [], files, skipped };
	let names: string[];
	try {
		names = (await readdir(directory)).sort();
	} catch {
		return { seeds: [], files, skipped };
	}
	for (const name of selectEnvFiles(names, patterns)) {
		try {
			const path = join(directory, name);
			const info = await stat(path);
			if (!info.isFile()) continue;
			if (info.size > MAX_BYTES) {
				skipped.push(`${name}: larger than 1 MiB`);
				continue;
			}
			for (const seed of seedsFrom(await readFile(path, "utf8"))) seeds.add(seed);
			files.push(name);
		} catch (error) {
			skipped.push(`${name}: ${(error as NodeJS.ErrnoException).code ?? "could not be read"}`);
		}
	}
	return { seeds: [...seeds], files, skipped };
}
