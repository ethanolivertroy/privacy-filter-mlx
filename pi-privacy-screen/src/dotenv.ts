import { readFile, readdir, stat } from "node:fs/promises";
import { join } from "node:path";

import { findPatterns } from "./patterns.ts";

const MAX_BYTES = 1024 * 1024;
const MIN_LENGTH = 8;

const SECRET_NAME =
	/SECRET|TOKEN|PASSWORD|PASSWD|PASSPHRASE|CREDENTIAL|PRIVATE|SIGNING|ENCRYPTION|SALT|HMAC|API_?KEY|ACCESS_?KEY|AUTH_?KEY|(^|_)(KEY|PAT|PWD|PASS)$/i;
const PUBLIC_NAME = /^(NEXT_PUBLIC_|NUXT_PUBLIC_|EXPO_PUBLIC_|PUBLIC_|VITE_|REACT_APP_|GATSBY_)|PUBLIC_?KEY|PUBLISHABLE/i;
const EXAMPLE_FILE = /\.(example|sample|template|dist|defaults)$/i;

const ASSIGNMENT = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(.*)$/;

function isSetting(value: string): boolean {
	return (
		/^[-+]?[0-9.,_]+$/.test(value) ||
		/^(true|false|yes|no|on|off|null|none|undefined)$/i.test(value) ||
		/^(~|\.{0,2})\//.test(value) ||
		/^[a-z][a-z0-9+.-]*:\/\//i.test(value) ||
		value.includes("$")
	);
}

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

function matches(name: string, pattern: string): boolean {
	const body = pattern
		.split("*")
		.map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
		.join(".*");
	if (pattern.startsWith("*") && name.startsWith(".")) return false;
	return new RegExp(`^${body}$`).test(name);
}

export function selectEnvFiles(names: readonly string[], patterns: readonly string[]): string[] {
	const usable = patterns.filter((pattern) => !pattern.includes("/") && !pattern.includes("\\"));
	return names.filter((name) => !EXAMPLE_FILE.test(name) && usable.some((pattern) => matches(name, pattern)));
}

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
