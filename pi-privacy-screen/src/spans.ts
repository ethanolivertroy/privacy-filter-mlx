import type { Found } from "./patterns.ts";
import type { Kind } from "./vault.ts";

export interface Span {
	start: number;
	end: number;
	label: string;
}

export const KIND_OF_LABEL: Readonly<Record<string, Kind>> = {
	secret: "SECRET",
	private_email: "EMAIL",
	private_person: "PERSON",
	private_phone: "PHONE",
	private_address: "ADDRESS",
	account_number: "ACCOUNT",
	private_url: "URL",
	private_date: "DATE",
};
export const ALL_LABELS: readonly string[] = Object.keys(KIND_OF_LABEL);
export const DEFAULT_LABELS: readonly string[] = ["secret", "private_email", "private_phone", "private_person", "private_address", "account_number"];

// The part before the "@" is at most 64 characters long. Without that bound, a long span with no address in it
// would be read to its end from every one of its positions.
const EMAIL = /[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}/;
const ASSIGNED_NAME = /^[A-Za-z_][A-Za-z0-9_.-]*["']?\s*[=:](?!\/\/)\s*/;

const LEADING = /[\s"'`([{<=:,;]/;
const TRAILING = /[\s"'`)\]}>,;:.]/;
const PHONE_LEADING = /[\s"'`=:,;]/;
const PHONE_TRAILING = /[\s"'`,;:.]/;

type Range = [start: number, end: number];

export function cleanSpans(text: string, spans: readonly Span[], labels: ReadonlySet<string>): Found[] {
	const found: Found[] = [];
	for (const span of spans) {
		const kind = KIND_OF_LABEL[span.label];
		if (kind === undefined || !labels.has(span.label)) continue;
		if (!Number.isInteger(span.start) || !Number.isInteger(span.end)) continue;
		if (span.start < 0 || span.end > text.length || span.start >= span.end) continue;
		for (const line of lines(text, span.start, span.end)) {
			const value = narrow(text, line, span.label);
			if (value) found.push({ start: value[0], end: value[1], kind });
		}
	}
	return found.sort((a, b) => a.start - b.start || b.end - a.end);
}

function lines(text: string, start: number, end: number): Range[] {
	const ranges: Range[] = [];
	let from = start;
	for (let index = start; index <= end; index++) {
		if (index === end || text[index] === "\n" || text[index] === "\r") {
			if (index > from) ranges.push([from, index]);
			from = index + 1;
		}
	}
	return ranges;
}

function trim(text: string, [start, end]: Range, leading = LEADING, trailing = TRAILING): Range {
	for (;;) {
		if (start < end && leading.test(text[start]!)) start++;
		else if (start + 1 < end && text[start] === "\\" && leading.test(text[start + 1]!)) start += 2;
		else break;
	}
	while (end > start && (trailing.test(text[end - 1]!) || text[end - 1] === "\\")) end--;
	return [start, end];
}

function count(value: string, pattern: RegExp): number {
	return value.match(pattern)?.length ?? 0;
}

function narrow(text: string, range: Range, label: string): Range | undefined {
	if (label === "private_phone") {
		const [start, end] = trim(text, range, PHONE_LEADING, PHONE_TRAILING);
		return count(text.slice(start, end), /[0-9]/g) >= 7 ? [start, end] : undefined;
	}

	let [start, end] = trim(text, range);
	const value = () => text.slice(start, end);
	switch (label) {
		case "secret": {
			const name = ASSIGNED_NAME.exec(value());
			if (name) [start, end] = trim(text, [start + name[0].length, end]);
			if (/\s/.test(value())) {
				const words = [...value().matchAll(/\S+/g)];
				const longest = words.reduce((a, b) => (b[0].length > a[0].length ? b : a));
				if (longest[0].length >= 16 && /[0-9]/.test(longest[0])) {
					[start, end] = trim(text, [start + longest.index, start + longest.index + longest[0].length]);
				}
			}
			return end - start >= 6 ? [start, end] : undefined;
		}
		case "private_email": {
			const email = EMAIL.exec(value());
			return email ? [start + email.index, start + email.index + email[0].length] : undefined;
		}
		case "account_number":
			return count(value(), /[0-9]/g) >= 6 && count(value(), /[A-Za-z0-9]/g) >= 8 ? [start, end] : undefined;
		case "private_person":
			return count(value(), /\p{L}/gu) >= 2 ? [start, end] : undefined;
		case "private_address":
			return end - start >= 5 && /\p{L}/u.test(value()) ? [start, end] : undefined;
		default:
			return end > start ? [start, end] : undefined;
	}
}
