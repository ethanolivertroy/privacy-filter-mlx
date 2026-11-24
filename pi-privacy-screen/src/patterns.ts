/**
 * Secrets that can be recognized by their shape, without a model.
 *
 * The model is good at secrets it has to understand ("the password is ...")
 * and can miss one that only its format gives away. These patterns are the
 * other half: key formats that are unmistakable, and values assigned to names
 * like PASSWORD or API_KEY. Everything found here is of kind SECRET.
 *
 * Precision matters more than reach. A false match is replaced by a
 * placeholder the model cannot read, so code and configuration that merely
 * mention a sensitive name must be left alone (`plausibleSecret`).
 *
 * The text is whatever a tool read, so it may have been written to hurt. No
 * pattern may take longer than in proportion to the text: none starts over at
 * every word and then reads on to the end of a long run. Each one is tied to
 * something rare (a prefix, `://`, a sensitive name), and what cannot be tied
 * down is bounded or checked after the match. "Twenty or more of these" is
 * written as twenty of them and then any number (`X{20}X*`, not `X{20,}`):
 * the engine remembers every step of a counted loop, and a run of megabytes
 * would exhaust it.
 */

import type { Kind } from "./vault.ts";

export interface Found {
	start: number;
	end: number;
	kind: Kind;
}

interface Rule {
	pattern: RegExp;
	/** The capture groups that can hold the value; the first one that matched is it. Default: the whole match. */
	groups?: number[];
	/** The value runs on to the end of the first match of this pattern after it. */
	until?: RegExp;
	check?: (value: string) => boolean;
}

const SENSITIVE_NAME = "(?:secret|token|passw(?:or)?d|passphrase|api[_-]?key|private[_-]?key|access[_-]?key|auth[_-]?key|credentials?)";

const RULES: Rule[] = [
	// PEM or PGP private key block, any type: from its first line to its last
	{ pattern: /-----BEGIN (?:[A-Z0-9]+ ){0,4}PRIVATE KEY(?: BLOCK)?-----/g, until: /-----END (?:[A-Z0-9]+ ){0,4}PRIVATE KEY(?: BLOCK)?-----/g },
	// AWS access key id
	{ pattern: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g },
	// GitHub tokens
	{ pattern: /\bgh[pousr]_[A-Za-z0-9]{36,255}\b/g },
	{ pattern: /\bgithub_pat_[A-Za-z0-9_]{22,255}\b/g },
	// OpenAI, Anthropic, OpenRouter and others: "sk-" then a long tail with a digit and a capital in it
	{ pattern: /\bsk-[A-Za-z0-9_-]{20}[A-Za-z0-9_-]*/g, check: (value) => /[0-9]/.test(value) && /[A-Z]/.test(value) },
	// Stripe secret and restricted keys, webhook secrets
	{ pattern: /\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16}[A-Za-z0-9]*\b/g },
	{ pattern: /\bwhsec_[A-Za-z0-9]{24}[A-Za-z0-9]*\b/g },
	// Slack
	{ pattern: /\bxox[abposr]-[A-Za-z0-9-]{10}[A-Za-z0-9-]*/g },
	// Google API key
	{ pattern: /\bAIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])/g },
	// JSON Web Token, where it is not the tail of a longer run of the same characters
	{ pattern: /(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8}[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]{8}[A-Za-z0-9_-]*\.[A-Za-z0-9_-]{8}[A-Za-z0-9_-]*/g },
	// npm, Hugging Face, GitLab
	{ pattern: /\bnpm_[A-Za-z0-9]{36}\b/g },
	{ pattern: /\bhf_[A-Za-z0-9]{30}[A-Za-z0-9]*\b/g },
	{ pattern: /\bglpat-[A-Za-z0-9_-]{20}[A-Za-z0-9_-]*/g },
	// scheme://user:PASSWORD@host. Found from the "://", then a look back for the scheme.
	{ pattern: /(?<=\b[a-z][a-z0-9+.-]{0,31}):\/\/[^\s:@/]+:([^\s@/]{3}[^\s@/]*)@/dgi, groups: [1], check: (value) => !/[${}<>%*]/.test(value) },
	// Authorization: Bearer TOKEN
	{ pattern: /\bBearer\s+([A-Za-z0-9._~+/-]{20}[A-Za-z0-9._~+/-]*=*)/dg, groups: [1] },
	// PASSWORD=value, "api_key": "value", token: value. A quoted value runs to the quote that ends it, past
	// escaped ones. A bare value does not end in a backslash: inside a string that is itself quoted, that
	// backslash escapes the quote after the value.
	{
		pattern: new RegExp(
			`${SENSITIVE_NAME}[A-Za-z0-9_]{0,64}["']?\\s*[=:]\\s*` +
				`(?:"([^"\\\\\\n]*(?:\\\\[^\\n][^"\\\\\\n]*)*)"|'([^'\\\\\\n]*(?:\\\\[^\\n][^'\\\\\\n]*)*)'|([^\\s"'\`,;]{6}[^\\s"'\`,;]*(?<!\\\\)))`,
			"dgi",
		),
		groups: [1, 2, 3],
		check: plausibleSecret,
	},
];

/** Every secret a pattern recognizes in `text`: in order, without overlaps. */
export function findPatterns(text: string): Found[] {
	const candidates: Found[] = [];
	for (const rule of RULES) {
		let past = 0;
		for (const match of text.matchAll(rule.pattern)) {
			let start = match.index;
			let end = start + match[0].length;
			if (rule.until) {
				if (start < past) continue; // inside the block before
				const close = new RegExp(rule.until.source, "g");
				close.lastIndex = end;
				const last = close.exec(text);
				if (!last) break; // nothing ends after this point, so nothing does after a later start either
				end = past = last.index + last[0].length;
			}
			if (rule.groups) {
				const group = rule.groups.find((index) => match[index] !== undefined);
				if (group === undefined) continue;
				[start, end] = match.indices![group]!;
			}
			if (rule.check && !rule.check(text.slice(start, end))) continue;
			candidates.push({ start, end, kind: "SECRET" });
		}
	}
	candidates.sort((a, b) => a.start - b.start || b.end - a.end);
	const found: Found[] = [];
	for (const candidate of candidates) {
		if (found.length === 0 || candidate.start >= found[found.length - 1]!.end) found.push(candidate);
	}
	return found;
}

/**
 * Could `value`, assigned to a sensitive name, be a real secret rather than code or a setting?
 *
 * Used for `NAME=value` in text and for values read from `.env` files. It asks
 * for a digit and refuses anything that reads as a reference, a path, a plain
 * URL or a number.
 */
export function plausibleSecret(value: string): boolean {
	if (value.length < 8) return false;
	if (/[\s(){}[\]<>$]/.test(value)) return false; // code, a reference to a variable, a placeholder
	if (/^[a-z][a-z0-9+.-]*:\/\//i.test(value)) {
		return /^[a-z][a-z0-9+.-]*:\/\/[^\s/@:]+:[^\s/@]+@/i.test(value); // a URL only counts with a password in it
	}
	if (/^(?:\/|\.\.?\/|~)/.test(value)) return false; // a path
	if (/^[0-9.,:+-]+$/.test(value)) return false; // a number, a version, a date
	if (/^[A-Za-z0-9.-]+:[0-9]+$/.test(value)) return false; // host:port
	if (/^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$/.test(value)) return false; // settings.API_KEY_V2
	return /[0-9]/.test(value);
}
