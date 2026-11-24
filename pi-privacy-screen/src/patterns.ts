import type { Kind } from "./vault.ts";

export interface Found {
	start: number;
	end: number;
	kind: Kind;
}

interface Rule {
	name: string;
	pattern: RegExp;
	groups?: number[];
	until?: RegExp;
	check?: (value: string) => boolean;
}

const SENSITIVE_NAME = "(?:secret|token|passw(?:or)?d|passphrase|api[_-]?key|private[_-]?key|access[_-]?key|auth[_-]?key|credentials?)";

const RULES: Rule[] = [
	{ name: "private-key-block", pattern: /-----BEGIN (?:[A-Z0-9]+ ){0,4}PRIVATE KEY(?: BLOCK)?-----/g, until: /-----END (?:[A-Z0-9]+ ){0,4}PRIVATE KEY(?: BLOCK)?-----/g },
	{ name: "aws-access-key", pattern: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g },
	{ name: "github-token", pattern: /\bgh[pousr]_[A-Za-z0-9]{36,255}\b/g },
	{ name: "github-fine-grained-pat", pattern: /\bgithub_pat_[A-Za-z0-9_]{22,255}\b/g },
	{ name: "openai-or-anthropic-key", pattern: /\bsk-[A-Za-z0-9_-]{20}[A-Za-z0-9_-]*/g, check: (value) => /[0-9]/.test(value) && /[A-Z]/.test(value) },
	{ name: "stripe-key", pattern: /\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16}[A-Za-z0-9]*\b/g },
	{ name: "stripe-webhook-secret", pattern: /\bwhsec_[A-Za-z0-9]{24}[A-Za-z0-9]*\b/g },
	{ name: "slack-token", pattern: /\bxox[abposr]-[A-Za-z0-9-]{10}[A-Za-z0-9-]*/g },
	{ name: "google-api-key", pattern: /\bAIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])/g },
	{ name: "jwt", pattern: /(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8}[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]{8}[A-Za-z0-9_-]*\.[A-Za-z0-9_-]{8}[A-Za-z0-9_-]*/g },
	{ name: "npm-token", pattern: /\bnpm_[A-Za-z0-9]{36}\b/g },
	{ name: "huggingface-token", pattern: /\bhf_[A-Za-z0-9]{30}[A-Za-z0-9]*\b/g },
	{ name: "gitlab-pat", pattern: /\bglpat-[A-Za-z0-9_-]{20}[A-Za-z0-9_-]*/g },
	{ name: "url-password", pattern: /(?<=\b[a-z][a-z0-9+.-]{0,31}):\/\/[^\s:@/]+:([^\s@/]{3}[^\s@/]*)@/dgi, groups: [1], check: (value) => !/[${}<>%*]/.test(value) },
	{ name: "bearer-token", pattern: /\bBearer\s+([A-Za-z0-9._~+/-]{20}[A-Za-z0-9._~+/-]*=*)/dg, groups: [1] },
	{
		name: "sensitive-assignment",
		pattern: new RegExp(
			`${SENSITIVE_NAME}[A-Za-z0-9_]{0,64}["']?\\s*[=:]\\s*` +
				`(?:"([^"\\\\\\n]*(?:\\\\[^\\n][^"\\\\\\n]*)*)"|'([^'\\\\\\n]*(?:\\\\[^\\n][^'\\\\\\n]*)*)'|([^\\s"'\`,;]{6}[^\\s"'\`,;]*(?<!\\\\)))`,
			"dgi",
		),
		groups: [1, 2, 3],
		check: plausibleSecret,
	},
];

export function findPatterns(text: string): Found[] {
	const candidates: Found[] = [];
	for (const rule of RULES) {
		let past = 0;
		for (const match of text.matchAll(rule.pattern)) {
			let start = match.index;
			let end = start + match[0].length;
			if (rule.until) {
				if (start < past) continue;
				const close = new RegExp(rule.until.source, "g");
				close.lastIndex = end;
				const last = close.exec(text);
				if (!last) break;
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

const WHITESPACE_OR_DELIMITER = /[\s(){}[\]<>$]/;
const URL_SCHEME = /^[a-z][a-z0-9+.-]*:\/\//i;
const URL_WITH_CREDENTIALS = /^[a-z][a-z0-9+.-]*:\/\/[^\s/@:]+:[^\s/@]+@/i;
const FILE_PATH = /^(?:\/|\.\.?\/|~)/;
const NUMERIC_OR_DATE_LITERAL = /^[0-9.,:+-]+$/;
const HOST_PORT = /^[A-Za-z0-9.-]+:[0-9]+$/;
const DOTTED_IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$/;
const HAS_DIGIT = /[0-9]/;

export function plausibleSecret(value: string): boolean {
	if (value.length < 8) return false;
	if (WHITESPACE_OR_DELIMITER.test(value)) return false;
	if (URL_SCHEME.test(value)) return URL_WITH_CREDENTIALS.test(value);
	if (FILE_PATH.test(value)) return false;
	if (NUMERIC_OR_DATE_LITERAL.test(value)) return false;
	if (HOST_PORT.test(value)) return false;
	if (DOTTED_IDENTIFIER.test(value)) return false;
	return HAS_DIGIT.test(value);
}
