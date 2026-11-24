import assert from "node:assert/strict";
import { test } from "node:test";

import { ALL_LABELS, DEFAULT_LABELS, cleanSpans } from "../src/spans.ts";

const labels = new Set(DEFAULT_LABELS);

function clean(text: string, raw: string, label: string, enabled: ReadonlySet<string> = labels): string[] {
	const start = text.indexOf(raw);
	assert.notEqual(start, -1, `"${raw}" is not in the text`);
	return cleanSpans(text, [{ start, end: start + raw.length, label }], enabled).map((found) => text.slice(found.start, found.end));
}

test("each label becomes a kind of placeholder", () => {
	const text = "Lena hunter22x";
	const found = cleanSpans(
		text,
		[
			{ start: 0, end: 4, label: "private_person" },
			{ start: 5, end: 14, label: "secret" },
		],
		labels,
	);

	assert.deepEqual(found, [
		{ start: 0, end: 4, kind: "PERSON" },
		{ start: 5, end: 14, kind: "SECRET" },
	]);
});

test("labels that are not screened are dropped", () => {
	assert.deepEqual(clean("see https://example.com/x", "https://example.com/x", "private_url"), []);
	assert.deepEqual(clean("on 2026-11-23", "2026-11-23", "private_date"), []);
	assert.deepEqual(clean("on 2026-11-23", "2026-11-23", "private_date", new Set(ALL_LABELS)), ["2026-11-23"]);
	assert.deepEqual(clean("some text", "some", "no_such_label", new Set(ALL_LABELS)), []);
});

test("the syntax around a value is not part of it", () => {
	assert.deepEqual(clean("MAIL=support@example.com\n", "=support@example.com", "private_email"), ["support@example.com"]);
	assert.deepEqual(clean("<lena.fischer@example.com>, x", "lena.fischer@example.com>,", "private_email"), ["lena.fischer@example.com"]);
	assert.deepEqual(clean('PHONE="+81 90-5550-0199"', '="+81 90-5550-0199', "private_phone"), ["+81 90-5550-0199"]);
	assert.deepEqual(clean("GITHUB_TOKEN=ghx_x7KqLm29ZpQr\n", "GITHUB_TOKEN=ghx_x7KqLm29ZpQr", "secret"), ["ghx_x7KqLm29ZpQr"]);
	assert.deepEqual(clean("(Lena Fischer).", "(Lena Fischer).", "private_person"), ["Lena Fischer"]);
	assert.deepEqual(clean('password: "Sup3rS3cret!"', 'password: "Sup3rS3cret!"', "secret"), ["Sup3rS3cret!"]);
});

test("the escaping around a value is not part of it", () => {
	const call = String.raw`bash(command="echo \"token: ghx_x7KqLm29ZpQrStUvWxYz0123456789\" >> deploy.log")`;
	assert.deepEqual(clean(call, String.raw`ghx_x7KqLm29ZpQrStUvWxYz0123456789\"`, "secret"), ["ghx_x7KqLm29ZpQrStUvWxYz0123456789"]);
	assert.deepEqual(clean(String.raw`echo \"ghx_x7KqLm29ZpQrStUvWxYz0123456789\"`, "ghx_x7KqLm29ZpQrStUvWxYz0123456789\\", "secret"), ["ghx_x7KqLm29ZpQrStUvWxYz0123456789"]);
	assert.deepEqual(clean(String.raw`say(\"Lena Fischer\")`, String.raw`\"Lena Fischer\"`, "private_person"), ["Lena Fischer"]);
	assert.deepEqual(clean(String.raw`call \"+81 90-5550-0199\" now`, String.raw`\"+81 90-5550-0199\"`, "private_phone"), ["+81 90-5550-0199"]);
});

test("a backslash inside a value stays", () => {
	assert.deepEqual(clean(String.raw`at C:\Users\lena\notes today`, String.raw`C:\Users\lena\notes`, "private_address"), [String.raw`C:\Users\lena\notes`]);
	assert.deepEqual(clean(String.raw`KEY=ab\ncd9876543210xyz`, String.raw`KEY=ab\ncd9876543210xyz`, "secret"), [String.raw`ab\ncd9876543210xyz`]);
});

test("a long span without an address in it is not read over and over", () => {
	for (const text of ["a.".repeat(60_000), "a".repeat(120_000), "a@".repeat(60_000)]) {
		const start = performance.now();
		assert.deepEqual(clean(text, text, "private_email"), []);
		const ms = performance.now() - start;
		assert.ok(ms < 1000, `${ms.toFixed(0)} ms for ${text.length} characters of "${text.slice(0, 4)}"`);
	}
});

test("padding at the end of a secret is kept", () => {
	assert.deepEqual(clean("KEY=YWJjZGVmZ2hpamtsbW5vcA==\n", "KEY=YWJjZGVmZ2hpamtsbW5vcA==", "secret"), ["YWJjZGVmZ2hpamtsbW5vcA=="]);
});

test("a span over several lines is one value per line", () => {
	const text = "AWS_ACCESS_KEY_ID=AKIXIOSFODNN7EXAMPLE\nAWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n";
	const raw = "AKIXIOSFODNN7EXAMPLE\nAWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

	assert.deepEqual(clean(text, raw, "secret"), ["AKIXIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"]);
});

test("a word in front of a long token is not part of the secret, but a passphrase keeps its words", () => {
	const sha = "9f8e7d6c5b4a39281706f5e4d3c2b1a098765432";

	assert.deepEqual(clean(`about commit ${sha}.`, `commit ${sha}`, "secret"), [sha]);
	assert.deepEqual(clean("the phrase is correct horse battery staple", "correct horse battery staple", "secret"), [
		"correct horse battery staple",
	]);
});

test("a connection string stays whole", () => {
	const url = "postgres://admin:s3cr3tP4ss@db.internal:5432/prod";

	assert.deepEqual(clean(`DATABASE_URL=${url}`, `DATABASE_URL=${url}`, "secret"), [url]);
	assert.deepEqual(clean(`url: ${url}`, url, "secret"), [url]);
});

test("spans that cannot be what their label says are dropped", () => {
	assert.deepEqual(clean("dana@Danas-MacBook-Pro ~ %", "dana@Danas-MacBook-Pro", "private_email"), []);
	assert.deepEqual(clean("order 88231 shipped", "88231", "account_number"), []);
	assert.deepEqual(clean("call 555", "555", "private_phone"), []);
	assert.deepEqual(clean("x = ab;", "ab", "secret"), []);
	assert.deepEqual(clean("a ' b", "'", "private_person"), []);
	assert.deepEqual(clean("IBAN DE89 3704 0044 0532 0130 00", "DE89 3704 0044 0532 0130 00", "account_number"), ["DE89 3704 0044 0532 0130 00"]);
	assert.deepEqual(clean("call +1 (415) 555-0132 now", "+1 (415) 555-0132", "private_phone"), ["+1 (415) 555-0132"]);
});

test("spans that do not fit the text are ignored, and the rest are returned in order", () => {
	const text = "Lena and Omar";
	const found = cleanSpans(
		text,
		[
			{ start: 9, end: 13, label: "private_person" },
			{ start: -1, end: 3, label: "private_person" },
			{ start: 5, end: 99, label: "private_person" },
			{ start: 4, end: 4, label: "private_person" },
			{ start: 0, end: 4, label: "private_person" },
		],
		labels,
	);

	assert.deepEqual(found.map((span) => text.slice(span.start, span.end)), ["Lena", "Omar"]);
});
