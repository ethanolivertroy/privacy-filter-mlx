import assert from "node:assert/strict";
import { test } from "node:test";

import { findPatterns, plausibleSecret } from "../src/patterns.ts";

// Made-up keys, assembled from pieces so that no secret scanner mistakes this file for a leak.
const join = (...parts: string[]) => parts.join("");
const GITHUB = join("ghp_", "x7KqLm29ZpQrStUvWxYz", "0123456789AbCdEf");
const AWS_ID = join("AKIA", "IOSFODNN7EXAMPLE");
const OPENAI = join("sk-", "proj-", "Ab3dEf6hIj9kLmN2pQrS5tUv");
const ANTHROPIC = join("sk-", "ant-", "api03-", "Zx9Yw8Vu7Ts6Rq5Po4Nm3Lk2");
const STRIPE = join("sk_", "live_", "4eC39HqLyjWDarjtT1zdp7dc");
const SLACK = join("xox", "b-", "123456789012-abcdefghijkl");
const GOOGLE = join("AIza", "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q");
const JWT = join("eyJ", "hbGciOiJIUzI1NiJ9", ".", "eyJ", "zdWIiOiIxMjM0NTY3ODkwIn0", ".", "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV");
const PEM = join("-----BEGIN RSA ", "PRIVATE KEY-----", "\nMIIEow\nIBAAKC\n", "-----END RSA ", "PRIVATE KEY-----");

const found = (text: string) => findPatterns(text).map((match) => text.slice(match.start, match.end));

test("well-known key formats are found wherever they stand", () => {
	for (const key of [GITHUB, AWS_ID, OPENAI, ANTHROPIC, STRIPE, SLACK, GOOGLE, JWT]) {
		assert.deepEqual(found(`before ${key} after`), [key], key);
		assert.deepEqual(found(`x="${key}";`), [key], key);
	}
});

test("a private key block is one value, line breaks included", () => {
	assert.deepEqual(found(`cat > key.pem <<EOF\n${PEM}\nEOF`), [PEM]);
});

test("a PGP private key block is found, too", () => {
	const PGP = join("-----BEGIN PGP ", "PRIVATE KEY BLOCK-----", "\n\nlQOYBF\n=AbCd\n", "-----END PGP ", "PRIVATE KEY BLOCK-----");
	assert.deepEqual(found(`key:\n${PGP}\nend`), [PGP]);
	assert.deepEqual(found(`two: ${PEM} and ${PGP}`), [PEM, PGP]);
	assert.deepEqual(found("-----BEGIN PGP PUBLIC KEY BLOCK-----\nmQENBF\n-----END PGP PUBLIC KEY BLOCK-----"), []);
});

test("everything found is a secret, in order, without overlaps", () => {
	const text = `A=${AWS_ID} and ${GITHUB}\n${PEM}`;
	const matches = findPatterns(text);

	assert.deepEqual(matches.map((match) => match.kind), ["SECRET", "SECRET", "SECRET"]);
	assert.deepEqual(matches.map((match) => text.slice(match.start, match.end)), [AWS_ID, GITHUB, PEM]);
});

test("the password in a URL is found, the rest of the URL is not", () => {
	assert.deepEqual(found("DATABASE_URL=postgres://admin:s3cr3tP4ss@db.internal:5432/prod"), ["s3cr3tP4ss"]);
	assert.deepEqual(found("https://example.com/a:b@c"), []);
	assert.deepEqual(found("redis://user:${REDIS_PASSWORD}@cache:6379"), []);
});

test("a bearer token is found without the word in front of it", () => {
	assert.deepEqual(found("Authorization: Bearer abcdefghij0123456789ABCDEFGHIJ=="), ["abcdefghij0123456789ABCDEFGHIJ=="]);
	assert.deepEqual(found("Bearer token required"), []);
});

test("a value assigned to a sensitive name is found", () => {
	assert.deepEqual(found("SESSION_SECRET=correct-horse-battery-staple-2291"), ["correct-horse-battery-staple-2291"]);
	assert.deepEqual(found('export DB_PASSWORD="Tr0ub4dor&3xyz"'), ["Tr0ub4dor&3xyz"]);
	assert.deepEqual(found(`"api_key": "a1B2c3D4e5F6g7H8"`), ["a1B2c3D4e5F6g7H8"]);
	assert.deepEqual(found("password: Sup3rS3cret!"), ["Sup3rS3cret!"]);
});

test("inside a string that is itself quoted, the escaping is not part of the value", () => {
	assert.deepEqual(found(String.raw`bash(command="echo \"token: ${GITHUB}\" >> deploy.log")`), [GITHUB]);
	assert.deepEqual(found(String.raw`run("export DB_PASSWORD=hunter22x9\"; echo done")`), ["hunter22x9"]);
	assert.deepEqual(found(String.raw`{"cmd": "login --password=hunter22x9\\"}`), ["hunter22x9"]);
	assert.deepEqual(found(String.raw`PASSWORD=ab\cd\ef2345 next`), [String.raw`ab\cd\ef2345`], "a backslash inside a value stays");
});

test("a quoted value runs to the quote that ends it, past the escaped ones inside", () => {
	assert.deepEqual(found(String.raw`{"password": "ab\"cd9876xyz", "user": "lena"}`), [String.raw`ab\"cd9876xyz`]);
	assert.deepEqual(found(String.raw`password = 'it\'s-4-secret'`), [String.raw`it\'s-4-secret`]);
	assert.deepEqual(found(String.raw`"password": "abc12345\\", "user": "x"`), [String.raw`abc12345\\`], "an escaped backslash does not escape the quote after it");
});

test("no text makes the patterns slow: the time grows with its length, not with the square of it", () => {
	const size = 300_000;
	const fill = (unit: string, tail = "") => unit.repeat(Math.ceil(size / unit.length)) + tail;
	const crafted: Record<string, string> = {
		"a key prefix over and over": fill("sk-"),
		"words joined by dots": fill("a."),
		"words joined by dashes": fill("xoxb-"),
		"schemes without a colon": fill("ab+", "://"),
		"sensitive names in a row": fill("token"),
		"sensitive names joined": fill("password_"),
		"sensitive names with dots": fill("secret."),
		"token headers joined by dashes": fill("eyJ-"),
		"token headers": fill("eyJabcdefgh."),
		"key blocks that never end": fill("-----BEGIN PRIVATE KEY-----\n"),
		"key block types": "-----BEGIN " + fill("A "),
		"a URL without a host": "x://u:" + fill("p"),
		"URL starts": fill("a://b:"),
		"an open quote": 'password: "' + fill("a1"),
		"escaped quotes": 'secret: "' + fill(String.raw`\"`),
		"backslashes": "secret=" + fill("\\"),
		"quotes": fill('secret="'),
		"assignments": fill("password="),
		"bearer": fill("Bearer "),
	};
	for (const [name, text] of Object.entries(crafted)) {
		const start = performance.now();
		findPatterns(text);
		const ms = performance.now() - start;
		assert.ok(ms < 1500, `${name}: ${ms.toFixed(0)} ms for ${text.length} characters`);
	}
});

test("a run of many megabytes is read to its end: the pattern engine does not give up on it", () => {
	const size = 12 * 1024 * 1024;
	const runs: Record<string, string> = {
		"after a key prefix": "sk-" + "a".repeat(size),
		"after a token prefix": "xoxb-" + "a".repeat(size),
		"an assigned value": "secret=" + "a1".repeat(size / 2),
		"a quoted value": 'secret: "' + "a1".repeat(size / 2) + '"',
		"a bearer token": "Bearer " + "a".repeat(size),
		"a token header": "eyJ" + "a".repeat(size),
		"the words of a key block": "-----BEGIN " + "A ".repeat(size / 2),
		"a URL password": "x://u:" + "p".repeat(size) + "@h",
	};
	for (const [name, text] of Object.entries(runs)) {
		assert.doesNotThrow(() => findPatterns(text), name);
	}
});

test("ordinary code and configuration around sensitive names is left alone", () => {
	for (const line of [
		"password = getpass.getpass()",
		'token = response.json()["access_token"]',
		"def login(password: str, token: Optional[str] = None):",
		"max_tokens=4096",
		'TOKEN_URL = "https://example.com/oauth/token"',
		"SECRET_KEY=changeme",
		"API_KEY=${API_KEY}",
		"API_KEY=<SECRET_1>",
		"PASSWORD=$DB_PASSWORD",
		'secret_name = "my-app/prod/db"',
		"private_key_path: /Users/someone/.ssh/id_ed25519",
		"use_api_key: true",
		"tokenizer = Tokenizer('o200k_base')",
		"git commit 9f8e7d6c5b4a39281706f5e4d3c2b1a098765432",
		"sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=",
		"sk-toggle-switch-container-wrapper-element",
		"sk-CircleFadeAnimationContainerElement",
		"version 1.2.3 at 10.0.0.1 on 2026-11-23",
	]) {
		assert.deepEqual(found(line), [], line);
	}
});

test("a value is a plausible secret only if it could not be ordinary text", () => {
	for (const value of ["correct-horse-battery-staple-2291", "Tr0ub4dor&3xyz", "hunter22", "postgres://u:pw123456@h/db"]) {
		assert.equal(plausibleSecret(value), true, value);
	}
	for (const value of [
		"short1", "changeme", "password", "12345678", "true", "localhost:5432", "https://example.com/path1",
		"/usr/local/bin/tool2", "./relative/path2", "~/.ssh/id_rsa2", "${TOKEN_VALUE1}", "$TOKEN_VALUE1", "<SECRET_12>",
		"getpass.getpass()", "os.environ['A1']", "my-app/prod/db", "",
	]) {
		assert.equal(plausibleSecret(value), false, value);
	}
});
