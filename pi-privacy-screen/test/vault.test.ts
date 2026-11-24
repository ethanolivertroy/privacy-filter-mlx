import assert from "node:assert/strict";
import { test } from "node:test";

import { Vault, tokenPattern } from "../src/vault.ts";

test("each value gets one numbered placeholder of its kind", () => {
	const vault = new Vault();

	assert.equal(vault.allocate("SECRET", "ghp_abc").token, "<SECRET_1>");
	assert.equal(vault.allocate("EMAIL", "a@example.com").token, "<EMAIL_1>");
	assert.equal(vault.allocate("SECRET", "hunter22").token, "<SECRET_2>");
	assert.equal(vault.allocate("SECRET", "ghp_abc").token, "<SECRET_1>");
});

test("a value keeps its first placeholder whatever it is called later", () => {
	const vault = new Vault();
	vault.allocate("ACCOUNT", "12345678");

	assert.equal(vault.allocate("SECRET", "12345678").token, "<ACCOUNT_1>");
	assert.equal(vault.entries.length, 1);
});

test("placeholders and values look each other up", () => {
	const vault = new Vault();
	const entry = vault.allocate("PERSON", "Lena Fischer");

	assert.equal(vault.lookup("<PERSON_1>"), entry);
	assert.equal(vault.find("Lena Fischer"), entry);
	assert.equal(vault.lookup("<PERSON_2>"), undefined);
	assert.equal(vault.find("lena fischer"), undefined);
});

test("every new placeholder is announced once, for saving", () => {
	const saved: string[] = [];
	const vault = new Vault();
	vault.onAllocate = (entry) => saved.push(`${entry.token}=${entry.value}`);

	vault.allocate("SECRET", "one");
	vault.allocate("SECRET", "one");
	vault.allocate("SECRET", "two");

	assert.deepEqual(saved, ["<SECRET_1>=one", "<SECRET_2>=two"]);
});

test("restoring a saved session keeps its numbers and continues after the highest", () => {
	const saved: string[] = [];
	const vault = new Vault();
	vault.onAllocate = (entry) => saved.push(entry.token);

	vault.restore("<SECRET_7>", "old-seven");
	vault.restore("<SECRET_2>", "old-two");
	vault.restore("<EMAIL_1>", "a@example.com");

	assert.equal(vault.lookup("<SECRET_7>")?.value, "old-seven");
	assert.equal(vault.allocate("SECRET", "new").token, "<SECRET_8>");
	assert.equal(vault.allocate("EMAIL", "b@example.com").token, "<EMAIL_2>");
	assert.deepEqual(saved, ["<SECRET_8>", "<EMAIL_2>"], "restored entries are not saved again");
});

test("a saved entry that is not a placeholder, or contradicts one already loaded, is refused", () => {
	const vault = new Vault();
	vault.restore("<SECRET_1>", "one");

	assert.equal(vault.restore("<SECRET_1>", "one"), true, "the same pair again is fine");
	assert.equal(vault.restore("<SECRET_1>", "other"), false);
	assert.equal(vault.restore("<SECRET_2>", "one"), false);
	assert.equal(vault.restore("<BOGUS_1>", "x"), false);
	assert.equal(vault.restore("SECRET_3", "x"), false);
	assert.equal(vault.restore("<SECRET_3>", ""), false);
	assert.equal(vault.entries.length, 1);
});

test("a revealed value keeps its placeholder but is marked", () => {
	const vault = new Vault();
	vault.allocate("SECRET", "one");

	assert.equal(vault.setRevealed("<SECRET_1>", true), true);
	assert.equal(vault.lookup("<SECRET_1>")?.revealed, true);
	assert.equal(vault.setRevealed("<SECRET_9>", true), false);
});

test("the placeholder pattern finds whole placeholders only", () => {
	const text = "a <SECRET_1> b <EMAIL_23> c <SECRET_0> <secret_1> <SECRET_1 > <LITERAL_4> <OTHER_1> <SECRET_1234567>";

	assert.deepEqual(text.match(tokenPattern()), ["<SECRET_1>", "<EMAIL_23>", "<LITERAL_4>"]);
});
