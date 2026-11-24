import assert from "node:assert/strict";
import { test } from "node:test";

import { findPatterns } from "../src/patterns.ts";
import { Screen } from "../src/screen.ts";
import type { Detect, Item } from "../src/screen.ts";
import type { Span } from "../src/spans.ts";

type Rule = (text: string) => boolean;

function detector(words: Record<string, string | [label: string, when: Rule]> = {}) {
	const calls: string[][] = [];
	const detect: Detect = async (texts) => {
		calls.push([...texts]);
		return texts.map((text) => {
			const spans: Span[] = [];
			for (const [word, rule] of Object.entries(words)) {
				const [label, when] = typeof rule === "string" ? [rule, () => true] : rule;
				if (!when(text)) continue;
				for (let at = text.indexOf(word); at !== -1; at = text.indexOf(word, at + 1)) {
					spans.push({ start: at, end: at + word.length, label });
				}
			}
			return spans;
		});
	};
	return { detect, calls };
}

function screenWith(words: Parameters<typeof detector>[0] = {}, labels?: string[]) {
	const { detect, calls } = detector(words);
	return { screen: new Screen(detect, labels ? { labels } : {}), calls };
}

const tool = (text: string): Item => ({ text, origin: "input" });
const user = tool;
const system = (text: string): Item => ({ text, origin: "system" });
const provider = (text: string): Item => ({ text, origin: "provider" });

// A made-up key, in pieces so that no scanner takes this file for a leak.
const GITHUB = ["ghp_", "x7KqLm29ZpQrStUvWxYz", "0123456789AbCdEf"].join("");

test("what the detector finds is replaced by numbered placeholders, and nothing else changes", async () => {
	const { screen } = screenWith({ "hunter22x!": "secret", "Lena Fischer": "private_person" });

	const result = await screen.mask([tool("Lena Fischer set the password to hunter22x! today")]);

	assert.deepEqual(result.texts, ["<PERSON_1> set the password to <SECRET_1> today"]);
	assert.equal(result.problem, undefined);
	assert.equal(screen.vault.lookup("<SECRET_1>")?.value, "hunter22x!");
});

test("a value found once is replaced wherever it appears later, also where the detector misses it", async () => {
	const { screen } = screenWith({ Lena: ["private_person", (text) => text.startsWith("From:")] });

	const result = await screen.mask([tool("From: Lena"), tool("ask Lena about it"), user("Lena said yes")]);

	assert.deepEqual(result.texts, ["From: <PERSON_1>", "ask <PERSON_1> about it", "<PERSON_1> said yes"]);
});

test("a value found in a later text is replaced in the earlier texts of the same request", async () => {
	const { screen } = screenWith({ Lena: ["private_person", (text) => text.startsWith("From:")] });

	const result = await screen.mask([system("Maintainer: Lena"), tool("ask Lena about it"), tool("From: Lena")]);

	assert.deepEqual(result.texts, ["Maintainer: <PERSON_1>", "ask <PERSON_1> about it", "From: <PERSON_1>"]);
});

test("a known value is replaced where it stands alone, a long key anywhere", async () => {
	const { screen } = screenWith();
	screen.vault.allocate("PERSON", "Lena");
	screen.vault.allocate("SECRET", "a1B2c3D4e5F6g7H8");
	screen.vault.allocate("PHONE", "5550199");

	const result = await screen.mask([tool("Lena, Lenard, Helena and (Lena) met; id=xa1B2c3D4e5F6g7H8y; 5550199 not 15550199")]);

	assert.deepEqual(result.texts, ["<PERSON_1>, Lenard, Helena and (<PERSON_1>) met; id=x<SECRET_1>y; <PHONE_1> not 15550199"]);
});

test("a value shorter than three characters is replaced only where it was detected", async () => {
	const { screen } = screenWith({ Bo: ["private_person", (text) => text.startsWith("name")] });

	const result = await screen.mask([tool("name: Bo"), tool("Bo went to Bolivia")]);

	assert.deepEqual(result.texts, ["name: <PERSON_1>", "Bo went to Bolivia"]);
});

test("a key with a well-known format is replaced without the detector's help", async () => {
	const { screen } = screenWith();

	const result = await screen.mask([tool(`export GITHUB_TOKEN=${GITHUB}`)]);

	assert.deepEqual(result.texts, ["export GITHUB_TOKEN=<SECRET_1>"]);
});

test("only the chosen labels are screened", async () => {
	const words = { "https://example.com/u/42": "private_url", "2026-11-23": "private_date" };
	const text = "see https://example.com/u/42 on 2026-11-23";

	const byDefault = await screenWith(words).screen.mask([tool(text)]);
	const everything = await screenWith(words, ["private_url", "private_date"]).screen.mask([tool(text)]);

	assert.deepEqual(byDefault.texts, [text]);
	assert.deepEqual(everything.texts, ["see <URL_1> on <DATE_1>"]);
});

test("the longer of two overlapping values wins", async () => {
	const { screen } = screenWith();
	screen.vault.allocate("PERSON", "Fischer");
	screen.vault.allocate("PERSON", "Lena Fischer");

	const result = await screen.mask([tool("Lena Fischer and Fischer")]);

	assert.deepEqual(result.texts, ["<PERSON_2> and <PERSON_1>"]);
});

test("two detections that overlap are one value, and only that one gets a placeholder", async () => {
	const { screen } = screenWith({ "Lena Fischer": "private_person", "Fischer-Mayer": "private_person" });

	const result = await screen.mask([tool("ask Lena Fischer-Mayer today")]);

	assert.deepEqual(result.texts, ["ask <PERSON_1> today"]);
	assert.deepEqual(screen.vault.entries.map((entry) => entry.value), ["Lena Fischer-Mayer"]);
});

test("a detection that sticks out of a replacement becomes one value with it, so no fragment is left", async () => {
	const { screen } = screenWith({ "Lena Fischer-Braun": "private_person" });
	screen.vault.allocate("PERSON", "Fischer-Braun of Hamburg");
	const text = "Dr. Lena Fischer-Braun of Hamburg called";

	const [masked] = (await screen.mask([tool(text)])).texts;

	assert.equal(masked, "Dr. <PERSON_3> called");
	assert.equal(screen.unmask(masked!), text);
});

test("an empty text needs no screening, and equal texts are checked once", async () => {
	const { screen, calls } = screenWith({ Lena: "private_person" });

	const result = await screen.mask([tool(""), tool("hi Lena"), tool("hi Lena")]);

	assert.deepEqual(result.texts, ["", "hi <PERSON_1>", "hi <PERSON_1>"]);
	assert.deepEqual(calls, [["hi Lena"]]);
});

test("text that looks like a placeholder is passed as a literal and comes back as itself, whoever it is from", async () => {
	const { screen } = screenWith();
	screen.vault.allocate("SECRET", "hunter22x!");
	const read = "the README says <SECRET_1> stands for a key";
	const pasted = "my test has expect(out).toBe('<SECRET_1>') and never <SECRET_9>";

	const masked = (await screen.mask([tool(read), user(pasted)])).texts;

	assert.deepEqual(masked, ["the README says <LITERAL_1> stands for a key", "my test has expect(out).toBe('<LITERAL_1>') and never <LITERAL_2>"]);
	assert.deepEqual(masked.map((text) => screen.unmask(text!)), [read, pasted], "so that a file is never given a value where it had this text");
});

test("a detection around a placeholder the model was given does not replace the reference", async () => {
	const { screen } = screenWith({ "password <SECRET_1>": "secret" });
	screen.vault.allocate("SECRET", "hunter22x!");
	screen.remember("the password hunter22x! is wrong", "the password <SECRET_1> is wrong");

	const result = await screen.mask([{ text: "the password hunter22x! is wrong", origin: "foreign" }]);

	assert.deepEqual(result.texts, ["the password <SECRET_1> is wrong"]);
	assert.equal(screen.vault.entries.length, 1);
});

test("the system prompt gets known values and key formats, but is not sent to the detector", async () => {
	const { screen, calls } = screenWith({ Omar: "private_person" });
	screen.vault.allocate("PERSON", "Lena");

	const result = await screen.mask([system(`Maintainer: Lena. Reviewer: Omar. Token ${GITHUB}. Placeholders look like <SECRET_1>.`)]);

	assert.deepEqual(result.texts, ["Maintainer: <PERSON_1>. Reviewer: Omar. Token <SECRET_1>. Placeholders look like <LITERAL_1>."]);
	assert.deepEqual(calls, []);
});

test("text the provider wrote goes back exactly as it was sent", async () => {
	const { screen, calls } = screenWith();
	screen.vault.allocate("SECRET", "it's here");
	const sent = 'echo <SECRET_1> > "$OUT"';
	const run = `echo 'it'\\''s here' > "$OUT"`;

	screen.remember(run, sent);

	assert.deepEqual((await screen.mask([provider(run)])).texts, [sent]);
	assert.deepEqual(calls, []);
});

test("what was remembered can be saved and loaded into a later session", async () => {
	const first = screenWith().screen;
	const saved: [string, string | null][] = [];
	first.onRemember = (key, masked) => saved.push([key, masked]);
	first.remember("the real text", "the <SECRET_1> text");

	const later = screenWith().screen;
	for (const [key, masked] of saved) later.restore(key, masked);

	assert.equal(saved.length, 1);
	assert.doesNotMatch(saved[0]![0], /real/);
	assert.deepEqual((await later.mask([provider("the real text")])).texts, ["the <SECRET_1> text"]);
});

test("a placeholder the provider wrote and that was left in place stays a reference to its value", async () => {
	const { screen } = screenWith();
	const saved: [string, string | null][] = [];
	screen.onRemember = (key, masked) => saved.push([key, masked]);
	screen.vault.allocate("SECRET", "hunter22x!");
	const kept = "run it with <SECRET_1> and <SECRET_7>";

	screen.remember(kept, kept);
	screen.remember("no placeholder in here", "no placeholder in here");

	assert.deepEqual((await screen.mask([provider(kept)])).texts, [kept]);
	assert.deepEqual(saved.map(([, masked]) => masked), [null], "text without placeholders needs no remembering");
});

test("a value learned later is replaced in remembered text too, around its placeholders", async () => {
	const { screen } = screenWith();
	screen.vault.allocate("SECRET", "hunter22x!");
	screen.remember("Omar should use hunter22x!", "Omar should use <SECRET_1>");
	screen.remember("tell Omar: <SECRET_1>", "tell Omar: <SECRET_1>");
	const items = [provider("Omar should use hunter22x!"), provider("tell Omar: <SECRET_1>")];

	const before = await screen.mask(items);
	screen.vault.allocate("PERSON", "Omar");
	const after = await screen.mask(items);

	assert.deepEqual(before.texts, ["Omar should use <SECRET_1>", "tell Omar: <SECRET_1>"]);
	assert.deepEqual(after.texts, ["<PERSON_1> should use <SECRET_1>", "tell <PERSON_1>: <SECRET_1>"]);
});

test("provider text nobody remembered gets its known values replaced, and what looks like a placeholder is a literal", async () => {
	const { screen, calls } = screenWith({ Omar: "private_person" });
	screen.vault.allocate("SECRET", "hunter22x!");
	screen.vault.allocate("LITERAL", "<SECRET_3>");
	const text = "Omar set it to hunter22x! and wrote <SECRET_3> and <SECRET_1> in the README";

	const [masked] = (await screen.mask([provider(text)])).texts;

	assert.equal(masked, "Omar set it to <SECRET_1> and wrote <LITERAL_1> and <LITERAL_2> in the README");
	assert.equal(screen.unmask(masked!), text);
	assert.deepEqual(calls, []);
});

test("placeholders in a reply become their values, once", () => {
	const { screen } = screenWith();
	screen.vault.allocate("SECRET", "hunter22x!");
	screen.vault.allocate("LITERAL", "<SECRET_1>");

	assert.equal(screen.unmask("set <SECRET_1>, see <SECRET_9> and <LITERAL_1>"), "set hunter22x!, see <SECRET_9> and <SECRET_1>");
	assert.equal(screen.unmask("nothing here"), "nothing here");
});

test("a text is sent to the detector once, however often it is screened", async () => {
	const { screen, calls } = screenWith({ Lena: "private_person" });

	await screen.mask([tool("hi Lena"), user("who is Lena?")]);
	const again = await screen.mask([tool("hi Lena"), user("who is Lena?"), tool("new text")]);

	assert.deepEqual(again.texts, ["hi <PERSON_1>", "who is <PERSON_1>?", "new text"]);
	assert.deepEqual(calls, [["hi Lena", "who is Lena?"], ["new text"]]);
});

test("a value learned later is replaced in earlier text without asking the detector again", async () => {
	const { screen, calls } = screenWith({ Omar: ["private_person", (text) => text.startsWith("name:")] });

	const before = await screen.mask([tool("Omar wrote this")]);
	const after = await screen.mask([tool("Omar wrote this"), tool("name: Omar")]);

	assert.deepEqual(before.texts, ["Omar wrote this"]);
	assert.deepEqual(after.texts, ["<PERSON_1> wrote this", "name: <PERSON_1>"]);
	assert.deepEqual(calls, [["Omar wrote this"], ["name: Omar"]]);
});

test("a text the detector fails on is withheld, the others are not", async () => {
	const { screen } = screenWith();
	const failing = new Screen(async (texts) => texts.map((text) => (text.includes("poison") ? new Error("detection failed") : [])));

	const result = await failing.mask([tool("fine"), tool("poison"), system("prompt")]);

	assert.deepEqual(result.texts, ["fine", undefined, "prompt"]);
	assert.deepEqual({ down: result.problem?.down, count: result.problem?.count }, { down: false, count: 1 });
	assert.equal((await screen.mask([])).problem, undefined);
});

test("a text the patterns cannot get through is withheld, the others are not", async (t) => {
	const hostile = 'secret: "' + String.raw`\"`.repeat(6 * 1024 * 1024);
	let thrown = false;
	try {
		findPatterns(hostile);
	} catch {
		thrown = true;
	}
	if (!thrown) return t.skip("this engine gets through the text, so there is nothing to withhold");
	const { screen } = screenWith({ Lena: "private_person" });

	const result = await screen.mask([tool("ask Lena"), tool(hostile), system(hostile), user("Lena again")]);

	assert.deepEqual(result.texts, ["ask <PERSON_1>", undefined, undefined, "<PERSON_1> again"]);
	assert.equal(result.problem?.down, false);
	assert.equal(result.problem?.count, 2);
	assert.match(result.problem!.error.message, /could not be searched/);
});

test("a detector that answers for a different number of texts is not believed for any of them", async () => {
	const screen = new Screen(async () => [[]]);

	const result = await screen.mask([tool("one"), tool("two")]);

	assert.deepEqual(result.texts, [undefined, undefined]);
	assert.deepEqual({ down: result.problem?.down, count: result.problem?.count }, { down: true, count: 2 });
});

test("when the detector cannot be reached, only text that was already screened is released", async () => {
	let up = true;
	const screen = new Screen(async (texts) => {
		if (!up) throw new Error("connect ENOENT");
		return texts.map(() => []);
	});
	screen.vault.allocate("SECRET", "hunter22x!");
	await screen.mask([tool("old hunter22x!")]);
	up = false;

	const result = await screen.mask([tool("old hunter22x!"), tool("new hunter22x!"), user("hello"), provider("reply")]);

	assert.deepEqual(result.texts, ["old <SECRET_1>", undefined, undefined, "reply"]);
	assert.deepEqual({ down: result.problem?.down, count: result.problem?.count }, { down: true, count: 2 });
	assert.match(result.problem!.error.message, /ENOENT/);
});

test("text that has left the conversation is forgotten", async () => {
	const { screen, calls } = screenWith();

	await screen.mask([tool("first")]);
	screen.endPass();
	await screen.mask([tool("first")]);
	screen.endPass();
	screen.endPass();
	screen.endPass();
	await screen.mask([tool("first")]);

	assert.deepEqual(calls, [["first"], ["first"]]);
});

test("a value from an environment file is replaced where it occurs, and only then gets a placeholder", async () => {
	const { screen } = screenWith();
	screen.seed(["correct-horse-battery-staple", "another-seed-value-1"]);

	const none = await screen.mask([tool("nothing to see")]);
	const some = await screen.mask([tool("DB_PASSWORD is correct-horse-battery-staple")]);

	assert.deepEqual(none.texts, ["nothing to see"]);
	assert.deepEqual(some.texts, ["DB_PASSWORD is <SECRET_1>"]);
	assert.deepEqual(screen.vault.entries.map((entry) => entry.value), ["correct-horse-battery-staple"]);
});

test("a revealed value is sent as it is, wherever it or its placeholder appears, until it is concealed again", async () => {
	const { screen } = screenWith({ "hunter22x!": "secret" });
	screen.vault.allocate("SECRET", "hunter22x!");
	screen.remember("I used hunter22x!", "I used <SECRET_1>");
	const items = [tool("pw hunter22x!"), user("why hunter22x!?"), provider("I used hunter22x!")];

	const masked = await screen.mask(items);
	assert.equal(screen.reveal("<SECRET_1>", true), true);
	const revealed = await screen.mask(items);
	screen.reveal("<SECRET_1>", false);
	const concealed = await screen.mask(items);

	assert.deepEqual(masked.texts, ["pw <SECRET_1>", "why <SECRET_1>?", "I used <SECRET_1>"]);
	assert.deepEqual(revealed.texts, ["pw hunter22x!", "why hunter22x!?", "I used hunter22x!"]);
	assert.deepEqual(concealed.texts, masked.texts);
	assert.equal(screen.reveal("<SECRET_9>", true), false);
});

test("the last look replaces known values in whatever is about to be sent, and counts them", () => {
	const { screen } = screenWith();
	screen.vault.allocate("SECRET", "hunter22x!");
	screen.vault.allocate("PERSON", "Lena");
	screen.vault.allocate("LITERAL", "<SECRET_1>");

	assert.deepEqual(screen.sweep("Lena used hunter22x! twice: hunter22x!"), { text: "<PERSON_1> used <SECRET_1> twice: <SECRET_1>", hits: 3 });
	assert.deepEqual(screen.sweep("already <SECRET_1> and <LITERAL_1>, Lenard"), { text: "already <SECRET_1> and <LITERAL_1>, Lenard", hits: 0 });
});

test("the last look also finds a value in the form it has inside a JSON string", () => {
	const { screen } = screenWith();
	screen.vault.allocate("SECRET", 'pa"ss\\word\n2');
	screen.seed(["seeded-value-77"]);

	const result = screen.sweep('{"command":"echo pa\\"ss\\\\word\\n2 seeded-value-77"}');

	assert.deepEqual(result, { text: '{"command":"echo <SECRET_1> <SECRET_2>"}', hits: 2 });
});

test("the last look leaves a revealed value alone", () => {
	const { screen } = screenWith();
	screen.vault.allocate("SECRET", "hunter22x!");
	screen.reveal("<SECRET_1>", true);

	assert.deepEqual(screen.sweep("pw hunter22x!"), { text: "pw hunter22x!", hits: 0 });
});

test("text another model wrote is new to this provider, and is screened like a tool's", async () => {
	const { screen, calls } = screenWith({ "lena@example.com": "private_email" });
	screen.vault.allocate("SECRET", "hunter22x!");

	const { texts } = await screen.mask([{ text: "The admin is lena@example.com; see <SECRET_1>.", origin: "foreign" }]);

	assert.deepEqual(texts, ["The admin is <EMAIL_1>; see <LITERAL_1>."]);
	assert.deepEqual(calls, [["The admin is lena@example.com; see <SECRET_1>."]]);
});

test("text another model wrote keeps the placeholders it was given, and is screened around them", async () => {
	const { screen, calls } = screenWith({ "lena@example.com": "private_email" });
	screen.vault.allocate("SECRET", "hunter22x!");
	screen.remember("Use hunter22x! and tell lena@example.com", "Use <SECRET_1> and tell lena@example.com");
	const real = "Use hunter22x! and tell lena@example.com";

	const same = await screen.mask([{ text: real, origin: "provider" }]);
	const other = await screen.mask([{ text: real, origin: "foreign" }]);
	const again = await screen.mask([{ text: real, origin: "foreign" }]);

	assert.deepEqual(same.texts, ["Use <SECRET_1> and tell lena@example.com"], "its own provider has seen all of it");
	assert.deepEqual(other.texts, ["Use <SECRET_1> and tell <EMAIL_1>"]);
	assert.deepEqual(calls, [["Use <SECRET_1> and tell lena@example.com"]], "the detector sees the text as it was written, once");
	assert.deepEqual(again.texts, other.texts);
});

test("text another model wrote is withheld when it cannot be checked", async () => {
	const screen = new Screen(async () => {
		throw new Error("connect ENOENT");
	});

	const { texts, problem } = await screen.mask([{ text: "The admin is lena@example.com", origin: "foreign" }, provider("its own words")]);

	assert.deepEqual(texts, [undefined, "its own words"]);
	assert.equal(problem?.down, true);
});
