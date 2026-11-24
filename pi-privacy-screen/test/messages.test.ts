import assert from "node:assert/strict";
import { test } from "node:test";

import { WITHHELD, sweepPayload, transcript, unmaskAssistant, unmaskInputInPlace } from "../src/messages.ts";
import type { Message } from "../src/messages.ts";
import { Screen } from "../src/screen.ts";
import type { Span } from "../src/spans.ts";

const EMAIL = "lena@example.com";

function emailScreen(down = false): Screen {
	return new Screen(async (texts) => {
		if (down) throw new Error("connect ENOENT");
		return texts.map((text) => {
			const spans: Span[] = [];
			for (let at = text.indexOf(EMAIL); at !== -1; at = text.indexOf(EMAIL, at + 1)) {
				spans.push({ start: at, end: at + EMAIL.length, label: "private_email" });
			}
			return spans;
		});
	});
}

async function masked(screen: Screen, messages: Message[], target: Parameters<typeof transcript>[1] = {}) {
	const plan = transcript(messages, target);
	return plan.build((await screen.mask(plan.items)).texts);
}

const bashTool = { name: "bash", description: "Run a command", parameters: { type: "object" }, execute: () => "ran" };

function conversation(): Message[] {
	return [
		{ role: "system", content: `Contact ${EMAIL}.`, sections: { rules: `Ask ${EMAIL} first.`, gone: null }, toolsAdded: [bashTool], timestamp: 0 },
		{ role: "user", content: `mail ${EMAIL}`, timestamp: 1 },
		{
			role: "user",
			content: [
				{ type: "text", text: `again: ${EMAIL}` },
				{ type: "image", data: "aGk=", mimeType: "image/png" },
			],
			timestamp: 2,
		},
		{
			role: "assistant",
			content: [
				{ type: "thinking", thinking: `thinking of ${EMAIL}`, thinkingSignature: "sig" },
				{ type: "text", text: `I will mail ${EMAIL}`, textSignature: "t1" },
				{ type: "toolCall", id: "call-1", name: "bash", arguments: { command: `mail '${EMAIL}'`, nested: { list: [EMAIL, 3, null] } } },
			],
			provider: "acme",
			stopReason: "toolUse",
			timestamp: 3,
		},
		{
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "bash",
			content: [{ type: "text", text: `sent to ${EMAIL}` }],
			details: { raw: EMAIL },
			isError: false,
			timestamp: 4,
		},
		{ role: "bashExecution", command: `echo ${EMAIL}`, output: `${EMAIL}\n`, exitCode: 0, timestamp: 5 },
		{ role: "bashExecution", command: "cat private", output: EMAIL, excludeFromContext: true, timestamp: 6 },
		{ role: "custom", customType: "other-extension", content: `note for ${EMAIL}`, display: true, timestamp: 7 },
		{ role: "custom", customType: "other-extension", content: [{ type: "text", text: `more for ${EMAIL}` }], display: true, timestamp: 7 },
		{ role: "compactionSummary", summary: `The user mailed ${EMAIL}`, tokensBefore: 10, timestamp: 8 },
		{ role: "branchSummary", summary: `A branch about ${EMAIL}`, fromId: "x", timestamp: 9 },
		{ role: "somethingNew", text: EMAIL },
	];
}

test("every text that reaches the provider is masked, and nothing else is touched", async () => {
	const input = conversation();
	const before = structuredClone(input.map((message) => ({ ...message, toolsAdded: undefined })));

	const { messages, withheld } = await masked(emailScreen(), input);

	const e = "<EMAIL_1>";
	assert.equal(withheld, 0);
	assert.deepEqual(messages, [
		{ role: "system", content: `Contact ${e}.`, sections: { rules: `Ask ${e} first.`, gone: null }, toolsAdded: [bashTool], timestamp: 0 },
		{ role: "user", content: `mail ${e}`, timestamp: 1 },
		{
			role: "user",
			content: [
				{ type: "text", text: `again: ${e}` },
				{ type: "image", data: "aGk=", mimeType: "image/png" },
			],
			timestamp: 2,
		},
		{
			role: "assistant",
			content: [
				{ type: "thinking", thinking: `thinking of ${e}`, thinkingSignature: "sig" },
				{ type: "text", text: `I will mail ${e}`, textSignature: "t1" },
				{ type: "toolCall", id: "call-1", name: "bash", arguments: { command: `mail '${e}'`, nested: { list: [e, 3, null] } } },
			],
			provider: "acme",
			stopReason: "toolUse",
			timestamp: 3,
		},
		{ role: "toolResult", toolCallId: "call-1", toolName: "bash", content: [{ type: "text", text: `sent to ${e}` }], details: { raw: EMAIL }, isError: false, timestamp: 4 },
		{ role: "bashExecution", command: `echo ${e}`, output: `${e}\n`, exitCode: 0, timestamp: 5 },
		{ role: "bashExecution", command: "cat private", output: EMAIL, excludeFromContext: true, timestamp: 6 },
		{ role: "custom", customType: "other-extension", content: `note for ${e}`, display: true, timestamp: 7 },
		{ role: "custom", customType: "other-extension", content: [{ type: "text", text: `more for ${e}` }], display: true, timestamp: 7 },
		{ role: "compactionSummary", summary: `The user mailed ${e}`, tokensBefore: 10, timestamp: 8 },
		{ role: "branchSummary", summary: `A branch about ${e}`, fromId: "x", timestamp: 9 },
		{ role: "somethingNew", text: EMAIL },
	]);
	assert.equal((messages[0] as { toolsAdded: unknown[] }).toolsAdded[0], bashTool, "tools are passed on, not copied");
	assert.deepEqual(input.map((message) => ({ ...message, toolsAdded: undefined })), before, "the messages given are not changed");
});

test("text that could not be checked is withheld; what needs no detector still goes", async () => {
	const { messages, withheld } = await masked(emailScreen(true), conversation().slice(0, 5), { provider: "acme" });

	assert.equal(withheld, 3);
	assert.equal(messages[0]!.content, `Contact ${EMAIL}.`, "nothing has taught the screen this address");
	assert.equal(messages[1]!.content, WITHHELD);
	assert.deepEqual(messages[2]!.content, [{ type: "text", text: WITHHELD }, { type: "image", data: "aGk=", mimeType: "image/png" }]);
	assert.equal((messages[3]!.content as { text?: string }[])[1]!.text, `I will mail ${EMAIL}`);
	assert.deepEqual(messages[4]!.content, [{ type: "text", text: WITHHELD }]);
	assert.doesNotMatch(WITHHELD, /<[A-Z]+_\d+>/);
});

test("what another model wrote, and every summary, is checked before this provider sees it", async () => {
	const call = { type: "toolCall", id: "c", name: "bash", arguments: { command: `mail ${EMAIL}` } };
	const written = (provider: string | undefined): Message => ({ role: "assistant", provider, content: [{ type: "text", text: `Ask ${EMAIL}` }, call] });
	const summary: Message = { role: "compactionSummary", summary: `About ${EMAIL}`, tokensBefore: 1 };
	const messages = [written("acme"), written("llama"), written(undefined), summary];

	const checked = await masked(emailScreen(), messages, { provider: "acme" });
	const unchecked = await masked(emailScreen(true), messages, { provider: "acme" });

	const texts = (result: typeof checked) => result.messages.map((message) => (message.summary ?? (message.content as { text?: string }[])[0]!.text) as string);
	assert.deepEqual(texts(checked), ["Ask <EMAIL_1>", "Ask <EMAIL_1>", "Ask <EMAIL_1>", "About <EMAIL_1>"]);
	assert.deepEqual(texts(unchecked), [`Ask ${EMAIL}`, WITHHELD, WITHHELD, WITHHELD], "its own words go back to a provider; nobody else's do unchecked");
	assert.equal(unchecked.withheld, 5);
	assert.deepEqual((unchecked.messages[1]!.content as { arguments?: unknown }[])[1]!.arguments, { command: WITHHELD });
	const toNobody = await masked(emailScreen(true), [written(undefined)]);
	assert.equal(toNobody.withheld, 2, "a reply that names no provider is no provider's own, also when none is named as the target");
});

test("what another model thought is checked like what it said; a provider's own thinking is passed on untouched", async () => {
	const thought = (provider: string): Message => ({
		role: "assistant",
		provider,
		content: [{ type: "thinking", thinking: `the user means ${EMAIL}`, thinkingSignature: "sig" }, { type: "text", text: "ok" }],
	});
	const messages = [thought("acme"), thought("llama")];
	const thinking = (result: { messages: Message[] }, index: number) => (result.messages[index]!.content as { thinking?: string }[])[0]!;

	const checked = await masked(emailScreen(), messages, { provider: "acme" });
	const unchecked = await masked(emailScreen(true), messages, { provider: "acme" });

	assert.equal(thinking(checked, 0), (messages[0]!.content as unknown[])[0], "signed, and already known to the provider that signed it");
	assert.deepEqual(thinking(checked, 1), { type: "thinking", thinking: "the user means <EMAIL_1>", thinkingSignature: "sig" });
	assert.equal(thinking(unchecked, 1).thinking, WITHHELD);
	assert.deepEqual(messages[1], thought("llama"), "the stored message is not changed");
});

test("only the screen makes placeholders: text that looks like one is text, whether the user, a tool or an extension wrote it", async () => {
	const screen = emailScreen();
	screen.vault.allocate("EMAIL", EMAIL);

	const { messages } = await masked(screen, [
		{ role: "user", content: "this test expects <EMAIL_1>", timestamp: 1 },
		{ role: "toolResult", toolCallId: "c", toolName: "read", content: [{ type: "text", text: "docs: write <EMAIL_1> for the address" }], timestamp: 2 },
		{ role: "bashExecution", command: "grep '<EMAIL_1>' notes", output: "notes:3: <EMAIL_1> is how it is written", exitCode: 0, timestamp: 3 },
		{ role: "custom", customType: "privacy-screen", content: "<EMAIL_1> in a note", display: true, timestamp: 4 },
	]);

	assert.equal(messages[0]!.content, "this test expects <LITERAL_1>");
	assert.deepEqual(messages[1]!.content, [{ type: "text", text: "docs: write <LITERAL_1> for the address" }]);
	assert.equal(messages[2]!.command, "grep '<LITERAL_1>' notes");
	assert.equal(messages[2]!.output, "notes:3: <LITERAL_1> is how it is written");
	assert.equal(messages[3]!.content, "<LITERAL_1> in a note");
	assert.equal(screen.unmask("<LITERAL_1>"), "<EMAIL_1>");
});

test("a reply is shown with its values, and goes back out with its placeholders", async () => {
	const screen = emailScreen();
	screen.vault.allocate("EMAIL", EMAIL);
	const thinking = { type: "thinking", thinking: "use <EMAIL_1>", thinkingSignature: "sig" };
	const call = { type: "toolCall", id: "call-1", name: "write", arguments: { path: "to.txt", content: "<EMAIL_1>\n" } };
	const reply: Message = { role: "assistant", content: [thinking, { type: "text", text: "Write to <EMAIL_1>.", textSignature: "t" }, call], timestamp: 1 };

	const shown = unmaskAssistant(reply, screen)!.message;

	assert.deepEqual(shown.content, [
		thinking,
		{ type: "text", text: `Write to ${EMAIL}.`, textSignature: "t" },
		{ type: "toolCall", id: "call-1", name: "write", arguments: { path: "to.txt", content: `${EMAIL}\n` } },
	]);
	assert.equal((shown.content as unknown[])[0], thinking, "signed thinking is passed on untouched");
	assert.deepEqual((await masked(screen, [shown])).messages, [reply]);
	assert.deepEqual(reply.content, [thinking, { type: "text", text: "Write to <EMAIL_1>.", textSignature: "t" }, call], "the reply given is not changed");
	assert.deepEqual(call.arguments, { path: "to.txt", content: "<EMAIL_1>\n" });
	assert.deepEqual((await masked(screen, [reply])).messages, [reply], "stored as written, it goes back as written, too");
});

test("the tool calls of a reply get their values where that is safe, and are left for the tool hook where it is not", () => {
	const screen = emailScreen();
	screen.vault.allocate("SECRET", "it's; rm -rf ~");
	const call = (id: string, name: string, args: object) => ({ type: "toolCall", id, name, arguments: args });
	const reply: Message = {
		role: "assistant",
		content: [
			call("a", "bash", { command: "echo <SECRET_1>", timeout: 5 }),
			call("b", "bash", { command: "echo `echo <SECRET_1>`" }),
			call("c", "codemode", { code: "await tools.write({ content: '<SECRET_1>' })" }),
			call("d", "read", { path: "plain.txt" }),
		],
	};
	const rules = { shell: { bash: ["command"] }, code: ["codemode"] };

	const { message, settled } = unmaskAssistant(reply, screen, rules)!;

	assert.deepEqual(message.content, [
		call("a", "bash", { command: `echo 'it'\\''s; rm -rf ~'`, timeout: 5 }),
		call("b", "bash", { command: "echo `echo <SECRET_1>`" }),
		call("c", "codemode", { code: "await tools.write({ content: '<SECRET_1>' })" }),
		call("d", "read", { path: "plain.txt" }),
	]);
	assert.deepEqual([...settled.keys()], ["a"]);
	assert.equal((message.content as unknown[])[3], (reply.content as unknown[])[3], "what did not change is passed on as it is");
});

test("arguments are given their values once: what merely looks like a placeholder afterwards is left alone", () => {
	const screen = emailScreen();
	screen.vault.allocate("SECRET", "hunter22x!");
	screen.vault.allocate("LITERAL", "<SECRET_1>");
	const written = { note: "the file said <LITERAL_1>", key: "<SECRET_1>" };
	const reply: Message = { role: "assistant", content: [{ type: "toolCall", id: "a", name: "write", arguments: written }] };

	const { message, settled } = unmaskAssistant(reply, screen)!;
	const stored = structuredClone((message.content as { arguments: typeof written }[])[0]!.arguments);
	const asStored = structuredClone(stored);
	const asWritten = structuredClone(written);
	unmaskInputInPlace(asStored, screen, [], { settled: settled.get("a") });
	unmaskInputInPlace(asWritten, screen, [], { settled: settled.get("a") });

	assert.deepEqual(stored, { note: "the file said <SECRET_1>", key: "hunter22x!" });
	assert.deepEqual(asStored, stored, "run with the arguments as stored: nothing more happens");
	assert.deepEqual(asWritten, stored, "run with the arguments as written: they get their values now");
});

test("a reply without placeholders needs no replacing", () => {
	const screen = emailScreen();

	assert.equal(unmaskAssistant({ role: "assistant", content: [{ type: "text", text: "Nothing here." }] }, screen), undefined);
	assert.equal(unmaskAssistant({ role: "assistant", content: [{ type: "toolCall", id: "a", name: "read", arguments: { path: "a.txt" } }] }, screen), undefined);
	assert.equal(unmaskAssistant({ role: "user", content: "<EMAIL_1>" }, screen), undefined);
	assert.equal(unmaskAssistant({ role: "assistant", content: "odd" }, screen), undefined);
});

test("the arguments of a tool call get their values, in place, and go back out as the model wrote them", async () => {
	const screen = emailScreen();
	screen.vault.allocate("PERSON", "Lena");
	const written = { path: "/tmp/<PERSON_1>.txt", content: "Dear <PERSON_1>,\n<SECRET_9>", edits: [{ oldText: "<PERSON_1>", newText: "x" }], limit: 3 };
	const input = structuredClone(written);

	const result = unmaskInputInPlace(input, screen, []);

	assert.deepEqual(result, { ok: true });
	assert.deepEqual(input, { path: "/tmp/Lena.txt", content: "Dear Lena,\n<SECRET_9>", edits: [{ oldText: "Lena", newText: "x" }], limit: 3 });
	const stored: Message = { role: "assistant", content: [{ type: "toolCall", id: "c", name: "edit", arguments: input }] };
	assert.deepEqual((await masked(screen, [stored])).messages[0]!.content, [{ type: "toolCall", id: "c", name: "edit", arguments: written }]);
});

test("a shell command gets its values quoted for the shell", async () => {
	const screen = emailScreen();
	screen.vault.allocate("SECRET", "it's; rm -rf ~");
	const input = { command: "echo <SECRET_1> > out.txt", description: "save <SECRET_1>", timeout: 5 };

	const result = unmaskInputInPlace(input, screen, ["command"]);

	assert.deepEqual(result, { ok: true });
	assert.deepEqual(input, { command: `echo 'it'\\''s; rm -rf ~' > out.txt`, description: "save it's; rm -rf ~", timeout: 5 });
	const stored: Message = { role: "assistant", content: [{ type: "toolCall", id: "c", name: "bash", arguments: input }] };
	assert.deepEqual((await masked(screen, [stored])).messages[0]!.content, [
		{ type: "toolCall", id: "c", name: "bash", arguments: { command: "echo <SECRET_1> > out.txt", description: "save <SECRET_1>", timeout: 5 } },
	]);
});

test("a command that cannot take a value safely is refused whole, and nothing is changed", () => {
	const screen = emailScreen();
	screen.vault.allocate("SECRET", "x; rm -rf ~");
	const input = { command: "echo `echo <SECRET_1>`", description: "<SECRET_1>" };

	const result = unmaskInputInPlace(input, screen, ["command"]);

	assert.equal(result.ok, false);
	if (!result.ok) assert.match(result.refusal, /SECRET_1.*backquotes/);
	assert.deepEqual(input, { command: "echo `echo <SECRET_1>`", description: "<SECRET_1>" });
});

test("arguments that are not a record of values are left alone", () => {
	const screen = emailScreen();
	screen.vault.allocate("SECRET", "hunter22x!");

	assert.deepEqual(unmaskInputInPlace("return '<SECRET_1>'", screen, []), { ok: true });
	assert.deepEqual(unmaskInputInPlace(null, screen, []), { ok: true });
	assert.deepEqual(unmaskInputInPlace(undefined, screen, []), { ok: true });
});

test("the last look goes through a payload of any shape and leaves signed and binary parts alone", () => {
	const screen = emailScreen();
	screen.vault.allocate("EMAIL", EMAIL);
	screen.vault.allocate("SECRET", 'pa"ss word');
	const payload = {
		model: EMAIL,
		system: [{ type: "text", text: `Contact ${EMAIL}` }],
		messages: [
			{
				role: "user",
				content: [
					{ type: "text", text: `hi ${EMAIL}` },
					{ type: "image", source: { type: "base64", media_type: "image/png", data: EMAIL } },
					{ type: "image_url", image_url: { url: `data:text/plain,${EMAIL}` } },
				],
			},
			{
				role: "assistant",
				content: [
					{ type: "thinking", thinking: EMAIL, signature: EMAIL },
					{ type: "tool_use", id: EMAIL, name: EMAIL, input: { command: `mail ${EMAIL}` } },
				],
				tool_calls: [{ id: "c", type: "function", function: { name: "bash", arguments: JSON.stringify({ command: 'echo pa"ss word' }) } }],
			},
		],
		tools: [{ name: "bash", description: `for ${EMAIL}`, input_schema: { type: "object" } }],
		max_tokens: 100,
		stream: true,
	};
	const before = structuredClone(payload);

	const swept = sweepPayload(payload, screen);

	const expected = structuredClone(before);
	expected.system[0]!.text = "Contact <EMAIL_1>";
	(expected.messages[0]!.content[0] as { text: string }).text = "hi <EMAIL_1>";
	(expected.messages[1]!.content[1] as { input: { command: string } }).input.command = "mail <EMAIL_1>";
	expected.messages[1]!.tool_calls![0]!.function.arguments = '{"command":"echo <SECRET_1>"}';
	expected.tools[0]!.description = "for <EMAIL_1>";
	assert.deepEqual(swept, { payload: expected, hits: 5 });
	assert.deepEqual(payload, before, "the payload given is not changed");
});

test("the last look returns the very same payload when there is nothing to replace", () => {
	const screen = emailScreen();
	screen.vault.allocate("EMAIL", EMAIL);
	const payload = { messages: [{ role: "user", content: "hi <EMAIL_1>" }], n: 1 };

	const swept = sweepPayload(payload, screen);

	assert.equal(swept.payload, payload);
	assert.equal(swept.hits, 0);
});
