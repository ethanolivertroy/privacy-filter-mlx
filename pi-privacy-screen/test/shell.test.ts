import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { after, test } from "node:test";

import { embed } from "../src/shell.ts";
import { tokenPattern } from "../src/vault.ts";

const T = "<SECRET_1>";
const valueIs = (value: string) => (token: string) => (token === T ? value : undefined);

function placed(command: string, value: string): string {
	const result = embed(command, valueIs(value));
	assert.ok("command" in result, `refused: ${"refused" in result ? result.refused : ""}`);
	return result.command;
}

function refused(command: string, value: string): string {
	const result = embed(command, valueIs(value));
	assert.ok("refused" in result, `expected a refusal, got: ${"command" in result ? result.command : ""}`);
	return result.refused;
}

// ------------------------------------------------------------ on paper

test("a command without a known placeholder is returned as it is", () => {
	for (const command of ["ls -la", "echo <SECRET_9> | cat", "echo `date` $((1+2)) ${HOME} \"$(case x in x) echo y;; esac)\"", "echo 'unterminated"]) {
		assert.deepEqual(embed(command, valueIs("x")), { command });
	}
});

test("outside quotes the value is put in single quotes", () => {
	assert.equal(placed("echo <SECRET_1>", "it's"), `echo 'it'\\''s'`);
	assert.equal(placed("export TOKEN=<SECRET_1>", "abc def"), "export TOKEN='abc def'");
	assert.equal(placed("curl https://x/<SECRET_1>/y", "a b"), "curl https://x/'a b'/y");
});

test("inside quotes the value is escaped for that kind of quote", () => {
	assert.equal(placed('curl -H "Authorization: Bearer <SECRET_1>" x', 'a"b$c`d\\e'), 'curl -H "Authorization: Bearer a\\"b\\$c\\`d\\\\e" x');
	assert.equal(placed("echo 'key=<SECRET_1>'", "it's"), `echo 'key=it'\\''s'`);
	assert.equal(placed("echo $'a <SECRET_1>'", "it's a\\b"), "echo $'a it\\'s a\\\\b'");
});

test("a value never joins a variable name in front of it", () => {
	assert.equal(placed('echo "$PATH:<SECRET_1>" "$HOME<SECRET_1>" "${HOME}<SECRET_1>"', "head"), 'echo "$PATH:""head" "$HOME""head" "${HOME}head"');
});

test("every occurrence is placed, each by its own surroundings", () => {
	assert.equal(placed(`a <SECRET_1> "<SECRET_1>" '<SECRET_1>' <SECRET_9>`, "x'y"), `a 'x'\\''y' "x'y" 'x'\\''y' <SECRET_9>`);
});

test("a value that would be read as an option is refused, unless options have ended or it is glued to a word", () => {
	assert.match(refused("rm <SECRET_1>", "-rf"), /option/);
	assert.match(refused('rm "<SECRET_1>"', "-rf"), /option/);
	assert.match(refused("rm '<SECRET_1>'", "-rf"), /option/);
	assert.match(refused("rm -- a; rm <SECRET_1>", "-rf"), /option/);
	assert.equal(placed("rm -- <SECRET_1>", "-rf"), "rm -- '-rf'");
	assert.equal(placed("tool --name=<SECRET_1> x<SECRET_1>", "-rf"), "tool --name='-rf' x'-rf'");
	assert.equal(placed("rm <SECRET_1>", "file-rf"), "rm 'file-rf'");
});

test("where a value cannot be placed safely the command is refused with the reason", () => {
	const cases: [command: string, value: string, reason: RegExp][] = [
		["echo `echo <SECRET_1>`", "x", /backquotes/],
		['echo "${X:-<SECRET_1>}"', "x", /\$\{/],
		["echo $((1 + <SECRET_1>))", "1", /arithmetic/],
		["((x = <SECRET_1>))", "1", /arithmetic/],
		["echo $<SECRET_1>", "x", /after a \$/],
		['echo "$<SECRET_1>"', "x", /after a \$/],
		["echo \\<SECRET_1>", "x", /backslash/],
		['echo "C:\\<SECRET_1>"', "x", /backslash/],
		["echo $'a\\<SECRET_1>'", "x", /backslash/],
		['echo "$arr[<SECRET_1>]"', "1", /subscript/],
		["cat <<<SECRET_1>\nx\n<SECRET_1>", "END", /here-document/],
		["cat <<EOF\n<SECRET_1>\nEOF", "a\nEOF\ntouch x", /here-document/],
		["cat <<-EOF\n\t<SECRET_1>\n\tEOF", "a\n\tb", /here-document/],
		["cat <<EOF\n$(date) <SECRET_1>\nEOF", "x", /here-document/],
		["cat <<EOF\n$X<SECRET_1>\nEOF", "x", /here-document/],
		["echo a # <SECRET_1>", "x\ntouch y", /comment/],
		["echo <SECRET_1>", "a\0b", /NUL/],
	];
	for (const [command, value, reason] of cases) {
		const message = refused(command, value);
		assert.match(message, reason, command);
		assert.match(message, /the value of SECRET_1 /, command);
		if (value.length > 1) assert.ok(!message.includes(value), `the refusal repeats the value: ${message}`);
	}
});

test("inside a command substitution in double quotes, a value that old bash would brace-expand is refused", () => {
	const nested = 'X="$(curl -H "Authorization: Bearer <SECRET_1>" https://example.com)"';

	assert.equal(placed(nested, "abc-123_x.y"), 'X="$(curl -H "Authorization: Bearer abc-123_x.y" https://example.com)"');
	assert.equal(placed(nested, `a,b'c"d`), `X="$(curl -H "Authorization: Bearer a,b'c\\"d" https://example.com)"`);
	assert.match(refused(nested, "{a,b}"), /bash 3\.2/);
	assert.match(refused(`X="$(tool {x,y} '<SECRET_1>')"`, "a'b"), /bash 3\.2/);
	assert.equal(placed('X=$(tool "<SECRET_1>"); Y="<SECRET_1>"', "{a,b}"), 'X=$(tool "{a,b}"); Y="{a,b}"');
});

test("inside a command substitution, old bash's reading of here-documents and $'...' is respected", () => {
	const commit = `git commit -m "$(cat <<'EOF'\nThanks <SECRET_1>\nEOF\n)"`;

	assert.equal(placed(commit, "Lena Fischer <lena@example.com>"), `git commit -m "$(cat <<'EOF'\nThanks Lena Fischer <lena@example.com>\nEOF\n)"`);
	assert.match(refused(commit, ") ; touch x ; ("), /bash 3\.2/);
	assert.match(refused(commit, "it's"), /bash 3\.2/);
	assert.match(refused(`X=$(cat <<EOF\nit's here\nEOF\n); echo <SECRET_1>`, "x"), /here-document inside/);
	assert.match(refused("X=$(printf %s $'a <SECRET_1>')", "it's"), /bash 3\.2/);
	assert.equal(placed("X=$(printf %s $'a <SECRET_1>')", "plain value"), "X=$(printf %s $'a plain value')");
	assert.match(refused("X=$(printf %s $'a\\'b'); echo <SECRET_1>", "x"), /escaped quote/);
	assert.equal(placed("printf %s $'a\\'b' $'<SECRET_1>'", "it's"), "printf %s $'a\\'b' $'it\\'s'");
});

test("after syntax it does not follow, a placeholder is refused rather than guessed at", () => {
	for (const command of [
		'echo "$(case x in x) echo y;; esac)" <SECRET_1>',
		'echo "$(echo a # note)\n)" <SECRET_1>',
		'(( x = "1" )); echo <SECRET_1>',
		"a[1<<2]=5\necho <SECRET_1>",
		"echo $[1+2] <SECRET_1>",
		'echo "${X:-$(date)}" <SECRET_1>',
		"echo `date\necho <SECRET_1>",
	]) {
		assert.match(refused(command, "x"), /the value of SECRET_1 /, command);
	}
});

// ---------------------------------------------------- in the real shells

const CANARY = "CANARY";
const HOSTILE = [
	"plain-word1",
	"two words",
	"it's",
	'say "hi"',
	"back\\slash",
	"trailing\\",
	"$HOME",
	"${HOME}",
	`$(touch ${CANARY})`,
	`\`touch ${CANARY}\``,
	`; touch ${CANARY}`,
	`&& touch ${CANARY}`,
	`| touch ${CANARY}`,
	`> ${CANARY}`,
	"line one\nline two",
	`a\n; touch ${CANARY}\nb`,
	`x\nEOF\ntouch ${CANARY}`,
	"EOF",
	"*",
	"~",
	"!!",
	"#not a comment",
	`'; touch ${CANARY}; '`,
	`"; touch ${CANARY}; "`,
	`'$(touch ${CANARY})'`,
	"$'\\n'",
	"\\'",
	'\\"',
	`\\$(touch ${CANARY})`,
	`\\\\$(touch ${CANARY})`,
	"$((1+1))",
	"tab\there",
	"x\n\ty",
	"é 😀 ü",
	"%s%n %d",
	"a'b\"c`d$e\\f",
	"{a,b}",
	"-n",
	"--help",
	"=ls",
];

interface Template {
	command: string;
	/** Shells (by name) that have the syntax; all three by default. */
	shells?: string[];
	/** Values this use must refuse. */
	refuses?: (value: string) => boolean;
	/** False for a use whose output does not show the value (a comment). */
	shows?: boolean;
}

const dashFirst = (value: string) => value.startsWith("-");
const endsHeredoc = (body: string) => (value: string) => body.replaceAll(T, () => value).split("\n").some((line) => line === "EOF");
const BASH_ZSH = ["bash", "zsh"];
/** bash 3.2 expands braces inside "$( ... )": see the note in shell.ts. */
const hasBraces = (value: string) => /[{}]/.test(value);

const TEMPLATES: Template[] = [
	{ command: "printf -- '%s\\0' <SECRET_1>" },
	{ command: "printf '%s\\0' <SECRET_1>", refuses: dashFirst },
	{ command: "printf -- '%s\\0' pre<SECRET_1>post" },
	{ command: 'printf -- \'%s\\0\' "<SECRET_1>"' },
	{ command: 'printf -- \'%s\\0\' "a <SECRET_1> b" "<SECRET_1><SECRET_1>"' },
	{ command: "printf -- '%s\\0' '<SECRET_1>'" },
	{ command: "printf -- '%s\\0' 'a <SECRET_1> b'" },
	{ command: "printf -- '%s\\0' $'a <SECRET_1> b\\tc'", shells: BASH_ZSH },
	{ command: "printf -- '%s\\0' a<SECRET_1>\"b<SECRET_1>\"'c<SECRET_1>'" },
	{ command: 'printf -- \'%s\\0\' "$(printf -- \'%s\' <SECRET_1>)"', refuses: hasBraces },
	{ command: 'printf -- \'%s\\0\' "$(printf -- \'%s\' "x <SECRET_1>" \'y <SECRET_1>\')"', refuses: hasBraces },
	{ command: 'printf -- \'%s\\0\' "a $(printf -- \'%s\' "$(printf -- \'%s\' <SECRET_1>)") <SECRET_1>"', refuses: hasBraces },
	{ command: 'printf -- \'%s\\0\' $(printf -- \'%s\' "x") "$(printf -- \'%s\' y)" "<SECRET_1>"' },
	{ command: 'X=<SECRET_1>; printf -- \'%s\\0\' "$X"' },
	{ command: 'X="<SECRET_1>" Y=\'<SECRET_1>\' sh -c \'printf -- "%s\\0" "$X" "$Y"\'' },
	{ command: 'printf -- \'%s\\0\' "$HOME/<SECRET_1>" "${HOME:+set}" <SECRET_1>' },
	{ command: "(printf -- '%s\\0' <SECRET_1>)" },
	{ command: "{ printf -- '%s\\0' <SECRET_1>; }" },
	{ command: 'if true; then printf -- \'%s\\0\' "<SECRET_1>"; fi' },
	{ command: 'for x in <SECRET_1> "<SECRET_1>"; do printf -- \'%s\\0\' "$x"; done', refuses: dashFirst },
	{ command: "case x in x) printf -- '%s\\0' <SECRET_1>;; esac" },
	{ command: 'printf -- \'%s\\0\' "<SECRET_1>" | cat' },
	{ command: 'arr=(<SECRET_1> "<SECRET_1>"); printf -- \'%s\\0\' "${arr[@]}"', shells: BASH_ZSH, refuses: dashFirst },
	{ command: "cat <<<<SECRET_1>", shells: BASH_ZSH, refuses: dashFirst },
	{ command: 'cat <<< "a <SECRET_1>"', shells: BASH_ZSH },
	{ command: "cat <<EOF\na <SECRET_1> b\nEOF", refuses: endsHeredoc("a <SECRET_1> b") },
	{ command: "cat <<EOF\n<SECRET_1>\nEOF\nprintf done", refuses: endsHeredoc("<SECRET_1>") },
	{ command: "cat <<'EOF'\na <SECRET_1> $HOME\n<SECRET_1>\nEOF", refuses: endsHeredoc("a <SECRET_1> $HOME\n<SECRET_1>") },
	{ command: 'cat <<"EOF"\n<SECRET_1> b\nEOF', refuses: endsHeredoc("<SECRET_1> b") },
	{
		command: "cat <<-EOF\n\ta <SECRET_1>\n\tEOF",
		refuses: (value) => /\n\t/.test(value) || endsHeredoc("a <SECRET_1>")(value),
	},
	{ command: "cat <<EOF; printf -- '%s\\0' <SECRET_1>\nbody <SECRET_1>\nEOF", refuses: endsHeredoc("body <SECRET_1>") },
	{ command: "printf -- '%s\\0' a # note <SECRET_1>\nprintf -- '%s\\0' b", refuses: (value) => /[\n\r]/.test(value), shows: false },
	{ command: "printf -- '%s\\0' a \\\n  <SECRET_1> b" },
];

const SHELLS = ["/bin/bash", "/bin/zsh", "/bin/dash"].filter((shell) => existsSync(shell));
const scratch = mkdtempSync(join(tmpdir(), "pi-privacy-shell-"));
after(() => rmSync(scratch, { recursive: true, force: true }));

function run(shell: string, command: string) {
	const result = spawnSync(shell, ["-c", command], {
		cwd: scratch,
		encoding: "utf8",
		timeout: 20_000,
		env: { PATH: "/usr/bin:/bin", HOME: scratch, FIXED: "fixed" },
	});
	return { stdout: result.stdout, stderr: result.stderr, status: result.status };
}

for (const shell of SHELLS) {
	test(`in ${basename(shell)}, a hostile value arrives as plain data wherever a placeholder may stand`, () => {
		let checked = 0;
		for (const template of TEMPLATES) {
			if (template.shells && !template.shells.includes(basename(shell))) continue;
			// What the model wrote, with a harmless word for the value: that is what the command must keep meaning.
			const harmless = run(shell, template.command.replaceAll(T, "MARK"));
			assert.deepEqual([harmless.status, harmless.stderr], [0, ""], `the template itself fails: ${template.command}`);
			if (template.shows !== false) assert.ok(harmless.stdout.includes("MARK"), `the template shows nothing: ${template.command}`);

			for (const value of HOSTILE) {
				const context = `${template.command}\nvalue: ${JSON.stringify(value)}`;
				const result = embed(template.command, valueIs(value));
				if (template.refuses?.(value)) {
					assert.ok("refused" in result, `expected a refusal\n${context}`);
					continue;
				}
				assert.ok("command" in result, `refused: ${"refused" in result ? result.refused : ""}\n${context}`);
				const actual = run(shell, result.command);
				const expected = { ...harmless, stdout: harmless.stdout.replaceAll("MARK", () => value) };
				assert.deepEqual(actual, expected, `${context}\nran: ${result.command}`);
				assert.deepEqual(readdirSync(scratch), [], `the value did something\n${context}\nran: ${result.command}`);
				checked++;
			}
		}
		assert.ok(checked > 500, `only ${checked} runs`);
	});

	test(`in ${basename(shell)}, a value after a variable does not change which variable is read`, () => {
		for (const value of ["tail", "head", "x", "_y", "9", "[1]", ":h"]) {
			const command = placed('printf -- \'%s\\0\' "$FIXED<SECRET_1>" "$FIXED:<SECRET_1>" $FIXED<SECRET_1> $FIXED:<SECRET_1>', value);
			assert.equal(run(shell, command).stdout, `fixed${value}\0fixed:${value}\0fixed${value}\0fixed:${value}\0`, command);
		}
	});
}

test("a refusal names the placeholder without its brackets, so that it reaches the model as written", () => {
	const placed = embed("echo `echo <SECRET_1>`", (token) => (token === "<SECRET_1>" ? "hunter22x!" : undefined));

	assert.ok("refused" in placed);
	assert.match(placed.refused, /the value of SECRET_1 /);
	assert.equal(tokenPattern().test(placed.refused), false, "what looks like a placeholder in a tool's output is passed to the model as a literal");
});
