/**
 * Random commands, hostile values, real shells: whatever `embed` agrees to place must never run.
 *
 * Every value below tries, in its own way, to create a file called CANARY. A command is built at
 * random from the constructs `embed` supports, nested in each other, with placeholders scattered
 * through it. If `embed` refuses, fine. If it places the value, the command is run, and the file
 * must not appear.
 *
 * SHELL_FUZZ=2000 node --test test/shell-fuzz.test.ts   runs more commands (default 80, seeded).
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { after, test } from "node:test";

import { embed } from "../src/shell.ts";

const T = "<SECRET_1>";
const ROUNDS = Number(process.env.SHELL_FUZZ ?? 80);
const SEED = Number(process.env.SHELL_FUZZ_SEED ?? 20261123);

const ATTACKS = [
	"$(touch CANARY)",
	"`touch CANARY`",
	"; touch CANARY ;",
	"\ntouch CANARY\n",
	"'; touch CANARY; '",
	'"; touch CANARY; "',
	"' $(touch CANARY) '",
	'" $(touch CANARY) "',
	"\\$(touch CANARY)",
	"\\\\$(touch CANARY)",
	"\\",
	"x\nEOF\ntouch CANARY\n:",
	"'\ntouch CANARY\n'",
	") ; touch CANARY ; (",
	"}; touch CANARY; {",
	"') ; touch CANARY ; ('",
	'") ; touch CANARY ; ("',
	"| touch CANARY",
	"& touch CANARY &",
	"\\'; touch CANARY; \\'",
	"$'\\n'touch CANARY$'\\n'",
	"x\\\ntouch CANARY",
	"${X:-$(touch CANARY)}",
	"$((1)) $(touch CANARY)",
	"--version; touch CANARY",
	"> CANARY",
	"a' b\" c` d$(touch CANARY)",
];

/** A small seeded generator (mulberry32), so that a failure can be reproduced. */
function random(seed: number) {
	let state = seed >>> 0;
	const next = () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
	return { pick: <V>(items: readonly V[]): V => items[Math.floor(next() * items.length)]!, chance: (p: number) => next() < p };
}

function generator(seed: number, ansiC: boolean) {
	const { pick, chance } = random(seed);
	const filler = () => pick(["lit", "a.b", "-x", "'q r'", '"s t"', "x=1", "$FIXED", '"$FIXED"', "\\;", "a\\ b"]);

	function word(depth: number): string {
		const kinds = [
			() => T,
			() => `pre${T}post`,
			() => `"a ${T} b"`,
			() => `"${T}"`,
			() => `'a ${T} b'`,
			() => `'${T}'`,
			() => `"$FIXED${T}"`,
			() => `"$FIXED:${T} \${FIXED}${T}"`,
			() => `--name=${T}`,
			() => `a${T}"b${T}"'c${T}'`,
			() => filler(),
		];
		if (ansiC) kinds.push(() => `$'a ${T} \\t b'`);
		if (depth < 3) {
			kinds.push(
				() => `"x $(${command(depth + 1)}) y"`,
				() => `"$(${command(depth + 1)})"`,
				() => `$(${command(depth + 1)})`,
			);
		}
		return pick(kinds)();
	}

	function simple(depth: number): string {
		const words = [word(depth)];
		while (chance(0.5) && words.length < 4) words.push(word(depth));
		return `printf -- '%s|' ${words.join(" ")}`;
	}

	function heredoc(): string {
		const marker = pick(["EOF", "'EOF'", '"EOF"', "\\EOF"]);
		const lines = [pick([`a ${T} b`, T, `${T} $FIXED`, `"${T}" '${T}'`, "plain"])];
		if (chance(0.4)) lines.push(pick([`second ${T}`, "more text", `x=${T}`]));
		return `cat <<${marker}\n${lines.join("\n")}\nEOF\n`;
	}

	function command(depth: number): string {
		const kinds = [() => simple(depth), () => simple(depth), () => simple(depth)];
		if (depth < 3) {
			kinds.push(
				() => heredoc(),
				() => `(${command(depth + 1)})`,
				() => `{ ${command(depth + 1)}\n}`,
				() => `${command(depth + 1)}\n${simple(depth)}`,
				() => `${simple(depth)} | cat`,
				() => `X=${word(depth)}; ${simple(depth)}`,
				() => `if true; then ${command(depth + 1)}\nfi`,
				() => `case x in x) ${simple(depth)};; esac`,
				() => `${simple(depth)} # note ${T}\n${simple(depth)}`,
				() => `for v in ${word(depth)}; do ${simple(depth)}; done`,
			);
		}
		return pick(kinds)();
	}

	return { command: () => command(0) };
}

const SHELLS = ["/bin/bash", "/bin/zsh", "/bin/dash"].filter((shell) => existsSync(shell));
const scratch = mkdtempSync(join(tmpdir(), "pi-privacy-fuzz-"));
after(() => rmSync(scratch, { recursive: true, force: true }));

function run(shell: string, command: string) {
	return spawnSync(shell, ["-c", command], {
		cwd: scratch,
		encoding: "utf8",
		timeout: 20_000,
		env: { PATH: "/usr/bin:/bin", HOME: scratch, FIXED: "fixed" },
	});
}

for (const shell of SHELLS) {
	test(`in ${basename(shell)}, no value that was placed in a random command gets to run (${ROUNDS} commands, seed ${SEED})`, () => {
		const make = generator(SEED, basename(shell) !== "dash");
		let placed = 0;
		let refused = 0;
		let commands = 0;
		for (let round = 0; round < ROUNDS; round++) {
			const command = make.command();
			if (!command.includes(T)) continue;
			// With a harmless word for the value the command has to be a working one, or the run proves nothing.
			const harmless = run(shell, command.replaceAll(T, "MARK"));
			if (harmless.status !== 0 || harmless.stderr !== "") continue;
			commands++;
			for (const attack of ATTACKS) {
				const result = embed(command, (token) => (token === T ? attack : undefined));
				if ("refused" in result) {
					refused++;
					continue;
				}
				run(shell, result.command);
				const left = readdirSync(scratch);
				for (const name of left) rmSync(join(scratch, name), { recursive: true, force: true });
				assert.deepEqual(left, [], `the value ran\ncommand: ${JSON.stringify(command)}\nvalue: ${JSON.stringify(attack)}\nran: ${JSON.stringify(result.command)}`);
				placed++;
			}
		}
		assert.ok(commands >= ROUNDS / 4, `only ${commands} of ${ROUNDS} generated commands were usable`);
		assert.ok(placed > refused, `placed ${placed}, refused ${refused}: the generator is not testing what is supported`);
	});
}

test("the same commands with the value pasted in do run it: the check above can fail", () => {
	const shell = SHELLS[0]!;
	const make = generator(SEED, true);
	let ran = 0;
	for (let round = 0; round < 40; round++) {
		const command = make.command();
		if (!command.includes(T)) continue;
		for (const attack of ATTACKS.slice(0, 6)) {
			run(shell, command.replaceAll(T, () => attack));
			if (readdirSync(scratch).length > 0) ran++;
			for (const name of readdirSync(scratch)) rmSync(join(scratch, name), { recursive: true, force: true });
		}
	}
	assert.ok(ran > 20, `pasting the value in ran it only ${ran} times`);
});
