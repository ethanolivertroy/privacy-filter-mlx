import { tokenPattern } from "./vault.ts";

export type Embedded = { command: string } | { refused: string };

const HINT = "Use the placeholder inside plain single or double quotes, or write the value to a file with a file tool and use the file.";

interface Spot {
	end: number;
	token: string;
	value: string;
}

interface Plain {
	kind: "plain";
	substitution: boolean;
	parenDepth: number;
	wordLength: number;
	wordPrefix: string;
	optionsEnded: boolean;
}

interface Double {
	kind: "double";
	owner: Plain;
}

interface Heredoc {
	delimiter: string;
	literal: boolean;
	stripTabs: boolean;
	nested: boolean;
}

class Refusal extends Error {}

const VARIABLE = /\$(?:[A-Za-z_][A-Za-z0-9_]*|[0-9@*#?!-]):?$/; // $NAME or $NAME: right before (zsh reads $NAME:h as a modifier)
const SUBSCRIPT = /\$[A-Za-z_][A-Za-z0-9_]*\[[^\]\n]*$/; // $NAME[ ... right before (a zsh subscript)
const WORD_END = " \t\n;|&()<>";
const BARE_BRACE = /(?<!\$)\{/;

const single = (value: string) => `'${inSingle(value)}'`;
const inSingle = (value: string) => value.replaceAll("'", `'\\''`);
const inDouble = (value: string) => value.replace(/[\\"$`]/g, "\\$&");
const inAnsiC = (value: string) => value.replace(/[\\']/g, "\\$&");
const inHeredoc = (value: string) => value.replace(/[\\$`]/g, "\\$&");

export function embed(command: string, valueOf: (token: string) => string | undefined): Embedded {
	const spots = new Map<number, Spot>();
	for (const match of command.matchAll(tokenPattern())) {
		const value = valueOf(match[0]);
		if (value !== undefined) spots.set(match.index, { end: match.index + match[0].length, token: match[0], value });
	}
	if (spots.size === 0) return { command };
	try {
		return { command: place(command, spots) };
	} catch (error) {
		if (error instanceof Refusal) return { refused: error.message };
		throw error;
	}
}

const stripPlaceholderBrackets = (token: string) => token.slice(1, -1);

function refuse(spot: Spot, why: string): never {
	throw new Refusal(`The privacy screen cannot put the value of ${stripPlaceholderBrackets(spot.token)} into this command: ${why}. Nothing was run. ${HINT}`);
}

function place(command: string, spots: ReadonlyMap<number, Spot>): string {
	const n = command.length;
	const order = [...spots.keys()].sort((a, b) => a - b);
	const out: string[] = [];
	const frames: (Plain | Double)[] = [plain(false)];
	const pending: Heredoc[] = [];
	let i = 0;

	for (const spot of spots.values()) {
		if (spot.value.includes("\0")) refuse(spot, "its value contains a NUL character");
	}

	function plain(substitution: boolean): Plain {
		return { kind: "plain", substitution, parenDepth: 0, wordLength: 0, wordPrefix: "", optionsEnded: false };
	}
	function copy(to: number): void {
		const end = Math.min(n, to);
		if (end > i) out.push(command.slice(i, end));
		i = Math.max(i, end);
	}
	function spotFrom(from: number): Spot | undefined {
		const index = order.find((at) => at >= from);
		return index === undefined ? undefined : spots.get(index);
	}
	function lose(what: string): void {
		const next = spotFrom(i);
		if (next) refuse(next, `it comes after ${what}, which the screen does not follow`);
		copy(n);
	}
	function grow(frame: Plain, text: string): void {
		frame.wordLength++;
		if (frame.wordPrefix.length < 16) frame.wordPrefix += text;
	}
	function endWord(frame: Plain): void {
		if (frame.wordPrefix === "--") frame.optionsEnded = true;
		const wasCase = frame.wordPrefix === "case";
		frame.wordPrefix = "";
		frame.wordLength = 0;
		if (wasCase && insideSubstitution()) lose("a case statement inside $( )");
	}
	function endCommand(frame: Plain): void {
		endWord(frame);
		frame.optionsEnded = false;
	}
	function insideSubstitution(): boolean {
		return frames.some((frame) => frame.kind === "plain" && frame.substitution);
	}
	function checkOption(spot: Spot, owner: Plain): void {
		if (owner.wordLength === 0 && !owner.optionsEnded && spot.value.startsWith("-")) {
			refuse(spot, "its value starts with a dash and would be read as an option here; put -- before it, or attach it to its option as in --name=…");
		}
	}
	function behind(): string {
		return command.slice(Math.max(0, i - 200), i);
	}
	let bareBrace: boolean | undefined;
	/** bash 3.2 brace-expands the text of a `$( )` that stands in double quotes, ignoring the quotes inside it. */
	function checkBraces(spot: Spot): void {
		let quoted = false;
		const affected = frames.some((frame) => {
			if (frame.kind === "double") quoted = true;
			return frame.kind === "plain" && frame.substitution && quoted;
		});
		if (!affected) return;
		// Braces in the value could be expanded; quotes in it could expose braces the command has itself.
		bareBrace ??= BARE_BRACE.test(command);
		if (/[{}]/.test(spot.value) || (bareBrace && /[,"'\\]/.test(spot.value))) {
			refuse(spot, "it stands in a $( ) inside double quotes, where bash 3.2 would expand braces in its value; assign the value to a variable first");
		}
	}

	function plainStep(frame: Plain): void {
		const spot = spots.get(i);
		if (spot) {
			if (SUBSCRIPT.test(behind())) refuse(spot, "it stands in the subscript of a variable");
			checkOption(spot, frame);
			checkBraces(spot);
			out.push(single(spot.value));
			grow(frame, "?");
			i = spot.end;
			return;
		}
		const c = command[i]!;
		switch (c) {
			case "\\": {
				if (spots.has(i + 1)) refuse(spots.get(i + 1)!, "it follows a backslash");
				if (command[i + 1] !== "\n") grow(frame, command.slice(i, i + 2));
				return copy(i + 2);
			}
			case "'":
				return singleQuoted(frame);
			case '"':
				frame.wordPrefix += '"';
				frames.push({ kind: "double", owner: frame });
				return copy(i + 1);
			case "`":
				grow(frame, "`");
				return backquoted();
			case "$":
				return dollar(frame, frame);
			case "#":
				if (frame.wordLength === 0 && frame.wordPrefix === "") return comment();
				break;
			case " ":
			case "\t":
				endWord(frame);
				return copy(i + 1);
			case "\n":
				endCommand(frame);
				copy(i + 1);
				if (pending.length > 0) heredocBodies();
				return;
			case ";":
			case "|":
			case "&":
				endCommand(frame);
				return copy(i + 1);
			case "(":
				if (command[i + 1] === "(") return arithmetic(i);
				endCommand(frame);
				frame.parenDepth++;
				return copy(i + 1);
			case ")":
				endCommand(frame);
				if (frame.parenDepth > 0) frame.parenDepth--;
				else if (frame.substitution) frames.pop();
				return copy(i + 1);
			case "<":
				if (command[i + 1] === "<" && !spots.has(i + 1)) {
					if (command[i + 2] === "<" && !spots.has(i + 2)) {
						endWord(frame);
						return copy(i + 3);
					}
					return heredocStart(frame);
				}
				endWord(frame);
				return copy(i + 1);
			case ">":
				endWord(frame);
				return copy(i + 1);
		}
		grow(frame, c);
		copy(i + 1);
	}

	function singleQuoted(frame: Plain): void {
		const close = command.indexOf("'", i + 1);
		if (close === -1) return lose("an unterminated quote");
		if (pending.length > 0 && command.slice(i, close).includes("\n")) return lose("a quoted line break before the text of a here-document");
		frame.wordPrefix += "'";
		copy(i + 1);
		while (i < close) {
			const spot = spots.get(i);
			if (spot) {
				checkOption(spot, frame);
				checkBraces(spot);
				out.push(inSingle(spot.value));
				i = spot.end;
			} else {
				copy(i + 1);
			}
			frame.wordLength++;
		}
		copy(close + 1);
	}

	function ansiC(frame: Plain): void {
		frame.wordPrefix += "$'";
		copy(i + 2);
		while (i < n) {
			const spot = spots.get(i);
			const c = command[i]!;
			if (spot) {
				checkOption(spot, frame);
				checkBraces(spot);
				if (insideSubstitution() && /['\\]/.test(spot.value)) {
					refuse(spot, "it stands in $'...' inside $( ), and its value has quotes or backslashes, which bash 3.2 would misread there");
				}
				out.push(inAnsiC(spot.value));
				i = spot.end;
			} else if (c === "\\") {
				if (spots.has(i + 1)) refuse(spots.get(i + 1)!, "it follows a backslash");
				if (command[i + 1] === "'" && insideSubstitution()) return lose("an escaped quote in $'...' inside $( )");
				copy(i + 2);
			} else if (c === "'") {
				return copy(i + 1);
			} else {
				if (c === "\n" && pending.length > 0) return lose("a quoted line break before the text of a here-document");
				copy(i + 1);
			}
			frame.wordLength++;
		}
	}

	function comment(): void {
		if (insideSubstitution()) return lose("a comment inside $( )");
		let end = command.indexOf("\n", i);
		if (end === -1) end = n;
		while (i < end) {
			const spot = spots.get(i);
			if (!spot) {
				copy(i + 1);
				continue;
			}
			if (/[\n\r]/.test(spot.value)) refuse(spot, "it stands in a comment and its value has more than one line");
			out.push(spot.value);
			i = spot.end;
		}
	}

	function backquoted(): void {
		let close = i + 1;
		while (close < n && command[close] !== "`") {
			if (spots.has(close)) refuse(spots.get(close)!, "it stands inside backquotes; use $( ) instead");
			close += command[close] === "\\" ? 2 : 1;
		}
		if (close >= n) return lose("an unterminated backquote");
		copy(close + 1);
	}

	function doubleStep(frame: Double): void {
		const spot = spots.get(i);
		if (spot) {
			const before = behind();
			if (SUBSCRIPT.test(before)) refuse(spot, "it stands in the subscript of a variable");
			checkOption(spot, frame.owner);
			checkBraces(spot);
			out.push((VARIABLE.test(before) ? '""' : "") + inDouble(spot.value));
			grow(frame.owner, "?");
			i = spot.end;
			return;
		}
		const c = command[i]!;
		if (c === "\\") {
			const next = command[i + 1];
			if (next !== undefined && '$`"\\\n'.includes(next)) {
				grow(frame.owner, c);
				return copy(i + 2);
			}
			if (spots.has(i + 1)) refuse(spots.get(i + 1)!, "it follows a backslash");
		} else if (c === '"') {
			frame.owner.wordPrefix += '"';
			frames.pop();
			return copy(i + 1);
		} else if (c === "`") {
			grow(frame.owner, c);
			return backquoted();
		} else if (c === "$") {
			return dollar(frame, frame.owner);
		} else if (c === "\n" && pending.length > 0) {
			return lose("a quoted line break before the text of a here-document");
		}
		grow(frame.owner, c);
		copy(i + 1);
	}

	function dollar(frame: Plain | Double, owner: Plain): void {
		grow(owner, "$");
		if (command.startsWith("$((", i)) return arithmetic(i + 1);
		if (command.startsWith("$(", i)) {
			copy(i + 2);
			frames.push(plain(true));
			return;
		}
		if (command.startsWith("${", i)) return parameter();
		if (command[i + 1] === "[") return lose("a $[ ] expression");
		if (frame.kind === "plain" && command[i + 1] === "'") return ansiC(frame);
		if (spots.has(i + 1)) refuse(spots.get(i + 1)!, "it stands right after a $");
		copy(i + 1);
	}

	function arithmetic(open: number): void {
		let depth = 0;
		for (let at = open; at < n; at++) {
			const c = command[at]!;
			if (spots.has(at)) refuse(spots.get(at)!, "it stands inside an arithmetic expression");
			if (c === "'" || c === '"' || c === "`") {
				copy(at);
				return lose("an arithmetic expression with quotes in it");
			}
			if (c === "(") depth++;
			else if (c === ")" && --depth === 0) return copy(at + 1);
		}
		lose("an unterminated arithmetic expression");
	}

	function parameter(): void {
		let depth = 0;
		for (let at = i + 1; at < n; at++) {
			const c = command[at]!;
			if (spots.has(at)) refuse(spots.get(at)!, "it stands inside a ${ } expansion");
			if (c === "\\") at++;
			else if (c === "'" || c === '"' || c === "`" || command.startsWith("$(", at)) {
				copy(at);
				return lose("a ${ } expansion with quotes or commands in it");
			} else if (c === "{" && command[at - 1] === "$") depth++;
			else if (c === "}" && --depth === 0) return copy(at + 1);
		}
		lose("an unterminated ${ } expansion");
	}

	function heredocStart(frame: Plain): void {
		if (/\[[^\]]*$/.test(frame.wordPrefix)) return lose("a << inside a subscript");
		endWord(frame);
		let at = i + 2;
		const stripTabs = command[at] === "-";
		if (stripTabs) at++;
		while (command[at] === " " || command[at] === "\t") at++;
		const start = at;
		let delimiter = "";
		let literal = false;
		while (at < n && !WORD_END.includes(command[at]!)) {
			const c = command[at]!;
			if (spots.has(at)) refuse(spots.get(at)!, "it would be the end marker of a here-document");
			if (c === "\\") {
				literal = true;
				delimiter += command[at + 1] ?? "";
				at += 2;
			} else if (c === "'" || c === '"') {
				const close = command.indexOf(c, at + 1);
				const inside = spotFrom(at);
				if (close === -1) return lose("an unterminated quote");
				if (inside && spots.get(close) !== inside && command.indexOf(inside.token, at) < close) {
					refuse(inside, "it would be the end marker of a here-document");
				}
				literal = true;
				delimiter += command.slice(at + 1, close);
				at = close + 1;
			} else {
				delimiter += c;
				at++;
			}
		}
		if (at === start) return lose("a here-document without an end marker");
		pending.push({ delimiter, literal, stripTabs, nested: insideSubstitution() });
		copy(at);
	}

	function heredocBodies(): void {
		while (pending.length > 0 && i < n) {
			const doc = pending.shift()!;
			const unindented = (line: string) => (doc.stripTabs ? line.replace(/^\t+/, "") : line);
			const nested = doc.nested || insideSubstitution();
			let hasCommandSubstitution = false;
			for (;;) {
				let end = command.indexOf("\n", i);
				if (end === -1) end = n;
				const line = unindented(command.slice(i, end));
				if (line === doc.delimiter) {
					copy(end + 1);
					break;
				}
				if (line.startsWith(doc.delimiter) && doc.delimiter !== "") return lose("a here-document whose end marker has text after it");
				// bash 3.2 looks for the end of the $( ) in this text too, counting its quotes and parentheses.
				const written = command.slice(i, end).replace(tokenPattern(), "");
				if (nested && /['"()`\\]/.test(written)) return lose("a here-document inside $( ) with quotes or parentheses in its text");

				const lineStart = i;
				const from = out.length;
				let last: Spot | undefined;
				while (i < end) {
					const spot = spots.get(i);
					const c = command[i]!;
					if (spot) {
						const before = command.slice(lineStart, i);
						checkBraces(spot);
						if (nested && /['"()`\\$]/.test(spot.value)) {
							refuse(spot, "it stands in a here-document inside $( ), and its value has quotes, parentheses, $ or backslashes, which bash 3.2 would misread there");
						}
						if (doc.stripTabs && (/\n\t/.test(spot.value) || (spot.value.startsWith("\t") && /^\t*$/.test(before)))) {
							refuse(spot, "its value has a tab at the start of a line, which this here-document (<<-) would remove");
						}
						if (doc.literal) {
							out.push(spot.value);
						} else {
							if (hasCommandSubstitution) {
								refuse(spot, "it stands in a here-document that also runs commands; quote the end marker (<<'EOF') to make the text literal");
							}
							if (VARIABLE.test(before) || SUBSCRIPT.test(before)) {
								refuse(spot, "it follows a variable in a here-document; write the variable as ${NAME}");
							}
							out.push(inHeredoc(spot.value));
						}
						last = spot;
						i = spot.end;
					} else if (doc.literal) {
						copy(i + 1);
					} else if (c === "\\") {
						if (i + 1 >= end) return lose("a line continuation in a here-document");
						if (spots.has(i + 1)) refuse(spots.get(i + 1)!, "it follows a backslash");
						copy("$`\\".includes(command[i + 1]!) ? i + 2 : i + 1);
					} else if (c === "`" || command.startsWith("$(", i)) {
						hasCommandSubstitution = true;
						copy(i + 1);
					} else if (command.startsWith("${", i)) {
						parameter();
					} else if (c === "$" && spots.has(i + 1)) {
						refuse(spots.get(i + 1)!, "it stands right after a $");
					} else {
						copy(i + 1);
					}
				}
				if (last) {
					const lines = out.slice(from).join("").split("\n");
					if (lines.some((one) => unindented(one) === doc.delimiter)) refuse(last, "its value would end the here-document early");
				}
				if (i >= n) return;
				copy(i + 1);
			}
		}
	}

	while (i < n) {
		const frame = frames[frames.length - 1]!;
		if (frame.kind === "plain") plainStep(frame);
		else doubleStep(frame);
	}
	return out.join("");
}
