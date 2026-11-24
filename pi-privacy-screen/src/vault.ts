/**
 * The placeholder map: one numbered placeholder per real value, both ways.
 *
 * `<SECRET_1>` is what the model provider sees; the value it stands for never
 * leaves this machine. A value keeps its placeholder for the whole session, so
 * the model can refer to it consistently and the text sent on every turn stays
 * the same.
 */

export const KINDS = ["SECRET", "EMAIL", "PERSON", "PHONE", "ADDRESS", "ACCOUNT", "URL", "DATE", "LITERAL"] as const;
export type Kind = (typeof KINDS)[number];

const TOKEN_SOURCE = `<(${KINDS.join("|")})_([1-9][0-9]{0,5})>`;
const WHOLE_TOKEN = new RegExp(`^${TOKEN_SOURCE}$`);

/** A new pattern each time: a global pattern keeps its position between uses. */
export function tokenPattern(): RegExp {
	return new RegExp(TOKEN_SOURCE, "g");
}

export interface Entry {
	readonly token: string;
	readonly kind: Kind;
	readonly value: string;
	/** The user chose to let the model see this value: it is no longer replaced on the way out. */
	revealed: boolean;
}

export class Vault {
	/** Every entry, oldest first. Its length is the version that caches compare against. */
	readonly entries: Entry[] = [];
	/** Called once for each newly created placeholder, so that it can be saved with the session. */
	onAllocate?: (entry: Entry) => void;

	private readonly byToken = new Map<string, Entry>();
	private readonly byValue = new Map<string, Entry>();
	private readonly highest = new Map<Kind, number>();

	lookup(token: string): Entry | undefined {
		return this.byToken.get(token);
	}

	find(value: string): Entry | undefined {
		return this.byValue.get(value);
	}

	/** The placeholder for `value`: the one it already has, else the next number of `kind`. */
	allocate(kind: Kind, value: string): Entry {
		const existing = this.byValue.get(value);
		if (existing) return existing;
		const entry = this.add(kind, (this.highest.get(kind) ?? 0) + 1, value);
		this.onAllocate?.(entry);
		return entry;
	}

	/** Load a pair saved earlier. False if it is malformed or contradicts what is already loaded. */
	restore(token: string, value: string, revealed = false): boolean {
		const match = WHOLE_TOKEN.exec(token);
		if (!match || typeof value !== "string" || value === "") return false;
		const byToken = this.byToken.get(token);
		const byValue = this.byValue.get(value);
		if (byToken || byValue) {
			if (byToken !== byValue) return false;
			byToken!.revealed = byToken!.revealed || revealed;
			return true;
		}
		this.add(match[1] as Kind, Number(match[2]), value).revealed = revealed;
		return true;
	}

	setRevealed(token: string, revealed: boolean): boolean {
		const entry = this.byToken.get(token);
		if (!entry) return false;
		entry.revealed = revealed;
		return true;
	}

	private add(kind: Kind, number: number, value: string): Entry {
		const entry: Entry = { token: `<${kind}_${number}>`, kind, value, revealed: false };
		this.entries.push(entry);
		this.byToken.set(entry.token, entry);
		this.byValue.set(value, entry);
		this.highest.set(kind, Math.max(this.highest.get(kind) ?? 0, number));
		return entry;
	}
}
