export const KINDS = ["SECRET", "EMAIL", "PERSON", "PHONE", "ADDRESS", "ACCOUNT", "URL", "DATE", "LITERAL"] as const;
export type Kind = (typeof KINDS)[number];

const TOKEN_SOURCE = `<(${KINDS.join("|")})_([1-9][0-9]{0,5})>`;
const WHOLE_TOKEN = new RegExp(`^${TOKEN_SOURCE}$`);

export function tokenPattern(): RegExp {
	return new RegExp(TOKEN_SOURCE, "g");
}

export interface Entry {
	readonly token: string;
	readonly kind: Kind;
	readonly value: string;
	revealed: boolean;
}

export class Vault {
	readonly entries: Entry[] = [];
	onAllocate?: (entry: Entry) => void;

	private readonly byToken = new Map<string, Entry>();
	private readonly byValue = new Map<string, Entry>();
	private readonly highest = new Map<Kind, number>();
	private _version = 0;

	get version(): number {
		return this._version;
	}

	lookup(token: string): Entry | undefined {
		return this.byToken.get(token);
	}

	find(value: string): Entry | undefined {
		return this.byValue.get(value);
	}

	allocate(kind: Kind, value: string): Entry {
		const existing = this.byValue.get(value);
		if (existing) return existing;
		const entry = this.add(kind, (this.highest.get(kind) ?? 0) + 1, value);
		this.onAllocate?.(entry);
		return entry;
	}

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
		this._version++;
		return true;
	}

	private add(kind: Kind, number: number, value: string): Entry {
		const entry: Entry = { token: `<${kind}_${number}>`, kind, value, revealed: false };
		this.entries.push(entry);
		this.byToken.set(entry.token, entry);
		this.byValue.set(value, entry);
		this.highest.set(kind, Math.max(this.highest.get(kind) ?? 0, number));
		this._version++;
		return entry;
	}
}
