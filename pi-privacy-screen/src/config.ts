import { join } from "node:path";

import { ALL_LABELS, DEFAULT_LABELS } from "./spans.ts";

export type Mode = "mask" | "report" | "off";
export const MODES: readonly Mode[] = ["mask", "report", "off"];

export interface ServerConfig {
	socket: string;
	command: string[] | undefined;
	autoStart: boolean;
	idleExit: number;
	startTimeout: number;
}

export interface Config {
	mode: Mode;
	labels: string[];
	screenLocalModels: boolean;
	localProviders: string[];
	envFiles: string[];
	server: ServerConfig;
	shellTools: Record<string, string[]>;
	codeTools: string[];
}

export function defaults(home: string): Config {
	return {
		mode: "mask",
		labels: [...DEFAULT_LABELS],
		screenLocalModels: false,
		localProviders: [],
		envFiles: [".env", ".env.local", ".env.*"],
		server: { socket: join(home, ".opf", "opf-mlx.sock"), command: undefined, autoStart: true, idleExit: 1800, startTimeout: 60 },
		shellTools: { bash: ["command"] },
		codeTools: ["codemode", "powershell"],
	};
}

function expand(path: string, home: string): string {
	return path === "~" ? home : path.startsWith("~/") ? join(home, path.slice(2)) : path;
}

export function configPath(env: Record<string, string | undefined>, home: string): string {
	if (env.PI_PRIVACY_SCREEN_CONFIG) return expand(env.PI_PRIVACY_SCREEN_CONFIG, home);
	const agentDir = env.PI_CODING_AGENT_DIR ? expand(env.PI_CODING_AGENT_DIR, home) : join(home, ".pi", "agent");
	return join(agentDir, "privacy-screen.json");
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const isNames = (value: unknown): value is string[] => Array.isArray(value) && value.every((one) => typeof one === "string" && one !== "");
const isSeconds = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value > 0;

export function parseConfig(text: string | undefined, home: string): { config: Config; problems: string[] } {
	const config = defaults(home);
	const problems: string[] = [];
	if (text === undefined) return { config, problems };

	let file: unknown;
	try {
		file = JSON.parse(text);
	} catch (error) {
		return { config, problems: [`the file is not valid JSON (${(error as Error).message}); nothing in it is used`] };
	}
	if (!isRecord(file)) return { config, problems: ["the file must be a JSON object; nothing in it is used"] };

	const take = <V>(holder: Record<string, unknown>, key: string, name: string, valid: (value: unknown) => value is V, expected: string, set: (value: V) => void) => {
		if (!(key in holder)) return;
		if (valid(holder[key])) set(holder[key] as V);
		else problems.push(`${name}: must be ${expected}; the default is used`);
	};
	const isMode = (value: unknown): value is Mode => MODES.includes(value as Mode);
	const isLabels = (value: unknown): value is string[] => isNames(value) && value.every((label) => ALL_LABELS.includes(label));
	const isBoolean = (value: unknown): value is boolean => typeof value === "boolean";
	const isPath = (value: unknown): value is string => typeof value === "string" && value !== "";
	const isCommand = (value: unknown): value is string[] => isNames(value) && value.length > 0;
	const isToolArguments = (value: unknown): value is Record<string, string[]> => isRecord(value) && Object.values(value).every(isNames);

	const settings: Record<string, () => void> = {
		mode: () => take(file, "mode", "mode", isMode, '"mask", "report" or "off"', (value) => (config.mode = value)),
		labels: () =>
			take(file, "labels", "labels", isLabels, `a list of labels (${JSON.stringify(file.labels)} is not: choose from ${ALL_LABELS.join(", ")})`, (value) => (config.labels = value)),
		screenLocalModels: () => take(file, "screenLocalModels", "screenLocalModels", isBoolean, "true or false", (value) => (config.screenLocalModels = value)),
		localProviders: () => take(file, "localProviders", "localProviders", isNames, "a list of provider names", (value) => (config.localProviders = value)),
		envFiles: () => take(file, "envFiles", "envFiles", isNames, "a list of file names", (value) => (config.envFiles = value)),
		server: () => {
			const server = file.server;
			if (!isRecord(server)) return void problems.push("server: must be an object; the default is used");
			const fields: Record<string, () => void> = {
				socket: () => take(server, "socket", "server.socket", isPath, "a path", (value) => (config.server.socket = expand(value, home))),
				command: () => take(server, "command", "server.command", isCommand, "a command as a list of arguments", (value) => (config.server.command = value)),
				autoStart: () => take(server, "autoStart", "server.autoStart", isBoolean, "true or false", (value) => (config.server.autoStart = value)),
				idleExit: () => take(server, "idleExit", "server.idleExit", isSeconds, "a number of seconds above 0", (value) => (config.server.idleExit = value)),
				startTimeout: () => take(server, "startTimeout", "server.startTimeout", isSeconds, "a number of seconds above 0", (value) => (config.server.startTimeout = value)),
			};
			for (const key of Object.keys(server)) (fields[key] ?? (() => problems.push(`server.${key}: unknown setting; it is ignored`)))();
		},
		shellTools: () =>
			take(file, "shellTools", "shellTools", isToolArguments, "an object of tool names, each with a list of argument names", (value) => (config.shellTools = value)),
		codeTools: () => take(file, "codeTools", "codeTools", isNames, "a list of tool names", (value) => (config.codeTools = value)),
	};
	for (const key of Object.keys(file)) (settings[key] ?? (() => problems.push(`${key}: unknown setting; it is ignored`)))();
	return { config, problems };
}

export function isLocalModel(model: { provider?: string; baseUrl?: string } | undefined, config: Pick<Config, "localProviders">): boolean {
	if (!model) return false;
	if (model.provider !== undefined && config.localProviders.includes(model.provider)) return true;
	if (!model.baseUrl) return false;
	let host: string;
	try {
		host = new URL(model.baseUrl).hostname.toLowerCase();
	} catch {
		return false;
	}
	return host === "localhost" || host === "[::1]" || /^127(\.\d{1,3}){3}$/.test(host);
}
