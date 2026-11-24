import assert from "node:assert/strict";
import { test } from "node:test";

import { configPath, defaults, isLocalModel, parseConfig } from "../src/config.ts";

const HOME = "/Users/someone";

test("without a file the screen masks, with the detector beside the default checkpoint", () => {
	assert.deepEqual(defaults(HOME), {
		mode: "mask",
		labels: ["secret", "private_email", "private_phone", "private_person", "private_address", "account_number"],
		screenLocalModels: false,
		localProviders: [],
		envFiles: [".env", ".env.local", ".env.*"],
		server: { socket: "/Users/someone/.opf/opf-mlx.sock", command: undefined, autoStart: true, idleExit: 1800, startTimeout: 60 },
		shellTools: { bash: ["command"] },
		codeTools: ["codemode", "powershell"],
	});
	assert.deepEqual(parseConfig(undefined, HOME), { config: defaults(HOME), problems: [] });
});

test("a file changes what it names and leaves the rest", () => {
	const { config, problems } = parseConfig(
		JSON.stringify({
			mode: "report",
			labels: ["secret", "private_url"],
			screenLocalModels: true,
			localProviders: ["ollama"],
			envFiles: [],
			server: { socket: "~/run/opf.sock", command: ["/opt/opf/bin/opf-mlx", "serve"], idleExit: 60 },
			shellTools: { bash: ["command"], ssh: ["command", "script"] },
			codeTools: [],
		}),
		HOME,
	);

	assert.deepEqual(problems, []);
	assert.deepEqual(config, {
		mode: "report",
		labels: ["secret", "private_url"],
		screenLocalModels: true,
		localProviders: ["ollama"],
		envFiles: [],
		server: { socket: "/Users/someone/run/opf.sock", command: ["/opt/opf/bin/opf-mlx", "serve"], autoStart: true, idleExit: 60, startTimeout: 60 },
		shellTools: { bash: ["command"], ssh: ["command", "script"] },
		codeTools: [],
	});
});

test("a file that is not JSON is reported and nothing in it is used", () => {
	const { config, problems } = parseConfig('{ "mode": "off", }', HOME);

	assert.deepEqual(config, defaults(HOME));
	assert.equal(problems.length, 1);
	assert.match(problems[0]!, /not valid JSON/);
});

test("a file that is not an object is reported", () => {
	assert.match(parseConfig('["mask"]', HOME).problems[0]!, /must be a JSON object/);
});

test("each setting that is wrong is reported and keeps its default; the others apply", () => {
	const { config, problems } = parseConfig(
		JSON.stringify({
			mode: "of",
			labels: ["secret", "passwords"],
			screenLocalModels: "yes",
			localProviders: "ollama",
			envFiles: [".env", 3],
			server: { socket: "", command: [], autoStart: 1, idleExit: -5, startTimeout: 0, port: 8000 },
			shellTools: { bash: "command" },
			codeTools: [""],
			colour: "blue",
		}),
		HOME,
	);

	assert.deepEqual(config, defaults(HOME));
	assert.deepEqual(
		problems.map((problem) => problem.split(":")[0]),
		[
			"mode",
			"labels",
			"screenLocalModels",
			"localProviders",
			"envFiles",
			"server.socket",
			"server.command",
			"server.autoStart",
			"server.idleExit",
			"server.startTimeout",
			"server.port",
			"shellTools",
			"codeTools",
			"colour",
		],
	);
	assert.match(problems[0]!, /"mask", "report" or "off"/);
	assert.match(problems[1]!, /passwords.*private_date/);
	assert.match(problems.at(-1)!, /unknown setting/);
});

test("a wrong mode never switches the screen off", () => {
	for (const mode of ["OFF", "disabled", false, null, 0]) {
		assert.equal(parseConfig(JSON.stringify({ mode }), HOME).config.mode, "mask");
	}
});

test("one good setting next to a bad one still applies", () => {
	const { config, problems } = parseConfig(JSON.stringify({ labels: ["secret", "private_date"], mode: 7 }), HOME);

	assert.deepEqual(config.labels, ["secret", "private_date"]);
	assert.equal(problems.length, 1);
});

test("the file is looked for in Pi's agent directory, unless the environment names one", () => {
	assert.equal(configPath({}, HOME), "/Users/someone/.pi/agent/privacy-screen.json");
	assert.equal(configPath({ PI_CODING_AGENT_DIR: "/tmp/agent" }, HOME), "/tmp/agent/privacy-screen.json");
	assert.equal(configPath({ PI_CODING_AGENT_DIR: "~/alt" }, HOME), "/Users/someone/alt/privacy-screen.json");
	assert.equal(configPath({ PI_CODING_AGENT_DIR: "/tmp/agent", PI_PRIVACY_SCREEN_CONFIG: "~/p.json" }, HOME), "/Users/someone/p.json");
});

test("a model counts as local only on this machine, or when its provider is listed", () => {
	const config = { ...defaults(HOME), localProviders: ["lan-box"] };
	const local = (baseUrl: string | undefined, provider = "x") => isLocalModel({ provider, baseUrl }, config);

	for (const url of ["http://localhost:11434/v1", "http://127.0.0.1:8080", "http://127.8.9.10/v1", "http://[::1]:1234/v1", "HTTP://LOCALHOST:1"]) {
		assert.equal(local(url), true, url);
	}
	for (const url of [
		"https://api.anthropic.com",
		"http://192.168.1.20:11434",
		"http://localhost.example.com/v1",
		"http://127.0.0.1.example.com/v1",
		"http://example.com/?x=localhost",
		"http://0.0.0.0:8080",
		"not a url",
		"",
		undefined,
	]) {
		assert.equal(local(url), false, String(url));
	}
	assert.equal(local("http://192.168.1.20:11434", "lan-box"), true);
	assert.equal(isLocalModel(undefined, config), false);
});
