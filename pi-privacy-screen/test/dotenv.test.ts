import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";

import { readSeeds, seedsFrom, selectEnvFiles } from "../src/dotenv.ts";

const scratch = mkdtempSync(join(tmpdir(), "pi-privacy-env-"));
after(() => rmSync(scratch, { recursive: true, force: true }));

test("the values that look like secrets are taken from a .env file", () => {
	const seeds = seedsFrom(
		[
			"# database",
			"DB_PASSWORD=correct-horse-battery",
			'export API_TOKEN="tok_9f8e7d6c5b4a"',
			"SESSION_SECRET='keep it quiet, please'",
			"SIGNING_SALT=Zx81kq0pLm42vv # generated",
			"  STRIPE_KEY = sk_live_51Habcdefghijklmnop",
			"",
		].join("\n"),
	);

	assert.deepEqual(seeds.sort(), ["Zx81kq0pLm42vv", "correct-horse-battery", "keep it quiet, please", "sk_live_51Habcdefghijklmnop", "tok_9f8e7d6c5b4a"]);
});

test("a quoted value with escapes is taken as written and as it reads", () => {
	assert.deepEqual(seedsFrom('APP_SECRET="line one\\nline \\"two\\""\n').sort(), ['line one\nline "two"', 'line one\\nline \\"two\\"'].sort());
});

test("ordinary settings are not taken", () => {
	const seeds = seedsFrom(
		[
			"NODE_ENV=development",
			"PORT=3000",
			"DEBUG=true",
			"HOST=localhost",
			"LOG_DIR=/var/log/app2",
			"API_URL=https://api.example.com/v2",
			"REDIS=redis.internal.example:6379",
			"PASSWORD=short",
			"API_KEY=$OTHER_KEY",
			"TOKEN=${VAULT_TOKEN}",
			"SECRET=",
			"NEXT_PUBLIC_API_KEY=public-browser-key",
			"TOKEN_LIFETIME=3600",
			"TOKEN_TTL_MS=86400000",
			"PRIVATE_KEY_PATH=/etc/keys/app.pem",
			"RANDOM_THING=Zx81kq0pLm42vv",
			"AWS_REGION=us-east-1",
			"BUCKET=my-bucket-2024",
			'SIGNING="unterminated',
			"not an assignment at all",
		].join("\n"),
	);

	assert.deepEqual(seeds, []);
});

test("of a connection string only the password is taken", () => {
	assert.deepEqual(seedsFrom("DATABASE_URL=postgres://app:s3cr3t-pass@db.example.com:5432/app\n"), ["s3cr3t-pass"]);
});

test("key formats are found anywhere in the file, also in a value that spans lines", () => {
	const pem = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----";
	const seeds = seedsFrom(`NOTE="see AKIAIOSFODNN7EXAMPLE"\nSIGNING="${pem}"\n`);

	assert.deepEqual(seeds.sort(), ["AKIAIOSFODNN7EXAMPLE", pem].sort());
});

test("the files are picked by name: real ones, not the examples that get committed", () => {
	const names = [".env", ".env.local", ".env.production", ".env.example", ".env.sample", ".env.template", ".env.dist", ".envrc", "env", "x.env", ".env.EXAMPLE"];

	assert.deepEqual(selectEnvFiles(names, [".env", ".env.local", ".env.*"]), [".env", ".env.local", ".env.production"]);
	assert.deepEqual(selectEnvFiles(names, ["*.env"]), ["x.env"]);
	assert.deepEqual(selectEnvFiles(names, []), []);
	assert.deepEqual(selectEnvFiles(names, ["../.env", "sub/.env", ".env"]), [".env"], "only names in the working directory");
});

test("the files are read from the working directory", async () => {
	const dir = join(scratch, "project");
	mkdirSync(join(dir, ".env.d"), { recursive: true });
	writeFileSync(join(dir, ".env"), "DB_PASSWORD=correct-horse-battery\n");
	writeFileSync(join(dir, ".env.local"), "API_TOKEN=tok_9f8e7d6c5b4a\nDB_PASSWORD=correct-horse-battery\n");
	writeFileSync(join(dir, ".env.example"), "DB_PASSWORD=example-password-1\n");
	writeFileSync(join(dir, ".env.huge"), `BIG_SECRET=${"x1".repeat(600_000)}\n`);
	writeFileSync(join(scratch, "outside"), "OUTSIDE_TOKEN=tok_outside_123456\n");
	symlinkSync(join(scratch, "outside"), join(dir, ".env.link"));

	const found = await readSeeds(dir, [".env", ".env.local", ".env.*"]);

	assert.deepEqual(found.seeds.sort(), ["correct-horse-battery", "tok_9f8e7d6c5b4a", "tok_outside_123456"]);
	assert.deepEqual(found.files.sort(), [".env", ".env.link", ".env.local"]);
	assert.deepEqual(found.skipped, [".env.huge: larger than 1 MiB"]);
});

test("a directory that cannot be read gives nothing, without failing", async () => {
	assert.deepEqual(await readSeeds(join(scratch, "missing"), [".env"]), { seeds: [], files: [], skipped: [] });
});
