import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { JsonVectorStore } from "../src/stores/json-store.js";
import type { MemoryRecord } from "../src/types.js";

function record(id: string, scope: "global" | "project" = "global", projectId?: string): MemoryRecord {
  return { id, text: id, kind: "fact", scope, ...(projectId ? { projectId } : {}), confidence: 1, status: "active", source: { actor: "user", sessionId: "s", cwd: "/x", cause: "test", reason: "test fixture" }, createdAt: "2026-01-01", updatedAt: "2026-01-01", embeddingModel: "test", schemaVersion: 1 };
}

async function temporary(testBody: (path: string) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), "active-memory-json-"));
  try { await testBody(join(dir, "db.json")); } finally { await rm(dir, { recursive: true, force: true }); }
}

test("malformed JSON fails closed", () => temporary(async path => {
  await writeFile(path, "not json");
  const before = await readFile(path, "utf8");
  const store = new JsonVectorStore(path);
  await assert.rejects(store.initialize(), /malformed/);
  await assert.rejects(store.insert({ record: record("one"), vector: [1, 0] }), /malformed/);
  await assert.rejects(store.migrateLegacyProvenance(), /malformed/);
  assert.equal(await readFile(path, "utf8"), before);
}));

test("invalid data-file shape fails closed", () => temporary(async path => {
  await writeFile(path, JSON.stringify({ version: 2, rows: [] }));
  const before = await readFile(path, "utf8");
  await assert.rejects(new JsonVectorStore(path).initialize(), /unsupported/);
  assert.equal(await readFile(path, "utf8"), before);
}));

test("malformed record shapes fail closed while v1 provenance omissions migrate", () => temporary(async path => {
  await writeFile(path, JSON.stringify({ version: 1, dimension: 2, rows: [{ record: { ...record("bad"), confidence: "high" }, vector: [1, 0] }] }));
  const before = await readFile(path, "utf8");
  await assert.rejects(new JsonVectorStore(path).migrateLegacyProvenance(), /invalid record shape/);
  assert.equal(await readFile(path, "utf8"), before);
  await writeFile(path, JSON.stringify({ version: 1, dimension: 2, rows: [{ record: { ...record("legacy"), source: {} }, vector: [1, 0] }] }));
  assert.equal(await new JsonVectorStore(path).migrateLegacyProvenance(), 1);
}));

test("empty, whitespace-only, and multiline persisted text fail closed", () => temporary(async path => {
  for (const text of ["", "   ", "first\nsecond", "first\rsecond"]) {
    await writeFile(path, JSON.stringify({ version: 1, dimension: 2, rows: [{ record: { ...record("bad"), text }, vector: [1, 0] }] }));
    await assert.rejects(new JsonVectorStore(path).initialize(), /invalid record shape/);
  }
}));

test("lineage migration removes self references and malformed v1 confidence fails closed", () => temporary(async path => {
  const complete = { ...record("self"), schemaVersion: 2 as const, priority: 1, supersedes: ["self", "older", "older"] };
  await writeFile(path, JSON.stringify({ version: 1, dimension: 2, rows: [{ record: complete, vector: [1, 0] }] }));
  const store = new JsonVectorStore(path);
  await store.initialize();
  assert.equal(await store.migrateLegacyProvenance(), 1);
  assert.deepEqual((await store.get("self"))?.supersedes, ["older"]);
  await writeFile(path, JSON.stringify({ version: 1, dimension: 2, rows: [{ record: { ...record("bad"), confidence: -5 }, vector: [1, 0] }] }));
  await assert.rejects(new JsonVectorStore(path).initialize(), /confidence/);
}));

test("v2 records require complete bounded provenance and dimensionless files infer one dimension", () => temporary(async path => {
  const v2 = { ...record("v2"), schemaVersion: 2, priority: .8, source: { sessionId: "s", cwd: "/", cause: "x", reason: "x" } };
  await writeFile(path, JSON.stringify({ version: 1, rows: [{ record: v2, vector: [1, 0] }] }));
  await assert.rejects(new JsonVectorStore(path).initialize(), /provenance/);
  await writeFile(path, JSON.stringify({ version: 1, rows: [{ record: { ...v2, source: { ...v2.source, actor: "user" } }, vector: [1, 0] }] }));
  await new JsonVectorStore(path).initialize(2);
  await writeFile(path, JSON.stringify({ version: 1, rows: [{ record: { ...v2, source: { ...v2.source, actor: "user" } }, vector: [1, 0] }, { record: { ...v2, id: "other", source: { ...v2.source, actor: "user" } }, vector: [1, 0, 0] }] }));
  await assert.rejects(new JsonVectorStore(path).initialize(), /inconsistent vector dimensions/);
}));

test("initializing an empty JSON store persists the requested dimension", () => temporary(async path => {
  const store = new JsonVectorStore(path);
  await store.initialize(3);
  assert.equal(JSON.parse(await readFile(path, "utf8")).dimension, 3);
  await assert.rejects(store.insert({ record: record("wrong"), vector: [1, 0] }), /inconsistent vector dimensions/);
}));

test("two initialized stores preserve concurrent inserts and readers refresh", () => temporary(async path => {
  const left = new JsonVectorStore(path), right = new JsonVectorStore(path);
  await Promise.all([left.initialize(), right.initialize()]);
  assert.deepEqual(await Promise.all([left.insert({ record: record("left"), vector: [1, 0] }), right.insert({ record: record("right"), vector: [0, 1] })]), ["inserted", "inserted"]);
  assert.deepEqual((await left.list({}, 10)).map(item => item.id).sort(), ["left", "right"]);
}));

test("atomic mutation merges feedback with a text/vector change", () => temporary(async path => {
  const first = new JsonVectorStore(path), second = new JsonVectorStore(path);
  await first.insert({ record: record("one"), vector: [1, 0] });
  await Promise.all([
    first.mutate("one", latest => ({ record: { ...latest, feedback: { useful: 1, unhelpful: 0, lastAt: "2026-01-02", history: [] } } })),
    second.mutate("one", latest => ({ record: { ...latest, text: "changed", embeddingModel: "new", sourceHistory: [latest.source] }, vector: [0, 1] })),
  ]);
  const found = await first.get("one");
  assert.equal(found?.text, "changed");
  assert.equal(found?.feedback?.useful, 1);
  assert.equal((await first.search([0, 1], {}, 1))[0]?.record.id, "one");
}));

test("JSON compaction is one atomic commit and empty rebuild stores dimension", () => temporary(async path => {
  const store = new JsonVectorStore(path);
  await store.insert({ record: record("a"), vector: [1, 0] });
  await store.insert({ record: record("b"), vector: [0, 1] });
  const compacted = await store.compact(["a", "b"], latest => ({ record: { ...record("c"), text: latest.map(row => row.text).join(" ") }, vector: [1, 1] }));
  assert.equal(compacted.status, "active");
  assert.deepEqual((await store.list({}, 10)).filter(row => row.status === "active").map(row => row.id), ["c"]);
  const empty = new JsonVectorStore(join(tmpdir(), `active-memory-empty-${Date.now()}.json`));
  await empty.rebuildVectors(3, async () => []);
  await assert.doesNotReject(empty.initialize(3));
}));
