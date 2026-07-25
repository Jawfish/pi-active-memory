import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { JsonVectorStore } from "../src/stores/json-store.js";
import type { MemoryRecord } from "../src/types.js";

function record(id: string, scope: "global" | "project", projectId?: string): MemoryRecord {
  return { id, text: id, kind: "fact", scope, ...(projectId ? { projectId } : {}), confidence: 1, status: "active", source: { actor: "user", sessionId: "s", cwd: "/x", cause: "test", reason: "test fixture" }, createdAt: "2026-01-01", updatedAt: "2026-01-01", embeddingModel: "test", schemaVersion: 1 };
}

test("JSON store migrates legacy records to complete provenance", async () => {
  const dir = await mkdtemp(join(tmpdir(), "active-memory-provenance-"));
  try {
    const path = join(dir, "db.json");
    const legacy = { ...record("legacy", "global"), source: { sessionId: "old-session", cwd: "/old" }, priority: undefined, schemaVersion: 1 };
    await writeFile(path, JSON.stringify({ version: 1, dimension: 2, rows: [{ record: legacy, vector: [1, 0] }] }));
    const store = new JsonVectorStore(path);
    assert.equal(await store.migrateLegacyProvenance(), 1);
    const migrated = (await store.list({}, 10))[0]!;
    assert.equal(migrated.source.actor, "user");
    assert.equal(migrated.source.sessionId, "old-session");
    assert.equal(migrated.source.cause, "legacy_memory_migration");
    assert.match(migrated.source.reason, /before cause and storage rationale/);
    assert.equal(migrated.priority, 1);
    assert.equal(migrated.schemaVersion, 2);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("JSON store performs vector search and scope filtering", async () => {
  const dir = await mkdtemp(join(tmpdir(), "active-memory-"));
  try {
    const store = new JsonVectorStore(join(dir, "db.json"));
    await store.upsert(record("global", "global"), [1, 0]);
    await store.upsert(record("project-a", "project", "a"), [0.9, 0.1]);
    await store.upsert(record("project-b", "project", "b"), [1, 0]);
    await store.upsert({ ...record("legacy", "global"), kind: "decision" } as unknown as MemoryRecord, [1, 0]);
    const found = await store.search([1, 0], { status: "active", scopes: ["global", "project"], kinds: ["user_profile", "fact", "skill_workflow"], projectId: "a" }, 10);
    assert.deepEqual(found.map((m) => m.record.id), ["global", "project-a"]);
    assert.equal(await store.markDeleted("global"), true);
    assert.deepEqual((await store.list({ status: "active", kinds: ["user_profile", "fact", "skill_workflow"] }, 10)).map((m) => m.id).sort(), ["project-a", "project-b"]);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
