import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { beginEmbeddingMigration, completeEmbeddingMigration, embeddingStoreKey, legacyEmbeddingStoreKey, readEmbeddingGeneration, writeEmbeddingModels } from "../src/embedding-metadata.js";
import { guardEmbeddingGeneration } from "../src/embedding-generation.js";
import { withStorageLock } from "../src/storage-lock.js";
import { ensureEmbeddingCompatibility } from "../src/index.js";
import type { EmbeddingProvider, MemoryRecord, VectorStore } from "../src/types.js";

function store(onSearch: () => void): VectorStore {
  return {
    contractVersion: 2, initialize: async () => {}, get: async () => undefined, insert: async () => "exists", mutate: async () => ({ status: "missing" }), compact: async () => { throw new Error("unused"); }, scan: async () => 0, rebuildVectors: async () => 0,
    search: async () => { onSearch(); return []; }, list: async () => [], migrateLegacyProvenance: async () => 0, close: async () => {},
  };
}

test("unguarded adapter methods preserve their private receiver", async () => {
  class PrivateStore {
    readonly contractVersion = 2 as const;
    #closed = false;
    initialize = async () => {};
    get = async () => undefined;
    insert = async () => "exists" as const;
    mutate = async () => ({ status: "missing" as const });
    compact = async () => { throw new Error("unused"); };
    scan = async () => 0;
    rebuildVectors = async () => 0;
    search = async () => [];
    list = async () => [];
    migrateLegacyProvenance = async () => 0;
    async close() { this.#closed = true; }
    closed() { return this.#closed; }
  }
  const raw = new PrivateStore();
  const guarded = guardEmbeddingGeneration(raw, "/unused", "store", { query: "q", document: "d" });
  await guarded.close();
  assert.equal(raw.closed(), true);
});

test("pending embedding generations fence old sessions and recover a published target", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-generation-"));
  const path = join(directory, "models.json");
  const old = { query: "q1", document: "d1" }, next = { query: "q2", document: "d2" };
  let searched = 0;
  try {
    await writeEmbeddingModels(path, "store", old);
    const oldSession = guardEmbeddingGeneration(store(() => searched++), path, "store", old);
    await beginEmbeddingMigration(path, "store", old, next);
    await assert.rejects(oldSession.search([1], {}, 1), /migrating/);
    assert.equal(searched, 0, "a fenced old session must not reach raw search");
    // Models the crash window after a vector rebuild published d2 but before metadata completion.
    assert.deepEqual((await readEmbeddingGeneration(path, "store"))?.pending, next);
    await completeEmbeddingMigration(path, "store");
    assert.deepEqual(await readEmbeddingGeneration(path, "store"), { current: next });
    await assert.rejects(oldSession.search([1], {}, 1), /generation changed/);
    const resumedSession = guardEmbeddingGeneration(store(() => searched++), path, "store", next);
    await resumedSession.search([1], {}, 1);
    assert.equal(searched, 1, "only the resumed generation may reach raw search");
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("pending recovery rebuilds an empty store before publishing its new dimension", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-empty-recovery-"));
  const path = join(directory, "models.json");
  const current = { query: "q-old", document: "d-old" }, desired = { query: "q-new", document: "d-new" };
  const rag = { adapter: "json", config: { path: "/empty-memory" } };
  const key = embeddingStoreKey(rag);
  let rebuilds = 0;
  const raw = store(() => {});
  raw.rebuildVectors = async dimension => { assert.equal(dimension, 2); rebuilds++; return 0; };
  const embedder: EmbeddingProvider = { queryModel: desired.query, documentModel: desired.document, embedQuery: async () => [[1, 0]], embedDocuments: async () => [[1, 0]] };
  const ctx = { hasUI: false, ui: { notify() {}, setWorkingMessage() {}, confirm: async () => false } };
  try {
    await writeEmbeddingModels(path, key, current);
    await beginEmbeddingMigration(path, key, current, desired);
    assert.equal(await ensureEmbeddingCompatibility(ctx as never, raw, embedder, rag, path), true);
    assert.equal(rebuilds, 1);
    assert.deepEqual(await readEmbeddingGeneration(path, key), { current: desired });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("canonical keys import legacy asymmetric metadata without rebuilding", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-legacy-key-"));
  const path = join(directory, "models.json");
  const desired = { query: "q-asymmetric", document: "d-asymmetric" };
  const priorRag = { adapter: "qdrant", config: { url: "http://qdrant", collection: "mem", apiKeyEnv: "OLD_TOKEN" } };
  const rag = { adapter: "qdrant", config: { url: "HTTP://QDRANT:80/", collection: "mem", apiKeyEnv: "NEW_TOKEN" } };
  const key = embeddingStoreKey(rag), legacyKey = legacyEmbeddingStoreKey(priorRag);
  let rebuilds = 0;
  const raw = store(() => {});
  raw.scan = async (_filter, visit) => { await visit([{ id: "one", embeddingModel: desired.document } as MemoryRecord]); return 1; };
  raw.rebuildVectors = async () => { rebuilds++; return 1; };
  const embedder: EmbeddingProvider = { queryModel: desired.query, documentModel: desired.document, embedQuery: async () => [[1]], embedDocuments: async () => [[1]] };
  const ctx = { hasUI: false, ui: { notify() {}, setWorkingMessage() {}, confirm: async () => false } };
  try {
    assert.notEqual(key, legacyKey);
    assert.notEqual(legacyEmbeddingStoreKey(rag), legacyKey);
    await writeEmbeddingModels(path, legacyKey, desired);
    assert.equal(await ensureEmbeddingCompatibility(ctx as never, raw, embedder, rag, path), true);
    assert.equal(rebuilds, 0);
    assert.deepEqual(await readEmbeddingGeneration(path, key), { current: desired });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("compatibility never rewrites unrelated store metadata", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-unrelated-key-"));
  const path = join(directory, "models.json");
  const old = { query: "q-old", document: "d-old" }, desired = { query: "q-new", document: "d-new" };
  const rag = { adapter: "json", config: { path: "/current-memory" } };
  const raw = store(() => {});
  raw.scan = async (_filter, visit) => { await visit([{ id: "one", text: "one", embeddingModel: old.document } as MemoryRecord]); return 1; };
  raw.rebuildVectors = async (_dimension, build) => { await build([{ id: "one", text: "one", embeddingModel: old.document } as MemoryRecord]); return 1; };
  const embedder: EmbeddingProvider = { queryModel: desired.query, documentModel: desired.document, embedQuery: async () => [[1]], embedDocuments: async texts => texts.map(() => [1]) };
  const ctx = { hasUI: true, ui: { notify() {}, setWorkingMessage() {}, confirm: async () => true } };
  try {
    await writeEmbeddingModels(path, "unrelated-store", old);
    assert.equal(await ensureEmbeddingCompatibility(ctx as never, raw, embedder, rag, path), true);
    assert.deepEqual(await readEmbeddingGeneration(path, "unrelated-store"), { current: old });
    assert.deepEqual(await readEmbeddingGeneration(path, embeddingStoreKey(rag)), { current: desired });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("an unrelated asymmetric generation cannot influence a store with matching document vectors", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-unrelated-asymmetric-"));
  const path = join(directory, "models.json");
  const desired = { query: "q-current", document: "shared-document" };
  const unrelated = { query: "q-private-unrelated", document: desired.document };
  const rag = { adapter: "json", config: { path: "/isolated-memory" } };
  let confirmations = 0, rebuilds = 0;
  const raw = store(() => {});
  raw.scan = async (_filter, visit) => { await visit([{ id: "one", text: "one", embeddingModel: desired.document } as MemoryRecord]); return 1; };
  raw.rebuildVectors = async () => { rebuilds++; return 1; };
  const embedder: EmbeddingProvider = { queryModel: desired.query, documentModel: desired.document, embedQuery: async () => [[1]], embedDocuments: async () => [[1]] };
  const ctx = { hasUI: true, ui: { notify() {}, setWorkingMessage() {}, confirm: async () => { confirmations++; return false; } } };
  try {
    await writeEmbeddingModels(path, "unrelated-store", unrelated);
    assert.equal(await ensureEmbeddingCompatibility(ctx as never, raw, embedder, rag, path), true);
    assert.equal(confirmations, 0);
    assert.equal(rebuilds, 0);
    assert.deepEqual(await readEmbeddingGeneration(path, "unrelated-store"), { current: unrelated });
    assert.deepEqual(await readEmbeddingGeneration(path, embeddingStoreKey(rag)), { current: desired });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("compatibility never adopts another store's pending target", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-unrelated-pending-"));
  const path = join(directory, "models.json");
  const old = { query: "q-old", document: "d-old" }, desired = { query: "q-new", document: "d-new" };
  const rag = { adapter: "json", config: { path: "/pending-isolation" } };
  let rebuilds = 0, confirmations = 0;
  const raw = store(() => {});
  raw.scan = async (_filter, visit) => { await visit([{ id: "one", text: "one", embeddingModel: old.document } as MemoryRecord]); return 1; };
  raw.rebuildVectors = async () => { rebuilds++; return 1; };
  const embedder: EmbeddingProvider = { queryModel: desired.query, documentModel: desired.document, embedQuery: async () => [[1]], embedDocuments: async () => [[1]] };
  const ctx = { hasUI: true, ui: { notify() {}, setWorkingMessage() {}, confirm: async () => { confirmations++; return false; } } };
  try {
    await writeEmbeddingModels(path, "unrelated-store", old);
    await beginEmbeddingMigration(path, "unrelated-store", old, desired);
    assert.equal(await ensureEmbeddingCompatibility(ctx as never, raw, embedder, rag, path), false);
    assert.equal(confirmations, 1);
    assert.equal(rebuilds, 0);
    assert.deepEqual(await readEmbeddingGeneration(path, "unrelated-store"), { current: old, pending: desired });
    assert.equal(await readEmbeddingGeneration(path, embeddingStoreKey(rag)), undefined);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("an empty store ignores unrelated legacy generations", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-empty-unrelated-"));
  const path = join(directory, "models.json");
  const desired = { query: "q-new", document: "d-new" };
  const rag = { adapter: "json", config: { path: "/new-empty-memory" } };
  const raw = store(() => {});
  const embedder: EmbeddingProvider = { queryModel: desired.query, documentModel: desired.document, embedQuery: async () => [[1]], embedDocuments: async () => [[1]] };
  const ctx = { hasUI: false, ui: { notify() {}, setWorkingMessage() {}, confirm: async () => false } };
  try {
    await writeEmbeddingModels(path, "unrelated-store", { query: "other-q", document: "other-d" });
    assert.equal(await ensureEmbeddingCompatibility(ctx as never, raw, embedder, rag, path), true);
    assert.deepEqual(await readEmbeddingGeneration(path, "unrelated-store"), { current: { query: "other-q", document: "other-d" } });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("metadata writes preserve unrelated legacy entry representation", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-legacy-shape-"));
  const path = join(directory, "models.json");
  const legacy = { query: "legacy-q", document: "legacy-d" };
  try {
    await writeFile(path, JSON.stringify({ version: 1, stores: { unrelated: legacy } }));
    await writeEmbeddingModels(path, "target", { query: "new-q", document: "new-d" });
    const raw = JSON.parse(await readFile(path, "utf8")) as { stores: Record<string, unknown> };
    assert.deepEqual(raw.stores.unrelated, legacy);
    assert.deepEqual(raw.stores.target, { current: { query: "new-q", document: "new-d" } });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("custom store identities ignore credentials and malformed legacy metadata fails closed", async () => {
  const first = embeddingStoreKey({ adapter: "custom", config: { storeIdentity: "shared", token: "one" } });
  const second = embeddingStoreKey({ adapter: "custom", config: { storeIdentity: "shared", token: "two", tuning: 3 } });
  assert.equal(first, second);
  assert.throws(() => embeddingStoreKey({ adapter: "custom", config: { token: "one" } }), /storeIdentity/);
  const directory = await mkdtemp(join(tmpdir(), "active-memory-malformed-metadata-"));
  const path = join(directory, "models.json");
  try {
    await writeFile(path, JSON.stringify({ version: 1, stores: { bad: { query: 123, document: null } } }));
    await assert.rejects(readEmbeddingGeneration(path, "bad"), /invalid model identities|hybrid generation shape/);
    await writeFile(path, JSON.stringify({ version: 1, stores: { bad: { query: "q", document: "d", pending: { query: "next-q", document: "next-d" } } } }));
    await assert.rejects(readEmbeddingGeneration(path, "bad"), /hybrid generation shape/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("embedding compatibility never displays opaque adapter failures", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-secret-error-"));
  const path = join(directory, "models.json");
  const raw = store(() => {});
  raw.scan = async () => { throw new Error("adapter secret=abc"); };
  const embedder: EmbeddingProvider = { model: "desired", embed: async () => [[1]] };
  const notifications: string[] = [];
  const ctx = { hasUI: false, ui: { notify: (message: string) => notifications.push(message), setWorkingMessage() {}, confirm: async () => false } };
  try {
    assert.equal(await ensureEmbeddingCompatibility(ctx as never, raw, embedder, { adapter: "custom", config: { storeIdentity: "secret-test" } }, path), false);
    assert.doesNotMatch(notifications.join("\n"), /secret=abc/);
    assert.match(notifications.join("\n"), /compatibility failed/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("superseded compatibility scans cannot prompt or rebuild", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-stale-compatibility-"));
  let release!: () => void, started!: () => void, current = true, rebuilds = 0, confirms = 0;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const scanning = new Promise<void>(resolve => { started = resolve; });
  const raw = store(() => {});
  raw.scan = async (_filter, visit) => { started(); await gate; await visit([{ id: "one", embeddingModel: "old" } as MemoryRecord]); return 1; };
  raw.rebuildVectors = async () => { rebuilds++; return 1; };
  const notifications: string[] = [];
  const ctx = { hasUI: true, ui: { notify: (message: string) => notifications.push(message), setWorkingMessage() {}, confirm: async () => { confirms++; return true; } } };
  const embedder: EmbeddingProvider = { model: "new", embed: async () => [[1]] };
  try {
    const compatibility = ensureEmbeddingCompatibility(ctx as never, raw, embedder, { adapter: "custom", config: { storeIdentity: "stale" } }, join(directory, "models.json"), () => current);
    await scanning;
    current = false;
    release();
    assert.equal(await compatibility, false);
    assert.equal(rebuilds, 0);
    assert.equal(confirms, 0);
    assert.deepEqual(notifications, []);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("a stale affirmative migration confirmation cannot publish pending metadata", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-stale-confirm-"));
  const path = join(directory, "models.json");
  const old = { query: "old", document: "old" }, desired = { query: "new", document: "new" };
  const rag = { adapter: "custom", config: { storeIdentity: "stale-confirm" } };
  const key = embeddingStoreKey(rag);
  let current = true, rebuilds = 0, release!: (answer: boolean) => void, prompted!: () => void;
  const answer = new Promise<boolean>(resolve => { release = resolve; });
  const promptStarted = new Promise<void>(resolve => { prompted = resolve; });
  const raw = store(() => {});
  raw.scan = async (_filter, visit) => { await visit([{ id: "one", embeddingModel: old.document } as MemoryRecord]); return 1; };
  raw.rebuildVectors = async () => { rebuilds++; return 1; };
  const ctx = { hasUI: true, ui: { notify() {}, setWorkingMessage() {}, confirm: async () => { prompted(); return answer; } } };
  const embedder: EmbeddingProvider = { queryModel: desired.query, documentModel: desired.document, embedQuery: async () => [[1]], embedDocuments: async () => [[1]] };
  try {
    await writeEmbeddingModels(path, key, old);
    const compatibility = ensureEmbeddingCompatibility(ctx as never, raw, embedder, rag, path, () => current);
    await promptStarted;
    current = false;
    release(true);
    assert.equal(await compatibility, false);
    assert.equal(rebuilds, 0);
    assert.deepEqual(await readEmbeddingGeneration(path, key), { current: old });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("a superseded empty-store write cannot publish obsolete metadata", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-stale-empty-write-"));
  const path = join(directory, "models.json");
  const rag = { adapter: "custom", config: { storeIdentity: "stale-empty" } };
  let release!: () => void, acquired!: () => void, current = true;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const lockAcquired = new Promise<void>(resolve => { acquired = resolve; });
  const holding = withStorageLock(path, async () => { acquired(); await gate; });
  try {
    await lockAcquired;
    const compatibility = ensureEmbeddingCompatibility({ hasUI: false, ui: { notify() {}, setWorkingMessage() {}, confirm: async () => false } } as never, store(() => {}), { model: "model-A", embed: async () => [[1]] }, rag, path, () => current);
    current = false;
    release();
    await holding;
    assert.equal(await compatibility, false);
    assert.equal(await readEmbeddingGeneration(path, embeddingStoreKey(rag)), undefined);
  } finally { release(); await holding.catch(() => {}); await rm(directory, { recursive: true, force: true }); }
});

test("embedding model accessors cannot leak through compatibility diagnostics", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-secret-model-getter-"));
  const notifications: string[] = [];
  const embedder = Object.defineProperty({}, "model", { get() { throw new Error("PRIVATE_ADAPTER_ERROR"); } }) as EmbeddingProvider;
  const ctx = { hasUI: false, ui: { notify: (message: string) => notifications.push(message), setWorkingMessage() {}, confirm: async () => false } };
  try {
    assert.equal(await ensureEmbeddingCompatibility(ctx as never, store(() => {}), embedder, { adapter: "custom", config: { storeIdentity: "getter-secret" } }, join(directory, "models.json")), false);
    assert.doesNotMatch(notifications.join("\n"), /PRIVATE_ADAPTER_ERROR/);
    assert.match(notifications.join("\n"), /compatibility failed/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("pending recovery never labels vectors from a different configured embedder", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-generation-recovery-"));
  const path = join(directory, "models.json");
  const current = { query: "q-old", document: "d-old" };
  const pending = { query: "q-pending", document: "d-pending" };
  const desired = { query: "q-configured", document: "d-configured" };
  let rebuilds = 0;
  const oldRecord = { id: "one", embeddingModel: current.document } as MemoryRecord;
  const raw = store(() => {});
  raw.scan = async (_filter, visit) => { await visit([oldRecord]); return 1; };
  raw.rebuildVectors = async () => { rebuilds++; return 1; };
  const embedder: EmbeddingProvider = { queryModel: desired.query, documentModel: desired.document, embedQuery: async () => [[1]], embedDocuments: async () => [[1]] };
  const notifications: string[] = [];
  const ctx = { hasUI: false, ui: { notify: (message: string) => notifications.push(message), setWorkingMessage() {}, confirm: async () => false } };
  const rag = { adapter: "json", config: { path: "/memory" } };
  const key = embeddingStoreKey(rag);
  try {
    await writeEmbeddingModels(path, key, current);
    await beginEmbeddingMigration(path, key, current, pending);
    assert.equal(await ensureEmbeddingCompatibility(ctx as never, raw, embedder, rag, path), false);
    assert.equal(rebuilds, 0);
    assert.deepEqual((await readEmbeddingGeneration(path, key))?.pending, pending);
    assert.ok(notifications.some(message => message.includes("different models")));
  } finally { await rm(directory, { recursive: true, force: true }); }
});
