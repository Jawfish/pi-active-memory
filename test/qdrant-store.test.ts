import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { embeddingStoreKey } from "../src/embedding-metadata.js";
import { QdrantVectorStore } from "../src/stores/qdrant-store.js";
import type { MemoryRecord, VectorRow } from "../src/types.js";

interface Point { id: string; payload: MemoryRecord; vector: number[] }
interface Collection {
  config: Record<string, unknown>;
  payloadSchema: Record<string, unknown>;
  points: Map<string, Point>;
  createBody?: Record<string, unknown>;
  indexes: Array<{ field_name: string; field_schema: unknown }>;
}

function record(id: string, overrides: Partial<MemoryRecord> = {}): MemoryRecord {
  return {
    schemaVersion: 2,
    id,
    text: id,
    kind: "fact",
    scope: "global",
    status: "active",
    confidence: 0.8,
    priority: 0.8,
    embeddingModel: "m",
    createdAt: "2025-01-01T00:00:00.000Z",
    updatedAt: "2025-01-01T00:00:00.000Z",
    source: { actor: "user", sessionId: "s", cwd: "/", cause: "test", reason: "test", evidence: "test" },
    sourceHistory: [],
    ...overrides,
  };
}

class FakeQdrant {
  readonly aliases = new Map<string, string>();
  readonly collections = new Map<string, Collection>();
  readonly routes: Array<{ method: string; path: string; body?: unknown }> = [];
  maxScrollPage = Number.POSITIVE_INFINITY;
  stageUpsertStatus: "completed" | "acknowledged" | "wait_timeout" = "completed";
  swapMode: "ok" | "throw-before" | "throw-after" | "non-ok-before" | "ok-without-commit" = "ok";
  failManagedDelete = false;
  hangManagedDelete = false;
  malformedScroll = false;
  malformedLookup = false;
  missingLookupOnce = false;
  missingScroll = false;

  seed(name: string, rows: readonly VectorRow[] = [], options: {
    dimension?: number;
    strictBatch?: number;
    payloadSchema?: Record<string, unknown>;
    metadata?: unknown;
  } = {}): void {
    const dimension = options.dimension ?? rows[0]?.vector.length ?? 2;
    this.collections.set(name, {
      config: {
        params: {
          vectors: { size: dimension, distance: "Cosine", on_disk: true },
          shard_number: 2,
          replication_factor: 2,
          read_fan_out_factor: 2,
          read_fan_out_delay_ms: 25,
          payload: { memory: "cold" },
        },
        optimizer_config: { indexing_threshold: 1000 },
        strict_mode_config: options.strictBatch ? { upsert_max_batchsize: options.strictBatch } : undefined,
        metadata: options.metadata ?? { owner: "active-memory" },
      },
      payloadSchema: options.payloadSchema ?? {},
      points: new Map(rows.map(row => [row.record.id, { id: row.record.id, payload: structuredClone(row.record), vector: [...row.vector] }])),
      indexes: [],
    });
  }

  fetch = async (input: string | URL | Request, init: RequestInit = {}): Promise<Response> => {
    const url = new URL(String(input));
    const path = `${url.pathname}${url.search}`;
    const method = init.method ?? "GET";
    const body = init.body ? JSON.parse(String(init.body)) : undefined;
    this.routes.push({ method, path, body });

    if (url.pathname === "/aliases" && method === "GET") {
      return Response.json({ result: { aliases: [...this.aliases].map(([alias_name, collection_name]) => ({ alias_name, collection_name })) } });
    }
    if (url.pathname === "/collections/aliases" && method === "POST") {
      if (this.swapMode === "throw-before") throw new Error("connection lost before response");
      if (this.swapMode === "non-ok-before") return new Response("timeout", { status: 500 });
      if (this.swapMode === "ok-without-commit") return Response.json({ result: true });
      for (const action of body.actions as Array<Record<string, { alias_name: string; collection_name?: string }>>) {
        if (action.delete_alias) this.aliases.delete(action.delete_alias.alias_name);
        if (action.create_alias) this.aliases.set(action.create_alias.alias_name, action.create_alias.collection_name!);
      }
      if (this.swapMode === "throw-after") throw new Error("response lost after commit");
      return Response.json({ result: true });
    }

    const collectionMatch = url.pathname.match(/^\/collections\/([^/]+)$/);
    if (collectionMatch) {
      const name = decodeURIComponent(collectionMatch[1]!);
      if (method === "GET") {
        const collection = this.collections.get(name);
        return collection
          ? Response.json({ result: { config: collection.config, payload_schema: collection.payloadSchema } })
          : new Response("missing", { status: 404 });
      }
      if (method === "PUT") {
        const params = Object.fromEntries(Object.entries(body).filter(([key]) => ["vectors", "shard_number", "sharding_method", "replication_factor", "write_consistency_factor", "on_disk_payload", "sparse_vectors", "payload"].includes(key)));
        const config: Record<string, unknown> = {
          params,
          hnsw_config: body.hnsw_config,
          wal_config: body.wal_config,
          optimizer_config: body.optimizers_config,
          quantization_config: body.quantization_config,
          strict_mode_config: body.strict_mode_config,
          metadata: body.metadata,
        };
        this.collections.set(name, { config, payloadSchema: {}, points: new Map(), createBody: body, indexes: [] });
        return Response.json({ result: true });
      }
      if (method === "PATCH") {
        const collection = this.resolve(name);
        collection.config.params = { ...(collection.config.params as Record<string, unknown>), ...(body.params as Record<string, unknown>) };
        return Response.json({ result: true });
      }
      if (method === "DELETE") {
        if (this.hangManagedDelete && name.includes("__active_memory_v2_old")) return new Promise<Response>(() => {});
        if (this.failManagedDelete && name.includes("__active_memory_v2_old")) return new Response("cleanup failed", { status: 500 });
        this.collections.delete(name);
        return Response.json({ result: true });
      }
    }

    const pointsMatch = url.pathname.match(/^\/collections\/([^/]+)\/points$/);
    if (pointsMatch && method === "POST") {
      if (this.missingLookupOnce) { this.missingLookupOnce = false; return new Response("missing", { status: 404 }); }
      if (this.malformedLookup) return Response.json({ status: "ok" });
      const collection = this.resolve(decodeURIComponent(pointsMatch[1]!));
      const points = (body.ids as string[]).flatMap((id: string) => collection.points.get(id) ? [collection.points.get(id)!] : []);
      return Response.json({ result: points.map(point => ({ id: point.id, payload: structuredClone(point.payload), ...(body.with_vector ? { vector: [...point.vector] } : {}) })) });
    }

    if (pointsMatch && method === "PUT") {
      const requested = decodeURIComponent(pointsMatch[1]!);
      const physical = this.physical(requested);
      const collection = this.resolve(requested);
      const points = body.points as Point[];
      const strict = collection.config.strict_mode_config as { upsert_max_batchsize?: number } | undefined;
      if (strict?.upsert_max_batchsize && points.length > strict.upsert_max_batchsize) return new Response("batch too large", { status: 400 });
      const isStage = physical.includes("__active_memory_v2_") && !physical.endsWith("_old");
      const status = isStage ? this.stageUpsertStatus : "completed";
      if (status === "completed") for (const point of points) collection.points.set(point.id, structuredClone(point));
      return Response.json({ result: { status } });
    }

    const payloadMatch = url.pathname.match(/^\/collections\/([^/]+)\/points\/payload$/);
    if (payloadMatch && method === "PUT") {
      const collection = this.resolve(decodeURIComponent(payloadMatch[1]!));
      for (const id of body.points as string[]) {
        const point = collection.points.get(id);
        if (point) point.payload = structuredClone(body.payload);
      }
      return Response.json({ result: { status: "completed" } });
    }

    const scrollMatch = url.pathname.match(/^\/collections\/([^/]+)\/points\/scroll$/);
    if (scrollMatch && method === "POST") {
      if (this.missingScroll) return new Response("missing", { status: 404 });
      if (this.malformedScroll) return Response.json({ status: "ok" });
      const collection = this.resolve(decodeURIComponent(scrollMatch[1]!));
      const all = [...collection.points.values()].sort((a, b) => a.id.localeCompare(b.id));
      const start = typeof body.offset === "string" ? Math.max(0, all.findIndex(point => point.id === body.offset) + 1) : 0;
      const limit = Math.min(Number(body.limit ?? 10), this.maxScrollPage);
      const page = all.slice(start, start + limit);
      const next = start + page.length < all.length ? page.at(-1)?.id : undefined;
      return Response.json({ result: { points: page.map(point => ({ id: point.id, payload: structuredClone(point.payload), ...(body.with_vector ? { vector: [...point.vector] } : {}) })), next_page_offset: next } });
    }

    const searchMatch = url.pathname.match(/^\/collections\/([^/]+)\/points\/search$/);
    if (searchMatch && method === "POST") return Response.json({ result: [] });

    const indexMatch = url.pathname.match(/^\/collections\/([^/]+)\/index$/);
    if (indexMatch && method === "PUT") {
      const collection = this.resolve(decodeURIComponent(indexMatch[1]!));
      collection.indexes.push(structuredClone(body));
      return Response.json({ result: { status: "completed" } });
    }

    throw new Error(`unexpected route ${method} ${path}`);
  };

  private physical(name: string): string { return this.aliases.get(name) ?? name; }
  private resolve(name: string): Collection {
    const physical = this.physical(name);
    const collection = this.collections.get(physical);
    if (!collection) throw new Error(`missing collection ${physical}`);
    return collection;
  }
}

async function withServer(body: (server: FakeQdrant, lock: string) => Promise<void>): Promise<void> {
  const original = globalThis.fetch;
  const directory = await mkdtemp(join(tmpdir(), "active-memory-qdrant-"));
  const server = new FakeQdrant();
  globalThis.fetch = server.fetch as typeof fetch;
  try { await body(server, join(directory, "lock")); }
  finally { globalThis.fetch = original; await rm(directory, { recursive: true, force: true }); }
}

test("Qdrant adopts a concrete collection and writes through its managed alias", () => withServer(async (server, lock) => {
  server.seed("mem");
  const store = new QdrantVectorStore("http://qdrant/", "mem", undefined, lock);
  assert.equal(await store.insert({ record: record("one"), vector: [1, 0] }), "inserted");
  assert.equal(server.aliases.get("mem__active_memory_live_v2"), "mem");
  assert.ok(server.routes.some(route => route.method === "POST" && route.path === "/collections/aliases"));
  assert.ok(server.routes.some(route => route.method === "PUT" && route.path === "/collections/mem__active_memory_live_v2/points?wait=true"));
}));

test("configured aliases converge on physical coordination and metadata identity", () => withServer(async (server, lock) => {
  server.seed("physical");
  server.aliases.set("mem", "physical");
  server.aliases.set("alternate", "physical");
  const store = new QdrantVectorStore("HTTP://QDRANT:80///", "mem", undefined, lock);
  const alternate = new QdrantVectorStore("http://qdrant", "alternate", undefined, lock);
  await store.initialize(2);
  await alternate.initialize(2);
  assert.equal(server.aliases.get("physical__active_memory_live_v2"), "physical");
  assert.equal(store.metadataStoreKey(), alternate.metadataStoreKey());
  const first = embeddingStoreKey({ adapter: "qdrant", config: { url: "HTTP://QDRANT:80/", collection: "mem", apiKeyEnv: "ONE", lockPath: "/a" } });
  const second = embeddingStoreKey({ adapter: "qdrant", config: { url: "http://qdrant", collection: "mem", apiKeyEnv: "TWO", lockPath: "/b" } });
  assert.equal(first, second);
  const other = new QdrantVectorStore("http://qdrant", "other") as unknown as { lockPath: string };
  const same = new QdrantVectorStore("HTTP://QDRANT:80/", "mem") as unknown as { lockPath: string };
  const canonical = new QdrantVectorStore("http://qdrant", "mem") as unknown as { lockPath: string };
  assert.equal(same.lockPath, canonical.lockPath);
  assert.equal(other.lockPath, canonical.lockPath);
}));

test("payload-only mutation overwrites the exact payload", () => withServer(async (server, lock) => {
  server.seed("physical", [{ record: record("one", { scope: "project", projectId: "old" }), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", "physical");
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await store.mutate("one", latest => ({ record: { ...latest, scope: "global", projectId: undefined } }));
  const payload = server.collections.get("physical")!.points.get("one")!.payload;
  assert.equal(payload.scope, "global");
  assert.equal(Object.hasOwn(payload, "projectId"), false);
  assert.ok(server.routes.some(route => route.method === "PUT" && route.path.includes("/points/payload?wait=true")));
}));

test("malformed payloads and unsupported vector topology fail closed", () => withServer(async (server, lock) => {
  server.seed("physical", [{ record: record("bad"), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", "physical");
  const collection = server.collections.get("physical")!;
  collection.points.get("bad")!.payload = { ...record("bad"), confidence: "high" } as unknown as MemoryRecord;
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await assert.rejects(store.rebuildVectors(3, async () => []), /invalid record shape/);
  assert.equal(server.aliases.get("mem__active_memory_live_v2"), "physical");
  assert.equal([...server.collections.keys()].filter(name => name.includes("__active_memory_v2_")).length, 0);
  await assert.rejects(store.insert({ record: record("self", { supersedes: ["self"] }), vector: [1, 0] }), /supersedes/);

  (collection.config.params as Record<string, unknown>).vectors = { default: { size: 2, distance: "Cosine" } };
  const named = new QdrantVectorStore("http://qdrant", "mem", undefined, `${lock}-named`);
  await assert.rejects(named.initialize(), /single unnamed Qdrant vector/);
}));

test("malformed lookups cannot turn insert-if-absent into an overwrite", () => withServer(async (server, lock) => {
  server.seed("physical", [{ record: record("one", { text: "original" }), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", "physical");
  server.malformedLookup = true;
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await assert.rejects(store.insert({ record: record("one", { text: "replacement" }), vector: [0, 1] }), /malformed result/);
  assert.equal(server.collections.get("physical")!.points.get("one")!.payload.text, "original");
}));

test("a live lookup 404 cannot turn insert-if-absent into an overwrite", () => withServer(async (server, lock) => {
  server.seed("physical", [{ record: record("one", { text: "original" }), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", "physical");
  server.missingLookupOnce = true;
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await assert.rejects(store.insert({ record: record("one", { text: "replacement" }), vector: [0, 1] }), /disappeared during lookup/);
  assert.equal(server.collections.get("physical")!.points.get("one")!.payload.text, "original");
  assert.equal(server.routes.filter(route => route.method === "PUT" && route.path.includes("/points?wait=true")).length, 0);
}));

test("malformed scroll responses cannot publish an empty replacement", () => withServer(async (server, lock) => {
  server.seed("physical", [{ record: record("one"), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", "physical");
  server.malformedScroll = true;
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await assert.rejects(store.rebuildVectors(3, async () => []), /malformed result/);
  assert.equal(server.aliases.get("mem__active_memory_live_v2"), "physical");
  assert.equal(server.collections.get("physical")!.points.size, 1);
}));

test("a live scroll 404 cannot publish an empty replacement", () => withServer(async (server, lock) => {
  server.seed("physical", [{ record: record("one"), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", "physical");
  server.missingScroll = true;
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await assert.rejects(store.rebuildVectors(3, async () => []), /disappeared during scroll/);
  assert.equal(server.aliases.get("mem__active_memory_live_v2"), "physical");
  assert.equal(server.collections.get("physical")!.points.size, 1);
}));

test("a dimensioned initializer retries a joined absent-store bootstrap", () => withServer(async (server, lock) => {
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await Promise.all([store.initialize(), store.initialize(3)]);
  const active = server.aliases.get("mem__active_memory_live_v2")!;
  const vectors = server.collections.get(active)!.config.params as { vectors: { size: number } };
  assert.equal(vectors.vectors.size, 3);
}));

test("an absent Qdrant store can publish an empty target-dimension generation", () => withServer(async (server, lock) => {
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  assert.equal(await store.rebuildVectors(3, async () => []), 0);
  const active = server.aliases.get("mem__active_memory_live_v2")!;
  const vectors = server.collections.get(active)!.config.params as { vectors: { size: number } };
  assert.equal(vectors.vectors.size, 3);
}));

test("new collection initialization caches its dimension for same-process compaction", () => withServer(async (server, lock) => {
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await store.insert({ record: record("one"), vector: [1, 0] });
  await store.insert({ record: record("two"), vector: [0, 1] });
  const compacted = await store.compact(["one", "two"], () => ({ record: record("combined"), vector: [1, 0] }));
  assert.equal(compacted.id, "combined");
}));

test("cached actual dimensions reject incompatible search before its request", () => withServer(async (server, lock) => {
  server.seed("physical", [], { dimension: 2 });
  server.aliases.set("mem__active_memory_live_v2", "physical");
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await store.initialize(2);
  await assert.rejects(store.search([1, 2, 3], {}, 1), /dimension mismatch/);
  assert.equal(server.routes.some(route => route.path.includes("/search")), false);
}));

test("scroll pagination migrates every legacy provenance payload", () => withServer(async (server, lock) => {
  const rows: VectorRow[] = ["a", "b", "c"].map(id => ({
    record: { ...record(id), schemaVersion: 1 as const, source: { sessionId: "old", cwd: "/old" } as MemoryRecord["source"], sourceHistory: [{ sessionId: "older", cwd: "/older" } as MemoryRecord["source"]] },
    vector: [1, 0],
  }));
  rows.push({ record: record("self", { supersedes: ["self", "older", "older"] }), vector: [1, 0] });
  server.seed("physical", rows);
  server.aliases.set("mem__active_memory_live_v2", "physical");
  server.maxScrollPage = 1;
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  assert.equal(await store.migrateLegacyProvenance(), 4);
  for (const point of server.collections.get("physical")!.points.values()) {
    assert.equal(point.payload.source.actor, "user");
    if (point.payload.sourceHistory?.length) assert.equal(point.payload.sourceHistory[0]?.actor, "user");
    assert.equal(point.payload.schemaVersion, 2);
    assert.equal(point.payload.supersedes?.includes(point.payload.id) ?? false, false);
  }
  assert.ok(server.routes.filter(route => route.path.includes("/points/scroll")).length >= 3);
}));

test("staged compaction preserves topology, indexes, strict batches, and atomic visibility", () => withServer(async (server, lock) => {
  const rows = Array.from({ length: 130 }, (_, index) => ({ record: record(`m${String(index).padStart(3, "0")}`), vector: [1, 0] }));
  const previous = "mem__active_memory_v2_old";
  server.seed(previous, rows, { strictBatch: 64, payloadSchema: { status: { data_type: "keyword", params: null, points: 130 } }, metadata: { tier: "cold" } });
  server.aliases.set("mem__active_memory_live_v2", previous);
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  const compacted = await store.compact(["m000", "m001"], latest => ({ record: record("combined", { text: latest.map(item => item.text).join("+") }), vector: [1, 0] }));
  assert.equal(compacted.id, "combined");
  const active = server.aliases.get("mem__active_memory_live_v2")!;
  assert.notEqual(active, previous);
  const staged = server.collections.get(active)!;
  assert.equal((staged.createBody?.optimizers_config as { indexing_threshold?: number }).indexing_threshold, 1000);
  assert.deepEqual(staged.createBody?.payload, { memory: "cold" });
  assert.deepEqual(staged.createBody?.metadata, { tier: "cold" });
  assert.equal(Object.hasOwn(staged.createBody ?? {}, "read_fan_out_factor"), false);
  assert.deepEqual(server.routes.find(route => route.method === "PATCH" && route.path.includes(active))?.body, { params: { read_fan_out_factor: 2, read_fan_out_delay_ms: 25 } });
  assert.equal((staged.config.params as Record<string, unknown>).read_fan_out_factor, 2);
  assert.equal((staged.config.params as Record<string, unknown>).read_fan_out_delay_ms, 25);
  assert.deepEqual(staged.indexes, [{ field_name: "status", field_schema: "keyword" }]);
  assert.ok(server.routes.some(route => route.path.includes("/index?wait=true")));
  const batches = server.routes.filter(route => route.method === "PUT" && route.path.includes("/points?wait=true") && !route.path.includes("live_v2"));
  assert.ok(batches.every(route => ((route.body as { points: unknown[] }).points.length <= 64)));
  assert.equal(staged.points.get("m000")?.payload.status, "superseded");
  assert.equal(staged.points.get("combined")?.payload.status, "active");
}));

test("a malformed successful swap response cannot delete the live generation", () => withServer(async (server, lock) => {
  const previous = "mem__active_memory_v2_old";
  server.seed(previous, [{ record: record("one"), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", previous);
  server.swapMode = "ok-without-commit";
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await assert.rejects(store.rebuildVectors(3, async page => page.map(item => ({ record: { ...item, embeddingModel: "new" }, vector: [1, 0, 0] }))), /writes remain fenced/);
  assert.equal(server.aliases.get("mem__active_memory_live_v2"), previous);
  assert.equal(server.collections.has(previous), true);
}));

test("an incomplete staged upload never publishes and removes the invisible stage", () => withServer(async (server, lock) => {
  server.seed("physical", [{ record: record("one"), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", "physical");
  server.stageUpsertStatus = "wait_timeout";
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  await assert.rejects(store.rebuildVectors(3, async page => page.map(item => ({ record: { ...item, embeddingModel: "new" }, vector: [1, 0, 0] }))), /not completed/);
  assert.equal(server.aliases.get("mem__active_memory_live_v2"), "physical");
  assert.equal([...server.collections].some(([name]) => name.includes("__active_memory_v2_") && name !== "physical"), false);
}));

test("a submitted ambiguous swap retains staging and a committed lost response succeeds", async t => {
  await t.test("uncommitted ambiguity", () => withServer(async (server, lock) => {
    server.seed("physical", [{ record: record("one"), vector: [1, 0] }]);
    server.aliases.set("mem__active_memory_live_v2", "physical");
    server.swapMode = "throw-before";
    const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
    await assert.rejects(store.rebuildVectors(3, async page => page.map(item => ({ record: { ...item, embeddingModel: "new" }, vector: [1, 0, 0] }))), /writes remain fenced/);
    assert.equal(server.aliases.get("mem__active_memory_live_v2"), "physical");
    const staging = [...server.collections.keys()].find(name => name.includes("__active_memory_v2_"))!;
    await assert.rejects(store.mutate("one", latest => ({ record: { ...latest, confidence: 0.9 } })), /writes are fenced/);
    assert.equal(server.collections.get("physical")!.points.get("one")!.payload.confidence, 0.8);
    // If the submitted request commits later, the next writer resolves the fence
    // against the alias before mutating the newly visible generation.
    server.aliases.set("mem__active_memory_live_v2", staging);
    assert.equal((await store.mutate("one", latest => ({ record: { ...latest, confidence: 0.9 } }))).status, "updated");
    assert.equal(server.collections.get(staging)!.points.get("one")!.payload.confidence, 0.9);
  }));
  await t.test("committed response loss", () => withServer(async (server, lock) => {
    server.seed("physical", [{ record: record("one"), vector: [1, 0] }]);
    server.aliases.set("mem__active_memory_live_v2", "physical");
    server.swapMode = "throw-after";
    const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
    assert.equal(await store.rebuildVectors(3, async page => page.map(item => ({ record: { ...item, embeddingModel: "new" }, vector: [1, 0, 0] }))), 1);
    assert.notEqual(server.aliases.get("mem__active_memory_live_v2"), "physical");
  }));
});

test("stalled cleanup after a committed swap does not delay the rebuild", () => withServer(async (server, lock) => {
  const previous = "mem__active_memory_v2_old";
  server.seed(previous, [{ record: record("one"), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", previous);
  server.hangManagedDelete = true;
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  const rebuilt = store.rebuildVectors(3, async page => page.map(item => ({ record: { ...item, embeddingModel: "new" }, vector: [1, 0, 0] })));
  // Leave enough headroom for contended Node 22 CI runners while remaining
  // well below the implementation's five-second cleanup abort timeout.
  const result = await Promise.race([rebuilt, new Promise<never>((_resolve, reject) => setTimeout(() => reject(new Error("cleanup blocked commit")), 1_000))]);
  assert.equal(result, 1);
  assert.notEqual(server.aliases.get("mem__active_memory_live_v2"), previous);
}));

test("cleanup failure after a committed swap does not reject the rebuild", () => withServer(async (server, lock) => {
  const previous = "mem__active_memory_v2_old";
  server.seed(previous, [{ record: record("one"), vector: [1, 0] }]);
  server.aliases.set("mem__active_memory_live_v2", previous);
  server.failManagedDelete = true;
  const store = new QdrantVectorStore("http://qdrant", "mem", undefined, lock);
  assert.equal(await store.rebuildVectors(3, async page => page.map(item => ({ record: { ...item, embeddingModel: "new" }, vector: [1, 0, 0] }))), 1);
  assert.notEqual(server.aliases.get("mem__active_memory_live_v2"), previous);
  assert.equal(server.collections.has(previous), true);
}));
