import test from "node:test";
import assert from "node:assert/strict";
import { AdapterRegistry, configuredEmbeddingModels, createBuiltInAdapterRegistry } from "../src/adapters.js";
import { embedDocuments, embeddingModels, embedQuery } from "../src/embeddings.js";

const context = {
  cwd: "/project",
  projectId: "project",
  sessionId: "session",
  extensionContext: { modelRegistry: { getApiKeyForProvider: async () => undefined } },
};

test("third-party factories can register and create each adapter component", async () => {
  const registry = new AdapterRegistry();
  const rag = {
    contractVersion: 2 as const,
    initialize: async () => {}, get: async () => undefined, insert: async () => "inserted" as const,
    mutate: async () => ({ status: "missing" as const }), compact: async () => { throw new Error("unused"); },
    scan: async () => 0, rebuildVectors: async () => 0, search: async () => [], list: async () => [],
    migrateLegacyProvenance: async () => 0, close: async () => {},
  };
  const embedding = { model: "custom/embed", embed: async () => [[1, 0]] };
  const llm = { selectedModel: () => "custom/llm", json: async <T>() => ({ ok: true }) as T };
  registry.registerRag("custom-rag", (config, received) => {
    assert.equal(config.endpoint, "memory://test");
    assert.equal(received.projectId, "project");
    return rag as never;
  });
  registry.registerEmbedding("custom-embedding", () => embedding);
  registry.registerLlm("custom-llm", () => llm);

  assert.equal(await registry.createRag({ adapter: "custom-rag", config: { endpoint: "memory://test" } }, context as never), rag);
  assert.equal(await registry.createEmbedding({ adapter: "custom-embedding", config: {} }, context as never), embedding);
  assert.equal(await registry.createLlm({ adapter: "custom-llm", config: {} }, context as never), llm);
});

test("legacy RAG adapters are closed and fail with the transactional v2 migration message", async () => {
  const registry = new AdapterRegistry();
  let closed = false;
  registry.registerRag("legacy", () => ({ initialize: async () => {}, close: async () => { closed = true; } }) as never);
  await assert.rejects(registry.createRag({ adapter: "legacy", config: {} }, context as never), /Transactional VectorStore v2 required/);
  assert.equal(closed, true);
});

test("stores with throwing contract accessors are still closed", async () => {
  const registry = new AdapterRegistry();
  let closed = false;
  const invalid = { get contractVersion(): never { throw new Error("PRIVATE_CONTRACT_ERROR"); }, close: async () => { closed = true; } };
  registry.registerRag("throwing", () => invalid as never);
  await assert.rejects(registry.createRag({ adapter: "throwing", config: {} }, context as never), /Transactional VectorStore v2 required/);
  assert.equal(closed, true);
});

test("invalid callable stores are closed before rejection", async () => {
  const registry = new AdapterRegistry();
  let closed = false;
  const invalid = Object.assign(() => {}, { close: async () => { closed = true; } });
  registry.registerRag("callable", () => invalid as never);
  await assert.rejects(registry.createRag({ adapter: "callable", config: {} }, context as never), /Transactional VectorStore v2 required/);
  assert.equal(closed, true);
});

test("adapter registration rejects duplicate ids and unknown selections", async () => {
  const registry = new AdapterRegistry();
  registry.registerEmbedding("same", () => ({ model: "x", embed: async () => [] }));
  assert.throws(() => registry.registerEmbedding("same", () => ({ model: "y", embed: async () => [] })), /already registered/);
  await assert.rejects(registry.createEmbedding({ adapter: "missing", config: {} }, context as never), /Unknown embedding adapter/);
});

test("built-in registry exposes independent rag, embedding, and llm adapters", async () => {
  const registry = createBuiltInAdapterRegistry();
  const rag = await registry.createRag({ adapter: "json", config: { path: "/tmp/memory.json" } }, context as never);
  const embedding = await registry.createEmbedding({ adapter: "ollama", config: { model: "embed", baseUrl: "http://localhost:11434" } }, context as never);
  const llm = await registry.createLlm({
    adapter: "pi-model",
    config: { candidates: ["openai-codex/test"], thinking: "off", maxTokens: 100 },
  }, context as never);

  assert.equal(typeof rag.search, "function");
  assert.deepEqual(embeddingModels(embedding), { query: "ollama/embed", document: "ollama/embed" });
  assert.equal(llm.selectedModel(), undefined);
});

test("Qdrant coordination cannot be split by configuration", async () => {
  const registry = createBuiltInAdapterRegistry();
  const first = await registry.createRag({ adapter: "qdrant", config: { url: "HTTP://QDRANT:80/", collection: "mem", lockPath: "/tmp/private-a" } }, context as never) as unknown as { lockPath: string; fencePath: string };
  const second = await registry.createRag({ adapter: "qdrant", config: { url: "http://qdrant", collection: "mem", lockPath: "/tmp/private-b" } }, context as never) as unknown as { lockPath: string; fencePath: string };
  assert.equal(first.lockPath, second.lockPath);
  assert.equal(first.fencePath, second.fencePath);
  assert.doesNotMatch(first.lockPath, /private-[ab]/);
});

test("embedding configuration requires unified or complete separate models", () => {
  assert.deepEqual(configuredEmbeddingModels({ model: "unified" }), { query: "unified", document: "unified" });
  assert.deepEqual(configuredEmbeddingModels({ queryModel: "query", documentModel: "document" }), { query: "query", document: "document" });
  assert.throws(() => configuredEmbeddingModels({ model: "unified", queryModel: "query", documentModel: "document" }), /not both/);
  assert.throws(() => configuredEmbeddingModels({ queryModel: "query" }), /both queryModel and documentModel/);
});

test("separate embedding providers route query and document inputs independently", async () => {
  const calls: string[] = [];
  const provider = {
    queryModel: "query/model",
    documentModel: "document/model",
    embedQuery: async (texts: string[]) => { calls.push(`query:${texts.join(",")}`); return [[1]]; },
    embedDocuments: async (texts: string[]) => { calls.push(`document:${texts.join(",")}`); return [[2]]; },
  };
  assert.deepEqual(embeddingModels(provider), { query: "query/model", document: "document/model" });
  assert.throws(() => embeddingModels({ ...provider, model: "unified/model" }), /both unified and separate/);
  assert.deepEqual(await embedQuery(provider, ["search"]), [[1]]);
  assert.deepEqual(await embedDocuments(provider, ["memory"]), [[2]]);
  assert.deepEqual(calls, ["query:search", "document:memory"]);
});
