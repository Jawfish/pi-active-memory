import test from "node:test";
import assert from "node:assert/strict";
import { AdapterRegistry, createBuiltInAdapterRegistry } from "../src/adapters.js";

const context = {
  cwd: "/project",
  projectId: "project",
  sessionId: "session",
  extensionContext: { modelRegistry: { getApiKeyForProvider: async () => undefined } },
};

test("third-party factories can register and create each adapter component", async () => {
  const registry = new AdapterRegistry();
  const rag = { initialize: async () => {}, close: async () => {} };
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
  assert.equal(embedding.model, "ollama/embed");
  assert.equal(llm.selectedModel(), undefined);
});
