import test from "node:test";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_CONFIG, loadConfig, saveUserCompactionThreshold, saveUserMemoryLifecycle } from "../src/config.js";

test("user compaction setting preserves unrelated extension configuration", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-config-"));
  const path = join(directory, "active-memory.json");
  try {
    await writeFile(path, JSON.stringify({ recall: { topK: 7 }, compaction: { maximumProposals: 3 } }));
    await saveUserCompactionThreshold(0.6, path);
    assert.deepEqual(JSON.parse(await readFile(path, "utf8")), {
      recall: { topK: 7 },
      compaction: { maximumProposals: 3, similarityThreshold: 0.6 },
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("user compaction setting rejects an invalid similarity threshold", async () => {
  await assert.rejects(saveUserCompactionThreshold(1.1, "/unused/active-memory.json"), /between 0 and 1/);
});

test("grouped memory lifecycle settings persist without clobbering other configuration", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-lifecycle-config-"));
  const path = join(directory, "active-memory.json");
  try {
    await writeFile(path, JSON.stringify({ recall: { topK: 7 } }));
    const lifecycle = structuredClone(DEFAULT_CONFIG.memoryLifecycle);
    lifecycle.decay.initialRate = 0.2;
    await saveUserMemoryLifecycle(lifecycle, path);
    const saved = JSON.parse(await readFile(path, "utf8"));
    assert.equal(saved.recall.topK, 7);
    assert.equal(saved.memoryLifecycle.decay.initialRate, 0.2);
    assert.equal(saved.memoryLifecycle.confidence.deletionThreshold, 0.1);
    assert.equal(saved.memoryLifecycle.feedback.historyLimit, 50);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("prompt templates can be overridden individually without losing defaults", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-prompts-config-"));
  const cwd = join(directory, "project");
  const agentDir = join(directory, "agent");
  try {
    await mkdir(cwd, { recursive: true });
    await mkdir(agentDir, { recursive: true });
    await writeFile(join(agentDir, "active-memory.json"), JSON.stringify({
      prompts: {
        query: "CUSTOM QUERY: {{context}}",
        tools: { memoryFeedback: { guidelines: ["Custom feedback policy"] } },
      },
    }));
    const config = await loadConfig(cwd, false, agentDir);
    assert.equal(config.prompts.query, "CUSTOM QUERY: {{context}}");
    assert.match(config.prompts.extraction, /NEWEST USER MESSAGE/);
    assert.equal(config.prompts.tools.memoryFeedback.snippet, DEFAULT_CONFIG.prompts.tools.memoryFeedback.snippet);
    assert.deepEqual(config.prompts.tools.memoryFeedback.guidelines, ["Custom feedback policy"]);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("legacy provider configuration migrates to independent adapter selections", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-provider-migration-"));
  const cwd = join(directory, "project");
  const agentDir = join(directory, "agent");
  try {
    await mkdir(cwd, { recursive: true });
    await mkdir(agentDir, { recursive: true });
    await writeFile(join(agentDir, "active-memory.json"), JSON.stringify({
      database: { provider: "qdrant", url: "http://qdrant", collection: "memories" },
      embedding: { provider: "ollama", model: "embed", baseUrl: "http://ollama" },
      fastModel: { candidates: ["openai-codex/test"], thinking: "low", maxTokens: 42 },
    }));
    const config = await loadConfig(cwd, false, agentDir);
    assert.deepEqual(config.providers.rag, { adapter: "qdrant", config: { url: "http://qdrant", collection: "memories" } });
    assert.deepEqual(config.providers.embedding, { adapter: "ollama", config: { model: "embed", baseUrl: "http://ollama" } });
    assert.deepEqual(config.providers.llm, { adapter: "pi-model", config: { candidates: ["openai-codex/test"], thinking: "low", maxTokens: 42 } });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("separate embedding models replace the default unified model", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-separate-embedding-"));
  const cwd = join(directory, "project");
  const agentDir = join(directory, "agent");
  try {
    await mkdir(cwd, { recursive: true });
    await mkdir(agentDir, { recursive: true });
    await writeFile(join(agentDir, "active-memory.json"), JSON.stringify({
      providers: { embedding: { config: { queryModel: "query-model", documentModel: "document-model" } } },
    }));
    const config = await loadConfig(cwd, false, agentDir);
    assert.deepEqual(config.providers.embedding.config, {
      queryModel: "query-model",
      documentModel: "document-model",
      baseUrl: "https://api.openai.com/v1",
      apiKeyEnv: "OPENAI_API_KEY",
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("trusted TypeScript configuration as code overlays JSON configuration", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-code-config-"));
  const cwd = join(directory, "project");
  const agentDir = join(directory, "agent");
  const projectDir = join(cwd, ".pi");
  try {
    await mkdir(projectDir, { recursive: true });
    await mkdir(agentDir, { recursive: true });
    await writeFile(join(agentDir, "active-memory.json"), JSON.stringify({ recall: { topK: 3 } }));
    await writeFile(join(agentDir, "active-memory.config.ts"), "export default ({ cwd }) => ({ activityLog: { includeText: cwd.endsWith('project') } });");
    await writeFile(join(projectDir, "active-memory.config.ts"), "export default { providers: { llm: { adapter: 'custom-llm', config: { tier: 'subscription' } } } };" );

    const untrusted = await loadConfig(cwd, false, agentDir);
    assert.equal(untrusted.recall.topK, 3);
    assert.equal(untrusted.activityLog.includeText, true);
    assert.equal(untrusted.providers.llm.adapter, "pi-model");

    const trusted = await loadConfig(cwd, true, agentDir);
    assert.equal(trusted.providers.llm.adapter, "custom-llm");
    assert.deepEqual(trusted.providers.llm.config, { tier: "subscription" });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
