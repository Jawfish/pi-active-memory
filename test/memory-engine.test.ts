import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_CONFIG } from "../src/config.js";
import { isRecentCurrentSessionMemory, MemoryEngine, rankMemoryMatches } from "../src/memory-engine.js";
import type { FastModelRunner, MemoryFilter, MemoryMatch, MemoryRecord, VectorStore } from "../src/types.js";

class Store implements VectorStore {
  records: MemoryRecord[] = [];
  matches: MemoryMatch[] = [];
  async initialize(): Promise<void> {}
  async upsert(record: MemoryRecord): Promise<void> { this.records.push(record); }
  async search(_vector: number[], _filter: MemoryFilter, _limit: number): Promise<MemoryMatch[]> { return this.matches; }
  async list(): Promise<MemoryRecord[]> { return this.records; }
  async markDeleted(): Promise<boolean> { return false; }
  async migrateLegacyProvenance(): Promise<number> { return 0; }
  async close(): Promise<void> {}
}

class Fast implements FastModelRunner {
  private index = 0;
  constructor(private readonly responses: unknown[]) {}
  async json<T>(): Promise<T> { return this.responses[Math.min(this.index++, this.responses.length - 1)] as T; }
  selectedModel(): string { return "test/fast"; }
}

const embedder = { model: "test/embed", embed: async (texts: string[]) => texts.map(() => [1, 0]) };

function engine(response: unknown | unknown[], store: Store): MemoryEngine {
  return new MemoryEngine(DEFAULT_CONFIG, store, embedder as never, new Fast(Array.isArray(response) ? response : [response]), "project", "session", "/cwd");
}

test("capture rejects a candidate sourced only from assistant context", async () => {
  const store = new Store();
  const subject = engine({ memories: [{ text: "Use tabs.", kind: "user_profile", scope: "global", confidence: 0.99, evidence: "use tabs" }] }, store);
  const stored = await subject.capture("Please fix the formatter.", "assistant: I think the user likes tabs");
  assert.equal(stored, 0);
  assert.equal(store.records.length, 0);
});

test("capture accepts supported user evidence and records its provenance", async () => {
  const store = new Store();
  const subject = engine([
    { memories: [{ text: "The user's favourite color is orange.", kind: "user_profile", scope: "global", confidence: 0.99, evidence: "favourite color is orange" }] },
    { accept: true, reason: "Explicit durable profile preference" },
  ], store);
  assert.equal(await subject.capture("My favourite color is orange.", "assistant: unrelated reasoning"), 1);
  assert.equal(store.records[0]?.kind, "user_profile");
  assert.equal(store.records[0]?.source.evidence, "favourite color is orange");
  assert.equal(store.records[0]?.source.actor, "user");
  assert.equal(store.records[0]?.source.cause, "explicit_user_statement");
  assert.equal(store.records[0]?.priority, 1);
});

test("capture rejects legacy or unsupported memory categories", async () => {
  const store = new Store();
  const subject = engine({ memories: [{ text: "Add tests next.", kind: "decision", scope: "project", confidence: 0.99, evidence: "Add tests next" }] }, store);
  assert.equal(await subject.capture("Add tests next.", ""), 0);
  assert.equal(store.records.length, 0);
});

test("capture rejects transient task instructions after semantic validation", async () => {
  const store = new Store();
  const subject = engine([
    { memories: [{ text: "Add tests as the next step.", kind: "skill_workflow", scope: "project", confidence: 0.99, evidence: "Add tests next" }] },
    { accept: false, reason: "Current-task next step" },
  ], store);
  assert.equal(await subject.capture("Add tests next.", ""), 0);
  assert.equal(store.records.length, 0);
});

test("assistant capture is time-gated and stores lower-priority provenance", async () => {
  const store = new Store();
  const subject = engine([
    { memories: [{ text: "The parser registry is in src/parsers/registry.ts.", kind: "fact", scope: "project", confidence: 0.95, evidence: "parser registry is in src/parsers/registry.ts", whyStored: "It required tracing several generated imports and documentation references." }] },
    { accept: true, reason: "Substantial documentation tracing" },
  ], store);
  const investigation = "After tracing generated imports, the parser registry is in src/parsers/registry.ts and owns plugin ordering.";
  assert.equal(await subject.captureAssistantInvestigation(investigation, "Find parser registration", 59_999), 0);
  assert.equal(await subject.captureAssistantInvestigation(investigation, "Find parser registration", 61_000), 1);
  const record = store.records[0]!;
  assert.equal(record.source.actor, "assistant");
  assert.equal(record.source.cause, "Find parser registration");
  assert.match(record.source.reason ?? "", /generated imports/);
  assert.equal(record.source.elapsedMs, 61_000);
  assert.equal(record.confidence, DEFAULT_CONFIG.assistantCapture.maximumConfidence);
  assert.equal(record.priority, DEFAULT_CONFIG.assistantCapture.priority);
});

test("assistant candidate searches first and updates an assistant memory", async () => {
  const store = new Store();
  const existing = { ...memory("existing", "older-session", "2020-01-01T00:00:00Z"), source: { actor: "assistant" as const, sessionId: "older-session", cwd: "/cwd", cause: "old investigation", reason: "hard to find" }, priority: 0.55 };
  store.matches = [{ record: existing, score: 0.95 }];
  const subject = engine([
    { memories: [{ text: "The parser registry is in src/parser-registry.ts.", kind: "fact", scope: "global", confidence: 0.7, evidence: "parser registry moved to src/parser-registry.ts", whyStored: "Locating the move required tracing generated imports." }] },
    { accept: true, reason: "Non-trivial trace" },
    { action: "replace", targetId: "existing", text: "The parser registry is in src/parser-registry.ts." },
  ], store);
  assert.equal(await subject.captureAssistantInvestigation("The parser registry moved to src/parser-registry.ts after tracing generated imports.", "Locate parser registry", 80_000), 1);
  assert.equal(store.records[0]?.id, "existing");
  assert.equal(store.records[0]?.sourceHistory?.[0]?.cause, "old investigation");
  assert.equal(store.records[0]?.source.cause, "Locate parser registry");
});

test("assistant candidate cannot overwrite a user-sourced memory", async () => {
  const store = new Store();
  const existing = memory("user-memory", "older-session", "2020-01-01T00:00:00Z");
  store.matches = [{ record: existing, score: 0.95 }];
  const subject = engine([
    { memories: [{ text: "Use pnpm for this project.", kind: "fact", scope: "global", confidence: 0.7, evidence: "Use pnpm for this project", whyStored: "Package-manager behavior took time to diagnose." }] },
    { accept: true, reason: "Substantial diagnosis" },
    { action: "replace", targetId: "user-memory", text: "Use pnpm for this project." },
  ], store);
  assert.equal(await subject.captureAssistantInvestigation("Use pnpm for this project after a long package-manager diagnosis.", "Diagnose install", 80_000), 0);
  assert.equal(store.records.length, 0);
});

function memory(id: string, sourceSession: string, createdAt: string): MemoryRecord {
  return { id, text: id, kind: "fact", scope: "global", confidence: 1, status: "active", source: { actor: "user", sessionId: sourceSession, cwd: "/cwd", cause: "test", reason: "test fixture" }, createdAt, updatedAt: createdAt, embeddingModel: "test", schemaVersion: 1 };
}

test("ranking lowers assistant memories by confidence and priority", () => {
  const user = memory("user", "old", "2020-01-01T00:00:00Z");
  const assistant = { ...memory("assistant", "old", "2020-01-01T00:00:00Z"), confidence: 0.75, priority: 0.55, source: { actor: "assistant" as const, sessionId: "old", cwd: "/cwd", cause: "test", reason: "test fixture" } };
  const ranked = rankMemoryMatches([{ record: assistant, score: 0.99 }, { record: user, score: 0.8 }]);
  assert.deepEqual(ranked.map((match) => match.record.id), ["user", "assistant"]);
  assert.ok(ranked[1]!.score < 0.55);
});

test("recent-memory filter suppresses only young memories from the current session", () => {
  const now = Date.parse("2026-07-25T12:00:00Z");
  assert.equal(isRecentCurrentSessionMemory(memory("recent", "session", "2026-07-25T11:50:00Z"), "session", now, 30 * 60_000), true);
  assert.equal(isRecentCurrentSessionMemory(memory("old", "session", "2026-07-25T11:00:00Z"), "session", now, 30 * 60_000), false);
  assert.equal(isRecentCurrentSessionMemory(memory("other", "other-session", "2026-07-25T11:59:00Z"), "session", now, 30 * 60_000), false);
  assert.equal(isRecentCurrentSessionMemory(memory("disabled", "session", "2026-07-25T11:59:00Z"), "session", now, 0), false);
});

test("recall excludes recent current-session memory but keeps older and other-session memory", async () => {
  const store = new Store();
  const recent = memory("recent", "session", new Date().toISOString());
  const old = memory("old", "session", "2020-01-01T00:00:00Z");
  const other = memory("other", "other-session", new Date().toISOString());
  store.matches = [recent, old, other].map((record, index) => ({ record, score: 0.9 - index * 0.1 }));
  const subject = engine([
    { query: "relevant facts" },
    { relevantIds: ["recent", "old", "other"], instruction: "Apply relevant facts", reason: "Relevant" },
  ], store);
  const recalled = await subject.recall("current task");
  assert.deepEqual(recalled?.relevant.map((match) => match.record.id), ["old", "other"]);
});
