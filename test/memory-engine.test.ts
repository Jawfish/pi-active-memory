import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_CONFIG } from "../src/config.js";
import { isRecentCurrentSessionMemory, MemoryEngine, rankMemoryMatches } from "../src/memory-engine.js";
import type { FastModelRunner, MemoryFilter, MemoryMatch, MemoryRecord, VectorStore } from "../src/types.js";

class Store implements VectorStore {
  records: MemoryRecord[] = [];
  matches: MemoryMatch[] = [];
  async initialize(): Promise<void> {}
  async upsert(record: MemoryRecord): Promise<void> {
    const index = this.records.findIndex((candidate) => candidate.id === record.id);
    if (index >= 0) this.records[index] = record;
    else this.records.push(record);
  }
  async update(record: MemoryRecord): Promise<boolean> {
    const index = this.records.findIndex((candidate) => candidate.id === record.id);
    if (index < 0) return false;
    this.records[index] = record;
    return true;
  }
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
  assert.equal(store.records[0]?.confidence, DEFAULT_CONFIG.memoryLifecycle.confidence.initial);
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

test("capture deterministically rejects temporary try-it-for-now state", async () => {
  const store = new Store();
  const subject = engine({
    memories: [{
      text: "The user installed pi-tasks and wants to try it for now.",
      kind: "fact",
      scope: "project",
      confidence: 0.99,
      evidence: "I've installed pi-tasks; let's try it for now",
    }],
  }, store);
  assert.equal(await subject.capture("I've installed pi-tasks; let's try it for now.", ""), 0);
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
  assert.equal(record.confidence, DEFAULT_CONFIG.memoryLifecycle.confidence.initial);
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

test("recall suppresses a memory while its source text remains in active context", async () => {
  const store = new Store();
  const record = {
    ...memory("installed-tool", "older-session", "2020-01-01T00:00:00Z"),
    source: {
      actor: "user" as const,
      sessionId: "older-session",
      cwd: "/cwd",
      cause: "explicit_user_statement",
      reason: "fixture",
      userText: "I've installed pi-tasks; let's try it for now.",
      evidence: "let's try it for now",
    },
  };
  store.matches = [{ record, score: 0.95 }];
  const subject = engine({ query: "pi tasks" }, store);
  const recalled = await subject.recall("current task", undefined, new Set(), "user: I've installed pi-tasks; let's try it for now.");
  assert.equal(recalled, undefined);
});

test("lifecycle sweep migrates legacy records and soft-deletes expired records", async () => {
  const store = new Store();
  const legacy = memory("legacy", "old", "2020-01-01T00:00:00Z");
  const expired = {
    ...memory("expired", "old", "2020-01-01T00:00:00Z"),
    decayRate: DEFAULT_CONFIG.memoryLifecycle.decay.initialRate,
    lifecycle: {
      lastDecayDate: "2025-12-01",
      lastRelevantAt: "2025-12-01T00:00:00Z",
      lastRelevantSessionId: "old",
      reinforcementCount: 0,
      lastReinforcementCause: "created" as const,
    },
  };
  store.records = [legacy, expired];
  const result = await engine([], store).sweepLifecycle(new Date("2026-01-01T00:00:00Z"));
  assert.deepEqual(result, { initialized: 1, expired: 1 });
  assert.equal(store.records.find((record) => record.id === "legacy")?.status, "active");
  assert.equal(store.records.find((record) => record.id === "legacy")?.lifecycle?.lastDecayDate, "2026-01-01");
  assert.equal(store.records.find((record) => record.id === "expired")?.status, "deleted");
  assert.equal(store.records.find((record) => record.id === "expired")?.lifecycle?.deletionCause, "low_confidence");
});

test("feedback updates an active memory with auditable provenance", async () => {
  const store = new Store();
  const existing = { ...memory("rated", "old", "2020-01-01T00:00:00Z"), confidence: 0.5 };
  store.records = [existing];
  const subject = engine([], store);
  const updated = await subject.recordFeedback("rated", "steer-token", "useful", "Avoided rediscovering the location");
  assert.equal(updated?.confidence, 0.6);
  assert.equal(updated?.feedback?.useful, 1);
  assert.equal(updated?.feedback?.history[0]?.steerToken, "steer-token");
  assert.equal(updated?.feedback?.history[0]?.sessionId, "session");
});

test("user-approved compaction preserves authority and provenance while superseding sources", async () => {
  const store = new Store();
  const first = { ...memory("first", "old-a", "2020-01-01T00:00:00Z"), confidence: 0.8, priority: 0.7 };
  const second = {
    ...memory("second", "old-b", "2021-01-01T00:00:00Z"),
    confidence: 0.6,
    priority: 1,
    sourceHistory: [{ actor: "user" as const, sessionId: "old-c", cwd: "/cwd", cause: "older", reason: "history" }],
    feedback: { useful: 2, unhelpful: 1, lastAt: "2025-01-01T00:00:00Z", history: [] },
  };
  store.records = [first, second];
  const subject = engine({ accept: true, reason: "Entailed duplicate" }, store);
  const compacted = await subject.applyCompaction(
    { enabled: true, sourceIds: ["first", "second"], text: "Canonical durable fact.", reason: "Exact duplicates" },
    { records: [first, second], minimumSimilarity: 0.95 },
  );
  assert.equal(compacted.source.actor, "user");
  assert.equal(compacted.source.cause, "user_invoked_compaction");
  assert.equal(compacted.confidence, 0.8);
  assert.equal(compacted.decayRate, DEFAULT_CONFIG.memoryLifecycle.decay.initialRate);
  assert.equal(compacted.priority, 1);
  assert.equal(compacted.createdAt, "2020-01-01T00:00:00Z");
  assert.deepEqual(compacted.feedback && { useful: compacted.feedback.useful, unhelpful: compacted.feedback.unhelpful }, { useful: 2, unhelpful: 1 });
  assert.deepEqual(compacted.supersedes, ["first", "second"]);
  assert.deepEqual(compacted.sourceHistory?.map((source) => source.sessionId), ["old-a", "old-c", "old-b"]);
  assert.equal(store.records.find((record) => record.id === "first")?.status, "superseded");
  assert.equal(store.records.find((record) => record.id === "second")?.status, "superseded");
  assert.equal(store.records.find((record) => record.id === compacted.id)?.status, "active");
});

test("compaction refuses to cross user and assistant authority", async () => {
  const store = new Store();
  const user = memory("user", "old", "2020-01-01T00:00:00Z");
  const assistant = { ...memory("assistant", "old", "2020-01-01T00:00:00Z"), source: { actor: "assistant" as const, sessionId: "old", cwd: "/cwd", cause: "test", reason: "fixture" } };
  const subject = engine({ accept: true }, store);
  await assert.rejects(
    subject.applyCompaction(
      { enabled: true, sourceIds: ["user", "assistant"], text: "Unsafe merge.", reason: "test" },
      { records: [user, assistant], minimumSimilarity: 1 },
    ),
    /authority boundaries/,
  );
});

test("recall removes frequency-limited memories before relevance judgment", async () => {
  const store = new Store();
  const repeated = memory("repeated", "older-session", "2020-01-01T00:00:00Z");
  const available = memory("available", "older-session", "2020-01-01T00:00:00Z");
  store.matches = [{ record: repeated, score: 0.95 }, { record: available, score: 0.9 }];
  const subject = engine([
    { query: "relevant facts" },
    { relevantIds: ["repeated", "available"], instruction: "Use the available memory", reason: "Relevant" },
  ], store);

  const recalled = await subject.recall("current task", undefined, new Set(["repeated"]));
  assert.deepEqual(recalled?.relevant.map((match) => match.record.id), ["available"]);
});
