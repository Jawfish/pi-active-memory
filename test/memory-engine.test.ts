import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_CONFIG } from "../src/config.js";
import { isRecentCurrentSessionMemory, MemoryEngine, rankMemoryMatches } from "../src/memory-engine.js";
import type { EmbeddingProvider, FastModelRunner, MemoryFilter, MemoryMatch, MemoryRecord, VectorStore } from "../src/types.js";

class Store implements VectorStore {
  readonly contractVersion = 2 as const;
  records: MemoryRecord[] = [];
  matches: MemoryMatch[] = [];
  lastSearchVector?: number[];
  lastUpsertVector?: number[];
  beforeMutate?: (id: string) => void;
  async initialize(): Promise<void> {}
  async get(id: string): Promise<MemoryRecord | undefined> { return this.records.find(record => record.id === id); }
  async insert(row: { record: MemoryRecord; vector: number[] }): Promise<"inserted" | "exists"> { if (await this.get(row.record.id)) return "exists"; this.lastUpsertVector = row.vector; this.records.push(row.record); return "inserted"; }
  async mutate(id: string, apply: (latest: Readonly<MemoryRecord>) => { record: MemoryRecord; vector?: number[] } | undefined) {
    this.beforeMutate?.(id);
    const index = this.records.findIndex(record => record.id === id);
    if (index < 0) return { status: "missing" as const };
    const result = apply(this.records[index]!);
    if (!result) return { status: "unchanged" as const, record: this.records[index]! };
    this.records[index] = result.record;
    if (result.vector) this.lastUpsertVector = result.vector;
    return { status: "updated" as const, record: result.record };
  }
  async compact(sourceIds: readonly string[], build: (latest: readonly MemoryRecord[]) => { record: MemoryRecord; vector: number[] }): Promise<MemoryRecord> { const sources = this.records.filter(record => sourceIds.includes(record.id)); const row = build(sources); this.records = this.records.map(record => sourceIds.includes(record.id) ? { ...record, status: "superseded" as const } : record); this.records.push(row.record); this.lastUpsertVector = row.vector; return row.record; }
  async scan(_filter: MemoryFilter, visit: (page: readonly MemoryRecord[]) => Promise<void>): Promise<number> { await visit(this.records); return this.records.length; }
  async rebuildVectors(_dimension: number, buildPage: (page: readonly MemoryRecord[]) => Promise<readonly { record: MemoryRecord; vector: number[] }[]>): Promise<number> { const rows = await buildPage(this.records); this.records = rows.map(row => row.record); return rows.length; }
  async search(vector: number[], _filter: MemoryFilter, _limit: number): Promise<MemoryMatch[]> { this.lastSearchVector = vector; return this.matches; }
  async list(): Promise<MemoryRecord[]> { return this.records; }
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

function engine(
  response: unknown | unknown[],
  store: Store,
  embedding: EmbeddingProvider = embedder,
  onMemoryStored?: (record: Readonly<MemoryRecord>, created: boolean) => void,
): MemoryEngine {
  return new MemoryEngine(DEFAULT_CONFIG, store, embedding, new Fast(Array.isArray(response) ? response : [response]), "project", "session", "/cwd", undefined, onMemoryStored);
}

test("activity events never persist adapter-reported model identifiers", async () => {
  const events: unknown[] = [];
  const subject = new MemoryEngine(DEFAULT_CONFIG, new Store(), embedder, new Fast([{ memories: [] }]), "project", "session", "/cwd", (_event, data) => events.push(data));
  await subject.capture("This message is deliberately long enough to run extraction.", "context");
  assert.doesNotMatch(JSON.stringify(events), /test\/fast/);
});

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

test("capture reports the committed canonical memory to a display-only sink", async () => {
  const store = new Store();
  const notices: Array<{ text: string; created: boolean }> = [];
  const subject = new MemoryEngine(
    DEFAULT_CONFIG,
    store,
    embedder,
    new Fast([
      { memories: [{ text: "The user's favourite color is orange.", kind: "user_profile", scope: "global", confidence: 0.99, evidence: "favourite color is orange" }] },
      { accept: true, reason: "Explicit durable profile preference" },
    ]),
    "project",
    "session",
    "/cwd",
    undefined,
    (record, created) => notices.push({ text: record.text, created }),
  );

  assert.equal(await subject.capture("My favourite color is orange.", ""), 1);
  assert.deepEqual(notices, [{ text: "The user's favourite color is orange.", created: true }]);
});

test("display feedback failures cannot turn a committed write into a capture failure", async () => {
  const store = new Store();
  const subject = new MemoryEngine(
    DEFAULT_CONFIG,
    store,
    embedder,
    new Fast([
      { memories: [{ text: "The user's favourite color is orange.", kind: "user_profile", scope: "global", confidence: 0.99, evidence: "favourite color is orange" }] },
      { accept: true, reason: "Explicit durable profile preference" },
    ]),
    "project",
    "session",
    "/cwd",
    undefined,
    () => { throw new Error("TUI unavailable"); },
  );

  assert.equal(await subject.capture("My favourite color is orange.", ""), 1);
  assert.equal(store.records.length, 1);
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
  store.records = [existing];
  store.matches = [{ record: existing, score: 0.95 }];
  const notices: Array<{ id: string; created: boolean }> = [];
  const subject = engine([
    { memories: [{ text: "The parser registry is in src/parser-registry.ts.", kind: "fact", scope: "global", confidence: 0.7, evidence: "parser registry moved to src/parser-registry.ts", whyStored: "Locating the move required tracing generated imports." }] },
    { accept: true, reason: "Non-trivial trace" },
    { action: "replace", targetId: "existing", text: "The parser registry is in src/parser-registry.ts." },
  ], store, embedder, (record, created) => notices.push({ id: record.id, created }));
  assert.equal(await subject.captureAssistantInvestigation("The parser registry moved to src/parser-registry.ts after tracing generated imports.", "Locate parser registry", 80_000), 1);
  assert.equal(store.records[0]?.id, "existing");
  assert.equal(store.records[0]?.sourceHistory?.[0]?.cause, "old investigation");
  assert.equal(store.records[0]?.source.cause, "Locate parser registry");
  assert.deepEqual(notices, [{ id: "existing", created: false }]);
});

test("assistant memory correction replaces text and preserves provenance", async () => {
  const store = new Store();
  const existing = {
    ...memory("assistant-memory", "older-session", "2020-01-01T00:00:00Z"),
    text: "The parser registry is in src/old-registry.ts.",
    confidence: 0.9,
    priority: 0.8,
    source: { actor: "assistant" as const, sessionId: "older-session", cwd: "/old", cause: "old investigation", reason: "old trace" },
  };
  store.records = [existing];
  const corrected = await engine({}, store).correctAssistantMemory(
    existing.id,
    "The parser registry is in src/parser-registry.ts.",
    "Repository inspection showed the old path no longer exists.",
  );
  assert.equal(corrected.id, existing.id);
  assert.equal(corrected.text, "The parser registry is in src/parser-registry.ts.");
  assert.equal(corrected.source.actor, "assistant");
  assert.equal(corrected.source.cause, "correction_of_inaccurate_assistant_memory");
  assert.equal(corrected.sourceHistory?.at(-1)?.cause, "old investigation");
  assert.equal(corrected.confidence, DEFAULT_CONFIG.assistantCapture.maximumConfidence);
  assert.equal(corrected.priority, DEFAULT_CONFIG.assistantCapture.priority);
  assert.equal(store.records[0]?.text, corrected.text);
});

test("assistant correction cannot follow a record into another project", async () => {
  const store = new Store();
  const existing = {
    ...memory("assistant-memory", "older-session", "2020-01-01T00:00:00Z"),
    text: "Old assistant claim.",
    source: { actor: "assistant" as const, sessionId: "older-session", cwd: "/old", cause: "old investigation", reason: "old trace" },
  };
  store.records = [existing];
  store.beforeMutate = () => { store.records[0] = { ...store.records[0]!, scope: "project", projectId: "other-project" }; };
  await assert.rejects(
    engine({}, store).correctAssistantMemory(existing.id, "Corrected assistant claim.", "The old claim is wrong."),
    /changed while correction was being prepared/,
  );
  assert.equal(store.records[0]?.text, "Old assistant claim.");
  assert.equal(store.records[0]?.projectId, "other-project");
});

test("assistant memory correction cannot alter user-sourced memory", async () => {
  const store = new Store();
  store.records = [memory("user-memory", "older-session", "2020-01-01T00:00:00Z")];
  await assert.rejects(
    engine({}, store).correctAssistantMemory("user-memory", "Replacement text.", "The old claim is wrong."),
    /Only assistant-generated memories can be corrected/,
  );
  assert.equal(store.records[0]?.text, "user-memory");
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

test("a user statement can replace a reviewed assistant memory with user authority", async () => {
  const store = new Store();
  const existing = {
    ...memory("assistant-memory", "older-session", "2020-01-01T00:00:00Z"),
    text: "Use yarn for this project.",
    source: { actor: "assistant" as const, sessionId: "older-session", cwd: "/cwd", cause: "investigation", reason: "inferred package manager" },
    priority: DEFAULT_CONFIG.assistantCapture.priority,
  };
  store.records = [existing];
  store.matches = [{ record: existing, score: 0.95 }];
  const subject = engine([
    { memories: [{ text: "Use pnpm for this project.", kind: "fact", scope: "global", confidence: 0.99, evidence: "Use pnpm for this project" }] },
    { accept: true, reason: "Explicit user correction" },
    { action: "replace", targetId: "assistant-memory", text: "Use pnpm for this project." },
  ], store);
  assert.equal(await subject.capture("Use pnpm for this project.", ""), 1);
  assert.equal(store.records[0]?.source.actor, "user");
  assert.equal(store.records[0]?.priority, 1);
  assert.equal(store.records[0]?.text, "Use pnpm for this project.");
});

test("replacement may change a reviewed memory kind", async () => {
  const store = new Store();
  const existing = { ...memory("workflow", "older-session", "2020-01-01T00:00:00Z"), kind: "fact" as const, text: "Run the formatter manually." };
  store.records = [existing];
  store.matches = [{ record: existing, score: 0.95 }];
  const subject = engine([
    { memories: [{ text: "Use the formatter before every commit.", kind: "skill_workflow", scope: "global", confidence: 0.99, evidence: "Use the formatter before every commit" }] },
    { accept: true, reason: "Explicit durable workflow" },
    { action: "replace", targetId: "workflow", text: "Use the formatter before every commit." },
  ], store);
  assert.equal(await subject.capture("Use the formatter before every commit.", ""), 1);
  assert.equal(store.records[0]?.kind, "skill_workflow");
});

test("replacement conflicts are re-resolved instead of silently losing capture", async () => {
  const store = new Store();
  const existing = memory("existing", "older-session", "2020-01-01T00:00:00Z");
  store.records = [existing];
  store.matches = [{ record: existing, score: 0.95 }];
  store.beforeMutate = () => {
    store.beforeMutate = undefined;
    const changed = { ...existing, text: "concurrently changed", updatedAt: "2026-01-02T00:00:00Z" };
    store.records = [changed];
    store.matches = [{ record: changed, score: 0.95 }];
  };
  const subject = engine([
    { memories: [{ text: "The durable value is new.", kind: "fact", scope: "global", confidence: 0.99, evidence: "durable value is new" }] },
    { accept: true, reason: "Explicit durable fact" },
    { action: "replace", targetId: "existing", text: "The durable value is new." },
    { action: "add", text: "The durable value is new." },
  ], store);
  assert.equal(await subject.capture("The durable value is new.", ""), 1);
  assert.equal(store.records.length, 2);
  assert.ok(store.records.some(item => item.text === "The durable value is new."));
});

test("ranking lowers assistant memories by confidence and priority", () => {
  const user = memory("user", "old", "2020-01-01T00:00:00Z");
  const assistant = { ...memory("assistant", "old", "2020-01-01T00:00:00Z"), confidence: 0.75, priority: 0.55, source: { actor: "assistant" as const, sessionId: "old", cwd: "/cwd", cause: "test", reason: "test fixture" } };
  const ranked = rankMemoryMatches([{ record: assistant, score: 0.99 }, { record: user, score: 0.8 }]);
  assert.deepEqual(ranked.map((match) => match.record.id), ["user", "assistant"]);
  assert.ok(ranked[1]!.score < 0.55);
});

test("recall uses query embeddings while stored memories use document embeddings", async () => {
  const dual: EmbeddingProvider = {
    queryModel: "test/query",
    documentModel: "test/document",
    embedQuery: async (texts) => texts.map(() => [1, 0]),
    embedDocuments: async (texts) => texts.map(() => [0, 1]),
  };
  const store = new Store();
  const existing = memory("existing", "old", "2020-01-01T00:00:00Z");
  store.matches = [{ record: existing, score: 0.9 }];
  await engine([{ query: "fact" }, { relevantIds: ["existing"], reason: "Relevant" }], store, dual).recall("current task");
  assert.deepEqual(store.lastSearchVector, [1, 0]);

  store.matches = [];
  await engine([
    { memories: [{ text: "The project uses a generated cache.", kind: "fact", scope: "project", confidence: 0.8, evidence: "uses a generated cache" }] },
    { accept: true, reason: "Durable project fact" },
  ], store, dual).capture("The project uses a generated cache.", "");
  assert.deepEqual(store.lastUpsertVector, [0, 1]);
  assert.equal(store.records.at(-1)?.embeddingModel, "test/document");
});

test("recall relevance is reinforced only after delivery is recorded", async () => {
  const store = new Store();
  const existing = memory("delivered", "older-session", "2020-01-01T00:00:00Z");
  store.records = [existing];
  store.matches = [{ record: existing, score: 0.9 }];
  const subject = engine([{ query: "durable" }, { relevantIds: ["delivered"], reason: "relevant" }], store);
  const recalled = await subject.recall("Need the durable fact", undefined, new Set(), "unrelated context");
  assert.ok(recalled);
  assert.equal(store.records[0]?.lifecycle?.reinforcementCount ?? 0, 0);
  await subject.recordRecallDelivery(recalled!.relevant.map(match => match.record));
  assert.equal(store.records[0]?.lifecycle?.reinforcementCount, 1);
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
    { relevantIds: ["recent", "old", "other"], reason: "Relevant" },
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
    { relevantIds: ["repeated", "available"], reason: "Relevant" },
  ], store);

  const recalled = await subject.recall("current task", undefined, new Set(["repeated"]));
  assert.deepEqual(recalled?.relevant.map((match) => match.record.id), ["available"]);
});
