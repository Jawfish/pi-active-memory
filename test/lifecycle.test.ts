import test from "node:test";
import assert from "node:assert/strict";
import { advanceMemoryLifecycle, initializeMemoryLifecycle, reinforceMemoryLifecycle } from "../src/lifecycle.js";
import type { MemoryLifecycleConfig, MemoryRecord } from "../src/types.js";

const config: MemoryLifecycleConfig = {
  enabled: true,
  confidence: { initial: 0.5, deletionThreshold: 0.1, minimum: 0.05, maximum: 0.95, usefulDelta: 0.1, unhelpfulDelta: 0.15 },
  decay: { initialRate: 0.28, minimumRate: 0, maximumRate: 0.95, usefulDelta: 0.05 },
  feedback: { maxPerMemoryPerSession: 2, historyLimit: 50 },
};

function memory(overrides: Partial<MemoryRecord> = {}): MemoryRecord {
  return {
    id: "memory",
    text: "Durable fact.",
    kind: "fact",
    scope: "global",
    confidence: 0.5,
    status: "active",
    source: { actor: "user", sessionId: "old", cwd: "/cwd", cause: "test", reason: "fixture" },
    createdAt: "2020-01-01T00:00:00Z",
    updatedAt: "2020-01-01T00:00:00Z",
    embeddingModel: "test",
    schemaVersion: 2,
    ...overrides,
  };
}

test("legacy memories receive a fresh daily-decay clock without retroactive expiry", () => {
  const result = advanceMemoryLifecycle(memory({ confidence: 0.01 }), new Date("2026-01-01T12:00:00Z"), "session-1", config);
  assert.equal(result.initialized, true);
  assert.equal(result.expiredBy, undefined);
  assert.equal(result.record.status, "active");
  assert.equal(result.record.decayRate, 0.28);
  assert.equal(result.record.lifecycle?.lastDecayDate, "2026-01-01");
  assert.equal(result.record.lifecycle?.lastReinforcementCause, "legacy_migration_grace");
});

test("default confidence falls below deletion threshold after five unused days", () => {
  const created = initializeMemoryLifecycle(memory(), new Date("2026-01-01T23:00:00Z"), "session-1", config);
  const result = advanceMemoryLifecycle(created, new Date("2026-01-06T00:01:00Z"), "session-2", config);
  assert.equal(result.decayedDays, 5);
  assert.ok(result.record.confidence < config.confidence.deletionThreshold);
  assert.equal(result.expiredBy, "low_confidence");
  assert.equal(result.record.status, "deleted");
  assert.equal(result.record.lifecycle?.deletionCause, "low_confidence");
});

test("decay is applied only on the first sweep of a UTC day", () => {
  const created = initializeMemoryLifecycle(memory(), new Date("2026-01-01T10:00:00Z"), "session-1", config);
  const first = advanceMemoryLifecycle(created, new Date("2026-01-02T00:01:00Z"), "session-2", config);
  assert.equal(first.decayedDays, 1);
  assert.equal(first.record.confidence, 0.5 * 0.72);
  const sameDay = advanceMemoryLifecycle(first.record, new Date("2026-01-02T23:59:00Z"), "session-3", config);
  assert.equal(sameDay.decayedDays, 0);
  assert.equal(sameDay.record.confidence, first.record.confidence);
});

test("useful reinforcement resets the daily clock without restoring lost confidence", () => {
  const initial = initializeMemoryLifecycle(memory(), new Date("2026-01-01T00:00:00Z"), "session-1", config);
  const decayed = advanceMemoryLifecycle(initial, new Date("2026-01-03T00:00:00Z"), "session-2", config).record;
  const reinforced = reinforceMemoryLifecycle(decayed, new Date("2026-01-03T12:00:00Z"), "session-2", config, "useful_feedback");
  assert.equal(reinforced.confidence, decayed.confidence);
  assert.equal(reinforced.lifecycle?.lastDecayDate, "2026-01-03");
  assert.equal(reinforced.lifecycle?.lastRelevantSessionId, "session-2");
  assert.equal(reinforced.lifecycle?.reinforcementCount, 1);
});
