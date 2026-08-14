import test from "node:test";
import assert from "node:assert/strict";
import { applyConfidenceFeedback, SteerFeedbackLedger } from "../src/feedback.js";
import type { MemoryRecord } from "../src/types.js";

function memory(confidence = 0.5): MemoryRecord {
  return {
    id: "memory",
    text: "Durable fact.",
    kind: "fact",
    scope: "global",
    confidence,
    status: "active",
    source: { actor: "user", sessionId: "old", cwd: "/cwd", cause: "test", reason: "fixture" },
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
    embeddingModel: "test",
    schemaVersion: 2,
  };
}

const config = {
  enabled: true,
  confidence: { initial: 0.5, deletionThreshold: 0.1, minimum: 0.05, maximum: 0.95, usefulDelta: 0.1, unhelpfulDelta: 0.15 },
  decay: { initialRate: 0.28, minimumRate: 0, maximumRate: 0.95, usefulDelta: 0.05 },
  feedback: { maxPerMemoryPerSession: 2, historyLimit: 2 },
};

test("steer feedback ledger binds feedback to a steer and prevents reinforcement loops", () => {
  const ledger = new SteerFeedbackLedger();
  ledger.register("token-a", ["memory"]);
  ledger.register("token-b", ["memory"]);
  ledger.register("token-c", ["memory"]);
  assert.equal(ledger.consume("unknown", "memory", 2), "unknown_token");
  assert.equal(ledger.consume("token-a", "other", 2), "not_in_steer");
  assert.equal(ledger.consume("token-a", "memory", 2), "accepted");
  assert.equal(ledger.consume("token-a", "memory", 2), "duplicate");
  ledger.release("token-a", "memory");
  assert.equal(ledger.consume("token-a", "memory", 2), "accepted");
  assert.equal(ledger.consume("token-b", "memory", 2), "accepted");
  assert.equal(ledger.consume("token-c", "memory", 2), "session_limit");
});

test("useful and unhelpful feedback adjust bounded confidence and retain bounded provenance", () => {
  const useful = applyConfidenceFeedback(memory(0.9), {
    outcome: "useful", sessionId: "session", steerToken: "one", reason: "Prevented a repeat investigation", at: "2026-02-01T00:00:00Z",
  }, config);
  assert.equal(useful.confidence, 0.95);
  assert.ok(Math.abs(useful.decayRate! - 0.23) < 1e-12);
  const unhelpful = applyConfidenceFeedback(useful, {
    outcome: "unhelpful", sessionId: "session", steerToken: "two", reason: "Pointed to a stale path", at: "2026-02-02T00:00:00Z",
  }, config);
  const latest = applyConfidenceFeedback(unhelpful, {
    outcome: "unhelpful", sessionId: "session-2", steerToken: "three", reason: "Still stale", at: "2026-02-03T00:00:00Z",
  }, config);
  assert.ok(Math.abs(latest.confidence - 0.65) < 1e-12);
  assert.equal(unhelpful.decayRate, useful.decayRate);
  assert.equal(latest.decayRate, useful.decayRate);
  assert.equal(latest.feedback?.useful, 1);
  assert.equal(latest.feedback?.unhelpful, 2);
  assert.deepEqual(latest.feedback?.history.map((feedback) => feedback.steerToken), ["two", "three"]);
});
