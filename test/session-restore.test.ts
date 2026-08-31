import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_CONFIG } from "../src/config.js";
import { SteerFeedbackLedger } from "../src/feedback.js";
import { restoreSessionState } from "../src/session-restore.js";
import { MemorySteerLimiter } from "../src/steer-frequency.js";

test("malformed restored steer details are ignored before fingerprinting", () => {
  const config = structuredClone(DEFAULT_CONFIG);
  const ledger = new SteerFeedbackLedger();
  const limiter = new MemorySteerLimiter(config.recall);
  const result = restoreSessionState([
    { type: "custom_message", customType: "active-memory-steer", details: { feedbackToken: "t", memoryIds: ["a"], instruction: { private: true } } },
    { type: "custom_message", customType: "active-memory-steer", details: { feedbackToken: "t2", memoryIds: [], instruction: "empty ids" } },
    { type: "custom_message", customType: "active-memory-steer", details: { feedbackToken: "t3", memoryIds: ["a"], scores: [Number.NaN], reason: "bad", instruction: "bad score", projectId: "p", source: "active-memory" } },
    { type: "custom_message", customType: "active-memory-steer", details: { feedbackToken: "t4", memoryIds: ["a"] } },
    { type: "custom_message", customType: "active-memory-steer", details: { feedbackToken: "t5", memoryIds: ["a"], scores: [0.9], reason: "relevant", instruction: "Remember A", projectId: "p", source: "active-memory" } },
    { type: "custom_message", customType: "active-memory-steer", timestamp: Date.now() + 365 * 24 * 60 * 60_000, details: { feedbackToken: "t6", memoryIds: ["a"], scores: [0.9], reason: "relevant", instruction: "Remember A", projectId: "p", source: "active-memory" } },
  ], config, ledger, limiter, () => {}, detail => `${detail.memoryIds.join(",")}|${detail.instruction!.toLowerCase()}`);
  assert.equal(result.lastRecall, undefined);
  assert.equal(result.lastSteerFingerprint, "");
  assert.equal(ledger.consume("t", "a", 1), "unknown_token");
  assert.equal(ledger.consume("t5", "a", 1), "unknown_token");
  assert.equal(ledger.consume("t6", "a", 1), "unknown_token");
});

test("custom_message steers restore feedback, limiter, dedup, and displayed outcomes", () => {
  const config = structuredClone(DEFAULT_CONFIG);
  config.memoryLifecycle.feedback.maxPerMemoryPerSession = 1;
  config.recall.maxSteersPerMemoryPerSession = 1;
  const ledger = new SteerFeedbackLedger();
  const limiter = new MemorySteerLimiter(config.recall);
  const displayed: string[] = [];
  const result = restoreSessionState([
    { type: "message", message: { role: "user" } },
    { type: "message", message: { role: "assistant" } },
    { type: "custom_message", customType: "active-memory-steer", timestamp: 1000, details: { feedbackToken: "t", memoryIds: ["a"], scores: [0.9], reason: "relevant", instruction: "Remember A", projectId: "p", source: "active-memory" } },
    { type: "message", message: { role: "toolResult", toolName: "memory_feedback", details: { accepted: true, steerToken: "t", memoryId: "a", outcome: "forged" } } },
    { type: "message", message: { role: "toolResult", toolName: "memory_feedback", details: { accepted: true, steerToken: "t", memoryId: "a", outcome: "useful" } } },
    { type: "message", message: { role: "toolResult", toolName: "memory_feedback", details: { accepted: true, steerToken: "t", memoryId: "a", outcome: "unhelpful" } } },
  ], config, ledger, limiter, (token, id, outcome) => displayed.push(`${token}/${id}/${outcome}`), detail => `${detail.memoryIds.join(",")}|${detail.instruction}`);
  assert.equal(result.turnSequence, 1);
  assert.equal(result.lastSteerAt, 1000);
  assert.equal(result.lastSteerFingerprint, "a|Remember A");
  assert.deepEqual(displayed, ["t/a/useful"]);
  assert.equal(ledger.consume("t", "a", 1), "duplicate");
  assert.ok(limiter.suppressedIds(2000, 2).has("a"));
});
