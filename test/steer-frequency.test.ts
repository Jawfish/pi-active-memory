import test from "node:test";
import assert from "node:assert/strict";
import { MemorySteerLimiter } from "../src/steer-frequency.js";

const policy = {
  perMemoryCooldownMs: 1_000,
  perMemoryTurnCooldown: 3,
  maxSteersPerMemoryPerSession: 2,
};

test("memory steer limiter enforces time and turn cooldowns", () => {
  const limiter = new MemorySteerLimiter(policy);
  limiter.record(["a"], 1_000, 2);

  assert.deepEqual([...limiter.suppressedIds(1_999, 20)], ["a"], "time cooldown remains active");
  assert.deepEqual([...limiter.suppressedIds(5_000, 4)], ["a"], "turn cooldown remains active");
  assert.deepEqual([...limiter.suppressedIds(5_000, 5)], [], "both cooldowns elapsed");
});

test("memory steer limiter caps deliveries per session and deduplicates IDs", () => {
  const limiter = new MemorySteerLimiter(policy);
  limiter.record(["a", "a"], 0, 0);
  assert.deepEqual([...limiter.suppressedIds(2_000, 3)], []);

  limiter.record(["a"], 2_000, 3);
  assert.deepEqual([...limiter.suppressedIds(20_000, 30)], ["a"]);

  limiter.clear();
  assert.deepEqual([...limiter.suppressedIds(20_000, 30)], []);
});
