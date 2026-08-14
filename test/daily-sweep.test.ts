import test from "node:test";
import assert from "node:assert/strict";
import { DAILY_SWEEP_POLL_INTERVAL_MS, DailySweepGate, utcDateKey } from "../src/daily-sweep.js";

test("active sessions poll lifecycle decay hourly", () => {
  assert.equal(DAILY_SWEEP_POLL_INTERVAL_MS, 60 * 60_000);
});

test("UTC date keys cross only at the UTC day boundary", () => {
  assert.equal(utcDateKey(new Date("2026-01-01T23:59:59.999Z")), "2026-01-01");
  assert.equal(utcDateKey(new Date("2026-01-02T00:00:00.000Z")), "2026-01-02");
});

test("daily sweep gate claims one sweep per UTC date", () => {
  const gate = new DailySweepGate(new Date("2026-01-01T12:00:00Z"));
  assert.equal(gate.claim(new Date("2026-01-01T23:59:00Z")), undefined);
  assert.equal(gate.claim(new Date("2026-01-02T00:01:00Z")), "2026-01-02");
  assert.equal(gate.claim(new Date("2026-01-02T12:00:00Z")), undefined);
  gate.complete("2026-01-02", true);
  assert.equal(gate.claim(new Date("2026-01-02T23:00:00Z")), undefined);
  assert.equal(gate.claim(new Date("2026-01-04T00:01:00Z")), "2026-01-04");
});

test("failed daily sweep can be retried on the same date", () => {
  const gate = new DailySweepGate(new Date("2026-01-01T12:00:00Z"));
  const date = gate.claim(new Date("2026-01-02T00:01:00Z"));
  assert.equal(date, "2026-01-02");
  gate.complete(date!, false);
  assert.equal(gate.claim(new Date("2026-01-02T00:02:00Z")), "2026-01-02");
});
