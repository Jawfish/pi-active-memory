import test from "node:test";
import assert from "node:assert/strict";
import { renderActiveMemoryStatus } from "../src/footer-status.js";
import { activeMemoryStatus, compactionProgressStatus } from "../src/status.js";

test("active memory status respects paused, error, recalling, and ready precedence", () => {
  assert.equal(activeMemoryStatus(true, "failed", true), "paused");
  assert.equal(activeMemoryStatus(false, "failed", true), "error");
  assert.equal(activeMemoryStatus(false, undefined, true), "recalling");
  assert.equal(activeMemoryStatus(false, undefined, false), "ready");
});

test("recall completion resolves the footer to ready or error", () => {
  assert.equal(activeMemoryStatus(false, undefined, false), "ready");
  assert.equal(activeMemoryStatus(false, "recall failed", false), "error");
  assert.notEqual(activeMemoryStatus(false, undefined, false), "recalling");
  assert.notEqual(activeMemoryStatus(false, "recall failed", false), "recalling");
});

test("active memory state renders beside model info without consuming a status row", () => {
  const lines = renderActiveMemoryStatus(
    ["~/project", "usage                 gpt-5.6-sol • medium", "other:ready"],
    "memory:ready",
  );

  assert.deepEqual(lines, ["~/project", "usage    gpt-5.6-sol • medium memory:ready", "other:ready"]);
});

test("active memory state leaves a footer unchanged when it cannot fit inline", () => {
  assert.deepEqual(renderActiveMemoryStatus(["usage model"], "memory:ready"), ["usage model"]);
});

test("compaction progress exposes processing and terminal states", () => {
  assert.equal(compactionProgressStatus("processing"), "memory-compact:processing");
  assert.equal(compactionProgressStatus("reviewing", 2, 4), "memory-compact:reviewing 2/4");
  assert.equal(compactionProgressStatus("applying", 2, 4), "memory-compact:applying 2/4");
  assert.equal(compactionProgressStatus("completed"), "memory-compact:completed");
  assert.equal(compactionProgressStatus("cancelled"), "memory-compact:cancelled");
  assert.equal(compactionProgressStatus("error"), "memory-compact:error");
});
