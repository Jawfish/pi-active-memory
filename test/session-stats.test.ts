import test from "node:test";
import assert from "node:assert/strict";
import { formatSessionStats, SESSION_STATS_ENTRY_TYPE, sessionStatsFromEntries } from "../src/session-stats.js";

test("session stats restore valid events from the active branch", () => {
  const entries = [
    { type: "custom", customType: SESSION_STATS_ENTRY_TYPE, data: { name: "memoriesCreated", amount: 1 } },
    { type: "custom", customType: SESSION_STATS_ENTRY_TYPE, data: { name: "recallAttempts", amount: 2 } },
    { type: "custom", customType: SESSION_STATS_ENTRY_TYPE, data: { name: "memorySteers", amount: 1 } },
    { type: "custom", customType: SESSION_STATS_ENTRY_TYPE, data: { name: "useful", amount: 1 } },
    { type: "custom", customType: SESSION_STATS_ENTRY_TYPE, data: { name: "unhelpful", amount: 1 } },
    { type: "custom", customType: SESSION_STATS_ENTRY_TYPE, data: { name: "unknown", amount: 99 } },
    { type: "message", message: { role: "user", content: "ignored" } },
  ];

  assert.deepEqual(sessionStatsFromEntries(entries), {
    memoriesCreated: 1,
    recallAttempts: 2,
    memorySteers: 1,
    useful: 1,
    unhelpful: 1,
  });
});

test("session stats render the requested counters", () => {
  assert.equal(formatSessionStats({ memoriesCreated: 2, recallAttempts: 5, memorySteers: 3, useful: 1, unhelpful: 2 }), [
    "Current session memory stats",
    "Memories created: 2",
    "Recall attempts: 5",
    "Memory steers: 3",
    "Useful: 1",
    "Not useful: 2",
  ].join("\n"));
});
