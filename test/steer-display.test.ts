import test from "node:test";
import assert from "node:assert/strict";
import { formatSteerSentence, orderedFeedbackOutcomes } from "../src/steer-display.js";

test("memory steer text is collapsed to one sentence", () => {
  assert.equal(formatSteerSentence("  Preserve width\nwhile rendering! "), "Preserve width while rendering.");
  assert.equal(formatSteerSentence(""), "");
});

test("feedback outcomes follow memory order and omit unrated memories", () => {
  const outcomes = new Map([
    ["second", "unhelpful" as const],
    ["first", "useful" as const],
  ]);
  assert.deepEqual(orderedFeedbackOutcomes(["first", "missing", "second"], outcomes), ["useful", "unhelpful"]);
});
