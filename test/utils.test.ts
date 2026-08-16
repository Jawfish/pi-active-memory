import test from "node:test";
import assert from "node:assert/strict";
import { boundedAssistantInvestigation, cosineSimilarity, evidenceAppearsInUserMessage, isTransientTaskMemory, parseJsonResponse, redactSecrets, sourceEvidenceAppearsInContext, stableProjectId } from "../src/utils.js";

test("cosine similarity ranks identical vectors", () => {
  assert.equal(cosineSimilarity([1, 0], [1, 0]), 1);
  assert.equal(cosineSimilarity([1, 0], [0, 1]), 0);
});

test("parses fenced model JSON", () => {
  assert.deepEqual(parseJsonResponse<{ ok: boolean }>("```json\n{\"ok\":true}\n```"), { ok: true });
});

test("redacts common secrets", () => {
  assert.doesNotMatch(redactSecrets("api_key=abcdefghijklmno"), /abcdefghijklmno/);
  assert.doesNotMatch(redactSecrets("ghp_abcdefghijklmnopqrstuvwxyz"), /ghp_/);
});

test("assistant investigation context excludes user messages", () => {
  const context = boundedAssistantInvestigation([
    { type: "message", message: { role: "user", content: [{ type: "text", text: "user-only claim" }] } },
    { type: "message", message: { role: "toolResult", toolName: "search", content: [{ type: "text", text: "hard-won search result" }] } },
    { type: "message", message: { role: "assistant", content: [{ type: "text", text: "investigation conclusion" }] } },
  ], 1000);
  assert.doesNotMatch(context, /user-only claim/);
  assert.match(context, /hard-won search result/);
  assert.match(context, /investigation conclusion/);
});

test("memory evidence must appear in the explicit user message", () => {
  assert.equal(evidenceAppearsInUserMessage("favourite color is orange", "My favourite   color is orange."), true);
  assert.equal(evidenceAppearsInUserMessage("use tabs", "Please fix the formatter"), false);
});

test("source context suppression recognizes stored user evidence", () => {
  const record = { source: { userText: "I've installed pi-tasks; let's try it for now.", evidence: "let's try it for now" } };
  assert.equal(sourceEvidenceAppearsInContext(record, "user: I've installed pi-tasks; let's try it for now."), true);
  assert.equal(sourceEvidenceAppearsInContext(record, "user: Update the task widget."), false);
});

test("transient task memory detection rejects temporary trials", () => {
  assert.equal(isTransientTaskMemory("The user wants to try pi-tasks for now.", "let's try it for now"), true);
  assert.equal(isTransientTaskMemory("The user manages settings through cfg.", "I manage settings through cfg"), false);
});

test("project ids are stable and remote-sensitive", () => {
  assert.equal(stableProjectId("/x", "git@example/a"), stableProjectId("/x", "git@example/a"));
  assert.notEqual(stableProjectId("/x", "git@example/a"), stableProjectId("/x", "git@example/b"));
  assert.equal(stableProjectId("/worktree/a", "git@example/a"), stableProjectId("/worktree/b", "git@example/a"));
});
