import test from "node:test";
import assert from "node:assert/strict";
import { assistantExtractionPrompt, assistantValidationPrompt, extractionPrompt, mergePrompt, validationPrompt } from "../src/prompts.js";

test("extraction prompt makes the user message the only evidence source", () => {
  const prompt = extractionPrompt("I prefer it.", "assistant: it refers to early returns", "project");
  assert.match(prompt, /NEWEST USER MESSAGE \(the only allowed source\)/);
  assert.match(prompt, /CURRENT CONTEXT \(interpretation only; never a source\)/);
  assert.match(prompt, /user_profile\|fact\|skill_workflow/);
  assert.match(prompt, /next steps, TODOs/);
  assert.match(prompt, /Never store instructions/);
  assert.match(prompt, /always use pnpm in this repository/);
  assert.match(prompt, /Do not convert an instruction into a preference, convention, fact, or workflow/);
  assert.match(prompt, /one terse, self-contained sentence/);
});

test("validation prompt permits context only for reference resolution", () => {
  const prompt = validationPrompt("I prefer it.", "assistant: it refers to early returns", {
    text: "The user prefers early returns.", kind: "user_profile", scope: "global", evidence: "I prefer it",
  });
  assert.match(prompt, /CONTEXT \(reference resolution only\)/);
  assert.match(prompt, /USER MESSAGE \(only source\)/);
  assert.match(prompt, /must not supply the claim itself/);
  assert.match(prompt, /Reject every command, request, instruction/);
  assert.match(prompt, /Never rewrite an imperative instruction/);
});

test("assistant prompts require costly investigation rather than duration alone", () => {
  const extraction = assistantExtractionPrompt("assistant result", "investigate parser", 70_000, "project");
  assert.match(extraction, /Duration alone|one-minute duration/);
  assert.match(extraction, /simple search results/);
  assert.match(extraction, /whyStored/);
  assert.match(extraction, /Store findings, not instructions/);
  assert.match(extraction, /project work instructions/);
  assert.match(extraction, /one terse, self-contained sentence/);
  assert.doesNotMatch(extraction, /user_profile\|fact/);

  const validation = assistantValidationPrompt("assistant result", "investigate parser", 70_000, {
    text: "result", kind: "fact", scope: "project", evidence: "result", whyStored: "required tracing",
  });
  assert.match(validation, /Duration alone is insufficient/);
  assert.match(validation, /simple-search/);
  assert.match(validation, /Reject all commands, project instructions/);
});

test("merge prompt requires pre-write resolution and protects user authority", () => {
  const prompt = mergePrompt("candidate", "existing", "assistant");
  assert.match(prompt, /searched before this write/);
  assert.match(prompt, /lower authority/);
  assert.match(prompt, /add\|replace\|noop/);
  assert.match(prompt, /one terse, self-contained sentence/);
});
