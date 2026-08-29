import test from "node:test";
import assert from "node:assert/strict";
import { PiFastModel } from "../src/fast-model.js";

test("Pi fast model uses simple completion so subscription providers map disabled reasoning correctly", async () => {
  let receivedOptions: Record<string, unknown> | undefined;
  let tokenUsage: { input: number; output: number } | undefined;
  const model = { provider: "openai-codex", id: "gpt-test", api: "openai-codex-responses" };
  const ctx = {
    modelRegistry: {
      find: (provider: string, id: string) => provider === model.provider && id === model.id ? model : undefined,
      getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "oauth-token", headers: { "x-test": "yes" } }),
    },
  };
  const complete = async (_model: unknown, _context: unknown, options: Record<string, unknown>) => {
    receivedOptions = options;
    return {
      role: "assistant",
      content: [{ type: "text", text: "{\"ok\":true}" }],
      stopReason: "stop",
      usage: { input: 80, output: 20 },
    };
  };
  const runner = new PiFastModel({ candidates: ["openai-codex/gpt-test"], thinking: "off", maxTokens: 100 }, ctx as never, complete as never);
  runner.onTokenUsage((usage) => { tokenUsage = usage; });

  assert.deepEqual(await runner.json<{ ok: boolean }>("system", "prompt"), { ok: true });
  assert.equal(receivedOptions?.reasoning, undefined);
  assert.equal(receivedOptions?.apiKey, "oauth-token");
  assert.equal(runner.selectedModel(), "openai-codex/gpt-test");
  assert.deepEqual(tokenUsage, { input: 80, output: 20 });
});

test("Pi fast model falls back when a provider returns an error message", async () => {
  const models = new Map([
    ["openai-codex/bad", { provider: "openai-codex", id: "bad", api: "openai-codex-responses" }],
    ["openai/good", { provider: "openai", id: "good", api: "openai-responses" }],
  ]);
  const ctx = {
    modelRegistry: {
      find: (provider: string, id: string) => models.get(`${provider}/${id}`),
      getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "token" }),
    },
  };
  let calls = 0;
  const complete = async () => ++calls === 1
    ? { role: "assistant", content: [], stopReason: "error", errorMessage: "rejected" }
    : { role: "assistant", content: [{ type: "text", text: "{\"ok\":true}" }], stopReason: "stop" };
  const runner = new PiFastModel({ candidates: [...models.keys()], thinking: "low", maxTokens: 100 }, ctx as never, complete as never);

  assert.deepEqual(await runner.json<{ ok: boolean }>("system", "prompt"), { ok: true });
  assert.equal(runner.selectedModel(), "openai/good");
});
