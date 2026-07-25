import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { readFile } from "node:fs/promises";
import type { ActiveMemoryConfig } from "./types.js";

export const DEFAULT_CONFIG: ActiveMemoryConfig = {
  enabled: true,
  database: { provider: "json", path: join(homedir(), ".pi", "agent", "active-memory", "vectors.json") },
  // ChatGPT/Codex OAuth does not expose an embeddings API, so embeddings default to OpenAI API.
  embedding: { provider: "openai", model: "text-embedding-3-small", baseUrl: "https://api.openai.com/v1", apiKeyEnv: "OPENAI_API_KEY" },
  fastModel: {
    candidates: ["openai-codex/gpt-5.6-luna", "openai/gpt-5.6-luna", "openai/gpt-5.4-mini"],
    thinking: "off",
    maxTokens: 1200,
  },
  capture: { enabled: true, minCharacters: 8, contextCharacters: 12000, confidenceThreshold: 0.72, similarityThreshold: 0.82 },
  assistantCapture: { enabled: true, minimumElapsedMs: 60000, contextCharacters: 20000, confidenceThreshold: 0.62, maximumConfidence: 0.75, priority: 0.55, similarityThreshold: 0.78 },
  recall: { enabled: true, topK: 10, contextCharacters: 16000, everyTurns: 2, everyToolResults: 4, thinkingCharacters: 1200, cooldownMs: 15000, minVectorScore: 0.28, minimumMemoryAgeMinutes: 30 },
  security: { redactSecrets: true, maxMemoryCharacters: 1200 },
  activityLog: { enabled: true, includeText: true },
};

type JsonObject = Record<string, unknown>;
function merge<T>(base: T, patch: unknown): T {
  if (!patch || typeof patch !== "object" || Array.isArray(patch)) return base;
  const out: JsonObject = { ...(base as JsonObject) };
  for (const [key, value] of Object.entries(patch as JsonObject)) {
    const current = out[key];
    out[key] = current && typeof current === "object" && !Array.isArray(current) && value && typeof value === "object" && !Array.isArray(value)
      ? merge(current, value)
      : value;
  }
  return out as T;
}

async function readJson(path: string): Promise<unknown> {
  try { return JSON.parse(await readFile(path, "utf8")); } catch { return undefined; }
}

export async function loadConfig(cwd: string, projectTrusted: boolean): Promise<ActiveMemoryConfig> {
  let config = merge(DEFAULT_CONFIG, await readJson(join(homedir(), ".pi", "agent", "active-memory.json")));
  if (projectTrusted) config = merge(config, await readJson(join(cwd, ".pi", "active-memory.json")));
  if (config.database.provider === "json") config.database.path = resolve(config.database.path.replace(/^~(?=\/)/, homedir()));
  return config;
}

export function publicConfig(config: ActiveMemoryConfig): object {
  return config;
}
