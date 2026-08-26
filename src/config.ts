import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { createJiti } from "jiti";
import { DEFAULT_PROMPTS } from "./prompts.js";
import type { ActiveMemoryConfig } from "./types.js";

export const DEFAULT_CONFIG: ActiveMemoryConfig = {
  enabled: true,
  providers: {
    rag: { adapter: "json", config: { path: join(homedir(), ".pi", "agent", "active-memory", "vectors.json") } },
    // ChatGPT/Codex OAuth does not expose an embeddings API, so embeddings default to OpenAI API.
    embedding: { adapter: "openai", config: { model: "text-embedding-3-small", baseUrl: "https://api.openai.com/v1", apiKeyEnv: "OPENAI_API_KEY" } },
    llm: {
      adapter: "pi-model",
      config: {
        candidates: ["openai-codex/gpt-5.6-luna", "openai/gpt-5.6-luna", "openai/gpt-5.4-mini"],
        thinking: "off",
        maxTokens: 1200,
      },
    },
  },
  prompts: DEFAULT_PROMPTS,
  capture: { enabled: true, minCharacters: 8, contextCharacters: 12000, confidenceThreshold: 0.72, similarityThreshold: 0.82 },
  assistantCapture: { enabled: true, minimumElapsedMs: 60000, contextCharacters: 20000, confidenceThreshold: 0.62, maximumConfidence: 0.75, priority: 0.55, similarityThreshold: 0.78 },
  memoryLifecycle: {
    enabled: true,
    confidence: { initial: 0.5, deletionThreshold: 0.1, minimum: 0.05, maximum: 0.95, usefulDelta: 0.1, unhelpfulDelta: 0.15 },
    // 0.5 * (1 - 0.28)^5 ≈ 0.097, just below the deletion threshold.
    decay: { initialRate: 0.28, minimumRate: 0, maximumRate: 0.95, usefulDelta: 0.05 },
    feedback: { maxPerMemoryPerSession: 2, historyLimit: 50 },
  },
  compaction: { similarityThreshold: 0.5, maximumProposals: 10 },
  recall: {
    enabled: true,
    topK: 10,
    contextCharacters: 16000,
    everyTurns: 2,
    everyToolResults: 4,
    thinkingCharacters: 1200,
    cooldownMs: 15000,
    perMemoryCooldownMs: 30 * 60_000,
    perMemoryTurnCooldown: 4,
    maxSteersPerMemoryPerSession: 2,
    minVectorScore: 0.28,
    minimumMemoryAgeMinutes: 30,
  },
  security: { redactSecrets: true, maxMemoryCharacters: 1200 },
  activityLog: { enabled: true, includeText: true },
};

type JsonObject = Record<string, unknown>;
function merge<T>(base: T, patch: unknown): T {
  if (!patch || typeof patch !== "object" || Array.isArray(patch)) return base;
  const patchObject = patch as JsonObject;
  const baseObject = base as JsonObject;
  if (typeof patchObject.adapter === "string" && typeof baseObject?.adapter === "string" && patchObject.adapter !== baseObject.adapter) {
    return { ...patchObject } as T;
  }
  const out: JsonObject = { ...baseObject };
  const hasUnifiedEmbeddingModel = Object.hasOwn(patchObject, "model");
  const hasSeparateEmbeddingModel = Object.hasOwn(patchObject, "queryModel") || Object.hasOwn(patchObject, "documentModel");
  if (hasSeparateEmbeddingModel && !hasUnifiedEmbeddingModel) delete out.model;
  if (hasUnifiedEmbeddingModel && !hasSeparateEmbeddingModel) {
    delete out.queryModel;
    delete out.documentModel;
  }
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

export async function saveUserCompactionThreshold(
  similarityThreshold: number,
  path = join(homedir(), ".pi", "agent", "active-memory.json"),
): Promise<void> {
  if (!Number.isFinite(similarityThreshold) || similarityThreshold < 0 || similarityThreshold > 1) {
    throw new Error("Compaction similarity threshold must be between 0 and 1");
  }
  await saveUserConfigPatch({ compaction: { similarityThreshold } }, path);
}

export async function saveUserMemoryLifecycle(
  memoryLifecycle: ActiveMemoryConfig["memoryLifecycle"],
  path = join(homedir(), ".pi", "agent", "active-memory.json"),
): Promise<void> {
  await saveUserConfigPatch({ memoryLifecycle }, path);
}

async function saveUserConfigPatch(patch: JsonObject, path: string): Promise<void> {
  const existing = await readJson(path);
  const root = existing && typeof existing === "object" && !Array.isArray(existing) ? existing as JsonObject : {};
  const next = merge(root, patch);
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp`;
  await writeFile(temporary, JSON.stringify(next, null, 2), { encoding: "utf8", mode: 0o600 });
  await rename(temporary, path);
}

export interface ActiveMemoryCodeConfigContext {
  cwd: string;
  projectTrusted: boolean;
  defaults: ActiveMemoryConfig;
}

export async function loadConfig(
  cwd: string,
  projectTrusted: boolean,
  agentDir = join(homedir(), ".pi", "agent"),
): Promise<ActiveMemoryConfig> {
  const context: ActiveMemoryCodeConfigContext = { cwd, projectTrusted, defaults: structuredClone(DEFAULT_CONFIG) };
  let config = merge(DEFAULT_CONFIG, migrateConfig(await readJson(join(agentDir, "active-memory.json"))));
  config = merge(config, migrateConfig(await readCodeConfig(agentDir, "active-memory.config", context)));
  if (projectTrusted) {
    config = merge(config, migrateConfig(await readJson(join(cwd, ".pi", "active-memory.json"))));
    config = merge(config, migrateConfig(await readCodeConfig(join(cwd, ".pi"), "active-memory.config", context)));
  }
  const rag = config.providers.rag;
  if (rag.adapter === "json" && typeof rag.config.path === "string") {
    rag.config.path = resolve(rag.config.path.replace(/^~(?=\/)/, homedir()));
  }
  return config;
}

async function readCodeConfig(directory: string, basename: string, context: ActiveMemoryCodeConfigContext): Promise<unknown> {
  for (const extension of [".ts", ".mts", ".cts", ".js", ".mjs", ".cjs"]) {
    const path = join(directory, `${basename}${extension}`);
    try {
      await readFile(path, "utf8");
    } catch {
      continue;
    }
    const jiti = createJiti(import.meta.url, { moduleCache: false });
    const loaded = await jiti.import(path, { default: true }) as unknown;
    return typeof loaded === "function"
      ? await (loaded as (context: ActiveMemoryCodeConfigContext) => unknown | Promise<unknown>)(context)
      : loaded;
  }
  return undefined;
}

function migrateConfig(value: unknown): unknown {
  return migrateProviderConfig(migrateLifecycleConfig(value));
}

function migrateProviderConfig(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value;
  const root = value as JsonObject;
  const providers = root.providers && typeof root.providers === "object" && !Array.isArray(root.providers)
    ? { ...(root.providers as JsonObject) }
    : {};
  if (!providers.rag && root.database && typeof root.database === "object" && !Array.isArray(root.database)) {
    const { provider, ...config } = root.database as JsonObject;
    if (typeof provider === "string") providers.rag = { adapter: provider, config };
  }
  if (!providers.embedding && root.embedding && typeof root.embedding === "object" && !Array.isArray(root.embedding)) {
    const { provider, ...config } = root.embedding as JsonObject;
    if (typeof provider === "string") providers.embedding = { adapter: provider, config };
  }
  if (!providers.llm && root.fastModel && typeof root.fastModel === "object" && !Array.isArray(root.fastModel)) {
    providers.llm = { adapter: "pi-model", config: root.fastModel };
  }
  const { database: _database, embedding: _embedding, fastModel: _fastModel, ...rest } = root;
  return Object.keys(providers).length > 0 ? { ...rest, providers } : rest;
}

function migrateLifecycleConfig(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value;
  const root = value as JsonObject;
  if (root.memoryLifecycle) return value;
  const oldForgetting = root.forgetting && typeof root.forgetting === "object" ? root.forgetting as JsonObject : undefined;
  const oldFeedback = root.feedback && typeof root.feedback === "object" ? root.feedback as JsonObject : undefined;
  if (!oldForgetting && !oldFeedback) return value;
  const lifecycle = structuredClone(DEFAULT_CONFIG.memoryLifecycle) as unknown as JsonObject;
  const confidence = lifecycle.confidence as JsonObject;
  const feedback = lifecycle.feedback as JsonObject;
  if (typeof oldForgetting?.enabled === "boolean") lifecycle.enabled = oldForgetting.enabled;
  if (typeof oldForgetting?.minimumConfidenceToKeep === "number") confidence.deletionThreshold = oldForgetting.minimumConfidenceToKeep;
  if (typeof oldFeedback?.initialConfidence === "number") confidence.initial = oldFeedback.initialConfidence;
  if (typeof oldFeedback?.usefulDelta === "number") confidence.usefulDelta = oldFeedback.usefulDelta;
  if (typeof oldFeedback?.unhelpfulDelta === "number") confidence.unhelpfulDelta = oldFeedback.unhelpfulDelta;
  if (typeof oldFeedback?.minimumConfidence === "number") confidence.minimum = oldFeedback.minimumConfidence;
  if (typeof oldFeedback?.maximumConfidence === "number") confidence.maximum = oldFeedback.maximumConfidence;
  if (typeof oldFeedback?.maxFeedbackPerMemoryPerSession === "number") feedback.maxPerMemoryPerSession = oldFeedback.maxFeedbackPerMemoryPerSession;
  if (typeof oldFeedback?.historyLimit === "number") feedback.historyLimit = oldFeedback.historyLimit;
  const { forgetting: _forgetting, feedback: _feedback, ...rest } = root;
  return { ...rest, memoryLifecycle: lifecycle };
}

export function publicConfig(config: ActiveMemoryConfig): object {
  return config;
}
