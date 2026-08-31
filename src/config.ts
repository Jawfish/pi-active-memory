import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { readFile } from "node:fs/promises";
import { createJiti } from "jiti";
import { DEFAULT_PROMPTS } from "./prompts.js";
import type { ActiveMemoryConfig } from "./types.js";
import { atomicWriteFile, withStorageLock } from "./storage-lock.js";

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

export class ActiveMemoryConfigError extends Error {
  constructor(message: string, readonly configPath?: string) { super(message); this.name = "ActiveMemoryConfigError"; }
}

async function readJson(path: string): Promise<unknown> {
  let text: string;
  try { text = await readFile(path, "utf8"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw new ActiveMemoryConfigError("Could not read Active Memory configuration", path); }
  try { return JSON.parse(text); } catch { throw new ActiveMemoryConfigError("Active Memory configuration contains invalid JSON", path); }
}

export async function saveUserCompactionThreshold(
  similarityThreshold: number,
  path = join(homedir(), ".pi", "agent", "active-memory.json"),
  effectiveConfig?: ActiveMemoryConfig,
): Promise<void> {
  if (!Number.isFinite(similarityThreshold) || similarityThreshold < 0 || similarityThreshold > 1) {
    throw new Error("Compaction similarity threshold must be between 0 and 1");
  }
  await saveUserConfigPatch({ compaction: { similarityThreshold } }, path, effectiveConfig);
}

export type UserLifecycleSetting = "decay.initialRate" | "confidence.deletionThreshold";

/** Persist only a user-selected lifecycle leaf, never the merged project-effective policy. */
export async function saveUserMemoryLifecycleSetting(
  setting: UserLifecycleSetting,
  value: number,
  path = join(homedir(), ".pi", "agent", "active-memory.json"),
  effectiveConfig?: ActiveMemoryConfig,
): Promise<void> {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error("Memory lifecycle setting must be between 0 and 1");
  const patch = setting === "decay.initialRate"
    ? { memoryLifecycle: { decay: { initialRate: value } } }
    : { memoryLifecycle: { confidence: { deletionThreshold: value } } };
  await saveUserConfigPatch(patch, path, effectiveConfig);
}

/** @deprecated Use saveUserMemoryLifecycleSetting to avoid persisting merged config. */
export async function saveUserMemoryLifecycle(
  memoryLifecycle: ActiveMemoryConfig["memoryLifecycle"],
  path = join(homedir(), ".pi", "agent", "active-memory.json"),
): Promise<void> {
  await saveUserConfigPatch({ memoryLifecycle }, path);
}

async function saveUserConfigPatch(patch: JsonObject, path: string, effectiveConfig?: ActiveMemoryConfig): Promise<void> {
  await withStorageLock(path, async () => {
    const existing = await readJson(path);
    if (existing !== undefined && (!existing || typeof existing !== "object" || Array.isArray(existing))) throw new ActiveMemoryConfigError("Active Memory configuration root must be an object", path);
    const root = existing as JsonObject | undefined ?? {};
    const next = merge(root, patch);
    // A global JSON layer can be intentionally incomplete until later code or
    // project overlays are applied. Validate against the caller's effective
    // layered policy when available, while still persisting only the user leaf.
    const prospective = effectiveConfig
      ? merge(structuredClone(effectiveConfig), patch)
      : merge(DEFAULT_CONFIG, migrateConfig(next));
    validateMergedConfig(prospective);
    await atomicWriteFile(path, JSON.stringify(next, null, 2));
  });
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
  const globalJsonPath = join(agentDir, "active-memory.json");
  const globalCodePath = join(agentDir, "active-memory.config");
  let config = merge(DEFAULT_CONFIG, migrateConfig(configLayer(await readJson(globalJsonPath), globalJsonPath)));
  config = merge(config, migrateConfig(configLayer(await readCodeConfig(agentDir, "active-memory.config", context), globalCodePath)));
  if (projectTrusted) {
    const projectJsonPath = join(cwd, ".pi", "active-memory.json");
    const projectCodePath = join(cwd, ".pi", "active-memory.config");
    config = merge(config, migrateConfig(configLayer(await readJson(projectJsonPath), projectJsonPath)));
    config = merge(config, migrateConfig(configLayer(await readCodeConfig(join(cwd, ".pi"), "active-memory.config", context), projectCodePath)));
  }
  const validated = validateMergedConfig(config);
  const rag = validated.providers.rag;
  if (rag.adapter === "json" && typeof rag.config.path === "string") rag.config.path = resolve(rag.config.path.replace(/^~(?=\/)/, homedir()));
  return validated;
}

async function readCodeConfig(directory: string, basename: string, context: ActiveMemoryCodeConfigContext): Promise<unknown> {
  for (const extension of [".ts", ".mts", ".cts", ".js", ".mjs", ".cjs"]) {
    const path = join(directory, `${basename}${extension}`);
    try {
      await readFile(path, "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
      throw new ActiveMemoryConfigError("Could not read Active Memory code configuration", path);
    }
    const jiti = createJiti(import.meta.url, { moduleCache: false });
    const loaded = await jiti.import(path, { default: true }) as unknown;
    return typeof loaded === "function"
      ? await (loaded as (context: ActiveMemoryCodeConfigContext) => unknown | Promise<unknown>)(context)
      : loaded;
  }
  return undefined;
}

function configLayer(value: unknown, path: string): unknown {
  if (value === undefined) return undefined;
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new ActiveMemoryConfigError("Active Memory configuration layer must be an object", path);
  return value;
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

/** Safe status/log projection: custom adapter configuration may contain arbitrary secrets. */
export function validateMergedConfig(value: unknown): ActiveMemoryConfig {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new ActiveMemoryConfigError("Active Memory configuration root must be an object");
  const config = value as ActiveMemoryConfig;
  validateLikeDefault(config, DEFAULT_CONFIG, "config");
  for (const provider of [config.providers.rag, config.providers.embedding, config.providers.llm]) {
    if (!provider || typeof provider.adapter !== "string" || !provider.adapter.trim() || !provider.config || typeof provider.config !== "object" || Array.isArray(provider.config)) throw new ActiveMemoryConfigError("Active Memory providers require a non-empty adapter id and object config");
  }
  const lifecycle = config.memoryLifecycle;
  const unit = [config.capture.confidenceThreshold, config.capture.similarityThreshold, config.assistantCapture.confidenceThreshold, config.assistantCapture.maximumConfidence, config.assistantCapture.priority, config.assistantCapture.similarityThreshold, lifecycle.confidence.initial, lifecycle.confidence.deletionThreshold, lifecycle.confidence.minimum, lifecycle.confidence.maximum, lifecycle.confidence.usefulDelta, lifecycle.confidence.unhelpfulDelta, lifecycle.decay.initialRate, lifecycle.decay.minimumRate, lifecycle.decay.maximumRate, lifecycle.decay.usefulDelta, config.compaction.similarityThreshold];
  if (unit.some(number => number < 0 || number > 1) || config.recall.minVectorScore < -1 || config.recall.minVectorScore > 1) throw new ActiveMemoryConfigError("Active Memory confidence, priority, similarity, and rate values must be in range");
  const positiveIntegers = [config.capture.minCharacters, config.capture.contextCharacters, config.assistantCapture.contextCharacters, config.compaction.maximumProposals, config.recall.topK, config.recall.contextCharacters, config.recall.everyTurns, config.recall.everyToolResults, config.recall.thinkingCharacters, config.security.maxMemoryCharacters];
  const nonNegativeIntegers = [lifecycle.feedback.maxPerMemoryPerSession, lifecycle.feedback.historyLimit, config.recall.perMemoryTurnCooldown, config.recall.maxSteersPerMemoryPerSession];
  if (positiveIntegers.some(number => !Number.isInteger(number) || number <= 0) || nonNegativeIntegers.some(number => !Number.isInteger(number) || number < 0) || [config.assistantCapture.minimumElapsedMs, config.recall.cooldownMs, config.recall.perMemoryCooldownMs, config.recall.minimumMemoryAgeMinutes].some(number => number < 0)) throw new ActiveMemoryConfigError("Active Memory sizes and cadences must be valid integers or non-negative durations");
  if (lifecycle.confidence.minimum > lifecycle.confidence.initial || lifecycle.confidence.initial > lifecycle.confidence.maximum || lifecycle.decay.minimumRate > lifecycle.decay.initialRate || lifecycle.decay.initialRate > lifecycle.decay.maximumRate || lifecycle.confidence.deletionThreshold < lifecycle.confidence.minimum || lifecycle.confidence.deletionThreshold > lifecycle.confidence.maximum || config.assistantCapture.confidenceThreshold > config.assistantCapture.maximumConfidence) throw new ActiveMemoryConfigError("Active Memory configuration has inconsistent confidence or decay bounds");
  return config;
}

function validateLikeDefault(value: unknown, exemplar: unknown, path: string): void {
  if (path === "config.providers") return;
  if (typeof exemplar === "boolean") { if (typeof value !== "boolean") throw new ActiveMemoryConfigError(`${path} must be a boolean`); return; }
  if (typeof exemplar === "string") { if (typeof value !== "string") throw new ActiveMemoryConfigError(`${path} must be a string`); return; }
  if (typeof exemplar === "number") { if (typeof value !== "number" || !Number.isFinite(value)) throw new ActiveMemoryConfigError(`${path} must be a finite number`); return; }
  if (Array.isArray(exemplar)) { if (!Array.isArray(value) || value.some(item => typeof item !== "string")) throw new ActiveMemoryConfigError(`${path} must be a string array`); return; }
  if (exemplar && typeof exemplar === "object") {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new ActiveMemoryConfigError(`${path} must be an object`);
    for (const [key, child] of Object.entries(exemplar as Record<string, unknown>)) validateLikeDefault((value as Record<string, unknown>)[key], child, `${path}.${key}`);
  }
}

export function publicConfig(config: ActiveMemoryConfig): object {
  return {
    enabled: config.enabled,
    providers: { rag: { configured: true }, embedding: { configured: true }, llm: { configured: true } },
    capture: { enabled: config.capture.enabled, minCharacters: config.capture.minCharacters, confidenceThreshold: config.capture.confidenceThreshold },
    assistantCapture: { enabled: config.assistantCapture.enabled, minimumElapsedMs: config.assistantCapture.minimumElapsedMs, confidenceThreshold: config.assistantCapture.confidenceThreshold },
    memoryLifecycle: config.memoryLifecycle,
    compaction: config.compaction,
    recall: config.recall,
    security: { redactSecrets: config.security.redactSecrets, maxMemoryCharacters: config.security.maxMemoryCharacters },
    activityLog: config.activityLog,
  };
}
