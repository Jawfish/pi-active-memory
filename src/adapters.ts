import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { Embedder, RoutedEmbedder } from "./embeddings.js";
import { TransactionalVectorStoreError } from "./errors.js";
import { PiFastModel } from "./fast-model.js";
import { canonicalEndpoint } from "./embedding-metadata.js";
import { JsonVectorStore } from "./stores/json-store.js";
import { QdrantVectorStore } from "./stores/qdrant-store.js";
import type {
  ActiveMemoryAdapterContext,
  ActiveMemoryAdapterFactory,
  ActiveMemoryAdapterRegistry,
  ActiveMemoryProvidersConfig,
  EmbeddingConfig,
  EmbeddingProvider,
  FastModelConfig,
  FastModelRunner,
  VectorStore,
} from "./types.js";

export const ACTIVE_MEMORY_ADAPTER_EVENT = "pi-active-memory:register-adapters";

type Component = "rag" | "embedding" | "llm";

export class AdapterRegistry implements ActiveMemoryAdapterRegistry {
  private readonly rag = new Map<string, ActiveMemoryAdapterFactory<VectorStore>>();
  private readonly embedding = new Map<string, ActiveMemoryAdapterFactory<EmbeddingProvider>>();
  private readonly llm = new Map<string, ActiveMemoryAdapterFactory<FastModelRunner>>();

  registerRag(id: string, factory: ActiveMemoryAdapterFactory<VectorStore>): void { this.register("rag", this.rag, id, factory); }
  registerEmbedding(id: string, factory: ActiveMemoryAdapterFactory<EmbeddingProvider>): void { this.register("embedding", this.embedding, id, factory); }
  registerLlm(id: string, factory: ActiveMemoryAdapterFactory<FastModelRunner>): void { this.register("llm", this.llm, id, factory); }

  async createRag(selection: ActiveMemoryProvidersConfig["rag"], context: ActiveMemoryAdapterContext): Promise<VectorStore> {
    const store = await this.create("rag", this.rag, selection.adapter, selection.config, context);
    const required: Array<keyof VectorStore> = ["initialize", "get", "insert", "mutate", "compact", "scan", "rebuildVectors", "search", "list", "migrateLegacyProvenance", "close"];
    let valid = false;
    try { valid = Boolean(store && typeof store === "object" && store.contractVersion === 2 && required.every(method => typeof store[method] === "function")); }
    catch { valid = false; }
    if (!valid) {
      try {
        const resource = store && (typeof store === "object" || typeof store === "function") ? store as Partial<VectorStore> : undefined;
        const close = resource?.close;
        if (typeof close === "function") await close.call(store);
      } catch {}
      throw new TransactionalVectorStoreError();
    }
    return store;
  }

  async createEmbedding(selection: ActiveMemoryProvidersConfig["embedding"], context: ActiveMemoryAdapterContext): Promise<EmbeddingProvider> {
    return this.create("embedding", this.embedding, selection.adapter, selection.config, context);
  }

  async createLlm(selection: ActiveMemoryProvidersConfig["llm"], context: ActiveMemoryAdapterContext): Promise<FastModelRunner> {
    return this.create("llm", this.llm, selection.adapter, selection.config, context);
  }

  private register<T>(component: Component, target: Map<string, ActiveMemoryAdapterFactory<T>>, id: string, factory: ActiveMemoryAdapterFactory<T>): void {
    if (!id.trim()) throw new Error(`Cannot register an empty ${component} adapter id`);
    if (target.has(id)) throw new Error(`${component} adapter already registered: ${id}`);
    target.set(id, factory);
  }

  private async create<T>(
    component: Component,
    source: Map<string, ActiveMemoryAdapterFactory<T>>,
    id: string,
    config: Record<string, unknown>,
    context: ActiveMemoryAdapterContext,
  ): Promise<T> {
    const factory = source.get(id);
    if (!factory) throw new Error(`Unknown ${component} adapter: ${id}`);
    return factory(config, context);
  }
}

export function createBuiltInAdapterRegistry(): AdapterRegistry {
  const registry = new AdapterRegistry();
  registry.registerRag("json", (config) => new JsonVectorStore(requiredString(config.path, "providers.rag.config.path")));
  registry.registerRag("qdrant", (config) => {
    const url = requiredString(config.url, "providers.rag.config.url");
    const collection = requiredString(config.collection, "providers.rag.config.collection");
    // Coordination is part of canonical store identity. A configurable lock
    // path would let two sessions targeting the same store bypass each other's
    // transaction lock and ambiguous-publication fence.
    const lockTarget = join(getAgentDir(), "active-memory", `qdrant-${createQdrantLockKey(url)}`);
    return new QdrantVectorStore(url, collection, optionalEnv(config.apiKeyEnv), lockTarget);
  });

  for (const provider of ["openai", "openai-compatible", "ollama"] as const) {
    registry.registerEmbedding(provider, async (config, context) => {
      const models = configuredEmbeddingModels(config);
      const configuredApiKeyEnv = optionalString(config.apiKeyEnv, "providers.embedding.config.apiKeyEnv");
      const configuredApiKeyProvider = optionalString(config.apiKeyProvider, "providers.embedding.config.apiKeyProvider");
      const apiKeyEnv = configuredApiKeyEnv ?? (provider === "openai" ? "OPENAI_API_KEY" : undefined);
      const apiKeyProvider = configuredApiKeyProvider ?? (provider === "openai" ? "openai" : undefined);
      const baseConfig = {
        provider,
        baseUrl: requiredString(config.baseUrl, "providers.embedding.config.baseUrl"),
        ...(apiKeyEnv ? { apiKeyEnv } : {}),
        ...(apiKeyProvider ? { apiKeyProvider } : {}),
        ...(typeof config.dimensions === "number" ? { dimensions: config.dimensions } : {}),
      };
      const resolvedApiKey = provider !== "ollama" && apiKeyProvider && !(apiKeyEnv && process.env[apiKeyEnv])
        ? await context.extensionContext.modelRegistry.getApiKeyForProvider(apiKeyProvider)
        : undefined;
      const query = new Embedder({ ...baseConfig, model: models.query } satisfies EmbeddingConfig, resolvedApiKey);
      const document = models.document === models.query
        ? query
        : new Embedder({ ...baseConfig, model: models.document } satisfies EmbeddingConfig, resolvedApiKey);
      return new RoutedEmbedder(query, document);
    });
  }

  registry.registerLlm("pi-model", (config, context) => new PiFastModel({
    candidates: stringArray(config.candidates, "providers.llm.config.candidates"),
    thinking: thinking(config.thinking),
    maxTokens: requiredNumber(config.maxTokens, "providers.llm.config.maxTokens"),
  } satisfies FastModelConfig, context.extensionContext));
  return registry;
}

export function configuredEmbeddingModels(config: Record<string, unknown>): { query: string; document: string } {
  const unified = typeof config.model === "string" && config.model.trim() ? config.model.trim() : undefined;
  const query = typeof config.queryModel === "string" && config.queryModel.trim() ? config.queryModel.trim() : undefined;
  const document = typeof config.documentModel === "string" && config.documentModel.trim() ? config.documentModel.trim() : undefined;
  if (unified && (query || document)) throw new Error("Configure either providers.embedding.config.model or queryModel/documentModel, not both");
  if (unified) return { query: unified, document: unified };
  if (!query || !document) throw new Error("Configure either providers.embedding.config.model or both queryModel and documentModel");
  return { query, document };
}

function requiredString(value: unknown, path: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${path} must be a non-empty string`);
  return value;
}

function requiredNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be a finite number`);
  return value;
}

function optionalString(value: unknown, path: string): string | undefined {
  if (value === undefined) return undefined;
  return requiredString(value, path).trim();
}

function stringArray(value: unknown, path: string): string[] {
  if (!Array.isArray(value) || value.some(item => typeof item !== "string") || value.length === 0) {
    throw new Error(`${path} must be a non-empty string array`);
  }
  return value as string[];
}

function thinking(value: unknown): FastModelConfig["thinking"] {
  if (value === "off" || value === "minimal" || value === "low" || value === "medium" || value === "high") return value;
  throw new Error("providers.llm.config.thinking is invalid");
}

function createQdrantLockKey(url: string): string {
  return createHash("sha256").update(canonicalEndpoint(url)).digest("hex");
}

function optionalEnv(value: unknown): string | undefined {
  return typeof value === "string" ? process.env[value] : undefined;
}
