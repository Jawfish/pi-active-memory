import { Embedder } from "./embeddings.js";
import { PiFastModel } from "./fast-model.js";
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
    return this.create("rag", this.rag, selection.adapter, selection.config, context);
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
  registry.registerRag("qdrant", (config) => new QdrantVectorStore(
    requiredString(config.url, "providers.rag.config.url"),
    requiredString(config.collection, "providers.rag.config.collection"),
    optionalEnv(config.apiKeyEnv),
  ));

  for (const provider of ["openai", "openai-compatible", "ollama"] as const) {
    registry.registerEmbedding(provider, async (config, context) => {
      const embeddingConfig: EmbeddingConfig = {
        provider,
        model: requiredString(config.model, "providers.embedding.config.model"),
        baseUrl: requiredString(config.baseUrl, "providers.embedding.config.baseUrl"),
        ...(typeof config.apiKeyEnv === "string" ? { apiKeyEnv: config.apiKeyEnv } : {}),
        ...(typeof config.dimensions === "number" ? { dimensions: config.dimensions } : {}),
      };
      const envName = embeddingConfig.apiKeyEnv ?? "OPENAI_API_KEY";
      const resolvedApiKey = provider === "openai" && !process.env[envName]
        ? await context.extensionContext.modelRegistry.getApiKeyForProvider("openai")
        : undefined;
      return new Embedder(embeddingConfig, resolvedApiKey);
    });
  }

  registry.registerLlm("pi-model", (config, context) => new PiFastModel({
    candidates: stringArray(config.candidates, "providers.llm.config.candidates"),
    thinking: thinking(config.thinking),
    maxTokens: requiredNumber(config.maxTokens, "providers.llm.config.maxTokens"),
  } satisfies FastModelConfig, context.extensionContext));
  return registry;
}

function requiredString(value: unknown, path: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${path} must be a non-empty string`);
  return value;
}

function requiredNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be a finite number`);
  return value;
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

function optionalEnv(value: unknown): string | undefined {
  return typeof value === "string" ? process.env[value] : undefined;
}
