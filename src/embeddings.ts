import type { EmbeddingConfig, EmbeddingModels, EmbeddingProvider } from "./types.js";

export class Embedder {
  constructor(private readonly config: EmbeddingConfig, private readonly resolvedApiKey?: string) {}
  get model(): string { return `${this.config.provider}/${this.config.model}`; }

  async embed(texts: string[], signal?: AbortSignal): Promise<number[][]> {
    if (texts.length === 0) return [];
    if (this.config.provider === "ollama") {
      const response = await fetch(`${this.config.baseUrl.replace(/\/$/, "")}/api/embed`, {
        method: "POST", headers: { "content-type": "application/json" }, signal,
        body: JSON.stringify({ model: this.config.model, input: texts }),
      });
      if (!response.ok) throw new Error(`Ollama embeddings failed (${response.status}): ${await response.text()}`);
      const data = await response.json() as { embeddings?: number[][] };
      if (!data.embeddings?.length) throw new Error("Ollama returned no embeddings");
      return data.embeddings;
    }

    const key = (this.config.apiKeyEnv ? process.env[this.config.apiKeyEnv] : undefined) ?? this.resolvedApiKey;
    if (!key) throw new Error(`Embedding provider requires $${this.config.apiKeyEnv ?? "OPENAI_API_KEY"}`);
    const response = await fetch(`${this.config.baseUrl.replace(/\/$/, "")}/embeddings`, {
      method: "POST", signal,
      headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
      body: JSON.stringify({ model: this.config.model, input: texts, ...(this.config.dimensions ? { dimensions: this.config.dimensions } : {}) }),
    });
    if (!response.ok) throw new Error(`Embeddings failed (${response.status}): ${await response.text()}`);
    const data = await response.json() as { data?: Array<{ index: number; embedding: number[] }> };
    const rows = (data.data ?? []).sort((a, b) => a.index - b.index).map((row) => row.embedding);
    if (rows.length !== texts.length) throw new Error("Embedding provider returned an unexpected number of vectors");
    return rows;
  }
}

export class RoutedEmbedder implements EmbeddingProvider {
  readonly queryModel: string;
  readonly documentModel: string;

  constructor(private readonly query: Embedder, private readonly document: Embedder) {
    this.queryModel = query.model;
    this.documentModel = document.model;
  }

  embedQuery(texts: string[], signal?: AbortSignal): Promise<number[][]> { return this.query.embed(texts, signal); }
  embedDocuments(texts: string[], signal?: AbortSignal): Promise<number[][]> { return this.document.embed(texts, signal); }
}

export function embeddingModels(provider: EmbeddingProvider): EmbeddingModels {
  const hasSeparate = Boolean(provider.queryModel || provider.documentModel);
  if (provider.model && hasSeparate) throw new Error("Embedding adapter exposes both unified and separate model identities");
  if (provider.queryModel && provider.documentModel) return { query: provider.queryModel, document: provider.documentModel };
  if (provider.model) return { query: provider.model, document: provider.model };
  throw new Error("Embedding adapter must expose either model or both queryModel and documentModel");
}

export function embedQuery(provider: EmbeddingProvider, texts: string[], signal?: AbortSignal): Promise<number[][]> {
  if (provider.embedQuery) return provider.embedQuery(texts, signal);
  if (provider.embed) return provider.embed(texts, signal);
  throw new Error("Embedding adapter does not implement query embedding");
}

export function embedDocuments(provider: EmbeddingProvider, texts: string[], signal?: AbortSignal): Promise<number[][]> {
  if (provider.embedDocuments) return provider.embedDocuments(texts, signal);
  if (provider.embed) return provider.embed(texts, signal);
  throw new Error("Embedding adapter does not implement document embedding");
}
