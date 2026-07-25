export type MemoryScope = "global" | "project";
export type MemoryKind = "user_profile" | "fact" | "skill_workflow";

export type MemoryActor = "user" | "assistant";

export interface MemorySource {
  actor: MemoryActor;
  sessionId: string;
  cwd: string;
  cause: string;
  reason: string;
  elapsedMs?: number;
  userText?: string;
  evidence?: string;
}

export interface MemoryRecord {
  id: string;
  text: string;
  kind: MemoryKind;
  scope: MemoryScope;
  projectId?: string;
  confidence: number;
  /** Ranking weight. Legacy records default to 1. */
  priority?: number;
  status: "active" | "superseded" | "deleted";
  supersedes?: string[];
  source: MemorySource;
  sourceHistory?: MemorySource[];
  createdAt: string;
  updatedAt: string;
  embeddingModel: string;
  schemaVersion: 1 | 2;
}

export interface MemoryMatch { record: MemoryRecord; score: number }
export interface MemoryFilter { scopes?: MemoryScope[]; kinds?: MemoryKind[]; projectId?: string; status?: MemoryRecord["status"] }

export interface VectorStore {
  initialize(dimension?: number): Promise<void>;
  upsert(record: MemoryRecord, vector: number[]): Promise<void>;
  search(vector: number[], filter: MemoryFilter, limit: number): Promise<MemoryMatch[]>;
  list(filter: MemoryFilter, limit: number): Promise<MemoryRecord[]>;
  markDeleted(id: string): Promise<boolean>;
  migrateLegacyProvenance(): Promise<number>;
  close(): Promise<void>;
}

export interface EmbeddingConfig {
  provider: "openai" | "ollama" | "openai-compatible";
  model: string;
  baseUrl: string;
  apiKeyEnv?: string;
  dimensions?: number;
}

export interface FastModelConfig {
  candidates: string[];
  thinking: "off" | "minimal" | "low" | "medium" | "high";
  maxTokens: number;
}

export interface ActiveMemoryConfig {
  enabled: boolean;
  database:
    | { provider: "json"; path: string }
    | { provider: "qdrant"; url: string; collection: string; apiKeyEnv?: string };
  embedding: EmbeddingConfig;
  fastModel: FastModelConfig;
  capture: {
    enabled: boolean;
    minCharacters: number;
    contextCharacters: number;
    confidenceThreshold: number;
    similarityThreshold: number;
  };
  assistantCapture: {
    enabled: boolean;
    minimumElapsedMs: number;
    contextCharacters: number;
    confidenceThreshold: number;
    maximumConfidence: number;
    priority: number;
    similarityThreshold: number;
  };
  recall: {
    enabled: boolean;
    topK: number;
    contextCharacters: number;
    everyTurns: number;
    everyToolResults: number;
    thinkingCharacters: number;
    cooldownMs: number;
    minVectorScore: number;
    minimumMemoryAgeMinutes: number;
  };
  security: { redactSecrets: boolean; maxMemoryCharacters: number };
  activityLog: { enabled: boolean; includeText: boolean };
}

export interface FastModelRunner {
  json<T>(system: string, prompt: string, signal?: AbortSignal): Promise<T>;
  selectedModel(): string | undefined;
}
