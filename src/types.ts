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

export interface MemoryFeedback {
  outcome: "useful" | "unhelpful";
  sessionId: string;
  steerToken: string;
  reason: string;
  at: string;
}

export interface MemoryFeedbackSummary {
  useful: number;
  unhelpful: number;
  lastAt: string;
  history: MemoryFeedback[];
}

export interface MemoryLifecycle {
  /** UTC calendar date through which multiplicative decay has been applied. */
  lastDecayDate: string;
  lastRelevantAt: string;
  lastRelevantSessionId: string;
  reinforcementCount: number;
  lastReinforcementCause: "created" | "relevant_recall" | "useful_feedback" | "user_compaction" | "legacy_migration_grace";
  deletedAt?: string;
  deletionCause?: "low_confidence";
}

export interface MemoryRecord {
  id: string;
  text: string;
  kind: MemoryKind;
  scope: MemoryScope;
  projectId?: string;
  confidence: number;
  /** Daily fraction lost: confidence *= (1 - decayRate) ^ elapsedDays. */
  decayRate?: number;
  /** Ranking weight. Legacy records default to 1. */
  priority?: number;
  status: "active" | "superseded" | "deleted";
  supersedes?: string[];
  source: MemorySource;
  sourceHistory?: MemorySource[];
  feedback?: MemoryFeedbackSummary;
  lifecycle?: MemoryLifecycle;
  createdAt: string;
  updatedAt: string;
  embeddingModel: string;
  schemaVersion: 1 | 2;
}

export interface MemoryMatch { record: MemoryRecord; score: number }
export interface MemoryFilter { scopes?: MemoryScope[]; kinds?: MemoryKind[]; projectId?: string; status?: MemoryRecord["status"] }

export interface VectorRow { record: MemoryRecord; vector: number[] }

export interface MutationResult {
  status: "missing" | "unchanged" | "updated";
  record?: MemoryRecord;
}

/**
 * Transactional vector storage contract. Implementations must serialize `mutate`,
 * `compact`, and `rebuildVectors` against other writers and commit text/payload
 * changes together with their corresponding vector.
 */
export interface VectorStore {
  readonly contractVersion: 2;
  initialize(dimension?: number): Promise<void>;
  get(id: string): Promise<MemoryRecord | undefined>;
  insert(row: VectorRow): Promise<"inserted" | "exists">;
  mutate(id: string, apply: (latest: Readonly<MemoryRecord>) => { record: MemoryRecord; vector?: number[] } | undefined): Promise<MutationResult>;
  compact(sourceIds: readonly string[], build: (latest: readonly MemoryRecord[]) => VectorRow): Promise<MemoryRecord>;
  scan(filter: MemoryFilter, visit: (page: readonly MemoryRecord[]) => Promise<void>, pageSize?: number): Promise<number>;
  rebuildVectors(dimension: number, buildPage: (page: readonly MemoryRecord[]) => Promise<readonly VectorRow[]>): Promise<number>;
  search(vector: number[], filter: MemoryFilter, limit: number): Promise<MemoryMatch[]>;
  list(filter: MemoryFilter, limit: number): Promise<MemoryRecord[]>;
  migrateLegacyProvenance(): Promise<number>;
  close(): Promise<void>;
}

export interface EmbeddingModels {
  query: string;
  document: string;
}

export interface EmbeddingProvider {
  /** Legacy unified model identity. */
  readonly model?: string;
  /** Separate model identities for asymmetric embedding providers. */
  readonly queryModel?: string;
  readonly documentModel?: string;
  /** Legacy unified embedding method. */
  embed?(texts: string[], signal?: AbortSignal): Promise<number[][]>;
  embedQuery?(texts: string[], signal?: AbortSignal): Promise<number[][]>;
  embedDocuments?(texts: string[], signal?: AbortSignal): Promise<number[][]>;
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

export interface MemoryLifecycleConfig {
  enabled: boolean;
  confidence: {
    initial: number;
    deletionThreshold: number;
    minimum: number;
    maximum: number;
    usefulDelta: number;
    unhelpfulDelta: number;
  };
  decay: {
    initialRate: number;
    minimumRate: number;
    maximumRate: number;
    usefulDelta: number;
  };
  feedback: {
    maxPerMemoryPerSession: number;
    historyLimit: number;
  };
}

export interface AdapterSelection {
  adapter: string;
  config: Record<string, unknown>;
}

export interface ActiveMemoryProvidersConfig {
  rag: AdapterSelection;
  embedding: AdapterSelection;
  llm: AdapterSelection;
}

export interface ToolPromptConfig {
  snippet: string;
  guidelines: string[];
}

export interface ActiveMemoryPromptsConfig {
  jsonOnly: string;
  extraction: string;
  validation: string;
  merge: string;
  assistantExtraction: string;
  assistantValidation: string;
  compaction: string;
  compactionValidation: string;
  query: string;
  judge: string;
  steerFeedback: string;
  tools: {
    memoryStoreResult: ToolPromptConfig;
    memoryCorrect: ToolPromptConfig;
    memorySearch: ToolPromptConfig;
    memoryFeedback: ToolPromptConfig;
  };
}

export interface ActiveMemoryConfig {
  enabled: boolean;
  providers: ActiveMemoryProvidersConfig;
  prompts: ActiveMemoryPromptsConfig;
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
  memoryLifecycle: MemoryLifecycleConfig;
  compaction: {
    similarityThreshold: number;
    maximumProposals: number;
  };
  recall: {
    enabled: boolean;
    topK: number;
    contextCharacters: number;
    everyTurns: number;
    everyToolResults: number;
    thinkingCharacters: number;
    cooldownMs: number;
    perMemoryCooldownMs: number;
    perMemoryTurnCooldown: number;
    maxSteersPerMemoryPerSession: number;
    minVectorScore: number;
    minimumMemoryAgeMinutes: number;
  };
  security: { redactSecrets: boolean; maxMemoryCharacters: number };
  activityLog: { enabled: boolean; includeText: boolean };
}

export interface FastModelTokenUsage {
  input: number;
  output: number;
}

export interface FastModelRunner {
  json<T>(system: string, prompt: string, signal?: AbortSignal): Promise<T>;
  selectedModel(): string | undefined;
  onTokenUsage?(handler: (usage: FastModelTokenUsage) => void): void;
}

export interface ActiveMemoryAdapterContext {
  cwd: string;
  projectId: string;
  sessionId: string;
  extensionContext: import("@earendil-works/pi-coding-agent").ExtensionContext;
}

export type ActiveMemoryAdapterFactory<T> = (
  config: Record<string, unknown>,
  context: ActiveMemoryAdapterContext,
) => T | Promise<T>;

export interface ActiveMemoryAdapterRegistry {
  registerRag(id: string, factory: ActiveMemoryAdapterFactory<VectorStore>): void;
  registerEmbedding(id: string, factory: ActiveMemoryAdapterFactory<EmbeddingProvider>): void;
  registerLlm(id: string, factory: ActiveMemoryAdapterFactory<FastModelRunner>): void;
}
