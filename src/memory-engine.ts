import { randomUUID } from "node:crypto";
import type { ActivitySink } from "./activity-log.js";
import { pairSimilarMemories, type CompactionProposal, type MemoryCluster } from "./compaction.js";
import { embedDocuments, embeddingModels, embedQuery } from "./embeddings.js";
import type { ActiveMemoryConfig, EmbeddingProvider, FastModelRunner, MemoryActor, MemoryKind, MemoryMatch, MemoryRecord, MemoryScope, MemorySource, VectorStore } from "./types.js";
import { applyConfidenceFeedback, type FeedbackOutcome } from "./feedback.js";
import { advanceMemoryLifecycle, initializeMemoryLifecycle, reinforceMemoryLifecycle } from "./lifecycle.js";
import { assistantExtractionPrompt, assistantValidationPrompt, compactionPrompt, compactionValidationPrompt, extractionPrompt, judgePrompt, mergePrompt, queryPrompt, validationPrompt } from "./prompts.js";
import { evidenceAppearsInUserMessage, isTransientTaskMemory, redactSecrets, sanitizePersistedText, sourceEvidenceAppearsInContext } from "./utils.js";

interface Extracted { text: string; kind: MemoryKind; scope: MemoryScope; confidence: number; evidence: string }
interface AssistantExtracted extends Extracted { whyStored: string }
interface RecallResult { reason: string; relevant: MemoryMatch[] }
export interface AssistantResultInput { text: string; kind: Exclude<MemoryKind, "user_profile">; scope: MemoryScope; confidence: number; reason: string }
export interface CompactionPlan { clusters: MemoryCluster[]; proposals: CompactionProposal[] }

export class MemoryEngine {
  constructor(
    private readonly config: ActiveMemoryConfig,
    private readonly store: VectorStore,
    private readonly embedder: EmbeddingProvider,
    private readonly fast: FastModelRunner,
    private readonly projectId: string,
    private readonly sessionId: string,
    private readonly cwd: string,
    private readonly activity?: ActivitySink,
    private readonly onMemoryStored?: (record: Readonly<MemoryRecord>, created: boolean) => void,
  ) {}

  async capture(userText: string, context: string, signal?: AbortSignal): Promise<number> {
    if (!this.config.capture.enabled || userText.trim().length < this.config.capture.minCharacters) return 0;
    this.activity?.("capture.started", { characters: userText.length, ...(this.config.activityLog.includeText ? { userText } : {}) });
    const result = await this.fast.json<{ memories?: Extracted[] }>(this.config.prompts.jsonOnly, extractionPrompt(userText, context, this.projectId, this.config.prompts), signal);
    this.activity?.("capture.extracted", {
      count: result.memories?.length ?? 0,
      memories: (result.memories ?? []).map((memory) => ({ kind: memory.kind, scope: memory.scope, confidence: memory.confidence, evidenceValid: typeof memory.evidence === "string" && evidenceAppearsInUserMessage(memory.evidence, userText), ...(this.config.activityLog.includeText ? { text: memory.text, evidence: memory.evidence } : {}) })),
    });
    let stored = 0;
    for (const raw of result.memories ?? []) {
      if (!validUserExtracted(raw) || !evidenceAppearsInUserMessage(raw.evidence, userText) || isTransientTaskMemory(raw.text, raw.evidence) || raw.confidence < this.config.capture.confidenceThreshold) {
        this.activity?.("capture.rejected", {
          reason: !validUserExtracted(raw) ? "invalid_schema_or_kind" : !evidenceAppearsInUserMessage(raw.evidence, userText) ? "evidence_not_in_user_message" : isTransientTaskMemory(raw.text, raw.evidence) ? "transient_task_intent" : "low_confidence",
          ...(this.config.activityLog.includeText ? { candidate: raw } : {}),
        });
        continue;
      }
      const validate = async (text: string) => {
        const validation = await this.fast.json<{ accept?: boolean; reason?: string }>(
          this.config.prompts.jsonOnly,
          validationPrompt(userText, context, { text, kind: raw.kind, scope: raw.scope, evidence: raw.evidence }, this.config.prompts),
          signal,
        );
        this.activity?.("capture.validated", { accept: validation.accept === true, ...(this.config.activityLog.includeText ? { reason: validation.reason ?? "" } : {}) });
        return validation.accept === true;
      };
      if (!await validate(raw.text)) {
        this.activity?.("capture.rejected", { reason: "durability_or_entailment_validation_failed" });
        continue;
      }
      const source: MemorySource = {
        actor: "user",
        sessionId: this.sessionId,
        cwd: this.cwd,
        cause: "explicit_user_statement",
        reason: "Durable information explicitly supported by the user's message",
        userText: this.safeSourceText(userText, 500),
        evidence: this.safeSourceText(raw.evidence, 500),
      };
      const didStore = await this.resolveAndStore(raw, source, this.config.capture.similarityThreshold, validate, signal);
      if (didStore) stored++;
    }
    this.activity?.("capture.completed", { stored });
    return stored;
  }

  async captureAssistantInvestigation(investigation: string, cause: string, elapsedMs: number, signal?: AbortSignal): Promise<number> {
    if (!this.config.assistantCapture.enabled || elapsedMs < this.config.assistantCapture.minimumElapsedMs || !investigation.trim()) return 0;
    this.activity?.("assistant_capture.started", { elapsedMs, characters: investigation.length, ...(this.config.activityLog.includeText ? { cause } : {}) });
    const result = await this.fast.json<{ memories?: AssistantExtracted[] }>(
      this.config.prompts.jsonOnly,
      assistantExtractionPrompt(investigation, cause, elapsedMs, this.projectId, this.config.prompts),
      signal,
    );
    let stored = 0;
    for (const raw of result.memories ?? []) {
      if (!validAssistantExtracted(raw) || !evidenceAppearsInUserMessage(raw.evidence, investigation) || raw.confidence < this.config.assistantCapture.confidenceThreshold) {
        this.activity?.("assistant_capture.rejected", { reason: "invalid_evidence_kind_or_confidence", ...(this.config.activityLog.includeText ? { candidate: raw } : {}) });
        continue;
      }
      const validate = (text: string) => this.validateAssistantCandidate(
        investigation,
        cause,
        elapsedMs,
        { ...raw, text },
        signal,
      );
      if (!await validate(raw.text)) continue;
      const source: MemorySource = {
        actor: "assistant",
        sessionId: this.sessionId,
        cwd: this.cwd,
        cause: this.safeSourceText(cause, 500),
        reason: this.safeSourceText(raw.whyStored, 500),
        elapsedMs,
        evidence: this.safeSourceText(raw.evidence, 500),
      };
      const candidate = { ...raw, confidence: Math.min(raw.confidence, this.config.assistantCapture.maximumConfidence) };
      if (await this.resolveAndStore(candidate, source, this.config.assistantCapture.similarityThreshold, validate, signal)) stored++;
    }
    this.activity?.("assistant_capture.completed", { stored, elapsedMs });
    return stored;
  }

  async rememberAssistantResult(input: AssistantResultInput, investigation: string, cause: string, elapsedMs: number, signal?: AbortSignal): Promise<boolean> {
    if (!this.config.assistantCapture.enabled) throw new Error("Assistant-result capture is disabled");
    if (elapsedMs < this.config.assistantCapture.minimumElapsedMs) {
      throw new Error(`Assistant-result memories require at least ${Math.ceil(this.config.assistantCapture.minimumElapsedMs / 1000)} seconds of investigation`);
    }
    if (!validAssistantInput(input)) throw new Error("Invalid assistant-result memory");
    if (input.confidence < this.config.assistantCapture.confidenceThreshold) return false;
    const candidate: AssistantExtracted = {
      ...input,
      evidence: input.text,
      whyStored: input.reason,
      confidence: Math.min(input.confidence, this.config.assistantCapture.maximumConfidence),
    };
    const evidenceText = `${investigation}\n\nPROPOSED RESULT:\n${input.text}`;
    const validate = (text: string) => this.validateAssistantCandidate(evidenceText, cause, elapsedMs, { ...candidate, text }, signal);
    if (!await validate(candidate.text)) return false;
    const source: MemorySource = {
      actor: "assistant",
      sessionId: this.sessionId,
      cwd: this.cwd,
      cause: this.safeSourceText(cause, 500),
      reason: this.safeSourceText(input.reason, 500),
      elapsedMs,
      evidence: this.safeSourceText(input.text, 500),
    };
    return this.resolveAndStore(candidate, source, this.config.assistantCapture.similarityThreshold, validate, signal);
  }

  async correctAssistantMemory(memoryId: string, correctedText: string, reason: string, signal?: AbortSignal): Promise<MemoryRecord> {
    const records = await this.store.list({ scopes: ["global", "project"], kinds: ["user_profile", "fact", "skill_workflow"], projectId: this.projectId }, 10000);
    const record = records.find((candidate) => candidate.id === memoryId && candidate.status === "active");
    if (!record) throw new Error(`Active memory ${memoryId} was not found`);
    if (record.source.actor !== "assistant") throw new Error("Only assistant-generated memories can be corrected by the model");
    if (!reason.trim()) throw new Error("A concrete correction reason is required");
    const text = sanitizePersistedText(correctedText, this.config.security.maxMemoryCharacters, this.config.security.redactSecrets);
    if (text === record.text) throw new Error("Corrected memory text must differ from the existing text");
    const now = new Date().toISOString();
    const source: MemorySource = {
      actor: "assistant",
      sessionId: this.sessionId,
      cwd: this.cwd,
      cause: "correction_of_inaccurate_assistant_memory",
      reason: this.safeSourceText(reason, 500),
      evidence: this.safeSourceText(text, 500),
    };
    const updated: MemoryRecord = {
      ...record,
      text,
      source,
      sourceHistory: [...(record.sourceHistory ?? []), record.source],
      confidence: Math.min(record.confidence, this.config.assistantCapture.maximumConfidence),
      priority: Math.min(record.priority ?? this.config.assistantCapture.priority, this.config.assistantCapture.priority),
      updatedAt: now,
      embeddingModel: embeddingModels(this.embedder).document,
    };
    const [vector] = await embedDocuments(this.embedder, [text], signal);
    if (!vector) throw new Error("Embedding failed");
    const mutation = await this.store.mutate(memoryId, latest => {
      if (latest.status !== "active" || latest.source.actor !== "assistant" || latest.kind !== record.kind || latest.scope !== record.scope || latest.projectId !== record.projectId || latest.text !== record.text || JSON.stringify(latest.source) !== JSON.stringify(record.source)) return undefined;
      return { record: {
        ...latest,
        text,
        source,
        sourceHistory: [...(latest.sourceHistory ?? []), latest.source],
        confidence: Math.min(latest.confidence, this.config.assistantCapture.maximumConfidence),
        priority: Math.min(latest.priority ?? this.config.assistantCapture.priority, this.config.assistantCapture.priority),
        updatedAt: now,
        embeddingModel: embeddingModels(this.embedder).document,
      }, vector };
    });
    if (mutation.status !== "updated" || !mutation.record) throw new Error("Assistant memory changed while correction was being prepared");
    this.activity?.("memory.corrected", { id: memoryId, actor: "assistant", ...(this.config.activityLog.includeText ? { previousText: record.text, text, reason: source.reason } : {}) });
    return mutation.record;
  }

  async sweepLifecycle(now = new Date()): Promise<{ initialized: number; expired: number }> {
    if (!this.config.memoryLifecycle.enabled) return { initialized: 0, expired: 0 };
    const records = await this.store.list({ status: "active", scopes: ["global", "project"], kinds: ["user_profile", "fact", "skill_workflow"], projectId: this.projectId }, 10000);
    let initialized = 0;
    let expired = 0;
    for (const record of records) {
      let before: MemoryRecord | undefined;
      let committed: ReturnType<typeof advanceMemoryLifecycle> | undefined;
      const mutation = await this.store.mutate(record.id, latest => {
        const current = advanceMemoryLifecycle(latest, now, this.sessionId, this.config.memoryLifecycle);
        if (!current.initialized && !current.expiredBy && current.decayedDays === 0) return undefined;
        before = structuredClone(latest);
        committed = current;
        return { record: current.record };
      });
      if (mutation.status !== "updated" || !committed || !before) continue;
      if (committed.initialized) initialized++;
      if (committed.expiredBy) {
        expired++;
        this.activity?.("memory.expired", { id: record.id, cause: committed.expiredBy, confidenceBefore: before.confidence, confidenceAfter: mutation.record!.confidence, decayRate: mutation.record!.decayRate, decayedDays: committed.decayedDays });
      }
    }
    if (initialized) this.activity?.("lifecycle.migrated", { records: initialized, policy: "full_grace_from_first_post-upgrade_session" });
    return { initialized, expired };
  }

  async recordFeedback(memoryId: string, steerToken: string, outcome: FeedbackOutcome, reason: string): Promise<MemoryRecord | undefined> {
    const records = await this.store.list({ scopes: ["global", "project"], kinds: ["user_profile", "fact", "skill_workflow"], projectId: this.projectId }, 10000);
    const record = records.find((candidate) => candidate.id === memoryId && candidate.status === "active");
    if (!record) return undefined;
    const at = new Date().toISOString();
    let confidenceBefore: number | undefined;
    const mutation = await this.store.mutate(memoryId, latest => {
      if (latest.status !== "active") return undefined;
      confidenceBefore = latest.confidence;
      let next = applyConfidenceFeedback(latest, { outcome, sessionId: this.sessionId, steerToken, reason: this.safeSourceText(reason, 500), at }, this.config.memoryLifecycle);
      if (this.config.memoryLifecycle.enabled) next = outcome === "useful" ? reinforceMemoryLifecycle(next, new Date(at), this.sessionId, this.config.memoryLifecycle, "useful_feedback") : advanceMemoryLifecycle(next, new Date(at), this.sessionId, this.config.memoryLifecycle).record;
      return { record: next };
    });
    if (mutation.status !== "updated") return undefined;
    this.activity?.("feedback.recorded", { id: memoryId, outcome, confidenceBefore, confidenceAfter: mutation.record!.confidence, steerToken, ...(this.config.activityLog.includeText ? { reason: mutation.record?.feedback?.history.at(-1)?.reason } : {}) });
    return mutation.record;
  }

  async planCompaction(signal?: AbortSignal): Promise<CompactionPlan> {
    const records = await this.store.list({ status: "active", scopes: ["global", "project"], kinds: ["user_profile", "fact", "skill_workflow"], projectId: this.projectId }, 10000);
    if (records.length < 2) return { clusters: [], proposals: [] };
    const vectors = await embedDocuments(this.embedder, records.map((record) => record.text), signal);
    const clusters = pairSimilarMemories(records, vectors, this.config.compaction.similarityThreshold, this.config.compaction.maximumProposals);
    const proposals: CompactionProposal[] = [];
    for (const cluster of clusters) {
      const decision = await this.fast.json<{ merge?: boolean; text?: string; reason?: string }>(
        this.config.prompts.jsonOnly,
        compactionPrompt(cluster.records.map((record) => ({ id: record.id, text: record.text })), this.config.prompts),
        signal,
      );
      const longest = Math.max(...cluster.records.map((record) => record.text.length));
      const text = decision.text?.trim() ?? "";
      proposals.push({
        enabled: decision.merge === true && text.length > 0 && text.length <= Math.min(this.config.security.maxMemoryCharacters, Math.ceil(longest * 1.25)) && !/[\r\n]/.test(text),
        sourceIds: cluster.records.map((record) => record.id),
        text,
        reason: decision.reason?.trim() ?? "",
      });
    }
    this.activity?.("compaction.planned", { clusters: clusters.length, proposed: proposals.filter((proposal) => proposal.enabled).length });
    return { clusters, proposals };
  }

  async applyCompaction(proposal: CompactionProposal, cluster: MemoryCluster, signal?: AbortSignal): Promise<MemoryRecord> {
    const sourceIds = new Set(proposal.sourceIds);
    const records = cluster.records.filter((record) => sourceIds.has(record.id));
    if (records.length < 2 || records.length !== sourceIds.size) throw new Error("Compaction proposal does not match its reviewed cluster");
    const first = records[0]!;
    if (!records.every((record) => record.scope === first.scope && record.kind === first.kind && memoryActor(record) === memoryActor(first) && record.projectId === first.projectId)) {
      throw new Error("Compaction cannot cross scope, kind, project, or authority boundaries");
    }
    const text = sanitizePersistedText(proposal.text, this.config.security.maxMemoryCharacters, this.config.security.redactSecrets);
    const validation = await this.fast.json<{ accept?: boolean; reason?: string }>(
      this.config.prompts.jsonOnly,
      compactionValidationPrompt(records.map((record) => ({ id: record.id, text: record.text })), text, this.config.prompts),
      signal,
    );
    if (validation.accept !== true) throw new Error(`Compaction validation failed: ${validation.reason ?? "merge was not entailed by every source"}`);
    const now = new Date().toISOString();
    const actor = memoryActor(first);
    const history = uniqueSources(records.flatMap((record) => [...(record.sourceHistory ?? []), record.source]));
    const source: MemorySource = {
      actor,
      sessionId: this.sessionId,
      cwd: this.cwd,
      cause: "user_invoked_compaction",
      reason: this.safeSourceText(proposal.reason || "User reviewed and approved consolidation of highly similar memories", 500),
      evidence: this.safeSourceText(records.map((record) => record.id).join(","), 500),
    };
    const baseCompacted: MemoryRecord = {
      id: randomUUID(),
      text,
      kind: first.kind,
      scope: first.scope,
      ...(first.scope === "project" ? { projectId: first.projectId ?? this.projectId } : {}),
      // A combined memory inherits the strongest observed usefulness of its sources.
      confidence: Math.max(...records.map((record) => record.confidence)),
      decayRate: Math.min(...records.map((record) => record.decayRate ?? this.config.memoryLifecycle.decay.initialRate)),
      priority: Math.max(...records.map(memoryPriority)),
      status: "active",
      supersedes: [...new Set(records.flatMap((record) => [record.id, ...(record.supersedes ?? [])]))],
      source,
      sourceHistory: history,
      ...(records.some((record) => record.feedback) ? {
        feedback: {
          useful: records.reduce((sum, record) => sum + (record.feedback?.useful ?? 0), 0),
          unhelpful: records.reduce((sum, record) => sum + (record.feedback?.unhelpful ?? 0), 0),
          lastAt: records.flatMap((record) => record.feedback?.lastAt ?? []).sort().at(-1) ?? now,
          history: records.flatMap((record) => record.feedback?.history ?? []).sort((left, right) => left.at.localeCompare(right.at)).slice(-this.config.memoryLifecycle.feedback.historyLimit),
        },
      } : {}),
      createdAt: records.map((record) => record.createdAt).sort()[0] ?? now,
      updatedAt: now,
      embeddingModel: embeddingModels(this.embedder).document,
      schemaVersion: 2,
    };
    const compacted = this.config.memoryLifecycle.enabled
      ? reinforceMemoryLifecycle(baseCompacted, new Date(now), this.sessionId, this.config.memoryLifecycle, "user_compaction")
      : baseCompacted;
    const [vector] = await embedDocuments(this.embedder, [compacted.text], signal);
    if (!vector) throw new Error("Embedding failed");
    const reviewedById = new Map(records.map(record => [record.id, JSON.stringify(record)]));
    const stored = await this.store.compact([...sourceIds], latest => {
      // Storage may order source ids differently from the reviewed proposal. Match by
      // identity, while still rejecting any changed/missing/extra reviewed record.
      if (latest.length !== reviewedById.size || latest.some(record => reviewedById.get(record.id) !== JSON.stringify(record))) throw new Error("Compaction sources changed after review");
      return { record: compacted, vector };
    });
    this.activity?.("compaction.applied", { id: stored.id, sourceIds: [...sourceIds], actor, scope: stored.scope, kind: stored.kind, ...(this.config.activityLog.includeText ? { text: stored.text } : {}) });
    return stored;
  }

  async recall(context: string, signal?: AbortSignal, excludedIds: ReadonlySet<string> = new Set(), activeContext = context): Promise<RecallResult | undefined> {
    if (!this.config.recall.enabled || !context.trim()) return undefined;
    this.activity?.("recall.started", { contextCharacters: context.length });
    const query = await this.fast.json<{ query?: string }>(this.config.prompts.jsonOnly, queryPrompt(context, this.config.prompts), signal);
    this.activity?.("recall.query", { query: this.config.activityLog.includeText ? query.query ?? "" : undefined });
    if (!query.query?.trim()) return undefined;
    const [vector] = await embedQuery(this.embedder, [query.query.trim()], signal);
    if (!vector) return undefined;
    const searchLimit = Math.max(this.config.recall.topK * 5, this.config.recall.topK + 20);
    const searched = await this.store.search(vector, { status: "active", scopes: ["global", "project"], kinds: ["user_profile", "fact", "skill_workflow"], projectId: this.projectId }, searchLimit);
    const frequencySuppressed = searched.filter((match) => excludedIds.has(match.record.id));
    if (frequencySuppressed.length) this.activity?.("recall.frequency_filtered", { ids: frequencySuppressed.map((match) => match.record.id) });
    const contextSuppressed = searched.filter((match) => sourceEvidenceAppearsInContext(match.record, activeContext));
    if (contextSuppressed.length) this.activity?.("recall.source_context_filtered", { ids: contextSuppressed.map((match) => match.record.id) });
    const raw = searched.filter((match) => !excludedIds.has(match.record.id) && !sourceEvidenceAppearsInContext(match.record, activeContext) && match.score >= this.config.recall.minVectorScore);
    const scored = rankMemoryMatches(raw);
    const minimumAgeMs = Math.max(0, this.config.recall.minimumMemoryAgeMinutes) * 60_000;
    const now = Date.now();
    const suppressed = scored.filter((match) => isRecentCurrentSessionMemory(match.record, this.sessionId, now, minimumAgeMs));
    const matches = scored.filter((match) => !isRecentCurrentSessionMemory(match.record, this.sessionId, now, minimumAgeMs)).slice(0, this.config.recall.topK);
    if (suppressed.length) this.activity?.("recall.age_filtered", { minimumMemoryAgeMinutes: this.config.recall.minimumMemoryAgeMinutes, ids: suppressed.map((match) => match.record.id) });
    this.activity?.("recall.matches", { matches: matches.map((match) => ({ id: match.record.id, score: match.score, scope: match.record.scope, kind: match.record.kind, actor: memoryActor(match.record), confidence: match.record.confidence, ...(this.config.activityLog.includeText ? { text: match.record.text } : {}) })) });
    if (!matches.length) return undefined;
    const candidates = matches.map((match) => `${match.record.id} rank=${match.score.toFixed(3)} source=${memoryActor(match.record)} confidence=${match.record.confidence.toFixed(2)} scope=${match.record.scope} kind=${match.record.kind}\n${match.record.text}`).join("\n\n");
    const judged = await this.fast.json<{ relevantIds?: string[]; reason?: string }>(this.config.prompts.jsonOnly, judgePrompt(context, candidates, this.config.prompts), signal);
    const ids = new Set(judged.relevantIds ?? []);
    const relevant = matches.filter((match) => ids.has(match.record.id));
    this.activity?.("recall.judged", {
      relevantIds: relevant.map((match) => match.record.id),
      ...(this.config.activityLog.includeText ? { reason: judged.reason ?? "" } : {}),
    });
    if (!relevant.length) return undefined;
    return { reason: judged.reason?.trim() ?? "", relevant };
  }

  /** Apply relevance only after the runtime has actually queued the steer. */
  async recordRecallDelivery(records: readonly MemoryRecord[]): Promise<void> {
    if (!this.config.memoryLifecycle.enabled) return;
    const now = new Date();
    for (const record of records) {
      const mutation = await this.store.mutate(record.id, latest => latest.status === "active" ? { record: reinforceMemoryLifecycle(latest, now, this.sessionId, this.config.memoryLifecycle, "relevant_recall") } : undefined);
      if (mutation.status === "updated" && mutation.record) this.activity?.("memory.reinforced", { id: record.id, cause: "relevant_recall", confidence: mutation.record.confidence, decayRate: mutation.record.decayRate, lastDecayDate: mutation.record.lifecycle?.lastDecayDate });
    }
  }

  private async validateAssistantCandidate(investigation: string, cause: string, elapsedMs: number, candidate: AssistantExtracted, signal?: AbortSignal): Promise<boolean> {
    const validation = await this.fast.json<{ accept?: boolean; reason?: string }>(
      this.config.prompts.jsonOnly,
      assistantValidationPrompt(investigation, cause, elapsedMs, candidate, this.config.prompts),
      signal,
    );
    this.activity?.("assistant_capture.validated", { accept: validation.accept === true, ...(this.config.activityLog.includeText ? { reason: validation.reason ?? "" } : {}) });
    return validation.accept === true;
  }

  private async resolveAndStore(
    raw: Pick<Extracted, "text" | "kind" | "scope" | "confidence">,
    source: MemorySource,
    similarityThreshold: number,
    validateCanonical: (text: string) => Promise<boolean>,
    signal?: AbortSignal,
    conflictRetries = 0,
  ): Promise<boolean> {
    let text: string;
    try { text = sanitizePersistedText(raw.text, this.config.security.maxMemoryCharacters, this.config.security.redactSecrets); } catch { return false; }
    if (text.includes("[REDACTED_PRIVATE_KEY]")) return false;
    const [vector] = await embedDocuments(this.embedder, [text], signal);
    if (!vector) return false;

    const nearest = await this.store.search(vector, {
      status: "active",
      scopes: [raw.scope],
      kinds: ["user_profile", "fact", "skill_workflow"],
      ...(raw.scope === "project" ? { projectId: this.projectId } : {}),
    }, 8);
    const plausible = nearest.filter((match) => match.score >= similarityThreshold);
    this.activity?.("capture.neighbors", {
      actor: source.actor,
      matches: nearest.map((match) => ({ id: match.record.id, score: match.score, actor: memoryActor(match.record), ...(this.config.activityLog.includeText ? { text: match.record.text } : {}) })),
    });

    let action: "add" | "replace" | "noop" = "add";
    let targetId: string | undefined;
    if (plausible.length) {
      const existing = plausible.map((match) => `${match.record.id} (${match.score.toFixed(3)}, source=${memoryActor(match.record)}, confidence=${match.record.confidence.toFixed(2)}): ${match.record.text}`).join("\n");
      const decision = await this.fast.json<{ action?: "add" | "replace" | "noop"; targetId?: string | null; text?: string }>(
        this.config.prompts.jsonOnly,
        mergePrompt(text, existing, source.actor ?? "user", this.config.prompts),
        signal,
      );
      action = decision.action ?? "add";
      targetId = decision.targetId ?? undefined;
      if (decision.text?.trim()) {
        try { text = sanitizePersistedText(decision.text, this.config.security.maxMemoryCharacters, this.config.security.redactSecrets); } catch { return false; }
      }
    }
    if (action === "noop") return false;
    const target = action === "replace" ? plausible.find((match) => match.record.id === targetId)?.record : undefined;
    if (action === "replace" && !target) action = "add";
    if (source.actor === "assistant" && target && memoryActor(target) === "user") return false;
    if (text !== raw.text.trim() && !await validateCanonical(text)) return false;

    try { text = sanitizePersistedText(text, this.config.security.maxMemoryCharacters, this.config.security.redactSecrets); } catch { return false; }
    const [canonicalVector] = text === raw.text.trim() ? [vector] : await embedDocuments(this.embedder, [text], signal);
    if (!canonicalVector) return false;
    const replacing = action === "replace" ? target : undefined;
    const now = new Date().toISOString();
    const actor = source.actor ?? "user";
    const baseRecord: MemoryRecord = {
      id: replacing?.id ?? randomUUID(),
      text,
      kind: raw.kind,
      scope: raw.scope,
      ...(raw.scope === "project" ? { projectId: this.projectId } : {}),
      confidence: replacing?.confidence ?? this.config.memoryLifecycle.confidence.initial,
      decayRate: replacing?.decayRate ?? this.config.memoryLifecycle.decay.initialRate,
      priority: actor === "assistant" ? this.config.assistantCapture.priority : 1,
      status: "active",
      ...(replacing ? { supersedes: (replacing.supersedes ?? []).filter(id => id !== replacing.id), sourceHistory: [...(replacing.sourceHistory ?? []), replacing.source] } : {}),
      source,
      ...(replacing?.feedback ? { feedback: replacing.feedback } : {}),
      ...(replacing?.lifecycle ? { lifecycle: replacing.lifecycle } : {}),
      createdAt: replacing?.createdAt ?? now,
      updatedAt: now,
      embeddingModel: embeddingModels(this.embedder).document,
      schemaVersion: 2,
    };
    const record = this.config.memoryLifecycle.enabled
      ? initializeMemoryLifecycle(baseRecord, new Date(now), this.sessionId, this.config.memoryLifecycle)
      : baseRecord;
    let committed = record;
    if (!replacing) {
      if (await this.store.insert({ record, vector: canonicalVector }) !== "inserted") return false;
    } else {
      const mutation = await this.store.mutate(replacing.id, latest => {
        if (latest.status !== "active" || latest.scope !== replacing.scope || latest.kind !== replacing.kind || latest.projectId !== replacing.projectId || memoryActor(latest) !== memoryActor(replacing) || latest.text !== replacing.text || JSON.stringify(latest.source) !== JSON.stringify(replacing.source)) return undefined;
        const priority = actor === memoryActor(latest) ? latest.priority : actor === "user" ? 1 : this.config.assistantCapture.priority;
        return { record: { ...record, confidence: latest.confidence, decayRate: latest.decayRate, priority, feedback: latest.feedback, lifecycle: latest.lifecycle, sourceHistory: [...(latest.sourceHistory ?? []), latest.source], supersedes: (latest.supersedes ?? []).filter(id => id !== latest.id) }, vector: canonicalVector };
      });
      if (mutation.status !== "updated" || !mutation.record) {
        this.activity?.("capture.conflict", { targetId: replacing.id, retry: conflictRetries + 1 });
        if (conflictRetries < 2) return this.resolveAndStore(raw, source, similarityThreshold, validateCanonical, signal, conflictRetries + 1);
        throw new Error("Memory changed repeatedly while capture was being resolved; retry the capture");
      }
      committed = mutation.record;
    }
    const created = !replacing;
    this.activity?.("capture.stored", { ...this.memoryActivity(committed), created });
    try {
      this.onMemoryStored?.(committed, created);
    } catch {
      // Storage is already committed. Display-only feedback must never turn a
      // successful memory write into a reported capture failure.
      try { this.activity?.("capture.feedback_failed", { id: committed.id }); } catch {}
    }
    return true;
  }

  private safeSourceText(value: string, max: number): string {
    const redacted = this.config.security.redactSecrets ? redactSecrets(value) : value;
    return redacted.slice(0, max);
  }

  private memoryActivity(record: MemoryRecord): object {
    return {
      id: record.id, kind: record.kind, scope: record.scope, projectId: record.projectId,
      actor: memoryActor(record), confidence: record.confidence, priority: memoryPriority(record),
      elapsedMs: record.source.elapsedMs,
      ...(this.config.activityLog.includeText ? { text: record.text, cause: record.source.cause, reason: record.source.reason } : {}),
    };
  }
}

export function memoryActor(record: MemoryRecord): MemoryActor {
  return record.source.actor ?? "user";
}

export function memoryPriority(record: MemoryRecord): number {
  return Number.isFinite(record.priority) ? Math.max(0, record.priority ?? 1) : memoryActor(record) === "assistant" ? 0.55 : 1;
}

export function rankMemoryMatches(matches: MemoryMatch[]): MemoryMatch[] {
  return matches.map((match) => ({
    record: match.record,
    score: match.score * memoryPriority(match.record) * (0.5 + 0.5 * Math.max(0, Math.min(1, match.record.confidence))),
  })).sort((a, b) => b.score - a.score);
}

function uniqueSources(sources: MemorySource[]): MemorySource[] {
  const seen = new Set<string>();
  return sources.filter((source) => {
    const key = JSON.stringify([source.actor ?? "user", source.sessionId, source.cwd, source.cause, source.reason, source.elapsedMs, source.evidence, source.userText]);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function isRecentCurrentSessionMemory(record: MemoryRecord, sessionId: string, now: number, minimumAgeMs: number): boolean {
  if (minimumAgeMs <= 0 || record.source.sessionId !== sessionId) return false;
  const createdAt = Date.parse(record.createdAt);
  return Number.isFinite(createdAt) && now - createdAt < minimumAgeMs;
}

function validUserExtracted(value: Extracted): boolean {
  return Boolean(
    value && typeof value.text === "string" && typeof value.evidence === "string" &&
    ["user_profile", "fact", "skill_workflow"].includes(value.kind) &&
    ["global", "project"].includes(value.scope) && Number.isFinite(value.confidence),
  );
}

function validAssistantExtracted(value: AssistantExtracted): boolean {
  return Boolean(
    value && typeof value.text === "string" && typeof value.evidence === "string" && typeof value.whyStored === "string" && value.whyStored.trim() &&
    ["fact", "skill_workflow"].includes(value.kind) && ["global", "project"].includes(value.scope) && Number.isFinite(value.confidence),
  );
}

function validAssistantInput(value: AssistantResultInput): boolean {
  return Boolean(
    value && typeof value.text === "string" && value.text.trim() && typeof value.reason === "string" && value.reason.trim() &&
    ["fact", "skill_workflow"].includes(value.kind) && ["global", "project"].includes(value.scope) && Number.isFinite(value.confidence) && value.confidence >= 0 && value.confidence <= 1,
  );
}
