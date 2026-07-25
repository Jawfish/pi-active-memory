import { randomUUID } from "node:crypto";
import type { ActivitySink } from "./activity-log.js";
import type { ActiveMemoryConfig, FastModelRunner, MemoryActor, MemoryKind, MemoryMatch, MemoryRecord, MemoryScope, MemorySource, VectorStore } from "./types.js";
import { Embedder } from "./embeddings.js";
import { assistantExtractionPrompt, assistantValidationPrompt, extractionPrompt, JSON_ONLY, judgePrompt, mergePrompt, queryPrompt, validationPrompt } from "./prompts.js";
import { evidenceAppearsInUserMessage, redactSecrets } from "./utils.js";

interface Extracted { text: string; kind: MemoryKind; scope: MemoryScope; confidence: number; evidence: string }
interface AssistantExtracted extends Extracted { whyStored: string }
interface RecallResult { instruction: string; reason: string; relevant: MemoryMatch[] }
export interface AssistantResultInput { text: string; kind: Exclude<MemoryKind, "user_profile">; scope: MemoryScope; confidence: number; reason: string }

export class MemoryEngine {
  constructor(
    private readonly config: ActiveMemoryConfig,
    private readonly store: VectorStore,
    private readonly embedder: Embedder,
    private readonly fast: FastModelRunner,
    private readonly projectId: string,
    private readonly sessionId: string,
    private readonly cwd: string,
    private readonly activity?: ActivitySink,
  ) {}

  async capture(userText: string, context: string, signal?: AbortSignal): Promise<number> {
    if (!this.config.capture.enabled || userText.trim().length < this.config.capture.minCharacters) return 0;
    this.activity?.("capture.started", { characters: userText.length, ...(this.config.activityLog.includeText ? { userText } : {}) });
    const result = await this.fast.json<{ memories?: Extracted[] }>(JSON_ONLY, extractionPrompt(userText, context, this.projectId), signal);
    this.activity?.("capture.extracted", {
      count: result.memories?.length ?? 0,
      memories: (result.memories ?? []).map((memory) => ({ kind: memory.kind, scope: memory.scope, confidence: memory.confidence, evidenceValid: typeof memory.evidence === "string" && evidenceAppearsInUserMessage(memory.evidence, userText), ...(this.config.activityLog.includeText ? { text: memory.text, evidence: memory.evidence } : {}) })),
      fastModel: this.fast.selectedModel(),
    });
    let stored = 0;
    for (const raw of result.memories ?? []) {
      if (!validUserExtracted(raw) || !evidenceAppearsInUserMessage(raw.evidence, userText) || raw.confidence < this.config.capture.confidenceThreshold) {
        this.activity?.("capture.rejected", {
          reason: !validUserExtracted(raw) ? "invalid_schema_or_kind" : !evidenceAppearsInUserMessage(raw.evidence, userText) ? "evidence_not_in_user_message" : "low_confidence",
          ...(this.config.activityLog.includeText ? { candidate: raw } : {}),
        });
        continue;
      }
      const validate = async (text: string) => {
        const validation = await this.fast.json<{ accept?: boolean; reason?: string }>(
          JSON_ONLY,
          validationPrompt(userText, context, { text, kind: raw.kind, scope: raw.scope, evidence: raw.evidence }),
          signal,
        );
        this.activity?.("capture.validated", { accept: validation.accept === true, reason: validation.reason ?? "", fastModel: this.fast.selectedModel() });
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
      JSON_ONLY,
      assistantExtractionPrompt(investigation, cause, elapsedMs, this.projectId),
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

  async recall(context: string, signal?: AbortSignal): Promise<RecallResult | undefined> {
    if (!this.config.recall.enabled || !context.trim()) return undefined;
    this.activity?.("recall.started", { contextCharacters: context.length });
    const query = await this.fast.json<{ query?: string }>(JSON_ONLY, queryPrompt(context), signal);
    this.activity?.("recall.query", { query: this.config.activityLog.includeText ? query.query ?? "" : undefined, fastModel: this.fast.selectedModel() });
    if (!query.query?.trim()) return undefined;
    const [vector] = await this.embedder.embed([query.query.trim()], signal);
    if (!vector) return undefined;
    const searchLimit = Math.max(this.config.recall.topK * 5, this.config.recall.topK + 20);
    const raw = (await this.store.search(vector, { status: "active", scopes: ["global", "project"], kinds: ["user_profile", "fact", "skill_workflow"], projectId: this.projectId }, searchLimit))
      .filter((match) => match.score >= this.config.recall.minVectorScore);
    const scored = rankMemoryMatches(raw);
    const minimumAgeMs = Math.max(0, this.config.recall.minimumMemoryAgeMinutes) * 60_000;
    const now = Date.now();
    const suppressed = scored.filter((match) => isRecentCurrentSessionMemory(match.record, this.sessionId, now, minimumAgeMs));
    const matches = scored.filter((match) => !isRecentCurrentSessionMemory(match.record, this.sessionId, now, minimumAgeMs)).slice(0, this.config.recall.topK);
    if (suppressed.length) this.activity?.("recall.age_filtered", { minimumMemoryAgeMinutes: this.config.recall.minimumMemoryAgeMinutes, ids: suppressed.map((match) => match.record.id) });
    this.activity?.("recall.matches", { matches: matches.map((match) => ({ id: match.record.id, score: match.score, scope: match.record.scope, kind: match.record.kind, actor: memoryActor(match.record), confidence: match.record.confidence, ...(this.config.activityLog.includeText ? { text: match.record.text } : {}) })) });
    if (!matches.length) return undefined;
    const candidates = matches.map((match) => `${match.record.id} rank=${match.score.toFixed(3)} source=${memoryActor(match.record)} confidence=${match.record.confidence.toFixed(2)} scope=${match.record.scope} kind=${match.record.kind}\n${match.record.text}`).join("\n\n");
    const judged = await this.fast.json<{ relevantIds?: string[]; instruction?: string; reason?: string }>(JSON_ONLY, judgePrompt(context, candidates), signal);
    const ids = new Set(judged.relevantIds ?? []);
    const relevant = matches.filter((match) => ids.has(match.record.id));
    this.activity?.("recall.judged", {
      relevantIds: relevant.map((match) => match.record.id),
      ...(this.config.activityLog.includeText ? { instruction: judged.instruction ?? "", reason: judged.reason ?? "" } : {}),
      fastModel: this.fast.selectedModel(),
    });
    if (!judged.instruction?.trim() || !relevant.length) return undefined;
    return { instruction: judged.instruction.trim(), reason: judged.reason?.trim() ?? "", relevant };
  }

  private async validateAssistantCandidate(investigation: string, cause: string, elapsedMs: number, candidate: AssistantExtracted, signal?: AbortSignal): Promise<boolean> {
    const validation = await this.fast.json<{ accept?: boolean; reason?: string }>(
      JSON_ONLY,
      assistantValidationPrompt(investigation, cause, elapsedMs, candidate),
      signal,
    );
    this.activity?.("assistant_capture.validated", { accept: validation.accept === true, reason: validation.reason ?? "", fastModel: this.fast.selectedModel() });
    return validation.accept === true;
  }

  private async resolveAndStore(
    raw: Pick<Extracted, "text" | "kind" | "scope" | "confidence">,
    source: MemorySource,
    similarityThreshold: number,
    validateCanonical: (text: string) => Promise<boolean>,
    signal?: AbortSignal,
  ): Promise<boolean> {
    let text = raw.text.trim().slice(0, this.config.security.maxMemoryCharacters);
    if (this.config.security.redactSecrets) text = redactSecrets(text);
    if (!text || text.includes("[REDACTED_PRIVATE_KEY]")) return false;
    const [vector] = await this.embedder.embed([text], signal);
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
        JSON_ONLY,
        mergePrompt(text, existing, source.actor ?? "user"),
        signal,
      );
      action = decision.action ?? "add";
      targetId = decision.targetId ?? undefined;
      if (decision.text?.trim()) text = decision.text.trim().slice(0, this.config.security.maxMemoryCharacters);
    }
    if (action === "noop") return false;
    const target = action === "replace" ? plausible.find((match) => match.record.id === targetId)?.record : undefined;
    if (action === "replace" && !target) action = "add";
    if (source.actor === "assistant" && target && memoryActor(target) === "user") return false;
    if (text !== raw.text.trim() && !await validateCanonical(text)) return false;

    if (this.config.security.redactSecrets) text = redactSecrets(text);
    const [canonicalVector] = text === raw.text.trim() ? [vector] : await this.embedder.embed([text], signal);
    if (!canonicalVector) return false;
    const replacing = action === "replace" ? target : undefined;
    const now = new Date().toISOString();
    const actor = source.actor ?? "user";
    const record: MemoryRecord = {
      id: replacing?.id ?? randomUUID(),
      text,
      kind: raw.kind,
      scope: raw.scope,
      ...(raw.scope === "project" ? { projectId: this.projectId } : {}),
      confidence: Math.max(0, Math.min(1, raw.confidence)),
      priority: actor === "assistant" ? this.config.assistantCapture.priority : 1,
      status: "active",
      ...(replacing ? { supersedes: [...(replacing.supersedes ?? []), replacing.id], sourceHistory: [...(replacing.sourceHistory ?? []), replacing.source] } : {}),
      source,
      createdAt: replacing?.createdAt ?? now,
      updatedAt: now,
      embeddingModel: this.embedder.model,
      schemaVersion: 2,
    };
    await this.store.upsert(record, canonicalVector);
    this.activity?.("capture.stored", this.memoryActivity(record));
    return true;
  }

  private safeSourceText(value: string, max: number): string {
    const sliced = value.slice(0, max);
    return this.config.security.redactSecrets ? redactSecrets(sliced) : sliced;
  }

  private memoryActivity(record: MemoryRecord): object {
    return {
      id: record.id, kind: record.kind, scope: record.scope, projectId: record.projectId,
      actor: memoryActor(record), confidence: record.confidence, priority: memoryPriority(record),
      cause: record.source.cause, reason: record.source.reason, elapsedMs: record.source.elapsedMs,
      ...(this.config.activityLog.includeText ? { text: record.text } : {}),
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
