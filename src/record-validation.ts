import type { MemoryRecord, VectorRow } from "./types.js";

/** Validate every persisted record field, allowing v1's documented incomplete provenance only when requested. */
export function validateMemoryRecord(value: unknown, allowLegacyLineage: boolean): asserts value is MemoryRecord {
  if (!value || typeof value !== "object") throw new Error("Active Memory store contains an invalid record");
  const record = value as Partial<MemoryRecord>;
  const nonEmpty = (item: unknown): item is string => typeof item === "string" && item.length > 0;
  if (!nonEmpty(record.id) || typeof record.text !== "string" || !record.text.trim() || /[\r\n]/.test(record.text) || !["user_profile", "fact", "skill_workflow"].includes(record.kind ?? "") || !["global", "project"].includes(record.scope ?? "") || (record.scope === "project" && !nonEmpty(record.projectId)) || !Number.isFinite(record.confidence) || !["active", "superseded", "deleted"].includes(record.status ?? "") || !validTimestamp(record.createdAt) || !validTimestamp(record.updatedAt) || !nonEmpty(record.embeddingModel) || (record.schemaVersion !== 1 && record.schemaVersion !== 2)) throw new Error("Active Memory store contains an invalid record shape");
  if (record.confidence! < 0 || record.confidence! > 1 || (record.schemaVersion === 2 && !Number.isFinite(record.priority))) throw new Error("Active Memory store contains invalid confidence or required priority");
  if (record.decayRate !== undefined && (!Number.isFinite(record.decayRate) || record.decayRate < 0 || record.decayRate > 1)) throw new Error("Active Memory store contains an invalid decay rate");
  if (record.priority !== undefined && (!Number.isFinite(record.priority) || record.priority < 0 || record.priority > 1)) throw new Error("Active Memory store contains an invalid priority");
  if (record.supersedes !== undefined && (!Array.isArray(record.supersedes) || record.supersedes.some(id => !nonEmpty(id)) || (!allowLegacyLineage && (record.supersedes.includes(record.id) || new Set(record.supersedes).size !== record.supersedes.length)))) throw new Error("Active Memory store contains invalid supersedes");
  validateSource(record.source, record.schemaVersion === 1);
  if (record.sourceHistory !== undefined && (!Array.isArray(record.sourceHistory) || record.sourceHistory.some(source => !validateSource(source, record.schemaVersion === 1)))) throw new Error("Active Memory store contains invalid source history");
  if (record.feedback !== undefined) {
    const feedback = record.feedback;
    if (!Number.isInteger(feedback.useful) || feedback.useful < 0 || !Number.isInteger(feedback.unhelpful) || feedback.unhelpful < 0 || !validTimestamp(feedback.lastAt) || !Array.isArray(feedback.history)) throw new Error("Active Memory store contains invalid feedback");
    for (const item of feedback.history) if (!item || !["useful", "unhelpful"].includes(item.outcome) || !nonEmpty(item.sessionId) || !nonEmpty(item.steerToken) || typeof item.reason !== "string" || !validTimestamp(item.at)) throw new Error("Active Memory store contains invalid feedback history");
  }
  if (record.lifecycle !== undefined) {
    const lifecycle = record.lifecycle;
    if (!validTimestamp(lifecycle.lastDecayDate) || !validTimestamp(lifecycle.lastRelevantAt) || !nonEmpty(lifecycle.lastRelevantSessionId) || !Number.isInteger(lifecycle.reinforcementCount) || lifecycle.reinforcementCount < 0 || !["created", "relevant_recall", "useful_feedback", "user_compaction", "legacy_migration_grace"].includes(lifecycle.lastReinforcementCause) || (lifecycle.deletedAt !== undefined && !validTimestamp(lifecycle.deletedAt)) || (lifecycle.deletionCause !== undefined && lifecycle.deletionCause !== "low_confidence")) throw new Error("Active Memory store contains invalid lifecycle");
  }
}

export function validateVectorRow(row: unknown, dimension?: number, allowLegacyLineage = false): asserts row is VectorRow {
  if (!row || typeof row !== "object") throw new Error("Active Memory store contains an invalid row");
  const candidate = row as Partial<VectorRow>;
  validateMemoryRecord(candidate.record, allowLegacyLineage);
  if (!Array.isArray(candidate.vector) || candidate.vector.length === 0 || candidate.vector.some(value => typeof value !== "number" || !Number.isFinite(value))) throw new Error("Active Memory store contains a non-finite vector");
  if (dimension !== undefined && candidate.vector.length !== dimension) throw new Error("Active Memory store has inconsistent vector dimensions");
}

function validateSource(value: unknown, allowLegacy: boolean): boolean {
  if (!value || typeof value !== "object") throw new Error("Active Memory store contains invalid provenance");
  const source = value as Partial<MemoryRecord["source"]>;
  const stringFields = [source.sessionId, source.cwd, source.cause, source.reason];
  if ((!allowLegacy && (source.actor !== "user" && source.actor !== "assistant" || stringFields.some(field => typeof field !== "string" || !field.trim()))) || (allowLegacy && stringFields.some(field => field !== undefined && typeof field !== "string")) || (source.actor !== undefined && source.actor !== "user" && source.actor !== "assistant") || (source.elapsedMs !== undefined && (!Number.isFinite(source.elapsedMs) || source.elapsedMs < 0)) || [source.userText, source.evidence].some(field => field !== undefined && typeof field !== "string")) throw new Error("Active Memory store contains invalid provenance");
  return true;
}

function validTimestamp(value: unknown): value is string { return typeof value === "string" && value.length > 0 && Number.isFinite(Date.parse(value)); }
