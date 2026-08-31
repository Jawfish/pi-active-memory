import type { MemoryActor, MemoryRecord, MemorySource } from "./types.js";

function sourceIsComplete(source: Partial<MemorySource> | undefined): boolean {
  return Boolean(
    source &&
    (source.actor === "user" || source.actor === "assistant") &&
    typeof source.sessionId === "string" && source.sessionId.trim() &&
    typeof source.cwd === "string" && source.cwd.trim() &&
    typeof source.cause === "string" && source.cause.trim() &&
    typeof source.reason === "string" && source.reason.trim(),
  );
}

export function hasCompleteProvenance(record: MemoryRecord): boolean {
  const supersedes = record.supersedes ?? [];
  return record.schemaVersion === 2 &&
    sourceIsComplete(record.source as Partial<MemorySource> | undefined) &&
    (record.sourceHistory ?? []).every((source) => sourceIsComplete(source)) &&
    Number.isFinite(record.priority) &&
    !supersedes.includes(record.id) &&
    new Set(supersedes).size === supersedes.length;
}

function normalizeSource(value: Partial<MemorySource> | undefined): MemorySource {
  const legacy = value ?? {};
  const actor: MemoryActor = legacy.actor === "assistant" ? "assistant" : "user";
  return {
    ...legacy,
    actor,
    sessionId: typeof legacy.sessionId === "string" && legacy.sessionId.trim() ? legacy.sessionId : "unknown-legacy-session",
    cwd: typeof legacy.cwd === "string" && legacy.cwd.trim() ? legacy.cwd : "unknown-legacy-cwd",
    cause: typeof legacy.cause === "string" && legacy.cause.trim() ? legacy.cause : "legacy_memory_migration",
    reason: typeof legacy.reason === "string" && legacy.reason.trim()
      ? legacy.reason
      : "Retained from a memory created before cause and storage rationale were recorded",
  };
}

/** Upgrade records created before actor/cause/rationale were mandatory. */
export function normalizeProvenance(record: MemoryRecord): MemoryRecord {
  const source = normalizeSource(record.source as Partial<MemorySource> | undefined);
  const supersedes = record.supersedes ? [...new Set(record.supersedes.filter(id => id !== record.id))] : undefined;
  return {
    ...record,
    source,
    ...(record.sourceHistory ? { sourceHistory: record.sourceHistory.map((item) => normalizeSource(item)) } : {}),
    ...(supersedes ? { supersedes } : {}),
    priority: Number.isFinite(record.priority) ? record.priority : source.actor === "assistant" ? 0.55 : 1,
    schemaVersion: 2,
  };
}
