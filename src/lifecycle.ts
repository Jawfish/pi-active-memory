import type { MemoryLifecycle, MemoryLifecycleConfig, MemoryRecord } from "./types.js";

export type MemoryExpiryCause = "low_confidence";

export interface LifecycleSweepResult {
  record: MemoryRecord;
  initialized: boolean;
  decayedDays: number;
  expiredBy?: MemoryExpiryCause;
}

export function initializeMemoryLifecycle(
  record: MemoryRecord,
  now: Date,
  sessionId: string,
  config: MemoryLifecycleConfig,
): MemoryRecord {
  const date = utcDate(now);
  return {
    ...record,
    decayRate: clamp(record.decayRate ?? config.decay.initialRate, config.decay.minimumRate, config.decay.maximumRate),
    lifecycle: isCurrentLifecycle(record.lifecycle)
      ? record.lifecycle
      : createLifecycle(now, date, sessionId, "created"),
  };
}

export function advanceMemoryLifecycle(
  record: MemoryRecord,
  now: Date,
  sessionId: string,
  config: MemoryLifecycleConfig,
): LifecycleSweepResult {
  if (record.status !== "active") return { record, initialized: false, decayedDays: 0 };
  const date = utcDate(now);
  const initialized = !isCurrentLifecycle(record.lifecycle) || !Number.isFinite(record.decayRate);
  if (initialized) {
    // Legacy records start a fresh daily-decay clock; migration never retroactively
    // decays years of history or immediately deletes an old low-confidence record.
    return {
      initialized: true,
      decayedDays: 0,
      record: {
        ...record,
        decayRate: clamp(record.decayRate ?? config.decay.initialRate, config.decay.minimumRate, config.decay.maximumRate),
        lifecycle: createLifecycle(now, date, sessionId, "legacy_migration_grace"),
      },
    };
  }

  const decayRate = clamp(record.decayRate!, config.decay.minimumRate, config.decay.maximumRate);
  const decayedDays = Math.max(0, daysBetween(record.lifecycle!.lastDecayDate, date));
  const confidence = decayedDays > 0
    ? record.confidence * Math.pow(1 - decayRate, decayedDays)
    : record.confidence;
  const lifecycle: MemoryLifecycle = {
    ...record.lifecycle!,
    ...(decayedDays > 0 ? { lastDecayDate: date } : {}),
  };
  const next: MemoryRecord = { ...record, confidence, decayRate, lifecycle, ...(decayedDays > 0 ? { updatedAt: now.toISOString() } : {}) };
  if (confidence >= config.confidence.deletionThreshold) return { record: next, initialized: false, decayedDays };

  return {
    initialized: false,
    decayedDays,
    expiredBy: "low_confidence",
    record: {
      ...next,
      status: "deleted",
      lifecycle: { ...lifecycle, deletedAt: now.toISOString(), deletionCause: "low_confidence" },
      updatedAt: now.toISOString(),
    },
  };
}

export function reinforceMemoryLifecycle(
  record: MemoryRecord,
  now: Date,
  sessionId: string,
  config: MemoryLifecycleConfig,
  cause: MemoryLifecycle["lastReinforcementCause"],
): MemoryRecord {
  const initialized = initializeMemoryLifecycle(record, now, sessionId, config);
  return {
    ...initialized,
    lifecycle: {
      ...initialized.lifecycle!,
      lastDecayDate: utcDate(now),
      lastRelevantAt: now.toISOString(),
      lastRelevantSessionId: sessionId,
      reinforcementCount: initialized.lifecycle!.reinforcementCount + 1,
      lastReinforcementCause: cause,
      deletedAt: undefined,
      deletionCause: undefined,
    },
    updatedAt: now.toISOString(),
  };
}

function createLifecycle(
  now: Date,
  date: string,
  sessionId: string,
  cause: MemoryLifecycle["lastReinforcementCause"],
): MemoryLifecycle {
  return {
    lastDecayDate: date,
    lastRelevantAt: now.toISOString(),
    lastRelevantSessionId: sessionId,
    reinforcementCount: 0,
    lastReinforcementCause: cause,
  };
}

function isCurrentLifecycle(lifecycle: MemoryRecord["lifecycle"]): lifecycle is MemoryLifecycle {
  return Boolean(lifecycle && typeof lifecycle.lastDecayDate === "string" && /^\d{4}-\d{2}-\d{2}$/.test(lifecycle.lastDecayDate));
}

function utcDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function daysBetween(left: string, right: string): number {
  const leftMs = Date.parse(`${left}T00:00:00.000Z`);
  const rightMs = Date.parse(`${right}T00:00:00.000Z`);
  if (!Number.isFinite(leftMs) || !Number.isFinite(rightMs)) return 0;
  return Math.floor((rightMs - leftMs) / 86_400_000);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
