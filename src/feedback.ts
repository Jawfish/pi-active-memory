import type { MemoryFeedback, MemoryLifecycleConfig, MemoryRecord } from "./types.js";

export type FeedbackOutcome = "useful" | "unhelpful";

export class SteerFeedbackLedger {
  private readonly steers = new Map<string, Set<string>>();
  private readonly consumed = new Set<string>();
  private readonly perMemory = new Map<string, number>();

  register(token: string, memoryIds: readonly string[]): void {
    this.steers.set(token, new Set(memoryIds));
  }

  consume(token: string, memoryId: string, maximumPerMemory: number): "accepted" | "unknown_token" | "not_in_steer" | "duplicate" | "session_limit" {
    const ids = this.steers.get(token);
    if (!ids) return "unknown_token";
    if (!ids.has(memoryId)) return "not_in_steer";
    const key = `${token}\u0000${memoryId}`;
    if (this.consumed.has(key)) return "duplicate";
    if ((this.perMemory.get(memoryId) ?? 0) >= maximumPerMemory) return "session_limit";
    this.consumed.add(key);
    this.perMemory.set(memoryId, (this.perMemory.get(memoryId) ?? 0) + 1);
    return "accepted";
  }

  release(token: string, memoryId: string): void {
    const key = `${token}\u0000${memoryId}`;
    if (!this.consumed.delete(key)) return;
    const next = Math.max(0, (this.perMemory.get(memoryId) ?? 1) - 1);
    if (next === 0) this.perMemory.delete(memoryId);
    else this.perMemory.set(memoryId, next);
  }
}

export function applyConfidenceFeedback(
  record: MemoryRecord,
  feedback: MemoryFeedback,
  config: MemoryLifecycleConfig,
): MemoryRecord {
  const useful = feedback.outcome === "useful";
  const delta = useful ? config.confidence.usefulDelta : -config.confidence.unhelpfulDelta;
  const confidence = clamp(record.confidence + delta, config.confidence.minimum, config.confidence.maximum);
  // Useful evidence makes future passive decay slower. Unhelpful evidence only
  // lowers confidence; it never changes the decay rate.
  const decayRate = useful
    ? clamp((record.decayRate ?? config.decay.initialRate) - config.decay.usefulDelta, config.decay.minimumRate, config.decay.maximumRate)
    : record.decayRate ?? config.decay.initialRate;
  const history = [...(record.feedback?.history ?? []), feedback].slice(-config.feedback.historyLimit);
  return {
    ...record,
    confidence,
    decayRate,
    feedback: {
      useful: (record.feedback?.useful ?? 0) + (useful ? 1 : 0),
      unhelpful: (record.feedback?.unhelpful ?? 0) + (useful ? 0 : 1),
      lastAt: feedback.at,
      history,
    },
    updatedAt: feedback.at,
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
