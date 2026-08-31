import { SteerFeedbackLedger } from "./feedback.js";
import { MemorySteerLimiter } from "./steer-frequency.js";
import type { ActiveMemoryConfig } from "./types.js";
import type { SteerFeedbackOutcome } from "./steer-display.js";

export interface RestoredSteerDetails { memoryIds: string[]; scores: number[]; reason: string; instruction: string; projectId: string; feedbackToken: string; source: "active-memory" }
export interface SessionRestoreResult { turnSequence: number; lastSteerAt: number; lastSteerFingerprint: string; lastRecall?: RestoredSteerDetails }

/**
 * Restore Pi branch state. Pi steers are custom_message entries, not regular
 * messages; keeping this parsing separate prevents resumed sessions from silently
 * losing feedback and recall-rate limits.
 */
export function restoreSessionState(
  entries: readonly unknown[],
  config: ActiveMemoryConfig,
  ledger: SteerFeedbackLedger,
  limiter: MemorySteerLimiter,
  displayed: (token: string, memoryId: string, outcome: SteerFeedbackOutcome) => void,
  fingerprint: (details: RestoredSteerDetails) => string,
): SessionRestoreResult {
  let turnSequence = 0;
  let lastSteerAt = 0;
  let lastSteerFingerprint = "";
  let lastRecall: RestoredSteerDetails | undefined;
  for (const entry of entries) {
    const raw = entry as { type?: string; message?: { role?: string; toolName?: string; details?: unknown; timestamp?: unknown }; customType?: string; details?: unknown; timestamp?: unknown };
    const message = raw.type === "message" ? raw.message : undefined;
    // Runtime advances this counter at turn_end, once for each completed
    // assistant turn (including tool-driven continuation turns).
    if (message?.role === "assistant") turnSequence++;
    if (raw.type === "custom_message" && raw.customType === "active-memory-steer") {
      const details = raw.details;
      if (isRestoredSteerDetails(details)) {
        const parsed = typeof raw.timestamp === "number" ? raw.timestamp : typeof raw.timestamp === "string" ? Date.parse(raw.timestamp) : Number.NaN;
        if (!Number.isFinite(parsed) || parsed < 0 || parsed > Date.now() + 5 * 60_000) continue;
        ledger.register(details.feedbackToken, details.memoryIds);
        limiter.record(details.memoryIds, parsed, turnSequence);
        lastSteerAt = parsed;
        lastSteerFingerprint = fingerprint(details);
        lastRecall = details;
      }
      continue;
    }
    if (message?.role !== "toolResult" || message.toolName !== "memory_feedback") continue;
    const details = message.details as { accepted?: unknown; steerToken?: unknown; memoryId?: unknown; outcome?: unknown } | undefined;
    if (details?.accepted === true && typeof details.steerToken === "string" && details.steerToken.trim() && typeof details.memoryId === "string" && details.memoryId.trim() && (details.outcome === "useful" || details.outcome === "unhelpful")) {
      const restored = ledger.consume(details.steerToken, details.memoryId, config.memoryLifecycle.feedback.maxPerMemoryPerSession);
      if (restored === "accepted") displayed(details.steerToken, details.memoryId, details.outcome);
    }
  }
  return { turnSequence, lastSteerAt, lastSteerFingerprint, lastRecall };
}

function isRestoredSteerDetails(value: unknown): value is RestoredSteerDetails {
  if (!value || typeof value !== "object") return false;
  const details = value as Partial<RestoredSteerDetails>;
  if (typeof details.feedbackToken !== "string" || !details.feedbackToken.trim() || !Array.isArray(details.memoryIds) || details.memoryIds.length === 0 || details.memoryIds.some(id => typeof id !== "string" || !id.trim())) return false;
  if (!Array.isArray(details.scores) || details.scores.length !== details.memoryIds.length || details.scores.some(score => typeof score !== "number" || !Number.isFinite(score))) return false;
  if (typeof details.reason !== "string" || typeof details.instruction !== "string" || !details.instruction.trim() || typeof details.projectId !== "string" || !details.projectId.trim()) return false;
  if (details.source !== "active-memory") return false;
  return true;
}
