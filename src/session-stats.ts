export const SESSION_STATS_ENTRY_TYPE = "active-memory-session-stat";

export type SessionStatName = "memoriesCreated" | "recallAttempts" | "memorySteers" | "useful" | "unhelpful" | "fastModelInputTokens" | "fastModelOutputTokens";

export interface SessionStats {
  memoriesCreated: number;
  recallAttempts: number;
  memorySteers: number;
  useful: number;
  unhelpful: number;
  fastModelInputTokens: number;
  fastModelOutputTokens: number;
}

export function emptySessionStats(): SessionStats {
  return { memoriesCreated: 0, recallAttempts: 0, memorySteers: 0, useful: 0, unhelpful: 0, fastModelInputTokens: 0, fastModelOutputTokens: 0 };
}

export function sessionStatsFromEntries(entries: readonly unknown[]): SessionStats {
  const stats = emptySessionStats();
  for (const raw of entries) {
    const entry = raw as { type?: string; customType?: string; data?: { name?: unknown; amount?: unknown } };
    if (entry.type !== "custom" || entry.customType !== SESSION_STATS_ENTRY_TYPE) continue;
    const name = entry.data?.name;
    const amount = entry.data?.amount;
    if (!isSessionStatName(name) || typeof amount !== "number" || !Number.isFinite(amount) || amount <= 0) continue;
    stats[name] += amount;
  }
  return stats;
}

export function displayedSessionStats(stats: SessionStats, tokenUsageAvailable: boolean): Omit<SessionStats, "fastModelInputTokens" | "fastModelOutputTokens"> & { fastModelInputTokens: number | "N/A"; fastModelOutputTokens: number | "N/A" } {
  return {
    ...stats,
    fastModelInputTokens: tokenUsageAvailable ? stats.fastModelInputTokens : "N/A",
    fastModelOutputTokens: tokenUsageAvailable ? stats.fastModelOutputTokens : "N/A",
  };
}

export function formatSessionStats(stats: SessionStats, tokenUsageAvailable: boolean): string {
  const displayed = displayedSessionStats(stats, tokenUsageAvailable);
  return [
    "Current session memory stats",
    `Memories created: ${stats.memoriesCreated}`,
    `Recall attempts: ${stats.recallAttempts}`,
    `Memory steers: ${stats.memorySteers}`,
    `Useful: ${stats.useful}`,
    `Not useful: ${stats.unhelpful}`,
    `Fast-model input tokens: ${displayed.fastModelInputTokens}`,
    `Fast-model output tokens: ${displayed.fastModelOutputTokens}`,
  ].join("\n");
}

function isSessionStatName(value: unknown): value is SessionStatName {
  return value === "memoriesCreated" || value === "recallAttempts" || value === "memorySteers" || value === "useful" || value === "unhelpful" || value === "fastModelInputTokens" || value === "fastModelOutputTokens";
}
